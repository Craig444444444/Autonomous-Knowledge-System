import logging
import subprocess
import os
import re
import shutil
import time
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
import urllib.parse
import tempfile

LOGGER = logging.getLogger("aks")

class GitManager:
    """
    Enhanced Git repository manager with improved error handling, security,
    and advanced Git operations for the AKS system.
    """
    def __init__(self, repo_path: Path, github_token: Optional[str],
                 repo_owner: str, repo_name: str, repo_url: str):
        self.repo_path = repo_path.resolve()
        self.github_token = github_token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.repo_url = repo_url
        self.last_push_time = time.time()
        self._lock_file = self.repo_path / ".aks_git_lock"
        self._configured = False

        LOGGER.info(f"Initializing GitManager for {self.repo_path}")
        # Ensure repository directory exists
        self.repo_path.mkdir(parents=True, exist_ok=True)
        self._configure_git_environment()

    def _configure_git_environment(self):
        """Configure Git environment with safe defaults and authentication."""
        try:
            # First try local configuration
            LOGGER.info("Attempting local Git configuration")
            self._run_git_command(["config", "--local", "safe.directory", str(self.repo_path)])
            self._run_git_command(["config", "--local", "user.email", f"{self.repo_owner}@aks-ai.system"])
            self._run_git_command(["config", "--local", "user.name", "AKS Autonomous System"])
            self._run_git_command(["config", "--local", "pull.rebase", "false"])
            self._run_git_command(["config", "--local", "gc.auto", "0"])
            self._configured = True
            LOGGER.info("Git environment configured successfully (local config)")
        except Exception as e:
            LOGGER.warning(f"Local configuration failed: {e}. Using environment variables as fallback")
            try:
                # Fallback to environment variables
                os.environ['GIT_SAFE_DIRECTORY'] = str(self.repo_path)
                os.environ['GIT_AUTHOR_NAME'] = "AKS Autonomous System"
                os.environ['GIT_COMMITTER_NAME'] = "AKS Autonomous System"
                os.environ['GIT_AUTHOR_EMAIL'] = f"{self.repo_owner}@aks-ai.system"
                os.environ['GIT_COMMITTER_EMAIL'] = f"{self.repo_owner}@aks-ai.system"
                self._configured = True
                LOGGER.info("Git environment configured via environment variables")
            except Exception as env_e:
                LOGGER.error(f"Fallback configuration failed: {env_e}")
                raise RuntimeError("Git environment configuration failed") from env_e

    def _acquire_lock(self, timeout: int = 30) -> bool:
        """Acquire a filesystem lock for thread-safe operations."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self._lock_file.touch(exist_ok=False)
                return True
            except FileExistsError:
                time.sleep(0.1)
            except Exception as e:
                LOGGER.error(f"Lock acquisition failed: {e}")
                return False
        LOGGER.warning("Could not acquire Git lock within timeout")
        return False

    def _release_lock(self) -> None:
        """Release the filesystem lock."""
        try:
            self._lock_file.unlink(missing_ok=True)
        except Exception as e:
            LOGGER.error(f"Failed to release lock: {e}")

    def _sanitize_git_command(self, command: List[str]) -> List[str]:
        """Sanitize Git commands to prevent injection attacks."""
        sanitized = []
        for arg in command:
            if any(c in arg for c in [";", "|", "&", "$", "`", ">", "<"]):
                raise ValueError(f"Potentially dangerous character in Git command: {arg}")
            sanitized.append(arg)
        return sanitized

    def _run_git_command(self, command: List[str], cwd: Optional[Path] = None,
                        check: bool = True, timeout: int = 60, retries: int = 3) -> Tuple[bool, str]:
        """
        Enhanced Git command runner with improved security and reliability.

        Args:
            command: Git command arguments
            cwd: Working directory
            check: Raise exception on failure if True
            timeout: Command timeout in seconds
            retries: Number of retry attempts

        Returns:
            Tuple of (success, output)
        """
        if not self._configured:
            raise RuntimeError("GitManager not properly configured")

        cwd = cwd or self.repo_path
        if not cwd.exists():
            cwd.mkdir(parents=True, exist_ok=True)
            LOGGER.warning(f"Created missing working directory: {cwd}")

        # Sanitize command arguments
        try:
            command = self._sanitize_git_command(["git"] + command)
        except ValueError as e:
            LOGGER.error(f"Command sanitization failed: {e}")
            return False, str(e)

        # Prepare environment with authentication
        env = os.environ.copy()
        if self.github_token:
            safe_token = urllib.parse.quote(self.github_token, safe='')
            env["GIT_ASKPASS"] = "echo"
            env["GIT_USERNAME"] = "oauth2"
            env["GIT_PASSWORD"] = safe_token

        last_error = ""
        for attempt in range(1, retries + 1):
            try:
                LOGGER.debug(f"Attempt {attempt}: Running git {' '.join(command)} in {cwd}")

                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    env=env,
                    preexec_fn=os.setsid
                )

                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    raise

                if process.returncode != 0:
                    error_msg = stderr.strip() or stdout.strip()
                    last_error = f"Git command failed (exit {process.returncode}): {error_msg}"

                    # Handle specific error cases
                    if "safe.directory" in error_msg:
                        LOGGER.warning("Safe directory error detected, applying fallback")
                        # Set in both environment and current env copy
                        os.environ['GIT_SAFE_DIRECTORY'] = str(self.repo_path)
                        env['GIT_SAFE_DIRECTORY'] = str(self.repo_path)
                        continue
                    elif "Authentication failed" in error_msg:
                        LOGGER.error("Git authentication failed")
                        break
                    elif "would be overwritten by merge" in error_msg:
                        LOGGER.warning("Merge conflict detected")
                        break
                    elif "not a git repository" in error_msg:
                        LOGGER.warning("Git repository not initialized")
                        self.initialize_repo()
                        continue

                    if attempt < retries:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    continue

                return True, stdout.strip()

            except subprocess.TimeoutExpired:
                last_error = "Command timed out"
                LOGGER.warning(f"Git command timed out (attempt {attempt})")
            except Exception as e:
                last_error = str(e)
                LOGGER.error(f"Unexpected error running git command: {e}")

        if check:
            raise RuntimeError(last_error)
        return False, last_error

    def is_repo_initialized(self) -> bool:
        """Check if the repository is properly initialized."""
        return (self.repo_path / ".git").exists()

    def initialize_repo(self) -> bool:
        """Initialize or reinitialize the Git repository."""
        if not self._acquire_lock():
            return False

        try:
            if self.is_repo_initialized():
                LOGGER.info("Repository already initialized")
                return True

            LOGGER.info("Initializing new Git repository")
            success, _ = self._run_git_command(["init"])
            if not success:
                return False

            if self.repo_url:
                LOGGER.info(f"Setting remote origin to {self.repo_url}")
                return self._run_git_command(["remote", "add", "origin", self.repo_url])[0]

            return True
        finally:
            self._release_lock()

    def verify_repository(self) -> bool:
        """Verify repository integrity with fsck and status checks."""
        LOGGER.info("Verifying repository integrity")

        # Check for uncommitted changes
        success, status = self._run_git_command(["status", "--porcelain"])
        if not success:
            LOGGER.error("Failed to check repository status")
            return False
        if status:
            LOGGER.warning(f"Uncommitted changes detected:\n{status}")

        # Run filesystem check
        success, output = self._run_git_command(["fsck"], check=False)
        if not success:
            LOGGER.error(f"Repository integrity check failed:\n{output}")
            return False

        return True

    def fetch_and_reset(self, branch: Optional[str] = None) -> Tuple[bool, str]:
        """Safely fetch and reset repository to remote state."""
        if not self._acquire_lock():
            return False, "Could not acquire lock for fetch/reset"

        try:
            LOGGER.info("Fetching latest changes from remote")
            success, output = self._run_git_command(["fetch", "--all", "--prune"])
            if not success:
                return False, f"Fetch failed: {output}"

            current_branch = branch or self.get_current_branch()
            if not current_branch:
                return False, "Could not determine current branch"

            LOGGER.info(f"Resetting {current_branch} to origin/{current_branch}")
            return self._run_git_command(["reset", "--hard", f"origin/{current_branch}"])
        finally:
            self._release_lock()

    def get_current_branch(self) -> Optional[str]:
        """Get current branch name with verification."""
        success, branch = self._run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
        if not success or not branch:
            return None
        return branch.strip()

    def get_remote_branches(self) -> List[str]:
        """Get list of all remote branches."""
        success, output = self._run_git_command(["branch", "-r", "--format=%(refname:short)"])
        if not success:
            return []

        branches = [
            b.replace("origin/", "")
            for b in output.split('\n')
            if b and not b.endswith("/HEAD")
        ]
        return branches

    def create_and_checkout_branch(self, branch_name: str, base: str = "main") -> bool:
        """Create and checkout new branch with validation."""
        if not re.match(r"^[a-zA-Z0-9_\-\./]+$", branch_name):
            LOGGER.error(f"Invalid branch name: {branch_name}")
            return False

        if not self._acquire_lock():
            return False

        try:
            # Verify base branch exists
            success, _ = self._run_git_command(["show-ref", "--verify", f"refs/heads/{base}"])
            if not success:
                LOGGER.error(f"Base branch {base} does not exist")
                return False

            LOGGER.info(f"Creating branch {branch_name} from {base}")
            return self._run_git_command(["checkout", "-b", branch_name, base])[0]
        finally:
            self._release_lock()

    def add_and_commit(self, message: str, paths: Optional[List[str]] = None) -> bool:
        """Stage and commit changes with validation."""
        if not message or len(message) > 2000:
            LOGGER.error("Invalid commit message length")
            return False

        if not self._acquire_lock():
            return False

        try:
            # Stage changes
            add_cmd = ["add"]
            if paths:
                add_cmd.extend(paths)
            else:
                add_cmd.append(".")

            success, _ = self._run_git_command(add_cmd)
            if not success:
                return False

            # Verify there are changes to commit
            success, status = self._run_git_command(["status", "--porcelain"])
            if not success or not status:
                LOGGER.info("No changes to commit")
                return False

            # Create commit
            return self._run_git_command(["commit", "-m", message])[0]
        finally:
            self._release_lock()

    def push_changes(self, branch: Optional[str] = None, force: bool = False) -> bool:
        """Push changes to remote with safety checks."""
        branch = branch or self.get_current_branch()
        if not branch:
            LOGGER.error("No branch specified for push")
            return False

        if not self._acquire_lock():
            return False

        try:
            LOGGER.info(f"Pushing branch {branch} to remote")

            # First push with -u to set upstream if needed
            push_cmd = ["push", "-u", "origin", branch]
            if force:
                push_cmd.insert(1, "--force-with-lease")

            success, output = self._run_git_command(push_cmd)
            if success:
                self.last_push_time = time.time()
            return success
        finally:
            self._release_lock()

    def merge_branch(self, target_branch: str, source_branch: str) -> bool:
        """Safely merge branches with conflict handling."""
        if not self._acquire_lock(timeout=60):
            return False

        try:
            original_branch = self.get_current_branch()
            if not original_branch:
                return False

            LOGGER.info(f"Merging {source_branch} into {target_branch}")

            # Checkout target branch
            if not self._run_git_command(["checkout", target_branch])[0]:
                return False

            # Perform merge
            merge_cmd = [
                "merge", "--no-ff", "--no-commit",
                "-m", f"Merge branch '{source_branch}' into {target_branch}",
                source_branch
            ]
            success, output = self._run_git_command(merge_cmd)

            if not success:
                LOGGER.warning(f"Merge failed: {output}")
                self._run_git_command(["merge", "--abort"])
                return False

            # Verify merge result
            success, status = self._run_git_command(["status", "--porcelain"])
            if not success:
                return False

            if "UU" in status:  # Unmerged files
                LOGGER.warning("Merge conflicts detected")
                self._run_git_command(["merge", "--abort"])
                return False

            # Finalize merge
            return self._run_git_command(["commit", "--no-edit"])[0]

        finally:
            if original_branch:
                self._run_git_command(["checkout", original_branch])
            self._release_lock()

    def delete_branch(self, branch_name: str, remote: bool = True) -> bool:
        """Delete branch with safety checks."""
        if branch_name in ["main", "master"]:
            LOGGER.error(f"Cannot delete protected branch: {branch_name}")
            return False

        current_branch = self.get_current_branch()
        if current_branch == branch_name:
            LOGGER.warning(f"Cannot delete current branch {branch_name}, switching to main")
            if not self._run_git_command(["checkout", "main"])[0]:
                return False

        if not self._acquire_lock():
            return False

        try:
            # Delete local branch
            success, _ = self._run_git_command(["branch", "-D", branch_name], check=False)
            if not success:
                LOGGER.warning(f"Failed to delete local branch {branch_name}")

            # Delete remote branch if requested
            if remote:
                remote_success, _ = self._run_git_command(
                    ["push", "origin", "--delete", branch_name],
                    check=False
                )
                return success and remote_success

            return success
        finally:
            self._release_lock()

    def clean_old_branches(self, max_branches: int = 10, pattern: str = r"^(feature|enhance)-ai-") -> bool:
        """Clean old branches matching pattern, keeping most recent ones."""
        if not self._acquire_lock():
            return False

        try:
            # Get all branches matching pattern
            success, branches = self._run_git_command(["branch", "--list", "--format=%(refname:short)"])
            if not success:
                return False

            matched_branches = [
                b for b in branches.split('\n')
                if b and re.match(pattern, b)
            ]

            # Sort by commit date (newest first)
            dated_branches = []
            for branch in matched_branches:
                success, date = self._run_git_command([
                    "log", "-1", "--format=%ct", branch
                ])
                if success and date:
                    dated_branches.append((int(date), branch))

            # Sort branches by date (newest first)
            dated_branches.sort(reverse=True, key=lambda x: x[0])

            # Determine branches to delete
            to_delete = [b for _, b in dated_branches[max_branches:]]
            if not to_delete:
                LOGGER.info("No old branches to clean")
                return True

            LOGGER.info(f"Cleaning {len(to_delete)} old branches: {', '.join(to_delete)}")

            # Delete branches
            all_success = True
            for branch in to_delete:
                if not self.delete_branch(branch):
                    all_success = False

            return all_success
        finally:
            self._release_lock()

    def get_commit_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get formatted commit history."""
        success, output = self._run_git_command([
            "log",
            f"-{limit}",
            "--pretty=format:%H|%an|%ad|%s",
            "--date=iso"
        ])

        if not success:
            return []

        commits = []
        for line in output.split('\n'):
            parts = line.split('|', 3)
            if len(parts) == 4:
                commits.append({
                    "hash": parts[0],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3]
                })

        return commits

    def execute_shell_command(self, command: str, timeout: int = 60) -> Tuple[bool, str]:
        """Execute shell command with enhanced security."""
        if not command:
            return False, "Empty command"

        # Basic command validation
        if any(c in command for c in [";", "|", "&", "$", "`", ">", "<"]):
            return False, "Potentially dangerous command"

        if not self._acquire_lock():
            return False, "Could not acquire lock"

        try:
            return self._run_git_command(["bash", "-c", command], timeout=timeout)
        finally:
            self._release_lock()

    def create_tag(self, tag_name: str, message: str = "") -> bool:
        """Create annotated Git tag."""
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", tag_name):
            LOGGER.error(f"Invalid tag name: {tag_name}")
            return False

        cmd = ["tag", "-a", tag_name]
        if message:
            cmd.extend(["-m", message])

        return self._run_git_command(cmd)[0]

    def get_repository_status(self) -> Dict[str, Any]:
        """Get comprehensive repository status."""
        status = {
            "branch": self.get_current_branch(),
            "clean": True,
            "modified_files": [],
            "untracked_files": [],
            "ahead": 0,
            "behind": 0,
            "last_commit": None
        }

        # Check for changes
        success, output = self._run_git_command(["status", "--porcelain", "--branch"])
        if not success:
            return status

        # Parse status output
        for line in output.split('\n'):
            if line.startswith('##'):
                # Branch tracking info
                match = re.search(r"ahead (\d+)", line)
                if match:
                    status["ahead"] = int(match.group(1))
                match = re.search(r"behind (\d+)", line)
                if match:
                    status["behind"] = int(match.group(1))
            else:
                # File status
                if line:
                    status["clean"] = False
                    if line.startswith('??'):
                        status["untracked_files"].append(line[3:])
                    else:
                        status["modified_files"].append(line[3:])

        # Get last commit info
        commits = self.get_commit_history(1)
        if commits:
            status["last_commit"] = commits[0]

        return status
