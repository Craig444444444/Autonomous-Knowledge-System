import logging
import shutil
import zipfile
import os
import tempfile
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
import stat
import time
import json
import re
import random

LOGGER = logging.getLogger("aks")

class ResilienceManager:
    """
    Enhanced system resilience manager with improved snapshot handling,
    validation, and security features for the AKS system.
    """
    def __init__(self, repo_path: Path, snapshot_dir: Path, max_snapshots: int = 5):
        """
        Initialize the enhanced ResilienceManager.

        Args:
            repo_path: Path to the repository (validated)
            snapshot_dir: Directory to store snapshots (created if needed)
            max_snapshots: Maximum number of snapshots to retain (minimum 1)
        """
        self.repo_path = repo_path.resolve()
        self.snapshot_dir = snapshot_dir.resolve()
        self.max_snapshots = max(1, max_snapshots)  # Ensure at least 1 snapshot is kept
        self._setup_directories()
        self._processed_hashes = set()  # Track processed snapshots
        LOGGER.info(f"Initialized ResilienceManager for {self.repo_path}")

    def _setup_directories(self):
        """Secure directory setup with proper permissions."""
        try:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            # Set restrictive permissions on snapshot directory
            self.snapshot_dir.chmod(0o700)

            # Verify repository exists
            if not self.repo_path.exists():
                raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")

            LOGGER.debug("Verified/Created required directories")
        except Exception as e:
            LOGGER.error(f"Directory setup failed: {e}")
            raise RuntimeError("Could not initialize directories") from e

    def create_snapshot(self, tag: str = "auto", description: str = "") -> Optional[Path]:
        """
        Create a secure, validated snapshot of the repository.

        Args:
            tag: Short identifier for the snapshot
            description: Longer description of the snapshot purpose

        Returns:
            Path to the created snapshot or None if failed
        """
        # Validate tag
        if not re.match(r"^[a-zA-Z0-9_\-]+$", tag):
            LOGGER.error(f"Invalid snapshot tag: {tag}")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"aks_snapshot_{tag}_{timestamp}.zip"
        snapshot_path = self.snapshot_dir / snapshot_name

        LOGGER.info(f"Creating snapshot '{tag}' at {snapshot_path}")

        try:
            # Create temporary staging area
            with tempfile.TemporaryDirectory(dir=self.snapshot_dir) as temp_dir:
                temp_path = Path(temp_dir)
                temp_repo = temp_path / "repo"

                # Copy with ignore patterns and permission preservation
                self._copy_repo_to_temp(temp_repo)

                # Add metadata file
                self._create_snapshot_metadata(temp_repo, tag, description)

                # Create checksum of the snapshot content
                snapshot_hash = self._hash_directory(temp_repo)
                if snapshot_hash in self._processed_hashes:
                    LOGGER.warning("Duplicate snapshot content detected")
                    return None

                # Create secure zip archive
                self._create_secure_zip(temp_repo, snapshot_path)

                # Verify the created snapshot
                if not self._validate_snapshot(snapshot_path):
                    LOGGER.error("Snapshot validation failed")
                    snapshot_path.unlink(missing_ok=True)
                    return None

                self._processed_hashes.add(snapshot_hash)
                LOGGER.info(f"Snapshot created successfully: {snapshot_path.name}")

                # Clean up old snapshots
                self.cleanup_snapshots()

                return snapshot_path

        except Exception as e:
            LOGGER.error(f"Snapshot creation failed: {e}", exc_info=True)
            snapshot_path.unlink(missing_ok=True)
            return None

    def _copy_repo_to_temp(self, dest_path: Path) -> None:
        """Secure copy of repository to temporary directory."""
        ignore_patterns = shutil.ignore_patterns(
            '.git', '__pycache__', '*.pyc', '.ipynb_checkpoints',
            '.tmp', '.swp', '*.bak', '.DS_Store'
        )

        shutil.copytree(
            self.repo_path,
            dest_path,
            ignore=ignore_patterns,
            symlinks=False,
            dirs_exist_ok=False
        )

        # Set safe permissions on copied files
        for root, dirs, files in os.walk(dest_path):
            for d in dirs:
                os.chmod(Path(root) / d, 0o755)
            for f in files:
                os.chmod(Path(root) / f, 0o644)

    def _create_snapshot_metadata(self, dest_path: Path, tag: str, description: str) -> None:
        """Create metadata file for the snapshot."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "tag": tag,
            "description": description,
            "repo_path": str(self.repo_path),
            "system": {
                "python_version": os.sys.version,
                "platform": os.sys.platform
            }
        }

        metadata_path = dest_path / ".aks_snapshot_meta.json"
        with metadata_path.open('w') as f:
            json.dump(metadata, f, indent=2)
        metadata_path.chmod(0o644)

    def _hash_directory(self, directory: Path) -> str:
        """Create a hash of directory contents for duplicate detection."""
        hasher = hashlib.sha256()

        for root, _, files in os.walk(directory):
            for file in sorted(files):  # Sort for consistent ordering
                file_path = Path(root) / file
                hasher.update(file_path.read_bytes())

        return hasher.hexdigest()

    def _create_secure_zip(self, source_dir: Path, dest_zip: Path) -> None:
        """Create a zip archive with security checks."""
        # First create a temporary zip file
        temp_zip = dest_zip.with_suffix('.tmp')

        with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)

        # Validate the zip before moving to final location
        if not self._validate_zip(temp_zip):
            temp_zip.unlink()
            raise RuntimeError("Created zip file failed validation")

        # Atomic move to final location
        temp_zip.replace(dest_zip)

    def _validate_zip(self, zip_path: Path) -> bool:
        """Validate the integrity of a zip file."""
        try:
            # Basic zip validation
            if not zipfile.is_zipfile(zip_path):
                return False

            with zipfile.ZipFile(zip_path, 'r') as zipf:
                # Check for zip bombs or malicious paths
                total_size = 0
                for info in zipf.infolist():
                    # Prevent path traversal
                    if '..' in info.filename or info.filename.startswith('/'):
                        LOGGER.warning(f"Potentially malicious path in zip: {info.filename}")
                        return False

                    # Check for reasonable uncompressed size
                    total_size += info.file_size
                    if total_size > 100 * 1024 * 1024:  # 100MB limit
                        LOGGER.warning("Zip file exceeds size safety limit")
                        return False

            return True
        except Exception as e:
            LOGGER.warning(f"Zip validation error: {e}")
            return False

    def _validate_snapshot(self, snapshot_path: Path) -> bool:
        """Validate the integrity of a snapshot file."""
        if not snapshot_path.exists():
            return False

        try:
            # Basic zip validation
            if not self._validate_zip(snapshot_path):
                return False

            # Check for required metadata file
            with zipfile.ZipFile(snapshot_path, 'r') as zipf:
                if '.aks_snapshot_meta.json' not in zipf.namelist():
                    LOGGER.warning("Snapshot missing metadata file")
                    return False

            return True
        except Exception as e:
            LOGGER.warning(f"Snapshot validation error: {e}")
            return False

    def restore_snapshot(self, snapshot_path: Path, verify: bool = True) -> bool:
        """
        Securely restore the repository from a snapshot.

        Args:
            snapshot_path: Path to the snapshot file
            verify: Whether to validate the snapshot before restoration

        Returns:
            True if restoration succeeded, False otherwise
        """
        snapshot_path = snapshot_path.resolve()

        if verify and not self._validate_snapshot(snapshot_path):
            LOGGER.error(f"Snapshot validation failed: {snapshot_path}")
            return False

        LOGGER.warning(f"Initiating restoration from snapshot: {snapshot_path.name}")

        # Create secure temporary extraction directory
        temp_extract = self.snapshot_dir / f"restore_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        temp_extract.mkdir(mode=0o700)

        try:
            # Step 1: Extract to temporary location
            with zipfile.ZipFile(snapshot_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract)

            # Step 2: Verify extracted content
            extracted_repo = temp_extract / "repo"
            if not extracted_repo.exists():
                LOGGER.error("Snapshot extraction failed - missing repo directory")
                return False

            # Step 3: Prepare existing repo - move to backup
            backup_path = self.snapshot_dir / f"pre_restore_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self._create_backup(backup_path)

            # Step 4: Clear existing repo (preserve .git)
            self._clear_repo_for_restore()

            # Step 5: Move files from temp to repo (atomic where possible)
            self._move_extracted_files(extracted_repo)

            LOGGER.info("Snapshot restored successfully")
            return True

        except Exception as e:
            LOGGER.error(f"Restoration failed: {e}", exc_info=True)
            return False
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_extract, ignore_errors=True)

    def _create_backup(self, backup_path: Path) -> bool:
        """Create backup of current repository state."""
        try:
            backup_path.mkdir()
            LOGGER.info(f"Creating pre-restore backup at {backup_path}")

            for item in self.repo_path.iterdir():
                if item.name != ".git":
                    dest = backup_path / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)

            return True
        except Exception as e:
            LOGGER.error(f"Backup creation failed: {e}")
            return False

    def _clear_repo_for_restore(self) -> None:
        """Clear repository contents while preserving git data."""
        for item in self.repo_path.iterdir():
            if item.name != ".git":
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception as e:
                    LOGGER.error(f"Failed to remove {item}: {e}")
                    raise

    def _move_extracted_files(self, source_dir: Path) -> None:
        """Move files from extracted snapshot to repository."""
        for item in source_dir.iterdir():
            dest = self.repo_path / item.name
            try:
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.move(str(item), str(dest))
            except Exception as e:
                LOGGER.error(f"Failed to move {item.name}: {e}")
                raise

    def get_available_snapshots(self) -> List[Dict[str, Any]]:
        """Get list of available snapshots with metadata."""
        snapshots = []

        for snapshot_file in self.snapshot_dir.glob("aks_snapshot_*.zip"):
            try:
                with zipfile.ZipFile(snapshot_file, 'r') as zipf:
                    if '.aks_snapshot_meta.json' in zipf.namelist():
                        with zipf.open('.aks_snapshot_meta.json') as meta_file:
                            metadata = json.load(meta_file)
                    else:
                        metadata = {}

                snapshots.append({
                    "path": snapshot_file,
                    "size": snapshot_file.stat().st_size,
                    "modified": os.path.getmtime(snapshot_file),
                    "metadata": metadata
                })
            except Exception as e:
                LOGGER.warning(f"Could not read metadata for {snapshot_file.name}: {e}")

        # Sort by modification time (newest first)
        return sorted(snapshots, key=lambda x: x["modified"], reverse=True)

    def cleanup_snapshots(self) -> None:
        """Remove oldest snapshots to maintain configured limit."""
        snapshots = self.get_available_snapshots()

        if len(snapshots) <= self.max_snapshots:
            return

        # Delete oldest snapshots
        for snapshot in snapshots[self.max_snapshots:]:
            try:
                snapshot["path"].unlink()
                LOGGER.info(f"Removed old snapshot: {snapshot['path'].name}")
            except Exception as e:
                LOGGER.error(f"Failed to remove snapshot {snapshot['path'].name}: {e}")

    def archive_old_snapshots(self) -> None:
        """Archive old snapshots to compressed archive."""
        snapshots = self.get_available_snapshots()
        if len(snapshots) <= self.max_snapshots:
            return

        # Create archive of old snapshots
        archive_name = f"aks_snapshots_archive_{datetime.now().strftime('%Y%m%d')}.zip"
        archive_path = self.snapshot_dir.parent / "archive" / archive_name
        archive_path.parent.mkdir(exist_ok=True)

        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for snapshot in snapshots[self.max_snapshots:]:
                    zipf.write(snapshot["path"], snapshot["path"].name)

            # Remove archived snapshots
            for snapshot in snapshots[self.max_snapshots:]:
                snapshot["path"].unlink()

            LOGGER.info(f"Archived {len(snapshots) - self.max_snapshots} snapshots to {archive_path}")
        except Exception as e:
            LOGGER.error(f"Failed to archive snapshots: {e}")
