import logging
import json
import shutil
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
import importlib.util

LOGGER = logging.getLogger("aks")

class VersionMigrator:
    """
    Handles version migration and compatibility management for the AKS system.
    Provides safe upgrades/downgrades between different versions.
    """
    
    def __init__(self, repo_path: Path, config_path: Path):
        """
        Initialize the VersionMigrator.
        
        Args:
            repo_path: Path to the main repository
            config_path: Path to configuration directory
        """
        self.repo_path = repo_path.resolve()
        self.config_path = config_path.resolve()
        self.migration_scripts_dir = self.repo_path / "migrations"
        self._setup_directories()
        
        # Version tracking
        self.current_version = self._detect_current_version()
        self.available_migrations = self._scan_migration_scripts()
        
        LOGGER.info(f"VersionMigrator initialized. Current version: {self.current_version}")

    def _setup_directories(self) -> None:
        """Ensure required directories exist."""
        try:
            self.migration_scripts_dir.mkdir(parents=True, exist_ok=True)
            (self.config_path / "version_history").mkdir(exist_ok=True)
            LOGGER.debug("Verified/Created migration directories")
        except Exception as e:
            LOGGER.error(f"Directory setup failed: {e}")
            raise RuntimeError("Could not initialize migration directories") from e

    def _detect_current_version(self) -> str:
        """Detect the current system version from config files."""
        version_file = self.config_path / "version.json"
        
        try:
            if version_file.exists():
                with version_file.open() as f:
                    data = json.load(f)
                    return data.get("version", "0.0.0")
            
            # Fallback for fresh installations
            return "0.0.0"
        except Exception as e:
            LOGGER.error(f"Version detection failed: {e}")
            return "0.0.0"

    def _scan_migration_scripts(self) -> Dict[str, Dict[str, str]]:
        """
        Scan the migrations directory for available scripts.
        
        Returns:
            Dictionary mapping version strings to migration script paths
            {
                "1.2.3": {
                    "upgrade": "path/to/upgrade_script.py",
                    "downgrade": "path/to/downgrade_script.py"
                }
            }
        """
        migrations = {}
        pattern = re.compile(r"^(upgrade|downgrade)_([0-9]+\.[0-9]+\.[0-9]+)\.py$")
        
        try:
            for script in self.migration_scripts_dir.glob("*.py"):
                match = pattern.match(script.name)
                if match:
                    direction = match.group(1)
                    version = match.group(2)
                    
                    if version not in migrations:
                        migrations[version] = {}
                    migrations[version][direction] = str(script)
            
            LOGGER.info(f"Found {len(migrations)} version migrations available")
            return migrations
        except Exception as e:
            LOGGER.error(f"Migration script scan failed: {e}")
            return {}

    def get_migration_path(self, target_version: str) -> Optional[List[Tuple[str, str]]]:
        """
        Calculate the migration path from current to target version.
        
        Args:
            target_version: Version string to migrate to
            
        Returns:
            List of (version, direction) tuples representing the migration steps
            or None if migration path cannot be determined
        """
        try:
            current = self._parse_version(self.current_version)
            target = self._parse_version(target_version)
            
            if current == target:
                LOGGER.info("Already at target version")
                return []
                
            steps = []
            
            if current < target:
                # Upgrade path
                for version in sorted(self.available_migrations.keys()):
                    v = self._parse_version(version)
                    if current < v <= target:
                        if "upgrade" in self.available_migrations[version]:
                            steps.append((version, "upgrade"))
            else:
                # Downgrade path
                for version in sorted(self.available_migrations.keys(), reverse=True):
                    v = self._parse_version(version)
                    if current >= v > target:
                        if "downgrade" in self.available_migrations[version]:
                            steps.append((version, "downgrade"))
            
            if not steps:
                LOGGER.warning("No valid migration path found")
                return None
                
            return steps
        except Exception as e:
            LOGGER.error(f"Migration path calculation failed: {e}")
            return None

    def _parse_version(self, version_str: str) -> Tuple[int, int, int]:
        """Parse version string into (major, minor, patch) tuple."""
        try:
            return tuple(map(int, version_str.split('.')))
        except Exception as e:
            LOGGER.error(f"Invalid version string: {version_str} - {e}")
            return (0, 0, 0)

    def migrate(self, target_version: str, dry_run: bool = False) -> bool:
        """
        Execute migration to target version.
        
        Args:
            target_version: Version string to migrate to
            dry_run: If True, only simulate the migration
            
        Returns:
            True if migration succeeded or was not needed,
            False if migration failed
        """
        try:
            # Check if migration is needed
            if self.current_version == target_version:
                LOGGER.info(f"Already at version {target_version}")
                return True
                
            # Calculate migration path
            steps = self.get_migration_path(target_version)
            if steps is None:
                LOGGER.error("Cannot determine migration path")
                return False
                
            if not steps:
                LOGGER.info("No migration steps required")
                return True
                
            LOGGER.info(f"Migration path: {' -> '.join([f'{v} ({d})' for v, d in steps])}")
            
            if dry_run:
                LOGGER.info("Dry run completed successfully")
                return True
                
            # Create backup before migration
            backup_path = self._create_backup()
            if not backup_path:
                LOGGER.error("Backup creation failed - aborting migration")
                return False
                
            # Execute each migration step
            for version, direction in steps:
                success = self._execute_migration_script(version, direction)
                if not success:
                    LOGGER.error(f"Migration failed at {version} {direction}")
                    self._rollback_migration(backup_path)
                    return False
                    
                # Update version tracking after each successful step
                self._update_version_tracking(version, direction)
            
            LOGGER.info(f"Successfully migrated to version {target_version}")
            return True
        except Exception as e:
            LOGGER.error(f"Migration process failed: {e}")
            return False

    def _execute_migration_script(self, version: str, direction: str) -> bool:
        """Execute a single migration script."""
        try:
            script_path = self.available_migrations[version][direction]
            LOGGER.info(f"Executing {direction} script for version {version}")
            
            # Dynamically import the migration module
            spec = importlib.util.spec_from_file_location(
                f"migration_{version}_{direction}", 
                script_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Execute the migration
            result = module.migrate(self.repo_path, self.config_path)
            
            if not result:
                LOGGER.error(f"Migration script {script_path} returned failure")
                return False
                
            return True
        except Exception as e:
            LOGGER.error(f"Migration script execution failed: {e}")
            return False

    def _create_backup(self) -> Optional[Path]:
        """Create backup of current state before migration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.config_path / "backups" / f"pre_migration_{timestamp}"
        
        try:
            backup_dir.mkdir(parents=True)
            
            # Backup configuration
            shutil.copytree(
                self.config_path,
                backup_dir / "config",
                ignore=shutil.ignore_patterns('backups', 'version_history')
            )
            
            # Backup critical repository files
            shutil.copytree(
                self.repo_path,
                backup_dir / "repo",
                ignore=shutil.ignore_patterns(
                    '__pycache__', '*.pyc', '.git', 
                    '*.tmp', '*.bak', '*.log'
                )
            )
            
            # Create backup manifest
            manifest = {
                "timestamp": timestamp,
                "original_version": self.current_version,
                "backup_hash": self._hash_directory(backup_dir)
            }
            
            with (backup_dir / "manifest.json").open('w') as f:
                json.dump(manifest, f, indent=2)
                
            LOGGER.info(f"Created migration backup at {backup_dir}")
            return backup_dir
        except Exception as e:
            LOGGER.error(f"Backup creation failed: {e}")
            return None

    def _hash_directory(self, directory: Path) -> str:
        """Generate hash of directory contents."""
        hasher = hashlib.sha256()
        
        for root, _, files in os.walk(directory):
            for file in sorted(files):
                file_path = Path(root) / file
                hasher.update(file_path.read_bytes())
                
        return hasher.hexdigest()

    def _rollback_migration(self, backup_path: Path) -> bool:
        """Restore system state from backup after failed migration."""
        try:
            LOGGER.warning(f"Attempting rollback from backup {backup_path}")
            
            # Restore configuration
            shutil.rmtree(self.config_path)
            shutil.copytree(
                backup_path / "config",
                self.config_path
            )
            
            # Restore repository files
            shutil.rmtree(self.repo_path)
            shutil.copytree(
                backup_path / "repo",
                self.repo_path
            )
            
            LOGGER.info("Rollback completed successfully")
            return True
        except Exception as e:
            LOGGER.critical(f"Rollback failed: {e}")
            return False

    def _update_version_tracking(self, version: str, direction: str) -> None:
        """Update version tracking after successful migration step."""
        try:
            # Update current version file
            version_file = self.config_path / "version.json"
            with version_file.open('w') as f:
                json.dump({"version": version}, f, indent=2)
                
            # Record in version history
            history_file = self.config_path / "version_history" / "migrations.log"
            with history_file.open('a') as f:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "from_version": self.current_version,
                    "to_version": version,
                    "direction": direction
                }
                f.write(json.dumps(entry) + "\n")
                
            self.current_version = version
            LOGGER.debug(f"Updated version tracking to {version}")
        except Exception as e:
            LOGGER.error(f"Version tracking update failed: {e}")

    def validate_system_state(self) -> bool:
        """Validate system state against expected version configuration."""
        try:
            # Check version consistency
            detected_version = self._detect_current_version()
            if detected_version != self.current_version:
                LOGGER.error(
                    f"Version mismatch: config says {self.current_version} "
                    f"but detected {detected_version}"
                )
                return False
                
            # TODO: Add more comprehensive validation checks
            return True
        except Exception as e:
            LOGGER.error(f"System validation failed: {e}")
            return False

    def create_migration_template(self, version: str) -> bool:
        """
        Create template migration scripts for a new version.
        
        Args:
            version: Target version for the migration
            
        Returns:
            True if templates were created successfully
        """
        try:
            # Validate version format
            if not re.match(r"^\d+\.\d+\.\d+$", version):
                LOGGER.error(f"Invalid version format: {version}")
                return False
                
            # Create upgrade template
            upgrade_path = self.migration_scripts_dir / f"upgrade_{version}.py"
            if not upgrade_path.exists():
                with upgrade_path.open('w') as f:
                    f.write(MIGRATION_TEMPLATE.format(
                        version=version,
                        direction="upgrade",
                        description=f"Upgrade to version {version}"
                    ))
                    
            # Create downgrade template
            downgrade_path = self.migration_scripts_dir / f"downgrade_{version}.py"
            if not downgrade_path.exists():
                with downgrade_path.open('w') as f:
                    f.write(MIGRATION_TEMPLATE.format(
                        version=version,
                        direction="downgrade",
                        description=f"Downgrade from version {version}"
                    ))
                    
            LOGGER.info(f"Created migration templates for version {version}")
            return True
        except Exception as e:
            LOGGER.error(f"Template creation failed: {e}")
            return False

# Migration script template
MIGRATION_TEMPLATE = '''"""
{description}
Migration script for AKS system.
"""

import logging
from pathlib import Path
import shutil
import json

def migrate(repo_path: Path, config_path: Path) -> bool:
    """
    Execute the migration.
    
    Args:
        repo_path: Path to the main repository
        config_path: Path to configuration directory
        
    Returns:
        True if migration succeeded, False otherwise
    """
    logger = logging.getLogger("aks")
    logger.info("Starting {direction} to version {version}")
    
    try:
        # =============================================
        # IMPLEMENT MIGRATION LOGIC HERE
        #
        # Example operations:
        # - Rename/move files or directories
        # - Modify configuration files
        # - Transform data formats
        # - Update database schemas
        # =============================================
        
        logger.info("{direction} to version {version} completed successfully")
        return True
    except Exception as e:
        logger.error(f"{direction} failed: {{e}}")
        return False
'''
