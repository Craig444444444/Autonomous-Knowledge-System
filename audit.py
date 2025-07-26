import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import threading

LOGGER = logging.getLogger("aks")

class AuditManager:
    """
    Comprehensive auditing system for tracking system activities, changes,
    and security events with integrity verification.
    """
    def __init__(self, repo_path: Path):
        """
        Initialize the AuditManager with secure log handling.

        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = repo_path.resolve()
        self.audit_log_dir = self.repo_path / "audit_logs"
        self.current_log_file = self.audit_log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.json"
        self._lock = threading.Lock()
        self._setup_directories()

    def _setup_directories(self):
        """Initialize audit log directory with secure permissions."""
        try:
            self.audit_log_dir.mkdir(parents=True, exist_ok=True)
            self.audit_log_dir.chmod(0o750)  # Restrict permissions
            LOGGER.info("Audit log directory initialized")
        except Exception as e:
            LOGGER.error(f"Failed to initialize audit directory: {e}")
            raise RuntimeError("Audit system initialization failed") from e

    def log_event(
        self,
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        severity: str = "INFO"
    ) -> bool:
        """
        Record an audit event with integrity verification.

        Args:
            event_type: Category of event (e.g., "security", "system", "git")
            description: Human-readable description
            metadata: Additional contextual data
            severity: Severity level (INFO, WARNING, ERROR, CRITICAL)

        Returns:
            bool: True if logging succeeded
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "description": description,
            "severity": severity,
            "metadata": metadata or {},
            "system": {
                "python_version": ".".join(map(str, sys.version_info[:3])),
                "platform": sys.platform
            }
        }

        # Calculate content hash for integrity verification
        event_str = json.dumps(event, sort_keys=True)
        event["integrity_hash"] = self._calculate_hash(event_str)

        try:
            with self._lock:
                # Create new log file if it's a new day or doesn't exist
                if not self.current_log_file.exists() or \
                   datetime.now().date() != datetime.fromtimestamp(
                       self.current_log_file.stat().st_mtime).date():
                    self.current_log_file = (
                        self.audit_log_dir / 
                        f"audit_{datetime.now().strftime('%Y%m%d')}.json"
                    )

                # Append to log file with integrity checks
                if self.current_log_file.exists():
                    with open(self.current_log_file, "r+", encoding="utf-8") as f:
                        try:
                            existing_data = json.load(f)
                            if not isinstance(existing_data, list):
                                raise ValueError("Invalid audit log format")
                        except (json.JSONDecodeError, ValueError):
                            existing_data = []
                            LOGGER.warning("Reset corrupted audit log file")

                        # Verify existing entries' integrity
                        for entry in existing_data:
                            if not self._verify_integrity(entry):
                                LOGGER.error("Audit log integrity check failed")
                                return False

                        existing_data.append(event)
                        f.seek(0)
                        json.dump(existing_data, f, indent=2)
                        f.truncate()
                else:
                    with open(self.current_log_file, "w", encoding="utf-8") as f:
                        json.dump([event], f, indent=2)

                # Set secure permissions
                self.current_log_file.chmod(0o640)
                return True

        except Exception as e:
            LOGGER.error(f"Failed to log audit event: {e}")
            return False

    def _calculate_hash(self, data: str) -> str:
        """Generate SHA-256 hash of data for integrity verification."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _verify_integrity(self, entry: Dict[str, Any]) -> bool:
        """
        Verify the integrity of a log entry by recalculating its hash.
        
        Args:
            entry: The audit log entry to verify
            
        Returns:
            bool: True if integrity is valid
        """
        if "integrity_hash" not in entry:
            return False

        # Create copy without the existing hash for recalculation
        entry_copy = entry.copy()
        original_hash = entry_copy.pop("integrity_hash")
        entry_str = json.dumps(entry_copy, sort_keys=True)
        return self._calculate_hash(entry_str) == original_hash

    def load_audit_log(
        self, 
        days: int = 7,
        verify_integrity: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load audit log entries with optional integrity verification.

        Args:
            days: Number of days of logs to load
            verify_integrity: Whether to verify entry hashes

        Returns:
            List of audit log entries
        """
        logs = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            for log_file in sorted(self.audit_log_dir.glob("audit_*.json")):
                file_date = datetime.strptime(
                    log_file.stem.split("_")[1], 
                    "%Y%m%d"
                ).date()
                
                if start_date.date() <= file_date <= end_date.date():
                    with open(log_file, "r", encoding="utf-8") as f:
                        try:
                            entries = json.load(f)
                            if verify_integrity:
                                for entry in entries:
                                    if not self._verify_integrity(entry):
                                        LOGGER.error(
                                            f"Integrity check failed for entry in {log_file.name}"
                                        )
                                        continue
                                    logs.append(entry)
                            else:
                                logs.extend(entries)
                        except json.JSONDecodeError:
                            LOGGER.error(f"Corrupted audit log file: {log_file.name}")
                            continue

            # Sort all entries by timestamp
            logs.sort(key=lambda x: x["timestamp"])
            return logs

        except Exception as e:
            LOGGER.error(f"Failed to load audit logs: {e}")
            return []

    def analyze_log_for_anomalies(self) -> Dict[str, Any]:
        """
        Analyze audit logs for suspicious patterns or anomalies.

        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "security_events": 0,
            "error_events": 0,
            "suspicious_patterns": [],
            "statistics": {
                "events_per_day": {},
                "events_per_type": {}
            }
        }

        try:
            logs = self.load_audit_log(days=30)
            if not logs:
                return analysis

            # Basic statistics
            for entry in logs:
                date = entry["timestamp"][:10]
                analysis["statistics"]["events_per_day"][date] = (
                    analysis["statistics"]["events_per_day"].get(date, 0) + 1
                )
                analysis["statistics"]["events_per_type"][entry["event_type"]] = (
                    analysis["statistics"]["events_per_type"].get(entry["event_type"], 0) + 1
                )

                # Count security and error events
                if entry["event_type"] == "security":
                    analysis["security_events"] += 1
                if entry["severity"] in ("ERROR", "CRITICAL"):
                    analysis["error_events"] += 1

            # Pattern detection (simplified example)
            failed_logins = sum(
                1 for e in logs 
                if e.get("event_type") == "security" 
                and "failed login" in e.get("description", "").lower()
            )
            if failed_logins > 5:
                analysis["suspicious_patterns"].append(
                    f"Multiple failed login attempts ({failed_logins})"
                )

            return analysis

        except Exception as e:
            LOGGER.error(f"Log analysis failed: {e}")
            return analysis

    def export_audit_log(
        self,
        output_path: Path,
        days: int = 30,
        format: str = "json"
    ) -> bool:
        """
        Export audit logs to external file.

        Args:
            output_path: Destination path for export
            days: Number of days to export
            format: Export format (json or csv)

        Returns:
            bool: True if export succeeded
        """
        try:
            logs = self.load_audit_log(days=days)
            if not logs:
                LOGGER.warning("No audit logs to export")
                return False

            if format == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(logs, f, indent=2)
            elif format == "csv":
                import csv
                fieldnames = [
                    "timestamp", "event_type", "severity", 
                    "description", "integrity_hash"
                ]
                with open(output_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for entry in logs:
                        writer.writerow({k: v for k, v in entry.items() 
                                       if k in fieldnames})
            else:
                LOGGER.error(f"Unsupported export format: {format}")
                return False

            # Set secure permissions
            output_path.chmod(0o640)
            LOGGER.info(f"Exported {len(logs)} audit entries to {output_path}")
            return True

        except Exception as e:
            LOGGER.error(f"Audit log export failed: {e}")
            return False
