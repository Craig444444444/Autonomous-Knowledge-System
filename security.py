import logging
import hashlib
import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import tempfile
import json
import stat
import subprocess
import tarfile
import zipfile

LOGGER = logging.getLogger("aks")

class SecurityManager:
    """
    Comprehensive security management for the Autonomous Knowledge System (AKS).
    Handles system integrity checks, vulnerability scanning, and security monitoring.
    """
    def __init__(self):
        self.suspicious_patterns = [
            # System commands
            (r'(?:;|\||&|\$\(|`)\s*(?:rm\s+-[rf]|wget|curl|chmod|sudo|shutdown|reboot)', 
             "Dangerous system command"),
            
            # Network operations
            (r'(?:socket\.|http\.|requests\.|urllib\.)', 
             "Network operation detected"),
            
            # File system operations
            (r'(?:open\(|write\(|os\.(?:remove|rename|system|popen))', 
             "Suspicious file operation"),
            
            # Code execution
            (r'(?:exec\(|eval\(|compile\(|__import__\()', 
             "Dynamic code execution"),
            
            # Environment manipulation
            (r'(?:os\.environ|subprocess\.|sys\.)', 
             "Environment manipulation"),
            
            # Suspicious strings
            (r'(?:password|secret|key|token)\s*=', 
             "Potential credential exposure")
        ]
        
        self.whitelisted_domains = {
            'wikipedia.org',
            'github.com',
            'stackoverflow.com',
            'arxiv.org',
            'python.org',
            'pypi.org'
        }
        
        self.known_vulnerable_libraries = {
            'pickle': 'Use json instead for safe serialization',
            'yaml.load': 'Use yaml.safe_load instead',
            'marshal': 'Potentially unsafe serialization',
            'subprocess': 'Use with shell=False'
        }
        
        LOGGER.info("Security Manager initialized")

    def perform_system_integrity_checks(self) -> Dict[str, List[str]]:
        """
        Perform comprehensive system integrity checks.
        Returns dictionary of check categories with findings.
        """
        results = {
            'file_integrity': [],
            'code_analysis': [],
            'dependency_checks': [],
            'configuration_issues': [],
            'suspicious_content': []
        }
        
        try:
            LOGGER.info("Starting system integrity checks")
            
            # Check file permissions
            results['file_integrity'].extend(self._check_file_permissions())
            
            # Analyze code for vulnerabilities
            results['code_analysis'].extend(self._scan_for_vulnerable_patterns())
            
            # Check dependencies
            results['dependency_checks'].extend(self._check_dependencies())
            
            # Verify configuration
            results['configuration_issues'].extend(self._check_configurations())
            
            # Scan for suspicious content
            results['suspicious_content'].extend(self._scan_for_malicious_content())
            
            LOGGER.info(f"Completed system integrity checks with {sum(len(v) for v in results.values())} findings")
            return results
            
        except Exception as e:
            LOGGER.error(f"System integrity check failed: {e}")
            return {'error': [f"Check failed: {str(e)}"]}

    def _check_file_permissions(self) -> List[str]:
        """Check for insecure file permissions."""
        findings = []
        sensitive_dirs = [
            '/content',
            '/content/knowledge_base',
            '/content/snapshots',
            '/content/logs'
        ]
        
        for dir_path in sensitive_dirs:
            try:
                path = Path(dir_path)
                if not path.exists():
                    continue
                    
                mode = path.stat().st_mode
                if mode & stat.S_IROTH or mode & stat.S_IWOTH:
                    findings.append(f"Insecure permissions on {dir_path} (world readable/writable)")
                    
                for root, dirs, files in os.walk(dir_path):
                    for name in files + dirs:
                        item_path = Path(root) / name
                        try:
                            item_mode = item_path.stat().st_mode
                            if item_mode & stat.S_IROTH or item_mode & stat.S_IWOTH:
                                findings.append(f"Insecure permissions on {item_path} (world readable/writable)")
                        except Exception:
                            continue
                            
            except Exception as e:
                LOGGER.warning(f"Could not check permissions for {dir_path}: {e}")
                
        return findings

    def _scan_for_vulnerable_patterns(self, scan_dir: str = '/content') -> List[str]:
        """Scan code for known vulnerable patterns."""
        findings = []
        python_files = list(Path(scan_dir).rglob('*.py'))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for suspicious code patterns
                    for pattern, message in self.suspicious_patterns:
                        if re.search(pattern, content):
                            findings.append(f"{message} in {file_path}")
                            
                    # Check for known vulnerable libraries
                    for lib, warning in self.known_vulnerable_libraries.items():
                        if lib in content:
                            findings.append(f"Potential vulnerability: {warning} in {file_path}")
                            
                    # Validate AST for dangerous constructs
                    try:
                        tree = ast.parse(content)
                        self._analyze_ast(tree, file_path, findings)
                    except SyntaxError as e:
                        findings.append(f"Syntax error in {file_path}: {e}")
                        
            except Exception as e:
                LOGGER.warning(f"Could not scan {file_path}: {e}")
                
        return findings

    def _analyze_ast(self, node, file_path: str, findings: List[str]) -> None:
        """Recursively analyze AST for security issues."""
        if isinstance(node, ast.Call):
            # Check for dangerous function calls
            if (isinstance(node.func, ast.Name) and 
                node.func.id in ['eval', 'exec', 'open']):
                findings.append(f"Dangerous function call {node.func.id} in {file_path}")
                
        for child in ast.iter_child_nodes(node):
            self._analyze_ast(child, file_path, findings)

    def _check_dependencies(self) -> List[str]:
        """Check installed dependencies for known vulnerabilities."""
        findings = []
        try:
            result = subprocess.run(
                [os.sys.executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                for pkg in packages:
                    # In a real system, you'd check against a vulnerability database
                    if pkg['name'].lower() in ['django', 'flask', 'requests']:
                        findings.append(
                            f"Package {pkg['name']} {pkg['version']} - "
                            "Check for known vulnerabilities"
                        )
        except Exception as e:
            LOGGER.warning(f"Could not check dependencies: {e}")
            
        return findings

    def _check_configurations(self) -> List[str]:
        """Check system configurations for security issues."""
        findings = []
        
        # Check environment variables
        sensitive_vars = ['AKS_SECRET', 'GITHUB_TOKEN', 'GEMINI_KEY']
        for var in sensitive_vars:
            if var in os.environ:
                findings.append(f"Sensitive environment variable {var} is set")
                
        # Check file configurations
        config_files = [
            '/content/config.json',
            '/content/credentials.ini'
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                findings.append(f"Configuration file {config_file} should be secured")
                
        return findings

    def _scan_for_malicious_content(self) -> List[str]:
        """Scan knowledge base for potentially malicious content."""
        findings = []
        try:
            kb_dir = Path('/content/knowledge_base')
            for file_path in kb_dir.rglob('*'):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        for pattern, message in self.suspicious_patterns:
                            if re.search(pattern, content):
                                findings.append(f"{message} in knowledge file {file_path.name}")
                    except Exception:
                        continue
        except Exception as e:
            LOGGER.warning(f"Could not scan knowledge base: {e}")
            
        return findings

    def analyze_logs_for_anomalies(self, log_data: List[Dict]) -> Dict[str, List[str]]:
        """
        Analyze system logs for security anomalies.
        
        Args:
            log_data: List of log entries from AuditManager
            
        Returns:
            Dictionary of anomaly categories with findings
        """
        anomalies = {
            'failed_logins': [],
            'access_denied': [],
            'unusual_activity': [],
            'error_spikes': []
        }
        
        try:
            error_counts = {}
            last_timestamp = None
            
            for entry in log_data:
                # Check for failed authentication
                if 'authentication failed' in entry.get('message', '').lower():
                    anomalies['failed_logins'].append(entry)
                
                # Check for access denied
                if 'permission denied' in entry.get('message', '').lower():
                    anomalies['access_denied'].append(entry)
                
                # Track error frequencies
                if entry.get('level', '').upper() == 'ERROR':
                    error_key = entry.get('module', 'unknown')
                    error_counts[error_key] = error_counts.get(error_key, 0) + 1
                
                # Check timestamp sequence
                if last_timestamp and entry.get('timestamp'):
                    try:
                        time_diff = (entry['timestamp'] - last_timestamp).total_seconds()
                        if time_diff < 0:
                            anomalies['unusual_activity'].append(
                                f"Log timestamp anomaly at {entry.get('timestamp')}"
                            )
                    except Exception:
                        pass
                last_timestamp = entry.get('timestamp')
            
            # Identify error spikes
            avg_errors = sum(error_counts.values()) / len(error_counts) if error_counts else 0
            for module, count in error_counts.items():
                if count > avg_errors * 3:  # 3x average is a spike
                    anomalies['error_spikes'].append(
                        f"Error spike in {module}: {count} errors"
                    )
                    
            LOGGER.info(f"Found {sum(len(v) for v in anomalies.values())} log anomalies")
            return anomalies
            
        except Exception as e:
            LOGGER.error(f"Log analysis failed: {e}")
            return {'error': [f"Analysis failed: {str(e)}"]}

    def validate_url(self, url: str) -> Tuple[bool, str]:
        """
        Validate a URL for security purposes.
        
        Args:
            url: URL to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Basic URL structure
            if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', url, re.IGNORECASE):
                return False, "Invalid URL format"
            
            # Domain validation
            domain = re.match(r'^https?://([^/]+)', url).group(1).lower()
            if any(bad in domain for bad in ['localhost', '127.0.0.1', '192.168.', '10.']):
                return False, "Local network URL not allowed"
                
            # Whitelist check
            if not any(good in domain for good in self.whitelisted_domains):
                return False, f"Domain {domain} not in whitelist"
                
            return True, "Valid URL"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def sanitize_input(self, input_str: str) -> str:
        """
        Sanitize user input to prevent injection attacks.
        
        Args:
            input_str: Raw input string
            
        Returns:
            Sanitized string safe for processing
        """
        if not input_str:
            return ""
            
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', input_str)
        
        # Escape special characters for different contexts
        sanitized = (sanitized.replace('\\', '\\\\')
                              .replace('"', '\\"')
                              .replace("'", "\\'")
                              .replace('`', '\\`')
                              .replace('$', '\\$'))
                              
        # Truncate to reasonable length
        return sanitized[:1000]

    def verify_file_integrity(self, file_path: Path) -> Tuple[bool, str]:
        """
        Verify the integrity of a file.
        
        Args:
            file_path: Path to file to verify
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            if not file_path.exists():
                return False, "File does not exist"
                
            # Check file type
            if file_path.suffix.lower() in ['.zip', '.tar', '.gz']:
                return self._verify_archive(file_path)
                
            # Check for suspicious content in text files
            if file_path.suffix.lower() in ['.txt', '.py', '.json', '.md']:
                return self._verify_text_file(file_path)
                
            return True, "File appears valid"
            
        except Exception as e:
            return False, f"Verification failed: {str(e)}"

    def _verify_archive(self, file_path: Path) -> Tuple[bool, str]:
        """Verify archive file integrity."""
        try:
            if file_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(file_path) as zf:
                    if zf.testzip() is not None:
                        return False, "Corrupt ZIP file"
                    for name in zf.namelist():
                        if '..' in name or name.startswith('/'):
                            return False, f"Invalid path in archive: {name}"
            elif file_path.suffix.lower() in ['.tar', '.gz']:
                with tarfile.open(file_path) as tf:
                    for member in tf.getmembers():
                        if member.islnk() or member.issym():
                            return False, "Archive contains symbolic links"
                        if '..' in member.name or member.name.startswith('/'):
                            return False, f"Invalid path in archive: {member.name}"
            return True, "Archive appears valid"
        except Exception as e:
            return False, f"Archive verification failed: {str(e)}"

    def _verify_text_file(self, file_path: Path) -> Tuple[bool, str]:
        """Verify text file integrity."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Check for suspicious patterns
            for pattern, _ in self.suspicious_patterns:
                if re.search(pattern, content):
                    return False, f"File matches suspicious pattern: {pattern}"
                    
            # Check for binary content
            if '\x00' in content:
                return False, "File contains binary data"
                
            return True, "Text file appears valid"
        except Exception as e:
            return False, f"Text file verification failed: {str(e)}"
