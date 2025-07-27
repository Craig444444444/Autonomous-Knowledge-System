import logging
import time
import psutil
import platform
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Deque
from pathlib import Path
import socket
import threading
import json
from collections import deque

LOGGER = logging.getLogger("aks")

class Monitoring:
    """
    Comprehensive system monitoring for AKS with performance tracking,
    resource alerts, and health checks.
    """
    def __init__(self, config: Any):  # Changed type hint to Any
        """
        Initialize monitoring system with configuration.
        
        Args:
            config: Configuration object or dictionary
        """
        self.config = config
        self.metrics_history: Dict[str, Deque[Dict]] = {
            'cpu': deque(maxlen=60),
            'memory': deque(maxlen=60),
            'disk': deque(maxlen=60),
            'network': deque(maxlen=60)
        }
        self.alert_history: List[Dict] = []
        self._running: bool = False
        self._monitor_thread: Optional[threading.Thread] = None
        self.start_time: float = time.time()
        self._setup_directories()
        
        LOGGER.info("Monitoring system initialized")

    def _setup_directories(self):
        """Create required monitoring directories."""
        try:
            # Handle both dictionary and object configs
            if hasattr(self.config, 'monitoring_dir'):
                log_dir_value = self.config.monitoring_dir
            elif hasattr(self.config, 'get'):
                log_dir_value = self.config.get('monitoring_dir', '/content/logs/monitoring')
            else:
                log_dir_value = '/content/logs/monitoring'
                
            log_dir = Path(log_dir_value)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_dir.chmod(0o755)
            self.log_file = log_dir / 'aks_monitor.log'
            LOGGER.info(f"Created monitoring directory at {log_dir}")
        except Exception as e:
            LOGGER.error(f"Failed to setup monitoring directories: {e}")
            raise

    def start_monitoring(self, interval: int = 60):
        """
        Start continuous monitoring in background thread.
        
        Args:
            interval: Seconds between monitoring checks
        """
        if self._running:
            LOGGER.warning("Monitoring already running")
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        LOGGER.info(f"Started monitoring with {interval}s interval")

    def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self._running:
            return
            
        self._running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
            if self._monitor_thread.is_alive():
                LOGGER.warning("Monitoring thread did not stop gracefully")
        LOGGER.info("Monitoring stopped")

    def _monitor_loop(self, interval: int):
        """Main monitoring loop."""
        LOGGER.debug("Monitoring loop started")
        while self._running:
            try:
                metrics = self.collect_metrics()
                self._check_thresholds(metrics)
                self._log_metrics(metrics)
                time.sleep(interval)
            except Exception as e:
                LOGGER.error(f"Monitoring loop error: {e}")
                time.sleep(5)  # Prevent tight error loop

    def collect_metrics(self) -> Dict:
        """
        Collect comprehensive system metrics.
        
        Returns:
            Dictionary containing current system metrics
        """
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': self._get_system_info(),
                'cpu': self._get_cpu_metrics(),
                'memory': self._get_memory_metrics(),
                'disk': self._get_disk_metrics(),
                'network': self._get_network_metrics(),
                'aks': self._get_aks_metrics()
            }
            
            # Update history
            for key in ['cpu', 'memory', 'disk', 'network']:
                self.metrics_history[key].append(metrics[key])
                
            return metrics
        except Exception as e:
            LOGGER.error(f"Metrics collection failed: {e}")
            return {}

    def _get_system_info(self) -> Dict:
        """Get static system information."""
        try:
            return {
                'hostname': socket.gethostname(),
                'os': platform.system(),
                'os_version': platform.release(),
                'python_version': platform.python_version(),
                'uptime': int(time.time() - self.start_time)
            }
        except Exception as e:
            LOGGER.warning(f"System info collection failed: {e}")
            return {}

    def _get_cpu_metrics(self) -> Dict:
        """Collect CPU utilization metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_times = psutil.cpu_times_percent(interval=1)
            return {
                'percent': cpu_percent,
                'user': cpu_times.user,
                'system': cpu_times.system,
                'idle': cpu_times.idle,
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'load_avg': psutil.getloadavg()
            }
        except Exception as e:
            LOGGER.warning(f"CPU metrics failed: {e}")
            return {}

    def _get_memory_metrics(self) -> Dict:
        """Collect memory utilization metrics."""
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                'total': mem.total,
                'available': mem.available,
                'used': mem.used,
                'percent': mem.percent,
                'swap_total': swap.total,
                'swap_used': swap.used,
                'swap_percent': swap.percent
            }
        except Exception as e:
            LOGGER.warning(f"Memory metrics failed: {e}")
            return {}

    def _get_disk_metrics(self) -> Dict:
        """Collect disk I/O and space metrics."""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            return {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': disk_usage.percent,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count,
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes
            }
        except Exception as e:
            LOGGER.warning(f"Disk metrics failed: {e}")
            return {}

    def _get_network_metrics(self) -> Dict:
        """Collect network I/O metrics."""
        try:
            net_io = psutil.net_io_counters()
            connections = psutil.net_connections()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'active_connections': len(connections)
            }
        except Exception as e:
            LOGGER.warning(f"Network metrics failed: {e}")
            return {}

    def _get_aks_metrics(self) -> Dict:
        """Collect AKS-specific metrics."""
        try:
            # These would be populated from other AKS components
            return {
                'knowledge_items': 0,  # Should be updated from KnowledgeProcessor
                'active_tasks': 0,     # Should be updated from TaskScheduler
                'last_cycle_time': 0,  # Should be updated from AutonomousAgent
                'error_count': 0       # Should be updated from error tracking
            }
        except Exception as e:
            LOGGER.warning(f"AKS metrics failed: {e}")
            return {}

    def _check_thresholds(self, metrics: Dict):
        """Check metrics against configured thresholds and trigger alerts."""
        # Handle both dictionary and object configs
        if hasattr(self.config, 'thresholds'):
            thresholds = self.config.thresholds
        elif hasattr(self.config, 'get'):
            thresholds = self.config.get('thresholds', {})
        else:
            thresholds = {}
            
        alerts = []

        # CPU threshold check
        cpu_thresh = thresholds.get('cpu', 90) if isinstance(thresholds, dict) else getattr(thresholds, 'cpu', 90)
        if metrics['cpu'].get('percent', 0) > cpu_thresh:
            alerts.append(f"High CPU usage: {metrics['cpu']['percent']}%")

        # Memory threshold check
        mem_thresh = thresholds.get('memory', 90) if isinstance(thresholds, dict) else getattr(thresholds, 'memory', 90)
        if metrics['memory'].get('percent', 0) > mem_thresh:
            alerts.append(f"High memory usage: {metrics['memory']['percent']}%")

        # Disk threshold check
        disk_thresh = thresholds.get('disk', 90) if isinstance(thresholds, dict) else getattr(thresholds, 'disk', 90)
        if metrics['disk'].get('percent', 0) > disk_thresh:
            alerts.append(f"High disk usage: {metrics['disk']['percent']}%")

        # Log and store alerts
        for alert in alerts:
            timestamp = datetime.now().isoformat()
            alert_entry = {'timestamp': timestamp, 'message': alert}
            self.alert_history.append(alert_entry)
            LOGGER.warning(f"ALERT: {alert}")

    def _log_metrics(self, metrics: Dict):
        """Log metrics to file in JSON format."""
        try:
            with open(self.log_file, 'a') as f:
                json.dump(metrics, f)
                f.write('\n')
        except Exception as e:
            LOGGER.error(f"Failed to log metrics: {e}")

    def get_recent_metrics(self, minutes: int = 5) -> List[Dict]:
        """
        Get recent metrics from history.
        
        Args:
            minutes: Number of minutes of history to return
            
        Returns:
            List of metric snapshots
        """
        # This would need proper implementation based on storage backend
        return []

    def get_active_alerts(self) -> List[Dict]:
        """
        Get currently active alerts.
        
        Returns:
            List of active alert dictionaries
        """
        return self.alert_history[-10:]  # Return last 10 alerts

    def get_system_health(self) -> Dict:
        """
        Get overall system health assessment.
        
        Returns:
            Dictionary with health status and details
        """
        metrics = self.collect_metrics()
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'details': {}
        }

        # Check CPU health
        cpu_usage = metrics['cpu'].get('percent', 0)
        if cpu_usage > 90:
            health['status'] = 'degraded'
            health['details']['cpu'] = f"Critical: {cpu_usage}% usage"
        elif cpu_usage > 70:
            health['details']['cpu'] = f"Warning: {cpu_usage}% usage"
        else:
            health['details']['cpu'] = "Normal"

        # Check memory health
        mem_usage = metrics['memory'].get('percent', 0)
        if mem_usage > 90:
            health['status'] = 'degraded'
            health['details']['memory'] = f"Critical: {mem_usage}% usage"
        elif mem_usage > 70:
            health['details']['memory'] = f"Warning: {mem_usage}% usage"
        else:
            health['details']['memory'] = "Normal"

        # Check disk health
        disk_usage = metrics['disk'].get('percent', 0)
        if disk_usage > 90:
            health['status'] = 'degraded'
            health['details']['disk'] = f"Critical: {disk_usage}% usage"
        elif disk_usage > 70:
            health['details']['disk'] = f"Warning: {disk_usage}% usage"
        else:
            health['details']['disk'] = "Normal"

        return health

    def generate_report(self, duration_hours: int = 24) -> Dict:
        """
        Generate a system health report.
        
        Args:
            duration_hours: Time period to cover in report
            
        Returns:
            Dictionary containing formatted report
        """
        # This would analyze historical data and generate insights
        return {
            'period': f"Last {duration_hours} hours",
            'status': 'healthy',
            'metrics_analyzed': 0,
            'alerts_triggered': len(self.alert_history),
            'recommendations': []
        }
