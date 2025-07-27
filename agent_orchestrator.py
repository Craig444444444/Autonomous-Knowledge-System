import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set  # Added Any import
from pathlib import Path
import json
import hashlib

LOGGER = logging.getLogger("aks")

class AgentOrchestrator:
    """
    Central coordination system for managing autonomous agents in the AKS ecosystem.
    Handles agent lifecycle, task distribution, and system-wide coordination.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Agent Orchestrator with system configuration.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.task_queue: List[Dict[str, Any]] = []
        self.agent_lock = threading.RLock()
        self.task_lock = threading.Lock()
        self.heartbeat_interval = 30  # seconds
        self.last_cleanup = datetime.now()
        self._setup_directories()
        self._load_persistent_state()
        
        # Start background threads
        self._running = True
        threading.Thread(target=self._monitor_agents, daemon=True).start()
        threading.Thread(target=self._process_tasks, daemon=True).start()
        
        LOGGER.info("Agent Orchestrator initialized")

    def _setup_directories(self):
        """Create required directories for agent management."""
        self.agent_dir = Path(self.config.get('agent_dir', '/content/agents'))
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.agent_dir / 'orchestrator_state.json'
        self.task_log = self.agent_dir / 'task_history.log'

    def _load_persistent_state(self):
        """Load persistent state from disk if available."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.agents = state.get('agents', {})
                    self.task_queue = state.get('tasks', [])
                LOGGER.info(f"Loaded state for {len(self.agents)} agents and {len(self.task_queue)} queued tasks")
        except Exception as e:
            LOGGER.error(f"Failed to load persistent state: {e}")

    def _save_persistent_state(self):
        """Save current state to disk."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'agents': self.agents,
                    'tasks': self.task_queue,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            LOGGER.error(f"Failed to save persistent state: {e}")

    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str]) -> bool:
        """
        Register a new agent with the orchestrator.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type/class of the agent
            capabilities: List of capabilities the agent provides
            
        Returns:
            bool: True if registration succeeded
        """
        with self.agent_lock:
            if agent_id in self.agents:
                LOGGER.warning(f"Agent {agent_id} already registered")
                return False
                
            self.agents[agent_id] = {
                'type': agent_type,
                'capabilities': capabilities,
                'last_heartbeat': datetime.now().isoformat(),
                'status': 'idle',
                'current_task': None,
                'stats': {
                    'tasks_completed': 0,
                    'errors': 0,
                    'uptime': 0
                }
            }
            LOGGER.info(f"Registered new agent: {agent_id} ({agent_type})")
            self._save_persistent_state()
            return True

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the orchestrator.
        
        Args:
            agent_id: ID of the agent to unregister
            
        Returns:
            bool: True if unregistration succeeded
        """
        with self.agent_lock:
            if agent_id not in self.agents:
                LOGGER.warning(f"Agent {agent_id} not found")
                return False
                
            if self.agents[agent_id]['status'] != 'idle':
                LOGGER.warning(f"Cannot unregister busy agent {agent_id}")
                return False
                
            del self.agents[agent_id]
            LOGGER.info(f"Unregistered agent: {agent_id}")
            self._save_persistent_state()
            return True

    def update_agent_heartbeat(self, agent_id: str) -> bool:
        """
        Update agent heartbeat timestamp.
        
        Args:
            agent_id: ID of the agent to update
            
        Returns:
            bool: True if agent exists and was updated
        """
        with self.agent_lock:
            if agent_id not in self.agents:
                LOGGER.warning(f"Heartbeat update failed - agent {agent_id} not found")
                return False
                
            self.agents[agent_id]['last_heartbeat'] = datetime.now().isoformat()
            return True

    def submit_task(self, task_type: str, payload: Dict, priority: int = 1) -> str:
        """
        Submit a new task to the orchestrator's queue.
        
        Args:
            task_type: Type of task (must match agent capabilities)
            payload: Task data payload
            priority: Task priority (1-10, 1=highest)
            
        Returns:
            str: Unique task ID
        """
        task_id = hashlib.sha256(
            f"{task_type}{json.dumps(payload)}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        with self.task_lock:
            self.task_queue.append({
                'id': task_id,
                'type': task_type,
                'payload': payload,
                'priority': min(max(1, priority), 10),
                'status': 'queued',
                'created': datetime.now().isoformat(),
                'assigned_to': None
            })
            self._log_task(f"Task {task_id} queued ({task_type})")
            self._save_persistent_state()
            
        return task_id

    def get_agent_status(self, agent_id: str) -> Optional[Dict]:
        """
        Get current status of an agent.
        
        Args:
            agent_id: ID of the agent to query
            
        Returns:
            Optional[Dict]: Agent status dict or None if not found
        """
        with self.agent_lock:
            return self.agents.get(agent_id)

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """
        Get current status of a task.
        
        Args:
            task_id: ID of the task to query
            
        Returns:
            Optional[Dict]: Task status dict or None if not found
        """
        with self.task_lock:
            for task in self.task_queue:
                if task['id'] == task_id:
                    return task
                    
            # Check completed tasks log
            try:
                with open(self.task_log, 'r') as f:
                    for line in reversed(f.readlines()):
                        if task_id in line:
                            return json.loads(line.split(' - ')[1])
            except Exception:
                pass
                
            return None

    def _monitor_agents(self):
        """Background thread to monitor agent health and status."""
        while self._running:
            try:
                now = datetime.now()
                inactive_agents = []
                
                with self.agent_lock:
                    for agent_id, agent_data in self.agents.items():
                        last_active = datetime.fromisoformat(agent_data['last_heartbeat'])
                        inactive_for = (now - last_active).total_seconds()
                        
                        if inactive_for > self.heartbeat_interval * 3:
                            LOGGER.warning(f"Agent {agent_id} inactive for {inactive_for:.1f}s")
                            inactive_agents.append(agent_id)
                            
                        # Update uptime stats
                        agent_data['stats']['uptime'] = inactive_for
                
                # Clean up inactive agents
                for agent_id in inactive_agents:
                    self.unregister_agent(agent_id)
                
                # Periodic cleanup
                if (now - self.last_cleanup) > timedelta(hours=1):
                    self._cleanup_completed_tasks()
                    self.last_cleanup = now
                
            except Exception as e:
                LOGGER.error(f"Agent monitoring error: {e}")
                
            time.sleep(self.heartbeat_interval)

    def _process_tasks(self):
        """Background thread to process and assign tasks."""
        while self._running:
            try:
                with self.task_lock:
                    # Sort tasks by priority (highest first) and age (oldest first)
                    self.task_queue.sort(key=lambda x: (-x['priority'], x['created']))
                    
                    for task in self.task_queue:
                        if task['status'] != 'queued':
                            continue
                            
                        # Find available agent with matching capabilities
                        with self.agent_lock:
                            for agent_id, agent_data in self.agents.items():
                                if (agent_data['status'] == 'idle' and 
                                    task['type'] in agent_data['capabilities']):
                                    
                                    # Assign task
                                    task['status'] = 'assigned'
                                    task['assigned_to'] = agent_id
                                    task['assigned_at'] = datetime.now().isoformat()
                                    
                                    agent_data['status'] = 'busy'
                                    agent_data['current_task'] = task['id']
                                    
                                    self._log_task(f"Task {task['id']} assigned to {agent_id}")
                                    self._save_persistent_state()
                                    break
                
            except Exception as e:
                LOGGER.error(f"Task processing error: {e}")
                
            time.sleep(1)

    def complete_task(self, agent_id: str, task_id: str, result: Dict, success: bool = True) -> bool:
        """
        Mark a task as completed by an agent.
        
        Args:
            agent_id: ID of the completing agent
            task_id: ID of the completed task
            result: Task result data
            success: Whether the task completed successfully
            
        Returns:
            bool: True if completion was recorded
        """
        with self.agent_lock:
            if agent_id not in self.agents:
                LOGGER.error(f"Agent {agent_id} not found for task completion")
                return False
                
            if self.agents[agent_id]['current_task'] != task_id:
                LOGGER.error(f"Agent {agent_id} trying to complete wrong task")
                return False
                
            # Update agent status
            self.agents[agent_id]['status'] = 'idle'
            self.agents[agent_id]['current_task'] = None
            self.agents[agent_id]['stats']['tasks_completed'] += 1
            if not success:
                self.agents[agent_id]['stats']['errors'] += 1
                
        with self.task_lock:
            # Find and update task
            for task in self.task_queue:
                if task['id'] == task_id:
                    task['status'] = 'completed' if success else 'failed'
                    task['completed_at'] = datetime.now().isoformat()
                    task['result'] = result
                    
                    # Log completion
                    self._log_task(f"Task {task_id} completed by {agent_id}", task)
                    
                    # Remove from queue (will be archived in cleanup)
                    self.task_queue.remove(task)
                    self._save_persistent_state()
                    return True
                    
            LOGGER.error(f"Task {task_id} not found in queue")
            return False

    def _log_task(self, message: str, task_data: Optional[Dict] = None):
        """Log task activity to persistent log."""
        try:
            with open(self.task_log, 'a') as f:
                entry = {
                    'timestamp': datetime.now().isoformat(),
                    'message': message
                }
                if task_data:
                    entry.update(task_data)
                f.write(f"{datetime.now().isoformat()} - {json.dumps(entry)}\n")
        except Exception as e:
            LOGGER.error(f"Failed to log task: {e}")

    def _cleanup_completed_tasks(self):
        """Clean up old completed tasks from memory."""
        cutoff = datetime.now() - timedelta(days=1)
        
        with self.task_lock:
            # In a real implementation, we would archive old tasks here
            LOGGER.debug(f"Task queue cleanup - current size: {len(self.task_queue)}")

    def shutdown(self):
        """Gracefully shutdown the orchestrator."""
        self._running = False
        self._save_persistent_state()
        LOGGER.info("Agent Orchestrator shutting down")

    def get_system_status(self) -> Dict:
        """
        Get overall system status summary.
        
        Returns:
            Dict: System status information
        """
        with self.agent_lock:
            agent_counts = {
                'total': len(self.agents),
                'idle': sum(1 for a in self.agents.values() if a['status'] == 'idle'),
                'busy': sum(1 for a in self.agents.values() if a['status'] == 'busy')
            }
            
            capability_counts = {}
            for agent in self.agents.values():
                for cap in agent['capabilities']:
                    capability_counts[cap] = capability_counts.get(cap, 0) + 1
        
        with self.task_lock:
            task_counts = {
                'queued': len([t for t in self.task_queue if t['status'] == 'queued']),
                'assigned': len([t for t in self.task_queue if t['status'] == 'assigned']),
                'recent_completed': 0  # Would be calculated from task log in full implementation
            }
        
        return {
            'agents': agent_counts,
            'capabilities': capability_counts,
            'tasks': task_counts,
            'last_updated': datetime.now().isoformat()
                            }
