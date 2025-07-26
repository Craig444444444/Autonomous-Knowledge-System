import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import heapq

LOGGER = logging.getLogger("aks")

class ScheduledTask:
    """Represents a scheduled task with execution time and callback."""
    def __init__(self, 
                 name: str,
                 callback: Callable,
                 interval: Optional[timedelta] = None,
                 start_time: Optional[datetime] = None,
                 args: Optional[tuple] = None,
                 kwargs: Optional[Dict[str, Any]] = None):
        self.name = name
        self.callback = callback
        self.interval = interval
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.next_run = start_time or datetime.now()
        self.is_running = False
        self.last_run = None
        self.last_result = None
        self.failure_count = 0

    def __lt__(self, other):
        return self.next_run < other.next_run

    def should_run(self) -> bool:
        """Check if task should run now."""
        return datetime.now() >= self.next_run and not self.is_running

    def reschedule(self):
        """Calculate next run time if recurring."""
        if self.interval:
            self.next_run = datetime.now() + self.interval
        else:
            self.next_run = None

class TaskScheduler:
    """Advanced task scheduler with thread pooling and priority queue."""
    def __init__(self, max_workers: int = 4):
        self.tasks: List[ScheduledTask] = []
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._shutdown = False
        self._scheduler_thread = None
        self.task_history: Dict[str, List[Dict]] = {}
        LOGGER.info("Task scheduler initialized")

    def add_task(self, 
                name: str,
                callback: Callable,
                interval: Optional[timedelta] = None,
                start_time: Optional[datetime] = None,
                args: Optional[tuple] = None,
                kwargs: Optional[Dict[str, Any]] = None) -> bool:
        """Add a new scheduled task."""
        with self.lock:
            if any(task.name == name for task in self.tasks):
                LOGGER.warning(f"Task with name '{name}' already exists")
                return False

            task = ScheduledTask(
                name=name,
                callback=callback,
                interval=interval,
                start_time=start_time,
                args=args,
                kwargs=kwargs
            )
            heapq.heappush(self.tasks, task)
            self.task_history[name] = []
            LOGGER.info(f"Added task '{name}' scheduled for {task.next_run}")
            return True

    def remove_task(self, name: str) -> bool:
        """Remove a scheduled task by name."""
        with self.lock:
            for i, task in enumerate(self.tasks):
                if task.name == name:
                    self.tasks.pop(i)
                    heapq.heapify(self.tasks)
                    LOGGER.info(f"Removed task '{name}'")
                    return True
            LOGGER.warning(f"Task '{name}' not found")
            return False

    def run_task(self, task: ScheduledTask):
        """Execute a task and handle rescheduling."""
        try:
            task.is_running = True
            LOGGER.debug(f"Executing task '{task.name}'")
            
            # Record start time
            start_time = time.time()
            
            # Execute the task
            result = task.callback(*task.args, **task.kwargs)
            
            # Record successful execution
            duration = time.time() - start_time
            task.last_result = result
            task.failure_count = 0
            
            self._record_execution(
                task.name,
                success=True,
                duration=duration,
                result=str(result)[:200]  # Truncate long results
            )
            
        except Exception as e:
            LOGGER.error(f"Task '{task.name}' failed: {str(e)}", exc_info=True)
            task.failure_count += 1
            
            self._record_execution(
                task.name,
                success=False,
                error=str(e),
                failure_count=task.failure_count
            )
            
            # Exponential backoff for failing tasks
            if task.interval and task.failure_count < 5:
                backoff = min(2 ** task.failure_count * 60, 3600)  # Max 1 hour
                task.next_run = datetime.now() + timedelta(seconds=backoff)
        finally:
            task.is_running = False
            task.last_run = datetime.now()
            
            # Reschedule if recurring
            if task.interval and not self._shutdown:
                task.reschedule()
                if task.next_run:
                    with self.lock:
                        heapq.heappush(self.tasks, task)

    def _record_execution(self, 
                         task_name: str,
                         success: bool,
                         duration: Optional[float] = None,
                         result: Optional[str] = None,
                         error: Optional[str] = None,
                         failure_count: Optional[int] = None):
        """Record task execution details in history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "duration": duration,
            "result": result,
            "error": error,
            "failure_count": failure_count
        }
        
        with self.lock:
            if task_name in self.task_history:
                self.task_history[task_name].append(entry)
                # Keep only last 100 executions
                if len(self.task_history[task_name]) > 100:
                    self.task_history[task_name].pop(0)

    def _run_scheduler(self):
        """Main scheduler loop that processes tasks."""
        LOGGER.info("Task scheduler started")
        while not self._shutdown:
            try:
                with self.lock:
                    if not self.tasks:
                        time.sleep(1)
                        continue
                    
                    task = heapq.heappop(self.tasks)
                    if task.should_run():
                        self.executor.submit(self.run_task, task)
                    else:
                        heapq.heappush(self.tasks, task)
                        sleep_time = (task.next_run - datetime.now()).total_seconds()
                        time.sleep(max(0, min(sleep_time, 1)))
            except Exception as e:
                LOGGER.error(f"Scheduler error: {str(e)}", exc_info=True)
                time.sleep(1)

        LOGGER.info("Task scheduler stopped")

    def start(self):
        """Start the scheduler in a background thread."""
        if self._scheduler_thread is None:
            self._shutdown = False
            self._scheduler_thread = threading.Thread(
                target=self._run_scheduler,
                daemon=True,
                name="AKS-TaskScheduler"
            )
            self._scheduler_thread.start()
            LOGGER.info("Started task scheduler thread")

    def shutdown(self, wait: bool = True):
        """Shutdown the scheduler gracefully."""
        self._shutdown = True
        if wait and self._scheduler_thread:
            self._scheduler_thread.join(timeout=30)
        self.executor.shutdown(wait=wait)
        LOGGER.info("Task scheduler shutdown complete")

    def get_task_status(self, name: str) -> Optional[Dict]:
        """Get status of a specific task."""
        with self.lock:
            for task in self.tasks:
                if task.name == name:
                    return {
                        "name": task.name,
                        "next_run": task.next_run.isoformat() if task.next_run else None,
                        "last_run": task.last_run.isoformat() if task.last_run else None,
                        "is_running": task.is_running,
                        "failure_count": task.failure_count,
                        "interval": str(task.interval) if task.interval else None
                    }
        return None

    def list_tasks(self) -> List[Dict]:
        """List all scheduled tasks with their status."""
        with self.lock:
            return [{
                "name": task.name,
                "next_run": task.next_run.isoformat() if task.next_run else None,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "is_running": task.is_running,
                "failure_count": task.failure_count,
                "interval": str(task.interval) if task.interval else None
            } for task in self.tasks]

    def get_task_history(self, name: str, limit: int = 10) -> Optional[List[Dict]]:
        """Get execution history for a specific task."""
        with self.lock:
            if name in self.task_history:
                return self.task_history[name][-limit:]
        return None

    def clear_completed(self) -> int:
        """Remove all completed one-time tasks."""
        with self.lock:
            original_count = len(self.tasks)
            self.tasks = [task for task in self.tasks if task.interval is not None]
            heapq.heapify(self.tasks)
            removed = original_count - len(self.tasks)
            if removed > 0:
                LOGGER.info(f"Removed {removed} completed one-time tasks")
            return removed
