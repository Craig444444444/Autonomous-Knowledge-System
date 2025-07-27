import os
import sys
import time
import threading
import json
import pathlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import traceback

# Set up logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)

# Import all components with error handling
try:
    from ai_provider_manager import AIProviderManager
    from file_handler import FileHandler
    from git_manager import GitManager
    from knowledge_processor import KnowledgeProcessor
    from vector_db import VectorDB
    from nli import NaturalLanguageInterface
    from ai_generator import AIGenerator
    from resilience_manager import ResilienceManager
    from security_manager import SecurityManager
    from audit_manager import AuditManager
    from monitoring import Monitoring
    from task_scheduler import TaskScheduler
    from collaborative_processor import CollaborativeProcessor
    from information_sourcing import InformationSourcing
    from api_handler import APIHandler
    from plugin_manager import PluginManager
    from codebase_enhancer import CodebaseEnhancer
    from documentation_generator import DocumentationGenerator
    from testing_framework import TestingFramework
    from user_manager import UserManager
    from data_visualizer import DataVisualizer
    from version_migrator import VersionMigrator
    from agent_orchestrator import AgentOrchestrator
except ImportError as e:
    LOGGER.error(f"Critical import error: {str(e)}")
    LOGGER.error(traceback.format_exc())
    sys.exit(1)

class Config:
    """Central configuration class for the Autonomous Knowledge System"""
    def __init__(self):
        # Core system paths
        self.repo_path = pathlib.Path(os.getcwd())
        self.knowledge_base_dir = self.repo_path / "knowledge_base"
        self.vector_db_dir = self.repo_path / "vector_db"
        self.snapshot_dir = self.repo_path / "snapshots"
        self.user_feedback_dir = self.repo_path / "user_feedback"
        self.temp_dir = self.repo_path / "temp"
        
        # GitHub configuration
        self.repo_owner = "Craig444444444"
        self.repo_name = "Autonomous-Knowledge-System"
        self.repo_url = f"https://github.com/{self.repo_owner}/{self.repo_name}.git"
        self.github_token = os.environ.get("GITHUB_TOKEN", "")
        
        # AI providers configuration
        self.preferred_models = {
            "text_generation": "gemini-pro",
            "code_generation": "gemini-pro",
            "information_retrieval": "gemini-pro",
            "summarization": "gemini-pro"
        }
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "")
        
        # System parameters
        self.max_snapshots = 10
        self.max_concurrent_tasks = 5
        self.max_plugins = 10
        self.autonomous_cycle_interval = 1800  # 30 minutes
        
        # Vector database configuration
        self.vector_db_config = {
            "model": "all-MiniLM-L6-v2",
            "chunk_size": 512,
            "chunk_overlap": 50
        }
        
        # Create directories if missing
        self._create_directories()
    
    def _create_directories(self):
        """Ensure all required directories exist"""
        for directory in [
            self.knowledge_base_dir, 
            self.vector_db_dir,
            self.snapshot_dir,
            self.user_feedback_dir,
            self.temp_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Directory ensured: {directory}")

# Global configuration instance
config = Config()

class AutonomousAgent:
    """Core autonomous agent managing the system loop with all integrated components."""
    def __init__(self, config):  # Added config parameter
        self.active = True
        self.system_activities: List[str] = []
        self._system_activities_lock = threading.Lock()
        self.config = config  # Store config for component initialization
        
        # Initialize all core components
        self.ai_provider_manager = AIProviderManager(config.preferred_models)
        self.file_handler = FileHandler(config.repo_path)
        
        # Initialize GitManager with error handling
        self.git_manager = None
        try:
            self.git_manager = GitManager(
                config.repo_path, 
                config.github_token, 
                config.repo_owner, 
                config.repo_name, 
                config.repo_url
            )
            LOGGER.info("GitManager initialized successfully")
        except Exception as e:
            LOGGER.error(f"Critical GitManager initialization failed: {e}")
            # Fallback to minimal functionality
            LOGGER.warning("Operating without Git functionality - limited capabilities")
        
        # Knowledge and processing components
        self.knowledge_processor = KnowledgeProcessor(config.knowledge_base_dir)
        self.vector_db = VectorDB(config.vector_db_dir, config.vector_db_config)
        self.nli = NaturalLanguageInterface(self.ai_provider_manager)
        
        # AI generation components
        self.ai_generator = AIGenerator(
            self.ai_provider_manager, 
            config.repo_path, 
            self.file_handler,
            self.vector_db
        )
        
        # System management components
        self.resilience_manager = ResilienceManager(config.repo_path, config.snapshot_dir, config.max_snapshots)
        self.security_manager = SecurityManager()
        self.audit_manager = AuditManager(config.repo_path)
        self.monitoring = Monitoring(config)  # Fixed: Pass config to Monitoring
        self.task_scheduler = TaskScheduler(max_tasks=config.max_concurrent_tasks)
        
        # Collaboration and integration components
        self.collaborative_processor = CollaborativeProcessor(
            self.knowledge_processor, 
            config.user_feedback_dir, 
            config.temp_dir
        )
        self.information_sourcing = InformationSourcing(
            self.ai_provider_manager, 
            self.knowledge_processor, 
            config,
            self.vector_db
        )
        self.api_handler = APIHandler()
        self.plugin_manager = PluginManager(max_plugins=config.max_plugins)
        
        # Code and documentation components
        self.codebase_enhancer = CodebaseEnhancer(self.ai_generator)
        self.documentation_generator = DocumentationGenerator(self.ai_generator)
        self.testing_framework = TestingFramework()
        
        # User and data components
        self.user_manager = UserManager()
        self.data_visualizer = DataVisualizer()
        self.version_migrator = VersionMigrator()
        
        # Orchestration component
        self.agent_orchestrator = AgentOrchestrator(self)

        # System state
        self.last_push_time = 0
        self.cycle_count = 0
        self.start_time = time.time()
        
        # Initialize all plugins
        self._initialize_plugins()
        LOGGER.info("AutonomousAgent initialized successfully")

    def _initialize_plugins(self):
        """Load and initialize all available plugins"""
        try:
            plugins_dir = self.config.repo_path / "plugins"
            if plugins_dir.exists():
                self.plugin_manager.load_plugins(plugins_dir)
                LOGGER.info(f"Loaded {len(self.plugin_manager.plugins)} plugins")
            else:
                LOGGER.warning("Plugins directory not found")
        except Exception as e:
            LOGGER.error(f"Plugin initialization failed: {e}")

    def log_activity(self, activity: str):
        """Thread-safe logging of system activities"""
        with self._system_activities_lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.system_activities.append(f"[{timestamp}] {activity}")
            # Keep only the last 100 activities
            if len(self.system_activities) > 100:
                self.system_activities.pop(0)

    def autonomous_cycle(self):
        """Execute one full cycle of autonomous operation"""
        self.cycle_count += 1
        self.log_activity(f"Starting autonomous cycle #{self.cycle_count}")
        LOGGER.info(f"=== AUTONOMOUS CYCLE #{self.cycle_count} STARTED ===")
        
        try:
            # Create system snapshot before any operations
            self.resilience_manager.create_snapshot()
            
            # 1. Knowledge processing phase
            self.log_activity("Processing new knowledge")
            new_knowledge = self.knowledge_processor.process_new_knowledge()
            LOGGER.info(f"Processed {len(new_knowledge)} new knowledge items")
            
            # 2. Information sourcing and research
            self.log_activity("Sourcing new information")
            research_topics = self.knowledge_processor.identify_research_topics()
            research_results = self.information_sourcing.research_topics(research_topics)
            LOGGER.info(f"Researched {len(research_results)} topics")
            
            # 3. Collaborative processing
            self.log_activity("Processing user feedback")
            feedback_items = self.collaborative_processor.process_feedback()
            LOGGER.info(f"Processed {len(feedback_items)} user feedback items")
            
            # 4. Codebase enhancement
            self.log_activity("Enhancing codebase")
            enhancement_targets = self.codebase_enhancer.identify_enhancement_targets()
            enhancement_results = self.codebase_enhancer.enhance_codebase(enhancement_targets)
            LOGGER.info(f"Enhanced {len(enhancement_results)} files")
            
            # 5. Documentation generation
            self.log_activity("Generating documentation")
            doc_targets = self.documentation_generator.identify_documentation_targets()
            doc_results = self.documentation_generator.generate_documentation(doc_targets)
            LOGGER.info(f"Generated documentation for {len(doc_results)} components")
            
            # 6. Security audit
            self.log_activity("Performing security audit")
            security_report = self.security_manager.audit_system(self.config.repo_path)
            LOGGER.info(f"Security audit completed: {security_report['summary']}")
            
            # 7. Git operations (if GitManager is available)
            if self.git_manager:
                self.log_activity("Committing changes")
                commit_msg = f"Autonomous update cycle #{self.cycle_count}"
                self.git_manager.commit_and_push(commit_msg)
                self.last_push_time = time.time()
                LOGGER.info("Changes committed and pushed to repository")
            
            # 8. System monitoring and reporting
            self.log_activity("Generating system report")
            report = self.monitoring.generate_system_report()
            LOGGER.info(f"System report: {report['status']}")
            
            # 9. Plugin execution
            self.log_activity("Executing plugins")
            plugin_results = self.plugin_manager.execute_plugins()
            LOGGER.info(f"Executed {len(plugin_results)} plugins")
            
            self.log_activity(f"Cycle #{self.cycle_count} completed successfully")
            LOGGER.info(f"=== AUTONOMOUS CYCLE #{self.cycle_count} COMPLETED ===")
            
            return True
        except Exception as e:
            LOGGER.error(f"Cycle #{self.cycle_count} failed: {str(e)}")
            LOGGER.error(traceback.format_exc())
            self.log_activity(f"Cycle #{self.cycle_count} failed: {str(e)}")
            
            # Attempt recovery
            self.log_activity("Attempting system recovery")
            try:
                self.resilience_manager.restore_latest_snapshot()
                LOGGER.warning("System restored from latest snapshot")
            except Exception as recovery_error:
                LOGGER.critical(f"Recovery failed: {str(recovery_error)}")
                self.log_activity("CRITICAL: Recovery failed")
            
            return False

    def run(self, continuous: bool = True, max_cycles: int = 0):
        """Main system execution loop"""
        self.log_activity("System startup")
        LOGGER.info("Starting Autonomous Knowledge System")
        
        try:
            # Start monitoring subsystem
            self.monitoring.start_monitoring()
            
            cycle_count = 0
            while self.active:
                # Run autonomous cycle
                success = self.autonomous_cycle()
                cycle_count += 1
                
                # Break conditions
                if not continuous:
                    break
                if max_cycles > 0 and cycle_count >= max_cycles:
                    LOGGER.info(f"Reached maximum cycle count ({max_cycles})")
                    break
                
                # Wait for next cycle
                self.log_activity(f"Sleeping for {self.config.autonomous_cycle_interval} seconds")
                time.sleep(self.config.autonomous_cycle_interval)
        except KeyboardInterrupt:
            LOGGER.info("Shutdown signal received")
        except Exception as e:
            LOGGER.critical(f"Fatal error in main loop: {str(e)}")
            LOGGER.critical(traceback.format_exc())
        finally:
            # System shutdown procedures
            self.log_activity("System shutdown")
            self.monitoring.stop_monitoring()
            LOGGER.info("Stopping monitoring subsystem")
            
            # Final snapshot
            try:
                self.resilience_manager.create_snapshot()
                LOGGER.info("Created final system snapshot")
            except Exception as e:
                LOGGER.error(f"Final snapshot failed: {str(e)}")
            
            LOGGER.info("Autonomous Knowledge System stopped")

def run_aks_pipeline(user_zip_file, repo_url_input, analysis_claim, debate_topic, 
                     github_token_input, gemini_api_key_input, web_query, 
                     ftp_query, run_autonomous_cycle):
    """Main pipeline function for Gradio interface"""
    try:
        # Update configuration based on inputs
        if github_token_input:
            config.github_token = github_token_input
            os.environ["GITHUB_TOKEN"] = github_token_input
        
        if gemini_api_key_input:
            config.gemini_key = gemini_api_key_input
            os.environ["GEMINI_API_KEY"] = gemini_api_key_input
        
        if repo_url_input:
            config.repo_url = repo_url_input
        
        # 1. Initialize the Agent - pass config here
        agent = AutonomousAgent(config)  # Pass config to the constructor
        
        # 2. Run autonomous cycle if requested
        if run_autonomous_cycle:
            agent.autonomous_cycle()
            return "Autonomous cycle completed successfully!"
        else:
            return "System initialized without running autonomous cycle"
    except Exception as e:
        return f"Pipeline execution failed: {str(e)}"

# Console mode execution
if __name__ == "__main__":
    try:
        import gradio as gr
        has_gradio = True
    except ImportError:
        has_gradio = False
    
    if has_gradio:
        # Gradio interface setup
        with gr.Blocks(title="Autonomous Knowledge System") as demo:
            gr.Markdown("# Autonomous Knowledge System")
            # ... Gradio UI components ...
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    else:
        LOGGER.warning("Gradio not installed. Running in console-only mode.")
        LOGGER.info("--- Autonomous Knowledge System (AKS) - Console Mode ---")
        
        # Get GitHub token securely
        github_token = config.github_token if config.github_token else getpass("Enter your GitHub Personal Access Token (PAT): ")
        config.github_token = github_token

        # Get Gemini API Key securely
        gemini_api_key = config.gemini_key if config.gemini_key else getpass("Enter your Gemini API Key (Optional): ")
        config.gemini_key = gemini_api_key

        # Basic input for repo details
        config.repo_owner = input("Enter GitHub repo owner (or press Enter for default): ") or config.repo_owner
        config.repo_name = input("Enter GitHub repo name (or press Enter for default): ") or config.repo_name
        
        # Run the agent - pass config here
        agent = AutonomousAgent(config)  # Pass config to the constructor
        agent.run(continuous=True)
        LOGGER.info("Autonomous Knowledge System (AKS) finished.")
