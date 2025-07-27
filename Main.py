# Autonomous knowledge system
#
# Copyright (c) 2025 Craig Huckerby. All rights reserved

import os
import re
import json
import time
import random
import zipfile
import shutil
import threading
import subprocess
import logging
import sys
import hashlib
import ast  # For code syntax validation
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from getpass import getpass
from logging.handlers import RotatingFileHandler
from collections import Counter
import importlib.util  # For dynamic module loading

# --- Constants ---
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
MAX_LOG_BACKUPS = 5
DEFAULT_CYCLE_INTERVAL = 15 * 60  # 15 minutes in seconds
MAX_API_RETRIES = 5
MIN_DISK_SPACE = 1 * 1024 * 1024 * 1024  # 1GB

# Package import name mapping for installation checks
PACKAGE_MAPPING = {
    "pyfiglet": "pyfiglet",
    "google-generativeai": "google.generativeai",
    "transformers": "transformers",
    "torch": "torch",
    "python-dateutil": "dateutil",
    "requests": "requests",
    "beautifulsoup4": "bs4",
    "tqdm": "tqdm",
    "gitpython": "git",
    "gradio": "gradio"  # For the Gradio UI
}

# --- Install required packages ---
def install_required_packages():
    required_packages = {
        "pyfiglet": ">=0.8.post1",
        "google-generativeai": ">=0.5.4",  # For Gemini API
        "transformers": ">=4.40.0",       # For GPT-2 fallback
        "torch": ">=2.2.0",              # For GPT-2 fallback
        "python-dateutil": ">=2.8.2",
        "requests": ">=2.31.0",
        "beautifulsoup4": ">=4.12.0",
        "tqdm": ">=4.66.0",
        "gitpython": ">=3.1.43",
        "gradio": ">=4.0.0"  # For the Gradio UI
    }
    print("Checking and installing required packages...")
    for package, version in required_packages.items():
        try:
            # Attempt to import to check if already installed
            import_name = PACKAGE_MAPPING.get(package, package.split("==")[0].replace("-", "_"))
            __import__(import_name)
            print(f"DEBUG: Package already installed: {package}")
        except ImportError:
            print(f"INFO: Installing {package}{version}...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", f"{package}{version}"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"INFO: Successfully installed {package}{version}.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Failed to install {package}: {e.stderr.strip()}", file=sys.stderr)
            except Exception as e:
                print(f"ERROR: Unexpected error during installation of {package}: {e}", file=sys.stderr)

install_required_packages()

# --- Conditional Imports (after installation attempt) ---
try:
    import pyfiglet
except ImportError:
    pyfiglet = None

# Imports for AI Providers
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    import torch
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    set_seed = None
    torch = None

try:
    from dateutil.parser import parse as date_parse
except ImportError:
    date_parse = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import gradio as gr  # For UI
except ImportError:
    gr = None

# IMPORT ALL MODULAR COMPONENTS
try:
    from ai_generator import AIGenerator
    from knowledge_processor import KnowledgeProcessor
    from resilience_manager import ResilienceManager
    from natural_language_interface import NaturalLanguageInterface
    from collaborative_processor import CollaborativeProcessor
    from git_manager import GitManager
    from information_sourcing import InformationSourcing
    from codebase_enhancer import CodebaseEnhancer
    from security import SecurityManager
    from audit import AuditManager
    from file_handler import FileHandler
    from api_handler import APIHandler
    from task_scheduler import TaskScheduler
    from monitoring import Monitoring
    from plugin_manager import PluginManager
    from user_manager import UserManager
    from data_visualizer import DataVisualizer
    from version_migrator import VersionMigrator
    from documentation_generator import DocumentationGenerator
    # FIXED IMPORT: Use alias to resolve TestingFramework
    from testing_framework import AKSTestRunner as TestingFramework
    from agent_orchestrator import AgentOrchestrator
    from vector_db import VectorDB
except ImportError as e:
    print(f"\n\nCRITICAL ERROR: Could not import a core modular component: {e}", file=sys.stderr)
    print("Please ensure all required .py files are in the same directory as aks_main.py", file=sys.stderr)
    sys.exit(1)

# CONFIGURATION SYSTEM
class Config:
    """Centralized configuration management system."""
    def __init__(self):
        self._config_version: str = "1.3"  # Updated version
        self._repo_owner: Optional[str] = os.getenv("GITHUB_REPO_OWNER") or "Craig444444444"
        self._repo_name: str = os.getenv("GITHUB_REPO_NAME") or "AKS"
        self._repo_url: Optional[str] = f"https://github.com/{self._repo_owner}/{self._repo_name}.git"
        self._repo_path: Path = Path("/content") / self._repo_name
        self._monitor_dir: Path = Path("/content")
        self._user_feedback_dir: Path = Path("/content/user_feedback")
        self._archive_dir: Path = Path("/content/archive")
        self._temp_dir: Path = Path("/content/temp")
        self._knowledge_base_dir: Path = Path("/content/knowledge_base")
        self._snapshot_dir: Path = Path("/content/snapshots")
        self._quarantine_dir: Path = Path("/content/quarantine")
        self._user_uploads_dir: Path = Path("/content/aks_user_uploads")
        self._vector_db_dir: Path = Path("/content/vector_db")  # New directory for vector database
        
        # System configuration
        self._log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
        self._github_token: Optional[str] = os.getenv("GITHUB_TOKEN")
        self._gemini_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self._cycle_interval: int = int(os.getenv("CYCLE_INTERVAL", DEFAULT_CYCLE_INTERVAL))
        self._max_snapshots: int = int(os.getenv("MAX_SNAPSHOTS", 5))
        self._max_branches: int = int(os.getenv("MAX_BRANCHES", 10))
        self._push_interval: int = int(os.getenv("PUSH_INTERVAL", 60))
        self._ai_activity_chance: float = float(os.getenv("AI_ACTIVITY_CHANCE", 0.7))
        self._api_max_retries: int = int(os.getenv("API_MAX_RETRIES", MAX_API_RETRIES))
        self._max_plugins: int = int(os.getenv("MAX_PLUGINS", 10))  # New config for plugins
        self._max_concurrent_tasks: int = int(os.getenv("MAX_CONCURRENT_TASKS", 5))  # New config for task scheduler

        # Preferred models list
        if not self._gemini_key:
            self._preferred_models: List[str] = ["gpt2"]
        else:
            self._preferred_models: List[str] = [
                "gemini-1.5-pro-latest",
                "gpt2"
            ]

        # Enhanced scraper configuration
        self._scraper_config = {
            'user_agent': 'AKSBot/1.0',
            'max_retries': 3,
            'retry_delay': 5,
            'extraction_tags': ['p', 'h1', 'h2', 'h3', 'article'],
            'max_links': 15,
            'timeout': 30  # Added timeout parameter
        }
        
        # New vector database configuration
        self._vector_db_config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'max_connections': 10,
            'persist_interval': 300  # seconds
        }
        
        self._setup_directories()

    def _setup_directories(self):
        """Create required directories."""
        dirs = [
            self._repo_path, self._snapshot_dir, Path("/content/logs"),
            self._user_feedback_dir, self._archive_dir, self._temp_dir,
            self._knowledge_base_dir, self._quarantine_dir,
            self._user_uploads_dir, self._vector_db_dir  # Added vector db directory
        ]
        for dir_path in dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                dir_path.chmod(0o755)
            except OSError as e:
                sys.stderr.write(f"Error creating directory {dir_path}: {e}\n")

    # Properties for all configuration values
    @property
    def config_version(self) -> str: return self._config_version
    @property
    def repo_owner(self) -> str: return self._repo_owner
    @repo_owner.setter
    def repo_owner(self, value: str):
        if not value:
            return
        self._repo_owner = value
        self._repo_url = f"https://github.com/{self._repo_owner}/{self._repo_name}.git"
    @property
    def repo_name(self) -> str: return self._repo_name
    @repo_name.setter
    def repo_name(self, value: str):
        self._repo_name = value
        self._repo_url = f"https://github.com/{self._repo_owner}/{self._repo_name}.git"
        self._repo_path = Path("/content") / self._repo_name
    @property
    def repo_url(self) -> str: return self._repo_url
    @property
    def repo_path(self) -> Path: return self._repo_path
    @property
    def monitor_dir(self) -> Path: return self._monitor_dir
    @property
    def user_feedback_dir(self) -> Path: return self._user_feedback_dir
    @property
    def archive_dir(self) -> Path: return self._archive_dir
    @property
    def temp_dir(self) -> Path: return self._temp_dir
    @property
    def knowledge_base_dir(self) -> Path: return self._knowledge_base_dir
    @property
    def snapshot_dir(self) -> Path: return self._snapshot_dir
    @property
    def quarantine_dir(self) -> Path: return self._quarantine_dir
    @property
    def user_uploads_dir(self) -> Path: return self._user_uploads_dir
    @property
    def vector_db_dir(self) -> Path: return self._vector_db_dir  # New property
    @property
    def log_level(self) -> str: return self._log_level
    @property
    def github_token(self) -> Optional[str]: return self._github_token
    @github_token.setter
    def github_token(self, value: str): self._github_token = value
    @property
    def gemini_key(self) -> Optional[str]: return self._gemini_key
    @gemini_key.setter
    def gemini_key(self, value: str): self._gemini_key = value
    @property
    def cycle_interval(self) -> int: return self._cycle_interval
    @property
    def max_snapshots(self) -> int: return self._max_snapshots
    @property
    def max_branches(self) -> int: return self._max_branches
    @property
    def push_interval(self) -> int: return self._push_interval
    @property
    def ai_activity_chance(self) -> float: return self._ai_activity_chance
    @property
    def api_max_retries(self) -> int: return self._api_max_retries
    @property
    def max_plugins(self) -> int: return self._max_plugins  # New property
    @property
    def max_concurrent_tasks(self) -> int: return self._max_concurrent_tasks  # New property
    @property
    def preferred_models(self) -> List[str]: return self._preferred_models
    @property
    def scraper_config(self) -> Dict[str, Any]: return self._scraper_config
    @property
    def vector_db_config(self) -> Dict[str, Any]: return self._vector_db_config  # New property

    def validate(self) -> List[str]:
        """Validate configuration and return errors."""
        errors = []
        # Validate critical configuration
        required_config_keys = ['repo_path', 'repo_owner', 'repo_name']
        for key in required_config_keys:
            if key not in self.__dict__ or not self.__dict__[f"_{key}"]:
                errors.append(f"Missing required configuration: {key}")
                
        if not self._github_token:
            errors.append("GitHub token is required for remote repository operations.")
        if "gpt2" not in self._preferred_models or len(self._preferred_models) > 1:
            if not self._gemini_key:
                errors.append("Gemini API key required unless only using GPT-2 fallback.")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary, redacting sensitive data."""
        return {
            "config_version": self._config_version,
            "repository": {
                "owner": self._repo_owner, "name": self._repo_name, "url": self._repo_url,
                "path": str(self._repo_path), "monitor_dir": str(self._monitor_dir),
                "user_feedback_dir": str(self._user_feedback_dir), "archive_dir": str(self._archive_dir),
                "temp_dir": str(self._temp_dir), "knowledge_base_dir": str(self._knowledge_base_dir),
                "snapshot_dir": str(self._snapshot_dir), "quarantine_dir": str(self._quarantine_dir),
                "user_uploads_dir": str(self._user_uploads_dir),
                "vector_db_dir": str(self._vector_db_dir)  # Added vector db directory
            },
            "api": {
                "github_token": "***REDACTED***" if self._github_token else None,
                "gemini_key": "***REDACTED***" if self._gemini_key else None,
            },
            "system": {
                "log_level": self._log_level, "max_branches": self._max_branches,
                "max_snapshots": self._max_snapshots, "cycle_interval": self._cycle_interval,
                "push_interval": self._push_interval, "ai_activity_chance": self._ai_activity_chance,
                "api_max_retries": self._api_max_retries, "preferred_models": self._preferred_models,
                "max_plugins": self._max_plugins, "max_concurrent_tasks": self._max_concurrent_tasks,
                "scraper_config": self._scraper_config,
                "vector_db_config": self._vector_db_config  # Added vector db config
            }
        }

config = Config()

# LOGGING SYSTEM (unchanged from previous version)
class ColoredFormatter(logging.Formatter):
    """Custom colored formatter for console output."""
    COLORS = {'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m', 
              'ERROR': '\033[31m', 'CRITICAL': '\033[35m', 'RESET': '\033[0m'}
    
    def format(self, record):
        log_message = super().format(record)
        if sys.stderr.isatty():
            return f"{self.COLORS.get(record.levelname, '')}{log_message}{self.COLORS['RESET']}"
        return log_message

class LogManager:
    """Enhanced logging manager with file rotation and colored output."""
    def __init__(self, name: str = "aks"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        self._setup_handlers()
        self._start_time = time.time()
        
    def _setup_handlers(self):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(ColoredFormatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(console)
        log_file = Path("/content/logs/aks.log")
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                filename=log_file, 
                maxBytes=MAX_LOG_SIZE, 
                backupCount=MAX_LOG_BACKUPS, 
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.error(f"Failed to setup file logger: {e}")
            
    def _get_uptime(self) -> float: 
        return time.time() - self._start_time
    
    def debug(self, message: str, *args, **kwargs): 
        self.logger.debug(message, *args, extra={'uptime': self._get_uptime()}, **kwargs)
    def info(self, message: str, *args, **kwargs): 
        self.logger.info(message, *args, extra={'uptime': self._get_uptime()}, **kwargs)
    def warning(self, message: str, *args, **kwargs): 
        self.logger.warning(message, *args, extra={'uptime': self._get_uptime()}, **kwargs)
    def error(self, message: str, *args, **kwargs): 
        self.logger.error(message, *args, extra={'uptime': self._get_uptime()}, **kwargs)
    def critical(self, message: str, *args, **kwargs): 
        self.logger.critical(message, *args, extra={'uptime': self._get_uptime()}, **kwargs)
    def exception(self, message: str, *args, **kwargs): 
        self.logger.exception(message, *args, exc_info=True, **kwargs)

LOGGER = LogManager()

# AI PROVIDER IMPLEMENTATION (unchanged from previous version)
class AIProvider:
    """Base class for AI providers with common functionality."""
    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.last_used = 0
        self.usage_count = 0
        self.error_count = 0
        self.rate_limit = 60
        self.active = True
        self.quota_exceeded = False
        self.model_access_issue = False
        
    def is_available(self) -> bool:
        if not self.active or self.quota_exceeded or self.model_access_issue: 
            return False
        if time.time() - self.last_used < 60 / self.rate_limit: 
            return False
        return True
    
    def record_usage(self, success: bool = True):
        self.last_used = time.time()
        self.usage_count += 1
        if not success:
            self.error_count += 1
            if self.error_count > 5:
                self.active = False
                LOGGER.warning(f"Provider {self.name} disabled due to too many consecutive errors.")
                
    def reset_status(self):
        self.error_count = 0
        self.active = True
        self.quota_exceeded = False
        self.model_access_issue = False

    def generate_text(self, prompt: str, system_prompt: str, max_tokens: int = 2048) -> Optional[str]: 
        raise NotImplementedError
    def generate_code(self, prompt: str, system_prompt: str, max_tokens: int = 4096) -> Optional[str]: 
        raise NotImplementedError

# Retry Mechanism (Decorator) (unchanged from previous version)
def with_retries(func: Callable, max_retries: int = config.api_max_retries, backoff_factor: float = 1.0) -> Any:
    """Decorator to retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    LOGGER.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                    raise
                wait_time = backoff_factor * (2 ** attempt)
                LOGGER.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        return None
    return wrapper

class GeminiProvider(AIProvider):
    """Google Gemini API Provider."""
    def __init__(self, api_key: str):
        super().__init__("Gemini", api_key)
        self.model: Optional[genai.GenerativeModel] = None
        self._initialize()

    def _initialize(self):
        if genai is None:
            LOGGER.warning("Google Generative AI package not installed, GeminiProvider disabled.")
            self.active = False
            return
        if not self.api_key:
            LOGGER.warning("Gemini API key not provided, GeminiProvider disabled.")
            self.active = False
            return
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
            self.model.generate_content("test", generation_config=genai.types.GenerationConfig(max_output_tokens=1))
            self.active = True
            LOGGER.info("Gemini provider initialized successfully.")
        except Exception as e:
            LOGGER.error(f"Gemini client initialization failed: {e}")
            self.active = False
            self.model_access_issue = True

    @with_retries
    def generate_text(self, prompt: str, system_prompt: str, max_tokens: int = 2048) -> Optional[str]:
        if not self.is_available() or not self.model:
            LOGGER.warning("GeminiProvider is not available for text generation.")
            return None
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            generation_config = genai.types.GenerationConfig(
                temperature=0.5,
                max_output_tokens=max_tokens
            )
            response = self.model.generate_content(full_prompt, generation_config=generation_config)

            if response.candidates and response.candidates[0].content.parts:
                self.record_usage(True)
                return response.candidates[0].content.parts[0].text.strip()
            else:
                LOGGER.warning(f"Gemini generation returned no content. Reason: {response.prompt_feedback.block_reason.name if response.prompt_feedback else 'Unknown'}")
                self.record_usage(False)
                return None
        except Exception as e:
            LOGGER.error(f"Unexpected error during Gemini text generation: {e}", exc_info=True)
            self.record_usage(False)
            raise

    @with_retries
    def generate_code(self, prompt: str, system_prompt: str, max_tokens: int = 4096) -> Optional[str]:
        code_system_prompt = f"Write only Python code, no explanations or markdown. {system_prompt}"
        result = self.generate_text(prompt, code_system_prompt, max_tokens)
        if result:
            if "```python" in result:
                code_match = re.search(r"```python\s*(.*?)\s*```", result, re.DOTALL)
                if code_match: 
                    return code_match.group(1).strip()
            elif "```" in result:
                code_match = re.search(r"```\s*(.*?)\s*```", result, re.DOTALL)
                if code_match: 
                    return code_match.group(1).strip()
            return result.strip()
        return None

class FreeAIProvider(AIProvider):
    """Free AI provider using GPT-2 from Hugging Face Transformers."""
    def __init__(self):
        super().__init__("FreeAI_GPT2", api_key="free")
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self._initialize()

    def _initialize(self):
        if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
            LOGGER.warning("Transformers or PyTorch not imported. FreeAIProvider disabled.")
            self.active = False
            return
        try:
            model_name = "gpt2"
            LOGGER.info(f"Attempting to load free AI model: {model_name} on {self.device.upper()}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None: 
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.active = True
            LOGGER.info("Free AI provider (GPT-2) initialized successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to initialize Free AI provider (GPT-2): {e}", exc_info=True)
            self.active = False

    @with_retries
    def generate_text(self, prompt: str, system_prompt: str, max_tokens: int = 500) -> Optional[str]:
        if not self.is_available() or not self.model:
            LOGGER.warning("FreeAIProvider is not available for text generation.")
            return None
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024 - max_tokens).to(self.device)
            if set_seed: 
                set_seed(42)
            outputs = self.model.generate(
                inputs.input_ids, 
                max_new_tokens=max_tokens, 
                temperature=0.7,
                do_sample=True, 
                pad_token_id=self.tokenizer.eos_token_id, 
                num_return_sequences=1
            )
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated[len(full_prompt):].strip() if generated.startswith(full_prompt) else generated
            if not generated.strip():
                LOGGER.warning("FreeAIProvider generated empty content.")
                self.record_usage(False)
                return None
            self.record_usage(True)
            return generated
        except Exception as e:
            LOGGER.error(f"Free AI text generation failed: {e}", exc_info=True)
            self.record_usage(False)
            raise

    @with_retries
    def generate_code(self, prompt: str, system_prompt: str, max_tokens: int = 500) -> Optional[str]:
        if not self.is_available(): 
            return None
        code_system_prompt = f"# Python code only, no explanations.\n{system_prompt}"
        result = self.generate_text(prompt, code_system_prompt, max_tokens)
        if result:
            if "```python" in result:
                match = re.search(r"```python\s*(.*?)\s*```", result, re.DOTALL)
                result = match.group(1).strip() if match else result.split("```python")[1].split("```")[0].strip()
            elif "```" in result:
                match = re.search(r"```\s*(.*?)\s*```", result, re.DOTALL)
                result = match.group(1).strip() if match else result.split("```")[1].split("```")[0].strip()

            if not result.strip():
                self.record_usage(False)
                return None
            try:
                ast.parse(result)
                self.record_usage(True)
                return result
            except SyntaxError:
                self.record_usage(False)
                LOGGER.warning("FreeAIProvider generated invalid Python syntax.")
                return None
        self.record_usage(False)
        return None

class AIProviderManager:
    """Manages AI providers with fallback, prioritizing Gemini."""
    def __init__(self, preferred_models: List[str]):
        self.providers: List[AIProvider] = []
        self.preferred_models = preferred_models
        self.initialize_providers()

    def initialize_providers(self):
        self.providers = []
        if config.gemini_key:
            self.providers.append(GeminiProvider(config.gemini_key))
        self.providers.append(FreeAIProvider())
        self.providers.sort(key=lambda p: (p.api_key is None, p.name))
        LOGGER.info(f"Initialized {len(self.providers)} AI providers.")
        LOGGER.info(f"Provider order: {[p.name for p in self.providers]}")

    def _execute_with_backoff(self, method_name: str, prompt: str, system_prompt: str, max_tokens: int) -> Optional[str]:
        for provider in self.providers:
            if provider.is_available():
                generate_method = getattr(provider, method_name)
                try:
                    result = generate_method(prompt, system_prompt, max_tokens)
                    if result is not None:
                        return result
                except Exception as e:
                    LOGGER.warning(f"Provider {provider.name} failed for {method_name}: {e}. Trying next provider.")
                    continue
        LOGGER.error("All providers failed for generation.")
        return None

    def generate_text(self, prompt: str, system_prompt: str, max_tokens: int = 2048) -> Optional[str]:
        return self._execute_with_backoff("generate_text", prompt, system_prompt, max_tokens)
    def generate_code(self, prompt: str, system_prompt: str, max_tokens: int = 4096) -> Optional[str]:
        return self._execute_with_backoff("generate_code", prompt, system_prompt, max_tokens)
    def has_available_providers(self) -> bool:
        return any(provider.is_available() for provider in self.providers)
    def get_provider_status(self) -> Dict[str, str]:
        status = {}
        for provider in self.providers:
            status_text = "Available"
            if not provider.active: 
                status_text = "Disabled (Too many errors)"
            elif provider.quota_exceeded: 
                status_text = "Unavailable (Quota Exceeded)"
            elif provider.model_access_issue: 
                status_text = "Unavailable (Model Access Issue)"
            elif not provider.is_available(): 
                status_text = "Unavailable (Inactive or Rate Limited)"
            status[provider.name] = status_text
        return status

# AUTONOMOUS AGENT - UPDATED WITH ALL NEW COMPONENTS
class AutonomousAgent:
    """Core autonomous agent managing the system loop with all integrated components."""
    def __init__(self):
        self.active = True
        self.system_activities: List[str] = []
        self._system_activities_lock = threading.Lock()
        
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
        self.vector_db = VectorDB(config.vector_db_dir, config.vector_db_config)  # New vector database
        self.nli = NaturalLanguageInterface(self.ai_provider_manager)
        
        # AI generation components
        self.ai_generator = AIGenerator(
            self.ai_provider_manager, 
            config.repo_path, 
            self.file_handler,
            self.vector_db  # Pass vector db to AI generator
        )
        
        # System management components
        self.resilience_manager = ResilienceManager(config.repo_path, config.snapshot_dir, config.max_snapshots)
        self.security_manager = SecurityManager()
        self.audit_manager = AuditManager(config.repo_path)
        self.monitoring = Monitoring()  # New monitoring system
        self.task_scheduler = TaskScheduler(max_tasks=config.max_concurrent_tasks)  # New task scheduler
        
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
            self.vector_db  # Pass vector db to information sourcing
        )
        self.api_handler = APIHandler()  # New API handler
        self.plugin_manager = PluginManager(max_plugins=config.max_plugins)  # New plugin manager
        
        # Code and documentation components
        self.codebase_enhancer = CodebaseEnhancer(self.ai_generator)
        self.documentation_generator = DocumentationGenerator(self.ai_generator)  # New documentation generator
        self.testing_framework = TestingFramework()  # New testing framework
        
        # User and data components
        self.user_manager = UserManager()  # New user manager
        self.data_visualizer = DataVisualizer()  # New data visualizer
        self.version_migrator = VersionMigrator()  # New version migrator
        
        # Orchestration component
        self.agent_orchestrator = AgentOrchestrator(self)  # New orchestrator

        # System state
        self.last_push_time = 0
        self.cycle_count = 0
        self.start_time = time.time()
        
        # Initialize all plugins
        self._initialize_plugins()

    def _initialize_plugins(self):
        """Initialize all registered plugins."""
        try:
            LOGGER.info("Initializing plugins...")
            self.plugin_manager.load_plugins()
            LOGGER.info(f"Loaded {len(self.plugin_manager.get_plugins())} plugins")
        except Exception as e:
            LOGGER.error(f"Failed to initialize plugins: {e}")

    def add_system_activity(self, activity: str):
        """Adds an activity to the system's activity log."""
        with self._system_activities_lock:
            self.system_activities.append(activity)
            LOGGER.debug(f"System activity added: {activity}")

    def get_system_activities(self) -> List[str]:
        """Gets the system's activity log."""
        with self._system_activities_lock:
            return self.system_activities[:]

    def clear_system_activities(self):
        """Clears the system's activity log."""
        with self._system_activities_lock:
            self.system_activities.clear()
            LOGGER.debug("System activities cleared.")

    def _check_disk_space(self):
        """Checks available disk space."""
        try:
            if shutil.disk_usage("/").free < MIN_DISK_SPACE:
                LOGGER.warning("Low disk space detected. Consider archiving or removing files.")
                return False
            return True
        except Exception as e:
            LOGGER.error(f"Disk space check failed: {e}")
            return True

    def _perform_initial_setup(self):
        """Performs initial setup tasks."""
        try:
            # Initialize Git repository if needed
            if self.git_manager and not self.git_manager.is_repo_initialized():
                LOGGER.info("Initializing Git repository...")
                self.git_manager.initialize_repo()
            elif self.git_manager:
                LOGGER.info("Git repository already initialized.")

            # Create initial README if needed
            if not self.file_handler.file_exists("README.md"):
                LOGGER.info("Creating initial README.md...")
                self.file_handler.write_file("README.md", f"# {config.repo_name}\n\nAutonomous Knowledge System.")
                if self.git_manager:
                    self.git_manager.commit_and_push("Initial README.md creation")
            else:
                LOGGER.info("README.md already exists.")
                
            # Initialize vector database
            LOGGER.info("Initializing vector database...")
            self.vector_db.initialize()
            
            # Load initial plugins
            self._initialize_plugins()
            
        except Exception as e:
            LOGGER.exception(f"Initial setup failed: {e}")

    def _archive_old_data(self):
        """Archives old data to free up disk space."""
        try:
            LOGGER.info("Archiving old data...")
            self.file_handler.archive_directory(Path("/content/logs"), config.archive_dir / "logs")
            self.file_handler.archive_directory(config.user_feedback_dir, config.archive_dir / "user_feedback")
            if self.resilience_manager:
                self.resilience_manager.archive_old_snapshots()
            LOGGER.info("Archiving completed.")
        except Exception as e:
            LOGGER.exception(f"Archiving failed: {e}")

    def _run_codebase_enhancement(self) -> bool:
        """Runs the codebase enhancement process."""
        if random.random() < config.ai_activity_chance:
            try:
                LOGGER.info("Initiating codebase enhancement...")
                self.add_system_activity("Codebase Enhancement")
                
                # Schedule enhancement tasks
                task_results = self.task_scheduler.run_tasks([
                    lambda: self.codebase_enhancer.enhance_codebase(),
                    lambda: self.documentation_generator.update_documentation(),
                    lambda: self.testing_framework.run_tests()
                ])
                
                if task_results[0]:  # Code enhancement result
                    LOGGER.info("Codebase enhancement successful.")
                    if self.git_manager:
                        self.git_manager.commit_and_push("Enhanced codebase")
                    return True
                else:
                    LOGGER.info("Codebase enhancement skipped or failed.")
            except Exception as e:
                LOGGER.exception(f"Codebase enhancement failed: {e}")
        return False

    def _run_information_sourcing(self) -> bool:
        """Runs the information sourcing process."""
        if random.random() < config.ai_activity_chance:
            try:
                LOGGER.info("Initiating information sourcing...")
                self.add_system_activity("Information Sourcing")
                
                # Use task scheduler for parallel processing
                results = self.task_scheduler.run_tasks([
                    lambda: self.information_sourcing.gather_information(),
                    lambda: self.vector_db.update_indexes()  # Update vector indexes in parallel
                ])
                
                if results[0]:  # Information gathering result
                    if self.git_manager:
                        self.git_manager.commit_and_push("Updated knowledge base from information sourcing")
                    return True
            except Exception as e:
                LOGGER.exception(f"Information sourcing failed: {e}")
        return False

    def _run_collaborative_processing(self) -> bool:
        """Runs the collaborative processing tasks."""
        try:
            LOGGER.info("Initiating collaborative processing...")
            self.add_system_activity("Collaborative Processing")
            
            # Process feedback and update user profiles in parallel
            results = self.task_scheduler.run_tasks([
                self.collaborative_processor.process_feedback,
                self.user_manager.update_user_profiles
            ])
            
            if results[0]:  # Feedback processing result
                if self.git_manager:
                    self.git_manager.commit_and_push("Processed User Feedback")
                return True
        except Exception as e:
            LOGGER.exception(f"Collaborative processing failed: {e}")
            return False

    def _run_security_checks(self) -> bool:
        """Performs security checks and actions."""
        try:
            LOGGER.info("Running security checks...")
            self.add_system_activity("Security Checks")
            
            # Run security checks in parallel
            results = self.task_scheduler.run_tasks([
                lambda: self.security_manager.perform_system_integrity_checks(),
                lambda: self.security_manager.analyze_logs_for_anomalies(self.audit_manager.load_audit_log()),
                lambda: self.plugin_manager.scan_for_malicious_plugins()
            ])
            
            return any(results)  # Return True if any checks found issues
        except Exception as e:
            LOGGER.exception(f"Security checks failed: {e}")
            return False

    def _run_plugin_tasks(self) -> bool:
        """Execute scheduled plugin tasks."""
        try:
            LOGGER.info("Running plugin tasks...")
            self.add_system_activity("Plugin Execution")
            
            # Get all plugin tasks and execute them in parallel
            plugin_tasks = [plugin.execute for plugin in self.plugin_manager.get_plugins()]
            results = self.task_scheduler.run_tasks(plugin_tasks)
            
            return any(results)  # Return True if any plugins executed successfully
        except Exception as e:
            LOGGER.exception(f"Plugin execution failed: {e}")
            return False

    def _perform_snapshot(self):
        """Performs a system snapshot for resilience."""
        try:
            LOGGER.info("Creating system snapshot...")
            self.add_system_activity("Snapshot Creation")
            
            # Take snapshot of both filesystem and vector db state
            if self.resilience_manager:
                self.resilience_manager.create_snapshot()
            self.vector_db.create_snapshot()
            
            LOGGER.info("Snapshot created successfully.")
        except Exception as e:
            LOGGER.exception(f"Snapshot creation failed: {e}")

    def _push_changes(self):
        """Pushes local changes to the remote repository."""
        try:
            if self.git_manager and time.time() - self.last_push_time > config.push_interval:
                LOGGER.info("Pushing changes to remote repository...")
                self.add_system_activity("Pushing Changes")
                self.git_manager.push_changes()
                self.last_push_time = time.time()
                LOGGER.info("Changes pushed successfully.")
        except Exception as e:
            LOGGER.exception(f"Failed to push changes: {e}")

    def _analyze_logs(self):
        """Analyzes the system logs for anomalies and insights."""
        try:
            LOGGER.info("Analyzing logs...")
            self.add_system_activity("Log Analysis")
            
            # Run analysis tasks in parallel
            self.task_scheduler.run_tasks([
                lambda: self.audit_manager.analyze_log_for_anomalies(),
                lambda: self.monitoring.generate_performance_report(),
                lambda: self.data_visualizer.generate_activity_visualizations()
            ])
        except Exception as e:
            LOGGER.exception(f"Log analysis failed: {e}")

    def _run_autonomous_cycle(self) -> bool:
        """Runs a single autonomous cycle."""
        LOGGER.info(f"--- Starting Autonomous Cycle {self.cycle_count + 1} ---")
        self.add_system_activity(f"Cycle {self.cycle_count + 1} Start")

        # 1. Pre-Cycle Checks
        if not self._check_disk_space():
            self._archive_old_data()
            if not self._check_disk_space():
                LOGGER.critical("Disk space is critically low. Halting operation.")
                self.active = False
                return False

        # 2. Core Tasks - Run in optimized order with parallel execution
        any_activity = False
        any_activity |= self._run_security_checks()  # Security first
        any_activity |= self._run_plugin_tasks()  # Plugin tasks next
        any_activity |= self._run_information_sourcing()
        any_activity |= self._run_codebase_enhancement()
        any_activity |= self._run_collaborative_processing()
        
        # 3. System Maintenance Tasks
        self._analyze_logs()
        self._perform_snapshot()
        self._push_changes()

        # 4. Post-Cycle Cleanup
        self.clear_system_activities()
        LOGGER.info(f"--- Autonomous Cycle {self.cycle_count + 1} Complete ---")
        self.cycle_count += 1
        return True

    def run(self, continuous: bool = True):
        """Runs the autonomous agent."""
        LOGGER.info("Starting Autonomous Knowledge System...")
        
        # Validate configuration
        config_errors = config.validate()
        if config_errors:
            for error in config_errors:
                LOGGER.error(f"Configuration error: {error}")
            LOGGER.critical("Configuration validation failed. Exiting.")
            return

        # Check AI providers
        if not self.ai_provider_manager.has_available_providers():
            LOGGER.critical("No AI providers available. Check API keys and network connectivity. Exiting.")
            return

        # Initial setup
        self._perform_initial_setup()

        # Single cycle mode
        if not continuous:
            self._run_autonomous_cycle()
            LOGGER.info("Single cycle completed. Exiting.")
            return

        # Continuous operation mode
        try:
            while self.active:
                start_time = time.time()
                
                if not self._run_autonomous_cycle():
                    LOGGER.warning("Autonomous cycle failed. Halting.")
                    self.active = False
                    break
                    
                # Calculate sleep time based on cycle duration
                elapsed_time = time.time() - start_time
                sleep_time = max(0, config.cycle_interval - elapsed_time)
                
                LOGGER.info(f"Cycle completed in {elapsed_time:.2f} seconds. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            LOGGER.info("Shutdown initiated by user.")
        except Exception as e:
            LOGGER.exception(f"An unhandled exception occurred: {e}")
        finally:
            # Clean shutdown
            LOGGER.info("Shutting down Autonomous Knowledge System...")
            self.task_scheduler.shutdown()
            self.vector_db.close()
            LOGGER.info(f"System was active for {time.time() - self.start_time:.2f} seconds")

# --- UI / Main Execution ---
def run_aks_pipeline(user_zip_file, repo_url_input, analysis_claim, debate_topic, github_token_input, gemini_api_key_input, web_query, ftp_query, run_autonomous_cycle):
    """
    Main function for running the Autonomous Knowledge System pipeline.
    """
    log_output = ""
    summary_output = ""
    knowledge_output = ""
    try:
        # 1. Configuration Updates (from UI)
        config.github_token = github_token_input
        config.gemini_key = gemini_api_key_input
        
        # Update repository info if provided
        if repo_url_input:
            try:
                repo_owner = repo_url_input.split("/")[-2]
                config.repo_owner = repo_owner
                config.repo_name = repo_url_input.split("/")[-1].replace(".git", "")
                LOGGER.info(f"Setting repo owner: {config.repo_owner} and repo name: {config.repo_name}")
            except IndexError:
                LOGGER.warning("Could not determine repo owner/name from URL. Using default.")
                
        # Validate configuration
        config_errors = config.validate()
        if config_errors:
            for error in config_errors:
                LOGGER.error(f"Configuration error: {error}")
            return log_output, "Configuration errors. Check logs.", "Configuration errors. Check logs."

        # 2. Initialize the Agent
        agent = AutonomousAgent()
        agent._perform_initial_setup()
        
        if not agent.ai_provider_manager.has_available_providers():
            return log_output, "No AI providers available.", "Check API keys and network."

        # 3. Process User Uploads (if any)
        if user_zip_file:
            try:
                with zipfile.ZipFile(user_zip_file.name, 'r') as zip_ref:
                    zip_ref.extractall(config.user_uploads_dir)
                LOGGER.info(f"Extracted user files to: {config.user_uploads_dir}")
                
                # Process the uploaded files
                agent.task_scheduler.run_task(
                    lambda: agent.information_sourcing.process_uploaded_files(config.user_uploads_dir)
                )
            except Exception as e:
                LOGGER.error(f"Error processing uploaded zip file: {e}")
                log_output += f"Error processing uploaded zip file: {e}\n"

        # 4. Run Pipeline
        if run_autonomous_cycle:
            # Start continuous operation in a separate thread
            operation_thread = threading.Thread(target=agent.run, kwargs={'continuous': True})
            operation_thread.daemon = True
            operation_thread.start()
            
            log_output = "Autonomous cycle running in the background. Check the logs for status.\n"
        else:
            # Run single cycle
            agent._run_autonomous_cycle()
            log_output = "Single autonomous cycle completed. Check the logs for status.\n"

        # 5. Output Summary
        summary_output = f"AKS cycle completed. Logged to /content/logs/aks.log. Repo: {config.repo_owner}/{config.repo_name}"
        knowledge_output = "Knowledge base updated (check the Git repo)."
        
    except Exception as e:
        LOGGER.exception(f"Pipeline execution failed: {e}")
        log_output += f"Pipeline execution failed: {e}\n"
        summary_output = "Pipeline failed. Check the logs for details."
        knowledge_output = "Error during knowledge processing."
    finally:
        return log_output, summary_output, knowledge_output

# --- Gradio UI ---
if gr:
    with gr.Blocks(title="Autonomous Knowledge System (AKS)") as demo:
        with gr.Tab("Input"):
            with gr.Row():
                user_zip_file_input = gr.File(label="Upload Files (ZIP)", file_types=[".zip"])
            with gr.Row():
                repo_url_input = gr.Textbox(label="GitHub Repository URL", placeholder="https://github.com/Craig444444444/AKS")
            with gr.Row():
                analysis_claim_input = gr.Textbox(label="Knowledge Analysis Claim (Optional)", placeholder="Analyze the code structure")
            with gr.Row():
                debate_topic_input = gr.Textbox(label="Debate Topic (Optional)", placeholder="The ethics of AI")
            with gr.Row():
                github_token_input = gr.Textbox(label="GitHub Token (for Repository Operations)", type="password")
            with gr.Row():
                gemini_api_key_input = gr.Textbox(label="Gemini API Key (Optional - for Gemini API)", type="password")
            with gr.Row():
                web_query_input = gr.Textbox(label="Web Research Query (Optional)", placeholder="Best practices for Python code")
            with gr.Row():
                ftp_query_input = gr.Textbox(label="FTP Query (Optional)", placeholder="Download data from FTP")
            with gr.Row():
                run_autonomous_cycle_input = gr.Checkbox(label="Run Autonomous Cycle (Continuous)", value=True)
        with gr.Tab("Output"):
            with gr.Row():
                log_output = gr.Textbox(label="System Log Output", lines=10)
            with gr.Row():
                summary_output = gr.Textbox(label="Summary", lines=2)
            with gr.Row():
                knowledge_output = gr.Textbox(label="Knowledge Base Status", lines=2)
        run_button = gr.Button("Run Autonomous Knowledge System")
        run_button.click(
            fn=run_aks_pipeline,
            inputs=[user_zip_file_input, repo_url_input, analysis_claim_input, debate_topic_input, github_token_input, gemini_api_key_input, web_query_input, ftp_query_input, run_autonomous_cycle_input],
            outputs=[log_output, summary_output, knowledge_output]
        )
    demo.launch(debug=True, share=False)
else:
    LOGGER.warning("Gradio not installed. Running in console-only mode.")
    if __name__ == "__main__":
        LOGGER.info("--- Autonomous Knowledge System (AKS) - Console Mode ---")
        
        # Get GitHub token securely
        github_token = config.github_token if config.github_token else getpass("Enter your GitHub Personal Access Token (PAT): ")
        config.github_token = github_token

        # Get Gemini API Key securely
        gemini_api_key = config.gemini_key if config.gemini_key else getpass("Enter your Gemini API Key (Optional): ")
        config.gemini_key = gemini_api_key

        # Basic input for repo details
        repo_owner = input("Enter GitHub repo owner (or press Enter for default): ") or config.repo_owner
        config.repo_owner = repo_owner
        config.repo_name = input("Enter GitHub repo name (or press Enter for default): ") or config.repo_name
        
        # Run the agent
        agent = AutonomousAgent()
        agent.run(continuous=True)
        LOGGER.info("Autonomous Knowledge System (AKS) finished.")
