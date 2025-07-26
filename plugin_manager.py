import logging
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import sys
import json
from dataclasses import dataclass

LOGGER = logging.getLogger("aks")

@dataclass
class PluginInfo:
    """Metadata about a loaded plugin"""
    name: str
    version: str
    description: str
    author: str
    plugin_class: Type
    module_path: Path

class PluginManager:
    """
    Manages dynamic loading and unloading of plugins for the AKS system.
    Provides secure plugin execution and dependency management.
    """
    def __init__(self, plugins_dir: Path):
        self.plugins_dir = plugins_dir.resolve()
        self._loaded_plugins: Dict[str, PluginInfo] = {}
        self._setup_plugin_environment()
        LOGGER.info(f"Initialized PluginManager for directory: {self.plugins_dir}")

    def _setup_plugin_environment(self):
        """Create required directories and security checks"""
        try:
            self.plugins_dir.mkdir(parents=True, exist_ok=True)
            # Set restrictive permissions
            self.plugins_dir.chmod(0o755)
            
            # Create subdirectories
            (self.plugins_dir / "cache").mkdir(exist_ok=True)
            (self.plugins_dir / "disabled").mkdir(exist_ok=True)
            
            LOGGER.debug("Plugin environment setup complete")
        except Exception as e:
            LOGGER.error(f"Failed to setup plugin environment: {e}")
            raise RuntimeError("Plugin initialization failed") from e

    def discover_plugins(self) -> List[Path]:
        """Scan plugin directory for valid plugin modules"""
        plugin_files = []
        for file in self.plugins_dir.glob("*.py"):
            if file.name.startswith("plugin_") and file.is_file():
                plugin_files.append(file)
        
        LOGGER.info(f"Discovered {len(plugin_files)} potential plugins")
        return plugin_files

    def load_plugin(self, plugin_path: Path) -> Optional[PluginInfo]:
        """
        Load a plugin module with security checks.
        
        Args:
            plugin_path: Path to the plugin Python file
            
        Returns:
            PluginInfo if successfully loaded, None otherwise
        """
        try:
            # Security validation
            if not self._validate_plugin(plugin_path):
                LOGGER.warning(f"Plugin validation failed: {plugin_path.name}")
                return None

            # Generate module name
            module_name = f"aks_plugins.{plugin_path.stem}"
            
            # Load module
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not get spec for {plugin_path}")
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = self._find_plugin_class(module)
            if not plugin_class:
                raise ValueError("No valid plugin class found")
                
            # Get metadata
            metadata = self._extract_plugin_metadata(plugin_class, plugin_path)
            
            # Initialize plugin
            plugin_info = PluginInfo(
                name=metadata.get("name", plugin_path.stem),
                version=metadata.get("version", "1.0.0"),
                description=metadata.get("description", ""),
                author=metadata.get("author", "Unknown"),
                plugin_class=plugin_class,
                module_path=plugin_path
            )
            
            self._loaded_plugins[plugin_info.name] = plugin_info
            LOGGER.info(f"Loaded plugin: {plugin_info.name} v{plugin_info.version}")
            return plugin_info
            
        except Exception as e:
            LOGGER.error(f"Failed to load plugin {plugin_path.name}: {e}", exc_info=True)
            return None

    def _validate_plugin(self, plugin_path: Path) -> bool:
        """Perform security and structure validation on plugin file"""
        try:
            # Basic checks
            if not plugin_path.exists():
                return False
                
            # Size limit (1MB)
            if plugin_path.stat().st_size > 1024 * 1024:
                LOGGER.warning(f"Plugin too large: {plugin_path.name}")
                return False
                
            # Check for required sections
            content = plugin_path.read_text(encoding='utf-8')
            if not all(tag in content for tag in ["PLUGIN_METADATA", "class Plugin"]):
                LOGGER.warning(f"Plugin missing required sections: {plugin_path.name}")
                return False
                
            return True
        except Exception as e:
            LOGGER.warning(f"Plugin validation error: {e}")
            return False

    def _find_plugin_class(self, module) -> Optional[Type]:
        """Find the plugin class in a module"""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                name == "Plugin" and 
                hasattr(obj, "execute")):
                return obj
        return None

    def _extract_plugin_metadata(self, plugin_class: Type, plugin_path: Path) -> Dict[str, str]:
        """Extract metadata from plugin class or file"""
        metadata = {
            "name": plugin_path.stem.replace("plugin_", ""),
            "version": "1.0.0",
            "description": "",
            "author": "Unknown"
        }
        
        try:
            if hasattr(plugin_class, "METADATA"):
                metadata.update(plugin_class.METADATA)
                
            # Parse docstring
            if plugin_class.__doc__:
                doc_metadata = self._parse_docstring(plugin_class.__doc__)
                metadata.update(doc_metadata)
                
            return metadata
        except Exception as e:
            LOGGER.warning(f"Metadata extraction failed: {e}")
            return metadata

    def _parse_docstring(self, docstring: str) -> Dict[str, str]:
        """Parse metadata from docstring"""
        result = {}
        lines = [line.strip() for line in docstring.split("\n") if line.strip()]
        
        for line in lines:
            if line.startswith("@version:"):
                result["version"] = line.split(":", 1)[1].strip()
            elif line.startswith("@author:"):
                result["author"] = line.split(":", 1)[1].strip()
            elif line.startswith("@description:"):
                result["description"] = line.split(":", 1)[1].strip()
                
        return result

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin by name"""
        if plugin_name not in self._loaded_plugins:
            LOGGER.warning(f"Plugin not loaded: {plugin_name}")
            return False
            
        try:
            # Remove from sys.modules
            module_name = f"aks_plugins.{self._loaded_plugins[plugin_name].module_path.stem}"
            if module_name in sys.modules:
                del sys.modules[module_name]
                
            # Remove from loaded plugins
            del self._loaded_plugins[plugin_name]
            
            LOGGER.info(f"Unloaded plugin: {plugin_name}")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False

    def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """Get an instance of a loaded plugin"""
        if plugin_name not in self._loaded_plugins:
            return None
            
        try:
            return self._loaded_plugins[plugin_name].plugin_class()
        except Exception as e:
            LOGGER.error(f"Failed to instantiate plugin {plugin_name}: {e}")
            return None

    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """
        Execute a plugin's main functionality with error handling
        
        Args:
            plugin_name: Name of the plugin to execute
            *args: Positional arguments to pass to plugin
            **kwargs: Keyword arguments to pass to plugin
            
        Returns:
            Plugin execution result or None if failed
        """
        if plugin_name not in self._loaded_plugins:
            LOGGER.error(f"Plugin not loaded: {plugin_name}")
            return None
            
        try:
            plugin = self.get_plugin(plugin_name)
            if not plugin:
                return None
                
            LOGGER.info(f"Executing plugin: {plugin_name}")
            return plugin.execute(*args, **kwargs)
        except Exception as e:
            LOGGER.error(f"Plugin execution failed: {plugin_name} - {e}", exc_info=True)
            return None

    def load_all_plugins(self) -> Dict[str, PluginInfo]:
        """Load all valid plugins in the plugins directory"""
        plugin_files = self.discover_plugins()
        for plugin_file in plugin_files:
            self.load_plugin(plugin_file)
        return self._loaded_plugins.copy()

    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin by moving it to disabled directory"""
        if plugin_name not in self._loaded_plugins:
            return False
            
        try:
            plugin_info = self._loaded_plugins[plugin_name]
            disabled_path = self.plugins_dir / "disabled" / plugin_info.module_path.name
            
            # Move the file
            shutil.move(str(plugin_info.module_path), str(disabled_path))
            
            # Unload the plugin
            self.unload_plugin(plugin_name)
            
            LOGGER.info(f"Disabled plugin: {plugin_name}")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to disable plugin {plugin_name}: {e}")
            return False

    def enable_plugin(self, plugin_filename: str) -> Optional[PluginInfo]:
        """Enable a disabled plugin"""
        disabled_path = self.plugins_dir / "disabled" / plugin_filename
        if not disabled_path.exists():
            return None
            
        try:
            # Move back to plugins directory
            enabled_path = self.plugins_dir / plugin_filename
            shutil.move(str(disabled_path), str(enabled_path))
            
            # Load the plugin
            return self.load_plugin(enabled_path)
        except Exception as e:
            LOGGER.error(f"Failed to enable plugin {plugin_filename}: {e}")
            return None

    def list_plugins(self) -> Dict[str, Dict[str, str]]:
        """Get information about all loaded plugins"""
        return {
            name: {
                "version": info.version,
                "description": info.description,
                "author": info.author,
                "path": str(info.module_path)
            }
            for name, info in self._loaded_plugins.items()
        }

    def verify_plugin_dependencies(self, plugin_name: str) -> bool:
        """Check if a plugin's dependencies are satisfied"""
        if plugin_name not in self._loaded_plugins:
            return False
            
        plugin = self.get_plugin(plugin_name)
        if not plugin or not hasattr(plugin, "REQUIREMENTS"):
            return True
            
        try:
            requirements = plugin.REQUIREMENTS
            for req in requirements.get("plugins", []):
                if req not in self._loaded_plugins:
                    LOGGER.warning(f"Missing plugin dependency: {req}")
                    return False
                    
            # TODO: Add package dependency checking
            return True
        except Exception as e:
            LOGGER.error(f"Dependency check failed for {plugin_name}: {e}")
            return False

    def save_plugin_state(self) -> bool:
        """Save current plugin configuration to disk"""
        state = {
            "loaded_plugins": list(self._loaded_plugins.keys()),
            "disabled_plugins": [
                f.name for f in (self.plugins_dir / "disabled").iterdir() 
                if f.is_file() and f.name.endswith(".py")
            ]
        }
        
        try:
            state_path = self.plugins_dir / "cache" / "plugin_state.json"
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            LOGGER.error(f"Failed to save plugin state: {e}")
            return False

    def restore_plugin_state(self) -> bool:
        """Restore plugin state from saved configuration"""
        state_path = self.plugins_dir / "cache" / "plugin_state.json"
        if not state_path.exists():
            return False
            
        try:
            with open(state_path) as f:
                state = json.load(f)
                
            # Load enabled plugins
            for plugin_name in state.get("loaded_plugins", []):
                plugin_file = f"plugin_{plugin_name}.py"
                if (self.plugins_dir / plugin_file).exists():
                    self.load_plugin(self.plugins_dir / plugin_file)
                    
            # Ensure disabled plugins stay disabled
            for plugin_file in state.get("disabled_plugins", []):
                if (self.plugins_dir / "disabled" / plugin_file).exists():
                    continue
                if (self.plugins_dir / plugin_file).exists():
                    self.disable_plugin(plugin_file.replace(".py", "").replace("plugin_", ""))
                    
            return True
        except Exception as e:
            LOGGER.error(f"Failed to restore plugin state: {e}")
            return False
