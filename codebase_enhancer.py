import logging
import ast
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import shutil
import tempfile
import hashlib

LOGGER = logging.getLogger("aks")

class CodebaseEnhancer:
    """
    Autonomous codebase enhancement system that analyzes and improves
    Python code quality, structure, and documentation.
    """
    def __init__(self, ai_generator):
        self.ai_generator = ai_generator
        self._backup_dir = Path("/content/code_backups")
        self._backup_dir.mkdir(exist_ok=True)
        LOGGER.info("CodebaseEnhancer initialized")

    def enhance_codebase(self, target_dir: Optional[Path] = None) -> bool:
        """
        Main enhancement workflow that safely improves the codebase.
        
        Args:
            target_dir: Directory to enhance (defaults to repository root)
            
        Returns:
            bool: True if enhancements were made, False otherwise
        """
        target_dir = target_dir or self.ai_generator.repo_path
        LOGGER.info(f"Starting codebase enhancement in {target_dir}")

        try:
            # Create backup before making changes
            backup_path = self._create_backup(target_dir)
            if not backup_path:
                LOGGER.error("Backup failed, aborting enhancement")
                return False

            # Get all Python files in directory
            py_files = list(target_dir.rglob("*.py"))
            if not py_files:
                LOGGER.warning("No Python files found to enhance")
                return False

            # Process files with enhancements
            enhancements_made = False
            for file_path in py_files:
                if self._enhance_file(file_path):
                    enhancements_made = True

            if enhancements_made:
                LOGGER.info("Codebase enhancements completed successfully")
            else:
                LOGGER.info("No enhancements were needed")
            return enhancements_made

        except Exception as e:
            LOGGER.error(f"Codebase enhancement failed: {e}", exc_info=True)
            return False

    def _enhance_file(self, file_path: Path) -> bool:
        """
        Enhance a single Python file with multiple improvements.
        
        Args:
            file_path: Path to Python file to enhance
            
        Returns:
            bool: True if file was modified, False otherwise
        """
        LOGGER.debug(f"Enhancing file: {file_path}")
        
        try:
            with file_path.open('r', encoding='utf-8') as f:
                original_content = f.read()

            if not original_content.strip():
                return False

            # Generate enhanced version
            enhanced_content = self._generate_enhanced_version(original_content, file_path)
            if not enhanced_content or enhanced_content == original_content:
                return False

            # Validate syntax before writing
            if not self._validate_python_syntax(enhanced_content):
                LOGGER.warning(f"Enhanced version of {file_path.name} has syntax errors")
                return False

            # Write enhanced version
            with file_path.open('w', encoding='utf-8') as f:
                f.write(enhanced_content)

            LOGGER.info(f"Successfully enhanced {file_path.name}")
            return True

        except Exception as e:
            LOGGER.error(f"Failed to enhance {file_path.name}: {e}")
            return False

    def _generate_enhanced_version(self, original_content: str, file_path: Path) -> Optional[str]:
        """
        Generate enhanced version of code using AI with proper prompting.
        
        Args:
            original_content: Original source code
            file_path: Path to source file (for context)
            
        Returns:
            str: Enhanced code content or None if failed
        """
        prompt = f"""Improve this Python code while preserving all functionality:
{original_content}

Apply these enhancements:
1. Add complete Google-style docstrings
2. Add type hints
3. Follow PEP 8 style guidelines
4. Optimize imports
5. Break down complex functions
6. Add error handling where missing
7. Add logging where appropriate
8. Add unit test examples

Return only the improved code with no additional commentary or explanations."""

        try:
            enhanced_code = self.ai_generator.ai_manager.generate_code(
                prompt,
                system_prompt="You are an expert Python developer. Return only valid Python code.",
                max_tokens=4000
            )

            if not enhanced_code:
                return None

            # Clean up any markdown code blocks
            if "```python" in enhanced_code:
                enhanced_code = re.sub(r'```python\s*', '', enhanced_code)
                enhanced_code = re.sub(r'\s*```', '', enhanced_code)

            return enhanced_code.strip()

        except Exception as e:
            LOGGER.error(f"AI enhancement generation failed: {e}")
            return None

    def _validate_python_syntax(self, code: str) -> bool:
        """Validate Python syntax using AST."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            LOGGER.warning(f"Syntax validation failed: {e}")
            return False

    def _create_backup(self, target_dir: Path) -> Optional[Path]:
        """Create timestamped backup of directory before modifications."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self._backup_dir / f"pre_enhance_{timestamp}"
            
            shutil.copytree(
                target_dir,
                backup_path,
                ignore=shutil.ignore_patterns('*.pyc', '__pycache__'),
                dirs_exist_ok=False
            )
            
            LOGGER.info(f"Created backup at {backup_path}")
            return backup_path
        except Exception as e:
            LOGGER.error(f"Backup creation failed: {e}")
            return None

    def analyze_code_quality(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze code quality metrics for a Python file.
        
        Args:
            file_path: Path to Python file to analyze
            
        Returns:
            dict: Analysis results with quality metrics
        """
        analysis = {
            "file": str(file_path),
            "metrics": {},
            "issues": [],
            "timestamp": datetime.now().isoformat()
        }

        try:
            with file_path.open('r', encoding='utf-8') as f:
                content = f.read()

            # Basic metrics
            tree = ast.parse(content)
            analysis["metrics"] = {
                "lines": len(content.splitlines()),
                "functions": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                "classes": len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                "imports": len([node for node in ast.walk(tree) if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)]),
            }

            # Detect common issues
            issues = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for missing docstrings
                    if not ast.get_docstring(node):
                        issues.append(f"Missing docstring for function {node.name}")
                    
                    # Check for long functions
                    if len(node.body) > 30:
                        issues.append(f"Function {node.name} is too long ({len(node.body)} lines)")

            analysis["issues"] = issues
            analysis["hash"] = hashlib.sha256(content.encode()).hexdigest()

        except Exception as e:
            LOGGER.error(f"Code analysis failed for {file_path}: {e}")
            analysis["error"] = str(e)

        return analysis

    def get_enhancement_suggestions(self, file_path: Path) -> List[str]:
        """
        Get specific enhancement suggestions for a Python file.
        
        Args:
            file_path: Path to Python file to analyze
            
        Returns:
            list: Specific enhancement suggestions
        """
        try:
            with file_path.open('r', encoding='utf-8') as f:
                content = f.read()

            prompt = f"""Analyze this Python code and suggest specific improvements:
{content}

Return a bulleted list of specific, actionable improvements.
Focus on:
- Code structure
- Documentation
- Performance
- Error handling
- Modern Python features
- Testing"""

            suggestions = self.ai_generator.ai_manager.generate_text(
                prompt,
                system_prompt="You are a code quality expert. Return a bulleted list of specific improvements.",
                max_tokens=1000
            )

            if suggestions:
                return [s.strip() for s in suggestions.split('\n') if s.strip()]
            return []

        except Exception as e:
            LOGGER.error(f"Failed to get enhancement suggestions: {e}")
            return []
