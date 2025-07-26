import logging
import re
import json
import ast
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

LOGGER = logging.getLogger("aks")

class AIGenerator:
    """Handles AI-driven generation tasks with integrated file processing."""
    def __init__(self, ai_manager: Any, repo_path: Path, file_handler: Any, max_history: int = 50):
        """
        Initialize with file handling capabilities.

        Args:
            ai_manager: Reference to AI provider manager
            repo_path: Path to main repository
            file_handler: FileHandler instance for file operations
            max_history: Maximum history entries to maintain
        """
        self.ai_manager = ai_manager
        self.repo_path = repo_path
        self.file_handler = file_handler
        self.history: List[str] = []
        self.max_history = max_history
        LOGGER.info("AI Generator initialized with file handling")

    def generate_from_file(self, file_path: str, prompt: str = "", max_tokens: int = 4096) -> Optional[str]:
        """
        Generate content based on a user file with optional prompt.

        Args:
            file_path: Path to file in /content or subdirectories
            prompt: Additional instructions for generation
            max_tokens: Maximum tokens for AI response

        Returns:
            Generated content or None if failed
        """
        try:
            content = self.file_handler.get_file_content(file_path)
            if not content:
                LOGGER.error(f"Could not read file: {file_path}")
                return None

            full_prompt = f"""File Content:
{content}

Additional Instructions:
{prompt}"""

            result = self.ai_manager.generate_text(
                full_prompt,
                "You are an AI assistant that processes files",
                max_tokens
            )

            if result:
                self._add_to_history(f"Generated from {Path(file_path).name}")
                return result
            return None

        except Exception as e:
            LOGGER.error(f"File generation failed: {e}")
            return None

    def generate_new_code(self, requirements: str, context: str = "", max_tokens: int = 4096) -> Optional[str]:
        """Generate Python code from requirements."""
        try:
            prompt = f"""Create a complete Python script based on these requirements:
{requirements}

Additional Context:
{context}

Provide only the raw Python code without explanations."""

            code = self.ai_manager.generate_code(
                prompt,
                "You are an expert Python developer",
                max_tokens
            )

            if code and self._validate_python(code):
                self._add_to_history("Generated new Python code")
                return code
            return None

        except Exception as e:
            LOGGER.error(f"Code generation failed: {e}")
            return None

    def import_user_file(self, src_path: str, dest_subdir: str = "user_files") -> Optional[Path]:
        """
        Copy user file into repository workspace.

        Args:
            src_path: Source path in /content
            dest_subdir: Destination subdirectory in repo

        Returns:
            Path to copied file or None if failed
        """
        return self.file_handler.copy_to_repo(src_path, dest_subdir)

    def _validate_python(self, code: str) -> bool:
        """Validate Python syntax using AST."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            LOGGER.warning(f"Invalid Python syntax: {e}")
            return False

    def _add_to_history(self, activity: str) -> None:
        """Add timestamped activity to history."""
        entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M')}: {activity}"
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        LOGGER.debug(f"History updated: {activity}")

    def generate_documentation(self, content: str, content_type: str = "code", max_tokens: int = 1024) -> Optional[str]:
        """Generate documentation for various content types."""
        prompt = f"""Create comprehensive documentation for this {content_type}:
{content}

Include purpose, usage examples, and parameters where applicable."""

        return self.ai_manager.generate_text(
            prompt,
            f"You are a technical documentation expert for {content_type}",
            max_tokens
        )

    def generate_data_transformation(self, source_data: str, target_format: str, max_tokens: int = 2048) -> Optional[str]:
        """Convert data between formats."""
        prompt = f"""Convert this data to {target_format} format:
{source_data}

Provide only the converted output without explanations."""

        return self.ai_manager.generate_text(
            prompt,
            f"You are a data conversion specialist",
            max_tokens
        )

    def get_recent_activities(self, count: int = 5) -> List[str]:
        """Get recent generation activities."""
        return self.history[-count:] if self.history else ["No history available"]
