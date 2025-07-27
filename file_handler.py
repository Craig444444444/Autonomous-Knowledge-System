import logging
import shutil
import os
import zipfile
from pathlib import Path
from typing import Optional
import PyPDF2
from datetime import datetime
import json
import ast
import re

# SAFE DOCX IMPORT FIX
# This block is now correctly indented, resolving the IndentationError.
try:
    from docx import Document
except ImportError:
    # This fallback ensures the program can run even if python-docx is not installed.
    Document = None
    logging.warning("python-docx module not available - DOCX extraction will be limited.")

# It's good practice to get a logger specific to your module/application.
LOGGER = logging.getLogger("aks")


class FileHandler:
    """
    Enhanced file handler with multi-format support, atomic writes, and security features.
    - Reads text from .txt, .py, .md, .json, .pdf, and .docx.
    - Writes only to safe, text-based formats (.txt, .py, .md, .json).
    - Includes features for atomic writes, backups on deletion, and directory archiving.
    """
    def __init__(self, base_path: Path):
        """
        Initializes the FileHandler with a base directory.
        Args:
            base_path (Path): The root directory for all file operations.
        """
        self.base_path = base_path
        # Supported extensions for READING
        self.supported_read_extensions = ['.txt', '.py', '.md', '.pdf', '.docx', '.json']
        # Supported extensions for WRITING
        self.supported_write_extensions = ['.txt', '.py', '.md', '.json']
        LOGGER.info(f"FileHandler initialized at base path: {self.base_path}")

    def file_exists(self, relative_path: str) -> bool:
        """Checks if a file exists relative to the base path."""
        return (self.base_path / relative_path).is_file()

    def read_file(self, relative_path: str) -> Optional[str]:
        """
        Reads a file with automatic format handling and robust encoding detection.
        Args:
            relative_path (str): The path to the file relative to the base directory.
        Returns:
            Optional[str]: The file content as a string, or None if reading fails.
        """
        path = self.base_path / relative_path
        if not self.file_exists(relative_path):
            LOGGER.warning(f"File not found: {path}")
            return None

        try:
            # Handle different file formats based on their extension.
            if path.suffix == '.pdf':
                return self._read_pdf(path)
            elif path.suffix == '.docx':
                return self._read_docx(path)
            else:
                # For plain text files (including .json), try multiple common encodings.
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                    try:
                        with open(path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
                LOGGER.error(f"Failed to decode '{relative_path}' with any common encoding.")
                return None
        except Exception as e:
            LOGGER.error(f"Error reading file '{relative_path}': {e}", exc_info=True)
            return None

    def _read_pdf(self, path: Path) -> Optional[str]:
        """Helper method to extract text from PDF files."""
        try:
            text_parts = []
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text_parts.append(extracted_text)
            return "\n".join(text_parts)
        except Exception as e:
            LOGGER.error(f"PDF extraction failed for '{path}': {e}", exc_info=True)
            return None

    def _read_docx(self, path: Path) -> Optional[str]:
        """Helper method to extract text from DOCX files."""
        if Document is None:
            LOGGER.warning("Cannot read DOCX file because python-docx is not installed.")
            return None
        try:
            doc = Document(path)
            # Extracts text from all paragraphs in the document body.
            return "\n".join(para.text for para in doc.paragraphs)
        except Exception as e:
            LOGGER.error(f"DOCX extraction failed for '{path}': {e}", exc_info=True)
            return None

    def write_file(self, relative_path: str, content: str) -> bool:
        """
        Writes content to a text-based file using an atomic operation.
        This prevents file corruption if the write is interrupted.
        Args:
            relative_path (str): The path for the file to be written.
            content (str): The string content to write to the file.
        Returns:
            bool: True on success, False on failure.
        """
        path = self.base_path / relative_path
        
        try:
            # 1. Validate file extension to prevent writing to binary formats like .pdf
            if path.suffix not in self.supported_write_extensions:
                LOGGER.error(f"Write error: '{path.suffix}' is not a supported text format.")
                return False

            # 2. Validate content size to prevent accidentally writing huge files.
            if len(content.encode('utf-8')) > 10 * 1024 * 1024:  # 10MB limit
                LOGGER.error(f"Write error: Content for '{relative_path}' exceeds 10MB size limit.")
                return False
            
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # 3. Atomic Write: Write to a temporary file first.
            temp_path = path.with_suffix(f"{path.suffix}.tmp")
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # 4. Validate content if it's a Python file.
            if path.suffix == '.py':
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    LOGGER.error(f"Invalid Python syntax in '{relative_path}': {e}")
                    os.remove(temp_path)  # Clean up the invalid temp file
                    return False

            # 5. If all checks pass, replace the original file with the new one.
            os.replace(temp_path, path)
            LOGGER.info(f"Successfully wrote to '{relative_path}'")
            return True
        except Exception as e:
            LOGGER.error(f"Error writing to '{relative_path}': {e}", exc_info=True)
            # Clean up temp file on any failure
            if 'temp_path' in locals() and temp_path.exists():
                os.remove(temp_path)
            return False

    def delete_file(self, relative_path: str) -> bool:
        """
        Deletes a file, creating a backup first.
        Args:
            relative_path (str): The path of the file to delete.
        Returns:
            bool: True on success, False if the file doesn't exist or on error.
        """
        path = self.base_path / relative_path
        if not self.file_exists(relative_path):
            LOGGER.warning(f"Delete failed: File '{path}' does not exist.")
            return False

        try:
            # Create a dedicated directory for backups.
            backup_dir = self.base_path / "_deleted_backups"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Sanitize the filename to be cross-platform safe for the backup copy.
            sanitized_name = str(relative_path).replace('/', '_').replace('\\', '_')
            backup_path = backup_dir / f"{sanitized_name}_{timestamp}"
            
            shutil.copy(path, backup_path)

            # Delete the original file.
            path.unlink()
            LOGGER.info(f"Deleted '{relative_path}' (backup saved to '{backup_path}')")
            return True
        except Exception as e:
            LOGGER.error(f"Error deleting '{relative_path}': {e}", exc_info=True)
            return False

    def archive_directory(self, source_relative_path: str, target_relative_path: str) -> Optional[Path]:
        """
        Archives a directory into a zip file.
        Args:
            source_relative_path (str): The relative path of the directory to archive.
            target_relative_path (str): The relative path where the zip file will be saved.
        Returns:
            Optional[Path]: The path to the created zip file, or None on failure.
        """
        source_dir = self.base_path / source_relative_path
        target_dir = self.base_path / target_relative_path

        if not source_dir.is_dir():
            LOGGER.error(f"Archive failed: Source '{source_dir}' is not a valid directory.")
            return None

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d")
            zip_path = target_dir / f"archive_{source_dir.name}_{timestamp}.zip"

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Use pathlib's rglob for a clean, recursive file search.
                for file_path in source_dir.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(source_dir))

            LOGGER.info(f"Successfully archived '{source_dir}' to '{zip_path}'")
            return zip_path
        except Exception as e:
            LOGGER.error(f"Archive failed for '{source_dir}': {e}", exc_info=True)
            return None

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitizes a filename to remove dangerous characters and path information.
        Args:
            filename (str): The original filename.
        Returns:
            str: A sanitized, safe filename.
        """
        # 1. Strip any directory path to prevent path traversal attacks (e.g., "../../etc/passwd")
        clean_name = os.path.basename(filename)
        # 2. Remove characters that are illegal in Windows/Linux filesystems.
        clean_name = re.sub(r'[<>:"/\\|?*]', '_', clean_name)
        # 3. Limit the length of the filename to a reasonable maximum.
        return clean_name[:255]

