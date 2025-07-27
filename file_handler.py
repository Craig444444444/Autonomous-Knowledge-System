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

# DOCX Import Handling
try:
    from docx import Document
except ImportError:
    Document = None
    logging.warning("docx module not available. .docx files will not be readable.")

# Configure logger
LOGGER = logging.getLogger("aks")
if not LOGGER.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


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
        Raises:
            ValueError: If base_path is not a valid directory.
        """
        self.base_path = Path(base_path).resolve()
        if not self.base_path.is_dir():
            try:
                self.base_path.mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"Base path '{self.base_path}' created successfully.")
            except OSError as e:
                LOGGER.critical(f"Failed to create base path '{self.base_path}': {e}")
                raise ValueError(
                    f"Base path '{self.base_path}' is not a valid directory and could not be created."
                ) from e

        # Supported extensions for READING
        self.supported_read_extensions = ['.txt', '.py', '.md', '.pdf', '.docx', '.json']
        # Supported extensions for WRITING
        self.supported_write_extensions = ['.txt', '.py', '.md', '.json']
        LOGGER.info(f"FileHandler initialized at base path: {self.base_path}")

    def file_exists(self, relative_path: str) -> bool:
        """
        Checks if a file exists relative to the base path.
        Args:
            relative_path (str): The path to the file relative to the base directory.
        Returns:
            bool: True if the file exists, False otherwise.
        """
        absolute_path = (self.base_path / relative_path).resolve()
        if not absolute_path.is_relative_to(self.base_path):
            LOGGER.warning(f"Attempted to access file outside base path: {relative_path}")
            return False
        return absolute_path.is_file()

    def read_file(self, relative_path: str) -> Optional[str]:
        """
        Reads a file with automatic format handling and robust encoding detection.
        Args:
            relative_path (str): The path to the file relative to the base directory.
        Returns:
            Optional[str]: The file content as a string, or None if reading fails.
        """
        path = (self.base_path / relative_path).resolve()
        
        if not self.file_exists(relative_path):
            LOGGER.warning(f"File not found or inaccessible: {path}")
            return None

        # Check if the extension is supported for reading
        if path.suffix not in self.supported_read_extensions:
            LOGGER.error(f"Unsupported file type for reading: '{path.suffix}' in file '{relative_path}'")
            return None

        try:
            # Handle different file formats
            if path.suffix == '.pdf':
                return self._read_pdf(path)
            elif path.suffix == '.docx':
                return self._read_docx(path)
            else:
                # For text files, try multiple encodings
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        with open(path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        LOGGER.error(
                            f"Error opening/reading '{relative_path}' with encoding '{encoding}': {e}"
                        )
                        return None
                LOGGER.error(f"Failed to decode '{relative_path}' with any common encoding.")
                return None
        except Exception as e:
            LOGGER.error(f"Error reading '{relative_path}': {e}")
            return None

    def _read_pdf(self, path: Path) -> Optional[str]:
        """Extract text from PDF files."""
        try:
            text_parts = []
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                if reader.is_encrypted:
                    LOGGER.warning(f"PDF file '{path}' is encrypted. Attempting to decrypt.")
                    try:
                        # Handle different PyPDF2 versions
                        if hasattr(reader, 'decrypt'):
                            reader.decrypt('')
                        else:
                            LOGGER.error("PDF decryption not supported in this PyPDF2 version")
                            return None
                    except Exception:
                        LOGGER.error("Decryption failed - password may be required")
                        return None

                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            
            return "\n".join(text_parts) if text_parts else ""
        except PyPDF2.errors.PdfReadError as e:
            LOGGER.error(f"PDF read error for '{path}': {e}")
            return None
        except Exception as e:
            LOGGER.error(f"PDF processing failed for '{path}': {e}")
            return None

    def _read_docx(self, path: Path) -> Optional[str]:
        """Extract text from DOCX files."""
        if Document is None:
            LOGGER.error("Cannot read .docx file - python-docx module not installed")
            return None
            
        try:
            doc = Document(path)
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except Exception as e:
            LOGGER.error(f"Error reading DOCX file '{path}': {e}")
            return None

    def write_file(self, relative_path: str, content: str) -> bool:
        """
        Writes content to a text-based file using an atomic operation.
        Args:
            relative_path (str): The path for the file to be written.
            content (str): The string content to write to the file.
        Returns:
            bool: True on success, False on failure.
        """
        path = (self.base_path / relative_path).resolve()
        
        if not path.is_relative_to(self.base_path):
            LOGGER.error(f"Write error: Attempted to write outside base path: {relative_path}")
            return False

        try:
            if path.suffix not in self.supported_write_extensions:
                LOGGER.error(f"Write error: '{path.suffix}' is not a supported text format.")
                return False

            content_size = len(content.encode('utf-8'))
            if content_size > 10 * 1024 * 1024:  # 10MB
                LOGGER.error(f"Content too large: {content_size/1024/1024:.2f}MB")
                return False
            
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)

            if path.suffix == '.py':
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    LOGGER.error(f"Invalid Python syntax: {str(e)}")
                    if temp_path.exists():
                        os.remove(temp_path)
                    return False

            os.replace(temp_path, path)
            LOGGER.info(f"File written: {relative_path}")
            return True
        except OSError as e:
            LOGGER.error(f"Filesystem error: {str(e)}")
            return False
        except Exception as e:
            LOGGER.error(f"Unexpected write error: {str(e)}")
            return False
        finally:
            # Cleanup temp file if it exists
            if 'temp_path' in locals() and temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def delete_file(self, relative_path: str) -> bool:
        """
        Deletes a file, creating a backup first.
        Args:
            relative_path (str): The path of the file to delete.
        Returns:
            bool: True on success, False if the file doesn't exist or on error.
        """
        path = (self.base_path / relative_path).resolve()
        
        if not self.file_exists(relative_path):
            LOGGER.warning(f"Delete failed - file not found: {relative_path}")
            return False

        try:
            backup_dir = self.base_path / "_deleted_backups"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            sanitized_name = self.sanitize_filename(path.name)
            backup_path = backup_dir / f"{sanitized_name}_{timestamp}"
            
            shutil.copy2(path, backup_path)  # copy2 preserves metadata
            path.unlink()
            
            LOGGER.info(f"Deleted: {relative_path} (Backup: {backup_path.name})")
            return True
        except OSError as e:
            LOGGER.error(f"Delete operation failed: {str(e)}")
            return False
        except Exception as e:
            LOGGER.error(f"Unexpected delete error: {str(e)}")
            return False

    def archive_directory(self, source_relative: str, target_relative: str) -> Optional[Path]:
        """
        Create ZIP archive of directory
        Args:
            source_relative (str): Relative path of directory to archive
            target_relative (str): Relative path where to save the ZIP
        Returns:
            Optional[Path]: Path to created ZIP file or None on failure
        """
        source_dir = (self.base_path / source_relative).resolve()
        target_dir = (self.base_path / target_relative).resolve()

        # Security checks
        if not source_dir.is_relative_to(self.base_path):
            LOGGER.error("Source directory outside base path")
            return None
        if not target_dir.is_relative_to(self.base_path):
            LOGGER.error("Target directory outside base path")
            return None
        if not source_dir.is_dir():
            LOGGER.error(f"Source not directory: {source_relative}")
            return None

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_name = f"archive_{source_dir.name}_{timestamp}.zip"
            zip_path = target_dir / zip_name

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for item in source_dir.rglob('*'):
                    if item.is_file() and "_deleted_backups" not in item.parts:
                        arcname = item.relative_to(source_dir)
                        zipf.write(item, arcname)
            
            LOGGER.info(f"Archived {source_relative} to {zip_path.name}")
            return zip_path
        except OSError as e:
            LOGGER.error(f"Archive failed: {str(e)}")
            return None
        except Exception as e:
            LOGGER.error(f"Unexpected archive error: {str(e)}")
            return None

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Create filesystem-safe filename"""
        # Extract final filename component
        name = Path(filename).name
        
        # Remove problematic characters
        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', name)
        
        # Trim spaces and handle empty results
        safe_name = safe_name.strip()
        safe_name = safe_name or "untitled"
        
        # Length validation (max 255 bytes)
        if len(safe_name.encode('utf-8')) > 255:
            root, ext = os.path.splitext(safe_name)
            max_root_length = 255 - len(ext.encode('utf-8')) - 1
            safe_name = root[:max_root_length] + ext
        
        return safe_name
