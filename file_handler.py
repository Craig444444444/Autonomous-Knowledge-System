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

# Try to import python-docx for .docx support
try:
    from docx import Document
except ImportError:
    Document = None
    logging.warning("python-docx module not installed. .docx support disabled.")

# Configure logger
LOGGER = logging.getLogger("aks")
if not LOGGER.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def is_within_base(path: Path, base: Path) -> bool:
    """Check if path is inside base (safe for Python <3.9)."""
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False

class FileHandler:
    """
    FileHandler with secure operations:
    - Read from .txt, .py, .md, .pdf, .docx, .json
    - Write only to .txt, .py, .md, .json
    - Atomic writes, backups, and archiving
    """

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path).resolve()
        if not self.base_path.is_dir():
            try:
                self.base_path.mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"Created base directory: {self.base_path}")
            except OSError as e:
                LOGGER.critical(f"Failed to create base directory '{self.base_path}': {e}")
                raise ValueError(f"Invalid base path: {self.base_path}") from e

        self.supported_read_extensions = ['.txt', '.py', '.md', '.pdf', '.docx', '.json']
        self.supported_write_extensions = ['.txt', '.py', '.md', '.json']
        LOGGER.info(f"FileHandler initialized at {self.base_path}")

    def file_exists(self, relative_path: str) -> bool:
        absolute_path = (self.base_path / relative_path).resolve()
        if not is_within_base(absolute_path, self.base_path):
            LOGGER.warning(f"Access attempt outside base path: {relative_path}")
            return False
        return absolute_path.is_file()

    def read_file(self, relative_path: str) -> Optional[str]:
        path = (self.base_path / relative_path).resolve()

        if not self.file_exists(relative_path):
            LOGGER.warning(f"File not found: {path}")
            return None

        if path.suffix not in self.supported_read_extensions:
            LOGGER.error(f"Unsupported file type for reading: {path.suffix}")
            return None

        try:
            if path.suffix == '.pdf':
                return self._read_pdf(path)
            elif path.suffix == '.docx':
                return self._read_docx(path)
            else:
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        with open(path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
                LOGGER.error(f"Failed to decode file: {relative_path}")
                return None
        except Exception as e:
            LOGGER.error(f"Error reading file '{relative_path}': {e}", exc_info=True)
            return None

    def _read_pdf(self, path: Path) -> Optional[str]:
        try:
            text_parts = []
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)

                if getattr(reader, 'is_encrypted', False):
                    LOGGER.warning(f"PDF is encrypted: {path}. Attempting to decrypt.")
                    try:
                        # Handle different PyPDF2 versions
                        if hasattr(reader, 'decrypt'):
                            reader.decrypt('')
                        else:
                            LOGGER.error("PDF decryption not supported in this PyPDF2 version")
                            return None
                    except Exception as e:
                        LOGGER.error(f"PDF decryption failed: {e}")
                        return None

                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts) if text_parts else ""
        except Exception as e:
            LOGGER.error(f"PDF processing failed for '{path}': {e}", exc_info=True)
            return None

    def _read_docx(self, path: Path) -> Optional[str]:
        if Document is None:
            LOGGER.error("python-docx not installed. Cannot read DOCX.")
            return None

        try:
            doc = Document(path)
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except Exception as e:
            LOGGER.error(f"Error reading DOCX: {e}", exc_info=True)
            return None

    def write_file(self, relative_path: str, content: str) -> bool:
        path = (self.base_path / relative_path).resolve()

        if not is_within_base(path, self.base_path):
            LOGGER.error(f"Write attempt outside base path: {relative_path}")
            return False

        # Initialize temp_path here to ensure it's defined for finally block
        temp_path = None
        try:
            if path.suffix not in self.supported_write_extensions:
                LOGGER.error(f"Unsupported file type for writing: {path.suffix}")
                return False

            content_size = len(content.encode('utf-8'))
            if content_size > 10 * 1024 * 1024:  # 10 MB limit
                LOGGER.error(f"File too large ({content_size/1024/1024:.2f} MB)")
                return False

            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")

            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)

            if path.suffix == '.py':
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    LOGGER.error(f"Python syntax error: {e}")
                    return False

            os.replace(temp_path, path)
            LOGGER.info(f"File written successfully: {relative_path}")
            return True
        except Exception as e:
            LOGGER.error(f"Write error: {e}", exc_info=True)
            return False
        finally:
            # Clean up temp file if it exists
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    LOGGER.debug(f"Could not remove temp file: {e}")

    def delete_file(self, relative_path: str) -> bool:
        path = (self.base_path / relative_path).resolve()

        if not self.file_exists(relative_path):
            LOGGER.warning(f"Delete failed - file not found: {relative_path}")
            return False

        try:
            backup_dir = self.base_path / "_deleted_backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_name = self.sanitize_filename(path.name)
            backup_path = backup_dir / f"{safe_name}_{timestamp}"

            shutil.copy2(path, backup_path)
            path.unlink()

            LOGGER.info(f"Deleted: {relative_path}. Backup saved as {backup_path.name}")
            return True
        except Exception as e:
            LOGGER.error(f"Delete error: {e}", exc_info=True)
            return False

    def archive_directory(self, source_relative: str, target_relative: str) -> Optional[Path]:
        source_dir = (self.base_path / source_relative).resolve()
        target_dir = (self.base_path / target_relative).resolve()

        if not is_within_base(source_dir, self.base_path) or not is_within_base(target_dir, self.base_path):
            LOGGER.error("Archive paths outside base directory.")
            return None
        if not source_dir.is_dir():
            LOGGER.error(f"Source is not a directory: {source_relative}")
            return None

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_path = target_dir / f"archive_{source_dir.name}_{timestamp}.zip"

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for item in source_dir.rglob('*'):
                    if item.is_file() and "_deleted_backups" not in item.parts:
                        # Use relative path for proper archive structure
                        arcname = item.relative_to(source_dir)
                        zipf.write(item, arcname)

            LOGGER.info(f"Directory archived: {zip_path}")
            return zip_path
        except Exception as e:
            LOGGER.error(f"Archive error: {e}", exc_info=True)
            return None

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        # Extract filename and remove unsafe characters
        name = Path(filename).name
        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', name).strip()
        
        # Handle empty filenames
        if not safe_name:
            safe_name = "untitled"
            
        # Check byte length (max 255 bytes for filesystems)
        byte_length = len(safe_name.encode('utf-8'))
        if byte_length > 255:
            # Preserve file extension
            root, ext = os.path.splitext(safe_name)
            max_root_length = 255 - len(ext.encode('utf-8'))
            # Truncate root part to fit within byte limit
            root_bytes = root.encode('utf-8')
            truncated_root = root_bytes[:max_root_length].decode('utf-8', 'ignore')
            safe_name = truncated_root + ext
            
        return safe_name
