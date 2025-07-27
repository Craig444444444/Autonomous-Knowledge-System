import logging
import shutil
import os
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any
import PyPDF2
from datetime import datetime

# Safe docx import with fallback
try:
    from docx import Document
except ImportError:
    Document = None
    logging.warning("docx module not available - DOCX extraction will be limited")

LOGGER = logging.getLogger("aks")


class FileHandler:
    """Enhanced file handler with multi-format support and security features."""
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.supported_extensions = ['.txt', '.py', '.md', '.pdf', '.docx', '.json']
        LOGGER.info(f"FileHandler initialized at {base_path}")

    def file_exists(self, relative_path: str) -> bool:
        return (self.base_path / relative_path).exists()

    def read_file(self, relative_path: str) -> Optional[str]:
        """Read a file with automatic encoding detection and format handling."""
        path = self.base_path / relative_path
        if not path.exists():
            LOGGER.warning(f"File not found: {relative_path}")
            return None

        try:
            # Handle different file formats
            if path.suffix == '.pdf':
                return self._read_pdf(path)
            elif path.suffix == '.docx':
                return self._read_docx(path)
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    return json.dumps(json.load(f))
            else:
                # Try different encodings for text files
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                    try:
                        with open(path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
                LOGGER.error(f"Failed to decode {relative_path} with common encodings")
                return None
        except Exception as e:
            LOGGER.error(f"Error reading {relative_path}: {e}")
            return None

    def _read_pdf(self, path: Path) -> Optional[str]:
        """Extract text from PDF files."""
        try:
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() for page in reader.pages)
                return text
        except Exception as e:
            LOGGER.error(f"PDF extraction failed: {e}")
            return None

    def _read_docx(self, path: Path) -> Optional[str]:
        """Extract text from DOCX files."""
        if Document is None:
            LOGGER.warning("docx module not available, cannot read DOCX files")
            return None
        try:
            doc = Document(path)
            return "\n".join(para.text for para in doc.paragraphs)
        except Exception as e:
            LOGGER.error(f"DOCX extraction failed: {e}")
            return None

    def write_file(self, relative_path: str, content: str) -> bool:
        """Write content to a file with validation and atomic writes."""
        path = self.base_path / relative_path
        try:
            # Validate file extension
            if path.suffix not in self.supported_extensions:
                LOGGER.warning(f"Unsupported file type: {path.suffix}")
                return False

            # Validate content size
            if len(content) > 10 * 1024 * 1024:  # 10MB
                LOGGER.error("File content exceeds size limit (10MB)")
                return False

            # Write to temp file first
            temp_path = path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Validate content (basic check)
            if path.suffix == '.py':
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    LOGGER.error(f"Invalid Python syntax: {e}")
                    os.remove(temp_path)
                    return False

            # Replace original file
            os.replace(temp_path, path)
            LOGGER.info(f"Successfully wrote {relative_path}")
            return True
        except Exception as e:
            LOGGER.error(f"Error writing {relative_path}: {e}")
            if temp_path.exists():
                os.remove(temp_path)
            return False

    def delete_file(self, relative_path: str) -> bool:
        """Securely delete a file with backup."""
        path = self.base_path / relative_path
        if not path.exists():
            return False

        try:
            # Create backup before deletion
            backup_dir = self.base_path / "deleted_backups"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{relative_path.replace('/', '_')}_{timestamp}"
            shutil.copy(path, backup_path)

            # Delete original
            path.unlink()
            LOGGER.info(f"Deleted {relative_path} (backup at {backup_path})")
            return True
        except Exception as e:
            LOGGER.error(f"Error deleting {relative_path}: {e}")
            return False

    def archive_directory(self, source_dir: Path, target_dir: Path) -> bool:
        """Archive a directory to zip file."""
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d")
            zip_path = target_dir / f"archive_{timestamp}.zip"

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(source_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, source_dir)
                        zipf.write(file_path, arcname)

            LOGGER.info(f"Archived {source_dir} to {zip_path}")
            return True
        except Exception as e:
            LOGGER.error(f"Archive failed: {e}")
            return False

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filenames to prevent path traversal and injection."""
        # Remove directory paths
        clean_name = os.path.basename(filename)
        # Remove potentially dangerous characters
        clean_name = re.sub(r'[\\/*?:"<>|]', "", clean_name)
        # Limit length
        return clean_name[:255]
