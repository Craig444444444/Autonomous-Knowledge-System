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

class FileHandler:
    """Enhanced file operations handler with security, reliability, and extended functionality."""
    def __init__(self, repo_path: Path):
        """
        Initialize the FileHandler with repository path.
        
        Args:
            repo_path: Path to the main repository directory
        """
        self.repo_path = repo_path.resolve()
        self.content_root = Path("/content")
        self.logger = logging.getLogger("aks")
        self.logger.info("File Handler initialized")

    def resolve_path(self, path_str: str) -> Path:
        """Converts any path to absolute Path object with security checks"""
        try:
            path = Path(path_str)
            if path.is_absolute():
                return path.resolve()
            return (self.content_root / path).resolve()
        except Exception as e:
            self.logger.error(f"Path resolution failed: {path_str} - {e}")
            return Path("/invalid_path")

    def is_valid_path(self, path: Path) -> bool:
        """Security check for allowed paths with comprehensive validation"""
        try:
            # Normalize and resolve path
            resolved_path = path.resolve()
            
            # Check against content root
            if self.content_root not in resolved_path.parents and resolved_path != self.content_root:
                self.logger.warning(f"Blocked access outside content dir: {resolved_path}")
                return False
                
            # Check for dangerous paths
            if any(part.startswith(('.', '__')) and part not in ['.git', '.github']:
                self.logger.warning(f"Blocked access to hidden system path: {resolved_path}")
                return False
                
            return True
        except (ValueError, RuntimeError, OSError) as e:
            self.logger.error(f"Path validation error: {path} - {e}")
            return False

    def get_file_content(self, path_str: str, max_size: int = 1024 * 1024 * 10) -> Optional[str]:
        """Gets content from various file types with extraction and size limits"""
        path = self.resolve_path(path_str)
        if not self.is_valid_path(path) or not path.is_file():
            return None

        # Check file size
        try:
            file_size = path.stat().st_size
            if file_size > max_size:
                self.logger.warning(f"File too large ({file_size} bytes): {path}")
                return None
        except OSError as e:
            self.logger.error(f"File size check failed: {path} - {e}")
            return None

        # Handle different file types
        ext = path.suffix.lower()
        try:
            if ext in ['.txt', '.py', '.json', '.md', '.html', '.css', '.js', '.csv', '.yaml', '.yml']:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            elif ext == '.pdf':
                return self._extract_pdf_text(path)
            elif ext in ['.docx', '.doc']:
                return self._extract_docx_text(path)
            else:
                self.logger.warning(f"Unsupported file type: {ext}")
                return None
        except Exception as e:
            self.logger.error(f"Content extraction failed: {path} - {e}")
            return None

    def _extract_pdf_text(self, path: Path) -> Optional[str]:
        """Extracts text from PDF files with error handling"""
        try:
            text = []
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return "\n".join(text) if text else None
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            return None

    def _extract_docx_text(self, path: Path) -> Optional[str]:
        """Extracts text from DOCX files with fallback"""
        if Document is None:
            self.logger.warning("docx module not available - cannot extract DOCX content")
            return None
            
        try:
            doc = Document(path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            return "\n".join(full_text) if full_text else None
        except Exception as e:
            self.logger.error(f"DOCX extraction failed: {e}")
            return None

    def copy_to_repo(self, source_path_str: str, repo_subpath: str = "") -> Optional[Path]:
        """Copies user files into repository workspace with validation"""
        source = self.resolve_path(source_path_str)
        dest = (self.repo_path / repo_subpath).resolve() / source.name

        # Validate paths
        if not self.is_valid_path(source) or not source.is_file():
            return None
        if not self.is_valid_path(dest):
            return None

        try:
            # Ensure destination directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy with metadata preservation
            shutil.copy2(source, dest)
            self.logger.info(f"Copied {source} to repository: {dest}")
            return dest
        except Exception as e:
            self.logger.error(f"File copy failed: {source} -> {dest} - {e}")
            return None

    def archive_directory(self, source_dir_str: str, dest_dir_str: str) -> bool:
        """
        Archive a directory to specified destination with validation.
        
        Args:
            source_dir_str: Directory to archive (relative or absolute)
            dest_dir_str: Destination directory for archive (relative or absolute)
            
        Returns:
            True if successful, False otherwise
        """
        source_dir = self.resolve_path(source_dir_str)
        dest_dir = self.resolve_path(dest_dir_str)
        
        if not self.is_valid_path(source_dir) or not source_dir.is_dir():
            return False
        if not self.is_valid_path(dest_dir):
            return False

        try:
            # Create destination directory
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Create archive name with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"{source_dir.name}_{timestamp}.zip"
            archive_path = dest_dir / archive_name

            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(source_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(source_dir)
                        zipf.write(file_path, arcname)

            self.logger.info(f"Archived {source_dir} to {archive_path}")
            return True
        except Exception as e:
            self.logger.error(f"Directory archiving failed: {e}")
            return False

    def file_exists(self, path_str: str) -> bool:
        """
        Check if a file exists at the given path with validation.
        
        Args:
            path_str: Path to check (relative or absolute)
            
        Returns:
            True if file exists and is valid, False otherwise
        """
        path = self.resolve_path(path_str)
        return path.is_file() and self.is_valid_path(path)

    def write_file(self, path_str: str, content: str) -> bool:
        """
        Write content to a file with atomic operation and directory creation.
        
        Args:
            path_str: Path to file (relative or absolute)
            content: Content to write
            
        Returns:
            True if successful, False otherwise
        """
        path = self.resolve_path(path_str)
        if not self.is_valid_path(path):
            return False

        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write operation
            temp_path = path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            os.replace(temp_path, path)  # Atomic replace
            
            self.logger.debug(f"Wrote {len(content)} characters to {path}")
            return True
        except Exception as e:
            self.logger.error(f"File write failed: {path} - {e}")
            return False

    def create_directory(self, path_str: str) -> bool:
        """
        Create a directory with parent directories if needed.
        
        Args:
            path_str: Path to directory (relative or absolute)
            
        Returns:
            True if successful, False otherwise
        """
        path = self.resolve_path(path_str)
        if not self.is_valid_path(path):
            return False

        try:
            path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {path}")
            return True
        except Exception as e:
            self.logger.error(f"Directory creation failed: {path} - {e}")
            return False

    def list_directory(self, path_str: str, recursive: bool = False) -> List[Path]:
        """
        List contents of a directory with validation.
        
        Args:
            path_str: Path to directory (relative or absolute)
            recursive: Return all contents recursively if True
            
        Returns:
            List of Path objects or empty list on failure
        """
        path = self.resolve_path(path_str)
        if not self.is_valid_path(path) or not path.is_dir():
            return []

        try:
            if recursive:
                return [p for p in path.rglob('*') if p.is_file() and self.is_valid_path(p)]
            return [p for p in path.iterdir() if self.is_valid_path(p)]
        except Exception as e:
            self.logger.error(f"Directory listing failed: {path} - {e}")
            return []

    def delete_file(self, path_str: str) -> bool:
        """
        Delete a file with validation.
        
        Args:
            path_str: Path to file (relative or absolute)
            
        Returns:
            True if successful, False otherwise
        """
        path = self.resolve_path(path_str)
        if not self.is_valid_path(path) or not path.is_file():
            return False

        try:
            path.unlink()
            self.logger.info(f"Deleted file: {path}")
            return True
        except Exception as e:
            self.logger.error(f"File deletion failed: {path} - {e}")
            return False
