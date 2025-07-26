import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import PyPDF2
from docx import Document

class FileHandler:
    """Handles file operations across AKS repository and user directories."""
    def __init__(self, repo_path: Path):
        """
        Initialize the FileHandler with repository path.
        
        Args:
            repo_path: Path to the main repository directory
        """
        self.repo_path = repo_path
        self.content_root = Path("/content")
        self.logger = logging.getLogger("aks")
        self.logger.info("File Handler initialized")

    def resolve_path(self, path_str: str) -> Path:
        """Converts any path to absolute Path object"""
        path = Path(path_str)
        if not path.is_absolute():
            return self.content_root / path
        return path

    def is_valid_path(self, path: Path) -> bool:
        """Security check for allowed paths"""
        try:
            # Prevent path traversal attacks
            path.resolve().relative_to(self.content_root)
            return True
        except (ValueError, RuntimeError):
            self.logger.warning(f"Blocked access outside content dir: {path}")
            return False

    def get_file_content(self, path_str: str) -> Optional[str]:
        """Gets content from various file types with extraction"""
        path = self.resolve_path(path_str)
        if not self.is_valid_path(path) or not path.is_file():
            return None

        # Handle different file types
        ext = path.suffix.lower()
        try:
            if ext in ['.txt', '.py', '.json', '.md', '.html', '.csv']:
                with open(path, 'r', encoding='utf-8') as f:
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
        """Extracts text from PDF files"""
        try:
            text = []
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())
            return "\n".join(text)
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            return None

    def _extract_docx_text(self, path: Path) -> Optional[str]:
        """Extracts text from DOCX files"""
        try:
            doc = Document(path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            self.logger.error(f"DOCX extraction failed: {e}")
            return None

    def copy_to_repo(self, source_path_str: str, repo_subpath: str = "") -> Optional[Path]:
        """Copies user files into repository workspace"""
        source = self.resolve_path(source_path_str)
        dest = self.repo_path / repo_subpath / source.name

        if not self.is_valid_path(source) or not source.is_file():
            return None

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, dest)
            self.logger.info(f"Copied {source} to repository: {dest}")
            return dest
        except Exception as e:
            self.logger.error(f"File copy failed: {source} -> {dest} - {e}")
            return None

    def archive_directory(self, source_dir: Path, dest_dir: Path) -> bool:
        """
        Archive a directory to specified destination.
        
        Args:
            source_dir: Directory to archive
            dest_dir: Destination directory for archive
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_valid_path(source_dir) or not source_dir.is_dir():
                return False

            dest_dir.mkdir(parents=True, exist_ok=True)
            archive_name = f"{source_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
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
        Check if a file exists at the given path.
        
        Args:
            path_str: Path to check (relative or absolute)
            
        Returns:
            True if file exists, False otherwise
        """
        path = self.resolve_path(path_str)
        return path.is_file() and self.is_valid_path(path)

    def write_file(self, path_str: str, content: str) -> bool:
        """
        Write content to a file with atomic write operation.
        
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
            # Atomic write operation
            temp_path = path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            temp_path.replace(path)
            self.logger.debug(f"Wrote content to {path}")
            return True
        except Exception as e:
            self.logger.error(f"File write failed: {path} - {e}")
            return False
