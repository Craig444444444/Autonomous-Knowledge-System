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

# OPTION 1: Make python-docx a strict requirement.
# If 'python-docx' is not installed, this line will raise an ImportError
# immediately when the script starts. Ensure it's in your project's
# requirements (e.g., requirements.txt) and installed via pip.
try:
    from docx import Document
except ImportError:
    # If Document is strictly required, raise an error or log critically and exit.
    # For this option, we'll re-raise, as it's a "strict requirement".
    logging.critical("CRITICAL ERROR: 'python-docx' module is required but not installed. Please install it with 'pip install python-docx'.")
    raise # Re-raise the ImportError to stop execution if the dependency is missing.


# It's good practice to get a logger specific to your module/application.
LOGGER = logging.getLogger("aks")
# Configure basic logging if not already configured by the main application
# This is a fallback for testing or standalone use. In a larger application,
# logging configuration should typically happen once at the application's entry point.
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
        self.base_path = Path(base_path).resolve() # Resolve to absolute path for robustness
        
        if not self.base_path.is_dir():
            try:
                self.base_path.mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"Base path '{self.base_path}' created successfully.")
            except OSError as e:
                LOGGER.critical(f"Failed to create base path '{self.base_path}': {e}")
                raise ValueError(f"Base path '{self.base_path}' is not a valid directory and could not be created.") from e

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
        # Use resolve() to handle '..' and '.' in paths safely
        absolute_path = (self.base_path / relative_path).resolve()
        # Ensure the resolved path is still within the base_path to prevent path traversal issues
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
        
        if not self.file_exists(relative_path): # file_exists already checks path validity
            LOGGER.warning(f"File not found or inaccessible: {path}")
            return None

        # Check if the extension is supported for reading
        if path.suffix not in self.supported_read_extensions:
            LOGGER.error(f"Unsupported file type for reading: '{path.suffix}' in file '{relative_path}'")
            return None

        try:
            # Handle different file formats based on their extension.
            if path.suffix == '.pdf':
                return self._read_pdf(path)
            elif path.suffix == '.docx':
                return self._read_docx(path)
            else:
                # For plain text files (including .json), try multiple common encodings.
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']: # Added cp1252 as a common fallback
                    try:
                        with open(path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue # Try next encoding
                    except Exception as e:
                        # Catch other potential errors during open/read, e.g., permissions
                        LOGGER.error(f"Error opening/reading '{relative_path}' with encoding '{encoding}': {e}", exc_info=True)
                        return None # Fail early if file cannot be opened
                LOGGER.error(f"Failed to decode '{relative_path}' with any common encoding.")
                return None
        except Exception as e:
            LOGGER.error(f"Unexpected error during read_file operation for '{relative_path}': {e}", exc_info=True)
            return None

    def _read_pdf(self, path: Path) -> Optional[str]:
        """Helper method to extract text from PDF files."""
        try:
            text_parts = []
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                if reader.is_encrypted:
                    LOGGER.warning(f"PDF file '{path}' is encrypted. Attempting to decrypt (password might be needed).")
                    # PyPDF2 >= 3.0.0 uses decrypt(), older versions use decrypt(password)
                    try:
                        reader.decrypt('') # Try with empty password
                    except PyPDF2.utils.PdfReadError: # For older PyPDF2 versions
                        try: # Catch for newer PyPDF2 versions
                            reader.decrypt()
                        except Exception:
                            LOGGER.error(f"Could not decrypt PDF '{path}'. Password might be required.")
                            return None
                    except Exception as e:
                        LOGGER.error(f"Error decrypting PDF '{path}': {e}")
                        return None

                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text_parts.append(extracted_text)
                    else:
                        LOGGER.debug(f"No text extracted from page {page_num + 1} of '{path}'.")
            
            if not text_parts:
                LOGGER.warning(f"No text found in PDF file '{path}'. It might be image-based or empty.")
                return "" # Return empty string if no text is found, rather than None
            
            return "\n".join(text_parts)
        except PyPDF2.errors.PdfReadError as e:
            LOGGER.error(f"Invalid PDF file or PDF read error for '{path}': {e}", exc_info=True)
            return None
        except Exception as e:
            LOGGER.error(f"PDF extraction failed for '{path}': {e}", exc_info=True)
            return None

    def _read_docx(self, path: Path) -> Optional[str]:
        """Helper method to extract text from DOCX files."""
        # 'Document' is guaranteed to be imported at the top of the file
        # due to the strict dependency approach (Option 1).
        try:
            doc = Document(path)
            # Extracts text from all paragraphs in the document body.
            # Filters out empty paragraphs to get cleaner output.
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
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
        path = (self.base_path / relative_path).resolve()
        
        # Ensure the target path is within the base_path to prevent path traversal.
        if not path.is_relative_to(self.base_path):
            LOGGER.error(f"Write error: Attempted to write outside base path: {relative_path}")
            return False

        try:
            # 1. Validate file extension to prevent writing to binary formats like .pdf
            if path.suffix not in self.supported_write_extensions:
                LOGGER.error(f"Write error: '{path.suffix}' is not a supported text format for writing.")
                return False

            # 2. Validate content size to prevent accidentally writing huge files.
            # Using len(content.encode('utf-8')) for accurate byte size.
            if len(content.encode('utf-8')) > 10 * 1024 * 1024:  # 10MB limit
                LOGGER.error(f"Write error: Content for '{relative_path}' exceeds 10MB size limit.")
                return False
            
            # Create parent directories if they don't exist
            # Using exist_ok=True prevents error if directory already exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # 3. Atomic Write: Write to a temporary file first.
            # Use a more robust temporary filename (e.g., with a UUID or process ID)
            # if multiple processes might write to the same file, but .tmp is fine for most cases.
            temp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # 4. Validate content if it's a Python file.
            if path.suffix == '.py':
                try:
                    ast.parse(content) # Parses the string as an AST, raises SyntaxError if invalid
                except SyntaxError as e:
                    LOGGER.error(f"Invalid Python syntax in '{relative_path}': {e}")
                    # Clean up the invalid temporary file
                    if temp_path.exists():
                        os.remove(temp_path)
                    return False
                except Exception as e: # Catch other AST parsing errors (less common)
                    LOGGER.error(f"Unexpected error during Python syntax validation for '{relative_path}': {e}")
                    if temp_path.exists():
                        os.remove(temp_path)
                    return False

            # 5. If all checks pass, replace the original file with the new one.
            # os.replace is atomic on POSIX systems. On Windows, it tries to be atomic.
            os.replace(temp_path, path)
            LOGGER.info(f"Successfully wrote to '{relative_path}'")
            return True
        except OSError as e: # Catch OS-related errors like permissions, disk full
            LOGGER.error(f"Operating System error writing to '{relative_path}': {e}", exc_info=True)
            return False
        except Exception as e: # Catch any other unexpected errors
            LOGGER.error(f"Unhandled error writing to '{relative_path}': {e}", exc_info=True)
            return False
        finally:
            # Ensure temp file is cleaned up even if an error occurs after os.replace fails
            if 'temp_path' in locals() and temp_path.exists():
                try:
                    os.remove(temp_path)
                except OSError as e:
                    LOGGER.warning(f"Could not remove temporary file '{temp_path}': {e}")


    def delete_file(self, relative_path: str) -> bool:
        """
        Deletes a file, creating a backup first.
        Args:
            relative_path (str): The path of the file to delete.
        Returns:
            bool: True on success, False if the file doesn't exist or on error.
        """
        path = (self.base_path / relative_path).resolve()

        if not self.file_exists(relative_path): # file_exists checks path validity
            LOGGER.warning(f"Delete failed: File '{path}' does not exist or is outside base path.")
            return False

        try:
            # Create a dedicated directory for backups.
            backup_dir = self.base_path / "_deleted_backups"
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Add microseconds for higher uniqueness
            
            # Sanitize the filename to be cross-platform safe for the backup copy.
            # Using Path.name for the filename part and then sanitize.
            original_filename = path.name
            sanitized_name = self.sanitize_filename(original_filename)
            
            backup_path = backup_dir / f"{sanitized_name}_{timestamp}"
            
            shutil.copy(path, backup_path)

            # Delete the original file.
            path.unlink() # Use unlink() from Path for file deletion
            LOGGER.info(f"Deleted '{relative_path}' (backup saved to '{backup_path}')")
            return True
        except OSError as e: # Catch OS-related errors like permissions
            LOGGER.error(f"Operating System error deleting '{relative_path}': {e}", exc_info=True)
            return False
        except Exception as e:
            LOGGER.error(f"Unhandled error deleting '{relative_path}': {e}", exc_info=True)
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
        source_dir = (self.base_path / source_relative_path).resolve()
        target_dir = (self.base_path / target_relative_path).resolve()

        # Ensure source directory is within base_path
        if not source_dir.is_relative_to(self.base_path):
            LOGGER.error(f"Archive failed: Source directory '{source_relative_path}' is outside base path.")
            return None
        if not target_dir.is_relative_to(self.base_path):
            LOGGER.error(f"Archive failed: Target directory '{target_relative_path}' is outside base path.")
            return None

        if not source_dir.is_dir():
            LOGGER.error(f"Archive failed: Source '{source_dir}' is not a valid directory or does not exist.")
            return None

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Ensure the zip file name is descriptive and safe
            zip_filename = f"archive_{source_dir.name}_{timestamp}.zip"
            zip_path = target_dir / zip_filename

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Use pathlib's rglob for a clean, recursive file search.
                # filter out the backup directory to avoid archiving backups of backups etc.
                for file_path in source_dir.rglob('*'):
                    if file_path.is_file():
                        # Exclude files within the backup directory
                        if "_deleted_backups" in file_path.parts: # Using parts to check for directory name
                            LOGGER.debug(f"Excluding backup file from archive: {file_path}")
                            continue
                        zipf.write(file_path, file_path.relative_to(source_dir))
                    elif file_path.is_dir():
                        # Exclude the backup directory itself from being added as an empty dir
                        if file_path.name == "_deleted_backups":
                            LOGGER.debug(f"Excluding backup directory from archive: {file_path}")
                            continue

            LOGGER.info(f"Successfully archived '{source_dir}' to '{zip_path}'")
            return zip_path
        except OSError as e:
            LOGGER.error(f"Operating System error during archiving '{source_dir}': {e}", exc_info=True)
            return None
        except Exception as e:
            LOGGER.error(f"Unhandled error during archiving '{source_dir}': {e}", exc_info=True)
            return None

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitizes a filename to remove dangerous characters and path information.
        This function should only return the filename itself, not any path components.
        Args:
            filename (str): The original filename (can include path for robustness, but only filename is used).
        Returns:
            str: A sanitized, safe filename suitable for file systems.
        """
        # 1. Extract only the base filename to prevent path traversal issues.
        # This handles cases like 'dir/../file.txt' or '\..\file.txt'
        clean_name = Path(filename).name # Path.name safely extracts the filename part

        # 2. Replace characters that are illegal or problematic in Windows/Linux filesystems.
        # This regex removes characters often reserved or used for special purposes in file paths.
        clean_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', clean_name)
        
        # 3. Remove leading/trailing spaces, which can be problematic on some filesystems.
        clean_name = clean_name.strip()

        # 4. Handle empty string if sanitization results in one.
        if not clean_name:
            clean_name = "untitled" # Or raise an error, depending on desired strictness

        # 5. Limit the length of the filename to a reasonable maximum (e.g., 255 chars is common).
        # This prevents "filename too long" errors on various file systems.
        MAX_FILENAME_LENGTH = 250 # Leaving some room for extensions or timestamps
        if len(clean_name) > MAX_FILENAME_LENGTH:
            # Preserve extension if present
            name_part, ext_part = os.path.splitext(clean_name)
            if len(name_part) > MAX_FILENAME_LENGTH - len(ext_part):
                name_part = name_part[:MAX_FILENAME_LENGTH - len(ext_part)]
            clean_name = name_part + ext_part

        return clean_name
