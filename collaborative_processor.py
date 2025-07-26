import logging
import zipfile
import shutil
import os
import re
import tempfile
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
import mimetypes
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

LOGGER = logging.getLogger("aks")

class CollaborativeProcessor:
    """
    Enhanced collaborative processor for handling user-AI knowledge enhancement with
    improved security, performance, and file processing capabilities.
    """
    def __init__(self, knowledge_processor: Any, user_feedback_dir: Path, temp_dir: Path):
        """
        Initialize the CollaborativeProcessor with enhanced capabilities.

        Args:
            knowledge_processor: Instance of KnowledgeProcessor for content ingestion
            user_feedback_dir: Directory to monitor for user feedback files
            temp_dir: Temporary directory for processing files (will be secured)
        """
        self.knowledge_processor = knowledge_processor
        self.user_feedback_dir = user_feedback_dir.resolve()
        self.temp_dir = temp_dir.resolve()
        self._setup_directories()
        self._processed_hashes = set()  # Track processed files to prevent duplicates
        self.max_workers = 4  # For parallel processing
        self.max_file_size = 10 * 1024 * 1024  # 10MB file size limit
        LOGGER.info("Initialized CollaborativeProcessor with enhanced features")

    def _setup_directories(self):
        """Secure directory setup with proper permissions."""
        try:
            self.user_feedback_dir.mkdir(parents=True, exist_ok=True)
            self.user_feedback_dir.chmod(0o755)

            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir.chmod(0o700)  # More restrictive for temp files

            LOGGER.debug("Verified/Created required directories")
        except Exception as e:
            LOGGER.error(f"Directory setup failed: {e}")
            raise RuntimeError("Could not initialize directories") from e

    def process_feedback(self, ai_generator: Any, archive_dir: Path) -> Dict[str, Any]:
        """
        Process all pending feedback files with enhanced security and parallel processing.

        Args:
            ai_generator: AIGenerator instance for content enhancement
            archive_dir: Secure directory for archiving processed files

        Returns:
            Detailed processing statistics and error reports
        """
        archive_dir = archive_dir.resolve()
        archive_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Starting enhanced feedback processing")
        feedback_files = self._get_pending_files()

        if not feedback_files:
            LOGGER.info("No new feedback files found")
            return {
                "status": "success",
                "processed": 0,
                "warnings": ["No files to process"]
            }

        total_results = {
            "files_processed": 0,
            "archives_processed": 0,
            "files_extracted": 0,
            "content_enhanced": 0,
            "knowledge_added": 0,
            "errors": [],
            "start_time": datetime.now().isoformat()
        }

        # Process files in parallel with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_file,
                    file_path,
                    ai_generator,
                    archive_dir
                ): file_path for file_path in feedback_files
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    self._update_results(total_results, result)
                except Exception as e:
                    error_msg = f"Failed to process {file_path.name}: {str(e)}"
                    LOGGER.error(error_msg, exc_info=True)
                    total_results["errors"].append(error_msg)

        total_results["end_time"] = datetime.now().isoformat()
        total_results["status"] = "completed_with_errors" if total_results["errors"] else "success"

        LOGGER.info(
            f"Processing complete. Results: {json.dumps(total_results, indent=2)}"
        )
        return total_results

    def _get_pending_files(self) -> List[Path]:
        """Get pending files with security checks and duplicate prevention."""
        valid_files = []

        for file_path in self.user_feedback_dir.iterdir():
            if not file_path.is_file():
                continue

            # Check file size
            try:
                if file_path.stat().st_size > self.max_file_size:
                    LOGGER.warning(f"Skipping large file: {file_path.name}")
                    continue
            except OSError as e:
                LOGGER.warning(f"Could not check size of {file_path.name}: {e}")
                continue

            # Check file hash for duplicates
            file_hash = self._get_file_hash(file_path)
            if file_hash in self._processed_hashes:
                LOGGER.info(f"Skipping already processed file: {file_path.name}")
                continue

            valid_files.append(file_path)
            self._processed_hashes.add(file_hash)

        return valid_files

    def _get_file_hash(self, file_path: Path) -> str:
        """Get secure hash of file contents."""
        hasher = hashlib.sha256()
        try:
            with file_path.open('rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            LOGGER.error(f"Could not hash file {file_path.name}: {e}")
            return ""

    def _process_single_file(self, file_path: Path, ai_generator: Any, archive_dir: Path) -> Dict[str, Any]:
        """Process a single feedback file with enhanced security."""
        result = {
            "file_name": file_path.name,
            "file_type": mimetypes.guess_type(file_path.name)[0] or "unknown",
            "processed": False,
            "extracted": 0,
            "enhanced": 0,
            "ingested": 0,
            "errors": []
        }

        try:
            # Handle different file types
            if file_path.suffix.lower() == '.zip':
                return self._process_archive(file_path, ai_generator, archive_dir)
            elif file_path.suffix.lower() in ['.txt', '.md', '.py', '.json', '.csv']:
                return self._process_text_file(file_path, ai_generator, archive_dir)
            else:
                result["errors"].append(f"Unsupported file type: {file_path.suffix}")
                return result
        except Exception as e:
            result["errors"].append(f"Processing failed: {str(e)}")
            LOGGER.error(f"Error processing {file_path.name}: {e}", exc_info=True)
            return result

    def _process_archive(self, zip_path: Path, ai_generator: Any, archive_dir: Path) -> Dict[str, Any]:
        """Process archive files with enhanced security checks."""
        result = {
            "file_name": zip_path.name,
            "file_type": "archive",
            "processed": True,
            "extracted": 0,
            "enhanced": 0,
            "ingested": 0,
            "errors": []
        }

        # Create secure temp directory
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as extract_dir:
            extract_path = Path(extract_dir)

            try:
                # Validate zip file before extraction
                if not self._validate_zip(zip_path):
                    result["errors"].append("Invalid or malicious zip file")
                    result["processed"] = False
                    return result

                # Extract files
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                    result["extracted"] = len(zip_ref.infolist())

                # Process extracted files
                for extracted_file in extract_path.rglob('*'):
                    if extracted_file.is_file():
                        file_result = self._process_extracted_file(
                            extracted_file,
                            zip_path.name,
                            ai_generator
                        )
                        result["enhanced"] += file_result.get("enhanced", 0)
                        result["ingested"] += file_result.get("ingested", 0)
                        result["errors"].extend(file_result.get("errors", []))

            except zipfile.BadZipFile:
                result["errors"].append("Corrupted zip file")
                result["processed"] = False
            except Exception as e:
                result["errors"].append(f"Archive processing error: {str(e)}")
                result["processed"] = False

        # Archive the processed zip
        if result["processed"]:
            self._archive_file(zip_path, archive_dir)

        return result

    def _validate_zip(self, zip_path: Path) -> bool:
        """Validate zip file for security risks."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Check for zip bombs or malicious paths
                total_size = 0
                for info in zip_ref.infolist():
                    # Prevent path traversal
                    if '..' in info.filename or info.filename.startswith('/'):
                        LOGGER.warning(f"Potentially malicious path in zip: {info.filename}")
                        return False

                    # Check for reasonable uncompressed size
                    total_size += info.file_size
                    if total_size > 100 * 1024 * 1024:  # 100MB limit
                        LOGGER.warning("Zip file exceeds size safety limit")
                        return False
            return True
        except Exception:
            return False

    def _process_text_file(self, file_path: Path, ai_generator: Any, archive_dir: Path) -> Dict[str, Any]:
        """Process individual text-based files."""
        result = {
            "file_name": file_path.name,
            "file_type": file_path.suffix.lower(),
            "processed": True,
            "extracted": 0,
            "enhanced": 0,
            "ingested": 0,
            "errors": []
        }

        try:
            # Read file with encoding fallback
            content = self._read_file_with_fallback(file_path)
            if not content:
                result["errors"].append("Empty or unreadable file")
                result["processed"] = False
                return result

            # Prepare metadata
            metadata = {
                "source": "user_upload",
                "original_file": file_path.name,
                "upload_time": datetime.now().isoformat(),
                "content_length": len(content),
                "content_hash": hashlib.sha256(content.encode()).hexdigest()
            }

            # Ingest original content
            if self.knowledge_processor.ingest_source("text", content, metadata):
                result["ingested"] += 1

            # Enhance content if AI is available
            if ai_generator.ai_manager.has_available_providers():
                enhanced = self._enhance_content(content, file_path.suffix.lower(), ai_generator)
                if enhanced and enhanced != content:
                    metadata["enhanced"] = True
                    if self.knowledge_processor.ingest_source("text", enhanced, metadata):
                        result["ingested"] += 1
                        result["enhanced"] += 1

            # Archive the processed file
            self._archive_file(file_path, archive_dir)

        except Exception as e:
            result["errors"].append(f"Text processing failed: {str(e)}")
            result["processed"] = False
            LOGGER.error(f"Error processing text file {file_path.name}: {e}", exc_info=True)

        return result

    def _process_extracted_file(self, file_path: Path, source_archive: str, ai_generator: Any) -> Dict[str, Any]:
        """Process a single file extracted from an archive."""
        result = {
            "file_name": file_path.name,
            "processed": False,
            "enhanced": 0,
            "ingested": 0,
            "errors": []
        }

        try:
            # Skip non-text files
            if file_path.suffix.lower() not in ['.txt', '.md', '.py', '.json', '.csv']:
                result["errors"].append(f"Unsupported file type: {file_path.suffix}")
                return result

            # Read file content
            content = self._read_file_with_fallback(file_path)
            if not content:
                result["errors"].append("Empty or unreadable file")
                return result

            # Prepare metadata
            metadata = {
                "source": f"user_upload/{source_archive}",
                "original_file": file_path.name,
                "extracted_from": source_archive,
                "processed_at": datetime.now().isoformat(),
                "content_hash": hashlib.sha256(content.encode()).hexdigest()
            }

            # Ingest original content
            if self.knowledge_processor.ingest_source("text", content, metadata):
                result["ingested"] += 1
                result["processed"] = True

            # Enhance content if AI is available
            if ai_generator.ai_manager.has_available_providers():
                enhanced = self._enhance_content(content, file_path.suffix.lower(), ai_generator)
                if enhanced and enhanced != content:
                    metadata["enhanced"] = True
                    if self.knowledge_processor.ingest_source("text", enhanced, metadata):
                        result["ingested"] += 1
                        result["enhanced"] += 1
                        result["processed"] = True

        except Exception as e:
            result["errors"].append(f"File processing failed: {str(e)}")
            LOGGER.error(f"Error processing extracted file {file_path.name}: {e}", exc_info=True)

        return result

    def _read_file_with_fallback(self, file_path: Path) -> Optional[str]:
        """Read file content with encoding fallback."""
        encodings = ['utf-8', 'latin-1', 'utf-16']
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        LOGGER.warning(f"Could not decode file {file_path.name} with any encoding")
        return None

    def _enhance_content(self, content: str, file_type: str, ai_generator: Any) -> Optional[str]:
        """Enhanced content improvement with AI."""
        enhancement_prompts = {
            '.txt': (
                "Improve the clarity, structure and professional tone of this text while "
                "preserving all key information. Add section headers if appropriate:\n\n{content}"
            ),
            '.md': (
                "Enhance this Markdown document by improving formatting, adding "
                "structure and expanding key sections with additional details:\n\n{content}"
            ),
            '.py': (
                "Refactor this Python code to improve readability, add type hints, "
                "include docstrings, and follow PEP 8 guidelines:\n\n{content}"
            ),
            '.json': (
                "Analyze this JSON data and add a '_comment' field explaining the "
                "structure and purpose of the data:\n\n{content}"
            ),
            '.csv': (
                "Analyze this CSV data (first 50 lines shown) and generate a "
                "summary of its structure and contents:\n\n{content[:5000]}"
            )
        }

        prompt_template = enhancement_prompts.get(file_type.lower())
        if not prompt_template:
            return None

        system_prompt = (
            "You are a professional content enhancement assistant. "
            "Provide only the enhanced content without additional commentary."
        )

        try:
            if file_type.lower() == '.py':
                return ai_generator.ai_manager.generate_code(
                    prompt_template.format(content=content),
                    system_prompt,
                    max_tokens=4000
                )
            else:
                return ai_generator.ai_manager.generate_text(
                    prompt_template.format(content=content),
                    system_prompt,
                    max_tokens=3000
                )
        except Exception as e:
            LOGGER.error(f"AI enhancement failed: {e}")
            return None

    def _archive_file(self, file_path: Path, archive_dir: Path) -> bool:
        """Securely archive processed files with timestamp."""
        try:
            archive_path = archive_dir / (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                f"{file_path.name}"
            )
            shutil.move(str(file_path), str(archive_path))
            LOGGER.debug(f"Archived {file_path.name} to {archive_path}")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to archive {file_path.name}: {e}")
            return False

    def _update_results(self, total: Dict[str, Any], current: Dict[str, Any]) -> None:
        """Helper to aggregate processing results."""
        total["files_processed"] += 1
        if current.get("processed", False):
            total["files_extracted"] += current.get("extracted", 0)
            total["content_enhanced"] += current.get("enhanced", 0)
            total["knowledge_added"] += current.get("ingested", 0)
        total["errors"].extend(current.get("errors", []))
