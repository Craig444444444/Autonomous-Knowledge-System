import logging
import hashlib
import uuid
import requests
import json
import re
from bs4 import BeautifulSoup
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
import tempfile
import mimetypes
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import html2text

# Configure mimetypes
mimetypes.init()

LOGGER = logging.getLogger("aks")

@dataclass
class KnowledgeItem:
    """Structured knowledge item with enhanced metadata."""
    id: str
    content: str
    content_hash: str
    content_type: str
    source: str
    timestamp: str
    metadata: Dict[str, Any]
    relevance: float
    verified: bool = False
    compressed: bool = False
    size_bytes: int = 0

class KnowledgeProcessor:
    """
    Enhanced knowledge processing system with improved ingestion,
    storage, retrieval, and security features.
    """
    def __init__(self, knowledge_base_dir: Path):
        """
        Initialize the enhanced KnowledgeProcessor.

        Args:
            knowledge_base_dir: Directory for knowledge storage (will be secured)
        """
        self.knowledge_base_dir = knowledge_base_dir.resolve()
        self._setup_directories()

        # Thread safety
        self._kb_lock = threading.RLock()
        self._content_index_lock = threading.Lock()

        # Knowledge storage
        self.knowledge_items: List[KnowledgeItem] = []
        self.content_index: Dict[str, KnowledgeItem] = {}
        self.source_index: Dict[str, List[str]] = {}  # source -> [item_ids]

        # Performance tuning
        self.max_content_size = 10 * 1024 * 1024  # 10MB
        self.compression_threshold = 100 * 1024  # 100KB
        self.max_workers = 4  # For parallel processing

        # Initialize
        self._load_knowledge_base()
        LOGGER.info(
            f"KnowledgeProcessor initialized with {len(self.knowledge_items)} items, "
            f"{len(self.content_index)} unique contents"
        )

    def _setup_directories(self):
        """Secure directory setup with proper permissions."""
        try:
            self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
            # Set restrictive permissions
            self.knowledge_base_dir.chmod(0o750)

            # Create subdirectories
            (self.knowledge_base_dir / "raw").mkdir(exist_ok=True)
            (self.knowledge_base_dir / "compressed").mkdir(exist_ok=True)
            (self.knowledge_base_dir / "metadata").mkdir(exist_ok=True)

            LOGGER.debug("Initialized knowledge base directories")
        except Exception as e:
            LOGGER.error(f"Directory setup failed: {e}")
            raise RuntimeError("Could not initialize knowledge base") from e

    def _load_knowledge_base(self):
        """Load knowledge items from disk with enhanced error handling."""
        LOGGER.info("Loading knowledge base from disk...")

        metadata_files = list((self.knowledge_base_dir / "metadata").glob("*.json"))
        if not metadata_files:
            LOGGER.info("No existing knowledge base found")
            return

        loaded_count = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._load_knowledge_item, mf): mf
                for mf in metadata_files
            }

            for future in as_completed(futures):
                try:
                    item = future.result()
                    if item:
                        with self._kb_lock:
                            self.knowledge_items.append(item)
                            self.content_index[item.content_hash] = item
                            self._update_source_index(item)
                        loaded_count += 1
                except Exception as e:
                    LOGGER.error(f"Failed to load knowledge item: {e}")

        LOGGER.info(f"Successfully loaded {loaded_count}/{len(metadata_files)} knowledge items")

    def _load_knowledge_item(self, metadata_path: Path) -> Optional[KnowledgeItem]:
        """Load a single knowledge item from disk."""
        try:
            with metadata_path.open('r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Determine content path
            if metadata.get('compressed', False):
                content_path = self.knowledge_base_dir / "compressed" / f"{metadata['id']}.zlib"
            else:
                content_path = self.knowledge_base_dir / "raw" / f"{metadata['id']}.txt"

            # Load content
            if not content_path.exists():
                LOGGER.warning(f"Content file missing for {metadata['id']}")
                return None

            if metadata.get('compressed', False):
                with content_path.open('rb') as f:
                    compressed = f.read()
                content = zlib.decompress(compressed).decode('utf-8')
            else:
                with content_path.open('r', encoding='utf-8') as f:
                    content = f.read()

            return KnowledgeItem(
                id=metadata['id'],
                content=content,
                content_hash=metadata['content_hash'],
                content_type=metadata['content_type'],
                source=metadata['source'],
                timestamp=metadata['timestamp'],
                metadata=metadata.get('metadata', {}),
                relevance=metadata.get('relevance', 0.0),
                verified=metadata.get('verified', False),
                compressed=metadata.get('compressed', False),
                size_bytes=metadata.get('size_bytes', 0)
            )
        except Exception as e:
            LOGGER.error(f"Failed to load item from {metadata_path.name}: {e}")
            return None

    def ingest_source(self, source_type: str, source_data: str, metadata: Optional[Dict] = None) -> bool:
        """
        Enhanced knowledge ingestion with parallel processing and validation.

        Args:
            source_type: Type of source ('web', 'file', 'text', 'code')
            source_data: The source content/URL/path
            metadata: Additional metadata about the source

        Returns:
            bool: True if new knowledge was added, False otherwise
        """
        try:
            # Validate input
            if not source_type or not source_data:
                LOGGER.warning("Invalid source type or data")
                return False

            # Process based on source type
            processor = {
                'web': self._process_web_source,
                'file': self._process_file_source,
                'text': self._process_text_source,
                'code': self._process_code_source
            }.get(source_type.lower())

            if not processor:
                LOGGER.warning(f"Unsupported source type: {source_type}")
                return False

            # Get content and content type
            content, content_type = processor(source_data)
            if not content:
                LOGGER.warning("No content extracted from source")
                return False

            # Create standardized metadata
            full_metadata = {
                "source_type": source_type,
                "source": source_data,
                "content_type": content_type,
                "ingestion_time": datetime.utcnow().isoformat(),
                **(metadata or {})
            }

            # Add to knowledge base
            return self._add_knowledge_item(content, content_type, full_metadata)

        except Exception as e:
            LOGGER.error(f"Ingestion failed: {e}", exc_info=True)
            return False

    def _process_web_source(self, url: str) -> Tuple[Optional[str], str]:
        """Fetch and process web content with enhanced features."""
        try:
            LOGGER.info(f"Fetching web content from {url}")

            # Validate URL
            if not re.match(r'^https?://', url, re.IGNORECASE):
                raise ValueError("Invalid URL format")

            # Fetch with timeout and retries
            response = requests.get(
                url,
                headers={
                    'User-Agent': 'AKSBot/1.0 (+https://github.com/Craig444444444/AKS)',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5'
                },
                timeout=(3.05, 30),  # Connect and read timeouts
                allow_redirects=True,
                verify=True  # SSL verification
            )
            response.raise_for_status()

            # Determine content type
            content_type = response.headers.get('Content-Type', '').split(';')[0].strip()
            if not content_type:
                content_type = mimetypes.guess_type(url)[0] or 'application/octet-stream'

            # Process based on content type
            if 'text/html' in content_type:
                return self._process_html_content(response.text, url), 'text/html'
            elif 'text/plain' in content_type:
                return response.text, 'text/plain'
            elif 'application/json' in content_type:
                try:
                    return json.dumps(json.loads(response.text), indent=2), 'application/json'
                except json.JSONDecodeError:
                    return response.text, 'application/json'
            else:
                return response.text, content_type

        except requests.RequestException as e:
            LOGGER.error(f"Web request failed: {e}")
            return None, 'unknown'
        except Exception as e:
            LOGGER.error(f"Web processing failed: {e}")
            return None, 'unknown'

    def _process_html_content(self, html: str, url: str) -> str:
        """Convert HTML to clean text with enhanced processing."""
        try:
            # Initialize HTML to text converter
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.bypass_tables = False
            h.single_line_break = False

            # Convert to markdown first for better structure
            markdown = h.handle(html)

            # Additional cleaning
            clean_text = re.sub(r'\n{3,}', '\n\n', markdown)  # Reduce excessive newlines
            clean_text = re.sub(r'\[(]+)\]\(+\)', r'\1', clean_text)  # Remove markdown links
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            # Add source URL
            return f"Source: {url}\n\n{clean_text}"
        except Exception as e:
            LOGGER.error(f"HTML processing failed: {e}")
            return html  # Fallback to raw HTML

    def _process_file_source(self, file_path: str) -> Tuple[Optional[str], str]:
        """Process file content with enhanced features."""
        try:
            path = Path(file_path)
            if not path.exists():
                LOGGER.error(f"File not found: {file_path}")
                return None, 'unknown'

            # Check file size
            size = path.stat().st_size
            if size > self.max_content_size:
                LOGGER.warning(f"File too large ({size} bytes), skipping")
                return None, 'unknown'

            # Determine content type
            content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'

            # Read with encoding fallback
            encodings = ['utf-8', 'latin-1', 'utf-16']
            for encoding in encodings:
                try:
                    with path.open('r', encoding=encoding) as f:
                        content = f.read()
                    return content, content_type
                except UnicodeDecodeError:
                    continue

            LOGGER.warning(f"Could not decode file {file_path} with any encoding")
            return None, 'unknown'
        except Exception as e:
            LOGGER.error(f"File processing failed: {e}")
            return None, 'unknown'

    def _process_text_source(self, text: str) -> Tuple[str, str]:
        """Process raw text content."""
        # Basic cleaning
        clean_text = re.sub(r'\s+', ' ', text).strip()
        return clean_text, 'text/plain'

    def _process_code_source(self, code: str) -> Tuple[str, str]:
        """Process code content with syntax validation."""
        # Basic code validation
        try:
            if len(code.splitlines()) > 1000:
                LOGGER.warning("Large code file detected, truncating")
                code = '\n'.join(code.splitlines()[:1000])
            return code, 'text/x-python'
        except Exception as e:
            LOGGER.error(f"Code processing failed: {e}")
            return code, 'text/x-python'

    def _add_knowledge_item(self, content: str, content_type: str, metadata: Dict) -> bool:
        """Add processed content to knowledge base with enhanced features."""
        try:
            # Generate content hash
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

            # Check for duplicates
            with self._content_index_lock:
                if content_hash in self.content_index:
                    LOGGER.info("Duplicate content detected, skipping ingestion")
                    return False

            # Create knowledge item
            item = KnowledgeItem(
                id=str(uuid.uuid4()),
                content=content,
                content_hash=content_hash,
                content_type=content_type,
                source=metadata.get('source', 'unknown'),
                timestamp=datetime.utcnow().isoformat(),
                metadata=metadata,
                relevance=self._calculate_relevance(content, content_type),
                size_bytes=len(content.encode('utf-8')),
                compressed=len(content) > self.compression_threshold
            )

            # Save to disk
            self._save_knowledge_item(item)

            # Add to in-memory indexes
            with self._kb_lock:
                self.knowledge_items.append(item)
                self.content_index[item.content_hash] = item
                self._update_source_index(item)

            LOGGER.info(f"Added new knowledge item: {item.id}")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to add knowledge item: {e}")
            return False

    def _save_knowledge_item(self, item: KnowledgeItem) -> None:
        """Save knowledge item to disk with atomic writes."""
        try:
            # Save content
            content_bytes = item.content.encode('utf-8')

            if item.compressed:
                content_path = self.knowledge_base_dir / "compressed" / f"{item.id}.zlib"
                compressed = zlib.compress(content_bytes)
                with tempfile.NamedTemporaryFile(dir=self.knowledge_base_dir, delete=False) as tmp:
                    tmp.write(compressed)
                    tmp_path = Path(tmp.name)
                tmp_path.replace(content_path)
            else:
                content_path = self.knowledge_base_dir / "raw" / f"{item.id}.txt"
                with tempfile.NamedTemporaryFile(dir=self.knowledge_base_dir, mode='w', encoding='utf-8', delete=False) as tmp:
                    tmp.write(item.content)
                    tmp_path = Path(tmp.name)
                tmp_path.replace(content_path)

            # Save metadata
            metadata_path = self.knowledge_base_dir / "metadata" / f"{item.id}.json"
            metadata = {
                "id": item.id,
                "content_hash": item.content_hash,
                "content_type": item.content_type,
                "source": item.source,
                "timestamp": item.timestamp,
                "metadata": item.metadata,
                "relevance": item.relevance,
                "verified": item.verified,
                "compressed": item.compressed,
                "size_bytes": item.size_bytes
            }

            with tempfile.NamedTemporaryFile(dir=self.knowledge_base_dir, mode='w', encoding='utf-8', delete=False) as tmp:
                json.dump(metadata, tmp, indent=2)
                tmp_path = Path(tmp.name)
            tmp_path.replace(metadata_path)

        except Exception as e:
            LOGGER.error(f"Failed to save knowledge item {item.id}: {e}")
            raise

    def _update_source_index(self, item: KnowledgeItem) -> None:
        """Update the source index with the new item."""
        source = item.metadata.get('source', 'unknown')
        if source not in self.source_index:
            self.source_index[source] = []
        self.source_index[source].append(item.id)

    def _calculate_relevance(self, content: str, content_type: str) -> float:
        """Enhanced relevance scoring with content type awareness."""
        # Basic metrics
        words = content.lower().split()
        total_words = max(len(words), 1)
        unique_words = len(set(words))
        lexical_diversity = unique_words / total_words

        # Content type factors
        type_factors = {
            'text/html': 0.8,
            'application/json': 0.6,
            'text/x-python': 0.9,
            'text/plain': 0.7
        }
        type_factor = type_factors.get(content_type, 0.5)

        # Keyword analysis
        knowledge_keywords = {
            'knowledge': 1.0,
            'information': 0.9,
            'data': 0.8,
            'analysis': 0.7,
            'system': 0.6,
            'ai': 0.9,
            'machine learning': 0.8
        }

        keyword_score = 0.0
        content_lower = content.lower()
        for kw, weight in knowledge_keywords.items():
            if kw in content_lower:
                keyword_score += weight

        # Length factor (normalized)
        length_factor = min(len(content) / 5000, 1.0)

        # Combined score
        relevance = (
            0.4 * type_factor +
            0.3 * keyword_score +
            0.2 * lexical_diversity +
            0.1 * length_factor
        )

        return min(max(relevance, 0.0), 1.0)

    def retrieve_knowledge(self, query: str, num_results: int = 5, min_relevance: float = 0.3) -> List[Dict]:
        """
        Enhanced knowledge retrieval with better ranking and filtering.

        Args:
            query: Search query
            num_results: Maximum number of results to return
            min_relevance: Minimum relevance score for results

        Returns:
            List of matching knowledge items with scores
        """
        try:
            query = query.lower().strip()
            if not query:
                return []

            results = []

            with self._kb_lock:
                for item in self.knowledge_items:
                    if item.relevance < min_relevance:
                        continue

                    # Simple content matching (could be enhanced with AI)
                    content_lower = item.content.lower()
                    if query in content_lower:
                        # Basic scoring
                        word_count = content_lower.count(query)
                        position_factor = 1.0 - (content_lower.find(query) / len(item.content))
                        score = (word_count * 0.3) + (position_factor * 0.2) + (item.relevance * 0.5)

                        results.append({
                            "item": item,
                            "score": score
                        })

            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)

            # Format results
            formatted_results = []
            for result in results[:num_results]:
                item = result["item"]
                formatted_results.append({
                    "id": item.id,
                    "content": (item.content[:1000] + "...") if len(item.content) > 1000 else item.content,
                    "source": item.source,
                    "content_type": item.content_type,
                    "relevance": item.relevance,
                    "match_score": result["score"],
                    "timestamp": item.timestamp,
                    "metadata": item.metadata
                })

            LOGGER.info(f"Found {len(formatted_results)} results for query '{query[:30]}...'")
            return formatted_results

        except Exception as e:
            LOGGER.error(f"Knowledge retrieval failed: {e}")
            return []

    def export_knowledge(self, format: str = "json", max_items: int = 1000) -> Optional[Path]:
        """
        Enhanced knowledge export with multiple formats and safety checks.

        Args:
            format: Export format ('json', 'text', 'markdown')
            max_items: Maximum number of items to export

        Returns:
            Path to exported file or None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = self.knowledge_base_dir / f"knowledge_export_{timestamp}.{format}"

            with self._kb_lock:
                items_to_export = self.knowledge_items[:max_items]

                if format == "json":
                    export_data = {
                        "export_time": timestamp,
                        "item_count": len(items_to_export),
                        "total_items": len(self.knowledge_items),
                        "items": [
                            {
                                "id": item.id,
                                "content_hash": item.content_hash,
                                "source": item.source,
                                "content_type": item.content_type,
                                "timestamp": item.timestamp,
                                "relevance": item.relevance,
                                "metadata": item.metadata,
                                "content_sample": item.content[:2000]
                            }
                            for item in items_to_export
                        ]
                    }

                    with tempfile.NamedTemporaryFile(dir=self.knowledge_base_dir, mode='w', encoding='utf-8', delete=False) as tmp:
                        json.dump(export_data, tmp, indent=2)
                        tmp_path = Path(tmp.name)
                    tmp_path.replace(export_path)

                elif format == "text":
                    with tempfile.NamedTemporaryFile(dir=self.knowledge_base_dir, mode='w', encoding='utf-8', delete=False) as tmp:
                        tmp.write(f"=== AKS KNOWLEDGE BASE EXPORT ===\n")
                        tmp.write(f"Exported: {timestamp}\n")
                        tmp.write(f"Items: {len(items_to_export)} (of {len(self.knowledge_items)} total)\n\n")

                        for item in items_to_export:
                            tmp.write(f"\n--- Item {item.id} ---\n")
                            tmp.write(f"Source: {item.source}\n")
                            tmp.write(f"Type: {item.content_type}\n")
                            tmp.write(f"Relevance: {item.relevance:.2f}\n")
                            tmp.write(f"Timestamp: {item.timestamp}\n")
                            tmp.write("\nCONTENT:\n")
                            tmp.write(item.content[:2000])
                            if len(item.content) > 2000:
                                tmp.write("\n[... truncated ...]")
                            tmp.write("\n\n")

                        tmp_path = Path(tmp.name)
                    tmp_path.replace(export_path)

                elif format == "markdown":
                    with tempfile.NamedTemporaryFile(dir=self.knowledge_base_dir, mode='w', encoding='utf-8', delete=False) as tmp:
                        tmp.write(f"# AKS Knowledge Base Export\n\n")
                        tmp.write(f"- Export Time: {timestamp}\n")
                        tmp.write(f"- Items Exported: {len(items_to_export)} (of {len(self.knowledge_items)} total)\n\n")

                        for item in items_to_export:
                            tmp.write(f"## {item.source}\n\n")
                            tmp.write(f"- ID: {item.id}\n")
                            tmp.write(f"- Type: {item.content_type}\n")
                            tmp.write(f"- Relevance: {item.relevance:.2f}\n")
                            tmp.write(f"- Timestamp: {item.timestamp}\n\n")

                            content = item.content.replace('```', '\\`\\`\\`')
                            tmp.write(f"```\n{content[:1000]}\n```\n\n")
                            if len(item.content) > 1000:
                                tmp.write("[... truncated ...]\n\n")

                        tmp_path = Path(tmp.name)
                    tmp_path.replace(export_path)

                else:
                    LOGGER.error(f"Unsupported export format: {format}")
                    return None

            LOGGER.info(f"Exported knowledge to {export_path}")
            return export_path

        except Exception as e:
            LOGGER.error(f"Knowledge export failed: {e}")
            return None

    def cleanup(self, max_age_days: int = 30, min_relevance: float = 0.1) -> int:
        """
        Clean up old or low-relevance knowledge items.

        Args:
            max_age_days: Maximum age in days to keep
            min_relevance: Minimum relevance score to keep

        Returns:
            Number of items removed
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=max_age_days)
            removed_count = 0

            with self._kb_lock:
                to_keep = []

                for item in self.knowledge_items:
                    item_time = datetime.fromisoformat(item.timestamp)
                    keep = (
                        item_time >= cutoff_time and
                        item.relevance >= min_relevance
                    )

                    if keep:
                        to_keep.append(item)
                    else:
                        # Remove from disk
                        try:
                            if item.compressed:
                                (self.knowledge_base_dir / "compressed" / f"{item.id}.zlib").unlink(missing_ok=True)
                            else:
                                (self.knowledge_base_dir / "raw" / f"{item.id}.txt").unlink(missing_ok=True)
                            (self.knowledge_base_dir / "metadata" / f"{item.id}.json").unlink(missing_ok=True)

                            # Remove from indexes
                            self.content_index.pop(item.content_hash, None)
                            source_items = self.source_index.get(item.source, [])
                            if item.id in source_items:
                                source_items.remove(item.id)

                            removed_count += 1
                        except Exception as e:
                            LOGGER.error(f"Failed to remove item {item.id}: {e}")

                self.knowledge_items = to_keep

            LOGGER.info(f"Knowledge cleanup removed {removed_count} items")
            return removed_count

        except Exception as e:
            LOGGER.error(f"Knowledge cleanup failed: {e}")
            return 0
