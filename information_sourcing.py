import logging
import requests
import re
import time
import json
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Set
from duckduckgo_search import DDGS
from tqdm import tqdm
import socket
import urllib3
from fake_useragent import UserAgent

# Disable insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOGGER = logging.getLogger("aks")

class InformationSourcing:
    """
    Enhanced information sourcing module with improved web scraping, search capabilities,
    and integration with the system's KnowledgeProcessor and SecurityManager.
    """
    def __init__(self, ai_manager: Any, knowledge_processor: Any, config: Any):
        self.ai_manager = ai_manager
        self.knowledge_processor = knowledge_processor
        self.config = config
        self.scraper_config = self.config.scraper_config
        self.user_agent = UserAgent()
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification (use with caution)
        LOGGER.info("InformationSourcing initialized with enhanced capabilities")

    def _is_valid_url(self, url: str, visited_domains: Set[str]) -> bool:
        """Enhanced URL validation with domain rotation and security checks."""
        try:
            parsed = urlparse(url)

            # Basic URL validation
            if not all([parsed.scheme in ['http', 'https'], parsed.netloc]):
                LOGGER.debug(f"Invalid URL scheme or netloc: {url}")
                return False

            # Domain rotation check
            domain = parsed.netloc
            if domain in visited_domains and list(visited_domains).count(domain) >= 3:
                LOGGER.debug(f"Domain limit reached for {domain}")
                return False

            # Security checks
            if not self._is_safe_domain(domain):
                LOGGER.warning(f"Potentially unsafe domain: {domain}")
                return False

            return True

        except Exception as e:
            LOGGER.error(f"URL validation error for {url}: {e}")
            return False

    def _is_safe_domain(self, domain: str) -> bool:
        """Check domain against known safe/unsafe lists."""
        # Basic safety checks (expand with actual security manager integration)
        unsafe_patterns = [
            r'\.(exe|zip|rar|dmg|pkg)$',
            r'(malware|phishing|spam|scam)',
            r'(ad|track|analytics)\..+$'
        ]

        domain_lower = domain.lower()
        return not any(re.search(pattern, domain_lower) for pattern in unsafe_patterns)

    def scrape_page(self, url: str) -> Tuple[Optional[str], List[str]]:
        """
        Enhanced webpage scraping with better content extraction and error handling.
        Returns (content, links) tuple.
        """
        headers = {
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': 'https://www.google.com/',
            'DNT': '1'
        }

        for attempt in range(self.scraper_config['max_retries']):
            try:
                # Random delay between requests to avoid rate limiting
                time.sleep(random.uniform(0.5, 2.5))

                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=15,
                    allow_redirects=True
                )

                response.raise_for_status()

                # Content type validation
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' not in content_type:
                    LOGGER.debug(f"Non-HTML content at {url}: {content_type}")
                    return None, []

                # Parse with BeautifulSoup using lxml if available
                soup = BeautifulSoup(response.text, 'html.parser')

                # Enhanced content cleaning
                for element in soup(['script', 'style', 'nav', 'footer', 'header',
                                   'form', 'aside', 'iframe', 'noscript', 'svg']):
                    element.decompose()

                # Improved content extraction
                content_parts = []
                for tag in self.scraper_config['extraction_tags']:
                    for element in soup.find_all(tag):
                        text = element.get_text(separator=' ', strip=True)
                        if text and len(text.split()) > 5:  # Skip very short texts
                            content_parts.append(text)

                content = "\n\n".join(content_parts)
                content = re.sub(r'\s+', ' ', content).strip()

                # Enhanced link extraction with deduplication
                links = set()
                base_domain = urlparse(url).netloc

                for link in soup.find_all('a', href=True):
                    href = link['href'].strip()
                    if not href or href.startswith(('#', 'javascript:', 'mailto:')):
                        continue

                    full_url = urljoin(url, href)
                    parsed = urlparse(full_url)

                    # Clean URL and remove tracking parameters
                    clean_url = parsed._replace(
                        query='',
                        fragment='',
                        params=''
                    ).geturl()

                    # Only keep links from the same domain or trusted sources
                    if parsed.netloc == base_domain or self._is_trusted_domain(parsed.netloc):
                        links.add(clean_url)

                LOGGER.debug(f"Scraped {len(content.split())} words from {url}")
                return content, list(links)[:self.scraper_config['max_links']]

            except requests.exceptions.RequestException as e:
                LOGGER.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                time.sleep(self.scraper_config['retry_delay'] * (attempt + 1))
            except Exception as e:
                LOGGER.error(f"Unexpected error scraping {url}: {str(e)}")
                break

        LOGGER.error(f"All scraping attempts failed for {url}")
        return None, []

    def _is_trusted_domain(self, domain: str) -> bool:
        """Check if domain is in trusted sources list."""
        trusted_domains = [
            'wikipedia.org',
            'github.com',
            'stackoverflow.com',
            'arxiv.org',
            'medium.com',
            'towardsdatascience.com'
        ]
        return any(trusted in domain for trusted in trusted_domains)

    def conduct_research(self, topic: str, depth: int = 2, max_pages: int = 10) -> List[Dict[str, Any]]:
        """
        Enhanced research method with better search, scraping, and result processing.
        Returns list of dictionaries with detailed source information.
        """
        LOGGER.info(f"Starting research on '{topic}' (depth={depth}, max_pages={max_pages})")

        # Initialize tracking structures
        queue = []
        processed_urls = set()
        visited_domains = []
        results = []
        total_content_length = 0

        # Generate optimized search queries
        search_queries = self._generate_search_queries(topic)

        # Get initial seed URLs from multiple search queries
        for query in search_queries:
            queue.extend(self._search_web(query, num_results=5))
            if len(queue) >= max_pages * 2:  # Get enough seeds
                break

        # Process queue with progress tracking
        with tqdm(total=max_pages, desc="Research Progress") as pbar:
            while queue and len(processed_urls) < max_pages:
                url = queue.pop(0)
                
                if url in processed_urls:
                    continue
                
                domain = urlparse(url).netloc
                if not self._is_valid_url(url, visited_domains):
                    continue

                # Scrape the page
                content, links = self.scrape_page(url)
                if not content:
                    continue

                # Add to results
                result = {
                    "url": url,
                    "domain": domain,
                    "content": content,
                    "length": len(content),
                    "timestamp": datetime.now().isoformat(),
                    "depth": depth - (len(queue) // 5)  # Estimate current depth
                }
                results.append(result)
                processed_urls.add(url)
                visited_domains.append(domain)
                total_content_length += len(content)

                # Add new links to queue if we need more pages
                if len(processed_urls) < max_pages and links:
                    queue.extend(links[:3])  # Add a few links for deeper crawling

                pbar.update(1)
                pbar.set_postfix({
                    "pages": len(processed_urls),
                    "words": total_content_length // 5,
                    "domain": domain[:20]
                })

        LOGGER.info(f"Research completed. Found {len(results)} sources with {total_content_length} characters")
        return results

    def _generate_search_queries(self, topic: str) -> List[str]:
        """Generate multiple optimized search queries using AI."""
        base_queries = [topic]

        if not self.ai_manager.has_available_providers():
            return base_queries

        try:
            prompt = (
                f"Generate 3-5 optimized web search queries for researching: '{topic}'\n"
                "The queries should be diverse and cover different aspects.\n"
                "Return as a JSON list of strings."
            )

            response = self.ai_manager.generate_text(
                prompt,
                system_prompt="You are a research assistant. Return only valid JSON.",
                max_tokens=200
            )

            if response:
                try:
                    queries = json.loads(response)
                    if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                        LOGGER.info(f"Generated search queries: {queries}")
                        return queries[:5]  # Limit to 5 queries
                except json.JSONDecodeError:
                    LOGGER.warning("Failed to parse AI-generated search queries")
        except Exception as e:
            LOGGER.error(f"Search query generation failed: {e}")

        return base_queries

    def _search_web(self, query: str, num_results: int = 5) -> List[str]:
        """Perform web search with fallback options."""
        LOGGER.debug(f"Searching web for: '{query}'")

        # Try DuckDuckGo first
        try:
            with DDGS() as ddgs:
                results = [r['href'] for r in ddgs.text(query, max_results=num_results)]
                if results:
                    LOGGER.debug(f"Found {len(results)} DDG results")
                    return results
        except Exception as e:
            LOGGER.warning(f"DuckDuckGo search failed: {e}")

        # Fallback to other search methods if needed
        return self._fallback_search(query, num_results)

    def _fallback_search(self, query: str, num_results: int) -> List[str]:
        """Alternative search methods when primary fails."""
        # Implement other search APIs or methods here
        LOGGER.warning(f"Using fallback search for: {query}")
        return []

    def get_page_summary(self, url: str) -> Optional[Dict[str, Any]]:
        """Get a structured summary of a webpage."""
        content, _ = self.scrape_page(url)
        if not content:
            return None

        if self.ai_manager.has_available_providers():
            prompt = (
                f"Create a structured summary of this webpage content:\n\n{content[:5000]}\n\n"
                "Include key points, main ideas, and important facts."
            )

            summary = self.ai_manager.generate_text(
                prompt,
                system_prompt="Return the summary as a JSON object with fields: "
                            "'title', 'key_points', 'main_ideas', 'references'",
                max_tokens=500
            )

            try:
                return json.loads(summary) if summary else None
            except json.JSONDecodeError:
                LOGGER.warning("Failed to parse AI-generated summary")

        return {"content": content[:2000] + "..."}  # Fallback truncated content

    def verify_source(self, url: str) -> Dict[str, Any]:
        """Verify the credibility of a source."""
        domain = urlparse(url).netloc
        result = {
            "url": url,
            "domain": domain,
            "is_trusted": self._is_trusted_domain(domain),
            "is_safe": self._is_safe_domain(domain),
            "timestamp": datetime.now().isoformat()
        }

        # Add additional verification checks here
        return result

    def gather_information(self, topics: List[str] = None, max_pages_per_topic: int = 5) -> None:
        """
        Main method to gather information from multiple sources and add to knowledge base.
        """
        if not topics:
            LOGGER.warning("No topics provided for information gathering")
            return

        for topic in topics:
            try:
                LOGGER.info(f"Gathering information about: {topic}")
                results = self.conduct_research(topic, max_pages=max_pages_per_topic)
                
                for result in results:
                    metadata = {
                        "research_topic": topic,
                        "source_url": result["url"],
                        "domain": result["domain"],
                        "content_length": result["length"],
                        "gathered_at": result["timestamp"]
                    }
                    
                    if not self.knowledge_processor.ingest_source(
                        "web",
                        result["content"],
                        metadata
                    ):
                        LOGGER.warning(f"Failed to ingest content from {result['url']}")
                
                LOGGER.info(f"Completed processing {len(results)} sources for topic: {topic}")
            except Exception as e:
                LOGGER.error(f"Failed to gather information for topic {topic}: {e}")
