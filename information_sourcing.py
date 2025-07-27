import logging
import requests
import re
import time
import json
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote, parse_qs, urlunparse
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
from duckduckgo_search import DDGS
from tqdm import tqdm
import socket
import urllib3
from fake_useragent import UserAgent
import html2text
import concurrent.futures

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
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = True
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        LOGGER.info("InformationSourcing initialized with enhanced capabilities")

    def _is_valid_url(self, url: str, domain_counter: Counter) -> bool:
        """Enhanced URL validation with domain rotation and security checks."""
        try:
            parsed = urlparse(url)

            # Basic URL validation
            if not all([parsed.scheme in ['http', 'https'], parsed.netloc]):
                LOGGER.debug(f"Invalid URL scheme or netloc: {url}")
                return False

            # Domain rotation check
            domain = parsed.netloc
            if domain_counter[domain] >= self.scraper_config.get('max_per_domain', 5):
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
        # Expanded unsafe patterns
        unsafe_patterns = [
            r'\.(exe|zip|rar|dmg|pkg|apk|bat|cmd|com|msi|jar)$',
            r'(malware|phishing|spam|scam|exploit|hack)',
            r'(ad|track|analytics|pixel)\.',
            r'\.(ru|su|cn|cc|xyz|info|biz|top|pw|icu)$'
        ]

        domain_lower = domain.lower()
        return not any(re.search(pattern, domain_lower) for pattern in unsafe_patterns)

    def _clean_url(self, url: str) -> str:
        """Remove tracking parameters and fragments from URL."""
        parsed = urlparse(url)
        # Remove tracking parameters
        query_dict = parse_qs(parsed.query)
        filtered_query = {k: v for k, v in query_dict.items() 
                         if not k.startswith(('utm_', 'fbclid', 'gclid', 'msclkid'))}
        clean_query = "&".join(f"{k}={v[0]}" for k, v in filtered_query.items() if v)
        return urlunparse(parsed._replace(query=clean_query, fragment=""))

    def scrape_page(self, url: str) -> Tuple[Optional[str], List[str]]:
        """
        Enhanced webpage scraping with better content extraction and error handling.
        Returns (content, links) tuple.
        """
        # Rate limiting
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = current_time

        headers = {
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': 'https://www.google.com/',
            'DNT': '1'
        }

        for attempt in range(self.scraper_config.get('max_retries', 3)):
            try:
                # Random delay between requests to avoid rate limiting
                time.sleep(random.uniform(0.5, 2.5))

                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=self.scraper_config.get('timeout', 15),
                    allow_redirects=True
                )

                response.raise_for_status()

                # Content type validation
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' not in content_type:
                    LOGGER.debug(f"Non-HTML content at {url}: {content_type}")
                    return None, []

                # Parse with BeautifulSoup using lxml if available
                soup = BeautifulSoup(response.text, 'lxml')

                # Enhanced content cleaning
                for element in soup(['script', 'style', 'nav', 'footer', 'header',
                                   'form', 'aside', 'iframe', 'noscript', 'svg',
                                   'button', 'input', 'select', 'textarea']):
                    element.decompose()

                # Remove inline styles and classes
                for tag in soup.find_all(True):
                    tag.attrs = {}

                # Improved content extraction using html2text
                self.html_converter.reset()  # Reset state
                self.html_converter.handle(soup.prettify())
                content = self.html_converter.result()
                content = re.sub(r'\n{3,}', '\n\n', content)  # Reduce excessive newlines
                content = re.sub(r'\s{2,}', ' ', content).strip()

                # Skip if content is too short
                min_length = self.scraper_config.get('min_content_length', 200)
                if len(content) < min_length:
                    LOGGER.debug(f"Content too short at {url}: {len(content)} chars")
                    return None, []

                # Enhanced link extraction with deduplication
                links = set()
                base_domain = urlparse(url).netloc

                for link in soup.find_all('a', href=True):
                    href = link['href'].strip()
                    if not href or href.startswith(('#', 'javascript:', 'mailto:')):
                        continue

                    full_url = urljoin(url, href)
                    clean_url = self._clean_url(full_url)
                    parsed = urlparse(clean_url)

                    # Only keep links from the same domain or trusted sources
                    if parsed.netloc == base_domain or self._is_trusted_domain(parsed.netloc):
                        links.add(clean_url)

                LOGGER.info(f"Scraped {len(content)} chars from {url}")
                return content, list(links)[:self.scraper_config.get('max_links', 15)]

            except requests.exceptions.RequestException as e:
                LOGGER.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                time.sleep(self.scraper_config.get('retry_delay', 5) * (attempt + 1))
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
            'stackexchange.com',
            'arxiv.org',
            'medium.com',
            'towardsdatascience.com',
            'microsoft.com',
            'google.com',
            'openai.com',
            'python.org',
            'pytorch.org',
            'tensorflow.org',
            'ibm.com',
            'oracle.com',
            'mozilla.org'
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
        domain_counter = Counter()
        results = []
        total_content_length = 0

        # Generate optimized search queries
        search_queries = self._generate_search_queries(topic)

        # Get initial seed URLs from multiple search queries
        for query in search_queries:
            seed_urls = self._search_web(query, num_results=5)
            if seed_urls:
                # Add with initial depth 0
                queue.extend([(url, 0) for url in seed_urls])
                LOGGER.info(f"Added {len(seed_urls)} seed URLs for query: {query}")
            if len(queue) >= max_pages * 2:  # Get enough seeds
                break

        # Process queue with progress tracking
        with tqdm(total=max_pages, desc="Research Progress") as pbar:
            while queue and len(results) < max_pages:
                url, current_depth = queue.pop(0)
                
                if url in processed_urls:
                    continue
                
                domain = urlparse(url).netloc
                if not self._is_valid_url(url, domain_counter):
                    continue

                # Scrape the page
                content, links = self.scrape_page(url)
                if not content:
                    processed_urls.add(url)
                    continue

                # Add to results
                result = {
                    "url": url,
                    "domain": domain,
                    "content": content,
                    "length": len(content),
                    "timestamp": datetime.now().isoformat(),
                    "depth": current_depth
                }
                results.append(result)
                processed_urls.add(url)
                domain_counter[domain] += 1
                total_content_length += len(content)

                # Add new links to queue if we need more pages and depth allows
                if (len(results) < max_pages and 
                    links and 
                    current_depth < depth):
                    
                    for link in links:
                        if link not in processed_urls and link not in [u for u, _ in queue]:
                            queue.append((link, current_depth + 1))
                            if len(queue) >= max_pages * 3:  # Don't let queue grow too large
                                break

                pbar.update(1)
                pbar.set_postfix({
                    "pages": len(results),
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
                max_tokens=200,
                timeout=15  # Add timeout
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
        try:
            # Try a simple Google search (may be blocked)
            google_url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
            headers = {'User-Agent': self.user_agent.random}
            response = requests.get(google_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            for g in soup.find_all('div', class_='g'):
                anchor = g.find('a')
                if anchor and anchor.get('href'):
                    url = anchor['href']
                    if url.startswith('/url?q='):
                        url = url.split('/url?q=')[1].split('&')[0]
                        url = unquote(url)
                        links.append(url)
            return links[:num_results]
        except Exception:
            LOGGER.warning("Google fallback search failed")
            return []

    def _truncate_at_sentence(self, text: str, max_length: int) -> str:
        """Truncate text at sentence boundary near max_length."""
        if len(text) <= max_length:
            return text
            
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_excl = truncated.rfind('!')
        last_question = truncated.rfind('?')
        sentence_end = max(last_period, last_excl, last_question)
        
        if sentence_end > 0:
            return truncated[:sentence_end + 1]
        return truncated

    def get_page_summary(self, url: str) -> Optional[Dict[str, Any]]:
        """Get a structured summary of a webpage."""
        content, _ = self.scrape_page(url)
        if not content:
            return None

        if self.ai_manager.has_available_providers():
            # Truncate at sentence boundary
            truncated_content = self._truncate_at_sentence(content, 5000)
            
            prompt = (
                f"Create a structured summary of this webpage content:\n\n{truncated_content}\n\n"
                "Include key points, main ideas, and important facts."
            )

            try:
                summary = self.ai_manager.generate_text(
                    prompt,
                    system_prompt="Return the summary as a JSON object with fields: "
                                "'title', 'key_points', 'main_ideas', 'references'",
                    max_tokens=500,
                    timeout=20
                )
                if summary:
                    try:
                        return json.loads(summary)
                    except json.JSONDecodeError:
                        LOGGER.warning("Failed to parse AI-generated summary")
            except Exception as e:
                LOGGER.error(f"Summary generation failed: {e}")

        # Fallback: create simple summary
        return {
            "title": urlparse(url).netloc,
            "key_points": content[:500].split('. ')[:3],
            "content": content[:2000] + "..."
        }

    def verify_source(self, url: str) -> Dict[str, Any]:
        """Verify the credibility of a source."""
        domain = urlparse(url).netloc
        result = {
            "url": url,
            "domain": domain,
            "is_trusted": self._is_trusted_domain(domain),
            "is_safe": self._is_safe_domain(domain),
            "timestamp": datetime.now().isoformat(),
            "trust_score": 80 if self._is_trusted_domain(domain) else 40
        }

        # Add domain age check
        result['domain_age'] = self._estimate_domain_age(domain)
        return result

    def _estimate_domain_age(self, domain: str) -> Optional[int]:
        """Estimate domain age using WHOIS if available."""
        try:
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
                
            # Try to use python-whois if installed
            try:
                import whois
                from whois.exceptions import WhoisCommandFailed
                try:
                    domain_info = whois.whois(domain)
                    if domain_info.creation_date:
                        if isinstance(domain_info.creation_date, list):
                            create_date = domain_info.creation_date[0]
                        else:
                            create_date = domain_info.creation_date
                        age = (datetime.now() - create_date).days // 365
                        return max(1, age)
                except (WhoisCommandFailed, TypeError):
                    pass
            except ImportError:
                pass
        except Exception:
            pass
        return None

    def gather_information(self, topics: List[str] = None, max_pages_per_topic: int = 5) -> None:
        """
        Main method to gather information from multiple sources and add to knowledge base.
        """
        if not topics:
            LOGGER.warning("No topics provided for information gathering")
            return

        # Use thread pool for parallel processing
        max_workers = min(3, len(topics))  # Don't exceed number of topics
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for topic in topics:
                futures.append(
                    executor.submit(
                        self._process_topic, 
                        topic, 
                        max_pages_per_topic
                    )
                )
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    LOGGER.error(f"Topic processing failed: {e}")

    def _process_topic(self, topic: str, max_pages: int):
        """Process a single research topic."""
        LOGGER.info(f"Gathering information about: {topic}")
        results = self.conduct_research(topic, max_pages=max_pages)
        
        for result in results:
            metadata = {
                "research_topic": topic,
                "source_url": result["url"],
                "domain": result["domain"],
                "content_length": result["length"],
                "gathered_at": result["timestamp"],
                "source_verification": self.verify_source(result["url"])
            }
            
            if not self.knowledge_processor.ingest_source(
                "web",
                result["content"],
                metadata
            ):
                LOGGER.warning(f"Failed to ingest content from {result['url']}")
        
        LOGGER.info(f"Completed processing {len(results)} sources for topic: {topic}")
