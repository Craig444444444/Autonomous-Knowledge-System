import logging
import requests
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import time
import hashlib
import hmac
from urllib.parse import urlencode

LOGGER = logging.getLogger("aks")

class APIHandler:
    """
    Centralized API handler for the Autonomous Knowledge System with:
    - Rate limiting
    - Request retries
    - Authentication management
    - Response caching
    - Error handling
    """
    
    def __init__(self, cache_dir: Path = Path("/content/api_cache")):
        """
        Initialize the API handler with default configurations.
        
        Args:
            cache_dir: Directory for API response caching
        """
        self.cache_dir = cache_dir.resolve()
        self._setup_cache()
        
        # Rate limiting tracking
        self.rate_limits = {}  # {api_name: {'last_call': timestamp, 'limit': calls_per_second}}
        
        # Default configurations
        self.default_headers = {
            'User-Agent': 'AKS/1.0 (+https://github.com/Craig444444444/AKS)',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        # API configurations
        self.api_configs = {
            'github': {
                'base_url': 'https://api.github.com',
                'auth_type': 'token',
                'rate_limit': 30  # requests per minute
            },
            'gemini': {
                'base_url': 'https://generativelanguage.googleapis.com/v1beta',
                'auth_type': 'api_key',
                'rate_limit': 60
            },
            'arxiv': {
                'base_url': 'http://export.arxiv.org/api',
                'auth_type': None,
                'rate_limit': 10
            }
        }
        
        # Initialize caches
        self.response_cache = {}
        self.cache_expiry = 3600  # 1 hour cache expiry
        
        LOGGER.info("API Handler initialized with caching")

    def _setup_cache(self):
        """Set up the cache directory structure."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "responses").mkdir(exist_ok=True)
            (self.cache_dir / "errors").mkdir(exist_ok=True)
            LOGGER.debug("API cache directories initialized")
        except Exception as e:
            LOGGER.error(f"Failed to initialize API cache: {e}")

    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API call is within rate limits."""
        if api_name not in self.rate_limits:
            return True
            
        config = self.api_configs.get(api_name, {})
        limit = config.get('rate_limit', 60)
        elapsed = time.time() - self.rate_limits[api_name]['last_call']
        
        if elapsed < (60 / limit):  # Convert to calls per second
            time.sleep((60 / limit) - elapsed)
            return False
        return True

    def _generate_cache_key(self, api_name: str, endpoint: str, params: Dict) -> str:
        """Generate a unique cache key for API requests."""
        param_str = urlencode(sorted(params.items())) if params else ''
        request_str = f"{api_name}:{endpoint}:{param_str}"
        return hashlib.md5(request_str.encode()).hexdigest()

    def _cache_response(self, key: str, response: Dict, status_code: int):
        """Cache API responses with timestamp."""
        try:
            cache_entry = {
                'timestamp': datetime.now().isoformat(),
                'response': response,
                'status': status_code
            }
            
            cache_file = self.cache_dir / "responses" / f"{key}.json"
            with cache_file.open('w') as f:
                json.dump(cache_entry, f)
                
            self.response_cache[key] = cache_entry
        except Exception as e:
            LOGGER.error(f"Failed to cache API response: {e}")

    def _get_cached_response(self, key: str) -> Optional[Dict]:
        """Retrieve cached API response if available and fresh."""
        try:
            # First check memory cache
            if key in self.response_cache:
                entry = self.response_cache[key]
                cache_time = datetime.fromisoformat(entry['timestamp'])
                if (datetime.now() - cache_time).total_seconds() < self.cache_expiry:
                    return entry['response']
            
            # Fall back to disk cache
            cache_file = self.cache_dir / "responses" / f"{key}.json"
            if cache_file.exists():
                with cache_file.open('r') as f:
                    entry = json.load(f)
                cache_time = datetime.fromisoformat(entry['timestamp'])
                if (datetime.now() - cache_time).total_seconds() < self.cache_expiry:
                    self.response_cache[key] = entry
                    return entry['response']
        except Exception as e:
            LOGGER.warning(f"Cache retrieval failed: {e}")
        return None

    def _handle_error(self, api_name: str, endpoint: str, error: Exception, status_code: Optional[int] = None):
        """Log and store API errors for analysis."""
        error_data = {
            'api': api_name,
            'endpoint': endpoint,
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'status_code': status_code
        }
        
        try:
            error_file = self.cache_dir / "errors" / f"error_{int(time.time())}.json"
            with error_file.open('w') as f:
                json.dump(error_data, f)
        except Exception as e:
            LOGGER.error(f"Failed to log API error: {e}")
        
        LOGGER.error(f"API Error ({api_name}/{endpoint}): {error}")

    def make_request(
        self,
        api_name: str,
        endpoint: str,
        method: str = 'GET',
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        auth_token: Optional[str] = None,
        use_cache: bool = True,
        retries: int = 3
    ) -> Optional[Dict]:
        """
        Make an API request with built-in:
        - Rate limiting
        - Retries
        - Caching
        - Error handling
        
        Args:
            api_name: Name of the API service (github, gemini, etc.)
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Request body data
            headers: Additional headers
            auth_token: Authentication token
            use_cache: Whether to use cached responses
            retries: Number of retry attempts
            
        Returns:
            Dictionary with API response or None if failed
        """
        # Validate inputs
        if api_name not in self.api_configs:
            LOGGER.error(f"Unknown API: {api_name}")
            return None
            
        config = self.api_configs[api_name]
        base_url = config['base_url']
        full_url = f"{base_url}/{endpoint.lstrip('/')}"
        
        # Check rate limits
        self._check_rate_limit(api_name)
        
        # Generate cache key
        cache_key = self._generate_cache_key(api_name, endpoint, params or {})
        
        # Check cache first
        if use_cache and method == 'GET':
            cached = self._get_cached_response(cache_key)
            if cached is not None:
                LOGGER.debug(f"Using cached response for {api_name}/{endpoint}")
                return cached
        
        # Prepare request
        headers = {**self.default_headers, **(headers or {})}
        
        # Add authentication
        if config['auth_type'] == 'token' and auth_token:
            headers['Authorization'] = f"Bearer {auth_token}"
        elif config['auth_type'] == 'api_key' and auth_token:
            params = {**(params or {}), 'key': auth_token}
        
        # Make the request with retries
        last_error = None
        for attempt in range(retries):
            try:
                response = requests.request(
                    method=method,
                    url=full_url,
                    params=params,
                    json=data,
                    headers=headers,
                    timeout=(10, 30)  # Connect and read timeouts
                )
                
                # Update rate limit tracking
                self.rate_limits[api_name] = {
                    'last_call': time.time(),
                    'limit': config['rate_limit']
                }
                
                # Handle response
                if response.status_code == 200:
                    result = response.json()
                    self._cache_response(cache_key, result, response.status_code)
                    return result
                elif response.status_code in [401, 403]:
                    self._handle_error(api_name, endpoint, Exception("Authentication failed"), response.status_code)
                    return None
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    LOGGER.warning(f"Rate limited, retrying after {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                else:
                    error_msg = f"API request failed with status {response.status_code}"
                    self._handle_error(api_name, endpoint, Exception(error_msg), response.status_code)
                    
            except requests.exceptions.RequestException as e:
                last_error = e
                self._handle_error(api_name, endpoint, e)
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            except json.JSONDecodeError as e:
                last_error = e
                self._handle_error(api_name, endpoint, e)
                continue
                
        LOGGER.error(f"All retries failed for {api_name}/{endpoint}: {str(last_error)}")
        return None

    def batch_request(
        self,
        api_name: str,
        endpoints: List[str],
        method: str = 'GET',
        params_list: Optional[List[Dict]] = None,
        auth_token: Optional[str] = None,
        max_parallel: int = 5
    ) -> List[Optional[Dict]]:
        """
        Make multiple API requests with controlled parallelism.
        
        Args:
            api_name: Name of the API service
            endpoints: List of endpoint paths
            method: HTTP method
            params_list: List of parameter dictionaries (one per endpoint)
            auth_token: Authentication token
            max_parallel: Maximum parallel requests
            
        Returns:
            List of responses in same order as endpoints
        """
        # TODO: Implement parallel request handling
        # For now, process sequentially
        results = []
        params_list = params_list or [{}] * len(endpoints)
        
        for endpoint, params in zip(endpoints, params_list):
            result = self.make_request(
                api_name=api_name,
                endpoint=endpoint,
                method=method,
                params=params,
                auth_token=auth_token
            )
            results.append(result)
            time.sleep(1)  # Basic rate limiting between calls
            
        return results

    def clear_cache(self, max_age_hours: int = 24) -> int:
        """
        Clear cached responses older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours to keep
            
        Returns:
            Number of cache files removed
        """
        cache_dir = self.cache_dir / "responses"
        cutoff = time.time() - (max_age_hours * 3600)
        removed = 0
        
        try:
            for cache_file in cache_dir.glob('*.json'):
                if cache_file.stat().st_mtime < cutoff:
                    cache_file.unlink()
                    removed += 1
                    
            LOGGER.info(f"Cleared {removed} expired cache entries")
            return removed
        except Exception as e:
            LOGGER.error(f"Cache cleanup failed: {e}")
            return 0

    def get_api_config(self, api_name: str) -> Dict:
        """Get configuration for a specific API."""
        return self.api_configs.get(api_name, {})

    def register_api(
        self,
        api_name: str,
        base_url: str,
        auth_type: Optional[str] = None,
        rate_limit: int = 60
    ) -> bool:
        """
        Register a new API configuration.
        
        Args:
            api_name: Unique identifier for the API
            base_url: Base URL for API endpoints
            auth_type: Authentication type ('token', 'api_key', None)
            rate_limit: Requests per minute limit
            
        Returns:
            True if registration succeeded
        """
        if api_name in self.api_configs:
            LOGGER.warning(f"API {api_name} already registered")
            return False
            
        self.api_configs[api_name] = {
            'base_url': base_url.rstrip('/'),
            'auth_type': auth_type,
            'rate_limit': rate_limit
        }
        LOGGER.info(f"Registered new API: {api_name}")
        return True
