"""
FRED (Federal Reserve Economic Data) API Client

This module provides a client for accessing FRED economic data with
built-in error handling, circuit breaker pattern, and data normalization.
"""

import requests
import json
import time
import logging
import urllib3
import pybreaker
from typing import List, Dict, Optional, Union, Any, TypedDict
from datetime import datetime

from models import DataSource, DataFrequency
from config import get_config

# Disable SSL warnings for cleaner output
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class FREDAPIException(Exception):
    """Custom exception for FRED API errors"""
    pass


class FREDAPIPybreaker(pybreaker.CircuitBreakerListener):
    """Circuit breaker listener for FRED API monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def state_change(self, cb, old_state, new_state):
        """Log circuit breaker state changes"""
        self.logger.warning(f"FRED API Circuit Breaker: {old_state} -> {new_state}")
        
    def failure(self, cb, exc):
        """Log circuit breaker failures"""
        self.logger.error(f"FRED API Circuit Breaker failure: {exc}")
        
    def success(self, cb):
        """Log circuit breaker successes"""
        self.logger.info("FRED API Circuit Breaker success")


class FREDPayload(TypedDict, total=False):
    """API request payload for FRED"""
    api_key: str
    file_type: str
    series_id: str
    series_ids: str
    observation_start: str
    observation_end: str
    realtime_start: str
    realtime_end: str
    limit: int
    offset: int
    sort_order: str
    frequency: str
    units: str
    aggregation_method: str
    output_type: int


class FREDClient:
    """FRED API client with built-in error handling and data normalization"""
    
    BASE_URL = "https://api.stlouisfed.org/fred/"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    DEFAULT_FILE_TYPE = "json"
    MAX_SERIES_PER_REQUEST = 100  # FRED limits
    
    # FRED API endpoints
    ENDPOINTS = {
        'series': 'series',
        'observations': 'series/observations',
        'releases': 'releases',
        'release': 'release',
        'release_series': 'release/series',
        'categories': 'categories',
        'category': 'category',
        'category_series': 'category/series',
        'tags': 'tags',
        'tags_series': 'tags/series',
        'search': 'series/search',
        'search_tags': 'series/search/tags'
    }
    
    # FRED frequency mappings
    FREQUENCY_MAP = {
        'd': DataFrequency.DAILY,
        'w': DataFrequency.WEEKLY,
        'bw': DataFrequency.BIWEEKLY,
        'm': DataFrequency.MONTHLY,
        'q': DataFrequency.QUARTERLY,
        'sa': DataFrequency.SEMIANNUAL,
        'a': DataFrequency.ANNUAL
    }
    
    # FRED units mappings
    UNITS_MAP = {
        'lin': 'Levels',
        'chg': 'Change',
        'ch1': 'Change from Year Ago',
        'pch': 'Percent Change',
        'pc1': 'Percent Change from Year Ago',
        'pca': 'Compounded Annual Rate of Change',
        'cch': 'Continuously Compounded Rate of Change',
        'cca': 'Continuously Compounded Annual Rate of Change',
        'log': 'Natural Log'
    }
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        config = get_config()
        self.api_key = api_key or config.fred_api_key
        self.timeout = timeout
        
        if not self.api_key:
            raise FREDAPIException("FRED API key is required. Set FRED_API_KEY environment variable or pass api_key parameter.")
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EconoVault-FRED-Client/1.0',
            'Accept': 'application/json'
        })
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize circuit breaker
        self.circuit_breaker = pybreaker.CircuitBreaker(
            fail_max=5,                    # 5 consecutive failures before opening
            reset_timeout=60,              # 60 seconds before attempting reset
            success_threshold=3,           # 3 successful calls before closing
            name="FRED_API_BREAKER",
            listeners=[FREDAPIPybreaker()]
        )
        
        # Exclude 4xx errors from circuit breaker failures
        self.circuit_breaker.add_excluded_exception(requests.HTTPError)
        
        # Add alerting listener if alerting is available
        try:
            from alerting import get_alerting_service, CircuitBreakerAlertListener
            alerting_service = get_alerting_service()
            listener = CircuitBreakerAlertListener(alerting_service, "FRED_API")
            self.circuit_breaker.add_listeners(listener)
            self.logger.info("Circuit breaker alerting listener added")
        except Exception as e:
            self.logger.warning(f"Could not add circuit breaker alerting listener: {e}")
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with circuit breaker and retry logic"""
        if not self.api_key:
            raise FREDAPIException("FRED API key not configured")
        
        # Add required parameters
        params['api_key'] = self.api_key
        params['file_type'] = self.DEFAULT_FILE_TYPE
        
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(self.MAX_RETRIES):
            try:
                self.logger.debug(f"FRED API request (attempt {attempt + 1}): {url} with params {params}")
                
                @self.circuit_breaker
                def _request():
                    response = self.session.get(
                        url,
                        params=params,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        try:
                            return response.json()
                        except json.JSONDecodeError as e:
                            raise FREDAPIException(f"Invalid JSON response: {e}")
                    else:
                        error_msg = f"HTTP {response.status_code}: {response.text}"
                        if response.status_code == 400:
                            raise FREDAPIException(f"Bad request: {error_msg}")
                        elif response.status_code == 401:
                            raise FREDAPIException(f"Invalid API key: {error_msg}")
                        elif response.status_code == 429:
                            raise FREDAPIException(f"Rate limit exceeded: {error_msg}")
                        else:
                            response.raise_for_status()
                
                result = _request()
                
                # Check for API errors in response
                if isinstance(result, dict) and 'error_code' in result:
                    error_code = result.get('error_code')
                    error_message = result.get('error_message', 'Unknown error')
                    raise FREDAPIException(f"FRED API Error {error_code}: {error_message}")
                
                return result
                
            except pybreaker.CircuitBreakerError:
                raise FREDAPIException("FRED API circuit breaker is open. Service temporarily unavailable.")
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"FRED API timeout (attempt {attempt + 1})")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise FREDAPIException("FRED API request timed out after all retries")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"FRED API request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise FREDAPIException(f"FRED API request failed: {str(e)}")
                    
            except FREDAPIException:
                # Re-raise FRED API exceptions without retry
                raise
                
            except Exception as e:
                self.logger.error(f"Unexpected error in FRED API request (attempt {attempt + 1}): {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise FREDAPIException(f"Unexpected error: {str(e)}")
        
        raise FREDAPIException("FRED API request failed after all retries")
    
    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific series"""
        try:
            params = {
                'series_id': series_id
            }
            
            result = self._make_request(self.ENDPOINTS['series'], params)
            
            # Add source information
            if 'seriess' in result:
                for series in result['seriess']:
                    series['source'] = DataSource.FRED
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get FRED series info for {series_id}: {str(e)}")
            raise FREDAPIException(f"Failed to get series info: {str(e)}")
    
    def get_series_data(self, series_ids: Union[str, List[str]], 
                       observation_start: Optional[str] = None,
                       observation_end: Optional[str] = None,
                       frequency: Optional[str] = None,
                       units: Optional[str] = None,
                       aggregation_method: Optional[str] = None,
                       limit: Optional[int] = None,
                       sort_order: str = 'asc') -> Dict[str, Any]:
        """
        Get observations/data for one or more series
        
        Args:
            series_ids: Single series ID or list of series IDs
            observation_start: Start date (YYYY-MM-DD)
            observation_end: End date (YYYY-MM-DD)
            frequency: Frequency aggregation (d, w, bw, m, q, sa, a)
            units: Units transformation (lin, chg, ch1, pch, pc1, pca, cch, cca, log)
            aggregation_method: Aggregation method (avg, sum, eop)
            limit: Maximum number of observations to return
            sort_order: Sort order (asc, desc)
            
        Returns:
            Raw FRED API response
        """
        try:
            # Handle single series ID vs multiple
            if isinstance(series_ids, list):
                if len(series_ids) == 1:
                    series_id_param = series_ids[0]
                elif len(series_ids) > 1:
                    # FRED doesn't support multiple series in a single observations call
                    # We'll need to make multiple calls
                    return self._get_multiple_series_data(
                        series_ids, 
                        observation_start=observation_start, 
                        observation_end=observation_end,
                        frequency=frequency, 
                        units=units, 
                        aggregation_method=aggregation_method, 
                        limit=limit, 
                        sort_order=sort_order
                    )
                else:
                    raise FREDAPIException("No series IDs provided")
            else:
                series_id_param = series_ids
            
            params: Dict[str, Any] = {
                'series_id': series_id_param,
                'sort_order': sort_order
            }
            
            if observation_start:
                params['observation_start'] = observation_start
            if observation_end:
                params['observation_end'] = observation_end
            if frequency:
                params['frequency'] = frequency
            if units:
                params['units'] = units
            if aggregation_method:
                params['aggregation_method'] = aggregation_method
            if limit:
                params['limit'] = limit
            
            result = self._make_request(self.ENDPOINTS['observations'], params)
            
            # Add source information
            if 'observations' in result:
                result['source'] = DataSource.FRED
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get FRED series data: {str(e)}")
            raise FREDAPIException(f"Failed to get series data: {str(e)}")
    
    def _get_multiple_series_data(self, series_ids: List[str], **kwargs) -> Dict[str, Any]:
        """Get data for multiple series (makes multiple API calls)"""
        all_results = {}
        
        for series_id in series_ids:
            try:
                result = self.get_series_data(series_id, **kwargs)
                all_results[series_id] = result
            except Exception as e:
                self.logger.error(f"Failed to get data for series {series_id}: {str(e)}")
                all_results[series_id] = {'error': str(e)}
        
        return {
            'multi_series_results': all_results,
            'source': DataSource.FRED
        }
    
    def search_series(self, search_text: str, limit: int = 1000,
                     filter_variable: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for series by text
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            filter_variable: Filter by variable (e.g., 'cid', 'pop', 'rgdp')
            
        Returns:
            Search results
        """
        try:
            params = {
                'search_text': search_text,
                'limit': limit
            }
            
            if filter_variable:
                params['filter_variable'] = filter_variable
            
            result = self._make_request(self.ENDPOINTS['search'], params)
            
            # Add source information
            if 'seriess' in result:
                for series in result['seriess']:
                    series['source'] = DataSource.FRED
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to search FRED series: {str(e)}")
            raise FREDAPIException(f"Failed to search series: {str(e)}")
    
    def get_releases(self, limit: int = 1000, offset: int = 0) -> Dict[str, Any]:
        """Get all releases of economic data"""
        try:
            params = {
                'limit': limit,
                'offset': offset
            }
            
            result = self._make_request(self.ENDPOINTS['releases'], params)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get FRED releases: {str(e)}")
            raise FREDAPIException(f"Failed to get releases: {str(e)}")
    
    def get_release_info(self, release_id: int) -> Dict[str, Any]:
        """Get information about a specific release"""
        try:
            params = {
                'release_id': release_id
            }
            
            result = self._make_request(self.ENDPOINTS['release'], params)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get FRED release info for {release_id}: {str(e)}")
            raise FREDAPIException(f"Failed to get release info: {str(e)}")
    
    def get_release_series(self, release_id: int, limit: int = 1000, 
                          offset: int = 0, order_by: str = 'series_id',
                          sort_order: str = 'asc') -> Dict[str, Any]:
        """Get series for a specific release"""
        try:
            params = {
                'release_id': release_id,
                'limit': limit,
                'offset': offset,
                'order_by': order_by,
                'sort_order': sort_order
            }
            
            result = self._make_request(self.ENDPOINTS['release_series'], params)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get FRED release series for {release_id}: {str(e)}")
            raise FREDAPIException(f"Failed to get release series: {str(e)}")
    
    def get_categories(self, limit: int = 1000, offset: int = 0) -> Dict[str, Any]:
        """Get all categories"""
        try:
            params = {
                'limit': limit,
                'offset': offset
            }
            
            result = self._make_request(self.ENDPOINTS['categories'], params)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get FRED categories: {str(e)}")
            raise FREDAPIException(f"Failed to get categories: {str(e)}")
    
    def get_category_info(self, category_id: int) -> Dict[str, Any]:
        """Get information about a specific category"""
        try:
            params = {
                'category_id': category_id
            }
            
            result = self._make_request(self.ENDPOINTS['category'], params)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get FRED category info for {category_id}: {str(e)}")
            raise FREDAPIException(f"Failed to get category info: {str(e)}")
    
    def get_category_series(self, category_id: int, limit: int = 1000,
                           offset: int = 0, order_by: str = 'series_id',
                           sort_order: str = 'asc', filter_variable: Optional[str] = None) -> Dict[str, Any]:
        """Get series for a specific category"""
        try:
            params = {
                'category_id': category_id,
                'limit': limit,
                'offset': offset,
                'order_by': order_by,
                'sort_order': sort_order
            }
            
            if filter_variable:
                params['filter_variable'] = filter_variable
            
            result = self._make_request(self.ENDPOINTS['category_series'], params)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get FRED category series for {category_id}: {str(e)}")
            raise FREDAPIException(f"Failed to get category series: {str(e)}")
    
    def get_tags(self, limit: int = 1000, offset: int = 0) -> Dict[str, Any]:
        """Get all tags"""
        try:
            params = {
                'limit': limit,
                'offset': offset
            }
            
            result = self._make_request(self.ENDPOINTS['tags'], params)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get FRED tags: {str(e)}")
            raise FREDAPIException(f"Failed to get tags: {str(e)}")
    
    def get_tags_series(self, tags: Union[str, List[str]], limit: int = 1000,
                       offset: int = 0) -> Dict[str, Any]:
        """Get series for specific tags"""
        try:
            if isinstance(tags, list):
                tags = ';'.join(tags)
            
            params = {
                'tags': tags,
                'limit': limit,
                'offset': offset
            }
            
            result = self._make_request(self.ENDPOINTS['tags_series'], params)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get FRED tags series: {str(e)}")
            raise FREDAPIException(f"Failed to get tags series: {str(e)}")
    
    def get_gdp_data(self, observation_start: Optional[str] = None,
                    observation_end: Optional[str] = None) -> Dict[str, Any]:
        """Get Real GDP data (most common use case)"""
        return self.get_series_data(
            series_ids='GDPC1',  # Real Gross Domestic Product
            observation_start=observation_start,
            observation_end=observation_end
        )
    
    def get_cpi_data(self, observation_start: Optional[str] = None,
                    observation_end: Optional[str] = None) -> Dict[str, Any]:
        """Get Consumer Price Index data"""
        return self.get_series_data(
            series_ids='CPIAUCSL',  # Consumer Price Index for All Urban Consumers
            observation_start=observation_start,
            observation_end=observation_end
        )
    
    def get_unemployment_data(self, observation_start: Optional[str] = None,
                             observation_end: Optional[str] = None) -> Dict[str, Any]:
        """Get Unemployment Rate data"""
        return self.get_series_data(
            series_ids='UNRATE',  # Unemployment Rate
            observation_start=observation_start,
            observation_end=observation_end
        )
    
    def get_federal_funds_rate(self, observation_start: Optional[str] = None,
                              observation_end: Optional[str] = None) -> Dict[str, Any]:
        """Get Federal Funds Rate data"""
        return self.get_series_data(
            series_ids='FEDFUNDS',  # Federal Funds Effective Rate
            observation_start=observation_start,
            observation_end=observation_end
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Check FRED API health and availability"""
        try:
            # Try to get a simple series as health check
            self.get_series_info('GDPC1')
            
            return {
                'status': 'healthy',
                'source': 'FRED',
                'api_accessible': True,
                'circuit_breaker_state': self.circuit_breaker.state,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'source': 'FRED',
                'error': str(e),
                'circuit_breaker_state': self.circuit_breaker.state,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_series_updates(self, limit: int = 1000, 
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None) -> Dict[str, Any]:
        """Get recently updated series"""
        try:
            params: Dict[str, Any] = {
                'limit': limit
            }
            
            if start_time:
                params['start_time'] = start_time
            if end_time:
                params['end_time'] = end_time
            
            # FRED doesn't have a direct updates endpoint in the base API
            # This would require using a different endpoint or approach
            # For now, return a placeholder
            return {
                'message': 'Series updates not directly available in FRED API',
                'suggestion': 'Use series/observations with realtime parameters',
                'source': DataSource.FRED
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get FRED series updates: {str(e)}")
            raise FREDAPIException(f"Failed to get series updates: {str(e)}")


# Global instance will be created when needed
fred_client = None