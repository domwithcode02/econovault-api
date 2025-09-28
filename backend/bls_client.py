import requests
import pandas as pd
import json
import time
import logging
from typing import List, Dict, Optional, Union, Any, TypedDict
from datetime import datetime
import urllib3
import pybreaker

# Disable SSL warnings for cleaner output
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BLSAPIException(Exception):
    """Custom exception for BLS API errors"""
    pass


class BLSAPIMonitor(pybreaker.CircuitBreakerListener):
    """Circuit breaker listener for BLS API monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def state_change(self, cb, old_state, new_state):
        """Log circuit breaker state changes"""
        self.logger.warning(f"BLS API Circuit Breaker: {old_state} -> {new_state}")
        
    def failure(self, cb, exc):
        """Log circuit breaker failures"""
        self.logger.error(f"BLS API Circuit Breaker failure: {exc}")
        
    def success(self, cb):
        """Log circuit breaker successes"""
        self.logger.info("BLS API Circuit Breaker success")


class APIPayload(TypedDict, total=False):
    """API request payload with mixed value types."""
    seriesid: List[str]
    startyear: str
    endyear: str
    catalog: bool
    calculations: bool
    annualaverage: bool
    aspects: bool
    registrationkey: str


class BLSClient:
    """Minimal BLS API client with built-in error handling and data normalization"""
    
    BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    MAX_SERIES_PER_REQUEST = 25  # Conservative limit for unregistered users
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'BLS-Client/1.0'
        })
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize circuit breaker
        self.circuit_breaker = pybreaker.CircuitBreaker(
            fail_max=5,                    # 5 consecutive failures before opening
            reset_timeout=60,              # 60 seconds before attempting reset
            success_threshold=3,           # 3 successful calls before closing
            name="BLS_API_BREAKER",
            listeners=[BLSAPIMonitor()]
        )
        
        # Exclude 4xx errors from circuit breaker failures
        self.circuit_breaker.add_excluded_exception(requests.HTTPError)
        
        # Add alerting listener if alerting is available
        try:
            from alerting import get_alerting_service, CircuitBreakerAlertListener
            alerting_service = get_alerting_service()
            listener = CircuitBreakerAlertListener(alerting_service, "BLS_API")
            self.circuit_breaker.add_listeners(listener)
            self.logger.info("Circuit breaker alerting listener added")
        except Exception as e:
            self.logger.warning(f"Could not add circuit breaker alerting listener: {e}")
    
    def get_series_data(self, series_ids: Union[str, List[str]], 
                       start_year: Optional[int] = None, 
                       end_year: Optional[int] = None,
                       catalog: bool = False,
                       calculations: bool = False,
                       annual_average: bool = False,
                       aspects: bool = False) -> pd.DataFrame:
        """
        Retrieve time series data from BLS API
        
        Args:
            series_ids: Single series ID or list of series IDs
            start_year: Start year for data (4-digit)
            end_year: End year for data (4-digit)
            catalog: Include catalog information
            calculations: Include calculations
            annual_average: Include annual averages
            aspects: Include aspects
            
        Returns:
            pandas.DataFrame with normalized time series data
        """
        
        # Normalize series_ids to list
        if isinstance(series_ids, str):
            series_ids = [series_ids]
        
        # Validate and batch process
        results = []
        for i in range(0, len(series_ids), self.MAX_SERIES_PER_REQUEST):
            batch = series_ids[i:i + self.MAX_SERIES_PER_REQUEST]
            batch_data = self._make_request(batch, start_year, end_year, 
                                          catalog, calculations, annual_average, aspects)
            results.append(batch_data)
        
        # Combine all results
        combined_data = []
        for batch_result in results:
            combined_data.extend(self._normalize_response(batch_result))
        
        # Convert to DataFrame
        if not combined_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(combined_data)
        
        # Convert value to numeric, handling missing values
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Create proper datetime index
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + 
                                   df['period'].str.replace('M', '').str.zfill(2) + '-01')
        
        return df
    
    def _make_request(self, series_ids: List[str], start_year: Optional[int], 
                     end_year: Optional[int], catalog: bool, calculations: bool, 
                     annual_average: bool, aspects: bool) -> Dict[str, Any]:
        """Make API request with retry logic and circuit breaker protection"""
        
        payload: APIPayload = {
            "seriesid": series_ids
        }
        
        # Add optional parameters
        if start_year:
            payload["startyear"] = str(start_year)
        if end_year:
            payload["endyear"] = str(end_year)
        if catalog:
            payload["catalog"] = catalog
        if calculations:
            payload["calculations"] = calculations
        if annual_average:
            payload["annualaverage"] = annual_average
        if aspects:
            payload["aspects"] = aspects
        if self.api_key:
            payload["registrationkey"] = self.api_key
        
        # Use circuit breaker to protect the API call
        return self.circuit_breaker.call(
            self._make_request_with_retry, 
            payload, 
            series_ids
        )
    
    def _make_request_with_retry(self, payload: APIPayload, series_ids: List[str]) -> Dict[str, Any]:
        """Make API request with retry logic (protected by circuit breaker)"""
        
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.session.post(
                    self.BASE_URL,
                    json=payload,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                data = response.json()
                
                # Check BLS API status
                if data.get('status') != 'REQUEST_SUCCEEDED':
                    raise BLSAPIException(f"BLS API Error: {data.get('message', 'Unknown error')}")
                
                return data
                
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise BLSAPIException(f"API request failed after {self.MAX_RETRIES} attempts: {str(e)}")
                
                # Exponential backoff
                wait_time = 2 ** attempt
                self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
        
        # This should never be reached, but satisfies the type checker
        raise BLSAPIException(f"API request failed after {self.MAX_RETRIES} attempts: Unknown error")
    
    def _normalize_response(self, response_data: Dict) -> List[Dict]:
        """Normalize BLS API response to flat data structure"""
        normalized_data = []
        
        if 'Results' not in response_data:
            return normalized_data
        
        for series in response_data['Results'].get('series', []):
            series_id = series.get('seriesID', '')
            
            for data_point in series.get('data', []):
                normalized_point = {
                    'series_id': series_id,
                    'year': data_point.get('year'),
                    'period': data_point.get('period'),
                    'period_name': data_point.get('periodName'),
                    'value': data_point.get('value'),
                    'latest': data_point.get('latest', False)
                }
                
                # Handle footnotes
                footnotes = data_point.get('footnotes', [])
                if footnotes and footnotes[0]:
                    normalized_point['footnote_code'] = footnotes[0].get('code', '')
                    normalized_point['footnote_text'] = footnotes[0].get('text', '')
                else:
                    normalized_point['footnote_code'] = ''
                    normalized_point['footnote_text'] = ''
                
                # Include catalog info if available
                if 'catalog' in series:
                    catalog = series['catalog']
                    normalized_point['series_title'] = catalog.get('series_title', '')
                    normalized_point['survey_name'] = catalog.get('survey_name', '')
                    normalized_point['seasonality'] = catalog.get('seasonality', '')
                
                normalized_data.append(normalized_point)
        
        return normalized_data
    
    def get_latest_data(self, series_ids: Union[str, List[str]]) -> pd.DataFrame:
        """Get the most recent data point for given series"""
        
        if isinstance(series_ids, str):
            series_ids = [series_ids]
        
        results = []
        for series_id in series_ids:
            try:
                response = self.session.get(f"{self.BASE_URL}{series_id}?latest=true", timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 'REQUEST_SUCCEEDED':
                    results.append(data)
                else:
                    self.logger.warning(f"Failed to get latest data for {series_id}: {data.get('message')}")
                    
            except Exception as e:
                self.logger.error(f"Error getting latest data for {series_id}: {str(e)}")
                continue
        
        # Normalize all results
        combined_data = []
        for result in results:
            combined_data.extend(self._normalize_response(result))
        
        if not combined_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(combined_data)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + 
                                   df['period'].str.replace('M', '').str.zfill(2) + '-01')
        
        return df
    
    def get_popular_series(self, survey: Optional[str] = None) -> pd.DataFrame:
        """Get list of popular BLS series"""
        
        url = "https://api.bls.gov/publicAPI/v2/timeseries/popular"
        if survey:
            url += f"?survey={survey}"
        
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'REQUEST_SUCCEEDED':
            raise BLSAPIException(f"Failed to get popular series: {data.get('message')}")
        
        series_data = data.get('Results', {}).get('series', [])
        return pd.DataFrame(series_data)
    
    def __del__(self):
        """Clean up session"""
        if hasattr(self, 'session'):
            self.session.close()