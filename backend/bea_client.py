"""
BEA (Bureau of Economic Analysis) API Client

This module provides a client for accessing BEA economic data with
built-in error handling, circuit breaker pattern, and data normalization.
"""

import requests
import json
import time
import logging
import urllib3
import pybreaker
from typing import List, Dict, Optional, Any, TypedDict
from datetime import datetime

from models import DataSource
from config import get_config

# Disable SSL warnings for cleaner output
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BEAAPIException(Exception):
    """Custom exception for BEA API errors"""
    pass


class BEAAPIMonitor(pybreaker.CircuitBreakerListener):
    """Circuit breaker listener for BEA API monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def state_change(self, cb, old_state, new_state):
        """Log circuit breaker state changes"""
        self.logger.warning(f"BEA API Circuit Breaker: {old_state} -> {new_state}")
        
    def failure(self, cb, exc):
        """Log circuit breaker failures"""
        self.logger.error(f"BEA API Circuit Breaker failure: {exc}")
        
    def success(self, cb):
        """Log circuit breaker successes"""
        self.logger.info("BEA API Circuit Breaker success")


class BEAAPayload(TypedDict, total=False):
    """API request payload for BEA"""
    UserID: str
    Method: str
    datasetname: str
    TableName: str
    Frequency: str
    Year: str
    YearX: str
    ResultFormat: str


class BEAClient:
    """BEA API client with built-in error handling and data normalization"""
    
    BASE_URL = "https://apps.bea.gov/api/data/"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    DEFAULT_RESULT_FORMAT = "JSON"
    
    # BEA dataset names
    DATASETS = {
        'NIPA': 'National Income and Product Accounts',
        'NIUnderlyingDetail': 'NI Underlying Detail',
        'FixedAssets': 'Fixed Assets',
        'GDPByIndustry': 'GDP by Industry',
        'InputOutput': 'Input-Output Accounts',
        'Regional': 'Regional Data',
        'ITA': 'International Transactions',
        'IntlServTrade': 'International Services Trade',
        'IIP': 'International Investment Position',
        'MNE': 'Multinational Enterprises'
    }
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        config = get_config()
        self.api_key = api_key or config.bea_api_key
        self.timeout = timeout
        
        if not self.api_key:
            raise BEAAPIException("BEA API key is required. Set BEA_API_KEY environment variable or pass api_key parameter.")
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EconoVault-BEA-Client/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        })
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize circuit breaker
        self.circuit_breaker = pybreaker.CircuitBreaker(
            fail_max=5,                    # 5 consecutive failures before opening
            reset_timeout=60,              # 60 seconds before attempting reset
            success_threshold=3,           # 3 successful calls before closing
            name="BEA_API_BREAKER",
            listeners=[BEAAPIMonitor()]
        )
        
        # Exclude 4xx errors from circuit breaker failures
        self.circuit_breaker.add_excluded_exception(requests.HTTPError)
        
        # Add alerting listener if alerting is available
        try:
            from alerting import get_alerting_service, CircuitBreakerAlertListener
            alerting_service = get_alerting_service()
            listener = CircuitBreakerAlertListener(alerting_service, "BEA_API")
            self.circuit_breaker.add_listeners(listener)
            self.logger.info("Circuit breaker alerting listener added")
        except Exception as e:
            self.logger.warning(f"Could not add circuit breaker alerting listener: {e}")
    
    def _make_request(self, payload: Dict[str, str]) -> Dict[str, Any]:
        """Make API request with circuit breaker and retry logic"""
        if not self.api_key:
            raise BEAAPIException("BEA API key not configured")
        
        # Add UserID to payload
        payload['UserID'] = self.api_key
        
        # Set default result format
        if 'ResultFormat' not in payload:
            payload['ResultFormat'] = self.DEFAULT_RESULT_FORMAT
        
        for attempt in range(self.MAX_RETRIES):
            try:
                self.logger.debug(f"BEA API request (attempt {attempt + 1}): {payload}")
                
                @self.circuit_breaker
                def _request():
                    response = self.session.post(
                        self.BASE_URL,
                        data=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        try:
                            return response.json()
                        except json.JSONDecodeError as e:
                            raise BEAAPIException(f"Invalid JSON response: {e}")
                    else:
                        error_msg = f"HTTP {response.status_code}: {response.text}"
                        if response.status_code == 400:
                            raise BEAAPIException(f"Bad request: {error_msg}")
                        elif response.status_code == 401:
                            raise BEAAPIException(f"Invalid API key: {error_msg}")
                        elif response.status_code == 429:
                            raise BEAAPIException(f"Rate limit exceeded: {error_msg}")
                        else:
                            response.raise_for_status()
                
                result = _request()
                
                # Check for API errors in response
                if 'BEAAPI' in result and 'Results' in result['BEAAPI']:
                    results = result['BEAAPI']['Results']
                    if 'Error' in results:
                        error_info = results['Error']
                        raise BEAAPIException(f"BEA API Error: {error_info.get('APIErrorDescription', 'Unknown error')}")
                
                return result
                
            except pybreaker.CircuitBreakerError:
                raise BEAAPIException("BEA API circuit breaker is open. Service temporarily unavailable.")
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"BEA API timeout (attempt {attempt + 1})")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise BEAAPIException("BEA API request timed out after all retries")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"BEA API request failed (attempt {attempt + 1}): {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise BEAAPIException(f"BEA API request failed: {str(e)}")
                    
            except BEAAPIException:
                # Re-raise BEA API exceptions without retry
                raise
                
            except Exception as e:
                self.logger.error(f"Unexpected error in BEA API request (attempt {attempt + 1}): {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise BEAAPIException(f"Unexpected error: {str(e)}")
        
        raise BEAAPIException("BEA API request failed after all retries")
    
    def get_dataset_list(self) -> List[Dict[str, Any]]:
        """Get list of available BEA datasets"""
        try:
            payload = {
                'Method': 'GETDATASETLIST'
            }
            
            result = self._make_request(payload)
            
            if 'BEAAPI' in result and 'Results' in result['BEAAPI']:
                datasets = result['BEAAPI']['Results'].get('Dataset', [])
                if isinstance(datasets, dict):
                    datasets = [datasets]
                return datasets
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get BEA dataset list: {str(e)}")
            raise BEAAPIException(f"Failed to get dataset list: {str(e)}")
    
    def get_parameter_list(self, dataset_name: str) -> Dict[str, Any]:
        """Get parameter list for a specific dataset"""
        try:
            payload = {
                'Method': 'GETPARAMETERLIST',
                'datasetname': dataset_name
            }
            
            result = self._make_request(payload)
            
            if 'BEAAPI' in result and 'Results' in result['BEAAPI']:
                return result['BEAAPI']['Results']
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get BEA parameter list for {dataset_name}: {str(e)}")
            raise BEAAPIException(f"Failed to get parameter list: {str(e)}")
    
    def get_series_data(self, table_name: str, frequency: str = 'A', 
                       year: Optional[str] = None, year_x: Optional[str] = None,
                       dataset_name: str = 'NIPA') -> Dict[str, Any]:
        """
        Get data for a specific BEA table
        
        Args:
            table_name: BEA table name
            frequency: Data frequency (A, Q, M, etc.)
            year: Single year or comma-separated years
            year_x: Year range (e.g., '2015,2020')
            dataset_name: BEA dataset name
            
        Returns:
            Raw BEA API response
        """
        try:
            payload = {
                'Method': 'GetData',
                'datasetname': dataset_name,
                'TableName': table_name,
                'Frequency': frequency
            }
            
            if year:
                payload['Year'] = year
            if year_x:
                payload['YearX'] = year_x
            
            result = self._make_request(payload)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get BEA data for {table_name}: {str(e)}")
            raise BEAAPIException(f"Failed to get series data: {str(e)}")
    
    def get_gdp_data(self, frequency: str = 'A', year: Optional[str] = None) -> Dict[str, Any]:
        """Get GDP data (most common use case)"""
        return self.get_series_data(
            table_name='T10101',  # Real GDP
            frequency=frequency,
            year=year,
            dataset_name='NIPA'
        )
    
    def get_personal_income_data(self, frequency: str = 'A', year: Optional[str] = None) -> Dict[str, Any]:
        """Get personal income data"""
        return self.get_series_data(
            table_name='T20100',  # Personal Income
            frequency=frequency,
            year=year,
            dataset_name='NIPA'
        )
    
    def get_personal_consumption_data(self, frequency: str = 'A', year: Optional[str] = None) -> Dict[str, Any]:
        """Get personal consumption expenditures data"""
        return self.get_series_data(
            table_name='T20302',  # Personal Consumption Expenditures
            frequency=frequency,
            year=year,
            dataset_name='NIPA'
        )
    
    def get_regional_data(self, table_name: str, frequency: str = 'A', 
                         year: Optional[str] = None, geo_fips: Optional[str] = None) -> Dict[str, Any]:
        """Get regional economic data"""
        try:
            payload = {
                'Method': 'GetData',
                'datasetname': 'Regional',
                'TableName': table_name,
                'Frequency': frequency
            }
            
            if year:
                payload['Year'] = year
            if geo_fips:
                payload['GeoFIPS'] = geo_fips
            
            result = self._make_request(payload)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get BEA regional data for {table_name}: {str(e)}")
            raise BEAAPIException(f"Failed to get regional data: {str(e)}")
    
    def search_series(self, query: str, dataset_name: str = 'NIPA') -> List[Dict[str, Any]]:
        """
        Search for series by querying table names and descriptions
        
        Note: BEA doesn't have a direct search API, so this searches 
        through available tables and parameters.
        """
        try:
            # Get parameter list to find matching tables
            params = self.get_parameter_list(dataset_name)
            
            matches = []
            
            # Search through ParameterArray if available
            if 'ParameterArray' in params:
                for param in params['ParameterArray']:
                    if param.get('ParameterName') == 'TableName':
                        if 'ParameterValueArray' in param:
                            for table in param['ParameterValueArray']:
                                desc = table.get('Description', '').upper()
                                value = table.get('ParameterValue', '')
                                
                                if query.upper() in desc or query.upper() in value:
                                    matches.append({
                                        'TableName': value,
                                        'Description': desc,
                                        'Dataset': dataset_name
                                    })
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Failed to search BEA series: {str(e)}")
            raise BEAAPIException(f"Failed to search series: {str(e)}")
    
    def get_series_info(self, table_name: str, dataset_name: str = 'NIPA') -> Dict[str, Any]:
        """Get detailed information about a specific table"""
        try:
            # Get parameter list to find table info
            params = self.get_parameter_list(dataset_name)
            
            if 'ParameterArray' in params:
                for param in params['ParameterArray']:
                    if param.get('ParameterName') == 'TableName':
                        if 'ParameterValueArray' in param:
                            for table in param['ParameterValueArray']:
                                if table.get('ParameterValue') == table_name:
                                    return {
                                        'table_name': table_name,
                                        'description': table.get('Description', ''),
                                        'dataset': dataset_name,
                                        'source': DataSource.BEA
                                    }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get BEA series info for {table_name}: {str(e)}")
            raise BEAAPIException(f"Failed to get series info: {str(e)}")
    
    def get_available_tables(self, dataset_name: str = 'NIPA') -> List[Dict[str, str]]:
        """Get list of available tables for a dataset"""
        try:
            tables = []
            params = self.get_parameter_list(dataset_name)
            
            if 'ParameterArray' in params:
                for param in params['ParameterArray']:
                    if param.get('ParameterName') == 'TableName':
                        if 'ParameterValueArray' in param:
                            for table in param['ParameterValueArray']:
                                tables.append({
                                    'table_name': table.get('ParameterValue', ''),
                                    'description': table.get('Description', '')
                                })
            
            return tables
            
        except Exception as e:
            self.logger.error(f"Failed to get BEA tables for {dataset_name}: {str(e)}")
            raise BEAAPIException(f"Failed to get available tables: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check BEA API health and availability"""
        try:
            # Try to get dataset list as a simple health check
            datasets = self.get_dataset_list()
            
            return {
                'status': 'healthy',
                'source': 'BEA',
                'available_datasets': len(datasets),
                'circuit_breaker_state': self.circuit_breaker.state,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'source': 'BEA',
                'error': str(e),
                'circuit_breaker_state': self.circuit_breaker.state,
                'timestamp': datetime.utcnow().isoformat()
            }


# Global instance will be created when needed
bea_client = None