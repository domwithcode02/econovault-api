"""
EconoVault Python SDK

A comprehensive Python SDK for interacting with the EconoVault API.
Provides easy access to economic indicators, data streaming, and GDPR compliance features.
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Union, Iterator, Any, AsyncGenerator, cast
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
from urllib.parse import urlencode
import warnings

logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Types of economic indicators"""
    CPI = "CPI"  # Consumer Price Index
    EMPLOYMENT = "employment"
    UNEMPLOYMENT = "unemployment"
    GDP = "GDP"  # Gross Domestic Product
    PPI = "PPI"  # Producer Price Index
    INTEREST_RATE = "interest_rate"
    MONEY_SUPPLY = "money_supply"


class DataSource(Enum):
    """Data source organizations"""
    BLS = "BLS"  # Bureau of Labor Statistics
    BEA = "BEA"  # Bureau of Economic Analysis
    FRED = "FRED"  # Federal Reserve Economic Data


@dataclass
class EconomicIndicator:
    """Economic indicator metadata"""
    series_id: str
    title: str
    source: str
    indicator_type: str
    frequency: str
    seasonal_adjustment: str
    geography_level: str
    units: str
    latest_data: Optional[Dict] = None


@dataclass
class DataPoint:
    """Single data point in a time series"""
    date: str
    value: float
    period: str
    year: int


@dataclass
class TimeSeriesData:
    """Time series data with metadata"""
    series_id: str
    title: str
    source: str
    data: List[DataPoint]
    count: int
    date_range: Dict[str, str]


@dataclass
class APIKeyInfo:
    """API key information"""
    key_id: str
    name: str
    scopes: List[str]
    is_active: bool
    created_at: str
    last_used: Optional[str] = None


def _create_economic_indicator_safely(data: Dict[str, Any]) -> Optional[EconomicIndicator]:
    """Safely create EconomicIndicator from dictionary data with validation"""
    required_fields = ['series_id', 'title', 'source', 'indicator_type', 'frequency', 
                      'seasonal_adjustment', 'geography_level', 'units']
    
    # Check if all required fields are present
    if not all(field in data for field in required_fields):
        return None
    
    try:
        return EconomicIndicator(**data)
    except (TypeError, ValueError):
        return None


def _create_api_key_info_safely(data: Dict[str, Any]) -> Optional[APIKeyInfo]:
    """Safely create APIKeyInfo from dictionary data with validation"""
    required_fields = ['key_id', 'name', 'scopes', 'is_active', 'created_at']
    
    # Check if all required fields are present
    if not all(field in data for field in required_fields):
        return None
    
    try:
        return APIKeyInfo(**data)
    except (TypeError, ValueError):
        return None


class EconoVaultError(Exception):
    """Base exception for EconoVault SDK"""
    pass


class AuthenticationError(EconoVaultError):
    """Authentication failed"""
    pass


class RateLimitError(EconoVaultError):
    """Rate limit exceeded"""
    pass


class NotFoundError(EconoVaultError):
    """Resource not found"""
    pass


class ServerError(EconoVaultError):
    """Server error"""
    pass


class EconoVaultClient:
    """Synchronous EconoVault API client"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.econovault.com",
        api_version: str = "v1",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        user_agent: str = "EconoVault-Python-SDK/1.0.0"
    ):
        """
        Initialize the EconoVault client.
        
        Args:
            api_key: Your EconoVault API key
            base_url: Base URL for the API
            api_version: API version to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            user_agent: User agent string
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.api_version = api_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.user_agent = user_agent
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'User-Agent': user_agent,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        logger.info(f"EconoVault client initialized for {base_url}")
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Union[int, str]]] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        url = f"{self.base_url}/{self.api_version}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Making {method} request to {url} (attempt {attempt + 1})")
                
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    timeout=self.timeout
                )
                
                # Handle different response codes
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                elif response.status_code == 404:
                    raise NotFoundError("Resource not found")
                elif response.status_code == 429:
                    raise RateLimitError("Rate limit exceeded")
                elif response.status_code >= 500:
                    if attempt < self.max_retries:
                        logger.warning(f"Server error {response.status_code}, retrying in {self.retry_delay}s")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        raise ServerError(f"Server error {response.status_code}: {response.text}")
                else:
                    raise EconoVaultError(f"Unexpected status code {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    logger.warning(f"Request failed: {e}, retrying in {self.retry_delay}s")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise EconoVaultError(f"Request failed after {self.max_retries} retries: {e}")
        
        raise EconoVaultError(f"Request failed after {self.max_retries} retries")
    
    def get_indicators(
        self,
        source: Optional[str] = None,
        indicator_type: Optional[str] = None,
        limit: int = 100
    ) -> List[EconomicIndicator]:
        """
        Get list of economic indicators.
        
        Args:
            source: Filter by data source (BLS, BEA, FRED)
            indicator_type: Filter by indicator type
            limit: Maximum number of indicators to return
            
        Returns:
            List of EconomicIndicator objects
"""
        params: Dict[str, Union[int, str]] = {"limit": limit}
        if source:
            params["source"] = source
        if indicator_type:
            params["indicator_type"] = indicator_type
        
        response = self._make_request("GET", "indicators", params=params)
        
        indicators = []
        for item in response:
            item_dict = cast(Dict[str, Any], item)
            indicator = _create_economic_indicator_safely(item_dict)
            if indicator:
                indicators.append(indicator)
            else:
                logger.warning(f"Skipping indicator due to missing or invalid fields: {item}")
        
        logger.info(f"Retrieved {len(indicators)} indicators")
        return indicators
    
    def get_indicator(self, indicator_name: str) -> EconomicIndicator:
        """
        Get specific economic indicator by semantic name.
        
        Args:
            indicator_name: The indicator name (e.g., 'consumer-price-index' for CPI)
            
        Returns:
            EconomicIndicator object
        """
        response = self._make_request("GET", f"indicators/{indicator_name}")
        
        logger.info(f"Retrieved indicator {indicator_name}")
        return EconomicIndicator(**response)
    
    def get_indicator_data(
        self,
        indicator_name: str,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
        limit: int = 1000
    ) -> TimeSeriesData:
        """
        Get time series data for a specific indicator.
        
        Args:
            indicator_name: The indicator name (e.g., 'consumer-price-index')
            start_date: Start date for data range (string, date, or datetime)
            end_date: End date for data range (string, date, or datetime)
            limit: Maximum number of data points to return
            
        Returns:
            TimeSeriesData object
        """
        params: Dict[str, Union[int, str]] = {"limit": limit}
        
        if start_date:
            if isinstance(start_date, (date, datetime)):
                params["start_date"] = start_date.isoformat()
            else:
                params["start_date"] = start_date
        
        if end_date:
            if isinstance(end_date, (date, datetime)):
                params["end_date"] = end_date.isoformat()
            else:
                params["end_date"] = end_date
        
        response = self._make_request("GET", f"indicators/{indicator_name}/data", params=params)
        
# Convert data points
        data_points = []
        for item in response["data"]:
            data_points.append(DataPoint(**item))
        
        # Create TimeSeriesData object
        time_series = TimeSeriesData(
            series_id=response["series_id"],
            title=response["title"],
            source=response["source"],
            data=data_points,
            count=response["count"],
            date_range=response["date_range"]
        )
        
        logger.info(f"Retrieved {len(data_points)} data points for {indicator_name}")
        return time_series
    
    def stream_indicator_data(
        self,
        series_id: str,
        update_interval: int = 60
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream real-time updates for an indicator using Server-Sent Events.
        
        Args:
            series_id: The series ID to stream
            update_interval: Update interval in seconds
            
        Yields:
            Data update dictionaries
        """
        params = {"update_interval": update_interval}
        
        # Note: This is a simplified implementation
        # In a real implementation, you would use SSE client library
        warnings.warn(
            "Streaming is not fully implemented in sync client. Use EconoVaultAsyncClient for full streaming support.",
            UserWarning
        )
        
        # Simulate streaming with polling (for demonstration)
        while True:
            try:
                data = self.get_indicator_data(series_id, limit=1)
                yield {
                    "series_id": series_id,
                    "timestamp": datetime.now().isoformat(),
                    "data": data.data[0] if data.data else None
                }
                time.sleep(update_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                time.sleep(update_interval)
    
    def create_api_key(
        self,
        name: str,
        scopes: Optional[List[str]] = None,
        rate_limit_per_minute: int = 60,
        rate_limit_per_hour: int = 3600
    ) -> Dict[str, Any]:
        """
        Create a new API key.
        
        Args:
            name: Name for the API key
            scopes: List of permission scopes
            rate_limit_per_minute: Requests per minute limit
            rate_limit_per_hour: Requests per hour limit
            
        Returns:
            API key information including the raw key
        """
        if scopes is None:
            scopes = ["read:indicators"]
        
        data = {
            "name": name,
            "scopes": scopes,
            "rate_limit_per_minute": rate_limit_per_minute,
            "rate_limit_per_hour": rate_limit_per_hour
        }
        
        response = self._make_request("POST", "auth/api-keys", data=data)
        logger.info(f"Created API key: {name}")
        return response
    
    def get_api_keys(self) -> List[APIKeyInfo]:
        """
        Get list of API keys for the current user.
        
        Returns:
            List of APIKeyInfo objects
        """
        response = self._make_request("GET", "auth/api-keys")
        
        api_keys = []
        for item in response:
            item_dict = cast(Dict[str, Any], item)
            api_key = _create_api_key_info_safely(item_dict)
            if api_key:
                api_keys.append(api_key)
            else:
                logger.warning(f"Skipping API key due to missing or invalid fields: {item}")
        
        logger.info(f"Retrieved {len(api_keys)} API keys")
        return api_keys
    
    def delete_api_key(self, key_id: str) -> bool:
        """
        Delete an API key.
        
        Args:
            key_id: The key ID to delete
            
        Returns:
            True if successful
        """
        self._make_request("DELETE", f"auth/api-keys/{key_id}")
        logger.info(f"Deleted API key: {key_id}")
        return True
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get API health status.
        
        Returns:
            Health status information
        """
        return self._make_request("GET", "health")
    
    def close(self):
        """Close the client and cleanup resources"""
        if self.session:
            self.session.close()
        logger.info("EconoVault client closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class EconoVaultAsyncClient:
    """Asynchronous EconoVault API client"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.econovault.com",
        api_version: str = "v1",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        user_agent: str = "EconoVault-Python-SDK/1.0.0"
    ):
        """
        Initialize the async EconoVault client.
        
        Args:
            api_key: Your EconoVault API key
            base_url: Base URL for the API
            api_version: API version to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            user_agent: User agent string
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.api_version = api_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.user_agent = user_agent
        
        # Session will be created when needed
        self._session = None
        
        logger.info(f"EconoVault async client initialized for {base_url}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    'X-API-Key': self.api_key,
                    'User-Agent': self.user_agent,
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Union[int, str]]] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make async HTTP request with retry logic"""
        url = f"{self.base_url}/{self.api_version}/{endpoint.lstrip('/')}"
        session = await self._get_session()
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Making async {method} request to {url} (attempt {attempt + 1})")
                
                async with session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data
                ) as response:
                    
                    # Handle different response codes
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 401:
                        raise AuthenticationError("Invalid API key")
                    elif response.status == 404:
                        raise NotFoundError("Resource not found")
                    elif response.status == 429:
                        raise RateLimitError("Rate limit exceeded")
                    elif response.status >= 500:
                        if attempt < self.max_retries:
                            logger.warning(f"Server error {response.status}, retrying in {self.retry_delay}s")
                            await asyncio.sleep(self.retry_delay)
                            continue
                        else:
                            error_text = await response.text()
                            raise ServerError(f"Server error {response.status}: {error_text}")
                    else:
                        error_text = await response.text()
                        raise EconoVaultError(f"Unexpected status code {response.status}: {error_text}")
                        
            except aiohttp.ClientError as e:
                if attempt < self.max_retries:
                    logger.warning(f"Request failed: {e}, retrying in {self.retry_delay}s")
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    raise EconoVaultError(f"Request failed after {self.max_retries} retries: {e}")
        
        raise EconoVaultError(f"Request failed after {self.max_retries} retries")
    
    async def get_indicators(
        self,
        source: Optional[str] = None,
        indicator_type: Optional[str] = None,
        limit: int = 100
    ) -> List[EconomicIndicator]:
        """Get list of economic indicators (async version)"""
        params: Dict[str, Union[int, str]] = {"limit": limit}
        if source:
            params["source"] = source
        if indicator_type:
            params["indicator_type"] = indicator_type
        
        response = await self._make_request("GET", "indicators", params=params)
        
        indicators = []
        for item in response:
            item_dict = cast(Dict[str, Any], item)
            indicator = _create_economic_indicator_safely(item_dict)
            if indicator:
                indicators.append(indicator)
            else:
                logger.warning(f"Skipping indicator due to missing or invalid fields: {item}")
        
        logger.info(f"Retrieved {len(indicators)} indicators")
        return indicators
    
    async def get_indicator(self, series_id: str) -> EconomicIndicator:
        """Get specific economic indicator by series ID (async version)"""
        response = await self._make_request("GET", f"indicators/{series_id}")
        
        logger.info(f"Retrieved indicator {series_id}")
        return EconomicIndicator(**response)
    
    async def get_indicator_data(
        self,
        series_id: str,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
        limit: int = 1000
    ) -> TimeSeriesData:
        """Get time series data for a specific indicator (async version)"""
        params: Dict[str, Union[int, str]] = {"limit": limit}
        
        if start_date:
            if isinstance(start_date, (date, datetime)):
                params["start_date"] = start_date.isoformat()
            else:
                params["start_date"] = start_date
        
        if end_date:
            if isinstance(end_date, (date, datetime)):
                params["end_date"] = end_date.isoformat()
            else:
                params["end_date"] = end_date
        
        response = await self._make_request("GET", f"indicators/{series_id}/data", params=params)
        
        # Convert data points
        data_points = []
        for item in response["data"]:
            data_points.append(DataPoint(**item))
        
        # Create TimeSeriesData object
        time_series = TimeSeriesData(
            series_id=response["series_id"],
            title=response["title"],
            source=response["source"],
            data=data_points,
            count=response["count"],
            date_range=response["date_range"]
        )
        
        logger.info(f"Retrieved {len(data_points)} data points for {series_id}")
        return time_series
    
    async def stream_indicator_data(
        self,
        series_id: str,
        update_interval: int = 60
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream real-time updates for an indicator using Server-Sent Events (async version).
        
        Args:
            series_id: The series ID to stream
            update_interval: Update interval in seconds
            
        Yields:
            Data update dictionaries
        """
        # This would implement proper SSE streaming
        # For now, we'll use a polling approach
        
        while True:
            try:
                data = await self.get_indicator_data(series_id, limit=1)
                yield {
                    "series_id": series_id,
                    "timestamp": datetime.now().isoformat(),
                    "data": data.data[0] if data.data else None
                }
                await asyncio.sleep(update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                await asyncio.sleep(update_interval)
    
    async def create_api_key(
        self,
        name: str,
        scopes: Optional[List[str]] = None,
        rate_limit_per_minute: int = 60,
        rate_limit_per_hour: int = 3600
    ) -> Dict[str, Any]:
        """Create a new API key (async version)"""
        if scopes is None:
            scopes = ["read:indicators"]
        
        data = {
            "name": name,
            "scopes": scopes,
            "rate_limit_per_minute": rate_limit_per_minute,
            "rate_limit_per_hour": rate_limit_per_hour
        }
        
        response = await self._make_request("POST", "auth/api-keys", data=data)
        logger.info(f"Created API key: {name}")
        return response
    
    async def get_api_keys(self) -> List[APIKeyInfo]:
        """Get list of API keys for the current user (async version)"""
        response = await self._make_request("GET", "auth/api-keys")
        
        api_keys = []
        for item in response:
            item_dict = cast(Dict[str, Any], item)
            api_key = _create_api_key_info_safely(item_dict)
            if api_key:
                api_keys.append(api_key)
            else:
                logger.warning(f"Skipping API key due to missing or invalid fields: {item}")
        
        logger.info(f"Retrieved {len(api_keys)} API keys")
        return api_keys
    
    async def delete_api_key(self, key_id: str) -> bool:
        """Delete an API key (async version)"""
        await self._make_request("DELETE", f"auth/api-keys/{key_id}")
        logger.info(f"Deleted API key: {key_id}")
        return True
    
    async def get_health(self) -> Dict[str, Any]:
        """Get API health status (async version)"""
        return await self._make_request("GET", "health")
    
    async def close(self):
        """Close the async client and cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("EconoVault async client closed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Convenience functions
def get_client(api_key: str, **kwargs) -> EconoVaultClient:
    """Get a synchronous EconoVault client"""
    return EconoVaultClient(api_key, **kwargs)


def get_async_client(api_key: str, **kwargs) -> EconoVaultAsyncClient:
    """Get an asynchronous EconoVault client"""
    return EconoVaultAsyncClient(api_key, **kwargs)


# Example usage
if __name__ == "__main__":
    # Example synchronous usage
    def sync_example():
        with get_client("your-api-key") as client:
            try:
                # Get indicators
                indicators = client.get_indicators(source="BLS", limit=5)
                print(f"Found {len(indicators)} BLS indicators")
                
                # Get specific indicator
                cpi = client.get_indicator("CUUR0000SA0")
                print(f"CPI Title: {cpi.title}")
                
                # Get indicator data
                data = client.get_indicator_data("CUUR0000SA0", limit=12)
                print(f"Retrieved {len(data.data)} data points")
                
                # Get health status
                health = client.get_health()
                print(f"API Health: {health['status']}")
                
            except EconoVaultError as e:
                print(f"Error: {e}")
    
    # Example asynchronous usage
    async def async_example():
        async with get_async_client("your-api-key") as client:
            try:
                # Get indicators
                indicators = await client.get_indicators(source="BLS", limit=5)
                print(f"Found {len(indicators)} BLS indicators")
                
                # Get specific indicator
                cpi = await client.get_indicator("CUUR0000SA0")
                print(f"CPI Title: {cpi.title}")
                
                # Get indicator data
                data = await client.get_indicator_data("CUUR0000SA0", limit=12)
                print(f"Retrieved {len(data.data)} data points")
                
                # Stream data (example)
                print("Streaming data for 30 seconds...")
                start_time = time.time()
                async for update in client.stream_indicator_data("CUUR0000SA0", update_interval=5):
                    print(f"Update: {update['timestamp']} - {update['data']['value']}")
                    if time.time() - start_time > 30:  # Stream for 30 seconds
                        break
                
            except EconoVaultError as e:
                print(f"Error: {e}")
    
    # Run examples
    print("=== Synchronous Example ===")
    sync_example()
    
    print("\n=== Asynchronous Example ===")
    asyncio.run(async_example())