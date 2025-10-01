"""
Data Source Factory

This module provides a factory pattern for creating and managing
different economic data source clients (BLS, BEA, FRED).
"""

from typing import Union, Optional, TYPE_CHECKING, Any
import logging
from functools import lru_cache

from models import DataSource
from config import get_config
from bls_client import BLSClient

# Import new clients with proper type hints
try:
    from bea_client import BEAClient, BEAAPIException
except ImportError:
    BEAClient = None
    BEAAPIException = Exception

try:
    from fred_client import FREDClient, FREDAPIException
except ImportError:
    FREDClient = None
    FREDAPIException = Exception

logger = logging.getLogger(__name__)


class DataSourceFactory:
    """Factory class for creating data source clients"""
    
    _clients = {}
    _config = None
    
    @classmethod
    def get_config(cls):
        """Get configuration instance"""
        if cls._config is None:
            cls._config = get_config()
        return cls._config
    
    @classmethod
    @lru_cache(maxsize=3)
    def create_client(cls, source: DataSource, **kwargs) -> Union[BLSClient, Any]:
        """
        Create a client for the specified data source
        
        Args:
            source: The data source (BLS, BEA, FRED)
            **kwargs: Additional parameters for client creation
            
        Returns:
            Client instance for the specified source
            
        Raises:
            ValueError: If the source is not supported
            ImportError: If the client module is not available
        """
        config = cls.get_config()
        
        if source == DataSource.BLS:
            if BLSClient is None:
                raise ImportError("BLS client not available")
            
            api_key = kwargs.get('api_key') or config.bls_api_key
            timeout = kwargs.get('timeout', 30)
            
            logger.info(f"Creating BLS client with timeout={timeout}")
            return BLSClient(api_key=api_key, timeout=timeout)
        
        elif source == DataSource.BEA:
            if BEAClient is None:
                raise ImportError("BEA client not available - implementation pending")
            
            api_key = kwargs.get('api_key') or config.bea_api_key
            timeout = kwargs.get('timeout', 30)
            
            logger.info(f"Creating BEA client with timeout={timeout}")
            return BEAClient(api_key=api_key, timeout=timeout)
        
        elif source == DataSource.FRED:
            if FREDClient is None:
                raise ImportError("FRED client not available - implementation pending")
            
            api_key = kwargs.get('api_key') or config.fred_api_key
            timeout = kwargs.get('timeout', 30)
            
            logger.info(f"Creating FRED client with timeout={timeout}")
            return FREDClient(api_key=api_key, timeout=timeout)
        
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    @classmethod
    def get_available_sources(cls) -> list[DataSource]:
        """Get list of available data sources"""
        available = [DataSource.BLS]  # BLS is always available
        
        if BEAClient is not None:
            available.append(DataSource.BEA)
        
        if FREDClient is not None:
            available.append(DataSource.FRED)
        
        return available
    
    @classmethod
    def is_source_available(cls, source: DataSource) -> bool:
        """Check if a data source is available"""
        try:
            cls.create_client(source)
            return True
        except (ImportError, ValueError):
            return False
    
    @classmethod
    def validate_source_config(cls, source: DataSource) -> dict:
        """
        Validate configuration for a specific data source
        
        Returns:
            Dictionary with validation results
        """
        config = cls.get_config()
        result = {
            'source': source,
            'available': False,
            'configured': False,
            'api_key_present': False,
            'client_available': False,
            'errors': []
        }
        
        # Check if client is available
        if source == DataSource.BLS:
            result['client_available'] = BLSClient is not None
        elif source == DataSource.BEA:
            result['client_available'] = BEAClient is not None
        elif source == DataSource.FRED:
            result['client_available'] = FREDClient is not None
        
        if not result['client_available']:
            result['errors'].append(f"Client implementation not available for {source}")
            return result
        
        # Check API key
        api_key = None
        if source == DataSource.BLS:
            api_key = config.bls_api_key
        elif source == DataSource.BEA:
            api_key = config.bea_api_key
        elif source == DataSource.FRED:
            api_key = config.fred_api_key
        
        result['api_key_present'] = bool(api_key and api_key.strip())
        
        if not result['api_key_present']:
            result['errors'].append(f"API key not configured for {source}")
        else:
            result['configured'] = True
        
        # Test client creation
        try:
            cls.create_client(source)
            result['available'] = True
        except Exception as e:
            result['errors'].append(f"Failed to create client for {source}: {str(e)}")
        
        return result
    
    @classmethod
    def get_all_source_status(cls) -> dict:
        """Get status of all data sources"""
        status = {
            'total_sources': len(DataSource),
            'available_sources': [],
            'unavailable_sources': [],
            'configured_sources': [],
            'sources': {}
        }
        
        for source in DataSource:
            source_status = cls.validate_source_config(source)
            status['sources'][source.value] = source_status
            
            if source_status['available']:
                status['available_sources'].append(source.value)
            else:
                status['unavailable_sources'].append(source.value)
            
            if source_status['configured']:
                status['configured_sources'].append(source.value)
        
        status['available_count'] = len(status['available_sources'])
        status['configured_count'] = len(status['configured_sources'])
        
        return status


class DataSourceRouter:
    """Router for directing requests to appropriate data sources"""
    
    def __init__(self):
        self.factory = DataSourceFactory()
        self.logger = logging.getLogger(__name__)
    
    def route_request(self, source: DataSource, **kwargs):
        """
        Route a request to the appropriate data source
        
        Args:
            source: Target data source
            **kwargs: Request parameters
            
        Returns:
            Response from the appropriate client
        """
        try:
            client = self.factory.create_client(source)
            
            # Route based on method and parameters
            method = kwargs.get('method', 'get_series_data')
            
            if method == 'get_series_data':
                series_ids = kwargs.get('series_ids')
                if series_ids is None:
                    raise ValueError("series_ids parameter is required for get_series_data method")
                start_year = kwargs.get('start_year')
                end_year = kwargs.get('end_year')
                
                return client.get_series_data(
                    series_ids=series_ids,
                    start_year=start_year,
                    end_year=end_year
                )
            
            elif method == 'get_series_info':
                series_id = kwargs.get('series_id')
                if source == DataSource.BLS:
                    raise NotImplementedError(f"get_series_info method not implemented for BLS client")
                else:
                    # Use getattr to avoid static type checking issues
                    get_series_info_method = getattr(client, 'get_series_info', None)
                    if get_series_info_method:
                        return get_series_info_method(series_id)
                    else:
                        raise NotImplementedError(f"get_series_info method not available for {source}")
            
            elif method == 'search_series':
                query = kwargs.get('query')
                if source == DataSource.BLS:
                    raise NotImplementedError(f"search_series method not implemented for BLS client")
                else:
                    # Use getattr to avoid static type checking issues
                    search_series_method = getattr(client, 'search_series', None)
                    if search_series_method:
                        return search_series_method(query)
                    else:
                        raise NotImplementedError(f"search_series method not available for {source}")
            
            else:
                raise ValueError(f"Unsupported method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error routing request to {source}: {str(e)}")
            raise
    
    def get_source_for_series_id(self, series_id: str) -> Optional[DataSource]:
        """
        Determine the data source for a given series ID based on naming patterns
        
        Args:
            series_id: Series identifier
            
        Returns:
            Likely data source or None if unclear
        """
        series_id = series_id.upper()
        
        # BEA patterns (more specific patterns first)
        if any(pattern in series_id for pattern in ['A191RX', 'A191RL', 'A191RO']):
            return DataSource.BEA
        
        # FRED specific patterns (check before generic GDP)
        if any(pattern in series_id for pattern in ['GDPC1', 'GDPPOT']):
            return DataSource.FRED
        
        # FRED patterns (most series)
        if len(series_id) <= 10 and series_id.replace('_', '').isalnum():
            return DataSource.FRED
        
        # BLS patterns (typically longer with specific prefixes)
        if any(pattern in series_id for pattern in ['CUUR', 'SUUR', 'CES', 'LNS', 'CEU']):
            return DataSource.BLS
        
        # Default assumption
        return DataSource.FRED  # FRED has the broadest coverage


# Singleton instance
data_source_factory = DataSourceFactory()
data_source_router = DataSourceRouter()