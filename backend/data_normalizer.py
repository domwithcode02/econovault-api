"""
Data Normalization Layer

This module provides data normalization functions to convert data from
different sources (BLS, BEA, FRED) into unified formats.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

from models import DataSource, DataFrequency, EconomicIndicatorType, SeasonalAdjustment
from unified_models import (
    UnifiedDataPoint, UnifiedIndicator, UnifiedSeriesResponse,
    validate_numeric_value, parse_frequency_from_source, parse_date_from_source,
    DataNormalizationError
)

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalizes data from different sources into unified formats"""
    
    @staticmethod
    def normalize_bea_data(raw_data: Dict[str, Any], series_metadata: Dict[str, Any]) -> List[UnifiedDataPoint]:
        """
        Normalize BEA API response data
        
        Args:
            raw_data: Raw BEA API response
            series_metadata: Series metadata from BEA
            
        Returns:
            List of normalized data points
        """
        try:
            data_points = []
            
            # Extract data from BEA response structure
            if 'BEAAPI' not in raw_data or 'Results' not in raw_data['BEAAPI']:
                raise DataNormalizationError("Invalid BEA response structure")
            
            results = raw_data['BEAAPI']['Results']
            
            # Handle different BEA response formats
            if 'Data' not in results:
                logger.warning("No data found in BEA response")
                return data_points
            
            data_array = results['Data']
            if not isinstance(data_array, list):
                data_array = [data_array]
            
            for item in data_array:
                try:
                    # Parse date
                    time_period = item.get('TimePeriod', '')
                    timestamp = parse_date_from_source(DataSource.BEA, time_period)
                    
                    # Parse value
                    raw_value = item.get('DataValue', '')
                    value = validate_numeric_value(raw_value)
                    
                    if value is not None:
                        data_point = UnifiedDataPoint(
                            series_id=series_metadata.get('SeriesID', ''),
                            timestamp=timestamp,
                            value=value,
                            source=DataSource.BEA,
                            frequency=parse_frequency_from_source(DataSource.BEA, time_period),
                            units=series_metadata.get('UnitName'),
                            # BEA typically doesn't have seasonal adjustment in the same way
                            seasonal_adjustment=SeasonalAdjustment.NOT_SEASONALLY_ADJUSTED,
                            realtime_start=None,
                            realtime_end=None
                        )
                        data_points.append(data_point)
                        
                except Exception as e:
                    logger.warning(f"Failed to normalize BEA data point {item}: {str(e)}")
                    continue
            
            logger.info(f"Normalized {len(data_points)} BEA data points")
            return data_points
            
        except Exception as e:
            logger.error(f"BEA data normalization failed: {str(e)}")
            raise DataNormalizationError(f"Failed to normalize BEA data: {str(e)}")
    
    @staticmethod
    def normalize_fred_data(raw_data: Dict[str, Any], series_metadata: Dict[str, Any]) -> List[UnifiedDataPoint]:
        """
        Normalize FRED API response data
        
        Args:
            raw_data: Raw FRED API response
            series_metadata: Series metadata from FRED
            
        Returns:
            List of normalized data points
        """
        try:
            data_points = []
            
            # Extract observations from FRED response
            if 'observations' not in raw_data:
                raise DataNormalizationError("Invalid FRED response structure - no observations found")
            
            observations = raw_data['observations']
            if not isinstance(observations, list):
                observations = [observations]
            
            for obs in observations:
                try:
                    # Parse date
                    date_str = obs.get('date', '')
                    if not date_str:
                        continue
                        
                    timestamp = parse_date_from_source(DataSource.FRED, date_str)
                    
                    # Parse value
                    raw_value = obs.get('value', '')
                    value = validate_numeric_value(raw_value)
                    
                    if value is not None:
                        data_point = UnifiedDataPoint(
                            series_id=series_metadata.get('id', ''),
                            timestamp=timestamp,
                            value=value,
                            source=DataSource.FRED,
                            frequency=parse_frequency_from_source(DataSource.FRED, series_metadata.get('frequency_short')),
                            units=series_metadata.get('units_short'),
                            seasonal_adjustment=DataNormalizer._parse_fred_seasonal_adjustment(
                                series_metadata.get('seasonal_adjustment_short')
                            ),
                            realtime_start=DataNormalizer._parse_fred_datetime(obs.get('realtime_start')),
                            realtime_end=DataNormalizer._parse_fred_datetime(obs.get('realtime_end'))
                        )
                        data_points.append(data_point)
                        
                except Exception as e:
                    logger.warning(f"Failed to normalize FRED data point {obs}: {str(e)}")
                    continue
            
            logger.info(f"Normalized {len(data_points)} FRED data points")
            return data_points
            
        except Exception as e:
            logger.error(f"FRED data normalization failed: {str(e)}")
            raise DataNormalizationError(f"Failed to normalize FRED data: {str(e)}")
    
    @staticmethod
    def normalize_bls_data(raw_data: pd.DataFrame, series_metadata: Dict[str, Any]) -> List[UnifiedDataPoint]:
        """
        Normalize BLS API response data
        
        Args:
            raw_data: BLS data as pandas DataFrame
            series_metadata: Series metadata
            
        Returns:
            List of normalized data points
        """
        try:
            data_points = []
            
            if raw_data.empty:
                logger.warning("Empty BLS DataFrame received")
                return data_points
            
            for _, row in raw_data.iterrows():
                try:
                    # Parse date from BLS year and period
                    year = int(row.get('year', 0))
                    period = row.get('period', '')
                    
                    timestamp = DataNormalizer._parse_bls_date(year, period)
                    
                    # Parse value
                    raw_value = row.get('value', '')
                    value = validate_numeric_value(raw_value)
                    
                    if value is not None:
                        data_point = UnifiedDataPoint(
                            series_id=series_metadata.get('series_id', ''),
                            timestamp=timestamp,
                            value=value,
                            source=DataSource.BLS,
                            frequency=DataNormalizer._parse_bls_frequency(period),
                            units=series_metadata.get('units'),
                            seasonal_adjustment=DataNormalizer._parse_bls_seasonal_adjustment(
                                series_metadata.get('seasonal_adjustment')
                            ),
                            realtime_start=None,
                            realtime_end=None
                        )
                        data_points.append(data_point)
                        
                except Exception as e:
                    logger.warning(f"Failed to normalize BLS data point {row}: {str(e)}")
                    continue
            
            logger.info(f"Normalized {len(data_points)} BLS data points")
            return data_points
            
        except Exception as e:
            logger.error(f"BLS data normalization failed: {str(e)}")
            raise DataNormalizationError(f"Failed to normalize BLS data: {str(e)}")
    
    @staticmethod
    def normalize_indicator_metadata(source: DataSource, raw_metadata: Dict[str, Any]) -> UnifiedIndicator:
        """
        Normalize indicator metadata from different sources
        
        Args:
            source: Data source
            raw_metadata: Raw metadata from source
            
        Returns:
            Normalized indicator metadata
        """
        try:
            if source == DataSource.BEA:
                return DataNormalizer._normalize_bea_indicator(raw_metadata)
            elif source == DataSource.FRED:
                return DataNormalizer._normalize_fred_indicator(raw_metadata)
            elif source == DataSource.BLS:
                return DataNormalizer._normalize_bls_indicator(raw_metadata)
            else:
                raise DataNormalizationError(f"Unsupported source: {source}")
                
        except Exception as e:
            logger.error(f"Indicator metadata normalization failed for {source}: {str(e)}")
            raise DataNormalizationError(f"Failed to normalize indicator metadata: {str(e)}")
    
    @staticmethod
    def create_unified_response(source: DataSource, series_id: str, 
                              data_points: List[UnifiedDataPoint],
                              raw_metadata: Dict[str, Any],
                              query_params: Optional[Dict[str, Any]] = None) -> UnifiedSeriesResponse:
        """
        Create a unified series response
        
        Args:
            source: Data source
            series_id: Series identifier
            data_points: Normalized data points
            raw_metadata: Raw metadata from source
            query_params: Query parameters used
            
        Returns:
            Unified series response
        """
        try:
            # Normalize indicator metadata
            indicator = DataNormalizer.normalize_indicator_metadata(source, raw_metadata)
            
            # Temporal coverage determined from data points if needed
            # start_date = min(dp.timestamp.date() for dp in data_points) if data_points else None
            # end_date = max(dp.timestamp.date() for dp in data_points) if data_points else None
            
            # Create response
            response = UnifiedSeriesResponse(
                series_id=series_id,
                indicator=indicator,
                data_points=data_points,
                total_count=len(data_points),
                returned_count=len(data_points),
                query_start_date=query_params.get('start_date') if query_params else None,
                query_end_date=query_params.get('end_date') if query_params else None,
                frequency_filter=parse_frequency_from_source(source, query_params.get('frequency') if query_params else None),
                units_filter=query_params.get('units') if query_params else None,
                source=source
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to create unified response: {str(e)}")
            raise DataNormalizationError(f"Failed to create unified response: {str(e)}")
    
    # Helper methods for source-specific parsing
    
    @staticmethod
    def _parse_fred_seasonal_adjustment(sa_short: Optional[str]) -> Optional[SeasonalAdjustment]:
        """Parse FRED seasonal adjustment code"""
        if not sa_short:
            return None
        
        sa_short = sa_short.upper()
        if sa_short == 'SA':
            return SeasonalAdjustment.SEASONALLY_ADJUSTED
        elif sa_short == 'NSA':
            return SeasonalAdjustment.NOT_SEASONALLY_ADJUSTED
        
        return None
    
    @staticmethod
    def _parse_fred_datetime(date_str: Optional[str]) -> Optional[datetime]:
        """Parse FRED datetime string"""
        if not date_str:
            return None
        
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return None
    
    @staticmethod
    def _parse_bls_date(year: int, period: str) -> datetime:
        """Parse BLS year and period into datetime"""
        if period.startswith('M'):  # Monthly
            month = int(period[1:])
            return datetime(year, month, 1)
        elif period.startswith('Q'):  # Quarterly
            quarter = int(period[1:])
            month = quarter * 3 - 2  # First month of quarter
            return datetime(year, month, 1)
        elif period == 'A01':  # Annual
            return datetime(year, 12, 31)
        else:
            # Default to first day of year
            return datetime(year, 1, 1)
    
    @staticmethod
    def _parse_bls_frequency(period: str) -> Optional[DataFrequency]:
        """Parse BLS period to frequency"""
        if period.startswith('M'):
            return DataFrequency.MONTHLY
        elif period.startswith('Q'):
            return DataFrequency.QUARTERLY
        elif period.startswith('A'):
            return DataFrequency.ANNUAL
        elif period.startswith('S'):
            return DataFrequency.SEMIANNUAL
        
        return None
    
    @staticmethod
    def _parse_bls_seasonal_adjustment(sa_code: Optional[str]) -> Optional[SeasonalAdjustment]:
        """Parse BLS seasonal adjustment code"""
        if not sa_code:
            return None
        
        sa_code = sa_code.upper()
        if sa_code == 'S':
            return SeasonalAdjustment.SEASONALLY_ADJUSTED
        elif sa_code == 'U':
            return SeasonalAdjustment.NOT_SEASONALLY_ADJUSTED
        
        return None
    
    @staticmethod
    def _normalize_bea_indicator(metadata: Dict[str, Any]) -> UnifiedIndicator:
        """Normalize BEA indicator metadata"""
        return UnifiedIndicator(
            series_id=metadata.get('SeriesID', ''),
            title=metadata.get('SeriesDescription', ''),
            description=metadata.get('SeriesDescription', ''),
            source=DataSource.BEA,
            indicator_type=DataNormalizer._map_bea_indicator_type(metadata.get('TableName', '')),
            frequency=DataNormalizer._map_bea_frequency(metadata.get('Frequency', '')),
            units=metadata.get('UnitName'),
            seasonal_adjustment=SeasonalAdjustment.NOT_SEASONALLY_ADJUSTED,
            start_date=None,
            end_date=None,
            geography_level=None,
            notes=None,
            source_specific=metadata
        )
    
    @staticmethod
    def _normalize_fred_indicator(metadata: Dict[str, Any]) -> UnifiedIndicator:
        """Normalize FRED indicator metadata"""
        return UnifiedIndicator(
            series_id=metadata.get('id', ''),
            title=metadata.get('title', ''),
            description=metadata.get('description', ''),
            source=DataSource.FRED,
            indicator_type=DataNormalizer._map_fred_indicator_type(metadata.get('notes', '')),
            frequency=DataNormalizer._map_fred_frequency(metadata.get('frequency_short', '')),
            units=metadata.get('units_short'),
            seasonal_adjustment=DataNormalizer._parse_fred_seasonal_adjustment(metadata.get('seasonal_adjustment_short')),
            start_date=None,
            end_date=None,
            geography_level=None,
            notes=metadata.get('notes'),
            source_specific=metadata
        )
    
    @staticmethod
    def _normalize_bls_indicator(metadata: Dict[str, Any]) -> UnifiedIndicator:
        """Normalize BLS indicator metadata"""
        return UnifiedIndicator(
            series_id=metadata.get('series_id', ''),
            title=metadata.get('series_title', ''),
            description=metadata.get('series_title', ''),
            source=DataSource.BLS,
            indicator_type=DataNormalizer._map_bls_indicator_type(metadata.get('series_title', '')),
            frequency=DataNormalizer._map_bls_frequency(metadata.get('periodicity_code', '')),
            units=metadata.get('units'),
            seasonal_adjustment=DataNormalizer._parse_bls_seasonal_adjustment(metadata.get('seasonal_adjustment_code')),
            start_date=None,
            end_date=None,
            geography_level=None,
            notes=None,
            source_specific=metadata
        )
    
    # Indicator type mapping methods
    
    @staticmethod
    def _map_bea_indicator_type(table_name: str) -> EconomicIndicatorType:
        """Map BEA table name to indicator type"""
        table_name = table_name.upper()
        
        if 'GDP' in table_name or 'GROSSDOMESTIC' in table_name:
            return EconomicIndicatorType.GDP
        elif 'PERSONALINCOME' in table_name:
            return EconomicIndicatorType.PERSONAL_INCOME
        elif 'PCE' in table_name or 'CONSUMERSPENDING' in table_name:
            return EconomicIndicatorType.CONSUMER_SPENDING
        
        return EconomicIndicatorType.GDP  # Default
    
    @staticmethod
    def _map_fred_indicator_type(notes: str) -> EconomicIndicatorType:
        """Map FRED notes to indicator type"""
        notes = notes.upper() if notes else ''
        
        if 'GDP' in notes or 'GROSS DOMESTIC PRODUCT' in notes:
            return EconomicIndicatorType.GDP
        elif 'CPI' in notes or 'CONSUMER PRICE INDEX' in notes:
            return EconomicIndicatorType.CPI
        elif 'UNEMPLOYMENT' in notes:
            return EconomicIndicatorType.UNEMPLOYMENT
        elif 'INTEREST RATE' in notes or 'FEDERAL FUNDS' in notes:
            return EconomicIndicatorType.INTEREST_RATE
        elif 'PERSONAL INCOME' in notes:
            return EconomicIndicatorType.PERSONAL_INCOME
        elif 'CONSUMER SPENDING' in notes or 'PCE' in notes:
            return EconomicIndicatorType.CONSUMER_SPENDING
        
        return EconomicIndicatorType.GDP  # Default
    
    @staticmethod
    def _map_bls_indicator_type(title: str) -> EconomicIndicatorType:
        """Map BLS title to indicator type"""
        title = title.upper() if title else ''
        
        if 'CPI' in title or 'CONSUMER PRICE INDEX' in title:
            return EconomicIndicatorType.CPI
        elif 'UNEMPLOYMENT' in title:
            return EconomicIndicatorType.UNEMPLOYMENT
        elif 'EMPLOYMENT' in title or 'PAYROLL' in title:
            return EconomicIndicatorType.EMPLOYMENT
        elif 'PPI' in title or 'PRODUCER PRICE' in title:
            return EconomicIndicatorType.PPI
        
        return EconomicIndicatorType.UNEMPLOYMENT  # Default
    
    # Frequency mapping methods
    
    @staticmethod
    def _map_bea_frequency(frequency: str) -> DataFrequency:
        """Map BEA frequency to DataFrequency enum"""
        frequency = frequency.upper() if frequency else ''
        
        if frequency == 'A' or 'ANNUAL' in frequency:
            return DataFrequency.ANNUAL
        elif frequency == 'Q' or 'QUARTERLY' in frequency:
            return DataFrequency.QUARTERLY
        elif frequency == 'M' or 'MONTHLY' in frequency:
            return DataFrequency.MONTHLY
        
        return DataFrequency.ANNUAL  # Default
    
    @staticmethod
    def _map_fred_frequency(frequency: str) -> DataFrequency:
        """Map FRED frequency to DataFrequency enum"""
        frequency = frequency.upper() if frequency else ''
        
        if frequency == 'A':
            return DataFrequency.ANNUAL
        elif frequency == 'Q':
            return DataFrequency.QUARTERLY
        elif frequency == 'M':
            return DataFrequency.MONTHLY
        elif frequency == 'W':
            return DataFrequency.WEEKLY
        elif frequency == 'D':
            return DataFrequency.DAILY
        
        return DataFrequency.ANNUAL  # Default
    
    @staticmethod
    def _map_bls_frequency(periodicity: str) -> DataFrequency:
        """Map BLS periodicity to DataFrequency enum"""
        periodicity = periodicity.upper() if periodicity else ''
        
        if periodicity == 'A':
            return DataFrequency.ANNUAL
        elif periodicity == 'Q':
            return DataFrequency.QUARTERLY
        elif periodicity == 'M':
            return DataFrequency.MONTHLY
        elif periodicity == 'S':
            return DataFrequency.SEMIANNUAL
        
        return DataFrequency.MONTHLY  # Default


# Singleton instance
data_normalizer = DataNormalizer()