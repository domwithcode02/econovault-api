"""
Unified Data Models for Economic Data Sources

This module provides standardized data models that can be used across
BLS, BEA, and FRED data sources to ensure consistent API responses.
"""

from datetime import datetime, date
from typing import List, Optional, Dict, Union, Any
from decimal import Decimal

from pydantic import BaseModel, Field

from models import (
    DataFrequency, SeasonalAdjustment, DataSource, 
    EconomicIndicatorType
)


class UnifiedDataPoint(BaseModel):
    """Standardized data point model for all economic data sources"""
    
    series_id: str = Field(..., description="Unique series identifier")
    timestamp: datetime = Field(..., description="Data point timestamp")
    value: Union[float, int, str, Decimal] = Field(..., description="Data value")
    source: DataSource = Field(..., description="Data source (BLS, BEA, FRED)")
    frequency: Optional[DataFrequency] = Field(None, description="Data frequency")
    units: Optional[str] = Field(None, description="Units of measurement")
    seasonal_adjustment: Optional[SeasonalAdjustment] = Field(None, description="Seasonal adjustment")
    realtime_start: Optional[datetime] = Field(None, description="Real-time period start")
    realtime_end: Optional[datetime] = Field(None, description="Real-time period end")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: float(v) if v else None
        }


class UnifiedIndicator(BaseModel):
    """Standardized indicator metadata model for all economic data sources"""
    
    series_id: str = Field(..., description="Unique series identifier")
    title: str = Field(..., description="Indicator title/name")
    description: str = Field(..., description="Indicator description")
    source: DataSource = Field(..., description="Data source (BLS, BEA, FRED)")
    indicator_type: EconomicIndicatorType = Field(..., description="Type of economic indicator")
    frequency: DataFrequency = Field(..., description="Data frequency")
    units: Optional[str] = Field(None, description="Units of measurement")
    seasonal_adjustment: Optional[SeasonalAdjustment] = Field(None, description="Seasonal adjustment")
    
    # Source-specific metadata
    source_specific: Dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata")
    
    # Temporal coverage
    start_date: Optional[date] = Field(None, description="Earliest available data date")
    end_date: Optional[date] = Field(None, description="Latest available data date")
    
    # Additional metadata
    geography_level: Optional[str] = Field(None, description="Geographic level (NATIONAL, STATE, etc.)")
    notes: Optional[str] = Field(None, description="Additional notes about the indicator")
    
    class Config:
        json_encoders = {
            date: lambda v: v.isoformat() if v else None
        }


class UnifiedSeriesResponse(BaseModel):
    """Standardized response for series data queries"""
    
    series_id: str = Field(..., description="Series identifier")
    indicator: UnifiedIndicator = Field(..., description="Series metadata")
    data_points: List[UnifiedDataPoint] = Field(..., description="Data points")
    total_count: int = Field(..., description="Total number of data points available")
    returned_count: int = Field(..., description="Number of data points returned")
    
    # Query metadata
    query_start_date: Optional[date] = Field(None, description="Query start date")
    query_end_date: Optional[date] = Field(None, description="Query end date")
    frequency_filter: Optional[DataFrequency] = Field(None, description="Applied frequency filter")
    units_filter: Optional[str] = Field(None, description="Applied units transformation")
    
    # Source metadata
    source: DataSource = Field(..., description="Data source")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Response generation time")
    
    class Config:
        json_encoders = {
            date: lambda v: v.isoformat() if v else None,
            datetime: lambda v: v.isoformat() if v else None
        }


class UnifiedMultiSeriesResponse(BaseModel):
    """Standardized response for multi-series data queries"""
    
    series_responses: List[UnifiedSeriesResponse] = Field(..., description="Individual series responses")
    total_series: int = Field(..., description="Total number of series requested")
    total_data_points: int = Field(..., description="Total data points across all series")
    
    # Query metadata
    query_params: Dict[str, Any] = Field(default_factory=dict, description="Applied query parameters")
    
    # Response metadata
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Response generation time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class DataValidationError(Exception):
    """Exception raised when data validation fails during normalization"""
    pass


class DataNormalizationError(Exception):
    """Exception raised when data normalization fails"""
    pass


# Source-specific data structures for internal processing
class BEARawDataPoint(BaseModel):
    """Raw BEA data point structure"""
    TimePeriod: str = Field(..., description="Time period (e.g., '2023Q1', '2023')")
    DataValue: str = Field(..., description="Data value as string")
    YEAR: Optional[str] = Field(None, description="Year")
    QUARTER: Optional[str] = Field(None, description="Quarter")
    MONTH: Optional[str] = Field(None, description="Month")
    DAY: Optional[str] = Field(None, description="Day")
    METRIC_NAME: Optional[str] = Field(None, description="Metric name")


class FREDRawDataPoint(BaseModel):
    """Raw FRED data point structure"""
    realtime_start: str = Field(..., description="Real-time start date")
    realtime_end: str = Field(..., description="Real-time end date")
    date: str = Field(..., description="Observation date")
    value: str = Field(..., description="Data value as string")


class BLSRawDataPoint(BaseModel):
    """Raw BLS data point structure"""
    year: str = Field(..., description="Year")
    period: str = Field(..., description="Period (e.g., 'M01', 'Q1')")
    periodName: str = Field(..., description="Period name (e.g., 'January', 'Q1')")
    value: str = Field(..., description="Data value as string")
    latest: str = Field(..., description="Whether this is the latest data point")


# Utility functions for data validation
def validate_numeric_value(value: Union[str, float, int, Decimal]) -> Union[float, int, Decimal, None]:
    """Validate and convert numeric values"""
    if isinstance(value, (int, float, Decimal)):
        return value
    
    if isinstance(value, str):
        try:
            # Handle common BEA/FRED notations
            if value.strip() in ['', '..', 'NA', 'N/A']:
                return None
            
            # Remove commas and whitespace
            cleaned = value.replace(',', '').strip()
            
            # Try integer first, then float
            try:
                return int(cleaned)
            except ValueError:
                return float(cleaned)
                
        except (ValueError, TypeError):
            return None
    
    return None


def parse_frequency_from_source(source: DataSource, frequency_str: Optional[str] = None) -> Optional[DataFrequency]:
    """Parse frequency from source-specific string"""
    if not frequency_str:
        return None
    
    frequency_str = frequency_str.upper().strip()
    
    # Common frequency mappings
    frequency_map = {
        'DAILY': DataFrequency.DAILY,
        'D': DataFrequency.DAILY,
        'WEEKLY': DataFrequency.WEEKLY,
        'W': DataFrequency.WEEKLY,
        'BIWEEKLY': DataFrequency.BIWEEKLY,
        'BW': DataFrequency.BIWEEKLY,
        'MONTHLY': DataFrequency.MONTHLY,
        'M': DataFrequency.MONTHLY,
        'QUARTERLY': DataFrequency.QUARTERLY,
        'Q': DataFrequency.QUARTERLY,
        'SEMIANNUAL': DataFrequency.SEMIANNUAL,
        'SA': DataFrequency.SEMIANNUAL,
        'ANNUAL': DataFrequency.ANNUAL,
        'A': DataFrequency.ANNUAL,
        'YEARLY': DataFrequency.ANNUAL,
        'Y': DataFrequency.ANNUAL
    }
    
    return frequency_map.get(frequency_str)


def parse_date_from_source(source: DataSource, date_str: str) -> datetime:
    """Parse date from source-specific format"""
    if source == DataSource.BEA:
        # BEA formats: '2023', '2023Q1', '2023M01'
        if len(date_str) == 4:  # Year only
            return datetime(int(date_str), 12, 31)  # End of year
        elif 'Q' in date_str:  # Quarterly
            year, quarter = date_str.split('Q')
            quarter_end_month = int(quarter) * 3
            return datetime(int(year), quarter_end_month, 1)  # First day of quarter end month
        elif 'M' in date_str:  # Monthly
            year, month = date_str.split('M')
            return datetime(int(year), int(month), 1)
    
    elif source == DataSource.FRED:
        # FRED format: YYYY-MM-DD
        return datetime.strptime(date_str, '%Y-%m-%d')
    
    elif source == DataSource.BLS:
        # BLS has separate year and period, handled in client
        pass
    
    # Default parsing attempt
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        raise DataValidationError(f"Unable to parse date '{date_str}' for source {source}")