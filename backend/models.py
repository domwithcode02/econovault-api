from enum import Enum
from datetime import date, datetime
from typing import List, Optional, Dict, Union
from decimal import Decimal
from pydantic import BaseModel, Field, validator


class DataFrequency(str, Enum):
    DAILY = "D"
    WEEKLY = "W"
    BIWEEKLY = "BW"
    MONTHLY = "M"
    QUARTERLY = "Q"
    SEMIANNUAL = "SA"
    ANNUAL = "A"


class SeasonalAdjustment(str, Enum):
    SEASONALLY_ADJUSTED = "SA"
    NOT_SEASONALLY_ADJUSTED = "NSA"
    SEASONAL_DIFFERENCE = "SD"


class DataSource(str, Enum):
    BLS = "BLS"
    BEA = "BEA"
    FRED = "FRED"
    CENSUS = "CENSUS"
    TREASURY = "TREASURY"


class Units(str, Enum):
    LEVELS = "lin"
    CHANGE = "chg"
    PERCENT_CHANGE = "pch"
    PERCENT_CHANGE_YOY = "pc1"
    COMPOUNDED_ANNUAL = "pca"
    LOG = "log"


class EconomicIndicatorType(str, Enum):
    GDP = "GDP"
    CPI = "CPI"
    UNEMPLOYMENT = "UNEMPLOYMENT"
    EMPLOYMENT = "EMPLOYMENT"
    INTEREST_RATE = "INTEREST_RATE"
    PPI = "PPI"
    PERSONAL_INCOME = "PERSONAL_INCOME"
    CONSUMER_SPENDING = "CONSUMER_SPENDING"


class GeographicLevel(str, Enum):
    NATIONAL = "NATIONAL"
    STATE = "STATE"
    METRO = "METRO"
    COUNTY = "COUNTY"


class DataQuality(str, Enum):
    PRELIMINARY = "P"
    REVISED = "R"
    FINAL = "F"
    SUPPRESSED = "S"


class Footnote(BaseModel):
    code: str
    text: str


class DataPoint(BaseModel):
    date: date
    value: Optional[Decimal]
    period: Optional[str] = Field(None, description="Period identifier (e.g., M01, Q1)")
    period_name: Optional[str] = Field(None, description="Period name (e.g., January, Q1)")
    footnotes: List[Footnote] = Field(default_factory=list)
    realtime_start: Optional['date'] = None
    realtime_end: Optional['date'] = None
    
    @validator('value')
    def validate_value(cls, v):
        if v is not None and v < 0:
            return v
        return v


class SeriesMetadata(BaseModel):
    series_id: str
    title: str
    description: Optional[str] = None
    source: DataSource
    indicator_type: EconomicIndicatorType
    frequency: DataFrequency
    seasonal_adjustment: SeasonalAdjustment
    geography: Optional[str] = None
    geography_level: GeographicLevel
    units: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    last_updated: Optional[datetime] = None
    registration_key_required: bool = False


class TimeSeriesData(BaseModel):
    series_metadata: SeriesMetadata
    data_points: List[DataPoint]
    
    @validator('data_points')
    def validate_data_points(cls, v):
        if not v:
            return v
        return sorted(v, key=lambda x: x.date)


class BLSSeriesData(BaseModel):
    series_id: str
    catalog: Optional[Dict] = None
    data: List[DataPoint]
    aspects: Optional[List[Dict]] = None
    
    @validator('series_id')
    def validate_bls_series_id(cls, v):
        if any(c.islower() for c in v):
            raise ValueError('BLS series ID cannot contain lowercase letters')
        return v


class BEADataPoint(BaseModel):
    year: int
    quarter: Optional[int] = None
    period: Optional[str] = None
    value: Decimal
    data_value: Optional[str] = None
    note_ref: Optional[str] = None
    
    @validator('year')
    def validate_year(cls, v):
        if v < 1900 or v > datetime.now().year + 5:
            raise ValueError('Year must be reasonable for economic data')
        return v


class FREDSeriesData(BaseModel):
    series_id: str
    observations: List[Dict[str, Optional[str]]]
    realtime_start: Optional[date] = None
    realtime_end: Optional[date] = None
    count: int = Field(ge=0)
    
    @validator('observations')
    def validate_observations(cls, v):
        for obs in v:
            if 'date' not in obs:
                raise ValueError('Each observation must have a date')
        return v


class GDPData(TimeSeriesData):
    gdp_type: str = Field(..., description="Nominal, Real, or Per Capita")
    base_year: Optional[int] = Field(None, description="Base year for real GDP")
    component: Optional[str] = Field(None, description="GDP component (C, I, G, NX)")
    
    @validator('gdp_type')
    def validate_gdp_type(cls, v):
        valid_types = ['NOMINAL', 'REAL', 'PER_CAPITA']
        if v.upper() not in valid_types:
            raise ValueError(f'GDP type must be one of {valid_types}')
        return v.upper()


class CPIData(TimeSeriesData):
    base_period: str = Field(..., description="Base period for index (e.g., 1982-84=100)")
    item_code: Optional[str] = Field(None, description="CPI item code")
    area_code: Optional[str] = Field(None, description="Geographic area code")
    
    @validator('base_period')
    def validate_base_period(cls, v):
        if '=' not in v:
            raise ValueError('Base period must include index value (e.g., 1982-84=100)')
        return v


class UnemploymentData(TimeSeriesData):
    unemployment_type: str = Field(..., description="Type of unemployment rate")
    demographic_group: Optional[str] = None
    
    @validator('unemployment_type')
    def validate_unemployment_type(cls, v):
        valid_types = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'TOTAL', 'WHITE', 'BLACK', 'HISPANIC']
        if v.upper() not in valid_types:
            raise ValueError(f'Unemployment type must be one of {valid_types}')
        return v.upper()


class InterestRateData(TimeSeriesData):
    rate_type: str = Field(..., description="Type of interest rate")
    maturity: Optional[str] = Field(None, description="Maturity period")
    instrument: Optional[str] = Field(None, description="Financial instrument")
    
    @validator('rate_type')
    def validate_rate_type(cls, v):
        valid_types = ['FEDERAL_FUNDS', 'PRIME', 'TREASURY', 'LIBOR', 'DISCOUNT']
        if v.upper() not in valid_types:
            raise ValueError(f'Rate type must be one of {valid_types}')
        return v.upper()


class EconomicDataNormalizer:
    """Utility class for normalizing economic data from different sources"""
    
    @staticmethod
    def normalize_date(date_input: Union[str, date, datetime], frequency: DataFrequency) -> date:
        """Normalize date to appropriate format based on frequency"""
        if isinstance(date_input, str):
            date_formats = ['%Y-%m-%d', '%Y-%m', '%Y', '%m/%d/%Y', '%m/%Y']
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_input, fmt).date()
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Cannot parse date: {date_input}")
        elif isinstance(date_input, datetime):
            parsed_date = date_input.date()
        elif isinstance(date_input, date):
            parsed_date = date_input
        else:
            raise ValueError(f"Unsupported date type: {type(date_input)}")
        
        # Adjust date based on frequency
        if frequency == DataFrequency.ANNUAL:
            return date(parsed_date.year, 1, 1)
        elif frequency == DataFrequency.QUARTERLY:
            quarter_months = [1, 4, 7, 10]
            quarter = (parsed_date.month - 1) // 3
            return date(parsed_date.year, quarter_months[quarter], 1)
        elif frequency == DataFrequency.MONTHLY:
            return date(parsed_date.year, parsed_date.month, 1)
        
        return parsed_date
    
    @staticmethod
    def normalize_value(value: Union[str, float, Decimal], 
                       indicator_type: EconomicIndicatorType) -> Optional[Decimal]:
        """Normalize value based on indicator type"""
        if value is None or str(value).strip() == '':
            return None
        
        if isinstance(value, str):
            cleaned_value = value.replace(',', '').replace('$', '').replace('%', '').strip()
            if cleaned_value == '.':
                return None
            try:
                return Decimal(cleaned_value)
            except ValueError:
                return None
        
        return Decimal(str(value))
    
    @staticmethod
    def validate_economic_value(value: Decimal, indicator_type: EconomicIndicatorType) -> bool:
        """Validate economic value based on indicator type"""
        if value is None:
            return True
        
        validation_rules = {
            EconomicIndicatorType.GDP: lambda x: x >= 0,
            EconomicIndicatorType.CPI: lambda x: x > 0,
            EconomicIndicatorType.UNEMPLOYMENT: lambda x: 0 <= x <= 100,
            EconomicIndicatorType.EMPLOYMENT: lambda x: x >= 0,
            EconomicIndicatorType.INTEREST_RATE: lambda x: -50 <= x <= 100,
            EconomicIndicatorType.PPI: lambda x: x > 0,
        }
        
        rule = validation_rules.get(indicator_type)
        if rule and not rule(value):
            return False
        
        return True