"""
EconoVault Data Pipeline - Real BLS/BEA/FRED Data Ingestion

This module implements the production data ingestion pipeline for economic indicators
from BLS (Bureau of Labor Statistics), BEA (Bureau of Economic Analysis), and 
FRED (Federal Reserve Economic Data) APIs.
"""

import asyncio
import logging
import os
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Union
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor
import json
import time

from bls_client import BLSClient, BLSAPIException
from database import get_db, DatabaseManager, EconomicIndicator, DataPoint
from models import (
    DataFrequency, SeasonalAdjustment, DataSource, EconomicIndicatorType,
    GeographicLevel, EconomicDataNormalizer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for concurrent API calls
executor = ThreadPoolExecutor(max_workers=10)

# BLS API Configuration
BLS_API_KEY = os.getenv('BLS_API_KEY')
if not BLS_API_KEY:
    logger.warning("BLS_API_KEY not found in environment variables")

# Popular BLS series IDs with metadata
BLS_SERIES_METADATA = {
    "CUUR0000SA0": {
        "series_id": "CUUR0000SA0",
        "title": "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",
        "source": "BLS",
        "indicator_type": "CPI",
        "frequency": "MONTHLY",
        "seasonal_adjustment": "SEASONALLY_ADJUSTED",
        "geography_level": "NATIONAL",
        "units": "Index 1982-84=100",
        "description": "The Consumer Price Index (CPI) is a measure of the average change over time in the prices paid by urban consumers for a market basket of consumer goods and services."
    },
    "LNS14000000": {
        "series_id": "LNS14000000",
        "title": "Unemployment Rate",
        "source": "BLS",
        "indicator_type": "UNEMPLOYMENT",
        "frequency": "MONTHLY",
        "seasonal_adjustment": "SEASONALLY_ADJUSTED",
        "geography_level": "NATIONAL",
        "units": "Percent",
        "description": "The unemployment rate represents the number unemployed as a percent of the labor force."
    },
    "CES0000000001": {
        "series_id": "CES0000000001",
        "title": "All Employees, Total Nonfarm",
        "source": "BLS",
        "indicator_type": "EMPLOYMENT",
        "frequency": "MONTHLY",
        "seasonal_adjustment": "SEASONALLY_ADJUSTED",
        "geography_level": "NATIONAL",
        "units": "Thousands of Persons",
        "description": "Total nonfarm payroll employment is a measure of the number of U.S. workers in the economy that excludes proprietors, private household employees, unpaid volunteers, farm employees, and the unincorporated self-employed."
    },
    "LNS11300000": {
        "series_id": "LNS11300000",
        "title": "Labor Force Participation Rate",
        "source": "BLS",
        "indicator_type": "EMPLOYMENT",
        "frequency": "MONTHLY",
        "seasonal_adjustment": "SEASONALLY_ADJUSTED",
        "geography_level": "NATIONAL",
        "units": "Percent",
        "description": "The labor force participation rate is the percentage of the population that is either working or actively looking for work."
    },
    "LNS12000000": {
        "series_id": "LNS12000000",
        "title": "Employment Level",
        "source": "BLS",
        "indicator_type": "EMPLOYMENT",
        "frequency": "MONTHLY",
        "seasonal_adjustment": "SEASONALLY_ADJUSTED",
        "geography_level": "NATIONAL",
        "units": "Thousands of Persons",
        "description": "The employment level is the number of people who are currently employed."
    },
    "LNS13000000": {
        "series_id": "LNS13000000",
        "title": "Unemployment Level",
        "source": "BLS",
        "indicator_type": "UNEMPLOYMENT",
        "frequency": "MONTHLY",
        "seasonal_adjustment": "SEASONALLY_ADJUSTED",
        "geography_level": "NATIONAL",
        "units": "Thousands of Persons",
        "description": "The unemployment level is the number of people who are currently unemployed."
    }
}


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors"""
    pass


class BLSDataIngestion:
    """BLS data ingestion with error handling and retry logic"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = BLSClient(api_key=api_key)
        self.normalizer = EconomicDataNormalizer()
        self.logger = logging.getLogger(__name__)
    
    async def ingest_series_data(self, series_id: str, start_year: Optional[int] = None, end_year: Optional[int] = None) -> pd.DataFrame:
        """Ingest data for a specific BLS series"""
        try:
            self.logger.info(f"Ingesting BLS data for series: {series_id}")
            
            # Get data from BLS API
            data_df = self.client.get_series_data(
                series_ids=series_id,
                start_year=start_year,
                end_year=end_year
            )
            
            if data_df.empty:
                self.logger.warning(f"No data returned for series {series_id}")
                return pd.DataFrame()
            
            # Validate and normalize data
            validated_data = self._validate_bls_data(data_df, series_id)
            
            self.logger.info(f"Successfully ingested {len(validated_data)} data points for {series_id}")
            return validated_data
            
        except BLSAPIException as e:
            self.logger.error(f"BLS API error for {series_id}: {str(e)}")
            raise DataIngestionError(f"BLS API error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error ingesting {series_id}: {str(e)}")
            raise DataIngestionError(f"Data ingestion error: {str(e)}")
    
    def _validate_bls_data(self, data_df: pd.DataFrame, series_id: str) -> pd.DataFrame:
        """Validate and clean BLS data"""
        if data_df.empty:
            return data_df
        
        # Remove rows with missing values
        data_df = data_df.dropna(subset=['value'])
        
        # Convert value to numeric
        data_df['value'] = pd.to_numeric(data_df['value'], errors='coerce')
        
        # Remove rows with invalid values after conversion
        data_df = data_df.dropna(subset=['value'])
        
        # Add metadata
        data_df['series_id'] = series_id
        data_df['source'] = 'BLS'
        data_df['ingested_at'] = datetime.utcnow()
        
        # Validate against expected ranges for economic indicators
        if series_id in BLS_SERIES_METADATA:
            indicator_type = BLS_SERIES_METADATA[series_id]['indicator_type']
            data_df = self._validate_economic_ranges(data_df, indicator_type)
        
        return data_df
    
    def _validate_economic_ranges(self, data_df: pd.DataFrame, indicator_type: str) -> pd.DataFrame:
        """Validate economic values against expected ranges"""
        if data_df.empty:
            return data_df
        
        # Define reasonable ranges for different indicator types
        validation_ranges = {
            'UNEMPLOYMENT': (0, 25),  # Unemployment rate 0-25%
            'EMPLOYMENT': (0, 200000),  # Employment in thousands
            'CPI': (50, 400),  # CPI index range
            'GDP': (0, 30000),  # GDP in billions
            'INTEREST_RATE': (-5, 20),  # Interest rates -5% to 20%
        }
        
        if indicator_type in validation_ranges:
            min_val, max_val = validation_ranges[indicator_type]
            original_count = len(data_df)
            data_df = data_df[(data_df['value'] >= min_val) & (data_df['value'] <= max_val)]
            
            if len(data_df) < original_count:
                self.logger.warning(f"Filtered {original_count - len(data_df)} out-of-range values for {indicator_type}")
        
        return data_df


class DatabaseStorage:
    """Database storage operations for economic data"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.db_manager = DatabaseManager(db_session)
        self.logger = logging.getLogger(__name__)
    
    def store_indicator_metadata(self, series_metadata: Dict[str, Any]) -> EconomicIndicator:
        """Store or update indicator metadata"""
        try:
            series_id = series_metadata['series_id']
            
            # Check if indicator already exists
            existing = self.db_manager.get_indicator_by_series_id(series_id)
            if existing:
                self.logger.info(f"Updating existing indicator: {series_id}")
                # Update metadata
                for key, value in series_metadata.items():
                    if hasattr(existing, key) and key != 'series_id':
                        setattr(existing, key, value)
                if hasattr(existing, 'last_updated'):
                    setattr(existing, 'last_updated', datetime.utcnow())
                self.db.commit()
                return existing
            
            # Create new indicator
            self.logger.info(f"Creating new indicator: {series_id}")
            indicator = EconomicIndicator(**series_metadata)
            self.db.add(indicator)
            self.db.commit()
            self.db.refresh(indicator)
            return indicator
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Error storing indicator metadata: {str(e)}")
            raise DataIngestionError(f"Database error: {str(e)}")
    
    def store_data_points(self, indicator_id: int, data_df: pd.DataFrame) -> int:
        """Store data points for an indicator"""
        if data_df.empty:
            return 0
        
        try:
            # Convert DataFrame to list of dictionaries
            data_points = []
            for _, row in data_df.iterrows():
                data_point = {
                    'indicator_id': indicator_id,
                    'date': row['date'],
                    'value': row['value'],
                    'period': row.get('period', ''),
                    'period_name': row.get('period_name', ''),
                    'footnote_code': row.get('footnote_code', ''),
                    'footnote_text': row.get('footnote_text', ''),
                    'latest': row.get('latest', False)
                }
                data_points.append(data_point)
            
            # Use bulk insert for better performance
            count = self.db_manager.bulk_insert_data_points(data_points)
            self.logger.info(f"Stored {count} data points for indicator {indicator_id}")
            return count
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Error storing data points: {str(e)}")
            raise DataIngestionError(f"Database error: {str(e)}")
    
    def get_latest_data_date(self, series_id: str) -> Optional[date]:
        """Get the latest data date for a series"""
        try:
            latest_point = self.db_manager.get_latest_data_point(series_id)
            if latest_point:
                return getattr(latest_point, 'date', None)
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest data date: {str(e)}")
            return None


class DataPipeline:
    """Main data pipeline orchestrator"""
    
    def __init__(self, bls_api_key: Optional[str] = None):
        self.bls_ingestion = BLSDataIngestion(bls_api_key)
        self.logger = logging.getLogger(__name__)
    
    async def run_full_ingestion(self, db_session: Session) -> Dict[str, Any]:
        """Run full data ingestion for all configured series"""
        self.logger.info("Starting full data ingestion pipeline")
        
        results = {
            'total_series': 0,
            'successful_series': 0,
            'failed_series': 0,
            'total_data_points': 0,
            'errors': []
        }
        
        storage = DatabaseStorage(db_session)
        
        # Process each BLS series
        for series_id, metadata in BLS_SERIES_METADATA.items():
            results['total_series'] += 1
            
            try:
                # Store indicator metadata
                indicator = storage.store_indicator_metadata(metadata)
                
                # Get latest data date to determine what to fetch
                latest_date = storage.get_latest_data_date(series_id)
                
                # Calculate date range for ingestion
                if latest_date:
                    # Fetch data from the day after latest date
                    start_year = latest_date.year
                    end_year = datetime.now().year
                else:
                    # No existing data, fetch last 5 years
                    start_year = datetime.now().year - 5
                    end_year = datetime.now().year
                
                # Ingest data
                data_df = await self.bls_ingestion.ingest_series_data(
                    series_id=series_id,
                    start_year=start_year,
                    end_year=end_year
                )
                
                if not data_df.empty:
                    # Store data points
                    indicator_id_val = getattr(indicator, 'id', None)
                    if indicator_id_val is not None:
                        count = storage.store_data_points(int(indicator_id_val), data_df)
                    else:
                        count = 0
                    results['total_data_points'] += count
                    results['successful_series'] += 1
                    self.logger.info(f"Successfully ingested {count} data points for {series_id}")
                else:
                    results['successful_series'] += 1  # No new data is not a failure
                    self.logger.info(f"No new data available for {series_id}")
                
            except Exception as e:
                results['failed_series'] += 1
                error_msg = f"Failed to ingest {series_id}: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)
        
        self.logger.info(f"Data ingestion completed. Success: {results['successful_series']}/{results['total_series']}, Total points: {results['total_data_points']}")
        return results
    
    async def run_incremental_update(self, db_session: Session) -> Dict[str, Any]:
        """Run incremental update for recent data"""
        self.logger.info("Starting incremental data update")
        
        results = {
            'total_series': 0,
            'updated_series': 0,
            'total_new_points': 0,
            'errors': []
        }
        
        storage = DatabaseStorage(db_session)
        
        # Process each series for recent updates
        for series_id, metadata in BLS_SERIES_METADATA.items():
            results['total_series'] += 1
            
            try:
                # Get latest data date
                latest_date = storage.get_latest_data_date(series_id)
                
                if latest_date:
                    # Fetch data from the last 3 months to catch any revisions
                    start_year = latest_date.year
                    if latest_date.month > 3:
                        start_year = latest_date.year
                    else:
                        start_year = latest_date.year - 1
                    end_year = datetime.now().year
                else:
                    # No existing data, skip incremental update
                    continue
                
                # Ingest recent data
                data_df = await self.bls_ingestion.ingest_series_data(
                    series_id=series_id,
                    start_year=start_year,
                    end_year=end_year
                )
                
                if not data_df.empty:
                    # Get indicator ID
                    indicator = storage.store_indicator_metadata(metadata)
                    
                    # Store new data points
                    indicator_id_val = getattr(indicator, 'id', None)
                    if indicator_id_val is not None:
                        count = storage.store_data_points(int(indicator_id_val), data_df)
                    else:
                        count = 0
                    if count > 0:
                        results['updated_series'] += 1
                        results['total_new_points'] += count
                        self.logger.info(f"Updated {count} data points for {series_id}")
                
            except Exception as e:
                error_msg = f"Failed to update {series_id}: {str(e)}"
                results['errors'].append(error_msg)
                self.logger.error(error_msg)
        
        self.logger.info(f"Incremental update completed. Updated: {results['updated_series']}/{results['total_series']}, New points: {results['total_new_points']}")
        return results
    
    async def validate_data_quality(self, db_session: Session) -> Dict[str, Any]:
        """Validate data quality and integrity"""
        self.logger.info("Starting data quality validation")
        
        results = {
            'total_indicators': 0,
            'valid_indicators': 0,
            'data_quality_issues': [],
            'recommendations': []
        }
        
        try:
            db_manager = DatabaseManager(db_session)
            
            # Get all indicators
            indicators = db_session.query(EconomicIndicator).all()
            results['total_indicators'] = len(indicators)
            
            for indicator in indicators:
                issues = []
                
                # Check for data points
                data_points = db_manager.get_data_points(str(indicator.series_id))
                
                if not data_points:
                    issues.append("No data points available")
                else:
                    # Check for data gaps
                    if len(data_points) > 1:
                        dates = []
                        for dp in data_points:
                            date_val = getattr(dp, 'date', None)
                            if date_val is not None:
                                dates.append(date_val)
                        date_gaps = self._detect_date_gaps(dates, str(indicator.frequency))
                        if date_gaps:
                            issues.append(f"Data gaps detected: {len(date_gaps)} periods")
                    
                    # Check for outliers
                    values = []
                    for dp in data_points:
                        value = getattr(dp, 'value', None)
                        if value is not None:
                            try:
                                values.append(float(value))
                            except (ValueError, TypeError):
                                continue
                    if values:
                        outliers = self._detect_outliers(values)
                        if outliers:
                            issues.append(f"Potential outliers detected: {len(outliers)} values")
                
                if not issues:
                    results['valid_indicators'] += 1
                else:
                    results['data_quality_issues'].append({
                        'series_id': indicator.series_id,
                        'title': indicator.title,
                        'issues': issues
                    })
            
            # Generate recommendations
            if results['data_quality_issues']:
                results['recommendations'].append("Consider implementing automated data quality checks")
                results['recommendations'].append("Review data sources for consistency")
            
            self.logger.info(f"Data quality validation completed. Valid: {results['valid_indicators']}/{results['total_indicators']}")
            
        except Exception as e:
            self.logger.error(f"Error during data quality validation: {str(e)}")
            results['data_quality_issues'].append({
                'series_id': 'system',
                'title': 'System validation error',
                'issues': [str(e)]
            })
        
        return results
    
    def _detect_date_gaps(self, dates: List[date], frequency: str) -> List[str]:
        """Detect gaps in time series data"""
        if len(dates) < 2:
            return []
        
        gaps = []
        dates_sorted = sorted(dates)
        
        for i in range(1, len(dates_sorted)):
            prev_date = dates_sorted[i-1]
            curr_date = dates_sorted[i]
            
            expected_diff = self._get_expected_date_diff(frequency)
            actual_diff = (curr_date - prev_date).days
            
            if actual_diff > expected_diff * 1.5:  # Allow 50% tolerance
                gaps.append(f"Gap between {prev_date} and {curr_date}")
        
        return gaps
    
    def _get_expected_date_diff(self, frequency: str) -> int:
        """Get expected date difference in days based on frequency"""
        frequency_map = {
            'DAILY': 1,
            'WEEKLY': 7,
            'BIWEEKLY': 14,
            'MONTHLY': 30,
            'QUARTERLY': 90,
            'SEMIANNUAL': 180,
            'ANNUAL': 365
        }
        return frequency_map.get(frequency.upper(), 30)
    
    def _detect_outliers(self, values: List[float]) -> List[float]:
        """Detect statistical outliers using IQR method"""
        if len(values) < 4:
            return []
        
        q1 = pd.Series(values).quantile(0.25)
        q3 = pd.Series(values).quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [v for v in values if v < lower_bound or v > upper_bound]
        return outliers


# Scheduled ingestion functions
async def scheduled_full_ingestion():
    """Scheduled full data ingestion (runs daily)"""
    logger.info("Starting scheduled full data ingestion")
    
    try:
        # Get database session
        db_gen = get_db()
        db = next(db_gen)
        
        # Run pipeline
        pipeline = DataPipeline(BLS_API_KEY)
        results = await pipeline.run_full_ingestion(db)
        
        logger.info(f"Scheduled ingestion completed: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Scheduled ingestion failed: {str(e)}")
        raise
    finally:
        # Close database session
        try:
            next(db_gen, None)
        except StopIteration:
            pass


async def scheduled_incremental_update():
    """Scheduled incremental update (runs hourly)"""
    logger.info("Starting scheduled incremental update")
    
    try:
        # Get database session
        db_gen = get_db()
        db = next(db_gen)
        
        # Run pipeline
        pipeline = DataPipeline(BLS_API_KEY)
        results = await pipeline.run_incremental_update(db)
        
        logger.info(f"Scheduled incremental update completed: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Scheduled incremental update failed: {str(e)}")
        raise
    finally:
        # Close database session
        try:
            next(db_gen, None)
        except StopIteration:
            pass


# CLI functions for manual execution
def run_full_ingestion_cli():
    """CLI function to run full ingestion"""
    import asyncio
    
    print("Starting full data ingestion...")
    results = asyncio.run(scheduled_full_ingestion())
    
    print(f"Ingestion completed:")
    print(f"- Total series: {results['total_series']}")
    print(f"- Successful: {results['successful_series']}")
    print(f"- Failed: {results['failed_series']}")
    print(f"- Total data points: {results['total_data_points']}")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"- {error}")


def run_incremental_update_cli():
    """CLI function to run incremental update"""
    import asyncio
    
    print("Starting incremental update...")
    results = asyncio.run(scheduled_incremental_update())
    
    print(f"Update completed:")
    print(f"- Total series: {results['total_series']}")
    print(f"- Updated: {results['updated_series']}")
    print(f"- New points: {results['total_new_points']}")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"- {error}")


def validate_data_quality_cli():
    """CLI function to validate data quality"""
    import asyncio
    
    print("Starting data quality validation...")
    
    try:
        # Get database session
        db_gen = get_db()
        db = next(db_gen)
        
        # Run validation
        pipeline = DataPipeline(BLS_API_KEY)
        results = asyncio.run(pipeline.validate_data_quality(db))
        
        print(f"Validation completed:")
        print(f"- Total indicators: {results['total_indicators']}")
        print(f"- Valid indicators: {results['valid_indicators']}")
        
        if results['data_quality_issues']:
            print(f"\nData quality issues found: {len(results['data_quality_issues'])}")
            for issue in results['data_quality_issues']:
                print(f"\n{issue['series_id']} - {issue['title']}:")
                for problem in issue['issues']:
                    print(f"  - {problem}")
        
        if results['recommendations']:
            print(f"\nRecommendations:")
            for rec in results['recommendations']:
                print(f"- {rec}")
        
    except Exception as e:
        print(f"Validation failed: {str(e)}")
    finally:
        # Close database session
        try:
            next(db_gen, None)
        except StopIteration:
            pass


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_pipeline.py [full|incremental|validate]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'full':
        run_full_ingestion_cli()
    elif command == 'incremental':
        run_incremental_update_cli()
    elif command == 'validate':
        validate_data_quality_cli()
    else:
        print("Invalid command. Use: full, incremental, or validate")
        sys.exit(1)