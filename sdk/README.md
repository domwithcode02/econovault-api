# EconoVault Python SDK

[![PyPI version](https://badge.fury.io/py/econovault.svg)](https://badge.fury.io/py/econovault)
[![Python versions](https://img.shields.io/pypi/pyversions/econovault.svg)](https://pypi.org/project/econovault/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python SDK for interacting with the EconoVault API, providing easy access to economic indicators, real-time data streaming, and advanced analytics features.

## Features

- **Easy Integration**: Simple, intuitive API for accessing economic data
- **Real-time Data**: Stream live economic indicators with Server-Sent Events
- **Comprehensive Coverage**: Access data from BLS, BEA, and Federal Reserve
- **Async Support**: Full async/await support for high-performance applications
- **Type Safety**: Full type hints and data validation
- **Error Handling**: Robust error handling with detailed exceptions
- **Rate Limiting**: Built-in rate limiting and retry logic
- **GDPR Compliant**: Privacy-focused with data export and deletion features

## Installation

```bash
pip install econovault
```

For development with additional dependencies:

```bash
pip install econovault[dev,pandas,numpy]
```

## Quick Start

### Synchronous Client

```python
from econovault import get_client

# Initialize client
with get_client("your-api-key") as client:
    # Get available indicators
    indicators = client.get_indicators(source="BLS", limit=10)
    print(f"Found {len(indicators)} BLS indicators")
    
    # Get CPI data using semantic endpoint (recommended)
    cpi_data = client.get_indicator_data("CUUR0000SA0", limit=12)  # Maps to consumer-price-index
    print(f"Latest CPI: {cpi_data.data[-1].value}")
    
    # Get unemployment rate
    unemployment = client.get_indicator("LNS14000000")  # Maps to unemployment-rate
    print(f"Unemployment Rate: {unemployment.title}")
    
    # Alternative: Use semantic names directly via requests
    import requests
    semantic_response = requests.get(
        f"{client.base_url}/v1/indicators/consumer-price-index/data",
        headers={"Authorization": f"Bearer {client.api_key}"}
    )
    print(f"Semantic CPI: {semantic_response.json()}")
```

### Asynchronous Client

```python
import asyncio
from econovault import get_async_client

async def main():
    async with get_async_client("your-api-key") as client:
        # Get indicators asynchronously
        indicators = await client.get_indicators(source="BLS", limit=5)
        
        # Stream real-time data
        async for update in client.stream_indicator_data("CUUR0000SA0", update_interval=60):
            print(f"CPI Update: {update['timestamp']} - {update['data']['value']}")

asyncio.run(main())
```

## API Reference

### Client Initialization

```python
from econovault import EconoVaultClient, EconoVaultAsyncClient

# Synchronous client
client = EconoVaultClient(
    api_key="your-api-key",
    base_url="https://api.econovault.com",  # Optional
    timeout=30,  # Request timeout in seconds
    max_retries=3,  # Maximum retry attempts
    retry_delay=1.0  # Delay between retries
)

# Asynchronous client
async_client = EconoVaultAsyncClient(
    api_key="your-api-key",
    base_url="https://api.econovault.com",
    timeout=30,
    max_retries=3,
    retry_delay=1.0
)
```

### Available Methods

#### `get_indicators(source=None, indicator_type=None, limit=100)`
Get list of economic indicators with optional filtering.

**Parameters:**
- `source` (str, optional): Filter by data source ("BLS", "BEA", "FRED")
- `indicator_type` (str, optional): Filter by indicator type
- `limit` (int): Maximum number of indicators to return (default: 100)

**Returns:** List of `EconomicIndicator` objects

#### `get_indicator(series_id)`
Get detailed information about a specific indicator.

**Parameters:**
- `series_id` (str): The series ID (e.g., "CUUR0000SA0" for CPI)

**Returns:** `EconomicIndicator` object

#### `get_indicator_data(series_id, start_date=None, end_date=None, limit=1000)`
Get historical time series data for an indicator.

**Parameters:**
- `series_id` (str): The series ID
- `start_date` (str/date/datetime, optional): Start date for data range
- `end_date` (str/date/datetime, optional): End date for data range
- `limit` (int): Maximum number of data points (default: 1000)

**Returns:** `TimeSeriesData` object

#### `stream_indicator_data(series_id, update_interval=60)`
Stream real-time updates for an indicator (async only).

**Parameters:**
- `series_id` (str): The series ID to stream
- `update_interval` (int): Update interval in seconds (default: 60)

**Yields:** Data update dictionaries

#### `create_api_key(name, scopes=None, rate_limit_per_minute=60, rate_limit_per_hour=3600)`
Create a new API key.

**Parameters:**
- `name` (str): Name for the API key
- `scopes` (List[str], optional): Permission scopes
- `rate_limit_per_minute` (int): Requests per minute limit
- `rate_limit_per_hour` (int): Requests per hour limit

**Returns:** API key information

#### `get_api_keys()`
Get list of API keys for the current user.

**Returns:** List of `APIKeyInfo` objects

#### `delete_api_key(key_id)`
Delete an API key.

**Parameters:**
- `key_id` (str): The key ID to delete

**Returns:** True if successful

#### `get_health()`
Get API health status.

**Returns:** Health status dictionary

## Data Types

### EconomicIndicator
```python
@dataclass
class EconomicIndicator:
    series_id: str              # Unique identifier
    title: str                  # Human-readable title
    source: str                 # Data source (BLS, BEA, FRED)
    indicator_type: str         # Type of indicator
    frequency: str              # Data frequency (MONTHLY, QUARTERLY, etc.)
    seasonal_adjustment: str    # Seasonal adjustment method
    geography_level: str        # Geographic coverage
    units: str                  # Measurement units
    latest_data: Optional[Dict] # Most recent data point
```

### DataPoint
```python
@dataclass
class DataPoint:
    date: str      # ISO 8601 date string
    value: float   # Numeric value
    period: str    # Time period (M01, Q1, etc.)
    year: int      # Calendar year
```

### TimeSeriesData
```python
@dataclass
class TimeSeriesData:
    series_id: str              # Series identifier
    title: str                  # Full title
    source: str                 # Data source
    data: List[DataPoint]       # Time series data points
    count: int                  # Number of data points
    date_range: Dict[str, str]  # First and last dates
```

## Error Handling

The SDK provides specific exceptions for different error types:

```python
from econovault import (
    EconoVaultError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    ServerError
)

try:
    data = client.get_indicator_data("INVALID_SERIES")
except NotFoundError as e:
    print(f"Indicator not found: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except ServerError as e:
    print(f"Server error: {e}")
except EconoVaultError as e:
    print(f"API error: {e}")
```

## Popular Series IDs

### New Semantic Endpoints (Recommended)

#### Consumer Price Index (CPI)
- `/v1/indicators/consumer-price-index` - CPI metadata and latest data
- `/v1/indicators/consumer-price-index/data` - CPI historical time series
- `/v1/indicators/consumer-price-index/stream` - CPI real-time streaming

#### Unemployment Rate
- `/v1/indicators/unemployment-rate` - Unemployment metadata and latest data
- `/v1/indicators/unemployment-rate/data` - Unemployment historical time series
- `/v1/indicators/unemployment-rate/stream` - Unemployment real-time streaming

#### Nonfarm Payrolls
- `/v1/indicators/nonfarm-payrolls` - Payrolls metadata and latest data
- `/v1/indicators/nonfarm-payrolls/data` - Payrolls historical time series
- `/v1/indicators/nonfarm-payrolls/stream` - Payrolls real-time streaming

#### Real GDP
- `/v1/indicators/real-gdp` - Real GDP metadata and latest data
- `/v1/indicators/real-gdp/data` - Real GDP historical time series
- `/v1/indicators/real-gdp/stream` - Real GDP real-time streaming

### Legacy Series IDs (Still Supported)

#### Consumer Price Index (CPI)
- `CUUR0000SA0`: CPI for All Urban Consumers
- `CUUR0000SA0L1E`: Core CPI (All Items Less Food and Energy)
- `CUUR0000SEEB01`: CPI for Energy Services

#### Employment Data
- `LNS14000000`: Unemployment Rate
- `LNS12000000`: Employment Level
- `LNS13000000`: Labor Force Participation Rate

#### Producer Price Index (PPI)
- `PCUOMFGOMFG`: PPI for All Manufacturing Industries
- `PCU325110325110`: PPI for Chemical Manufacturing

## Advanced Usage

### Pandas Integration

```python
import pandas as pd
from econovault import get_client

with get_client("your-api-key") as client:
    # Get CPI data (automatically uses semantic endpoint)
    cpi_data = client.get_indicator_data("CUUR0000SA0", limit=120)
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {"date": point.date, "value": point.value}
        for point in cpi_data.data
    ])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate year-over-year change
    df['yoy_change'] = df['value'].pct_change(periods=12) * 100
    
    print(df.tail())
    
    # Alternative: Use semantic endpoint directly
    semantic_response = requests.get(
        "https://api.econovault.com/v1/indicators/consumer-price-index/data",
        headers={"Authorization": "Bearer your-api-key"}
    )
    semantic_data = semantic_response.json()
    print(f"Semantic endpoint response: {semantic_data}")
```

### Real-time Streaming

```python
import asyncio
from econovault import get_async_client

async def stream_cpi_updates():
    async with get_async_client("your-api-key") as client:
        print("Streaming CPI updates...")
        
        async for update in client.stream_indicator_data("CUUR0000SA0", update_interval=30):
            timestamp = update['timestamp']
            value = update['data']['value']
            print(f"[{timestamp}] CPI: {value}")
            
            # Process the update
            if value > 300:  # Example threshold
                print("🚨 CPI above threshold!")

asyncio.run(stream_cpi_updates())
```

### Batch Data Retrieval

```python
from econovault import get_client
from concurrent.futures import ThreadPoolExecutor

def fetch_indicator_data(series_id):
    with get_client("your-api-key") as client:
        return client.get_indicator_data(series_id, limit=12)

# Fetch multiple indicators in parallel
series_ids = ["CUUR0000SA0", "LNS14000000", "PCUOMFGOMFG"]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(fetch_indicator_data, series_ids))

for series_id, data in zip(series_ids, results):
    print(f"{series_id}: {len(data.data)} data points")
```

## Configuration

### Environment Variables

```bash
export ECONOVAULT_API_KEY="your-api-key"
export ECONOVAULT_BASE_URL="https://api.econovault.com"
export ECONOVAULT_TIMEOUT="30"
export ECONOVAULT_MAX_RETRIES="3"
```

### Configuration File

```python
# config.py
import os
from econovault import EconoVaultClient

def get_configured_client():
    return EconoVaultClient(
        api_key=os.getenv("ECONOVAULT_API_KEY"),
        base_url=os.getenv("ECONOVAULT_BASE_URL", "https://api.econovault.com"),
        timeout=int(os.getenv("ECONOVAULT_TIMEOUT", "30")),
        max_retries=int(os.getenv("ECONOVAULT_MAX_RETRIES", "3"))
    )
```

## Rate Limiting

The SDK automatically handles rate limiting with exponential backoff:

- **Standard**: 60 requests/minute, 3600 requests/hour
- **Premium**: 300 requests/minute, 18000 requests/hour
- **Enterprise**: Custom limits

## Error Recovery

```python
from econovault import get_client, RateLimitError
import time

def fetch_with_backoff(client, series_id, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return client.get_indicator_data(series_id)
        except RateLimitError:
            wait_time = (2 ** attempt) + (attempt * 0.1)  # Exponential backoff
            print(f"Rate limited, waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
    raise Exception("Max attempts reached")

with get_client("your-api-key") as client:
    data = fetch_with_backoff(client, "CUUR0000SA0")
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Support

- **Documentation**: [https://docs.econovault.com](https://docs.econovault.com)
- **API Reference**: [https://api.econovault.com/docs](https://api.econovault.com/docs)
- **Support**: support@econovault.com
- **Issues**: [GitHub Issues](https://github.com/econovault/econovault-python-sdk/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.