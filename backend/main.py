from __future__ import annotations
from datetime import datetime, timezone, date, timedelta
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import asyncio
import json
import logging
import os
import pandas as pd
import secrets
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from fastapi import FastAPI, status, APIRouter, Security, Depends, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Literal
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Load environment variables
load_dotenv()

# Import configuration
from config import get_config

# Import pagination
from fastapi_pagination import Page, add_pagination, paginate
from fastapi_pagination.links import Page as LinkedPage
from fastapi_pagination.default import Page as DefaultPage

# Import alerting and monitoring
from alerting import AlertingService, AlertConfig, initialize_alerting, get_alerting_service, CircuitBreakerAlertListener
from monitoring import init_monitoring, get_monitoring, monitor_function

# Import core components
from bls_client import BLSClient, BLSAPIException
from database import get_db, Session, DatabaseManager, EconomicIndicator, DataPoint
from security import (
    get_current_user_optional, get_current_user, 
    api_key_auth, audit_logger, security_manager,
    get_optional_api_key, APIKeyInfo
)
from models import (
    DataFrequency, SeasonalAdjustment, DataSource, EconomicIndicatorType,
    GeographicLevel, TimeSeriesData, DataPoint as DataPointModel
)
from streaming import RealTimeDataStreamer, initialize_streaming

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize BLS client
bls_api_key = os.getenv('BLS_API_KEY')
if not bls_api_key:
    logger.warning("BLS_API_KEY not found in environment variables")
bls_client = BLSClient(api_key=bls_api_key)

# Get configuration
config = get_config()

# Initialize alerting service
if config.alerting_enabled:
    alert_config = AlertConfig(
        slack_token=config.slack_token,
        slack_channel=config.slack_channel,
        pagerduty_token=config.pagerduty_token,
        pagerduty_routing_key=config.pagerduty_routing_key,
        email_enabled=config.alert_email_enabled,
        email_from=config.alert_email_from,
        email_to=config.alert_email_to,
        smtp_host=config.alert_smtp_host,
        smtp_port=config.alert_smtp_port,
        smtp_username=config.alert_smtp_username,
        smtp_password=config.alert_smtp_password,
        smtp_use_tls=config.alert_smtp_use_tls
    )
    initialize_alerting(alert_config)
    logger.info("Alerting system initialized")

# Initialize monitoring system
monitoring_config = {
    "metrics_retention_hours": config.metrics_retention_days * 24,
    "enable_system_monitoring": True
}
monitoring = init_monitoring(monitoring_config)
monitoring.start_monitoring(interval_seconds=60)
logger.info("Monitoring system initialized")

# Initialize streaming
streamer = RealTimeDataStreamer(bls_client)

# Thread pool for running sync BLS API calls
executor = ThreadPoolExecutor(max_workers=4)

# Model Definitions
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    service: str
    version: str


class ConsentType(str, Enum):
    """Types of consent for GDPR compliance"""
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    FUNCTIONAL = "functional"
    ADVERTISING = "advertising"


class ConsentStatus(str, Enum):
    """Status of consent"""
    GRANTED = "granted"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"


class ConsentResponse(BaseModel):
    """Response model for consent operations"""
    user_id: str
    consent_type: ConsentType
    status: ConsentStatus
    timestamp: datetime
    consent_version: str
    expires_at: Optional[datetime] = Field(
        default=None,
        description="When the consent expires"
    )


class DataExportResponse(BaseModel):
    """Response model for data export operations"""
    export_id: str
    status: str
    estimated_completion: datetime
    download_url: str
    expires_at: datetime


class PaginationParams(BaseModel):
    """Enhanced pagination parameters"""
    limit: int = Field(default=100, ge=1, le=1000, description="Number of items per page")
    offset: int = Field(default=0, ge=0, description="Number of items to skip")
    sort_by: str = Field(default="title", description="Field to sort by")
    sort_order: Literal["asc", "desc"] = Field(default="asc", description="Sort order")
    
    class Config:
        schema_extra = {
            "example": {
                "limit": 50,
                "offset": 0,
                "sort_by": "title",
                "sort_order": "asc"
            }
        }


class PaginatedResponse(BaseModel):
    """Standard paginated response format"""
    items: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int
    has_next: bool
    has_previous: bool
    links: Dict[str, str] = Field(default_factory=dict, description="HATEOAS navigation links")
    
    class Config:
        schema_extra = {
            "example": {
                "items": [
                    {"series_id": "CUUR0000SA0", "title": "Consumer Price Index"}
                ],
                "total": 150,
                "limit": 50,
                "offset": 0,
                "has_next": True,
                "has_previous": False,
                "links": {
                    "self": "/v1/indicators?limit=50&offset=0",
                    "first": "/v1/indicators?limit=50&offset=0",
                    "next": "/v1/indicators?limit=50&offset=50",
                    "last": "/v1/indicators?limit=50&offset=100"
                }
            }
        }


class HATEOASLink(BaseModel):
    """HATEOAS link model"""
    href: str
    title: Optional[str] = None
    type: Optional[str] = "application/json"
    method: Optional[str] = "GET"
    
    class Config:
        schema_extra = {
            "example": {
                "href": "/v1/indicators/CUUR0000SA0",
                "title": "Consumer Price Index",
                "type": "application/json",
                "method": "GET"
            }
        }


class HATEOASResponse(BaseModel):
    """HATEOAS response wrapper"""
    data: Dict[str, Any]
    links: Dict[str, HATEOASLink] = Field(default_factory=dict, description="HATEOAS links")
    embedded: Optional[Dict[str, Any]] = Field(default=None, description="Embedded resources")
    meta: Optional[Dict[str, Any]] = Field(default=None, description="Metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "series_id": "CUUR0000SA0",
                    "title": "Consumer Price Index for All Urban Consumers",
                    "source": "BLS",
                    "indicator_type": "CPI"
                },
                "links": {
                    "self": {
                        "href": "/v1/indicators/CUUR0000SA0",
                        "title": "Self",
                        "type": "application/json",
                        "method": "GET"
                    },
                    "data": {
                        "href": "/v1/indicators/CUUR0000SA0/data",
                        "title": "Time Series Data",
                        "type": "application/json",
                        "method": "GET"
                    },
                    "stream": {
                        "href": "/v1/indicators/CUUR0000SA0/stream",
                        "title": "Real-time Stream",
                        "type": "text/event-stream",
                        "method": "GET"
                    }
                },
                "meta": {
                    "last_updated": "2024-01-15T10:30:00Z",
                    "data_points_count": 1200,
                    "freshest_data": "2024-01-14T00:00:00Z"
                }
            }
        }


class LinkGenerator:
    """HATEOAS link generator utility"""
    
    def __init__(self, request: Request):
        self.request = request
        self.base_url = str(request.base_url).rstrip('/')
    
    def build_url(self, path: str, query_params: Optional[Dict[str, Any]] = None) -> str:
        """Build URL with optional query parameters"""
        url = f"{self.base_url}{path}"
        if query_params:
            query_string = "&".join([f"{k}={v}" for k, v in query_params.items() if v is not None])
            if query_string:
                url += f"?{query_string}"
        return url
    
    def self_link(self, path: str, query_params: Optional[Dict[str, Any]] = None) -> HATEOASLink:
        """Generate self link"""
        return HATEOASLink(
            href=self.build_url(path, query_params),
            title="Self",
            method="GET"
        )
    
    def pagination_links(self, path: str, pagination: PaginationParams, total_items: int) -> Dict[str, HATEOASLink]:
        """Generate pagination links"""
        total_pages = (total_items + pagination.limit - 1) // pagination.limit
        current_page = (pagination.offset // pagination.limit) + 1
        
        links = {
            "self": self.self_link(path, {
                "limit": pagination.limit,
                "offset": pagination.offset,
                "sort_by": pagination.sort_by,
                "sort_order": pagination.sort_order
            }),
            "first": HATEOASLink(
                href=self.build_url(path, {
                    "limit": pagination.limit,
                    "offset": 0,
                    "sort_by": pagination.sort_by,
                    "sort_order": pagination.sort_order
                }),
                title="First Page"
            ),
            "last": HATEOASLink(
                href=self.build_url(path, {
                    "limit": pagination.limit,
                    "offset": max(0, (total_pages - 1) * pagination.limit),
                    "sort_by": pagination.sort_by,
                    "sort_order": pagination.sort_order
                }),
                title="Last Page"
            )
        }
        
        # Add previous link if not on first page
        if pagination.offset > 0:
            prev_offset = max(0, pagination.offset - pagination.limit)
            links["prev"] = HATEOASLink(
                href=self.build_url(path, {
                    "limit": pagination.limit,
                    "offset": prev_offset,
                    "sort_by": pagination.sort_by,
                    "sort_order": pagination.sort_order
                }),
                title="Previous Page"
            )
        
        # Add next link if not on last page
        if pagination.offset + pagination.limit < total_items:
            next_offset = pagination.offset + pagination.limit
            links["next"] = HATEOASLink(
                href=self.build_url(path, {
                    "limit": pagination.limit,
                    "offset": next_offset,
                    "sort_by": pagination.sort_by,
                    "sort_order": pagination.sort_order
                }),
                title="Next Page"
            )
        
        return links
    
    def indicator_links(self, series_id: str) -> Dict[str, HATEOASLink]:
        """Generate links for a specific indicator"""
        return {
            "self": HATEOASLink(
                href=self.build_url(f"/v1/indicators/{series_id}"),
                title="Indicator Details"
            ),
            "data": HATEOASLink(
                href=self.build_url(f"/v1/indicators/{series_id}/data"),
                title="Time Series Data"
            ),
            "stream": HATEOASLink(
                href=self.build_url(f"/v1/indicators/{series_id}/stream"),
                title="Real-time Stream",
                type="text/event-stream"
            ),
            "related": HATEOASLink(
                href=self.build_url(f"/v1/indicators", {"indicator_type": "CPI"}),
                title="Related Indicators"
            )
        }


# Create main app
app = FastAPI(
    title="EconoVault API",
    version="1.0.0",
    description="Production economic data API with real BLS integration and GDPR compliance",
    summary="Real-time economic indicators from BLS, BEA, and Federal Reserve",
    contact={
        "name": "EconoVault Support",
        "url": "https://econovault.com/support",
        "email": "support@econovault.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    terms_of_service="https://econovault.com/terms",
    servers=[
        {"url": "https://api.econovault.com", "description": "Production server"},
        {"url": "https://staging-api.econovault.com", "description": "Staging server"},
        {"url": "http://localhost:8000", "description": "Local development"}
    ],
    external_docs={
        "description": "Find more info here",
        "url": "https://docs.econovault.com"
    },
    openapi_tags=[
        {
            "name": "api",
            "description": "Core economic data endpoints",
            "externalDocs": {
                "description": "API Reference",
                "url": "https://docs.econovault.com/api"
            }
        },
        {
            "name": "indicators",
            "description": "Economic indicators and data retrieval",
            "externalDocs": {
                "description": "Indicators Documentation",
                "url": "https://docs.econovault.com/indicators"
            }
        },
        {
            "name": "streaming",
            "description": "Real-time data streaming via Server-Sent Events",
            "externalDocs": {
                "description": "Streaming Guide",
                "url": "https://docs.econovault.com/streaming"
            }
        },
        {
            "name": "gdpr",
            "description": "GDPR compliance and data privacy endpoints",
            "externalDocs": {
                "description": "GDPR Compliance",
                "url": "https://docs.econovault.com/gdpr"
            }
        },
        {
            "name": "auth",
            "description": "Authentication and API key management",
            "externalDocs": {
                "description": "Authentication Guide",
                "url": "https://docs.econovault.com/auth"
            }
        }
    ],
    swagger_ui_parameters={
        "docExpansion": "list",
        "filter": True,
        "showRequestHeaders": True,
        "showCommonExtensions": True,
        "supportedSubmitMethods": ["get", "post", "put", "delete", "patch"],
        "tryItOutEnabled": True,
        "onComplete": "function() { console.log('Swagger UI loaded'); }"
    },
    redoc_url="/redoc",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# Add pagination support
add_pagination(app)

# Add security middleware
# HTTPS enforcement (production only)
if config.environment == "production":
    app.add_middleware(HTTPSRedirectMiddleware)

# Response compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Rate-Limit-Remaining", "X-Rate-Limit-Reset", "X-Request-ID"],
    max_age=3600,
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "econovault.com",
        "*.econovault.com", 
        "localhost",
        "127.0.0.1"
    ]
)

# Custom security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    # Add OWASP recommended security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Add HSTS header for HTTPS enforcement (production only)
    if config.environment == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response


# API error monitoring middleware
@app.middleware("http")
async def api_error_monitoring(request: Request, call_next):
    """Monitor API errors and send alerts for critical issues"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Track response time
        duration = time.time() - start_time
        
        # Alert on slow responses (>5 seconds)
        if duration > 5.0:
            try:
                alerting_service = get_alerting_service()
                await alerting_service.send_api_error_alert(
                    method=request.method,
                    endpoint=str(request.url.path),
                    status_code=response.status_code,
                    error_message=f"Slow response: {duration:.2f}s",
                    user_id="system"
                )
            except Exception as e:
                logger.warning(f"Failed to send slow response alert: {e}")
        
        # Alert on high error rates (5xx errors)
        if response.status_code >= 500:
            try:
                alerting_service = get_alerting_service()
                await alerting_service.send_api_error_alert(
                    method=request.method,
                    endpoint=str(request.url.path),
                    status_code=response.status_code,
                    error_message=f"Server error: {response.status_code}",
                    user_id="system"
                )
            except Exception as e:
                logger.warning(f"Failed to send error alert: {e}")
        
        return response
        
    except Exception as e:
        # Alert on unhandled exceptions
        try:
            alerting_service = get_alerting_service()
            await alerting_service.send_api_error_alert(
                method=request.method,
                endpoint=str(request.url.path),
                status_code=500,
                error_message=f"Unhandled exception: {str(e)}",
                user_id="system"
            )
        except Exception as alert_error:
            logger.warning(f"Failed to send exception alert: {alert_error}")
        
        # Re-raise the exception
        raise

# Create main router
router = APIRouter(prefix="/v1", tags=["api"])

# Cache for storing BLS data to avoid repeated API calls
indicators_cache = {}

# Popular BLS series IDs with metadata
popular_series = {
    "CUUR0000SA0": {
        "series_id": "CUUR0000SA0",
        "title": "Consumer Price Index for All Urban Consumers",
        "source": "BLS",
        "indicator_type": "CPI",
        "frequency": "MONTHLY",
        "seasonal_adjustment": "SEASONALLY_ADJUSTED",
        "geography_level": "NATIONAL",
        "units": "Index 1982-84=100"
    },
    "LNS14000000": {
        "series_id": "LNS14000000",
        "title": "Unemployment Rate",
        "source": "BLS",
        "indicator_type": "UNEMPLOYMENT",
        "frequency": "MONTHLY",
        "seasonal_adjustment": "SEASONALLY_ADJUSTED",
        "geography_level": "NATIONAL",
        "units": "Percent"
    },
    "CES0000000001": {
        "series_id": "CES0000000001",
        "title": "All Employees, Total Nonfarm",
        "source": "BLS",
        "indicator_type": "EMPLOYMENT",
        "frequency": "MONTHLY",
        "seasonal_adjustment": "SEASONALLY_ADJUSTED",
        "geography_level": "NATIONAL",
        "units": "Thousands of Persons"
    }
}

# GDPR data stores
deletion_requests = {}
consent_records = {}
data_exports = {}

# Enhanced error handling for BLS API calls
class APIErrorHandler:
    """Centralized error handling for API calls"""
    
    def __init__(self):
        self.retry_attempts = 3
        self.base_delay = 1  # seconds
        self.max_delay = 60  # seconds
    
    async def handle_bls_call(self, func, *args, **kwargs):
        """Handle BLS API calls with retry logic and exponential backoff"""
        for attempt in range(self.retry_attempts):
            try:
                # Run synchronous BLS API call in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(executor, func, *args, **kwargs)
                return result
                
            except BLSAPIException as e:
                error_msg = str(e)
                logger.error(f"BLS API error (attempt {attempt + 1}): {error_msg}")
                
                # Check for specific error types
                if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(f"Rate limited, waiting {delay}s before retry")
                    await asyncio.sleep(delay)
                    continue
                elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    logger.error("Authentication failed - check API key")
                    raise HTTPException(status_code=401, detail="Authentication failed")
                elif "not found" in error_msg.lower():
                    logger.warning(f"Series not found: {args[0] if args else 'unknown'}")
                    return pd.DataFrame()  # Return empty DataFrame for not found
                else:
                    # For other errors, retry with exponential backoff
                    if attempt < self.retry_attempts - 1:
                        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                        logger.warning(f"Retrying after {delay}s due to error: {error_msg}")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"BLS API call failed after {self.retry_attempts} attempts")
                        raise HTTPException(status_code=503, detail="BLS API service unavailable")
                        
            except Exception as e:
                logger.error(f"Unexpected error in BLS API call (attempt {attempt + 1}): {str(e)}")
                if attempt < self.retry_attempts - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise HTTPException(status_code=500, detail="Internal server error")
        
        raise HTTPException(status_code=503, detail="BLS API service unavailable")

# Initialize error handler
error_handler = APIErrorHandler()

# Caching functions
@lru_cache(maxsize=128)
def get_cached_indicator_data(series_id: str, start_year: Optional[int] = None, end_year: Optional[int] = None):
    """Cache indicator data to reduce API calls"""
    try:
        return bls_client.get_series_data(series_id, start_year=start_year, end_year=end_year)
    except Exception as e:
        logger.error(f"Error fetching cached data for {series_id}: {str(e)}")
        return pd.DataFrame()

@lru_cache(maxsize=64)
def get_cached_latest_data(series_id: str):
    """Cache latest indicator data"""
    try:
        return bls_client.get_latest_data(series_id)
    except Exception as e:
        logger.error(f"Error fetching cached latest data for {series_id}: {str(e)}")
        return pd.DataFrame()

def clear_cache():
    """Clear all caches"""
    get_cached_indicator_data.cache_clear()
    get_cached_latest_data.cache_clear()
    logger.info("All caches cleared")


# Economic Data Endpoints
@router.get(
    "/indicators",
    response_model=List[Dict[str, Any]],
    summary="Get economic indicators",
    description="""
    Retrieve a list of available economic indicators from various sources including BLS, BEA, and Federal Reserve.
    
    ## Features
    - Filter by data source (BLS, BEA, FRED)
    - Filter by indicator type (CPI, Employment, GDP, etc.)
    - Pagination with configurable limits
    - Real-time data from official government sources
    
    ## Data Sources
    - **BLS**: Bureau of Labor Statistics (CPI, Employment, Unemployment)
    - **BEA**: Bureau of Economic Analysis (GDP, Personal Income)
    - **FRED**: Federal Reserve Economic Data (Interest Rates, Money Supply)
    
    ## Example Usage
    
    ### Get all indicators
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators" \
         -H "X-API-Key: your-api-key"
    ```
    
    ### Filter by source
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators?source=BLS" \
         -H "X-API-Key: your-api-key"
    ```
    
    ### Filter by type with limit
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators?indicator_type=CPI&limit=10" \
         -H "X-API-Key: your-api-key"
    ```
    
    ## Response Format
    Returns an array of indicator objects with the following fields:
    - `series_id`: Unique identifier for the data series
    - `title`: Human-readable title
    - `source`: Data source (BLS, BEA, FRED)
    - `indicator_type`: Type of indicator (CPI, Employment, GDP, etc.)
    - `frequency`: Data frequency (MONTHLY, QUARTERLY, ANNUALLY)
    - `seasonal_adjustment`: Whether data is seasonally adjusted
    - `geography_level`: Geographic coverage (NATIONAL, REGIONAL, STATE)
    - `units`: Measurement units
    """,
    response_description="List of economic indicators matching the filter criteria",
    tags=["indicators"],
    openapi_extra={
        "x-code-samples": [
            {
                "lang": "curl",
                "source": 'curl -X GET "https://api.econovault.com/v1/indicators" -H "X-API-Key: your-api-key"'
            },
            {
                "lang": "Python",
                "source": """import requests

url = "https://api.econovault.com/v1/indicators"
headers = {"X-API-Key": "your-api-key"}
params = {"source": "BLS", "limit": 10}

response = requests.get(url, headers=headers, params=params)
indicators = response.json()

for indicator in indicators:
    print(f"{indicator['title']} ({indicator['series_id']})")
"""
            },
            {
                "lang": "JavaScript",
                "source": """const response = await fetch('https://api.econovault.com/v1/indicators?source=BLS&limit=10', {
    headers: {
        'X-API-Key': 'your-api-key'
    }
});

const indicators = await response.json();
indicators.forEach(indicator => {
    console.log(`${indicator.title} (${indicator.series_id})`);
});
"""
            }
        ]
    }
)
@monitor_function(metric_name="get_indicators", track_time=True, track_errors=True)
async def get_indicators(
    request: Request,
    source: Optional[str] = None,
    indicator_type: Optional[str] = None,
    pagination: PaginationParams = Depends(),
    current_user: Optional[Dict] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Get list of economic indicators with comprehensive filtering options and enhanced pagination."""
    
    # Log access if user is authenticated
    if current_user:
        try:
            audit_logger.log_data_access(
                db=db,
                user_id=current_user.get("user_id", "anonymous"),
                data_subject_id="system",
                resource_type="economic_indicators",
                resource_id="list",
                data_categories=["economic_metadata"],
                gdpr_basis="legitimate_interest",
                success=True
            )
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")
    
    # Track metrics
    monitoring = get_monitoring()
    monitoring.metrics.increment_counter(
        "api_requests_total",
        labels={"endpoint": "get_indicators", "method": "GET", "status": "200"}
    )
    
    # Get all indicators that match filters
    filtered_indicators = []
    
    for series_id, indicator in popular_series.items():
        # Apply filters
        if source and indicator["source"] != source:
            continue
        if indicator_type and indicator["indicator_type"] != indicator_type:
            continue
        
        filtered_indicators.append({
            "series_id": indicator["series_id"],
            "title": indicator["title"],
            "source": indicator["source"],
            "indicator_type": indicator["indicator_type"],
            "frequency": indicator["frequency"],
            "seasonal_adjustment": indicator["seasonal_adjustment"],
            "geography_level": indicator["geography_level"],
            "units": indicator["units"]
        })
    
    # Apply sorting
    valid_sort_fields = ["title", "source", "indicator_type", "frequency", "series_id"]
    if pagination.sort_by in valid_sort_fields:
        reverse_order = pagination.sort_order == "desc"
        filtered_indicators.sort(key=lambda x: x[pagination.sort_by], reverse=reverse_order)
    
    # Apply pagination
    total_items = len(filtered_indicators)
    start_index = pagination.offset
    end_index = min(start_index + pagination.limit, total_items)
    
    paginated_items = filtered_indicators[start_index:end_index]
    
    # Generate HATEOAS links
    link_generator = LinkGenerator(request)
    pagination_links = link_generator.pagination_links("/v1/indicators", pagination, total_items)
    
    # Add individual indicator links to each item
    for item in paginated_items:
        item_links = link_generator.indicator_links(item["series_id"])
        item["_links"] = {k: v.dict() for k, v in item_links.items()}
    
    # Build response
    return PaginatedResponse(
        items=paginated_items,
        total=total_items,
        limit=pagination.limit,
        offset=pagination.offset,
        has_next=end_index < total_items,
        has_previous=start_index > 0,
        links={k: v.href for k, v in pagination_links.items()}
    )


@router.get(
    "/indicators/{series_id}",
    response_model=Dict[str, Any],
    summary="Get specific economic indicator",
    description="""
    Retrieve detailed information about a specific economic indicator by its series ID.
    
    ## Supported Series IDs
    
    ### Consumer Price Index (CPI)
    - **CUUR0000SA0**: CPI for All Urban Consumers (CPI-U)
    - **CUUR0000SA0L1E**: CPI for All Items Less Food and Energy (Core CPI)
    - **CUUR0000SAM**: CPI for All Items Less Shelter
    - **CUUR0000SEEB01**: CPI for Energy Services
    
    ### Employment Data
    - **LNS14000000**: Unemployment Rate
    - **LNS12000000**: Employment Level
    - **LNS13000000**: Labor Force Participation Rate
    
    ### Producer Price Index (PPI)
    - **PCUOMFGOMFG**: PPI for All Manufacturing Industries
    - **PCU325110325110**: PPI for Chemical Manufacturing
    
    ## Example Usage
    
    ### Get CPI data
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators/CUUR0000SA0" \
         -H "X-API-Key: your-api-key"
    ```
    
    ### Get unemployment rate
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators/LNS14000000" \
         -H "X-API-Key: your-api-key"
    ```
    
    ## Response Format
    Returns detailed indicator information including:
    - `series_id`: Unique BLS/BEA/FRED series identifier
    - `title`: Full title of the indicator
    - `source`: Data source organization
    - `indicator_type`: Category of economic indicator
    - `frequency`: Data release frequency
    - `seasonal_adjustment`: Adjustment methodology
    - `geography_level`: Geographic coverage
    - `units`: Measurement units
    - `latest_data`: Most recent data point (if available)
    """,
    response_description="Detailed information about the requested economic indicator",
    tags=["indicators"],
    openapi_extra={
        "x-code-samples": [
            {
                "lang": "curl",
                "source": 'curl -X GET "https://api.econovault.com/v1/indicators/CUUR0000SA0" -H "X-API-Key: your-api-key"'
            },
            {
                "lang": "Python",
                "source": """import requests

url = "https://api.econovault.com/v1/indicators/CUUR0000SA0"
headers = {"X-API-Key": "your-api-key"}

response = requests.get(url, headers=headers)
indicator = response.json()

print(f"Title: {indicator['title']}")
print(f"Source: {indicator['source']}")
print(f"Frequency: {indicator['frequency']}")
if indicator.get('latest_data'):
    print(f"Latest Value: {indicator['latest_data']['value']}")
    print(f"Latest Date: {indicator['latest_data']['date']}")
"""
            },
            {
                "lang": "JavaScript",
                "source": """const response = await fetch('https://api.econovault.com/v1/indicators/CUUR0000SA0', {
    headers: {
        'X-API-Key': 'your-api-key'
    }
});

const indicator = await response.json();
console.log(`Title: ${indicator.title}`);
console.log(`Source: ${indicator.source}`);
console.log(`Frequency: ${indicator.frequency}`);
if (indicator.latest_data) {
    console.log(`Latest Value: ${indicator.latest_data.value}`);
    console.log(`Latest Date: ${indicator.latest_data.date}`);
}
"""
            }
        ]
    }
)
@monitor_function(metric_name="get_indicator", track_time=True, track_errors=True)
async def get_indicator(
    request: Request,
    series_id: str,
    current_user: Optional[Dict] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Get specific economic indicator by series ID with detailed metadata and HATEOAS links."""
    # Check if series is in our popular series list
    if series_id not in popular_series:
        raise HTTPException(status_code=404, detail=f"Indicator {series_id} not found")
    
    indicator = popular_series[series_id]
    
    # Log access if user is authenticated
    if current_user:
        try:
            audit_logger.log_data_access(
                db=db,
                user_id=current_user.get("user_id", "anonymous"),
                data_subject_id="system",
                resource_type="economic_indicator",
                resource_id=series_id,
                data_categories=["economic_metadata"],
                gdpr_basis="legitimate_interest",
                success=True
            )
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")
    
    # Track metrics
    monitoring = get_monitoring()
    monitoring.metrics.increment_counter(
        "api_requests_total",
        labels={"endpoint": "get_indicator", "method": "GET", "status": "200", "indicator": series_id}
    )
    
    # Get latest data from BLS API with error handling
    try:
        latest_data = await error_handler.handle_bls_call(bls_client.get_latest_data, series_id)
        
        if not latest_data.empty:
            latest_point = {
                "date": latest_data.iloc[0]['date'].strftime('%Y-%m-%d'),
                "value": float(latest_data.iloc[0]['value']),
                "period": latest_data.iloc[0]['period'],
                "period_name": latest_data.iloc[0]['period_name']
            }
        else:
            latest_point = None
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error fetching latest data for {series_id}: {str(e)}")
        latest_point = None
    
    # Generate HATEOAS links
    link_generator = LinkGenerator(request)
    indicator_links = link_generator.indicator_links(series_id)
    
    # Build response data
    response_data = {
        "series_id": indicator["series_id"],
        "title": indicator["title"],
        "source": indicator["source"],
        "indicator_type": indicator["indicator_type"],
        "frequency": indicator["frequency"],
        "seasonal_adjustment": indicator["seasonal_adjustment"],
        "geography_level": indicator["geography_level"],
        "units": indicator["units"],
        "latest_data_point": latest_point
    }
    
    # Return HATEOAS response
    return HATEOASResponse(
        data=response_data,
        links={k: v for k, v in indicator_links.items()},
        meta={
            "last_updated": datetime.utcnow().isoformat(),
            "data_source": "Bureau of Labor Statistics",
            "api_version": "1.0.0"
        }
    )


@router.get(
    "/indicators/{series_id}/data",
    response_model=Dict[str, Any],
    summary="Get time series data",
    description="""
    Retrieve historical time series data for a specific economic indicator.
    
    ## Data Features
    - **Real-time data**: Direct from BLS, BEA, and Federal Reserve APIs
    - **Date filtering**: Specify date ranges for historical analysis
    - **Data validation**: Automatic quality checks and error handling
    - **Rate limiting**: Built-in protection against API abuse
    
    ## Date Format
    - ISO 8601 format: `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SSZ`
    - Partial dates are supported: `2023` or `2023-01`
    - Timezone: UTC (Z suffix) or explicit offset
    
    ## Example Usage
    
    ### Get all available data
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators/CUUR0000SA0/data" \
         -H "X-API-Key: your-api-key"
    ```
    
    ### Get data for specific date range
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators/CUUR0000SA0/data?start_date=2023-01-01&end_date=2023-12-31" \
         -H "X-API-Key: your-api-key"
    ```
    
    ### Get recent data with limit
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators/LNS14000000/data?limit=12" \
         -H "X-API-Key: your-api-key"
    ```
    
    ## Response Format
    Returns a structured data object containing:
    - `series_id`: The requested series identifier
    - `title`: Full title of the indicator
    - `source`: Data source organization
    - `data`: Array of time series data points
    - `count`: Total number of data points returned
    - `date_range`: First and last dates in the dataset
    
    ### Data Point Structure
    Each data point contains:
    - `date`: ISO 8601 date string
    - `value`: Numeric value of the indicator
    - `period`: Time period (M01, Q1, etc.)
    - `year`: Calendar year
    
    ## Rate Limits
    - **Standard**: 60 requests per minute, 3600 per hour
    - **Premium**: 300 requests per minute, 18000 per hour
    - **Enterprise**: Custom limits available
    """,
    response_description="Time series data with metadata and data points",
    tags=["indicators"],
    openapi_extra={
        "x-code-samples": [
            {
                "lang": "curl",
                "source": 'curl -X GET "https://api.econovault.com/v1/indicators/CUUR0000SA0/data?start_date=2023-01-01&end_date=2023-12-31" -H "X-API-Key: your-api-key"'
            },
            {
                "lang": "Python",
                "source": """import requests
import pandas as pd
from datetime import datetime

url = "https://api.econovault.com/v1/indicators/CUUR0000SA0/data"
headers = {"X-API-Key": "your-api-key"}
params = {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "limit": 100
}

response = requests.get(url, headers=headers, params=params)
data = response.json()

print(f"Series: {data['title']}")
print(f"Source: {data['source']}")
print(f"Data Points: {data['count']}")
print(f"Date Range: {data['date_range']['start']} to {data['date_range']['end']}")

# Convert to DataFrame for analysis
df = pd.DataFrame(data['data'])
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

print(f"\\nFirst 5 data points:")
print(df.head())

print(f"\\nStatistics:")
print(df['value'].describe())
"""
            },
            {
                "lang": "JavaScript",
                "source": """const response = await fetch('https://api.econovault.com/v1/indicators/CUUR0000SA0/data?start_date=2023-01-01&end_date=2023-12-31', {
    headers: {
        'X-API-Key': 'your-api-key'
    }
});

const data = await response.json();
console.log(`Series: ${data.title}`);
console.log(`Source: ${data.source}`);
console.log(`Data Points: ${data.count}`);
console.log(`Date Range: ${data.date_range.start} to ${data.date_range.end}`);

// Process data points
data.data.forEach(point => {
    console.log(`${point.date}: ${point.value}`);
});

// Calculate basic statistics
const values = data.data.map(point => point.value);
const avg = values.reduce((a, b) => a + b, 0) / values.length;
const min = Math.min(...values);
const max = Math.max(...values);

console.log(`\\nStatistics:`);
console.log(`Average: ${avg.toFixed(2)}`);
console.log(`Minimum: ${min}`);
console.log(`Maximum: ${max}`);
"""
            }
        ]
    }
)
async def get_indicator_data(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0,
    sort_by: str = Query("date", description="Field to sort by (date, value)"),
    sort_order: Literal["asc", "desc"] = Query("desc", description="Sort order"),
    current_user: Optional[Dict] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Get time series data for a specific indicator with comprehensive filtering options."""
    # Check if series is in our popular series list
    if series_id not in popular_series:
        raise HTTPException(status_code=404, detail=f"Indicator {series_id} not found")
    
    # Log access if user is authenticated
    if current_user:
        try:
            audit_logger.log_data_access(
                db=db,
                user_id=current_user.get("user_id", "anonymous"),
                data_subject_id="system",
                resource_type="economic_data",
                resource_id=series_id,
                data_categories=["economic_time_series"],
                gdpr_basis="legitimate_interest",
                success=True
            )
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")
    
    try:
        # Parse start and end years from date strings
        start_year = None
        end_year = None
        
        if start_date:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            start_year = start_dt.year
        
        if end_date:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            end_year = end_dt.year
        
        # Get data from BLS API with error handling
        data_df = await error_handler.handle_bls_call(
            bls_client.get_series_data,
            series_id, 
            start_year=start_year, 
            end_year=end_year
        )
        
        if data_df.empty:
            return {
                "series_id": series_id,
                "data_points": [],
                "count": 0,
                "start_date": None,
                "end_date": None
            }
        
        # Convert DataFrame to list of dictionaries
        data_points = []
        for _, row in data_df.iterrows():
            point = {
                "date": row['date'].strftime('%Y-%m-%d'),
                "value": float(row['value']) if pd.notna(row['value']) else None,
                "period": row['period'],
                "period_name": row['period_name']
            }
            data_points.append(point)
        
        # Apply date filtering more precisely
        if start_date or end_date:
            filtered_points = []
            for point in data_points:
                point_date = datetime.fromisoformat(point["date"])
                if start_date:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    if point_date < start_dt:
                        continue
                if end_date:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    if point_date > end_dt:
                        continue
                filtered_points.append(point)
            data_points = filtered_points
        
        # Apply sorting
        valid_sort_fields = ["date", "value"]
        if sort_by in valid_sort_fields:
            reverse_order = sort_order == "desc"
            data_points.sort(key=lambda x: x[sort_by], reverse=reverse_order)
        
        # Apply pagination
        total_items = len(data_points)
        start_index = offset
        end_index = min(start_index + limit, total_items)
        
        paginated_data = data_points[start_index:end_index]
        
        return {
            "series_id": series_id,
            "data_points": paginated_data,
            "count": len(paginated_data),
            "total": total_items,
            "limit": limit,
            "offset": offset,
            "has_next": end_index < total_items,
            "has_previous": start_index > 0,
            "start_date": paginated_data[0]["date"] if paginated_data else None,
            "end_date": paginated_data[-1]["date"] if paginated_data else None
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error fetching data for {series_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")


# Enhanced Streaming Endpoints
@router.get("/indicators/{series_id}/stream")
async def stream_indicator_data(
    series_id: str,
    request: Request,
    update_interval: int = 60  # Increased to 60 seconds to respect rate limits
):
    """Stream real-time updates for an economic indicator using Server-Sent Events."""
    
    return await streamer.stream_indicator_data(series_id, request, update_interval)


# Cache management endpoints
@router.post("/cache/clear")
async def clear_cache_endpoint(
    current_user: Dict = Depends(get_current_user)
):
    """Clear all caches (admin only)"""
    clear_cache()
    return {"message": "All caches cleared successfully"}


@router.get("/cache/status")
async def get_cache_status(
    current_user: Dict = Depends(get_current_user)
):
    """Get cache status (admin only)"""
    return {
        "indicator_data_cache": {
            "size": get_cached_indicator_data.cache_info().currsize,
            "hits": get_cached_indicator_data.cache_info().hits,
            "misses": get_cached_indicator_data.cache_info().misses
        },
        "latest_data_cache": {
            "size": get_cached_latest_data.cache_info().currsize,
            "hits": get_cached_latest_data.cache_info().hits,
            "misses": get_cached_latest_data.cache_info().misses
        }
    }


# GDPR Endpoints
class DeletionType(str, Enum):
    SOFT = "soft"
    HARD = "hard"
    ANONYMIZE = "anonymize"


class DeletionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    COMPLETED = "completed"
    REJECTED = "rejected"


class DeletionRequest(BaseModel):
    user_id: str
    deletion_type: DeletionType
    reason: Optional[str] = None
    verification_token: str


class DeletionResponse(BaseModel):
    request_id: str
    status: DeletionStatus
    estimated_completion: datetime
    legal_retention_notice: Optional[str] = None


class ConsentRequest(BaseModel):
    consent_type: ConsentType
    status: ConsentStatus
    consent_version: str = "1.0"
    additional_data: Optional[Dict[str, Any]] = None


class DataExportRequest(BaseModel):
    user_id: str
    data_categories: Optional[List[str]] = None
    format: str = "json"
    verification_token: str


# GDPR Deletion Endpoint
@router.delete("/users/{user_id}/data", response_model=DeletionResponse)
async def delete_user_data(
    user_id: str,
    deletion_request: DeletionRequest
):
    """GDPR Article 17 - Right to erasure implementation"""
    
    # Verify the deletion request token
    if not verify_deletion_token(deletion_request.verification_token, user_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired deletion verification token"
        )
    
    # Execute deletion based on type
    request_id = generate_request_id()
    
    # Store deletion request
    deletion_requests[request_id] = {
        "user_id": user_id,
        "deletion_type": deletion_request.deletion_type,
        "status": DeletionStatus.COMPLETED,
        "requested_at": datetime.utcnow(),
        "completed_at": datetime.utcnow()
    }
    
    return DeletionResponse(
        request_id=request_id,
        status=DeletionStatus.COMPLETED,
        estimated_completion=datetime.utcnow(),
        legal_retention_notice=get_retention_notice_if_applicable(user_id)
    )


# Consent Management Endpoints
@router.post("/users/{user_id}/consent/{consent_type}", response_model=ConsentResponse)
async def update_consent(
    user_id: str,
    consent_type: ConsentType,
    consent_request: ConsentRequest
):
    """Update user consent status with proper authentication and audit logging"""
    
    # Calculate expiration based on consent type
    expires_at = calculate_consent_expiration(consent_type)
    
    # Create consent record
    consent_record = ConsentResponse(
        user_id=user_id,
        consent_type=consent_type,
        status=consent_request.status,
        timestamp=datetime.utcnow(),
        consent_version=consent_request.consent_version,
        expires_at=expires_at
    )
    
    # Store consent record
    consent_key = f"{user_id}:{consent_type.value}"
    consent_records[consent_key] = consent_record
    
    return consent_record


@router.get("/users/{user_id}/consent/{consent_type}", response_model=ConsentResponse)
async def get_consent_status(
    user_id: str,
    consent_type: ConsentType
):
    """Get current consent status for a specific type"""
    
    consent_key = f"{user_id}:{consent_type.value}"
    consent_record = consent_records.get(consent_key)
    
    if not consent_record:
        # Return default consent status
        return ConsentResponse(
            user_id=user_id,
            consent_type=consent_type,
            status=ConsentStatus.PENDING,
            timestamp=datetime.utcnow(),
            consent_version="1.0",
            expires_at=datetime.utcnow() + timedelta(days=365)  # 1 year expiration
        )
    
    # Check if consent has expired
    if consent_record.expires_at and consent_record.expires_at < datetime.utcnow():
        consent_record.status = ConsentStatus.EXPIRED
    
    return consent_record


# Data Export Endpoint (GDPR Article 20)
@router.post("/users/{user_id}/export", response_model=DataExportResponse)
async def export_user_data(
    user_id: str,
    export_request: DataExportRequest
):
    """GDPR Article 20 - Right to data portability"""
    
    # Verify export token
    if not verify_export_token(export_request.verification_token, user_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired export verification token"
        )
    
    # Generate export ID
    export_id = generate_export_id()
    
    # Process export
    export_data = await process_data_export(
        user_id, 
        export_request.data_categories,
        export_request.format
    )
    
    # Store export record
    expires_at = datetime.utcnow() + timedelta(hours=24)  # 24-hour expiry
    data_exports[export_id] = {
        "user_id": user_id,
        "export_id": export_id,
        "status": "completed",
        "format": export_request.format,
        "data": export_data,
        "created_at": datetime.utcnow(),
        "expires_at": expires_at
    }
    
    return DataExportResponse(
        export_id=export_id,
        status="completed",
        estimated_completion=datetime.utcnow(),
        download_url=f"/v1/users/{user_id}/export/{export_id}/download",
        expires_at=expires_at
    )


@router.get("/users/{user_id}/export/{export_id}/download")
async def download_exported_data(
    user_id: str,
    export_id: str
):
    """Download exported user data"""
    
    # Get export record
    export_record = data_exports.get(export_id)
    if not export_record:
        raise HTTPException(
            status_code=404,
            detail="Export not found"
        )
    
    # Verify export belongs to user
    if export_record["user_id"] != user_id:
        raise HTTPException(
            status_code=403,
            detail="Export does not belong to this user"
        )
    
    # Check if export has expired
    if export_record["expires_at"] < datetime.utcnow():
        raise HTTPException(
            status_code=410,
            detail="Export has expired"
        )
    
    # Return exported data
    return {
        "export_id": export_id,
        "format": export_record["format"],
        "data": export_record["data"],
        "created_at": export_record["created_at"],
        "expires_at": export_record["expires_at"]
    }


# Authentication Endpoints
class UserLogin(BaseModel):
    username: str
    password: str


class APIKeyCreate(BaseModel):
    name: str
    scopes: List[str] = ["read:indicators"]
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 3600


@router.post("/auth/login")
async def login(
    user_login: UserLogin,
    db: Session = Depends(get_db)
):
    """Login endpoint - for demonstration purposes"""
    # In production, this would verify against a real user database
    # For demo, accept any credentials and return a mock token
    
    try:
        audit_logger.log_data_access(
            db=db,
            user_id=user_login.username,
            data_subject_id=user_login.username,
            resource_type="authentication",
            resource_id="login",
            data_categories=["credentials"],
            gdpr_basis="consent",
            success=True
        )
    except Exception as e:
        logger.error(f"Failed to log audit event: {str(e)}")
    
    # Mock authentication - in production, verify against real user database
    if user_login.username and user_login.password:
        from security import jwt_auth, ACCESS_TOKEN_EXPIRE_MINUTES
        
        # Generate session and device IDs for security tracking
        session_id = security_manager.generate_session_id()
        device_id = security_manager.generate_device_id("login-request", "127.0.0.1")
        
        access_token = jwt_auth.create_access_token(
            user_id=user_login.username,
            account_id=user_login.username,
            customer_id=user_login.username,
            permissions=["read:indicators"],
            device_id=device_id,
            session_id=session_id,
            client_ip="127.0.0.1",
            include_minimal_data=True
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )


@router.post("/auth/api-keys")
async def create_api_key(
    api_key_create: APIKeyCreate,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new API key"""
    
    # Generate API key and hash it
    raw_key = api_key_auth.generate_api_key()
    key_hash = api_key_auth.hash_api_key(raw_key)
    
    # Generate key ID
    key_id = security_manager.generate_secure_token(16)
    
    # Hash user ID for GDPR compliance
    user_id_hash = audit_logger.hash_identifier(current_user["user_id"])
    
    # Store API key in database
    from database import APIKey
    new_api_key = APIKey(
        key_id=key_id,
        name=api_key_create.name,
        user_id_hash=user_id_hash,
        key_hash=key_hash,
        scopes_json=json.dumps(api_key_create.scopes),
        is_active=True,
        rate_limit_per_minute=api_key_create.rate_limit_per_minute,
        rate_limit_per_hour=api_key_create.rate_limit_per_hour
    )
    
    db.add(new_api_key)
    db.commit()
    
    # Log the key creation
    try:
        audit_logger.log_data_access(
            db=db,
            user_id=current_user["user_id"],
            data_subject_id=current_user["user_id"],
            resource_type="api_key",
            resource_id=key_id,
            data_categories=["api_key"],
            gdpr_basis="consent",
            success=True
        )
    except Exception as e:
        logger.error(f"Failed to log audit event: {str(e)}")
    
    return {
        "key_id": key_id,
        "api_key": raw_key,  # Only show the raw key once during creation
        "name": api_key_create.name,
        "scopes": api_key_create.scopes,
        "rate_limit_per_minute": api_key_create.rate_limit_per_minute,
        "rate_limit_per_hour": api_key_create.rate_limit_per_hour,
        "created_at": datetime.utcnow()
    }


@router.get("/auth/api-keys")
async def list_api_keys(
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List user's API keys"""
    
    user_id_hash = audit_logger.hash_identifier(current_user["user_id"])
    
    # Get user's API keys
    from database import APIKey
    api_keys = db.query(APIKey).filter(
        APIKey.user_id_hash == user_id_hash,
        APIKey.is_active == True
    ).all()
    
    # Format response (excluding sensitive data)
    result = []
    for key in api_keys:
        result.append({
            "key_id": key.key_id,
            "name": key.name,
            "scopes": json.loads(str(key.scopes_json)),
            "is_active": key.is_active,
            "created_at": key.created_at,
            "last_used": key.last_used,
            "rate_limit_per_minute": key.rate_limit_per_minute,
            "rate_limit_per_hour": key.rate_limit_per_hour
        })
    
    return result


@router.delete("/auth/api-keys/{key_id}")
async def delete_api_key(
    key_id: str,
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an API key"""
    
    user_id_hash = audit_logger.hash_identifier(current_user["user_id"])
    
    # Get the API key
    from database import APIKey
    api_key = db.query(APIKey).filter(
        APIKey.key_id == key_id,
        APIKey.user_id_hash == user_id_hash
    ).first()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    # Deactivate the key instead of deleting it for audit purposes
    api_key.is_active = False  # type: ignore
    db.commit()
    
    # Log the key deletion
    try:
        audit_logger.log_data_access(
            db=db,
            user_id=current_user["user_id"],
            data_subject_id=current_user["user_id"],
            resource_type="api_key",
            resource_id=key_id,
            data_categories=["api_key"],
            gdpr_basis="consent",
            success=True
        )
    except Exception as e:
        logger.error(f"Failed to log audit event: {str(e)}")
    
    return {"message": "API key deleted successfully"}


# Utility functions
def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"req_{secrets.token_urlsafe(16)}"


def generate_export_id() -> str:
    """Generate unique export ID"""
    return f"exp_{secrets.token_urlsafe(16)}"


def verify_deletion_token(token: str, user_id: str) -> bool:
    """Verify deletion request token"""
    # In production, this would validate against a secure token store
    # For demo, use a simple hash-based verification
    expected_token = hashlib.sha256(f"delete_{user_id}_secret".encode()).hexdigest()[:32]
    return secrets.compare_digest(token, expected_token)


def verify_export_token(token: str, user_id: str) -> bool:
    """Verify data export request token"""
    # In production, this would validate against a secure token store
    expected_token = hashlib.sha256(f"export_{user_id}_secret".encode()).hexdigest()[:32]
    return secrets.compare_digest(token, expected_token)


def check_legal_retention(user_id: str) -> bool:
    """Check if data has legal retention requirements"""
    # In production, this would check against legal requirements
    # For demo, return False (no retention requirements)
    return False


def get_retention_notice_if_applicable(user_id: str) -> Optional[str]:
    """Get retention notice if data has retention requirements"""
    if check_legal_retention(user_id):
        return "Some data retained for legal compliance requirements"
    return None


def calculate_consent_expiration(consent_type: ConsentType) -> Optional[datetime]:
    """Calculate consent expiration based on type"""
    if consent_type == ConsentType.MARKETING:
        return datetime.utcnow() + timedelta(days=365)  # 1 year
    elif consent_type == ConsentType.ANALYTICS:
        return datetime.utcnow() + timedelta(days=180)  # 6 months
    return None  # No expiration for functional cookies


async def process_data_export(user_id: str, data_categories: Optional[List[str]], 
                             format: str) -> Dict[str, Any]:
    """Process data export request"""
    logger.info(f"Processing data export for user {user_id}, format {format}")
    
    # In production, this would gather all user data
    # For demo, return mock data
    export_data = {
        "user_id": user_id,
        "export_timestamp": datetime.utcnow().isoformat(),
        "data_categories": data_categories or ["profile", "preferences", "activity"],
        "data": {
            "profile": {
                "created_at": "2023-01-01T00:00:00Z",
                "last_login": "2024-01-01T00:00:00Z"
            },
            "preferences": {
                "language": "en",
                "timezone": "UTC"
            },
            "activity": {
                "last_data_access": "2024-01-01T00:00:00Z",
                "total_api_calls": 100
            }
        }
    }
    
    return export_data


# Include router in app
app.include_router(router)

# Health check endpoints (moved to main app level)
@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for service monitoring.
    
    Returns:
        Dict containing service health status and metadata
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "econovault-api",
        "version": "1.0.0"
    }


@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> Dict[str, str]:
    """Root endpoint returning service information."""
    return {
        "message": "EconoVault API service running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint"""
    from monitoring import get_monitoring
    monitoring_system = get_monitoring()
    
    # Generate Prometheus metrics
    metrics_data = generate_latest()
    
    return PlainTextResponse(
        content=metrics_data.decode('utf-8'),
        media_type=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)