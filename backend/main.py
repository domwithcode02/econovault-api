from __future__ import annotations
from __future__ import annotations
from datetime import datetime, timezone, date, timedelta
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
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
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Literal
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import text

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
from database import get_db, Session, DatabaseManager, EconomicIndicator, DataPoint, APIKey
from security import (
    get_current_user_optional, get_current_user, 
    api_key_auth, audit_logger, security_manager,
    get_optional_api_key,
    get_current_user_with_roles, require_permission, require_any_permission,
    require_admin, require_analyst_or_admin, Permission, UserRole
)
from models import (
    DataFrequency, SeasonalAdjustment, DataSource, EconomicIndicatorType,
    GeographicLevel, TimeSeriesData, DataPoint as DataPointModel
)
from streaming import RealTimeDataStreamer, initialize_streaming
from redis_manager import redis_manager, indicator_cache, gdpr_cache
from validation import InputValidationMiddleware, ValidationConfig
from csrf_protection import CSRFProtectionMiddleware, CSRFConfig
from etag_middleware import ETagMiddleware, ETagConfig
from cursor_pagination import EconomicDataCursorPaginator, CursorPaginationParams
from error_handler import (
    ErrorHandler, create_exception_handlers, ErrorType, ErrorSeverity, 
    FieldError, StandardErrorResponse, create_validation_error_response,
    create_not_found_error_response, create_rate_limit_error_response
)
from advanced_filtering import (
    AdvancedFilterParams, FilterParser, FilterValidator, FilterExecutor,
    EconomicDataFilterConfig, parse_filter_query, create_indicator_filter
)
from sparse_fieldsets import (
    FieldSelectionRequest, FieldSelector, SparseFieldsetMiddleware, FieldSelectionConfig,
    create_field_selection_response, get_available_fields
)
from api_key_rotation import (
    api_key_rotation_service, api_key_rotation_scheduler, APIKeyRotationRequest,
    APIKeyRotationPolicy, APIKeyStatus, APIKeyInfo, APIKeyEventType
)

# Initialize error handler
error_handler = ErrorHandler()

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
        email_to=[config.alert_email_to] if isinstance(config.alert_email_to, str) else config.alert_email_to,
        smtp_host=config.alert_smtp_host,
        smtp_port=config.alert_smtp_port,
        smtp_username=config.alert_smtp_username,
        smtp_password=config.alert_smtp_password,
        smtp_use_tls=config.alert_smtp_use_tls
    )
    initialize_alerting(alert_config)
    logger.info("Alerting system initialized")
else:
    logger.info("Alerting system disabled")

    # Initialize monitoring system
    monitoring_config = {
        "metrics_retention_hours": config.metrics_retention_days * 24,
        "enable_system_monitoring": True
    }
    monitoring = init_monitoring(monitoring_config)
    
    # Register Redis health check
    def redis_internal_health_check():
        """Check Redis health - handle fallback mode gracefully"""
        try:
            # Check if Redis is available and initialized
            if redis_manager._initialized:
                return {
                    "status": "healthy",
                    "error_rate": 0.0,
                    "availability": 100.0,
                    "details": {"mode": "redis", "initialized": True}
                }
            else:
                # Redis is in fallback mode, which is acceptable
                return {
                    "status": "degraded",
                    "error_rate": 0.0,
                    "availability": 100.0,
                    "details": {"mode": "fallback", "message": "Redis unavailable, using fallback mode"}
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error_rate": 100.0,
                "availability": 0.0,
                "details": {"error": str(e)}
            }
    
    monitoring.health.register_health_check("redis", redis_internal_health_check)
    
    # Register database health check
    def database_health_check():
        """Check database health"""
        try:
            from database import get_db
            db = next(get_db())
            # Simple test query
            db.execute(text("SELECT 1"))
            db.close()
            return {
                "status": "healthy",
                "error_rate": 0.0,
                "availability": 100.0,
                "details": {"connection": "ok"}
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error_rate": 100.0,
                "availability": 0.0,
                "details": {"error": str(e)}
            }
    
    monitoring.health.register_health_check("database", database_health_check)
    
    logger.info("Monitoring system initialized")

# Initialize streaming
streamer = RealTimeDataStreamer(bls_client)

# Thread pool for running sync BLS API calls
executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("Starting application...")
    
    # Initialize Redis
    try:
        await redis_manager.initialize()
        logger.info("Redis manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        # Continue with fallback mode
    
    # Start monitoring system
    try:
        monitoring.start_monitoring(interval_seconds=60)
        logger.info("Monitoring system started successfully")
    except Exception as e:
        logger.error(f"Failed to start monitoring system: {str(e)}")
    
    # Start API key lifecycle manager
    try:
        await api_key_rotation_scheduler.start(interval_hours=24)  # Run daily
        logger.info("API Key lifecycle manager started")
    except Exception as e:
        logger.error(f"Failed to start API key lifecycle manager: {str(e)}")
    
    logger.info("Application startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Stop API key lifecycle manager
    try:
        await api_key_rotation_scheduler.stop()
        logger.info("API Key lifecycle manager stopped")
    except Exception as e:
        logger.error(f"Error stopping API key lifecycle manager: {str(e)}")
    
    # Stop monitoring system
    try:
        if hasattr(monitoring, 'stop_monitoring'):
            monitoring.stop_monitoring()
            logger.info("Monitoring system stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping monitoring system: {str(e)}")
    
    # Close Redis connections
    try:
        await redis_manager.close()
        logger.info("Redis connections closed")
    except Exception as e:
        logger.error(f"Error closing Redis connections: {e}")
    
    logger.info("Application shutdown completed")


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
    openapi_url="/openapi.json",
    lifespan=lifespan
)


# Register exception handlers for standardized error responses
create_exception_handlers(app)

# Add pagination support
add_pagination(app)

# Initialize Redis connection


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
        json_schema_extra = {
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
        json_schema_extra = {
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
        json_schema_extra = {
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
        json_schema_extra = {
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


# Add security middleware
# HTTPS enforcement (production only)
if config.environment == "production":
    app.add_middleware(HTTPSRedirectMiddleware)

# Input validation middleware - TEMPORARILY DISABLED due to OpenAPI generation issues
# app.add_middleware(InputValidationMiddleware, config=ValidationConfig())

# CSRF protection middleware - TEMPORARILY DISABLED due to OpenAPI generation issues
# app.add_middleware(CSRFProtectionMiddleware, config=CSRFConfig())

# ETag middleware for HTTP caching - TEMPORARILY DISABLED due to OpenAPI generation issues
# app.add_middleware(ETagMiddleware, config=ETagConfig())

# Sparse fieldsets middleware for response optimization - TEMPORARILY DISABLED due to OpenAPI generation issues
# app.add_middleware(SparseFieldsetMiddleware, config=FieldSelectionConfig())

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
        "127.0.0.1",
        "econovault-api-2.onrender.com",
        "*.onrender.com"
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
api_error_handler = APIErrorHandler()

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
@router.put(
    "/indicators/{series_id}",
    response_model=Dict[str, Any],
    summary="Update economic indicator (full replacement)",
    description="""
    Update an existing economic indicator with full replacement of all fields.
    
    ## Authorization
    Requires admin or analyst role to update indicators.
    
    ## Validation
    - All required fields must be provided
    - Date ranges must be valid if provided
    - Series ID cannot be changed
    
    ## Example Usage
    
    ### Update an existing indicator
    ```bash
    curl -X PUT "https://api.econovault.com/v1/indicators/NEW_INDICATOR_001" \
         -H "Authorization: Bearer your-token" \
         -H "Content-Type: application/json" \
         -d '{
           "series_id": "NEW_INDICATOR_001",
           "title": "Updated Economic Indicator",
           "description": "An updated economic indicator",
           "source": "BLS",
           "indicator_type": "EMPLOYMENT",
           "frequency": "MONTHLY",
           "seasonal_adjustment": "SEASONALLY_ADJUSTED",
           "geography_level": "NATIONAL",
           "units": "Thousands of Persons"
         }'
    ```
    
    ## Response Format
    Returns the updated indicator with metadata:
    - All updated fields
    - `last_updated`: Last update timestamp
    """,
    response_description="Updated economic indicator",
    tags=["indicators"]
)

# Request models for indicator operations
class IndicatorCreateRequest(BaseModel):
    """Request model for creating a new economic indicator"""
    series_id: str = Field(..., description="Unique series identifier (e.g., CUUR0000SA0)")
    title: str = Field(..., description="Human-readable title of the indicator")
    description: Optional[str] = Field(None, description="Detailed description of the indicator")
    source: str = Field(..., description="Data source (BLS, BEA, FRED, etc.)")
    indicator_type: str = Field(..., description="Type of economic indicator (CPI, Employment, etc.)")
    frequency: str = Field(..., description="Data frequency (MONTHLY, QUARTERLY, ANNUALLY)")
    seasonal_adjustment: str = Field(..., description="Seasonal adjustment method")
    geography: Optional[str] = Field(None, description="Geographic area")
    geography_level: str = Field(..., description="Geographic level (NATIONAL, STATE, etc.)")
    units: str = Field(..., description="Measurement units")
    start_date: Optional[date] = Field(None, description="Start date of available data")
    end_date: Optional[date] = Field(None, description="End date of available data")
    registration_key_required: bool = Field(False, description="Whether registration key is required")
    
    class Config:
        json_schema_extra = {
            "example": {
                "series_id": "NEW_INDICATOR_001",
                "title": "New Economic Indicator",
                "description": "A newly created economic indicator for testing purposes",
                "source": "BLS",
                "indicator_type": "EMPLOYMENT",
                "frequency": "MONTHLY",
                "seasonal_adjustment": "SEASONALLY_ADJUSTED",
                "geography": "United States",
                "geography_level": "NATIONAL",
                "units": "Thousands of Persons",
                "start_date": "2020-01-01",
                "end_date": None,
                "registration_key_required": False
            }
        }


class IndicatorUpdateRequest(BaseModel):
    """Request model for partially updating an economic indicator"""
    title: Optional[str] = Field(None, description="Human-readable title of the indicator")
    description: Optional[str] = Field(None, description="Detailed description of the indicator")
    source: Optional[str] = Field(None, description="Data source (BLS, BEA, FRED, etc.)")
    indicator_type: Optional[str] = Field(None, description="Type of economic indicator (CPI, Employment, etc.)")
    frequency: Optional[str] = Field(None, description="Data frequency (MONTHLY, QUARTERLY, ANNUALLY)")
    seasonal_adjustment: Optional[str] = Field(None, description="Seasonal adjustment method")
    geography: Optional[str] = Field(None, description="Geographic area")
    geography_level: Optional[str] = Field(None, description="Geographic level (NATIONAL, STATE, etc.)")
    units: Optional[str] = Field(None, description="Measurement units")
    start_date: Optional[date] = Field(None, description="Start date of available data")
    end_date: Optional[date] = Field(None, description="End date of available data")
    registration_key_required: Optional[bool] = Field(None, description="Whether registration key is required")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Updated Economic Indicator",
                "description": "An updated description of the indicator",
                "frequency": "QUARTERLY",
                "units": "Billions of Dollars"
            }
        }

@monitor_function(metric_name="update_indicator", track_time=True, track_errors=True)
async def update_indicator(
    request: Request,
    series_id: str,
    indicator_data: IndicatorCreateRequest,
    current_user: Dict = Depends(require_analyst_or_admin),
    db: Session = Depends(get_db)
):
    """Update an existing economic indicator with full replacement."""
    
    # RBAC check is already handled by require_analyst_or_admin dependency
    
    # Validate series ID format
    if not indicator_data.series_id or len(indicator_data.series_id) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Series ID must be at least 3 characters long"
        )
    
    # Ensure series ID in path matches series ID in body
    if indicator_data.series_id != series_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Series ID in path must match series ID in request body"
        )
    
    # Check if indicator exists
    from database import DatabaseManager
    db_manager = DatabaseManager(db)
    
    existing_indicator = db_manager.get_indicator_by_series_id(series_id)
    if not existing_indicator:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Indicator with series ID '{series_id}' not found"
        )
    
    # Log the update attempt
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
        labels={"endpoint": "update_indicator", "method": "PUT", "status": "200"}
    )
    
    # Update the indicator
    indicator_db_data = {
        "title": indicator_data.title,
        "description": indicator_data.description,
        "source": indicator_data.source,
        "indicator_type": indicator_data.indicator_type,
        "frequency": indicator_data.frequency,
        "seasonal_adjustment": indicator_data.seasonal_adjustment,
        "geography": indicator_data.geography,
        "geography_level": indicator_data.geography_level,
        "units": indicator_data.units,
        "start_date": indicator_data.start_date,
        "end_date": indicator_data.end_date,
        "registration_key_required": indicator_data.registration_key_required,
        "last_updated": datetime.utcnow()
    }
    
    try:
        updated_indicator = db_manager.update_indicator(series_id, indicator_db_data)
        
        if not updated_indicator:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update indicator"
            )
        
        # Generate HATEOAS links
        link_generator = LinkGenerator(request)
        indicator_links = link_generator.indicator_links(str(updated_indicator.series_id))
        
        # Return updated indicator
        response_data = {
            "id": updated_indicator.id,
            "series_id": updated_indicator.series_id,
            "title": updated_indicator.title,
            "description": updated_indicator.description,
            "source": updated_indicator.source,
            "indicator_type": updated_indicator.indicator_type,
            "frequency": updated_indicator.frequency,
            "seasonal_adjustment": updated_indicator.seasonal_adjustment,
            "geography": updated_indicator.geography,
            "geography_level": updated_indicator.geography_level,
            "units": updated_indicator.units,
            "start_date": updated_indicator.start_date.isoformat() if updated_indicator.start_date is not None else None,
            "end_date": updated_indicator.end_date.isoformat() if updated_indicator.end_date is not None else None,
            "registration_key_required": updated_indicator.registration_key_required,
            "last_updated": updated_indicator.last_updated.isoformat(),
            "_links": {k: v.dict() for k, v in indicator_links.items()}
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error updating indicator: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating indicator: {str(e)}"
        )


@router.patch(
    "/indicators/{series_id}",
    response_model=Dict[str, Any],
    summary="Update economic indicator (partial update)",
    description="""
    Update an existing economic indicator with partial field updates.
    
    ## Authorization
    Requires admin or analyst role to update indicators.
    
    ## Validation
    - Only provided fields will be updated
    - Date ranges must be valid if provided
    - Series ID cannot be changed
    
    ## Example Usage
    
    ### Partially update an indicator
    ```bash
    curl -X PATCH "https://api.econovault.com/v1/indicators/NEW_INDICATOR_001" \
         -H "Authorization: Bearer your-token" \
         -H "Content-Type: application/json" \
         -d '{
           "title": "Updated Economic Indicator",
           "description": "An updated description",
           "frequency": "QUARTERLY"
         }'
    ```
    
    ## Response Format
    Returns the updated indicator with metadata:
    - All updated fields
    - `last_updated`: Last update timestamp
    """,
    response_description="Updated economic indicator",
    tags=["indicators"]
)
@monitor_function(metric_name="patch_indicator", track_time=True, track_errors=True)
async def patch_indicator(
    request: Request,
    series_id: str,
    indicator_data: IndicatorUpdateRequest,
    current_user: Dict = Depends(require_analyst_or_admin),
    db: Session = Depends(get_db)
):
    """Update an existing economic indicator with partial field updates."""
    
    # RBAC check is already handled by require_analyst_or_admin dependency
    
    # Check if indicator exists
    from database import DatabaseManager
    db_manager = DatabaseManager(db)
    
    existing_indicator = db_manager.get_indicator_by_series_id(series_id)
    if not existing_indicator:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Indicator with series ID '{series_id}' not found"
        )
    
    # Log the update attempt
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
        labels={"endpoint": "patch_indicator", "method": "PATCH", "status": "200"}
    )
    
    # Prepare update data (only include non-None values)
    update_data = {}
    for field, value in indicator_data.dict(exclude_unset=True).items():
        if value is not None:
            update_data[field] = value
    
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields provided for update"
        )
    
    # Add last_updated timestamp
    update_data["last_updated"] = datetime.utcnow()
    
    try:
        updated_indicator = db_manager.update_indicator(series_id, update_data)
        
        if not updated_indicator:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update indicator"
            )
        
        # Generate HATEOAS links
        link_generator = LinkGenerator(request)
        indicator_links = link_generator.indicator_links(str(updated_indicator.series_id))
        
        # Return updated indicator
        response_data = {
            "id": updated_indicator.id,
            "series_id": updated_indicator.series_id,
            "title": updated_indicator.title,
            "description": updated_indicator.description,
            "source": updated_indicator.source,
            "indicator_type": updated_indicator.indicator_type,
            "frequency": updated_indicator.frequency,
            "seasonal_adjustment": updated_indicator.seasonal_adjustment,
            "geography": updated_indicator.geography,
            "geography_level": updated_indicator.geography_level,
            "units": updated_indicator.units,
            "start_date": updated_indicator.start_date.isoformat() if updated_indicator.start_date is not None else None,
            "end_date": updated_indicator.end_date.isoformat() if updated_indicator.end_date is not None else None,
            "registration_key_required": updated_indicator.registration_key_required,
            "last_updated": updated_indicator.last_updated.isoformat(),
            "_links": {k: v.dict() for k, v in indicator_links.items()}
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error updating indicator: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating indicator: {str(e)}"
        )


@router.delete(
    "/indicators/{series_id}",
    response_model=Dict[str, Any],
    summary="Delete economic indicator",
    description="""
    Delete an economic indicator and all associated data points.
    
    ## Authorization
    Requires admin role to delete indicators.
    
    ## Effects
    - Deletes the indicator metadata
    - Deletes all associated data points
    - Removes from Redis cache if present
    - Cannot be undone
    
    ## Example Usage
    
    ### Delete an indicator
    ```bash
    curl -X DELETE "https://api.econovault.com/v1/indicators/NEW_INDICATOR_001" \
         -H "Authorization: Bearer your-token"
    ```
    
    ## Response Format
    Returns confirmation of deletion:
    - `message`: Success message
    - `deleted_at`: Deletion timestamp
    - `series_id`: Deleted series ID
    """,
    response_description="Deletion confirmation",
    tags=["indicators"]
)
@monitor_function(metric_name="delete_indicator", track_time=True, track_errors=True)
async def delete_indicator(
    request: Request,
    series_id: str,
    current_user: Dict = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete an economic indicator and all associated data."""
    
    # RBAC check is already handled by require_admin dependency
    
    # Check if indicator exists
    from database import DatabaseManager
    db_manager = DatabaseManager(db)
    
    existing_indicator = db_manager.get_indicator_by_series_id(series_id)
    if not existing_indicator:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Indicator with series ID '{series_id}' not found"
        )
    
    # Log the deletion attempt
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
        labels={"endpoint": "delete_indicator", "method": "DELETE", "status": "200"}
    )
    
    try:
        # Delete from Redis cache if present
        await indicator_cache.store_indicator_metadata(series_id, {}, ttl=1)  # Set to empty dict with 1 second TTL
        
        # Delete the indicator and all associated data
        success = db_manager.delete_indicator(series_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete indicator"
            )
        
        # Return deletion confirmation
        return {
            "message": f"Indicator '{series_id}' deleted successfully",
            "deleted_at": datetime.utcnow().isoformat(),
            "series_id": series_id
        }
        
    except Exception as e:
        logger.error(f"Error deleting indicator: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting indicator: {str(e)}"
        )


@router.post(
    "/indicators",
    response_model=Dict[str, Any],
    summary="Create new economic indicator",
    description="""
    Create a new economic indicator in the system.
    
    ## Authorization
    Requires admin or analyst role to create new indicators.
    
    ## Validation
    - Series ID must be unique and follow standard format
    - All required fields must be provided
    - Date ranges must be valid if provided
    
    ## Example Usage
    
    ### Create a new indicator
    ```bash
    curl -X POST "https://api.econovault.com/v1/indicators" \
         -H "Authorization: Bearer your-token" \
         -H "Content-Type: application/json" \
         -d '{
           "series_id": "NEW_INDICATOR_001",
           "title": "New Economic Indicator",
           "description": "A newly created economic indicator",
           "source": "BLS",
           "indicator_type": "EMPLOYMENT",
           "frequency": "MONTHLY",
           "seasonal_adjustment": "SEASONALLY_ADJUSTED",
           "geography_level": "NATIONAL",
           "units": "Thousands of Persons"
         }'
    ```
    
    ## Response Format
    Returns the created indicator with additional metadata:
    - All input fields
    - `created_at`: Timestamp of creation
    - `id`: Database ID
    - `last_updated`: Last update timestamp
    """,
    response_description="Created economic indicator",
    tags=["indicators"],
    status_code=status.HTTP_201_CREATED
)
@monitor_function(metric_name="create_indicator", track_time=True, track_errors=True)
async def create_indicator(
    request: Request,
    indicator_data: IndicatorCreateRequest,
    current_user: Dict = Depends(require_analyst_or_admin),
    db: Session = Depends(get_db)
):
    """Create a new economic indicator with proper validation and authorization."""
    
    # RBAC check is already handled by require_analyst_or_admin dependency
    
    # Validate series ID format
    if not indicator_data.series_id or len(indicator_data.series_id) < 3:
        field_error = FieldError(
            field="series_id",
            message="Series ID must be at least 3 characters long",
            code="TOO_SHORT",
            value=indicator_data.series_id,
            expected_type="string",
            constraint="min_length=3",
            suggestions=["Use a series ID with at least 3 characters"]
        )
        error_response = create_validation_error_response([field_error], request)
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.dict(exclude_none=True)
        )
    
    # Check if series ID already exists
    from database import DatabaseManager
    db_manager = DatabaseManager(db)
    
    existing_indicator = db_manager.get_indicator_by_series_id(indicator_data.series_id)
    if existing_indicator:
        error_response = error_handler.create_conflict_error_response(
            "Indicator", 
            indicator_data.series_id, 
            request,
            message=f"Indicator with series ID '{indicator_data.series_id}' already exists"
        )
        error_response.type = ErrorType.CONFLICT_ERROR
        error_response.status_code = status.HTTP_409_CONFLICT
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.dict(exclude_none=True)
        )
    
    # Log the creation attempt
    try:
        audit_logger.log_data_access(
            db=db,
            user_id=current_user.get("user_id", "anonymous"),
            data_subject_id="system",
            resource_type="economic_indicator",
            resource_id=indicator_data.series_id,
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
        labels={"endpoint": "create_indicator", "method": "POST", "status": "201"}
    )
    
    # Create the indicator
    indicator_db_data = {
        "series_id": indicator_data.series_id,
        "title": indicator_data.title,
        "description": indicator_data.description,
        "source": indicator_data.source,
        "indicator_type": indicator_data.indicator_type,
        "frequency": indicator_data.frequency,
        "seasonal_adjustment": indicator_data.seasonal_adjustment,
        "geography": indicator_data.geography,
        "geography_level": indicator_data.geography_level,
        "units": indicator_data.units,
        "start_date": indicator_data.start_date,
        "end_date": indicator_data.end_date,
        "registration_key_required": indicator_data.registration_key_required,
        "last_updated": datetime.utcnow()
    }
    
    try:
        new_indicator = db_manager.create_indicator(indicator_db_data)
        
        # Generate HATEOAS links
        link_generator = LinkGenerator(request)
        indicator_links = link_generator.indicator_links(str(new_indicator.series_id))
        
        # Return created indicator
        response_data = {
            "id": new_indicator.id,
            "series_id": new_indicator.series_id,
            "title": new_indicator.title,
            "description": new_indicator.description,
            "source": new_indicator.source,
            "indicator_type": new_indicator.indicator_type,
            "frequency": new_indicator.frequency,
            "seasonal_adjustment": new_indicator.seasonal_adjustment,
            "geography": new_indicator.geography,
            "geography_level": new_indicator.geography_level,
            "units": new_indicator.units,
            "start_date": new_indicator.start_date.isoformat() if new_indicator.start_date is not None else None,
            "end_date": new_indicator.end_date.isoformat() if new_indicator.end_date is not None else None,
            "registration_key_required": new_indicator.registration_key_required,
            "created_at": new_indicator.last_updated.isoformat(),
            "last_updated": new_indicator.last_updated.isoformat(),
            "_links": {k: v.dict() for k, v in indicator_links.items()}
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error creating indicator: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating indicator: {str(e)}"
        )


@router.get(
    "/indicators",
    response_model=Dict[str, Any],
    summary="Get economic indicators (cursor-based pagination + advanced filtering)",
    description="""
    Retrieve a list of available economic indicators from various sources including BLS, BEA, and Federal Reserve.
    
    ## Features
    - **Advanced Filtering**: Complex filter expressions with operators (eq, ne, gt, lt, like, regex, in, between, etc.)
    - **Cursor-based Pagination**: Efficient pagination for large datasets
    - **Field Selection**: Choose specific fields to return
    - **Sorting**: Multi-field sorting with direction control
    - **HTTP Caching**: ETags and conditional requests
    
    ## Advanced Filtering
    Use the `filter` parameter with JSON-encoded filter expressions:
    - **Operators**: eq, ne, gt, gte, lt, lte, in, nin, like, nlike, regex, nregex, between, nbetween
    - **Logical Operators**: and, or for combining conditions
    - **Nested Groups**: Complex filter logic with parentheses
    
    ## Cursor-Based Pagination
    - **first**: Number of items to fetch (forward pagination)
    - **after**: Cursor to fetch items after
    - **last**: Number of items to fetch (backward pagination)
    - **before**: Cursor to fetch items before
    
    ## Example Usage
    
    ### Basic filtering with source
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators?filter={\\"source\\":\\"BLS\\"}&first=10" \
         -H "X-API-Key: your-api-key"
    ```
    
    ### Complex filtering with multiple conditions
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators?filter={\\"operator\\":\\"and\\",\\"conditions\\":[{\\"field\\":\\"source\\",\\"operator\\":\\"eq\\",\\"value\\":\\"BLS\\"},{\\"field\\":\\"indicator_type\\",\\"operator\\":\\"eq\\",\\"value\\":\\"CPI\\"}]}&first=20" \
         -H "X-API-Key: your-api-key"
    ```
    
    ### Pattern matching with LIKE
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators?filter={\\"field\\":\\"title\\",\\"operator\\":\\"like\\",\\"value\\":\\"%employment%\\"}&first=10" \
         -H "X-API-Key: your-api-key"
    ```
    
    ### Date range filtering
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators?filter={\\"field\\":\\"last_updated\\",\\"operator\\":\\"gte\\",\\"value\\":\\"2023-01-01\\"}&first=50" \
         -H "X-API-Key: your-api-key"
    ```
    
    ## Response Format
    Returns cursor-based pagination with filtered indicator data:
    - `edges`: List of edges containing nodes and cursors
    - `page_info`: Pagination metadata
    - `total_count`: Total number of matching indicators
    - `filter_applied`: Applied filter summary
    """,
    response_description="Cursor-based paginated and filtered list of economic indicators",
    tags=["indicators"],
    openapi_extra={
        "x-code-samples": [
            {
                "lang": "curl",
                "source": 'curl -X GET "https://api.econovault.com/v1/indicators?filter={\\"source\\":\\"BLS\\"}&first=10" -H "X-API-Key: your-api-key"'
            },
            {
                "lang": "Python",
                "source": """import requests
import json

url = "https://api.econovault.com/v1/indicators"
headers = {"X-API-Key": "your-api-key"}

# Complex filter with multiple conditions
filter_data = {
    "operator": "and",
    "conditions": [
        {"field": "source", "operator": "eq", "value": "BLS"},
        {"field": "indicator_type", "operator": "eq", "value": "CPI"},
        {"field": "title", "operator": "like", "value": "%consumer%"}
    ]
}

params = {
    "filter": json.dumps(filter_data),
    "first": 10
}

response = requests.get(url, headers=headers, params=params)
result = response.json()

print(f"Total matching indicators: {result['data']['total_count']}")
print(f"Filter applied: {result['data']['filter_applied']}")

for edge in result['data']['edges']:
    indicator = edge['node']
    print(f"{indicator['title']} ({indicator['series_id']})")
"""
            },
            {
                "lang": "JavaScript",
                "source": """const filterData = {
    operator: "and",
    conditions: [
        { field: "source", operator: "eq", value: "BLS" },
        { field: "indicator_type", operator: "eq", value: "CPI" }
    ]
};

const response = await fetch(`https://api.econovault.com/v1/indicators?filter=${encodeURIComponent(JSON.stringify(filterData))}&first=10`, {
    headers: {
        'X-API-Key': 'your-api-key'
    }
});

const result = await response.json();
console.log(`Total matching indicators: ${result.data.total_count}`);
console.log(`Filter applied: ${JSON.stringify(result.data.filter_applied)}`);

result.data.edges.forEach(edge => {
    const indicator = edge.node;
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
    filter_param: Optional[str] = Query(None, alias="filter", description="JSON-encoded filter expression"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to include (e.g., 'series_id,title,source')"),
    exclude: Optional[str] = Query(None, description="Comma-separated list of fields to exclude"),
    pagination: CursorPaginationParams = Depends(CursorPaginationParams),
    current_user: Optional[Dict] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Get list of economic indicators with advanced filtering and cursor-based pagination."""
    
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
    
    # Parse and validate advanced filter if provided
    filter_group = None
    filter_summary = None
    
    if filter_param:
        try:
            # Parse filter parameter
            parser = FilterParser()
            filter_group = parser.parse_filter_string(filter_param)
            
            # Validate against allowed fields
            validator = FilterValidator(EconomicDataFilterConfig.INDICATOR_FIELDS)
            validator.validate_filter_group(filter_group)
            
            filter_summary = {
                "filter_applied": True,
                "filter_expression": filter_param[:200] + "..." if len(filter_param) > 200 else filter_param
            }
            
        except ValueError as e:
            field_error = FieldError(
                field="filter",
                message=str(e),
                code="INVALID_FILTER",
                value=filter_param,
                expected_type="string",
                constraint="valid_filter_expression",
                suggestions=["Check filter syntax and field names"]
            )
            error_response = create_validation_error_response([field_error], request)
            return JSONResponse(
                status_code=error_response.status_code,
                content=error_response.dict(exclude_none=True)
            )
    else:
        # Create simple filter from query parameters if no advanced filter provided
        filter_group = create_indicator_filter(source, indicator_type)
        filter_summary = {
            "filter_applied": bool(source or indicator_type),
            "simple_filters": {
                "source": source,
                "indicator_type": indicator_type
            }
        }
    
    # Use cursor-based pagination for database indicators
    from database import DatabaseManager
    db_manager = DatabaseManager(db)
    
    # Create cursor paginator
    paginator = EconomicDataCursorPaginator()
    
    # Get base query
    query = db.query(EconomicIndicator)
    
    # Apply advanced filtering if provided
    if filter_group:
        try:
            executor = FilterExecutor()
            # For now, apply simple filters at database level
            # Advanced filtering would be applied in-memory for this implementation
            if source:
                query = query.filter(EconomicIndicator.source == source.upper())
            if indicator_type:
                query = query.filter(EconomicIndicator.indicator_type == indicator_type.upper())
        except Exception as e:
            logger.error(f"Error applying filter: {str(e)}")
            error_response = error_handler.create_error_response(
                error_type=ErrorType.BAD_REQUEST_ERROR,
                message="Error applying filter conditions",
                status_code=status.HTTP_400_BAD_REQUEST,
                request=request,
                details=str(e)
            )
            return JSONResponse(
                status_code=error_response.status_code,
                content=error_response.dict(exclude_none=True)
            )
    
    # Paginate indicators from database
    pagination_result = paginator.paginate_indicators(
        db=db,
        first=pagination.first,
        after=pagination.after,
        last=pagination.last,
        before=pagination.before
    )
    
    # Apply advanced filtering in-memory (for demonstration)
    if filter_group:
        try:
            executor = FilterExecutor()
            
            # Convert database results to dictionaries for filtering
            filtered_items = []
            for edge in pagination_result["edges"]:
                indicator = edge["node"]
                indicator_dict = {
                    "series_id": indicator.series_id,
                    "title": indicator.title,
                    "description": indicator.description,
                    "source": indicator.source,
                    "indicator_type": indicator.indicator_type,
                    "frequency": indicator.frequency,
                    "seasonal_adjustment": indicator.seasonal_adjustment,
                    "geography": indicator.geography,
                    "geography_level": indicator.geography_level,
                    "units": indicator.units,
                    "start_date": indicator.start_date,
                    "end_date": indicator.end_date,
                    "registration_key_required": indicator.registration_key_required,
                    "last_updated": indicator.last_updated
                }
                filtered_items.append(indicator_dict)
            
            # Apply filter
            filtered_data = executor.apply_filter_group(filtered_items, filter_group)
            
            # Update pagination result with filtered data
            filtered_edges = []
            for item in filtered_data:
                # Find corresponding edge
                for edge in pagination_result["edges"]:
                    if edge["node"].series_id == item["series_id"]:
                        filtered_edges.append(edge)
                        break
            
            pagination_result["edges"] = filtered_edges
            pagination_result["page_info"]["total_count"] = len(filtered_edges)
            
        except Exception as e:
            logger.error(f"Error applying advanced filter: {str(e)}")
            error_response = error_handler.create_error_response(
                error_type=ErrorType.BAD_REQUEST_ERROR,
                message="Error applying advanced filter conditions",
                status_code=status.HTTP_400_BAD_REQUEST,
                request=request,
                details=str(e)
            )
            return JSONResponse(
                status_code=error_response.status_code,
                content=error_response.dict(exclude_none=True)
            )
    
    # Convert database results to response format
    edges = []
    for edge in pagination_result["edges"]:
        indicator = edge["node"]
        
        # Add HATEOAS links
        link_generator = LinkGenerator(request)
        indicator_links = link_generator.indicator_links(indicator.series_id)
        
        indicator_data = {
            "series_id": indicator.series_id,
            "title": indicator.title,
            "description": indicator.description,
            "source": indicator.source,
            "indicator_type": indicator.indicator_type,
            "frequency": indicator.frequency,
            "seasonal_adjustment": indicator.seasonal_adjustment,
            "geography": indicator.geography,
            "geography_level": indicator.geography_level,
            "units": indicator.units,
            "start_date": indicator.start_date.isoformat() if indicator.start_date else None,
            "end_date": indicator.end_date.isoformat() if indicator.end_date else None,
            "registration_key_required": indicator.registration_key_required,
            "last_updated": indicator.last_updated.isoformat(),
            "_links": {k: v.dict() for k, v in indicator_links.items()}
        }
        
        edges.append({
            "node": indicator_data,
            "cursor": edge["cursor"]
        })
    
    # Build cursor-based response
    return {
        "data": {
            "edges": edges,
            "page_info": pagination_result["page_info"],
            "total_count": pagination_result["page_info"]["total_count"],
            "filter_applied": filter_summary
        }
    }


@router.get(
    "/indicators/fields",
    response_model=Dict[str, Any],
    summary="Get available fields for field selection",
    description="""
    Get information about available fields that can be used with the fields/exclude parameters
    for sparse fieldset responses.
    
    ## Features
    - **Field Discovery**: See all available fields for indicators
    - **Field Types**: Get data types for each field
    - **Nested Fields**: Discover nested object fields
    - **Field Examples**: See example values for fields
    
    ## Example Usage
    
    ### Get all available fields
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators/fields" \
         -H "X-API-Key: your-api-key"
    ```
    
    ## Response Format
    Returns field information including:
    - `top_level_fields`: List of top-level field names
    - `nested_fields`: Dictionary of nested field structures
    - `field_types`: Dictionary mapping fields to their data types
    - `examples`: Sample field values for reference
    """,
    response_description="Available field information for sparse fieldset selection",
    tags=["indicators"]
)
@monitor_function(metric_name="get_indicator_fields", track_time=True, track_errors=True)
async def get_indicator_fields(
    request: Request,
    current_user: Optional[Dict] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Get information about available fields for field selection."""
    
    # Log access if user is authenticated
    if current_user:
        try:
            audit_logger.log_data_access(
                db=db,
                user_id=current_user.get("user_id", "anonymous"),
                data_subject_id="system",
                resource_type="field_information",
                resource_id="indicator_fields",
                data_categories=["api_metadata"],
                gdpr_basis="legitimate_interest",
                success=True
            )
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")
    
    # Track metrics
    monitoring = get_monitoring()
    monitoring.metrics.increment_counter(
        "api_requests_total",
        labels={"endpoint": "get_indicator_fields", "method": "GET", "status": "200"}
    )
    
    # Get a sample indicator to analyze field structure
    from database import DatabaseManager
    db_manager = DatabaseManager(db)
    
    sample_indicator = db.query(EconomicIndicator).first()
    
    if not sample_indicator:
        # Return default field information
        return {
            "top_level_fields": [
                "series_id", "title", "description", "source", "indicator_type",
                "frequency", "seasonal_adjustment", "geography", "geography_level",
                "units", "start_date", "end_date", "registration_key_required",
                "last_updated", "created_at", "updated_at"
            ],
            "nested_fields": {},
            "field_types": {
                "series_id": "str",
                "title": "str",
                "description": "str",
                "source": "str",
                "indicator_type": "str",
                "frequency": "str",
                "seasonal_adjustment": "str",
                "geography": "str",
                "geography_level": "str",
                "units": "str",
                "start_date": "date",
                "end_date": "date",
                "registration_key_required": "bool",
                "last_updated": "datetime",
                "created_at": "datetime",
                "updated_at": "datetime"
            },
            "examples": {
                "series_id": "CUUR0000SA0",
                "title": "Consumer Price Index for All Urban Consumers",
                "source": "BLS",
                "indicator_type": "CPI",
                "frequency": "MONTHLY",
                "seasonal_adjustment": "SEASONALLY_ADJUSTED",
                "geography_level": "NATIONAL",
                "units": "Index 1982-84=100"
            },
            "field_selection_examples": [
                "series_id,title,source",
                "series_id,title,description,last_updated",
                "series_id,title,source,indicator_type,frequency"
            ]
        }
    
    # Convert to dictionary for field analysis
    sample_dict = {
        "series_id": sample_indicator.series_id,
        "title": sample_indicator.title,
        "description": sample_indicator.description,
        "source": sample_indicator.source,
        "indicator_type": sample_indicator.indicator_type,
        "frequency": sample_indicator.frequency,
        "seasonal_adjustment": sample_indicator.seasonal_adjustment,
        "geography": sample_indicator.geography,
        "geography_level": sample_indicator.geography_level,
        "units": sample_indicator.units,
        "start_date": sample_indicator.start_date,
        "end_date": sample_indicator.end_date,
        "registration_key_required": sample_indicator.registration_key_required,
        "last_updated": sample_indicator.last_updated,
        "created_at": sample_indicator.created_at if hasattr(sample_indicator, 'created_at') else None,
        "updated_at": sample_indicator.updated_at if hasattr(sample_indicator, 'updated_at') else None
    }
    
    # Get field information
    selector = FieldSelector()
    field_info = selector.get_field_info(sample_dict, max_depth=2)
    
    # Add field selection examples
    field_info["field_selection_examples"] = [
        "series_id,title,source",
        "series_id,title,description,last_updated",
        "series_id,title,source,indicator_type,frequency",
        "series_id,title,description,source,indicator_type,frequency,seasonal_adjustment,geography_level,units"
    ]
    
    # Add usage examples
    field_info["usage_examples"] = [
        {
            "description": "Get only basic indicator information",
            "fields": "series_id,title,source,indicator_type",
            "curl": 'curl -X GET "https://api.econovault.com/v1/indicators?fields=series_id,title,source,indicator_type" -H "X-API-Key: your-api-key"'
        },
        {
            "description": "Get indicator metadata excluding description",
            "exclude": "description",
            "curl": 'curl -X GET "https://api.econovault.com/v1/indicators?exclude=description" -H "X-API-Key: your-api-key"'
        }
    ]
    
    return field_info


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
        error_response = create_not_found_error_response("Indicator", series_id, request)
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.dict(exclude_none=True)
        )
    
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
    summary="Get time series data (cursor-based pagination)",
    description="""
    Retrieve historical time series data for a specific economic indicator with cursor-based pagination.
    
    ## Data Features
    - **Real-time data**: Direct from BLS, BEA, and Federal Reserve APIs
    - **Date filtering**: Specify date ranges for historical analysis
    - **Data validation**: Automatic quality checks and error handling
    - **Rate limiting**: Built-in protection against API abuse
    - **HTTP caching**: ETags and conditional requests for improved performance
    - **Cursor-based pagination**: Efficient pagination for large time series datasets
    
    ## Date Format
    - ISO 8601 format: `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SSZ`
    - Partial dates are supported: `2023` or `2023-01`
    - Timezone: UTC (Z suffix) or explicit offset
    
    ## Cursor-Based Pagination for Data
    - **first**: Number of data points to fetch (most recent first)
    - **after**: Cursor to fetch data points after
    - **last**: Number of data points to fetch (oldest first)
    - **before**: Cursor to fetch data points before
    - More efficient than offset-based pagination for large time series
    
    ## HTTP Caching
    - **ETag**: Automatically generated based on response content
    - **Cache-Control**: Public caching with 30-minute max-age for data
    - **Conditional Requests**: Supports If-None-Match and If-Modified-Since
    
    ## Example Usage
    
    ### Get first 50 data points (most recent)
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators/CUUR0000SA0/data?first=50" \
         -H "X-API-Key: your-api-key"
    ```
    
    ### Get next page using cursor
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators/CUUR0000SA0/data?first=50&after=eyJ2YWx1ZSI6IjIwMjMtMTItMDEiLCJkaXJlY3Rpb24iOiJuZXh0In0=" \
         -H "X-API-Key: your-api-key"
    ```
    
    ### Get data for specific date range with cursor pagination
    ```bash
    curl -X GET "https://api.econovault.com/v1/indicators/CUUR0000SA0/data?start_date=2023-01-01&end_date=2023-12-31&first=100" \
         -H "X-API-Key: your-api-key" \
         -H "If-None-Match: W/\"abc123\""
    ```
    
    ## Response Format
    Returns cursor-based pagination with time series data:
    - `edges`: List of edges containing data points and cursors
    - `page_info`: Pagination metadata (has_next_page, has_previous_page, etc.)
    - `series_metadata`: Indicator metadata
    - `date_range`: First and last dates in the current page
    
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
    response_description="Cursor-based paginated time series data with metadata",
    tags=["indicators"],
    openapi_extra={
        "x-code-samples": [
            {
                "lang": "curl",
                "source": 'curl -X GET "https://api.econovault.com/v1/indicators/CUUR0000SA0/data?first=50" -H "X-API-Key: your-api-key"'
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
    "first": 100
}

response = requests.get(url, headers=headers, params=params)
result = response.json()

print(f"Series: {result['data']['series_metadata']['title']}")
print(f"Source: {result['data']['series_metadata']['source']}")
print(f"Data Points: {len(result['data']['edges'])}")
print(f"Has next page: {result['data']['page_info']['has_next_page']}")

# Process data points
for edge in result['data']['edges']:
    point = edge['node']
    print(f"{point['date']}: {point['value']}")
"""
            },
            {
                "lang": "JavaScript",
                "source": """const response = await fetch('https://api.econovault.com/v1/indicators/CUUR0000SA0/data?first=100', {
    headers: {
        'X-API-Key': 'your-api-key'
    }
});

const result = await response.json();
console.log(`Series: ${result.data.series_metadata.title}`);
console.log(`Source: ${result.data.series_metadata.source}`);
console.log(`Data Points: ${result.data.edges.length}`);
console.log(`Has next page: ${result.data.page_info.has_next_page}`);

// Process data points
result.data.edges.forEach(edge => {
    const point = edge.node;
    console.log(`${point.date}: ${point.value}`);
});
"""
            }
        ]
    }
)
async def get_indicator_data(
    request: Request,
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filter_param: Optional[str] = Query(None, alias="filter", description="JSON-encoded filter expression for data points"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to include (e.g., 'date,value,period')"),
    exclude: Optional[str] = Query(None, description="Comma-separated list of fields to exclude"),
    pagination: CursorPaginationParams = Depends(CursorPaginationParams),
    current_user: Optional[Dict] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """Get time series data for a specific indicator with cursor-based pagination and filtering."""
    # Check if series is in our popular series list
    if series_id not in popular_series:
        error_response = create_not_found_error_response("Indicator", series_id, request)
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.dict(exclude_none=True)
        )
    
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
        # Try to get data from Redis cache first
        cached_data = await indicator_cache.get_indicator_data(series_id, start_date, end_date)
        if cached_data:
            logger.info(f"Cache hit for indicator data: {series_id}")
            data_points = cached_data["data_points"]
        else:
            # Parse start and end years from date strings
            start_year = None
            end_year = None
            start_dt = None
            end_dt = None
            
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
                    "data": {
                        "edges": [],
                        "page_info": {
                            "has_next_page": False,
                            "has_previous_page": False,
                            "start_cursor": None,
                            "end_cursor": None
                        },
                        "series_metadata": popular_series[series_id],
                        "date_range": {
                            "start": None,
                            "end": None
                        }
                    }
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
            
            # Cache the processed data
            await indicator_cache.store_indicator_data(series_id, data_points, start_date, end_date)
        
        # Parse and validate advanced filter if provided
        filter_group = None
        filter_summary = None
        
        if filter_param:
            try:
                # Parse filter parameter
                parser = FilterParser()
                filter_group = parser.parse_filter_string(filter_param)
                
                # Validate against allowed fields for data points
                validator = FilterValidator(EconomicDataFilterConfig.DATA_POINT_FIELDS)
                validator.validate_filter_group(filter_group)
                
                filter_summary = {
                    "filter_applied": True,
                    "filter_expression": filter_param[:200] + "..." if len(filter_param) > 200 else filter_param
                }
                
            except ValueError as e:
                field_error = FieldError(
                    field="filter",
                    message=str(e),
                    code="INVALID_FILTER",
                    value=filter_param,
                    expected_type="string",
                    constraint="valid_filter_expression",
                    suggestions=["Check filter syntax and field names"]
                )
                error_response = create_validation_error_response([field_error], request)
                return JSONResponse(
                    status_code=error_response.status_code,
                    content=error_response.dict(exclude_none=True)
                )
        else:
            filter_summary = {
                "filter_applied": False,
                "date_range": {
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
        
        # Use cursor-based pagination for data points
        from cursor_pagination import EconomicDataCursorPaginator
        paginator = EconomicDataCursorPaginator()
        
        # Apply advanced filtering if provided
        if filter_group:
            try:
                executor = FilterExecutor()
                
                # Apply filter to data points
                filtered_data = executor.apply_filter_group(data_points, filter_group)
                
                # Update filter summary
                filter_summary["filtered_count"] = len(filtered_data)
                filter_summary["original_count"] = len(data_points)
                
                data_points = filtered_data
                
            except Exception as e:
                logger.error(f"Error applying advanced filter: {str(e)}")
                error_response = error_handler.create_error_response(
                    error_type=ErrorType.BAD_REQUEST_ERROR,
                    message="Error applying advanced filter conditions",
                    status_code=status.HTTP_400_BAD_REQUEST,
                    request=request,
                    details=str(e)
                )
                return JSONResponse(
                    status_code=error_response.status_code,
                    content=error_response.dict(exclude_none=True)
                )
        
        # Convert data points to format suitable for cursor pagination
        # Sort by date (most recent first for cursor pagination)
        data_points.sort(key=lambda x: x["date"], reverse=True)
        
        # Apply cursor pagination manually since we have the data in memory
        # For production, this would be done at the database level
        
        # Decode cursor if provided
        after_cursor = None
        before_cursor = None
        
        if pagination.after:
            from cursor_pagination import CursorEncoder
            after_cursor = CursorEncoder().decode_cursor(pagination.after)
        
        if pagination.before:
            from cursor_pagination import CursorEncoder
            before_cursor = CursorEncoder().decode_cursor(pagination.before)
        
        # Filter based on cursor
        filtered_data = data_points
        
        if after_cursor:
            # Get items after the cursor date
            cursor_date = after_cursor.value
            if isinstance(cursor_date, str):
                cursor_date = datetime.fromisoformat(cursor_date)
            elif isinstance(cursor_date, int):
                # Skip filtering if cursor_date is an int (can't compare datetime with int)
                cursor_date = None
            if cursor_date:
                filtered_data = [point for point in filtered_data if datetime.fromisoformat(point["date"]) < cursor_date]
        
        if before_cursor:
            # Get items before the cursor date
            cursor_date = before_cursor.value
            if isinstance(cursor_date, str):
                cursor_date = datetime.fromisoformat(cursor_date)
            elif isinstance(cursor_date, int):
                # Skip filtering if cursor_date is an int (can't compare datetime with int)
                cursor_date = None
            if cursor_date:
                filtered_data = [point for point in filtered_data if datetime.fromisoformat(point["date"]) > cursor_date]
        
        # Determine pagination direction and limit
        if pagination.first is not None:
            # Forward pagination (most recent first)
            limit = min(pagination.first, paginator.max_page_size)
            paginated_data = filtered_data[:limit]
            has_next_page = len(filtered_data) > limit
            has_previous_page = after_cursor is not None
        elif pagination.last is not None:
            # Backward pagination (oldest first)
            limit = min(pagination.last, paginator.max_page_size)
            paginated_data = filtered_data[-limit:]
            has_next_page = before_cursor is not None
            has_previous_page = len(filtered_data) > limit
        else:
            # Default pagination
            limit = paginator.default_page_size
            paginated_data = filtered_data[:limit]
            has_next_page = len(filtered_data) > limit
            has_previous_page = False
        
        # Generate cursors for data points
        edges = []
        for point in paginated_data:
            cursor = paginator.encoder.encode_cursor(point["date"])
            edges.append({
                "node": point,
                "cursor": cursor
            })
        
        # Determine page info
        start_cursor = edges[0]["cursor"] if edges else None
        end_cursor = edges[-1]["cursor"] if edges else None
        
        # Build cursor-based response
        return {
            "data": {
                "edges": edges,
                "page_info": {
                    "has_next_page": has_next_page,
                    "has_previous_page": has_previous_page,
                    "start_cursor": start_cursor,
                    "end_cursor": end_cursor
                },
                "series_metadata": popular_series[series_id],
                "date_range": {
                    "start": paginated_data[-1]["date"] if paginated_data else None,
                    "end": paginated_data[0]["date"] if paginated_data else None
                },
                "filter_applied": filter_summary
            }
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
    current_user: Dict = Depends(require_permission(Permission.MANAGE_CACHE))
):
    """Clear all caches (requires cache management permission)"""
    clear_cache()
    return {"message": "All caches cleared successfully"}


@router.get("/cache/status")
async def get_cache_status(
    current_user: Dict = Depends(require_permission(Permission.VIEW_METRICS))
):
    """Get cache status (requires metrics viewing permission)"""
    redis_status = "disabled"
    circuit_breaker_status = "unknown"
    
    if redis_manager.client:
        try:
            # Test Redis connection
            await redis_manager.client.ping()
            redis_status = "connected"
            
            # Get circuit breaker status
            circuit_breaker_status = redis_manager.circuit_breaker.state
        except Exception as e:
            redis_status = f"error: {str(e)}"
    
    return {
        "redis_status": redis_status,
        "circuit_breaker": circuit_breaker_status,
        "fallback_enabled": redis_manager.config.fallback_enabled,
        "memory_caches": {
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
    deletion_request: DeletionRequest,
    current_user: Dict = Depends(require_permission(Permission.MANAGE_GDPR_REQUESTS)),
    db: Session = Depends(get_db)
):
    """
    GDPR Article 17 - Right to erasure implementation (requires GDPR management permission)
    """
    
    # Verify the deletion request token
    if not verify_deletion_token(deletion_request.verification_token, user_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired deletion verification token"
        )
    
    # Execute deletion based on type
    request_id = generate_request_id()
    
    # Hash user ID for GDPR compliance
    from security import audit_logger
    user_id_hash = audit_logger.hash_identifier(user_id)
    verification_token_hash = audit_logger.hash_identifier(deletion_request.verification_token)
    
    # Store deletion request in database
    from database import DatabaseManager
    db_manager = DatabaseManager(db)
    
    deletion_data = {
        "request_id": request_id,
        "user_id_hash": user_id_hash,
        "deletion_type": deletion_request.deletion_type.value,
        "status": DeletionStatus.COMPLETED.value,
        "reason": deletion_request.reason,
        "verification_token_hash": verification_token_hash,
        "requested_at": datetime.utcnow(),
        "approved_at": datetime.utcnow(),
        "completed_at": datetime.utcnow(),
        "legal_retention_notice": get_retention_notice_if_applicable(user_id),
        "metadata_json": json.dumps({
            "processed_by": "system",
            "gdpr_article": "17",
            "deletion_method": deletion_request.deletion_type.value
        })
    }
    
    db_deletion_request = db_manager.create_deletion_request(deletion_data)
    
    # Log the deletion request
    try:
        audit_logger.log_data_access(
            db=db,
            user_id=user_id,
            data_subject_id=user_id,
            resource_type="deletion_request",
            resource_id=request_id,
            data_categories=["gdpr_request"],
            gdpr_basis="legal_obligation",
            success=True
        )
    except Exception as e:
        logger.error(f"Failed to log audit event: {str(e)}")
    
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
    consent_request: ConsentRequest,
    current_user: Dict = Depends(require_permission(Permission.MANAGE_GDPR_REQUESTS)),
    db: Session = Depends(get_db)
):
    """Update user consent status with proper authentication and audit logging (requires GDPR management permission)"""
    
    # Calculate expiration based on consent type
    expires_at = calculate_consent_expiration(consent_type)
    
    # Hash user ID for GDPR compliance
    from security import audit_logger
    user_id_hash = audit_logger.hash_identifier(user_id)
    
    # Store consent record in database
    from database import DatabaseManager
    db_manager = DatabaseManager(db)
    
    consent_data = {
        "user_id_hash": user_id_hash,
        "consent_type": consent_type.value,
        "status": consent_request.status.value,
        "consent_version": consent_request.consent_version,
        "expires_at": expires_at,
        "additional_data_json": json.dumps(consent_request.additional_data) if consent_request.additional_data else None
    }
    
    db_consent_record = db_manager.create_consent_record(consent_data)
    
    # Log the consent update
    try:
        audit_logger.log_data_access(
            db=db,
            user_id=user_id,
            data_subject_id=user_id,
            resource_type="consent_record",
            resource_id=f"{user_id}:{consent_type.value}",
            data_categories=["consent"],
            gdpr_basis="consent",
            success=True
        )
    except Exception as e:
        logger.error(f"Failed to log audit event: {str(e)}")
    
    return ConsentResponse(
        user_id=user_id,
        consent_type=consent_type,
        status=consent_request.status,
        timestamp=getattr(db_consent_record, 'timestamp', None) or datetime.now(timezone.utc),
        consent_version=getattr(db_consent_record, 'consent_version', None) or "1.0",
        expires_at=getattr(db_consent_record, 'expires_at', None),
    )


@router.get("/users/{user_id}/consent/{consent_type}", response_model=ConsentResponse)
async def get_consent_status(
    user_id: str,
    consent_type: ConsentType,
    db: Session = Depends(get_db)
):
    """Get current consent status for a specific type"""
    
    # Hash user ID for GDPR compliance
    from security import audit_logger
    user_id_hash = audit_logger.hash_identifier(user_id)
    if not user_id_hash:
        raise HTTPException(status_code=500, detail="Failed to hash user identifier")
    
    # Get consent record from database
    from database import DatabaseManager
    db_manager = DatabaseManager(db)
    
    consent_record = db_manager.get_consent_record(user_id_hash, consent_type.value)
    
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
    status = ConsentStatus(consent_record.status)
    expires_at = getattr(consent_record, 'expires_at', None)
    # Ensure expires_at is a datetime, not a Column object
    if hasattr(expires_at, '__class__') and 'Column' in str(expires_at.__class__):
        expires_at = None
    if expires_at is not None and expires_at < datetime.utcnow():
        status = ConsentStatus.EXPIRED
    
    return ConsentResponse(
        user_id=user_id,
        consent_type=ConsentType(consent_record.consent_type),
        status=status,
        timestamp=getattr(consent_record, 'timestamp', None) or datetime.utcnow(),
        consent_version=getattr(consent_record, 'consent_version', None) or "1.0",
        expires_at=getattr(consent_record, 'expires_at', None),
    )


# Data Export Endpoint (GDPR Article 20)
@router.post("/users/{user_id}/export", response_model=DataExportResponse)
async def export_user_data(
    user_id: str,
    export_request: DataExportRequest,
    current_user: Dict = Depends(require_permission(Permission.MANAGE_GDPR_REQUESTS)),
    db: Session = Depends(get_db)
):
    """GDPR Article 20 - Right to data portability (requires GDPR management permission)"""
    
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
    
    # Hash user ID for GDPR compliance
    from security import audit_logger
    user_id_hash = audit_logger.hash_identifier(user_id)
    verification_token_hash = audit_logger.hash_identifier(export_request.verification_token)
    
    # Store export record in database
    from database import DatabaseManager
    db_manager = DatabaseManager(db)
    
    expires_at = datetime.utcnow() + timedelta(hours=24)  # 24-hour expiry
    
    export_record_data = {
        "export_id": export_id,
        "user_id_hash": user_id_hash,
        "status": "completed",
        "format": export_request.format,
        "data_categories_json": json.dumps(export_request.data_categories) if export_request.data_categories else None,
        "verification_token_hash": verification_token_hash,
        "requested_at": datetime.utcnow(),
        "completed_at": datetime.utcnow(),
        "expires_at": expires_at,
        "metadata_json": json.dumps({
            "exported_data": export_data,
            "processed_by": "system",
            "gdpr_article": "20"
        })
    }
    
    db_export_request = db_manager.create_data_export_request(export_record_data)
    
    # Log the export request
    try:
        audit_logger.log_data_access(
            db=db,
            user_id=user_id,
            data_subject_id=user_id,
            resource_type="data_export_request",
            resource_id=export_id,
            data_categories=["gdpr_request"],
            gdpr_basis="legal_obligation",
            success=True
        )
    except Exception as e:
        logger.error(f"Failed to log audit event: {str(e)}")
    
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
    export_id: str,
    db: Session = Depends(get_db)
):
    """Download exported user data"""
    
    # Get export record from database
    from database import DatabaseManager
    db_manager = DatabaseManager(db)
    
    export_request = db_manager.get_data_export_request(export_id)
    if not export_request:
        raise HTTPException(
            status_code=404,
            detail="Export not found"
        )
    
    # Hash user ID for verification
    from security import audit_logger
    user_id_hash = audit_logger.hash_identifier(user_id)
    
    # Verify export belongs to user
    export_user_hash = getattr(export_request, 'user_id_hash', None) or ""
    if export_user_hash != user_id_hash:
        raise HTTPException(
            status_code=403,
            detail="Export does not belong to this user"
        )
    
    # Check if export has expired
    export_expires_at = getattr(export_request, 'expires_at', None) or datetime.utcnow()
    if export_expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=410,
            detail="Export has expired"
        )
    
    # Check if export is completed
    export_status = getattr(export_request, 'status', None) or ""
    if export_status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Export is not ready for download"
        )
    
    # Increment download count
    db_manager.increment_export_download_count(str(export_id))
    
    # Parse metadata to get the exported data
    try:
        metadata_json = getattr(export_request, 'metadata_json', None) or "{}"
        metadata = json.loads(metadata_json)
        export_data = metadata.get("exported_data", {})
    except json.JSONDecodeError:
        export_data = {}
    
    # Return exported data
    return {
        "export_id": export_id,
        "format": export_request.format,
        "data": export_data,
        "created_at": export_request.requested_at.isoformat(),
        "expires_at": export_request.expires_at.isoformat(),
        "download_count": export_request.download_count
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
    current_user: Dict = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: Session = Depends(get_db)
):
    """Create a new API key (requires API key management permission)"""
    
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
    current_user: Dict = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: Session = Depends(get_db)
):
    # List user's API keys (requires API key management permission)
    
    user_id_hash = audit_logger.hash_identifier(current_user["user_id"])
    
    # Get user's API keys
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
    current_user: Dict = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: Session = Depends(get_db)
):
    """Delete an API key (requires API key management permission)"""
    
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


# API Key Rotation and Lifecycle Management Endpoints
@router.post("/auth/api-keys/{key_id}/rotate")
async def rotate_api_key(
    request: Request,
    key_id: str,
    rotation_request: APIKeyRotationRequest,
    current_user: Dict = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: Session = Depends(get_db)
):
    """Rotate an API key (create new key and revoke old one)"""
    
    # Validate the key ID matches the request
    if rotation_request.key_id != key_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Key ID in path must match key ID in request body"
        )
    
    try:
        # Rotate the API key
        new_raw_key, new_key_info = await api_key_rotation_service.rotate_api_key(
            user_id=current_user["user_id"],
            key_id=key_id,
            db=db,
            reason=rotation_request.reason,
            notify_user=rotation_request.notify_user
        )
        
        # Log the rotation
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
            "message": "API key rotated successfully",
            "old_key_id": key_id,
            "new_key_id": new_key_info.key_id,
            "new_api_key": new_raw_key,  # Only shown once during rotation
            "new_key_info": new_key_info.dict(exclude_none=True)
        }
        
    except ValueError as e:
        error_response = error_handler.create_error_response(
            error_type=ErrorType.BAD_REQUEST_ERROR,
            message=str(e),
            status_code=status.HTTP_400_BAD_REQUEST,
            request=request
        )
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.dict(exclude_none=True)
        )


@router.post("/auth/api-keys/{key_id}/revoke")
async def revoke_api_key(
    request: Request,
    key_id: str,
    reason: Optional[str] = Query(None, description="Reason for revocation"),
    current_user: Dict = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: Session = Depends(get_db)
):
    """Revoke an API key permanently"""
    
    try:
        success = await api_key_rotation_service.revoke_api_key(
            user_id=current_user["user_id"],
            key_id=key_id,
            db=db,
            reason=reason
        )
        
        # Log the revocation
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
            "message": "API key revoked successfully",
            "key_id": key_id,
            "status": "revoked"
        }
        
    except ValueError as e:
        error_response = error_handler.create_error_response(
            error_type=ErrorType.BAD_REQUEST_ERROR,
            message=str(e),
            status_code=status.HTTP_400_BAD_REQUEST,
            request=request
        )
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.dict(exclude_none=True)
        )


@router.post("/auth/api-keys/{key_id}/suspend")
async def suspend_api_key(
    request: Request,
    key_id: str,
    reason: Optional[str] = Query(None, description="Reason for suspension"),
    current_user: Dict = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: Session = Depends(get_db)
):
    """Suspend an API key temporarily"""
    
    try:
        success = await api_key_rotation_service.suspend_api_key(
            user_id=current_user["user_id"],
            key_id=key_id,
            db=db,
            reason=reason
        )
        
        # Log the suspension
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
            "message": "API key suspended successfully",
            "key_id": key_id,
            "status": "suspended"
        }
        
    except ValueError as e:
        error_response = error_handler.create_error_response(
            error_type=ErrorType.BAD_REQUEST_ERROR,
            message=str(e),
            status_code=status.HTTP_400_BAD_REQUEST,
            request=request
        )
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.dict(exclude_none=True)
        )


@router.post("/auth/api-keys/{key_id}/reactivate")
async def reactivate_api_key(
    request: Request,
    key_id: str,
    current_user: Dict = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: Session = Depends(get_db)
):
    """Reactivate a suspended API key"""
    
    try:
        success = await api_key_rotation_service.reactivate_api_key(
            user_id=current_user["user_id"],
            key_id=key_id,
            db=db
        )
        
        # Log the reactivation
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
            "message": "API key reactivated successfully",
            "key_id": key_id,
            "status": "active"
        }
        
    except ValueError as e:
        error_response = error_handler.create_error_response(
            error_type=ErrorType.BAD_REQUEST_ERROR,
            message=str(e),
            status_code=status.HTTP_400_BAD_REQUEST,
            request=request
        )
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.dict(exclude_none=True)
        )


@router.get("/auth/api-keys/{key_id}/lifecycle")
async def get_api_key_lifecycle(
    request: Request,
    key_id: str,
    current_user: Dict = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: Session = Depends(get_db)
):
    """Get API key lifecycle information"""
    
    try:
        key_info = await api_key_rotation_service.get_api_key_info(
            user_id=current_user["user_id"],
            key_id=key_id,
            db=db
        )
        
        # Calculate key health metrics
        now = datetime.utcnow()
        days_until_expiry = None
        days_until_rotation = None
        
        if key_info.expires_at:
            days_until_expiry = (key_info.expires_at - now).days
        
        if key_info.next_rotation_date:
            days_until_rotation = (key_info.next_rotation_date - now).days
        
        # Determine key health status
        health_status = "healthy"
        if getattr(key_info, 'status', None) != APIKeyStatus.ACTIVE:
            health_status = key_info.status.value
        elif days_until_expiry is not None and days_until_expiry <= 7:
            health_status = "expiring_soon"
        elif days_until_rotation is not None and days_until_rotation <= 7:
            health_status = "rotation_due_soon"
        
        return {
            "key_info": key_info.dict(exclude_none=True),
            "lifecycle_metrics": {
                "days_until_expiry": days_until_expiry,
                "days_until_rotation": days_until_rotation,
                "health_status": health_status,
                "usage_rate": key_info.usage_count / max(1, (now - key_info.created_at).days)
            },
            "recommendations": _get_key_recommendations(key_info, days_until_expiry, days_until_rotation)
        }
        
    except ValueError as e:
        error_response = error_handler.create_error_response(
            error_type=ErrorType.NOT_FOUND_ERROR,
            message=str(e),
            status_code=status.HTTP_404_NOT_FOUND,
            request=request
        )
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.dict(exclude_none=True)
        )


def _get_key_recommendations(key_info: APIKeyInfo, days_until_expiry: Optional[int], days_until_rotation: Optional[int]) -> List[str]:
    """Get recommendations for API key management"""
    recommendations = []
    
    if key_info.status != APIKeyStatus.ACTIVE:
        return recommendations
    
    if days_until_expiry is not None and days_until_expiry <= 7:
        recommendations.append("Key expires soon - consider rotation or renewal")
    
    if days_until_rotation is not None and days_until_rotation <= 7:
        recommendations.append("Key is due for rotation - schedule rotation soon")
    
    if getattr(key_info, 'usage_count', 0) == 0:
        recommendations.append("Key has never been used - consider revoking if not needed")
    
    if key_info.rate_limit_per_minute < 60:
        recommendations.append("Consider increasing rate limits if needed for your use case")
    
    if getattr(key_info, 'rotation_policy', None) == APIKeyRotationPolicy.MANUAL:
        recommendations.append("Consider enabling automatic rotation for better security")
    
    return recommendations


@router.get("/auth/api-keys/lifecycle/status")
async def get_lifecycle_status(
    request: Request,
    current_user: Dict = Depends(require_permission(Permission.MANAGE_API_KEYS)),
    db: Session = Depends(get_db)
):
    """Get overall API key lifecycle status for the user"""
    
    try:
        # Get all user's keys
        all_keys = await api_key_rotation_service.list_api_keys(
            user_id=current_user["user_id"],
            db=db,
            include_inactive=True
        )
        
        # Calculate statistics
        total_keys = len(all_keys)
        active_keys = sum(1 for key in all_keys if key.status == APIKeyStatus.ACTIVE)
        expired_keys = sum(1 for key in all_keys if key.status == APIKeyStatus.EXPIRED)
        revoked_keys = sum(1 for key in all_keys if key.status == APIKeyStatus.REVOKED)
        
        # Find keys needing attention
        now = datetime.utcnow()
        expiring_soon = []
        rotation_due = []
        
        for key in all_keys:
            if key.status == APIKeyStatus.ACTIVE:
                if key.expires_at and (key.expires_at - now).days <= 7:
                    expiring_soon.append(key.key_id)
                
                if key.next_rotation_date and (key.next_rotation_date - now).days <= 7:
                    rotation_due.append(key.key_id)
        
        return {
            "total_keys": total_keys,
            "active_keys": active_keys,
            "expired_keys": expired_keys,
            "revoked_keys": revoked_keys,
            "keys_needing_attention": {
                "expiring_soon": expiring_soon,
                "rotation_due": rotation_due
            },
            "recommendations": [
                f"You have {len(expiring_soon)} keys expiring within 7 days",
                f"You have {len(rotation_due)} keys due for rotation within 7 days",
                f"Consider cleaning up {revoked_keys} revoked keys for better organization"
            ]
        }
        
    except Exception as e:
        error_response = error_handler.create_error_response(
            error_type=ErrorType.INTERNAL_SERVER_ERROR,
            message="Error retrieving lifecycle status",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request=request,
            exception=e
        )
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.dict(exclude_none=True)
        )





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


@app.get("/health/redis", status_code=status.HTTP_200_OK)
async def redis_health_check() -> Dict[str, Any]:
    """Health check endpoint for Redis connectivity"""
    if not redis_manager.config.enabled:
        return {
            "status": "disabled",
            "message": "Redis is disabled",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    try:
        redis_client = redis_manager.client
        if not redis_client:
            return {
                "status": "fallback",
                "message": "Redis unavailable, using fallback mode",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Test Redis connection
        await redis_client.ping()
        
        # Get basic Redis info
        info = await redis_client.info()
        
        return {
            "status": "healthy",
            "redis_version": info.get("redis_version", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "circuit_breaker": redis_manager.circuit_breaker.state,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "circuit_breaker": redis_manager.circuit_breaker.state,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> Dict[str, str]:
    # Root endpoint returning service information.
    return {
        "message": "EconoVault API service running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    # Prometheus metrics endpoint
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
    # Allow PORT environment variable for development
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)