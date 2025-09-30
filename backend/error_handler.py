"""
Standardized error response handling for EconoVault API.
Provides consistent error responses with correlation IDs and field-level validation details.
"""

from __future__ import annotations
import uuid

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
import traceback
import json

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Standardized error types"""
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND_ERROR = "not_found_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    SERVICE_UNAVAILABLE_ERROR = "service_unavailable_error"
    INTERNAL_SERVER_ERROR = "internal_server_error"
    BAD_REQUEST_ERROR = "bad_request_error"
    CONFLICT_ERROR = "conflict_error"
    TIMEOUT_ERROR = "timeout_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    DATABASE_ERROR = "database_error"
    CACHE_ERROR = "cache_error"
    NETWORK_ERROR = "network_error"


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FieldError(BaseModel):
    """Field-level validation error details"""
    field: str = Field(..., description="Field name that caused the error")
    message: str = Field(..., description="Human-readable error message")
    code: Optional[str] = Field(None, description="Error code for programmatic handling")
    value: Optional[Any] = Field(None, description="The invalid value provided")
    expected_type: Optional[str] = Field(None, description="Expected data type")
    constraint: Optional[str] = Field(None, description="Constraint that was violated")
    suggestions: Optional[List[str]] = Field(None, description="Suggested corrections")
    
    class Config:
        schema_extra = {
            "example": {
                "field": "series_id",
                "message": "Series ID must be 8-20 uppercase alphanumeric characters",
                "code": "INVALID_FORMAT",
                "value": "invalid-series-id",
                "expected_type": "string",
                "constraint": "pattern=^[A-Z0-9]{8,20}$",
                "suggestions": ["Use format like 'CUUR0000SA0' or 'LNS14000000'"]
            }
        }


class ErrorContext(BaseModel):
    """Additional context for error debugging"""
    timestamp: str = Field(..., description="ISO 8601 timestamp when error occurred")
    path: str = Field(..., description="Request path that caused the error")
    method: str = Field(..., description="HTTP method")
    user_agent: Optional[str] = Field(None, description="User agent string")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    request_id: str = Field(..., description="Unique request identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier if authenticated")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00.123Z",
                "path": "/v1/indicators/INVALID123/data",
                "method": "GET",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "ip_address": "192.168.1.100",
                "request_id": "req_abc123def456",
                "session_id": "sess_xyz789abc012",
                "user_id": "user_123456789"
            }
        }


class StandardErrorResponse(BaseModel):
    """Standardized error response format"""
    error: str = Field(..., description="Error type/category")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(None, description="Additional error details")
    request_id: str = Field(..., description="Unique request identifier for tracking")
    timestamp: str = Field(..., description="ISO 8601 timestamp when error occurred")
    status_code: int = Field(..., description="HTTP status code")
    severity: ErrorSeverity = Field(ErrorSeverity.MEDIUM, description="Error severity level")
    type: ErrorType = Field(..., description="Specific error type")
    code: Optional[str] = Field(None, description="Application-specific error code")
    help: Optional[str] = Field(None, description="Help text or documentation link")
    fields: Optional[List[FieldError]] = Field(None, description="Field-level validation errors")
    context: Optional[ErrorContext] = Field(None, description="Additional error context")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Request validation failed",
                "details": "One or more fields contain invalid data",
                "request_id": "req_abc123def456",
                "timestamp": "2024-01-15T10:30:00.123Z",
                "status_code": 400,
                "severity": "medium",
                "type": "validation_error",
                "code": "VALIDATION_FAILED",
                "help": "See https://docs.econovault.com/api/errors#validation",
                "fields": [
                    {
                        "field": "series_id",
                        "message": "Series ID must be 8-20 uppercase alphanumeric characters",
                        "code": "INVALID_FORMAT",
                        "value": "invalid-series-id",
                        "expected_type": "string",
                        "constraint": "pattern=^[A-Z0-9]{8,20}$",
                        "suggestions": ["Use format like 'CUUR0000SA0' or 'LNS14000000'"]
                    }
                ],
                "context": {
                    "timestamp": "2024-01-15T10:30:00.123Z",
                    "path": "/v1/indicators/INVALID123/data",
                    "method": "GET",
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "ip_address": "192.168.1.100",
                    "request_id": "req_abc123def456",
                    "session_id": "sess_xyz789abc012",
                    "user_id": "user_123456789"
                },
                "metadata": {
                    "api_version": "1.0.0",
                    "server_time": "2024-01-15T10:30:00.123Z"
                }
            }
        }


class ErrorHandler:
    """Centralized error handling for the API"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_error_response(
        self,
        error_type: ErrorType,
        message: str,
        status_code: int,
        request: Optional[Request] = None,
        details: Optional[str] = None,
        fields: Optional[List[FieldError]] = None,
        exception: Optional[Exception] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        code: Optional[str] = None,
        help_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StandardErrorResponse:
        """Create a standardized error response"""
        
        # Generate request ID if not provided
        request_id = self._generate_request_id()
        
        # Create error context
        context = None
        if request:
            context = ErrorContext(
                timestamp=datetime.utcnow().isoformat(),
                path=str(request.url.path),
                method=request.method,
                user_agent=request.headers.get("user-agent"),
                ip_address=self._get_client_ip(request),
                request_id=request_id,
                session_id=request.headers.get("x-session-id"),
                user_id=self._get_user_id(request)
            )
        
        # Create error response
        error_response = StandardErrorResponse(
            error=error_type.value,
            message=message,
            details=details,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            status_code=status_code,
            severity=severity,
            type=error_type,
            code=code,
            help=help_text,
            fields=fields,
            context=context,
            metadata=metadata or {}
        )
        
        # Log the error
        self._log_error(error_response, exception)
        
        return error_response
    
    def handle_http_exception(
        self,
        exception: HTTPException,
        request: Optional[Request] = None
    ) -> StandardErrorResponse:
        """Convert FastAPI HTTPException to standardized error response"""
        
        # Map HTTP status codes to error types
        error_type_map = {
            400: ErrorType.BAD_REQUEST_ERROR,
            401: ErrorType.AUTHENTICATION_ERROR,
            403: ErrorType.AUTHORIZATION_ERROR,
            404: ErrorType.NOT_FOUND_ERROR,
            409: ErrorType.CONFLICT_ERROR,
            429: ErrorType.RATE_LIMIT_ERROR,
            500: ErrorType.INTERNAL_SERVER_ERROR,
            502: ErrorType.SERVICE_UNAVAILABLE_ERROR,
            503: ErrorType.SERVICE_UNAVAILABLE_ERROR,
            504: ErrorType.TIMEOUT_ERROR
        }
        
        error_type = error_type_map.get(exception.status_code, ErrorType.INTERNAL_SERVER_ERROR)
        
        return self.create_error_response(
            error_type=error_type,
            message=exception.detail,
            status_code=exception.status_code,
            request=request,
            severity=self._get_severity_for_status_code(exception.status_code)
        )
    
    def handle_validation_error(
        self,
        validation_errors: List[Dict[str, Any]],
        request: Optional[Request] = None
    ) -> StandardErrorResponse:
        """Handle Pydantic validation errors"""
        
        field_errors = []
        for error in validation_errors:
            field_error = FieldError(
                field=error.get("loc", ["unknown"])[-1] if error.get("loc") else "unknown",
                message=error.get("msg", "Validation error"),
                code=error.get("type", "validation_error"),
                value=error.get("input"),
                expected_type=self._get_expected_type(error.get("type", "")),
                constraint=self._get_constraint(error.get("type", "")),
                suggestions=None
            )
            field_errors.append(field_error)
        
        return self.create_error_response(
            error_type=ErrorType.VALIDATION_ERROR,
            message="Request validation failed",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            request=request,
            details="One or more fields contain invalid data",
            fields=field_errors,
            severity=ErrorSeverity.LOW,
            code="VALIDATION_FAILED",
            help_text="See https://docs.econovault.com/api/errors#validation"
        )
    
    def handle_internal_error(
        self,
        exception: Exception,
        request: Optional[Request] = None,
        include_stack_trace: bool = False
    ) -> StandardErrorResponse:
        """Handle internal server errors"""
        
        # Create error details
        details = "An internal server error occurred"
        if include_stack_trace:
            details = f"{details}: {str(exception)}"
        
        # Create metadata with stack trace if requested
        metadata = {
            "api_version": "1.0.0",
            "server_time": datetime.utcnow().isoformat()
        }
        
        if include_stack_trace:
            metadata["stack_trace"] = traceback.format_exc()
        
        return self.create_error_response(
            error_type=ErrorType.INTERNAL_SERVER_ERROR,
            message="Internal server error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request=request,
            details=details,
            exception=exception,
            severity=ErrorSeverity.CRITICAL,
            code="INTERNAL_ERROR",
            help_text="Please contact support with the request ID",
            metadata=metadata
        )
    
    def handle_external_service_error(
        self,
        service_name: str,
        exception: Exception,
        request: Optional[Request] = None
    ) -> StandardErrorResponse:
        """Handle external service errors (BLS API, etc.)"""
        
        return self.create_error_response(
            error_type=ErrorType.EXTERNAL_SERVICE_ERROR,
            message=f"External service '{service_name}' is temporarily unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            request=request,
            details=f"Error from {service_name}: {str(exception)}",
            exception=exception,
            severity=ErrorSeverity.HIGH,
            code="EXTERNAL_SERVICE_ERROR",
            help_text="Please try again later or contact support if the issue persists"
        )
    
    def handle_rate_limit_error(
        self,
        limit: int,
        window: str,
        request: Optional[Request] = None
    ) -> StandardErrorResponse:
        """Handle rate limit errors"""
        
        return self.create_error_response(
            error_type=ErrorType.RATE_LIMIT_ERROR,
            message="Rate limit exceeded",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            request=request,
            details=f"Rate limit of {limit} requests per {window} exceeded",
            severity=ErrorSeverity.MEDIUM,
            code="RATE_LIMIT_EXCEEDED",
            help_text="Please reduce your request rate or upgrade your plan"
        )
    
    def handle_database_error(
        self,
        exception: Exception,
        request: Optional[Request] = None
    ) -> StandardErrorResponse:
        """Handle database errors"""
        
        return self.create_error_response(
            error_type=ErrorType.DATABASE_ERROR,
            message="Database operation failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request=request,
            details="A database error occurred while processing your request",
            exception=exception,
            severity=ErrorSeverity.HIGH,
            code="DATABASE_ERROR",
            help_text="Please try again later or contact support if the issue persists"
        )
    
    def handle_cache_error(
        self,
        exception: Exception,
        request: Optional[Request] = None
    ) -> StandardErrorResponse:
        """Handle cache errors"""
        
        return self.create_error_response(
            error_type=ErrorType.CACHE_ERROR,
            message="Cache operation failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            request=request,
            details="A cache error occurred while processing your request",
            exception=exception,
            severity=ErrorSeverity.MEDIUM,
            code="CACHE_ERROR",
            help_text="The system is operating in fallback mode. Please try again later."
        )
    
    async def handle_bls_call(
        self,
        bls_function,
        *args,
        **kwargs
    ):
        """Handle BLS API calls with error handling"""
        
        try:
            # Try to execute the BLS function
            result = await bls_function(*args, **kwargs)
            return result
        except Exception as e:
            # Log the error
            self.logger.error(f"BLS API call failed: {str(e)}")
            
            # Re-raise the exception to be handled by the caller
            raise e
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        return f"req_{uuid.uuid4().hex[:16]}"
    
    def create_not_found_error_response(
        self,
        resource_type: str,
        resource_id: str,
        request: Optional[Request] = None,
        message: Optional[str] = None
    ) -> StandardErrorResponse:
        """Create a standardized not found error response"""
        
        default_message = f"{resource_type} with ID '{resource_id}' not found"
        return self.create_error_response(
            error_type=ErrorType.NOT_FOUND_ERROR,
            message=message or default_message,
            status_code=status.HTTP_404_NOT_FOUND,
            request=request,
            details=f"The requested {resource_type.lower()} could not be found",
            severity=ErrorSeverity.LOW,
            code="NOT_FOUND",
            help_text="Please verify the resource ID and try again"
        )
    
    def create_conflict_error_response(
        self,
        resource_type: str,
        resource_id: str,
        request: Optional[Request] = None,
        message: Optional[str] = None
    ) -> StandardErrorResponse:
        """Create a standardized conflict error response"""
        
        default_message = f"{resource_type} with ID '{resource_id}' already exists"
        return self.create_error_response(
            error_type=ErrorType.CONFLICT_ERROR,
            message=message or default_message,
            status_code=status.HTTP_409_CONFLICT,
            request=request,
            details=f"A {resource_type.lower()} with the provided ID already exists",
            severity=ErrorSeverity.LOW,
            code="CONFLICT",
            help_text="Please use a different ID or update the existing resource"
        )
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Extract client IP address from request"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if request.client and hasattr(request.client, 'host'):
            return request.client.host
        
        return None
    
    def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request if available"""
        # This would integrate with your authentication system
        # For now, return None
        return None
    
    def _get_severity_for_status_code(self, status_code: int) -> ErrorSeverity:
        """Determine error severity based on HTTP status code"""
        if status_code >= 500:
            return ErrorSeverity.CRITICAL
        elif status_code >= 400:
            return ErrorSeverity.HIGH
        elif status_code >= 300:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _get_expected_type(self, error_type: str) -> Optional[str]:
        """Extract expected type from Pydantic error type"""
        type_mapping = {
            "string_type": "string",
            "int_type": "integer",
            "float_type": "float",
            "bool_type": "boolean",
            "list_type": "list",
            "dict_type": "dict",
            "date_type": "date",
            "datetime_type": "datetime"
        }
        
        for key, expected_type in type_mapping.items():
            if key in error_type:
                return expected_type
        
        return None
    
    def _get_constraint(self, error_type: str) -> Optional[str]:
        """Extract constraint information from Pydantic error type"""
        constraint_mapping = {
            "string_too_short": "min_length",
            "string_too_long": "max_length",
            "string_pattern_mismatch": "pattern",
            "int_too_small": "min_value",
            "int_too_large": "max_value",
            "float_too_small": "min_value",
            "float_too_large": "max_value",
            "date_past": "date_range",
            "date_future": "date_range"
        }
        
        for key, constraint in constraint_mapping.items():
            if key in error_type:
                return constraint
        
        return None
    
    def _log_error(self, error_response: StandardErrorResponse, exception: Optional[Exception] = None):
        """Log error for monitoring and debugging"""
        
        # Create log entry
        log_entry = {
            "request_id": error_response.request_id,
            "error_type": error_response.type.value,
            "status_code": error_response.status_code,
            "severity": error_response.severity.value,
            "message": error_response.message,
            "timestamp": error_response.timestamp,
            "user_id": error_response.context.user_id if error_response.context else None
        }
        
        # Add exception details if available
        if exception:
            log_entry["exception_type"] = type(exception).__name__
            log_entry["exception_message"] = str(exception)
        
        # Log based on severity
        if error_response.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical error: {json.dumps(log_entry)}", exc_info=exception)
        elif error_response.severity == ErrorSeverity.HIGH:
            self.logger.error(f"High severity error: {json.dumps(log_entry)}", exc_info=exception)
        elif error_response.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Medium severity error: {json.dumps(log_entry)}")
        else:
            self.logger.info(f"Low severity error: {json.dumps(log_entry)}")


# Global error handler instance
error_handler = ErrorHandler()


# FastAPI exception handlers
def create_exception_handlers(app):
    """Create FastAPI exception handlers"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        error_response = error_handler.handle_http_exception(exc, request)
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.dict(exclude_none=True)
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        # Don't expose internal errors in production
        include_stack_trace = False  # Set to True for development/debugging
        
        error_response = error_handler.handle_internal_error(
            exc, request, include_stack_trace=include_stack_trace
        )
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.dict(exclude_none=True)
        )
    
    @app.exception_handler(422)
    async def validation_exception_handler(request: Request, exc: Exception):
        """Handle validation errors"""
        # Extract validation errors from the exception
        validation_errors = []
        
        exc_body = getattr(exc, 'body', None)
        if exc_body:
            try:
                error_data = json.loads(exc_body)
                if isinstance(error_data, list):
                    validation_errors = error_data
            except (json.JSONDecodeError, TypeError):
                pass
        
        error_response = error_handler.handle_validation_error(validation_errors, request)
        return JSONResponse(
            status_code=error_response.status_code,
            content=error_response.dict(exclude_none=True)
        )


# Utility functions for common error scenarios
def create_validation_error_response(
    field_errors: List[FieldError],
    request: Optional[Request] = None,
    message: str = "Request validation failed"
) -> StandardErrorResponse:
    """Create validation error response"""
    return error_handler.create_error_response(
        error_type=ErrorType.VALIDATION_ERROR,
        message=message,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        request=request,
        fields=field_errors,
        severity=ErrorSeverity.LOW,
        code="VALIDATION_FAILED"
    )


def create_not_found_error_response(
    resource: str,
    identifier: str,
    request: Optional[Request] = None
) -> StandardErrorResponse:
    """Create not found error response"""
    return error_handler.create_error_response(
        error_type=ErrorType.NOT_FOUND_ERROR,
        message=f"{resource} not found",
        status_code=status.HTTP_404_NOT_FOUND,
        request=request,
        details=f"{resource} with identifier '{identifier}' was not found",
        severity=ErrorSeverity.LOW,
        code="NOT_FOUND"
    )


def create_rate_limit_error_response(
    limit: int,
    window: str,
    request: Optional[Request] = None
) -> StandardErrorResponse:
    """Create rate limit error response"""
    return error_handler.handle_rate_limit_error(limit, window, request)


def create_external_service_error_response(
    service_name: str,
    exception: Exception,
    request: Optional[Request] = None
) -> StandardErrorResponse:
    """Create external service error response"""
    return error_handler.handle_external_service_error(service_name, exception, request)