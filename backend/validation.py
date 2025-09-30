"""
Input validation and sanitization middleware for EconoVault API.
Provides comprehensive validation, sanitization, and security checks for all incoming requests.
"""

from __future__ import annotations
import re
import html
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, date
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, Field
import bleach
from urllib.parse import unquote_plus

logger = logging.getLogger(__name__)


class ValidationConfig:
    """Configuration for validation middleware"""
    
    # Maximum allowed sizes
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_STRING_LENGTH = 1000
    MAX_ARRAY_LENGTH = 1000
    MAX_OBJECT_DEPTH = 10
    
    # SQL Injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute|script|declare|truncate)\b)",
        r"(\b(or|and)\b.*=.*\b(or|and)\b)",
        r"(--|#|/\*|\*/)",
        r"(\bunion\b.*\bselect\b)",
        r"(\bselect\b.*\bfrom\b.*\bwhere\b)",
        r"(xp_cmdshell|sp_executesql|information_schema)"
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<form[^>]*>",
        r"<input[^>]*>",
        r"<svg[^>]*>",
        r"<img[^>]*on\w+",
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%2e%2e%5c",
        r"%252e%252e%252f",
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`]",
        r"\$\(.*?\)",
        r"`.*?`",
        r"\|\|",
        r"&&",
    ]
    
    # Allowed HTML tags and attributes for rich text fields
    ALLOWED_HTML_TAGS = [
        'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'blockquote', 'code', 'pre'
    ]
    
    ALLOWED_HTML_ATTRIBUTES = {
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'title'],
    }


class ValidationException(Exception):
    """Validation exception for raising errors"""
    
    def __init__(self, field: str, message: str, value: Any = None, 
                 expected_type: Optional[str] = None, constraint: Optional[str] = None):
        self.field = field
        self.message = message
        self.value = value
        self.expected_type = expected_type
        self.constraint = constraint
        super().__init__(self.message)


class ValidationError(BaseModel):
    """Validation error details"""
    field: str
    message: str
    value: Any = None
    expected_type: Optional[str] = None
    constraint: Optional[str] = None


class ValidationResult(BaseModel):
    """Result of validation operation"""
    is_valid: bool
    errors: List[ValidationError]
    sanitized_data: Optional[Dict[str, Any]] = None
    warnings: List[str] = []


class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
    
    def validate_request_size(self, content_length: Optional[int]) -> None:
        """Validate request size"""
        if content_length and content_length > self.config.MAX_REQUEST_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request too large. Maximum size: {self.config.MAX_REQUEST_SIZE} bytes"
            )
    
    def validate_string(self, value: str, field_name: str, max_length: Optional[int] = None) -> str:
        """Validate and sanitize string input"""
        if not isinstance(value, str):
            raise ValidationException(
                field=field_name,
                message=f"Expected string, got {type(value).__name__}",
                value=value,
                expected_type="string"
            )
        
        max_len = max_length or self.config.MAX_STRING_LENGTH
        if len(value) > max_len:
            raise ValidationException(
                field=field_name,
                message=f"String too long. Maximum length: {max_len}",
                value=value,
                constraint=f"max_length={max_len}"
            )
        
        # Check for SQL injection patterns
        for pattern in self.config.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                self.logger.warning(f"Potential SQL injection detected in field '{field_name}': {value[:100]}")
                raise ValidationException(
                    field=field_name,
                    message="Input contains potentially dangerous SQL patterns",
                    value=value[:100]
                )
        
        # Check for XSS patterns
        for pattern in self.config.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                self.logger.warning(f"Potential XSS detected in field '{field_name}': {value[:100]}")
                raise ValidationException(
                    field=field_name,
                    message="Input contains potentially dangerous HTML/JavaScript patterns",
                    value=value[:100]
                )
        
        # Check for path traversal patterns
        for pattern in self.config.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                self.logger.warning(f"Potential path traversal detected in field '{field_name}': {value[:100]}")
                raise ValidationException(
                    field=field_name,
                    message="Input contains potentially dangerous path patterns",
                    value=value[:100]
                )
        
        # Check for command injection patterns
        for pattern in self.config.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                self.logger.warning(f"Potential command injection detected in field '{field_name}': {value[:100]}")
                raise ValidationException(
                    field=field_name,
                    message="Input contains potentially dangerous command patterns",
                    value=value[:100]
                )
        
        # Sanitize HTML if present
        if "<" in value and ">" in value:
            sanitized = bleach.clean(
                value,
                tags=self.config.ALLOWED_HTML_TAGS,
                attributes=self.config.ALLOWED_HTML_ATTRIBUTES,
                strip=True
            )
            if sanitized != value:
                self.logger.info(f"HTML sanitized in field '{field_name}'")
                return sanitized
        
        return value.strip()
    
    def validate_numeric(self, value: Union[int, float], field_name: str, 
                        min_value: Optional[Union[int, float]] = None,
                        max_value: Optional[Union[int, float]] = None) -> Union[int, float]:
        """Validate numeric input"""
        if not isinstance(value, (int, float)):
            raise ValidationException(
                field=field_name,
                message=f"Expected numeric, got {type(value).__name__}",
                value=value,
                expected_type="numeric"
            )
        
        if min_value is not None and value < min_value:
            raise ValidationException(
                field=field_name,
                message=f"Value too small. Minimum: {min_value}",
                value=value,
                constraint=f"min_value={min_value}"
            )
        
        if max_value is not None and value > max_value:
            raise ValidationException(
                field=field_name,
                message=f"Value too large. Maximum: {max_value}",
                value=value,
                constraint=f"max_value={max_value}"
            )
        
        return value
    
    def validate_date(self, value: Union[str, date, datetime], field_name: str) -> date:
        """Validate date input"""
        if isinstance(value, date):
            return value
        elif isinstance(value, datetime):
            return value.date()
        elif isinstance(value, str):
            try:
                # Try multiple date formats
                date_formats = ['%Y-%m-%d', '%Y-%m', '%Y', '%m/%d/%Y', '%m/%Y']
                for fmt in date_formats:
                    try:
                        return datetime.strptime(value, fmt).date()
                    except ValueError:
                        continue
                raise ValueError(f"Cannot parse date: {value}")
            except ValueError as e:
                raise ValidationException(
                    field=field_name,
                    message=f"Invalid date format: {value}",
                    value=value,
                    expected_type="date (YYYY-MM-DD)"
                )
        else:
            raise ValidationException(
                field=field_name,
                message=f"Expected date, got {type(value).__name__}",
                value=value,
                expected_type="date"
            )
    
    def validate_array(self, value: List[Any], field_name: str, 
                      max_length: Optional[int] = None,
                      item_validator: Optional[Callable] = None) -> List[Any]:
        """Validate array input"""
        if not isinstance(value, list):
            raise ValidationException(
                field=field_name,
                message=f"Expected array, got {type(value).__name__}",
                value=value,
                expected_type="array"
            )
        
        max_len = max_length or self.config.MAX_ARRAY_LENGTH
        if len(value) > max_len:
            raise ValidationException(
                field=field_name,
                message=f"Array too long. Maximum length: {max_len}",
                value=value,
                constraint=f"max_length={max_len}"
            )
        
        # Validate each item if validator provided
        if item_validator:
            validated_items = []
            for i, item in enumerate(value):
                try:
                    validated_item = item_validator(item, f"{field_name}[{i}]")
                    validated_items.append(validated_item)
                except ValidationException as e:
                    e.field = f"{field_name}[{i}]"
                    raise e
            return validated_items
        
        return value
    
    def validate_object(self, value: Dict[str, Any], field_name: str, 
                       max_depth: Optional[int] = None,
                       depth: int = 0) -> Dict[str, Any]:
        """Validate object input recursively"""
        if not isinstance(value, dict):
            raise ValidationException(
                field=field_name,
                message=f"Expected object, got {type(value).__name__}",
                value=value,
                expected_type="object"
            )
        
        max_dep = max_depth or self.config.MAX_OBJECT_DEPTH
        if depth > max_dep:
            raise ValidationException(
                field=field_name,
                message=f"Object nesting too deep. Maximum depth: {max_dep}",
                value=value,
                constraint=f"max_depth={max_dep}"
            )
        
        validated_obj = {}
        for key, val in value.items():
            # Validate key
            if not isinstance(key, str):
                raise ValidationException(
                    field=f"{field_name}.{key}",
                    message="Object keys must be strings",
                    value=key
                )
            
            # Sanitize key
            sanitized_key = self.validate_string(key, f"{field_name}.{key}")
            
            # Recursively validate value based on type
            if isinstance(val, dict):
                validated_obj[sanitized_key] = self.validate_object(val, f"{field_name}.{key}", max_dep, depth + 1)
            elif isinstance(val, list):
                validated_obj[sanitized_key] = self.validate_array(val, f"{field_name}.{key}")
            elif isinstance(val, str):
                validated_obj[sanitized_key] = self.validate_string(val, f"{field_name}.{key}")
            elif isinstance(val, (int, float)):
                validated_obj[sanitized_key] = self.validate_numeric(val, f"{field_name}.{key}")
            else:
                validated_obj[sanitized_key] = val
        
        return validated_obj
    
    def validate_email(self, value: str, field_name: str) -> str:
        """Validate email address"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, value):
            raise ValidationException(
                field=field_name,
                message="Invalid email format",
                value=value,
                expected_type="email"
            )
        return value.lower().strip()
    
    def validate_url(self, value: str, field_name: str) -> str:
        """Validate URL"""
        url_pattern = r'^https?://[\w\-._~:/?#\[\]@!$&\'()*+,;=]+$'
        if not re.match(url_pattern, value):
            raise ValidationException(
                field=field_name,
                message="Invalid URL format",
                value=value,
                expected_type="URL"
            )
        return value.strip()
    
    def validate_series_id(self, value: str, field_name: str) -> str:
        """Validate economic indicator series ID"""
        # BLS series IDs are typically uppercase alphanumeric
        series_id_pattern = r'^[A-Z0-9]{8,20}$'
        if not re.match(series_id_pattern, value):
            raise ValidationException(
                field=field_name,
                message="Invalid series ID format. Must be 8-20 uppercase alphanumeric characters",
                value=value,
                expected_type="series_id"
            )
        return value
    
    def validate_indicator_data(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate complete indicator data structure"""
        errors = []
        warnings = []
        sanitized_data = {}
        
        # Required fields
        required_fields = ["series_id", "title", "source", "indicator_type", "frequency", "geography_level", "units"]
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(ValidationError(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    value=None
                ))
        
        # Validate series_id
        if "series_id" in data:
            try:
                sanitized_data["series_id"] = self.validate_series_id(data["series_id"], "series_id")
            except ValidationException as e:
                errors.append(ValidationError(
                    field=e.field,
                    message=e.message,
                    value=e.value,
                    expected_type=e.expected_type,
                    constraint=e.constraint
                ))
        
        # Validate title
        if "title" in data:
            try:
                sanitized_data["title"] = self.validate_string(data["title"], "title", max_length=500)
            except ValidationException as e:
                errors.append(ValidationError(
                    field=e.field,
                    message=e.message,
                    value=e.value,
                    expected_type=e.expected_type,
                    constraint=e.constraint
                ))
        
        # Validate description
        if "description" in data and data["description"] is not None:
            try:
                sanitized_data["description"] = self.validate_string(data["description"], "description", max_length=2000)
            except ValidationException as e:
                errors.append(ValidationError(
                    field=e.field,
                    message=e.message,
                    value=e.value,
                    expected_type=e.expected_type,
                    constraint=e.constraint
                ))
        
        # Validate source
        if "source" in data:
            try:
                source = self.validate_string(data["source"], "source", max_length=20)
                if source.upper() not in ["BLS", "BEA", "FRED", "CENSUS", "TREASURY"]:
                    errors.append(ValidationError(
                        field="source",
                        message="Invalid source. Must be one of: BLS, BEA, FRED, CENSUS, TREASURY",
                        value=source
                    ))
                sanitized_data["source"] = source.upper()
            except ValidationException as e:
                errors.append(ValidationError(
                    field=e.field,
                    message=e.message,
                    value=e.value,
                    expected_type=e.expected_type,
                    constraint=e.constraint
                ))
        
        # Validate frequency
        if "frequency" in data:
            try:
                frequency = self.validate_string(data["frequency"], "frequency", max_length=10)
                valid_frequencies = ["DAILY", "WEEKLY", "BIWEEKLY", "MONTHLY", "QUARTERLY", "SEMIANNUAL", "ANNUAL"]
                if frequency.upper() not in valid_frequencies:
                    errors.append(ValidationError(
                        field="frequency",
                        message=f"Invalid frequency. Must be one of: {', '.join(valid_frequencies)}",
                        value=frequency
                    ))
                sanitized_data["frequency"] = frequency.upper()
            except ValidationException as e:
                errors.append(ValidationError(
                    field=e.field,
                    message=e.message,
                    value=e.value,
                    expected_type=e.expected_type,
                    constraint=e.constraint
                ))
        
        # Validate dates
        if "start_date" in data and data["start_date"] is not None:
            try:
                sanitized_data["start_date"] = self.validate_date(data["start_date"], "start_date")
            except ValidationException as e:
                errors.append(ValidationError(
                    field=e.field,
                    message=e.message,
                    value=e.value,
                    expected_type=e.expected_type,
                    constraint=e.constraint
                ))
        
        if "end_date" in data and data["end_date"] is not None:
            try:
                sanitized_data["end_date"] = self.validate_date(data["end_date"], "end_date")
            except ValidationException as e:
                errors.append(ValidationError(
                    field=e.field,
                    message=e.message,
                    value=e.value,
                    expected_type=e.expected_type,
                    constraint=e.constraint
                ))
        
        # Validate date range
        if "start_date" in sanitized_data and "end_date" in sanitized_data:
            if sanitized_data["start_date"] > sanitized_data["end_date"]:
                errors.append(ValidationError(
                    field="date_range",
                    message="Start date must be before or equal to end date",
                    value=f"{sanitized_data['start_date']} - {sanitized_data['end_date']}"
                ))
        
        # Validate geography_level
        if "geography_level" in data:
            try:
                geo_level = self.validate_string(data["geography_level"], "geography_level", max_length=20)
                valid_levels = ["NATIONAL", "STATE", "METRO", "COUNTY"]
                if geo_level.upper() not in valid_levels:
                    errors.append(ValidationError(
                        field="geography_level",
                        message=f"Invalid geography level. Must be one of: {', '.join(valid_levels)}",
                        value=geo_level
                    ))
                sanitized_data["geography_level"] = geo_level.upper()
            except ValidationException as e:
                errors.append(ValidationError(
                    field=e.field,
                    message=e.message,
                    value=e.value,
                    expected_type=e.expected_type,
                    constraint=e.constraint
                ))
        
        # Validate units
        if "units" in data:
            try:
                sanitized_data["units"] = self.validate_string(data["units"], "units", max_length=100)
            except ValidationException as e:
                errors.append(ValidationError(
                    field=e.field,
                    message=e.message,
                    value=e.value,
                    expected_type=e.expected_type,
                    constraint=e.constraint
                ))
        
        # Validate boolean fields
        for bool_field in ["registration_key_required"]:
            if bool_field in data and data[bool_field] is not None:
                if not isinstance(data[bool_field], bool):
                    errors.append(ValidationError(
                        field=bool_field,
                        message=f"Expected boolean, got {type(data[bool_field]).__name__}",
                        value=data[bool_field],
                        expected_type="boolean"
                    ))
                sanitized_data[bool_field] = data[bool_field]
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_data=sanitized_data if errors else None,
            warnings=warnings
        )


class InputValidationMiddleware:
    """FastAPI middleware for input validation and sanitization"""
    
    def __init__(self, app, config: Optional[ValidationConfig] = None):
        self.app = app
        self.config = config or ValidationConfig()
        self.validator = InputValidator(self.config)
        self.logger = logging.getLogger(__name__)
    
    async def __call__(self, scope, receive, send):
        """Middleware entry point"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Skip validation for OpenAPI endpoints
        path = scope.get("path", "")
        if path in ["/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"]:
            await self.app(scope, receive, send)
            return
        
        # Create request object
        request = Request(scope, receive)
        
        try:
            # Validate request size
            content_length = request.headers.get("content-length")
            if content_length:
                self.validator.validate_request_size(int(content_length))
            
            # Process request body for validation
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                if body:
                    try:
                        # Try to parse as JSON
                        data = json.loads(body.decode())
                        
                        # Validate the data
                        if request.url.path.startswith("/v1/indicators"):
                            result = self.validator.validate_indicator_data(data)
                            if not result.is_valid:
                                error_response = {
                                    "error": "Validation failed",
                                    "message": "Request validation failed",
                                    "validation_errors": [
                                        {
                                            "field": e.field,
                                            "message": e.message,
                                            "value": e.value,
                                            "expected_type": e.expected_type,
                                            "constraint": e.constraint
                                        }
                                        for e in result.errors
                                    ],
                                    "request_id": self._generate_request_id()
                                }
                                response = JSONResponse(
                                    status_code=status.HTTP_400_BAD_REQUEST,
                                    content=error_response
                                )
                                await response(scope, receive, send)
                                return
                            
                            # Use sanitized data if validation passed with warnings
                            if result.sanitized_data:
                                self.logger.info(f"Request sanitized with {len(result.warnings)} warnings")
                                # Create new request with sanitized data
                                sanitized_body = json.dumps(result.sanitized_data).encode()
                                
                                async def new_receive():
                                    return {"type": "http.request", "body": sanitized_body, "more_body": False}
                                
                                await self.app(scope, new_receive, send)
                                return
                        
                    except json.JSONDecodeError:
                        # Not JSON, let it pass through for other content types
                        pass
            
            # Continue with normal processing
            await self.app(scope, receive, send)
            
        except HTTPException as e:
            # Convert HTTPException to proper response
            response = JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.detail,
                    "message": str(e.detail),
                    "request_id": self._generate_request_id()
                }
            )
            await response(scope, receive, send)
            
        except Exception as e:
            self.logger.error(f"Validation middleware error: {str(e)}")
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal validation error",
                    "message": "An error occurred during request validation",
                    "request_id": self._generate_request_id()
                }
            )
            await response(scope, receive, send)
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID for error tracking"""
        import uuid
        return str(uuid.uuid4())


# Utility functions for specific validation scenarios
def validate_economic_series_id(series_id: str) -> str:
    """Validate economic series ID format"""
    if not series_id or len(series_id) < 3:
        raise ValueError("Series ID must be at least 3 characters long")
    
    # BLS series IDs are typically uppercase alphanumeric
    if not re.match(r'^[A-Z0-9]{3,20}$', series_id):
        raise ValueError("Series ID must be 3-20 uppercase alphanumeric characters")
    
    return series_id


def validate_date_range(start_date: date, end_date: date) -> None:
    """Validate date range"""
    if start_date > end_date:
        raise ValueError("Start date must be before or equal to end date")


def sanitize_html_content(content: str, allowed_tags: Optional[List[str]] = None) -> str:
    """Sanitize HTML content"""
    if allowed_tags is None:
        allowed_tags = ValidationConfig.ALLOWED_HTML_TAGS
    
    return bleach.clean(
        content,
        tags=allowed_tags,
        attributes=ValidationConfig.ALLOWED_HTML_ATTRIBUTES,
        strip=True
    )


def validate_api_key_format(api_key: str) -> str:
    """Validate API key format"""
    if not api_key.startswith("pk_"):
        raise ValueError("API key must start with 'pk_'")
    
    if len(api_key) < 20:
        raise ValueError("API key too short")
    
    return api_key