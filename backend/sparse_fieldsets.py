"""
Sparse fieldsets support for EconoVault API response optimization.
Allows clients to specify exactly which fields they want returned, reducing bandwidth and improving performance.
"""

from __future__ import annotations
import re
from typing import Dict, Any, List, Optional, Set
from collections.abc import Callable
from pydantic import BaseModel, Field, validator
from fastapi import Request
import logging

logger = logging.getLogger(__name__)


class FieldSelectionConfig:
    """Configuration for field selection"""
    
    # Maximum number of fields that can be requested
    MAX_FIELDS = 50
    
    # Maximum nesting depth for field selection
    MAX_DEPTH = 5
    
    # Reserved fields that are always included
    RESERVED_FIELDS = {"id", "_links", "_meta"}
    
    # Field patterns that are not allowed
    FORBIDDEN_PATTERNS = [
        r"^_.*",  # Private fields starting with underscore
        r".*\.__.*",  # Double underscore fields (private)
        r".*\.password.*",  # Password fields
        r".*\.secret.*",  # Secret fields
        r".*\.token.*",  # Token fields
    ]


class FieldSelectionRequest(BaseModel):
    """Request for field selection"""
    fields: Optional[str] = Field(None, description="Comma-separated list of fields to include")
    exclude: Optional[str] = Field(None, description="Comma-separated list of fields to exclude")
    
    @validator('fields', 'exclude')
    def validate_field_selection(cls, v):
        """Validate field selection parameters"""
        if v is not None:
            if len(v) > 1000:  # Reasonable limit
                raise ValueError("Field selection string too long")
            
            # Check for suspicious patterns
            for pattern in FieldSelectionConfig.FORBIDDEN_PATTERNS:
                if re.search(pattern, v, re.IGNORECASE):
                    raise ValueError(f"Field selection contains forbidden pattern: {pattern}")
        
        return v


class FieldSelector:
    """Field selection and filtering for API responses"""
    
    def __init__(self, config: Optional[FieldSelectionConfig] = None):
        self.config = config or FieldSelectionConfig()
        self.logger = logging.getLogger(__name__)
    
    def parse_field_selection(self, fields_str: Optional[str], exclude_str: Optional[str] = None) -> Dict[str, Any]:
        """Parse field selection strings into structured format"""
        result = {
            "include": set(),
            "exclude": set(),
            "nested": {},
            "is_sparse": False
        }
        
        # Parse include fields
        if fields_str:
            include_fields = self._parse_field_string(fields_str)
            result["include"] = include_fields
            result["is_sparse"] = True
        
        # Parse exclude fields
        if exclude_str:
            exclude_fields = self._parse_field_string(exclude_str)
            result["exclude"] = exclude_fields
            result["is_sparse"] = True
        
        # Validate field count
        total_fields = len(result["include"]) + len(result["exclude"])
        if total_fields > self.config.MAX_FIELDS:
            raise ValueError(f"Too many fields selected. Maximum: {self.config.MAX_FIELDS}")
        
        # Process nested fields
        result["nested"] = self._extract_nested_fields(result["include"])
        
        return result
    
    def _parse_field_string(self, field_str: str) -> Set[str]:
        """Parse comma-separated field string into set"""
        if not field_str:
            return set()
        
        fields = set()
        for field in field_str.split(","):
            field = field.strip()
            if field:
                # Validate field format
                if not self._is_valid_field_name(field):
                    raise ValueError(f"Invalid field name: {field}")
                
                # Check depth
                if field.count(".") > self.config.MAX_DEPTH:
                    raise ValueError(f"Field nesting too deep: {field}")
                
                fields.add(field)
        
        return fields
    
    def _is_valid_field_name(self, field: str) -> bool:
        """Validate field name format"""
        # Basic validation - alphanumeric, underscore, and dots for nesting
        if not re.match(r'^[a-zA-Z0-9_.]+$', field):
            return False
        
        # Check for forbidden patterns
        for pattern in self.config.FORBIDDEN_PATTERNS:
            if re.search(pattern, field, re.IGNORECASE):
                return False
        
        return True
    
    def _extract_nested_fields(self, fields: Set[str]) -> Dict[str, Set[str]]:
        """Extract nested field structure"""
        nested = {}
        
        for field in fields:
            if "." in field:
                parent, child = field.split(".", 1)
                if parent not in nested:
                    nested[parent] = set()
                nested[parent].add(child)
        
        return nested
    
    def apply_field_selection(self, data: Any, field_selection: Dict[str, Any]) -> Any:
        """Apply field selection to data"""
        if not field_selection["is_sparse"]:
            return data
        
        if isinstance(data, list):
            return [self._filter_item(item, field_selection) for item in data]
        elif isinstance(data, dict):
            return self._filter_item(data, field_selection)
        else:
            return data
    
    def _filter_item(self, item: Dict[str, Any], field_selection: Dict[str, Any]) -> Dict[str, Any]:
        """Filter a single item based on field selection"""
        if not isinstance(item, dict):
            return item
        
        result = {}
        include_fields = field_selection["include"]
        exclude_fields = field_selection["exclude"]
        nested_fields = field_selection["nested"]
        
        # If include is specified, only include those fields (plus reserved)
        if include_fields:
            # Always include reserved fields
            fields_to_include = include_fields.union(self.config.RESERVED_FIELDS)
            
            for field in fields_to_include:
                if field in item:
                    # Handle nested fields
                    if field in nested_fields:
                        result[field] = self._filter_nested_field(item[field], nested_fields[field])
                    else:
                        result[field] = item[field]
        else:
            # Include all fields except excluded ones
            for key, value in item.items():
                if key not in exclude_fields:
                    # Handle nested fields
                    if key in nested_fields:
                        result[key] = self._filter_nested_field(value, nested_fields[key])
                    else:
                        result[key] = value
        
        return result
    
    def _filter_nested_field(self, data: Any, nested_fields: Set[str]) -> Any:
        """Filter nested field data"""
        if isinstance(data, list):
            return [self._filter_item(item, {"include": nested_fields, "exclude": set(), "nested": {}, "is_sparse": True}) 
                   if isinstance(item, dict) else item for item in data]
        elif isinstance(data, dict):
            return self._filter_item(data, {"include": nested_fields, "exclude": set(), "nested": {}, "is_sparse": True})
        else:
            return data
    
    def create_sparse_response(self, data: Any, fields: Optional[str] = None, exclude: Optional[str] = None) -> Any:
        """Create sparse response with field selection"""
        if not fields and not exclude:
            return data
        
        try:
            field_selection = self.parse_field_selection(fields, exclude)
            return self.apply_field_selection(data, field_selection)
        except Exception as e:
            self.logger.error(f"Error applying field selection: {str(e)}")
            # Return original data if field selection fails
            return data
    
    def get_field_info(self, data_sample: Dict[str, Any], max_depth: int = 2) -> Dict[str, Any]:
        """Get information about available fields in data structure"""
        field_info = {
            "top_level_fields": [],
            "nested_fields": {},
            "field_types": {}
        }
        
        self._extract_field_info(data_sample, field_info, "", max_depth, 0)
        
        return field_info
    
    def _extract_field_info(self, data: Any, field_info: Dict[str, Any], prefix: str, max_depth: int, current_depth: int):
        """Recursively extract field information"""
        if current_depth >= max_depth or not isinstance(data, dict):
            return
        
        for key, value in data.items():
            full_path = f"{prefix}.{key}" if prefix else key
            
            # Skip private fields
            if key.startswith("_"):
                continue
            
            # Add to appropriate category
            if not prefix:
                field_info["top_level_fields"].append(key)
            else:
                parent = prefix.split(".")[-1] if "." in prefix else prefix
                if parent not in field_info["nested_fields"]:
                    field_info["nested_fields"][parent] = []
                field_info["nested_fields"][parent].append(key)
            
            # Record field type
            field_info["field_types"][full_path] = type(value).__name__
            
            # Recurse for nested objects
            if isinstance(value, dict) and current_depth < max_depth - 1:
                self._extract_field_info(value, field_info, full_path, max_depth, current_depth + 1)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                self._extract_field_info(value[0], field_info, full_path, max_depth, current_depth + 1)


class SparseFieldsetMiddleware:
    """FastAPI middleware for sparse fieldset support"""
    
    def __init__(self, app, config: Optional[FieldSelectionConfig] = None):
        self.app = app
        self.config = config or FieldSelectionConfig()
        self.selector = FieldSelector(self.config)
        self.logger = logging.getLogger(__name__)
    
    async def __call__(self, scope, receive, send):
        """Middleware entry point"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Check if this is a GET request that might benefit from field selection
        if request.method != "GET":
            await self.app(scope, receive, send)
            return
        
        # Extract field selection parameters
        fields = request.query_params.get("fields")
        exclude = request.query_params.get("exclude")
        
        if not fields and not exclude:
            await self.app(scope, receive, send)
            return
        
        try:
            # Create response wrapper to capture and filter content
            response_wrapper = SparseFieldsetResponseWrapper(send, self.selector, fields, exclude)
            
            # Process request
            await self.app(scope, receive, response_wrapper)
            
            # Send filtered response
            await response_wrapper.send_filtered_response()
            
        except Exception as e:
            self.logger.error(f"Sparse fieldset middleware error: {str(e)}")
            # Fallback to original response
            await self.app(scope, receive, send)


class SparseFieldsetResponseWrapper:
    """Wrapper to capture and filter response content"""
    
    def __init__(self, original_send: Callable, selector: FieldSelector, fields: Optional[str], exclude: Optional[str]):
        self.original_send = original_send
        self.selector = selector
        self.fields = fields
        self.exclude = exclude
        self.body_chunks = []
        self.headers = {}
        self.status_code = 200
        self._response_started = False
    
    async def __call__(self, message: Dict[str, Any]) -> None:
        """Capture response messages"""
        if message["type"] == "http.response.start":
            self.status_code = message.get("status", 200)
            self.headers = dict(message.get("headers", []))
            self._response_started = True
        elif message["type"] == "http.response.body":
            body_chunk = message.get("body", b"")
            if body_chunk:
                self.body_chunks.append(body_chunk)
            
            # Don't send immediately, wait for field filtering
            if not message.get("more_body", False):
                # End of response, but don't send yet
                pass
    
    async def send_filtered_response(self) -> None:
        """Send response with field filtering applied"""
        try:
            # Combine body chunks
            full_body = b"".join(self.body_chunks)
            
            # Parse JSON if possible
            if full_body:
                import json
                try:
                    data = json.loads(full_body.decode('utf-8'))
                    
                    # Apply field selection
                    filtered_data = self.selector.create_sparse_response(data, self.fields, self.exclude)
                    
                    # Convert back to JSON
                    filtered_body = json.dumps(filtered_data).encode('utf-8')
                    
                    # Update content-length header
                    self._update_content_length(len(filtered_body))
                    
                    # Send filtered response
                    await self._send_response(filtered_body)
                    return
                    
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Not JSON, send original response
                    pass
            
            # Send original response if not JSON or filtering failed
            await self._send_response(full_body)
            
        except Exception as e:
            # Log error and send original response
            self.selector.logger.error(f"Error in sparse fieldset filtering: {str(e)}")
            await self._send_response(b"".join(self.body_chunks))
    
    def _update_content_length(self, new_length: int) -> None:
        """Update content-length header"""
        # Convert headers to list and update content-length
        header_list = []
        content_length_updated = False
        
        for name, value in self.headers.items():
            if name.lower() == b"content-length":
                header_list.append((name, str(new_length).encode()))
                content_length_updated = True
            else:
                if isinstance(value, str):
                    header_list.append((name, value.encode()))
                elif isinstance(value, bytes):
                    header_list.append((name, value))
        
        # Add content-length if not present
        if not content_length_updated:
            header_list.append((b"content-length", str(new_length).encode()))
        
        self.headers = dict(header_list)
    
    async def _send_response(self, body: bytes) -> None:
        """Send response with current headers and body"""
        # Send start message
        await self.original_send({
            "type": "http.response.start",
            "status": self.status_code,
            "headers": [(name, value) for name, value in self.headers.items()]
        })
        
        # Send body
        await self.original_send({
            "type": "http.response.body",
            "body": body,
            "more_body": False
        })


# Utility functions for field selection
def create_field_selection_response(fields: Optional[str], exclude: Optional[str], 
                                  data: Any, available_fields: List[str]) -> Any:
    """Create response with field selection applied"""
    selector = FieldSelector()
    
    try:
        # Validate requested fields against available fields
        if fields:
            requested_fields = set(field.strip() for field in fields.split(",") if field.strip())
            invalid_fields = requested_fields - set(available_fields)
            if invalid_fields:
                logger.warning(f"Requested invalid fields: {invalid_fields}")
                # Remove invalid fields
                valid_fields = ",".join(requested_fields - invalid_fields)
                return selector.create_sparse_response(data, valid_fields, exclude)
        
        return selector.create_sparse_response(data, fields, exclude)
        
    except Exception as e:
        logger.error(f"Error applying field selection: {str(e)}")
        return data


def get_available_fields(data_sample: Dict[str, Any], max_depth: int = 3) -> List[str]:
    """Get list of available fields from data sample"""
    selector = FieldSelector()
    field_info = selector.get_field_info(data_sample, max_depth)
    
    available_fields = field_info["top_level_fields"].copy()
    
    # Add nested fields
    for parent, children in field_info["nested_fields"].items():
        for child in children:
            available_fields.append(f"{parent}.{child}")
    
    return available_fields