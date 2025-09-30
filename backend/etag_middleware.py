"""
ETag generation and validation for HTTP caching in EconoVault API.
Provides efficient HTTP caching using ETags and conditional requests.
"""

from __future__ import annotations
import hashlib
import json
import time
import logging
from typing import Optional, Dict, Any, Union, Callable
from datetime import datetime
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import asyncio

logger = logging.getLogger(__name__)


class ETagConfig:
    """ETag configuration settings"""
    
    # ETag generation settings
    ENABLE_WEAK_ETAGS = True  # Use weak ETags for better cache efficiency
    ETAG_ALGORITHM = "sha256"  # Hash algorithm for ETag generation
    
    # Cache control settings
    DEFAULT_MAX_AGE = 3600  # 1 hour
    PUBLIC_CACHE_MAX_AGE = 86400  # 24 hours for public data
    PRIVATE_CACHE_MAX_AGE = 3600  # 1 hour for private/user-specific data
    
    # Conditional request settings
    ENABLE_IF_NONE_MATCH = True
    ENABLE_IF_MODIFIED_SINCE = True
    
    # Resource-specific cache settings (in seconds)
    RESOURCE_CACHE_SETTINGS = {
        "indicators": 3600,  # 1 hour for indicator metadata
        "indicator_data": 1800,  # 30 minutes for time series data
        "user_data": 300,  # 5 minutes for user-specific data
        "gdpr_data": 60,  # 1 minute for GDPR-sensitive data
    }


class ETagGenerator:
    """Generate ETags for HTTP responses"""
    
    def __init__(self, config: Optional[ETagConfig] = None):
        self.config = config or ETagConfig()
        self.logger = logging.getLogger(__name__)
    
    def generate_etag(self, content: Union[str, bytes, Dict[str, Any]], 
                     weak: bool = False, resource_type: str = "general") -> str:
        """Generate ETag from content"""
        try:
            # Convert content to bytes for hashing
            if isinstance(content, dict):
                content_str = json.dumps(content, sort_keys=True, default=str)
                content_bytes = content_str.encode('utf-8')
            elif isinstance(content, str):
                content_bytes = content.encode('utf-8')
            elif isinstance(content, bytes):
                content_bytes = content
            else:
                content_bytes = str(content).encode('utf-8')
            
            # Generate hash
            if self.config.ETAG_ALGORITHM == "sha256":
                content_hash = hashlib.sha256(content_bytes).hexdigest()
            elif self.config.ETAG_ALGORITHM == "md5":
                content_hash = hashlib.md5(content_bytes).hexdigest()
            else:
                content_hash = hashlib.sha1(content_bytes).hexdigest()
            
            # Format ETag
            if weak or self.config.ENABLE_WEAK_ETAGS:
                etag = f'W/"{content_hash[:16]}"'  # Weak ETag with first 16 chars
            else:
                etag = f'"{content_hash[:32]}"'  # Strong ETag with first 32 chars
            
            self.logger.debug(f"Generated ETag: {etag} for {resource_type}")
            return etag
            
        except Exception as e:
            self.logger.error(f"Error generating ETag: {str(e)}")
            # Return a fallback ETag
            timestamp = str(int(time.time()))
            return f'W/"{timestamp}"'
    
    def generate_etag_from_response(self, response: Response) -> str:
        """Generate ETag from response object"""
        try:
            # Get response content
            if hasattr(response, 'body') and response.body:
                content = response.body
            else:
                # Fallback to empty content
                content = b""
            
            return self.generate_etag(content)
            
        except Exception as e:
            self.logger.error(f"Error generating ETag from response: {str(e)}")
            return self.generate_etag("")


class CacheControlManager:
    """Manage Cache-Control headers for different resource types"""
    
    def __init__(self, config: Optional[ETagConfig] = None):
        self.config = config or ETagConfig()
        self.logger = logging.getLogger(__name__)
    
    def get_cache_control_header(self, resource_type: str, is_private: bool = False) -> str:
        """Get appropriate Cache-Control header for resource type"""
        
        # Get max age for resource type
        max_age = self.config.RESOURCE_CACHE_SETTINGS.get(resource_type, self.config.DEFAULT_MAX_AGE)
        
        if is_private:
            return f"private, max-age={self.config.PRIVATE_CACHE_MAX_AGE}, must-revalidate"
        else:
            return f"public, max-age={max_age}, must-revalidate"
    
    def add_cache_headers(self, response: Response, resource_type: str, 
                         etag: str, last_modified: Optional[datetime] = None,
                         is_private: bool = False) -> None:
        """Add cache-related headers to response"""
        
        # Add ETag
        response.headers["ETag"] = etag
        
        # Add Cache-Control
        response.headers["Cache-Control"] = self.get_cache_control_header(resource_type, is_private)
        
        # Add Last-Modified if provided
        if last_modified:
            response.headers["Last-Modified"] = last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT")
        
        # Add other cache headers
        response.headers["Vary"] = "Accept-Encoding"
        
        self.logger.debug(f"Added cache headers for {resource_type}: ETag={etag}")


class ConditionalRequestHandler:
    """Handle conditional requests (If-None-Match, If-Modified-Since)"""
    
    def __init__(self, config: Optional[ETagConfig] = None):
        self.config = config or ETagConfig()
        self.logger = logging.getLogger(__name__)
    
    def check_if_none_match(self, request: Request, current_etag: str) -> bool:
        """Check If-None-Match header"""
        if not self.config.ENABLE_IF_NONE_MATCH:
            return False
        
        if_none_match = request.headers.get("if-none-match")
        if not if_none_match:
            return False
        
        # Handle multiple ETags (comma-separated)
        client_etags = [tag.strip() for tag in if_none_match.split(",")]
        
        # Check if current ETag matches any client ETag
        for client_etag in client_etags:
            # Handle weak ETags
            if client_etag.startswith("W/"):
                client_etag = client_etag[2:]  # Remove W/
            
            # Remove quotes
            client_etag = client_etag.strip('"')
            current_etag_clean = current_etag.strip('"')
            
            if client_etag == "*" or client_etag == current_etag_clean:
                self.logger.debug(f"If-None-Match matched: {client_etag}")
                return True
        
        return False
    
    def check_if_modified_since(self, request: Request, last_modified: datetime) -> bool:
        """Check If-Modified-Since header"""
        if not self.config.ENABLE_IF_MODIFIED_SINCE:
            return False
        
        if_modified_since = request.headers.get("if-modified-since")
        if not if_modified_since:
            return False
        
        try:
            # Parse client timestamp
            client_timestamp = datetime.strptime(if_modified_since, "%a, %d %b %Y %H:%M:%S GMT")
            
            # Compare timestamps
            if last_modified <= client_timestamp:
                self.logger.debug(f"Resource not modified since {client_timestamp}")
                return False  # Not modified
            else:
                return True  # Modified
                
        except ValueError:
            self.logger.warning(f"Invalid If-Modified-Since format: {if_modified_since}")
            return True  # Assume modified if parsing fails
    
    def handle_conditional_request(self, request: Request, response: Response, 
                                 current_etag: str, last_modified: Optional[datetime] = None) -> Optional[Response]:
        """Handle conditional request and return 304 Not Modified if applicable"""
        
        # Check If-None-Match
        if self.check_if_none_match(request, current_etag):
            self.logger.info("Returning 304 Not Modified (If-None-Match)")
            return Response(status_code=304)
        
        # Check If-Modified-Since
        if last_modified and not self.check_if_modified_since(request, last_modified):
            self.logger.info("Returning 304 Not Modified (If-Modified-Since)")
            return Response(status_code=304)
        
        return None  # Continue with normal response


class ETagMiddleware:
    """FastAPI middleware for ETag generation and validation"""
    
    def __init__(self, app, config: Optional[ETagConfig] = None):
        self.app = app
        self.config = config or ETagConfig()
        self.etag_generator = ETagGenerator(self.config)
        self.cache_manager = CacheControlManager(self.config)
        self.conditional_handler = ConditionalRequestHandler(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Cache for storing ETags (optional, for performance)
        self._etag_cache: Dict[str, tuple[str, float]] = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def __call__(self, scope, receive, send):
        """Middleware entry point"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Only process GET requests for ETag caching
        if request.method != "GET":
            await self.app(scope, receive, send)
            return
        
        try:
            # Check if we have cached ETag for this request
            cache_key = self._get_cache_key(request)
            cached_etag = self._get_cached_etag(cache_key)
            
            if cached_etag:
                # Check conditional request
                conditional_response = self.conditional_handler.handle_conditional_request(
                    request, Response(), cached_etag
                )
                if conditional_response:
                    await conditional_response(scope, receive, send)
                    return
            
            # Create response wrapper to capture content
            response_wrapper = ETagResponseWrapper(send)
            
            # Process request
            await self.app(scope, receive, response_wrapper)
            
            # Generate ETag from response content
            if response_wrapper.body:
                etag = self.etag_generator.generate_etag(response_wrapper.body)
                
                # Cache the ETag
                self._cache_etag(cache_key, etag)
                
                # Add ETag and cache headers to response
                response_wrapper.headers["ETag"] = etag
                
                # Determine resource type from path
                resource_type = self._determine_resource_type(request.url.path)
                cache_control = self.cache_manager.get_cache_control_header(resource_type)
                response_wrapper.headers["Cache-Control"] = cache_control
                
                # Check conditional request again with actual ETag
                conditional_response = self.conditional_handler.handle_conditional_request(
                    request, Response(), etag
                )
                if conditional_response:
                    # Return 304 Not Modified
                    await conditional_response(scope, receive, send)
                    return
                
                # Send the response with ETag headers
                await response_wrapper.send_with_headers()
            else:
                # No content to generate ETag for
                await response_wrapper.send_original()
                
        except Exception as e:
            self.logger.error(f"ETag middleware error: {str(e)}")
            # Fallback to original response
            await self.app(scope, receive, send)
    
    def _get_cache_key(self, request: Request) -> str:
        """Generate cache key for request"""
        path = request.url.path
        query = request.url.query
        return f"{path}?{query}" if query else path
    
    def _get_cached_etag(self, cache_key: str) -> Optional[str]:
        """Get cached ETag for request"""
        if cache_key in self._etag_cache:
            etag, timestamp = self._etag_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return etag
            else:
                # Expired, remove from cache
                del self._etag_cache[cache_key]
        return None
    
    def _cache_etag(self, cache_key: str, etag: str) -> None:
        """Cache ETag for request"""
        self._etag_cache[cache_key] = (etag, time.time())
    
    def _determine_resource_type(self, path: str) -> str:
        """Determine resource type from request path"""
        if "/indicators/" in path and "/data" in path:
            return "indicator_data"
        elif "/indicators" in path:
            return "indicators"
        elif "/users/" in path:
            return "user_data"
        elif "/gdpr" in path or "/consent" in path or "/export" in path:
            return "gdpr_data"
        else:
            return "general"


class ETagResponseWrapper:
    """Wrapper to capture response content for ETag generation"""
    
    def __init__(self, original_send: Callable):
        self.original_send = original_send
        self.body = b""
        self.headers = {}
        self.status_code = 200
        self._response_started = False
    
    async def __call__(self, message: Dict[str, Any]) -> None:
        """Capture response messages"""
        if message["type"] == "http.response.start":
            self.status_code = message.get("status", 200)
            # Convert headers from bytes to strings for easier manipulation
            headers_dict = {}
            for name, value in message.get("headers", []):
                if isinstance(name, bytes):
                    name_str = name.decode()
                else:
                    name_str = name
                
                if isinstance(value, bytes):
                    value_str = value.decode()
                else:
                    value_str = value
                
                headers_dict[name_str] = value_str
            
            self.headers = headers_dict
            self._response_started = True
        elif message["type"] == "http.response.body":
            body_chunk = message.get("body", b"")
            if body_chunk:
                self.body += body_chunk
            
            # Don't send immediately, wait for ETag processing
            if not message.get("more_body", False):
                # End of response, but don't send yet
                pass
    
    async def send_with_headers(self) -> None:
        """Send response with additional ETag headers"""
        # Convert headers to list format
        header_list = []
        for name, value in self.headers.items():
            if isinstance(name, str):
                name_bytes = name.encode()
            else:
                name_bytes = name
            
            if isinstance(value, str):
                value_bytes = value.encode()
            else:
                value_bytes = value
            
            header_list.append((name_bytes, value_bytes))
        
        # Send start message
        await self.original_send({
            "type": "http.response.start",
            "status": self.status_code,
            "headers": header_list
        })
        
        # Send body
        await self.original_send({
            "type": "http.response.body",
            "body": self.body,
            "more_body": False
        })
    
    async def send_original(self) -> None:
        """Send original response without modifications"""
        # Convert headers back to list format
        header_list = []
        for name, value in self.headers.items():
            if isinstance(name, str):
                name_bytes = name.encode()
            else:
                name_bytes = name
            
            if isinstance(value, str):
                value_bytes = value.encode()
            else:
                value_bytes = value
            
            header_list.append((name_bytes, value_bytes))
        
        # Send start message
        await self.original_send({
            "type": "http.response.start",
            "status": self.status_code,
            "headers": header_list
        })
        
        # Send body
        await self.original_send({
            "type": "http.response.body",
            "body": self.body,
            "more_body": False
        })


# Utility functions for manual ETag handling
def generate_etag_for_data(data: Any, weak: bool = False) -> str:
    """Generate ETag for arbitrary data"""
    generator = ETagGenerator()
    return generator.generate_etag(data, weak=weak)


def add_etag_headers(response: Response, etag: str, resource_type: str = "general", 
                    last_modified: Optional[datetime] = None, is_private: bool = False) -> None:
    """Manually add ETag headers to a response"""
    manager = CacheControlManager()
    manager.add_cache_headers(response, resource_type, etag, last_modified, is_private)


def check_conditional_request(request: Request, current_etag: str, 
                             last_modified: Optional[datetime] = None) -> Optional[Response]:
    """Check if request is conditional and return 304 if applicable"""
    handler = ConditionalRequestHandler()
    return handler.handle_conditional_request(request, Response(), current_etag, last_modified)