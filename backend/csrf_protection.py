"""
CSRF protection middleware for EconoVault API.
Provides CSRF protection for state-changing operations in RESTful APIs.
"""

from __future__ import annotations
import secrets
import hashlib
import hmac
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import json

logger = logging.getLogger(__name__)


class CSRFConfig:
    """CSRF protection configuration"""
    
    # Token settings
    TOKEN_LENGTH = 32
    TOKEN_EXPIRY_MINUTES = 60
    
    # Protected HTTP methods
    PROTECTED_METHODS = ["POST", "PUT", "PATCH", "DELETE"]
    
    # Header names
    CSRF_TOKEN_HEADER = "X-CSRF-Token"
    CSRF_TOKEN_COOKIE = "csrf_token"
    
    # Allowed origins (for CORS + CSRF protection)
    ALLOWED_ORIGINS = [
        "https://econovault.com",
        "https://api.econovault.com",
        "https://app.econovault.com",
        "http://localhost:3000",  # Development
        "http://localhost:8000",  # Development
    ]
    
    # Skip CSRF for API key authentication
    SKIP_FOR_API_KEY = True
    
    # Enable double-submit cookie pattern
    ENABLE_DOUBLE_SUBMIT = True
    
    # Enable encrypted token pattern
    ENABLE_ENCRYPTED_TOKENS = True
    
    # Secret key for HMAC (should be from environment in production)
    SECRET_KEY = secrets.token_urlsafe(64)


class CSRFTokenGenerator:
    """Generate and validate CSRF tokens"""
    
    def __init__(self, config: Optional[CSRFConfig] = None):
        self.config = config or CSRFConfig()
        self.logger = logging.getLogger(__name__)
    
    def generate_token(self, user_id: str, session_id: str) -> str:
        """Generate a CSRF token for a user session"""
        # Create token components
        timestamp = str(int(time.time()))
        nonce = secrets.token_urlsafe(16)
        
        # Create token data
        token_data = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": timestamp,
            "nonce": nonce
        }
        
        # Create signature
        signature = self._create_signature(token_data)
        
        # Combine components
        token_parts = [
            timestamp,
            nonce,
            signature
        ]
        
        return ":".join(token_parts)
    
    def validate_token(self, token: str, user_id: str, session_id: str) -> bool:
        """Validate a CSRF token"""
        try:
            # Parse token
            parts = token.split(":")
            if len(parts) != 3:
                self.logger.warning(f"Invalid token format: {len(parts)} parts")
                return False
            
            timestamp_str, nonce, provided_signature = parts
            
            # Check timestamp
            try:
                timestamp = int(timestamp_str)
                current_time = int(time.time())
                
                # Check token expiry
                if current_time - timestamp > self.config.TOKEN_EXPIRY_MINUTES * 60:
                    self.logger.warning(f"Token expired: {current_time - timestamp} seconds old")
                    return False
                
            except ValueError:
                self.logger.warning(f"Invalid timestamp in token: {timestamp_str}")
                return False
            
            # Recreate token data
            token_data = {
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": timestamp_str,
                "nonce": nonce
            }
            
            # Verify signature
            expected_signature = self._create_signature(token_data)
            
            if not hmac.compare_digest(provided_signature, expected_signature):
                self.logger.warning("Token signature mismatch")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Token validation error: {str(e)}")
            return False
    
    def _create_signature(self, token_data: Dict[str, Any]) -> str:
        """Create HMAC signature for token data"""
        # Create message from token data
        message = json.dumps(token_data, sort_keys=True)
        
        # Create HMAC signature
        signature = hmac.new(
            self.config.SECRET_KEY.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature[:32]  # Use first 32 characters of signature


class CSRFProtectionMiddleware:
    """CSRF protection middleware for FastAPI"""
    
    def __init__(self, app, config: Optional[CSRFConfig] = None):
        self.app = app
        self.config = config or CSRFConfig()
        self.token_generator = CSRFTokenGenerator(self.config)
        self.logger = logging.getLogger(__name__)
    
    async def __call__(self, scope, receive, send):
        """Middleware entry point"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        try:
            # Only protect state-changing methods
            if request.method not in self.config.PROTECTED_METHODS:
                await self.app(scope, receive, send)
                return
            
            # Skip if using API key authentication
            if self.config.SKIP_FOR_API_KEY and self._is_api_key_auth(request):
                await self.app(scope, receive, send)
                return
            
            # Get user information from request
            user_info = self._get_user_info(request)
            if not user_info:
                # No authenticated user, skip CSRF protection
                await self.app(scope, receive, send)
                return
            
            # Validate origin/referer headers
            if not self._validate_origin(request):
                self.logger.warning(f"Invalid origin for CSRF request: {request.headers.get('origin')}")
                response = JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "CSRF validation failed",
                        "message": "Invalid request origin",
                        "request_id": self._generate_request_id()
                    }
                )
                await response(scope, receive, send)
                return
            
            # Get CSRF token from request
            csrf_token = self._extract_csrf_token(request)
            if not csrf_token:
                self.logger.warning("No CSRF token provided")
                response = JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "CSRF token required",
                        "message": f"CSRF token required in '{self.config.CSRF_TOKEN_HEADER}' header",
                        "request_id": self._generate_request_id()
                    }
                )
                await response(scope, receive, send)
                return
            
            # Validate the token
            if not self.token_generator.validate_token(
                csrf_token,
                user_info["user_id"],
                user_info.get("session_id", "default")
            ):
                self.logger.warning(f"Invalid CSRF token for user {user_info['user_id']}")
                response = JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "Invalid CSRF token",
                        "message": "The provided CSRF token is invalid or expired",
                        "request_id": self._generate_request_id()
                    }
                )
                await response(scope, receive, send)
                return
            
            # CSRF validation passed, continue with request
            await self.app(scope, receive, send)
            
        except HTTPException as e:
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
            self.logger.error(f"CSRF middleware error: {str(e)}")
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "CSRF validation error",
                    "message": "An error occurred during CSRF validation",
                    "request_id": self._generate_request_id()
                }
            )
            await response(scope, receive, send)
    
    def _is_api_key_auth(self, request: Request) -> bool:
        """Check if request is using API key authentication"""
        authorization = request.headers.get("authorization", "")
        return authorization.startswith("Bearer pk_")
    
    def _get_user_info(self, request: Request) -> Optional[Dict[str, Any]]:
        """Extract user information from request"""
        # This would typically integrate with your authentication system
        # For now, we'll extract from JWT token or session
        
        authorization = request.headers.get("authorization", "")
        if authorization.startswith("Bearer "):
            # Extract user info from JWT token
            # This is a simplified version - in production, you'd verify the token
            token = authorization[7:]  # Remove "Bearer " prefix
            try:
                # Decode JWT token (simplified - use proper JWT verification)
                import base64
                parts = token.split('.')
                if len(parts) == 3:
                    payload = json.loads(base64.urlsafe_b64decode(parts[1] + '=='))
                    return {
                        "user_id": payload.get("sub", "").replace("user:", ""),
                        "session_id": payload.get("session_id", "default")
                    }
            except Exception:
                pass
        
        return None
    
    def _validate_origin(self, request: Request) -> bool:
        """Validate request origin for CSRF protection"""
        origin = request.headers.get("origin")
        referer = request.headers.get("referer")
        
        # If no origin or referer, it's likely a direct API call (acceptable)
        if not origin and not referer:
            return True
        
        # Check origin against allowed list
        if origin:
            return any(origin.startswith(allowed) for allowed in self.config.ALLOWED_ORIGINS)
        
        # Check referer against allowed list
        if referer:
            return any(referer.startswith(allowed) for allowed in self.config.ALLOWED_ORIGINS)
        
        return False
    
    def _extract_csrf_token(self, request: Request) -> Optional[str]:
        """Extract CSRF token from request"""
        # Check header first
        token = request.headers.get(self.config.CSRF_TOKEN_HEADER)
        if token:
            return token
        
        # Check cookies if double-submit pattern is enabled
        if self.config.ENABLE_DOUBLE_SUBMIT:
            cookies = request.headers.get("cookie", "")
            for cookie in cookies.split(";"):
                cookie = cookie.strip()
                if cookie.startswith(f"{self.config.CSRF_TOKEN_COOKIE}="):
                    return cookie.split("=", 1)[1]
        
        return None
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID for error tracking"""
        import uuid
        return str(uuid.uuid4())


# Utility functions for CSRF token management
def generate_csrf_token_for_user(user_id: str, session_id: str, config: Optional[CSRFConfig] = None) -> str:
    """Generate CSRF token for a specific user session"""
    generator = CSRFTokenGenerator(config)
    return generator.generate_token(user_id, session_id)


def validate_csrf_token(token: str, user_id: str, session_id: str, config: Optional[CSRFConfig] = None) -> bool:
    """Validate CSRF token for a specific user session"""
    generator = CSRFTokenGenerator(config)
    return generator.validate_token(token, user_id, session_id)


def should_protect_endpoint(method: str, path: str, protected_methods: Optional[List[str]] = None) -> bool:
    """Determine if an endpoint should be CSRF protected"""
    if protected_methods is None:
        protected_methods = CSRFConfig.PROTECTED_METHODS
    
    return method.upper() in protected_methods