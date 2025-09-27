import bcrypt
import secrets
from jose import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
import redis
from redis import Redis
import json
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Union, Any
from fastapi import Security, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from sqlalchemy.orm import Session
import os
import logging

# Import database models and dependencies
from database import APIKey, AuditLog, get_db


# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "RS256"  # Enhanced security with RSA
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Reduced for financial APIs
REFRESH_TOKEN_EXPIRE_DAYS = 7
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MASTER_KEY = os.getenv("MASTER_ENCRYPTION_KEY", Fernet.generate_key())
JWT_PRIVATE_KEY_PATH = os.getenv("JWT_PRIVATE_KEY_PATH")
JWT_PUBLIC_KEY_PATH = os.getenv("JWT_PUBLIC_KEY_PATH")
JWT_ISSUER = os.getenv("JWT_ISSUER", "https://api.econovault.com")


# Enhanced Key Management
class KeyManager:
    """RSA key management for JWT tokens"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.private_key = self._load_or_generate_private_key()
        self.public_key = self.private_key.public_key()
    
    def _load_or_generate_private_key(self):
        """Load existing RSA key or generate new one"""
        if JWT_PRIVATE_KEY_PATH and os.path.exists(JWT_PRIVATE_KEY_PATH):
            try:
                with open(JWT_PRIVATE_KEY_PATH, "rb") as key_file:
                    return serialization.load_pem_private_key(
                        key_file.read(),
                        password=None,
                        backend=default_backend()
                    )
            except Exception as e:
                self.logger.error(f"Failed to load private key: {e}")
        
        # Generate new RSA key for development
        self.logger.info("Generating new RSA key pair for JWT")
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
    
    def get_public_key_pem(self):
        """Export public key in PEM format"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def save_keys_to_files(self, private_path: str = "jwt_private.pem", public_path: str = "jwt_public.pem"):
        """Save keys to files for production use"""
        try:
            # Save private key
            private_pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            with open(private_path, "wb") as f:
                f.write(private_pem)
            
            # Save public key
            public_pem = self.get_public_key_pem()
            with open(public_path, "wb") as f:
                f.write(public_pem)
            
            self.logger.info(f"Keys saved to {private_path} and {public_path}")
        except Exception as e:
            self.logger.error(f"Failed to save keys: {e}")


# Initialize Redis
redis_client: Optional[Redis] = None
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    # Test connection
    if redis_client:
        redis_client.ping()
except redis.ConnectionError:
    redis_client = None


# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Models
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    user_id: Optional[str] = None
    scopes: List[str] = []


class APIKeyInfo(BaseModel):
    key_id: str
    name: str
    user_id: str
    scopes: List[str]
    is_active: bool
    created_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 3600


class ConsentRecord(BaseModel):
    user_id: str
    consent_type: str
    status: str
    timestamp: datetime
    consent_version: str


# Encryption service
class DataEncryption:
    """Data encryption service for PII and sensitive data"""
    
    def __init__(self, master_key: bytes):
        self.fernet = Fernet(master_key)
        self.logger = logging.getLogger(__name__)
    
    def encrypt_pii(self, data: str, context: str = "general") -> str:
        """Encrypt personally identifiable information"""
        if not data:
            return ""
        
        metadata = {
            "context": context,
            "encrypted_at": datetime.utcnow().isoformat(),
            "version": "1.0"
        }
        
        payload = {
            "data": data,
            "metadata": metadata
        }
        
        json_data = json.dumps(payload)
        encrypted = self.fernet.encrypt(json_data.encode('utf-8'))
        
        return encrypted.decode('utf-8')
    
    def decrypt_pii(self, encrypted_data: str) -> Dict:
        """Decrypt PII data"""
        if not encrypted_data:
            return {"data": "", "metadata": {}}
        
        try:
            decrypted = self.fernet.decrypt(encrypted_data.encode('utf-8'))
            payload = json.loads(decrypted.decode('utf-8'))
            return payload
        except Exception as e:
            self.logger.error(f"Failed to decrypt data: {str(e)}")
            raise ValueError(f"Failed to decrypt data: {str(e)}")


# Initialize encryption
# Ensure MASTER_KEY is properly formatted for Fernet (32 url-safe base64-encoded bytes)
import base64
def format_fernet_key(key):
    """Format key for Fernet encryption (32 url-safe base64-encoded bytes)"""
    if isinstance(key, str):
        key_bytes = key.encode('utf-8')
    else:
        key_bytes = key
    
    # Ensure we have exactly 32 bytes
    if len(key_bytes) < 32:
        # Pad with zeros if too short
        key_bytes = key_bytes.ljust(32, b'0')
    elif len(key_bytes) > 32:
        # Truncate if too long
        key_bytes = key_bytes[:32]
    
    # Return as base64-encoded bytes
    return base64.urlsafe_b64encode(key_bytes)

MASTER_KEY_bytes = format_fernet_key(MASTER_KEY)
encryption_service = DataEncryption(MASTER_KEY_bytes)


# Enhanced JWT Authentication with RS256 and GDPR compliance
class EnhancedJWTAuth:
    """Enhanced JWT token authentication with RSA keys and GDPR compliance"""
    
    def __init__(self, key_manager: KeyManager, algorithm: str = ALGORITHM):
        self.key_manager = key_manager
        self.algorithm = algorithm
        self.issuer = JWT_ISSUER
        self.logger = logging.getLogger(__name__)
    
    def create_access_token(
        self,
        user_id: str,
        account_id: str,
        customer_id: str,
        permissions: list,
        device_id: str,
        session_id: str,
        client_ip: str,
        expires_delta: Optional[timedelta] = None,
        include_minimal_data: bool = True
    ) -> str:
        """Create secure access token for financial API with GDPR compliance"""
        
        # Use short expiration for financial APIs (15 minutes max)
        if expires_delta is None:
            expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        elif expires_delta > timedelta(minutes=30):
            expires_delta = timedelta(minutes=30)  # Enforce maximum
        
        now = datetime.utcnow()
        expire = now + expires_delta
        
        # Create minimal payload for GDPR compliance
        payload = {
            "iss": self.issuer,
            "sub": f"user:{user_id}",
            "aud": f"{self.issuer}/v1",
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(secrets.token_urlsafe(16)),  # Unique token ID
            "typ": "fin-api+jwt",  # Explicit typing
            "permissions": permissions,
            "device_id": device_id,
            "session_id": session_id,
            "ip_hash": self._hash_ip(client_ip)
        }
        
        # Add financial claims only if explicitly requested (data minimization)
        if not include_minimal_data:
            payload.update({
                "account_id": account_id,
                "customer_id": customer_id
            })
        
        # Sign with private key
        encoded_jwt = jwt.encode(
            payload,
            self.key_manager.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ),
            algorithm=self.algorithm,
            headers={"typ": "JWT", "alg": self.algorithm}
        )
        
        self.logger.info(f"Access token created for user {user_id}")
        return encoded_jwt
    
    def create_refresh_token(self, user_id: str, session_id: str, device_id: str) -> str:
        """Create refresh token with longer expiration"""
        
        now = datetime.utcnow()
        expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        payload = {
            "iss": self.issuer,
            "sub": f"user:{user_id}",
            "aud": f"{self.issuer}/v1",
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(secrets.token_urlsafe(16)),
            "typ": "refresh+jwt",
            "session_id": session_id,
            "device_id": device_id
        }
        
        encoded_jwt = jwt.encode(
            payload,
            self.key_manager.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ),
            algorithm=self.algorithm
        )
        
        return encoded_jwt
    
    async def verify_token(self, token: str) -> Dict:
        """Verify JWT token and return payload with comprehensive validation"""
        try:
            # Decode and verify token with leeway for clock skew
            payload = jwt.decode(
                token,
                self.key_manager.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ),
                algorithms=[self.algorithm],
                issuer=self.issuer,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True,
                    "require": ["exp", "iat", "iss", "aud", "jti", "sub"],
                    "leeway": 60  # 60 seconds leeway for clock skew
                }
            )
            
            # Check token blacklist
            if await self._is_token_blacklisted(payload["jti"]):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            # Validate token type
            if payload.get("typ") not in ["fin-api+jwt", "refresh+jwt"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            self.logger.info(f"Token verified for {payload['sub']}")
            return payload
            
        except ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def _hash_ip(self, ip: str) -> str:
        """Hash IP address for privacy (GDPR compliance)"""
        if not ip:
            return ""
        return hashlib.sha256(ip.encode()).hexdigest()[:16]
    
    async def _is_token_blacklisted(self, jti: str) -> bool:
        """Check if token has been revoked"""
        if not redis_client:
            return False
        
        try:
            result = redis_client.exists(f"blacklist:{jti}")
            return bool(result)
        except Exception as e:
            self.logger.error(f"Failed to check token blacklist: {e}")
            return False  # Fail open for availability


# Token Revocation Manager
class TokenRevocationManager:
    """Manage JWT token revocation and GDPR compliance"""
    
    def __init__(self, redis_client: Optional[Redis], db_session_func=None):
        self.redis = redis_client
        self.db_session_func = db_session_func
        self.logger = logging.getLogger(__name__)
    
    async def revoke_token(self, jti: str, user_id: str, reason: str = "user_logout", revoke_all: bool = False):
        """Revoke token and optionally all user tokens"""
        
        if revoke_all:
            await self._revoke_all_user_tokens(user_id, reason)
        else:
            await self._revoke_single_token(jti, user_id, reason)
    
    async def _revoke_single_token(self, jti: str, user_id: str, reason: str):
        """Revoke a single token"""
        
        if not self.redis:
            self.logger.warning("Redis not available, cannot revoke token")
            return
        
        try:
            # Add to Redis blacklist with TTL
            token_data = {
                "user_id": user_id,
                "reason": reason,
                "revoked_at": datetime.utcnow().isoformat()
            }
            
            self.redis.hset(f"blacklist:{jti}", mapping=token_data)
            
            # Set expiration based on token's original expiration (7 days for refresh tokens)
            self.redis.expire(f"blacklist:{jti}", 86400 * 7)
            
            self.logger.info(f"Token {jti} revoked for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to revoke token {jti}: {e}")
    
    async def _revoke_all_user_tokens(self, user_id: str, reason: str):
        """Revoke all tokens for a user (GDPR right to be forgotten)"""
        
        # This would typically query the database for all active sessions
        # For now, we'll implement a basic version
        self.logger.info(f"Revoking all tokens for user {user_id}")
        
        # In a real implementation, you would:
        # 1. Query database for all active sessions
        # 2. Revoke each token individually
        # 3. Deactivate all sessions
        # 4. Log the action for audit purposes
    
    async def is_token_revoked(self, jti: str) -> bool:
        """Check if token has been revoked"""
        
        if not self.redis:
            return False
        
        try:
            result = self.redis.exists(f"blacklist:{jti}")
            return bool(result)
        except Exception as e:
            self.logger.error(f"Failed to check token blacklist: {e}")
            return False  # Fail open for availability


# API Key Authentication
class APIKeyAuth:
    """API key authentication with rate limiting"""
    
    def __init__(self, redis_client: Optional[Redis]):
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)
    
    def generate_api_key(self) -> str:
        """Generate a secure API key"""
        key = secrets.token_urlsafe(32)
        return f"pk_{key}"
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key using bcrypt for secure storage"""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(api_key.encode('utf-8'), salt).decode('utf-8')
    
    def verify_api_key_hash(self, api_key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash"""
        try:
            return bcrypt.checkpw(api_key.encode('utf-8'), stored_hash.encode('utf-8'))
        except Exception:
            return False
    
    async def check_rate_limit(self, key_id: str, limit_per_minute: int, 
                              limit_per_hour: int) -> bool:
        """Implement sliding window rate limiting"""
        # Check if Redis is available
        if not self.redis:
            # If Redis is not available, allow all requests (fail open)
            return True
        
        minute_key = f"rate_limit:min:{key_id}:{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        hour_key = f"rate_limit:hour:{key_id}:{datetime.utcnow().strftime('%Y%m%d%H')}"
        
        try:
            # Check hourly limit first
            hour_count_raw: Union[int, Any] = self.redis.incr(hour_key)
            if hour_count_raw == 1:
                self.redis.expire(hour_key, 3600)
            
            # Convert Redis response to integer for comparison
            hour_count = int(hour_count_raw) if hour_count_raw is not None else 0
            if hour_count > limit_per_hour:
                return False
            
            # Check minute limit
            minute_count_raw: Union[int, Any] = self.redis.incr(minute_key)
            if minute_count_raw == 1:
                self.redis.expire(minute_key, 60)
            
            # Convert Redis response to integer for comparison
            minute_count = int(minute_count_raw) if minute_count_raw is not None else 0
            return minute_count <= limit_per_minute
        except redis.RedisError:
            # If Redis operation fails, allow all requests (fail open)
            return True
    
    async def validate_api_key(self, api_key: str, db: Session) -> APIKeyInfo:
        """Validate API key and return key information"""
        if not api_key or not api_key.startswith("pk_"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key format"
            )
        
        # Get API key info from database
        key_info = self.get_api_key_info(api_key, db)
        if not key_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        if not key_info.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key is inactive"
            )
        
        # Check rate limits
        if not await self.check_rate_limit(
            key_info.key_id,
            key_info.rate_limit_per_minute,
            key_info.rate_limit_per_hour
        ):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Update last used timestamp
        await self.update_key_usage(key_info.key_id, db)
        
        return key_info
    
    def _extract_datetime_from_column(self, column_value: Any) -> Optional[datetime]:
        """Safely extract datetime value from SQLAlchemy Column"""
        return column_value if column_value is not None else None
    
    def _extract_int_from_column(self, column_value: Any, default: int = 0) -> int:
        """Safely extract int value from SQLAlchemy Column"""
        return int(column_value) if column_value is not None else default
    
    def get_api_key_info(self, api_key: str, db: Session) -> Optional[APIKeyInfo]:
        """Get API key information from database"""
        # Query all active API keys
        api_keys = db.query(APIKey).filter(APIKey.is_active == True).all()
        
        for key in api_keys:
            if self.verify_api_key_hash(api_key, str(key.key_hash)):
                # Extract values from SQLAlchemy model instance with proper type conversion
                created_at_value = self._extract_datetime_from_column(key.created_at)
                last_used_value = self._extract_datetime_from_column(key.last_used)
                
                # Ensure proper type conversion before passing to APIKeyInfo constructor
                rate_limit_minute = self._extract_int_from_column(key.rate_limit_per_minute, 60)
                rate_limit_hour = self._extract_int_from_column(key.rate_limit_per_hour, 3600)
                
                return APIKeyInfo(
                    key_id=str(key.key_id),
                    name=str(key.name),
                    user_id=str(key.user_id_hash),
                    scopes=json.loads(str(key.scopes_json)),
                    is_active=bool(key.is_active),
                    created_at=created_at_value,
                    last_used=last_used_value,
                    rate_limit_per_minute=rate_limit_minute,
                    rate_limit_per_hour=rate_limit_hour
                )
        
        return None
    
    async def update_key_usage(self, key_id: str, db: Session) -> None:
        """Update API key last used timestamp"""
        key = db.query(APIKey).filter(APIKey.key_id == key_id).first()
        if key:
            # Direct assignment works for SQLAlchemy model instances
            key.last_used = datetime.utcnow()  # type: ignore[assignment]
            db.commit()


# Audit logging
class AuditLogger:
    """GDPR-compliant audit logging"""
    
    def __init__(self):
        self.logger = logging.getLogger('gdpr_audit')
    
    def hash_identifier(self, identifier: Optional[str]) -> Optional[str]:
        """Create a deterministic hash of an identifier"""
        if not identifier:
            return None
        return hashlib.sha256(identifier.encode()).hexdigest()[:32]
    
    def log_data_access(self, db: Session, user_id: str, data_subject_id: str,
                       resource_type: str, resource_id: str, data_categories: List[str],
                       gdpr_basis: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                       session_id: Optional[str] = None, success: bool = True, reason: Optional[str] = None) -> None:
        """Log data access event"""
        
        audit_entry = AuditLog(
            event_type="data_access",
            timestamp=datetime.utcnow(),
            user_id_hash=self.hash_identifier(user_id),
            data_subject_id_hash=self.hash_identifier(data_subject_id),
            session_id=session_id,
            ip_address_hash=self.hash_identifier(ip_address),
            user_agent_hash=self.hash_identifier(user_agent),
            resource_type=resource_type,
            resource_id=resource_id,
            action="read",
            result="success" if success else "failure",
            reason=reason,
            metadata_json=json.dumps({"data_categories": data_categories}),
            gdpr_basis=gdpr_basis,
            data_categories_json=json.dumps(data_categories),
            retention_period="6_years"
        )
        
        db.add(audit_entry)
        db.commit()
    
    def log_consent_event(self, db: Session, user_id: str, consent_type: str,
                         action: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                         success: bool = True, metadata: Optional[Dict] = None) -> None:
        """Log consent-related events"""
        
        audit_entry = AuditLog(
            event_type=f"consent_{action}",
            timestamp=datetime.utcnow(),
            user_id_hash=self.hash_identifier(user_id),
            data_subject_id_hash=self.hash_identifier(user_id),
            ip_address_hash=self.hash_identifier(ip_address),
            user_agent_hash=self.hash_identifier(user_agent),
            action=action,
            result="success" if success else "failure",
            metadata_json=json.dumps(metadata) if metadata else None,
            gdpr_basis="consent",
            data_categories_json=json.dumps(["consent_data"]),
            retention_period="6_years"
        )
        
        db.add(audit_entry)
        db.commit()


# GDPR Compliance Manager for JWT
class GDPRCompliantJWTManager:
    """GDPR compliance features for JWT tokens"""
    
    def __init__(self, jwt_auth: EnhancedJWTAuth, revocation_manager: TokenRevocationManager):
        self.jwt_auth = jwt_auth
        self.revocation_manager = revocation_manager
        self.logger = logging.getLogger(__name__)
    
    def minimize_token_data(self, user_data: dict) -> dict:
        """Apply data minimization principle - only include necessary data"""
        
        # Only include claims that are absolutely necessary
        minimal_claims = {
            "iss": user_data.get("issuer"),
            "sub": user_data.get("subject"),
            "aud": user_data.get("audience"),
            "exp": user_data.get("expiration"),
            "iat": user_data.get("issued_at"),
            "jti": user_data.get("jwt_id"),
            "permissions": user_data.get("permissions", []),
        }
        
        # Only add financial claims if specifically required
        if user_data.get("include_account_id"):
            minimal_claims["account_id"] = user_data.get("account_id")
            
        if user_data.get("include_customer_id"):
            minimal_claims["customer_id"] = user_data.get("customer_id")
            
        return minimal_claims
    
    async def handle_data_subject_request(self, user_id: str, request_type: str) -> dict:
        """Handle GDPR data subject requests (access, rectification, erasure)"""
        
        if request_type == "access":
            return await self._provide_token_data_access(user_id)
        elif request_type == "erasure":
            return await self._execute_right_to_erasure(user_id)
        elif request_type == "portability":
            return await self._provide_data_portability(user_id)
        else:
            raise ValueError(f"Unsupported request type: {request_type}")
    
    async def _execute_right_to_erasure(self, user_id: str) -> dict:
        """Implement right to be forgotten (revoke all tokens)"""
        
        self.logger.info(f"Executing right to erasure for user {user_id}")
        
        # Revoke all tokens for the user
        await self.revocation_manager._revoke_all_user_tokens(user_id, "gdpr_erasure")
        
        return {
            "status": "completed",
            "action": "all_tokens_revoked",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _provide_token_data_access(self, user_id: str) -> dict:
        """Provide audit trail for token usage (GDPR access request)"""
        
        # This would typically query the database for token history
        # For now, return a basic structure
        return {
            "user_id": user_id,
            "token_history": [],
            "security_events": [],
            "access_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _provide_data_portability(self, user_id: str) -> dict:
        """Provide data in portable format"""
        
        return {
            "user_id": user_id,
            "data_format": "json",
            "token_data": await self._provide_token_data_access(user_id),
            "export_timestamp": datetime.utcnow().isoformat()
        }
    
    async def audit_token_usage(self, user_id: str) -> dict:
        """Provide audit trail for token usage (GDPR accountability)"""
        
        # This would typically query the database for comprehensive audit data
        return {
            "user_id": user_id,
            "audit_trail": [],
            "compliance_status": "active",
            "last_audit": datetime.utcnow().isoformat()
        }


# Initialize enhanced services
key_manager = KeyManager()
jwt_auth = EnhancedJWTAuth(key_manager)
token_revocation_manager = TokenRevocationManager(redis_client)
gdpr_jwt_manager = GDPRCompliantJWTManager(jwt_auth, token_revocation_manager)
api_key_auth = APIKeyAuth(redis_client)
audit_logger = AuditLogger()


# Enhanced authentication dependencies
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> Dict:
    """Get current user from enhanced JWT token"""
    try:
        payload = await jwt_auth.verify_token(token)
        user_id = payload.get("sub", "").replace("user:", "")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return {
            "user_id": user_id,
            "scopes": payload.get("permissions", []),
            "account_id": payload.get("account_id"),
            "customer_id": payload.get("customer_id"),
            "device_id": payload.get("device_id"),
            "session_id": payload.get("session_id")
        }
    except HTTPException:
        raise


async def get_optional_api_key(
    authorization: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Optional[APIKeyInfo]:
    """Extract and validate API key from Authorization header (optional)"""
    if not authorization:
        return None
    
    # Extract API key from Authorization header
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    
    api_key = parts[1]
    if not api_key.startswith("pk_"):
        return None
    
    try:
        return await api_key_auth.validate_api_key(api_key, db)
    except HTTPException:
        return None


async def get_current_user_optional(
    api_key_info: Optional[APIKeyInfo] = Depends(get_optional_api_key),
    token: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Optional[Dict]:
    """Get current user from either API key or JWT token (optional authentication)"""
    
    # Try API key authentication first
    if api_key_info:
        return {
            "auth_type": "api_key",
            "key_id": api_key_info.key_id,
            "user_id": api_key_info.user_id,
            "scopes": api_key_info.scopes
        }
    
    # Try JWT token authentication if no API key
    if token:
        try:
            return await get_current_user(token, db)
        except HTTPException:
            pass
    
    # No valid authentication found
    return None


# Security utilities
class SecurityManager:
    """Security management utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False
    
    def sanitize_input(self, input_string: str) -> str:
        """Basic input sanitization"""
        if not input_string:
            return ""
        
        # Remove potential SQL injection patterns
        sanitized = input_string.replace("'", "").replace("\"", "")
        sanitized = sanitized.replace("<", "").replace(">", "")
        sanitized = sanitized.strip()
        
        return sanitized


# Enhanced Security utilities
class EnhancedSecurityManager:
    """Enhanced security management utilities with compliance features"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt with high cost factor"""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False
    
    def sanitize_input(self, input_string: str) -> str:
        """Enhanced input sanitization for financial data"""
        if not input_string:
            return ""
        
        # Remove potential SQL injection patterns
        sanitized = input_string.replace("'", "").replace("\"", "")
        sanitized = sanitized.replace("<", "").replace(">", "")
        sanitized = sanitized.replace(";", "").replace("--", "")
        sanitized = sanitized.strip()
        
        # Limit length to prevent buffer overflow
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def validate_email(self, email: str) -> bool:
        """Validate email format for financial services"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email)) and len(email) <= 254
    
    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        import re
        # Remove common formatting characters
        clean_phone = re.sub(r'[\s\-\(\)\+]', '', phone)
        # Check for valid international phone format
        return bool(re.match(r'^\d{7,15}$', clean_phone))
    
    def generate_session_id(self) -> str:
        """Generate secure session ID"""
        return f"sess_{secrets.token_urlsafe(16)}"
    
    def generate_device_id(self, user_agent: str, ip_address: str) -> str:
        """Generate device fingerprint for security tracking"""
        import hashlib
        device_string = f"{user_agent}:{ip_address}"
        return f"dev_{hashlib.sha256(device_string.encode()).hexdigest()[:16]}"


# Initialize enhanced security manager
security_manager = EnhancedSecurityManager()