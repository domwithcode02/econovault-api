"""
API Key rotation and lifecycle management for EconoVault API.
Provides automated key rotation, expiration, and lifecycle management.
"""

from __future__ import annotations
import secrets
import hashlib
import json
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta, timezone
import logging
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from database import APIKey, get_db
from security import audit_logger, security_manager

logger = logging.getLogger(__name__)


class APIKeyStatus(str, Enum):
    """API Key lifecycle statuses"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ROTATED = "rotated"
    SUSPENDED = "suspended"


class APIKeyRotationPolicy(str, Enum):
    """API Key rotation policies"""
    MANUAL = "manual"
    AUTOMATIC_30_DAYS = "30_days"
    AUTOMATIC_90_DAYS = "90_days"
    AUTOMATIC_180_DAYS = "180_days"
    AUTOMATIC_365_DAYS = "365_days"


class APIKeyEventType(str, Enum):
    """API Key event types for audit logging"""
    CREATED = "created"
    ROTATED = "rotated"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"
    REACTIVATED = "reactivated"
    USED = "used"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class APIKeyRotationRequest(BaseModel):
    """Request for API key rotation"""
    key_id: str = Field(..., description="ID of the key to rotate")
    reason: Optional[str] = Field(None, description="Reason for rotation")
    notify_user: bool = Field(True, description="Whether to notify the user")
    immediate: bool = Field(False, description="Whether to rotate immediately or schedule")
    
    class Config:
        json_schema_extra = {
            "example": {
                "key_id": "pk_abc123def456",
                "reason": "Scheduled rotation for security compliance",
                "notify_user": True,
                "immediate": False
            }
        }


class APIKeyLifecycleConfig(BaseModel):
    """Configuration for API key lifecycle management"""
    default_expiry_days: int = Field(365, description="Default expiry period in days")
    rotation_policy: APIKeyRotationPolicy = Field(APIKeyRotationPolicy.AUTOMATIC_90_DAYS, description="Rotation policy")
    max_active_keys_per_user: int = Field(5, description="Maximum active keys per user")
    rate_limit_threshold: int = Field(100, description="Rate limit threshold for suspicious activity")
    auto_suspend_after_rate_limit: bool = Field(True, description="Auto-suspend keys after rate limit violations")
    notification_before_expiry_days: int = Field(7, description="Days before expiry to send notification")
    
    @validator('default_expiry_days')
    def validate_expiry_days(cls, v):
        if v < 1 or v > 3650:
            raise ValueError("Expiry days must be between 1 and 3650")
        return v
    
    @validator('max_active_keys_per_user')
    def validate_max_keys(cls, v):
        if v < 1 or v > 100:
            raise ValueError("Max active keys must be between 1 and 100")
        return v


class APIKeyInfo(BaseModel):
    """API Key information for responses"""
    key_id: str
    name: str
    scopes: List[str]
    is_active: bool
    status: APIKeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    usage_count: int
    rate_limit_per_minute: int
    rate_limit_per_hour: int
    rotation_policy: APIKeyRotationPolicy
    next_rotation_date: Optional[datetime]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class APIKeyRotationService:
    """Service for managing API key rotation and lifecycle"""
    
    def __init__(self, config: Optional[APIKeyLifecycleConfig] = None):
        self.config = config or APIKeyLifecycleConfig(
            default_expiry_days=365,
            rotation_policy=APIKeyRotationPolicy.AUTOMATIC_90_DAYS,
            max_active_keys_per_user=5,
            rate_limit_threshold=100,
            auto_suspend_after_rate_limit=True,
            notification_before_expiry_days=7
        )
        self.logger = logging.getLogger(__name__)
    
    async def create_api_key(
        self,
        user_id: str,
        name: str,
        scopes: List[str],
        db: Session,
        expires_in_days: Optional[int] = None,
        rotation_policy: Optional[APIKeyRotationPolicy] = None
    ) -> tuple[str, APIKeyInfo]:
        """Create a new API key with lifecycle management"""
        
        # Generate secure API key
        raw_key = self._generate_api_key()
        key_hash = self._hash_api_key(raw_key)
        key_id = security_manager.generate_secure_token(16)
        
        # Hash user ID for GDPR compliance
        user_id_hash = audit_logger.hash_identifier(user_id)
        
        # Calculate expiry
        if expires_in_days is None:
            expires_in_days = self.config.default_expiry_days
        
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Calculate next rotation date
        rotation_policy = rotation_policy or self.config.rotation_policy
        next_rotation = self._calculate_next_rotation(rotation_policy, expires_at)
        
        # Check key limit
        active_keys_count = db.query(APIKey).filter(
            APIKey.user_id_hash == user_id_hash,
            APIKey.is_active == True,
            APIKey.status == APIKeyStatus.ACTIVE
        ).count()
        
        if active_keys_count >= self.config.max_active_keys_per_user:
            raise ValueError(f"Maximum active keys limit reached ({self.config.max_active_keys_per_user})")
        
        # Create API key record
        api_key_record = APIKey(
            key_id=key_id,
            name=name,
            user_id_hash=user_id_hash,
            key_hash=key_hash,
            scopes_json=json.dumps(scopes),
            is_active=True,
            status=APIKeyStatus.ACTIVE,
            rate_limit_per_minute=60,
            rate_limit_per_hour=3600,
            rotation_policy=rotation_policy.value,
            expires_at=expires_at,
            next_rotation_date=next_rotation
        )
        
        db.add(api_key_record)
        db.commit()
        db.refresh(api_key_record)
        
        # Log the creation
        await self._log_key_event(
            db=db,
            key_id=key_id,
            user_id=user_id,
            event_type=APIKeyEventType.CREATED,
            details={
                "name": name,
                "scopes": scopes,
                "expires_at": expires_at.isoformat(),
                "rotation_policy": rotation_policy.value
            }
        )
        
        self.logger.info(f"Created API key {key_id} for user {user_id}")
        
        # Return raw key (only shown once) and key info
        key_info = self._create_key_info(api_key_record)
        return raw_key, key_info
    
    async def rotate_api_key(
        self,
        user_id: str,
        key_id: str,
        db: Session,
        reason: Optional[str] = None,
        notify_user: bool = True
    ) -> tuple[str, APIKeyInfo]:
        """Rotate an existing API key"""
        
        # Get existing key
        user_id_hash = audit_logger.hash_identifier(user_id)
        existing_key = db.query(APIKey).filter(
            APIKey.key_id == key_id,
            APIKey.user_id_hash == user_id_hash
        ).first()
        
        if not existing_key:
            raise ValueError(f"API key {key_id} not found or not owned by user {user_id}")
        
        if str(existing_key.status) != APIKeyStatus.ACTIVE.value:
            raise ValueError(f"API key {key_id} is not active and cannot be rotated")
        
        # Mark existing key as rotated
        existing_key.status = APIKeyStatus.ROTATED.value
        existing_key.is_active = False
        existing_key.next_rotation_date = None
        
        # Create new key with same configuration
        scopes_json = str(existing_key.scopes_json) if existing_key.scopes_json else "[]"
        scopes = json.loads(scopes_json)
        expires_at = existing_key.expires_at
        expires_in_days = (expires_at - datetime.utcnow()).days if expires_at else None
        rotation_policy = APIKeyRotationPolicy(str(existing_key.rotation_policy))
        
        raw_key, key_info = await self.create_api_key(
            user_id=user_id,
            name=f"{existing_key.name} (Rotated)",
            scopes=scopes,
            db=db,
            expires_in_days=expires_in_days,
            rotation_policy=rotation_policy
        )
        
        # Log the rotation
        await self._log_key_event(
            db=db,
            key_id=key_id,
            user_id=user_id,
            event_type=APIKeyEventType.ROTATED,
            details={
                "reason": reason,
                "new_key_id": key_info.key_id,
                "notify_user": notify_user
            }
        )
        
        self.logger.info(f"Rotated API key {key_id} for user {user_id}, new key: {key_info.key_id}")
        
        return raw_key, key_info
    
    async def revoke_api_key(
        self,
        user_id: str,
        key_id: str,
        db: Session,
        reason: Optional[str] = None
    ) -> bool:
        """Revoke an API key"""
        
        user_id_hash = audit_logger.hash_identifier(user_id)
        api_key = db.query(APIKey).filter(
            APIKey.key_id == key_id,
            APIKey.user_id_hash == user_id_hash
        ).first()
        
        if not api_key:
            raise ValueError(f"API key {key_id} not found or not owned by user {user_id}")
        
        if str(api_key.status) == APIKeyStatus.REVOKED.value:
            return True  # Already revoked
        
        # Revoke the key
        api_key.status = APIKeyStatus.REVOKED.value
        api_key.is_active = False
        api_key.next_rotation_date = None
        
        db.commit()
        
        # Log the revocation
        await self._log_key_event(
            db=db,
            key_id=key_id,
            user_id=user_id,
            event_type=APIKeyEventType.REVOKED,
            details={"reason": reason}
        )
        
        self.logger.info(f"Revoked API key {key_id} for user {user_id}")
        return True
    
    async def suspend_api_key(
        self,
        user_id: str,
        key_id: str,
        db: Session,
        reason: Optional[str] = None
    ) -> bool:
        """Suspend an API key temporarily"""
        
        user_id_hash = audit_logger.hash_identifier(user_id)
        api_key = db.query(APIKey).filter(
            APIKey.key_id == key_id,
            APIKey.user_id_hash == user_id_hash
        ).first()
        
        if not api_key:
            raise ValueError(f"API key {key_id} not found or not owned by user {user_id}")
        
        if str(api_key.status) != APIKeyStatus.ACTIVE.value:
            raise ValueError(f"API key {key_id} is not active and cannot be suspended")
        
        # Suspend the key
        api_key.status = APIKeyStatus.SUSPENDED.value
        api_key.is_active = False
        
        db.commit()
        
        # Log the suspension
        await self._log_key_event(
            db=db,
            key_id=key_id,
            user_id=user_id,
            event_type=APIKeyEventType.SUSPENDED,
            details={"reason": reason}
        )
        
        self.logger.info(f"Suspended API key {key_id} for user {user_id}")
        return True
    
    async def reactivate_api_key(
        self,
        user_id: str,
        key_id: str,
        db: Session
    ) -> bool:
        """Reactivate a suspended API key"""
        
        user_id_hash = audit_logger.hash_identifier(user_id)
        api_key = db.query(APIKey).filter(
            APIKey.key_id == key_id,
            APIKey.user_id_hash == user_id_hash
        ).first()
        
        if not api_key:
            raise ValueError(f"API key {key_id} not found or not owned by user {user_id}")
        
        if str(api_key.status) != APIKeyStatus.SUSPENDED.value:
            raise ValueError(f"API key {key_id} is not suspended")
        
        # Reactivate the key
        api_key.status = APIKeyStatus.ACTIVE.value
        api_key.is_active = True
        
        db.commit()
        
        # Log the reactivation
        await self._log_key_event(
            db=db,
            key_id=key_id,
            user_id=user_id,
            event_type=APIKeyEventType.REACTIVATED
        )
        
        self.logger.info(f"Reactivated API key {key_id} for user {user_id}")
        return True
    
    async def get_api_key_info(
        self,
        user_id: str,
        key_id: str,
        db: Session
    ) -> APIKeyInfo:
        """Get information about an API key"""
        
        user_id_hash = audit_logger.hash_identifier(user_id)
        api_key = db.query(APIKey).filter(
            APIKey.key_id == key_id,
            APIKey.user_id_hash == user_id_hash
        ).first()
        
        if not api_key:
            raise ValueError(f"API key {key_id} not found or not owned by user {user_id}")
        
        return self._create_key_info(api_key)
    
    async def list_api_keys(
        self,
        user_id: str,
        db: Session,
        include_inactive: bool = False
    ) -> List[APIKeyInfo]:
        """List all API keys for a user"""
        
        user_id_hash = audit_logger.hash_identifier(user_id)
        
        query = db.query(APIKey).filter(APIKey.user_id_hash == user_id_hash)
        
        if not include_inactive:
            query = query.filter(APIKey.is_active == True)
        
        api_keys = query.order_by(APIKey.created_at.desc()).all()
        
        return [self._create_key_info(key) for key in api_keys]
    
    async def check_key_expiry_and_rotation(self, db: Session) -> Dict[str, Any]:
        """Check for expired keys and keys due for rotation"""
        now = datetime.utcnow()
        
        # Find expired keys
        expired_keys = db.query(APIKey).filter(
            APIKey.expires_at < now,
            APIKey.status == APIKeyStatus.ACTIVE
        ).all()
        
        # Find keys due for rotation
        rotation_due_keys = db.query(APIKey).filter(
            APIKey.next_rotation_date <= now,
            APIKey.status == APIKeyStatus.ACTIVE,
            APIKey.rotation_policy != APIKeyRotationPolicy.MANUAL
        ).all()
        
        # Process expired keys
        expired_count = 0
        for key in expired_keys:
            key.status = APIKeyStatus.EXPIRED.value
            key.is_active = False
            expired_count += 1
            
            # Log expiration
            await self._log_key_event(
                db=db,
                key_id=str(key.key_id),
                user_id=str(key.user_id_hash),  # This is hashed, would need to be decoded
                event_type=APIKeyEventType.EXPIRED
            )
        
        # Process keys due for rotation
        rotated_count = 0
        for key in rotation_due_keys:
            try:
                # This would need proper user ID resolution
                # For now, we'll just log the rotation due
                self.logger.info(f"Key {key.key_id} is due for rotation")
                rotated_count += 1
            except Exception as e:
                self.logger.error(f"Error rotating key {key.key_id}: {str(e)}")
        
        if expired_count > 0 or rotated_count > 0:
            db.commit()
        
        return {
            "expired_keys_processed": expired_count,
            "rotation_due_keys_found": rotated_count,
            "timestamp": now.isoformat()
        }
    
    async def cleanup_old_keys(self, db: Session, days_old: int = 365) -> int:
        """Clean up old revoked/expired keys"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        old_keys = db.query(APIKey).filter(
            APIKey.created_at < cutoff_date,
            APIKey.status.in_([APIKeyStatus.REVOKED, APIKeyStatus.EXPIRED])
        ).all()
        
        deleted_count = 0
        for key in old_keys:
            db.delete(key)
            deleted_count += 1
        
        if deleted_count > 0:
            db.commit()
            self.logger.info(f"Cleaned up {deleted_count} old API keys")
        
        return deleted_count
    
    async def send_expiry_notifications(self, db: Session) -> int:
        """Send notifications for keys expiring soon"""
        
        notification_date = datetime.utcnow() + timedelta(days=self.config.notification_before_expiry_days)
        
        expiring_keys = db.query(APIKey).filter(
            APIKey.expires_at <= notification_date,
            APIKey.expires_at > datetime.utcnow(),
            APIKey.status == APIKeyStatus.ACTIVE,
            APIKey.expiry_notification_sent == False  # This field would need to be added to the model
        ).all()
        
        # For now, just log the notification need
        notification_count = len(expiring_keys)
        
        for key in expiring_keys:
            self.logger.info(f"Key {key.key_id} expires in {self.config.notification_before_expiry_days} days")
            # In a real implementation, this would send email/SMS notifications
        
        return notification_count
    
    def _generate_api_key(self) -> str:
        """Generate a secure API key"""
        key = secrets.token_urlsafe(32)
        return f"pk_{key}"
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key using SHA-256"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def _calculate_next_rotation(self, rotation_policy: APIKeyRotationPolicy, expires_at: datetime | None) -> Optional[datetime]:
        """Calculate next rotation date based on policy"""
        if rotation_policy == APIKeyRotationPolicy.MANUAL:
            return None
        
        rotation_days = {
            APIKeyRotationPolicy.AUTOMATIC_30_DAYS: 30,
            APIKeyRotationPolicy.AUTOMATIC_90_DAYS: 90,
            APIKeyRotationPolicy.AUTOMATIC_180_DAYS: 180,
            APIKeyRotationPolicy.AUTOMATIC_365_DAYS: 365
        }
        
        days = rotation_days.get(rotation_policy, 90)
        next_rotation = datetime.utcnow() + timedelta(days=days)
        
        # Don't rotate after expiry
        if expires_at is not None and next_rotation >= expires_at:
            return None
        
        return next_rotation
    
    def _create_key_info(self, api_key: APIKey) -> APIKeyInfo:
        """Create APIKeyInfo from database record"""
        import json
        
        scopes_json = str(api_key.scopes_json) if api_key.scopes_json else "[]"
        scopes = json.loads(scopes_json)
        
        return APIKeyInfo(
            key_id=str(api_key.key_id),
            name=str(api_key.name),
            scopes=scopes,
            is_active=bool(api_key.is_active),
            status=APIKeyStatus(str(api_key.status)),
            created_at=api_key.created_at,
            expires_at=api_key.expires_at,
            last_used=api_key.last_used,
            usage_count=int(api_key.usage_count or 0),
            rate_limit_per_minute=int(api_key.rate_limit_per_minute),
            rate_limit_per_hour=int(api_key.rate_limit_per_hour),
            rotation_policy=APIKeyRotationPolicy(str(api_key.rotation_policy)),
            next_rotation_date=api_key.next_rotation_date
        )
    
    async def _log_key_event(
        self,
        db: Session,
        key_id: str,
        user_id: str,
        event_type: APIKeyEventType,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log API key event for audit purposes"""
        
        # This would integrate with your existing audit logging system
        # For now, just log locally
        self.logger.info(f"API Key Event: {event_type.value} for key {key_id} by user {user_id}")
        if details:
            self.logger.info(f"Event details: {details}")


class APIKeyRotationScheduler:
    """Scheduler for automated API key rotation tasks"""
    
    def __init__(self, rotation_service: APIKeyRotationService):
        self.rotation_service = rotation_service
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.scheduler_task = None
    
    async def start(self, interval_hours: int = 24):
        """Start the rotation scheduler"""
        if self.is_running:
            return
        
        self.is_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop(interval_hours))
        self.logger.info(f"API Key rotation scheduler started with {interval_hours} hour interval")
    
    async def stop(self):
        """Stop the rotation scheduler"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("API Key rotation scheduler stopped")
    
    async def _scheduler_loop(self, interval_hours: int):
        """Main scheduler loop"""
        while self.is_running:
            try:
                await self._run_rotation_tasks()
                await asyncio.sleep(interval_hours * 3600)  # Convert hours to seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in rotation scheduler: {str(e)}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def _run_rotation_tasks(self):
        """Run all rotation-related tasks"""
        self.logger.info("Running API key rotation tasks")
        
        db = next(get_db())
        try:
            # Check for expired keys and rotation due
            expiry_results = await self.rotation_service.check_key_expiry_and_rotation(db)
            
            # Send expiry notifications
            notification_count = await self.rotation_service.send_expiry_notifications(db)
            
            # Clean up old keys
            cleanup_count = await self.rotation_service.cleanup_old_keys(db)
            
            self.logger.info(
                f"Rotation tasks completed: "
                f"expired={expiry_results['expired_keys_processed']}, "
                f"rotation_due={expiry_results['rotation_due_keys_found']}, "
                f"notifications={notification_count}, "
                f"cleaned_up={cleanup_count}"
            )
            
        except Exception as e:
            self.logger.error(f"Error in rotation tasks: {str(e)}")
            db.rollback()
        finally:
            db.close()


# Global instances
api_key_rotation_config = APIKeyLifecycleConfig(
    default_expiry_days=365,
    rotation_policy=APIKeyRotationPolicy.AUTOMATIC_90_DAYS,
    max_active_keys_per_user=5,
    rate_limit_threshold=100,
    auto_suspend_after_rate_limit=True,
    notification_before_expiry_days=7
)
api_key_rotation_service = APIKeyRotationService(api_key_rotation_config)
api_key_rotation_scheduler = APIKeyRotationScheduler(api_key_rotation_service)