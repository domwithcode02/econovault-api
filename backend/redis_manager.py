"""
Redis connection and caching management for EconoVault API.
Provides async Redis operations with connection pooling and fallback mechanisms.
"""

from __future__ import annotations
import json
import hashlib
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Union, Callable, Any, TYPE_CHECKING
from functools import wraps
import asyncio

if TYPE_CHECKING:
    try:
        import redis.asyncio as aioredis
    except ImportError:
        pass

try:
    import redis.asyncio as aioredis
    from redis.exceptions import RedisError, ConnectionError, TimeoutError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
    RedisError = Exception
    ConnectionError = Exception
    TimeoutError = Exception

# Type aliases that work with TYPE_CHECKING
if REDIS_AVAILABLE:
    RedisClientType = Any
    ConnectionPoolType = Any
else:
    RedisClientType = Any
    ConnectionPoolType = Any

logger = logging.getLogger(__name__)


class RedisConfig:
    """Redis configuration settings"""
    
    def __init__(self):
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", "6379"))
        self.db = int(os.getenv("REDIS_DB", "0"))
        self.password = os.getenv("REDIS_PASSWORD", None)
        self.username = os.getenv("REDIS_USERNAME", None)
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
        self.socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))
        self.socket_connect_timeout = float(os.getenv("REDIS_CONNECT_TIMEOUT", "5.0"))
        self.retry_on_timeout = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
        self.health_check_interval = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))
        self.decode_responses = True
        
        # Cache TTL settings (in seconds)
        self.default_ttl = int(os.getenv("REDIS_DEFAULT_TTL", "3600"))  # 1 hour
        self.indicator_data_ttl = int(os.getenv("REDIS_INDICATOR_DATA_TTL", "14400"))  # 4 hours
        self.metadata_ttl = int(os.getenv("REDIS_METADATA_TTL", "86400"))  # 24 hours
        self.gdpr_data_ttl = int(os.getenv("REDIS_GDPR_DATA_TTL", "604800"))  # 7 days
        
        # Circuit breaker settings
        self.circuit_breaker_threshold = int(os.getenv("REDIS_CIRCUIT_BREAKER_THRESHOLD", "5"))
        self.circuit_breaker_timeout = int(os.getenv("REDIS_CIRCUIT_BREAKER_TIMEOUT", "60"))
        
        # Feature flags - disable Redis for development if not available
        redis_enabled_env = os.getenv("REDIS_ENABLED", "auto").lower()
        if redis_enabled_env == "auto":
            # Auto-detect based on Redis availability
            self.enabled = REDIS_AVAILABLE
        else:
            self.enabled = redis_enabled_env == "true"
        
        self.fallback_enabled = os.getenv("REDIS_FALLBACK_ENABLED", "true").lower() == "true"


class CircuitBreaker:
    """Circuit breaker pattern for Redis operations"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()
    
    async def can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        async with self._lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if datetime.now(timezone.utc).timestamp() - (self.last_failure_time or 0) >= self.recovery_timeout:
                    self.state = "half-open"
                    return True
                return False
            else:  # half-open
                return True
    
    async def record_success(self) -> None:
        """Record successful execution"""
        async with self._lock:
            self.failure_count = 0
            self.last_failure_time = None
            self.state = "closed"
    
    async def record_failure(self) -> None:
        """Record failed execution"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc).timestamp()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"


class RedisManager:
    """Manages Redis connections and operations with circuit breaker pattern"""
    
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or RedisConfig()
        self._redis_client: Optional[Any] = None
        self._pool: Optional[Any] = None
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold,
            recovery_timeout=self.config.circuit_breaker_timeout
        )
        self._fallback_cache: Dict[str, Any] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Redis connection with pooling"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available. Running in fallback mode.")
            return
            
        if not self.config.enabled:
            logger.info("Redis is disabled. Running in fallback mode.")
            return
        
        try:
            # Create connection pool with optimized settings
            self._pool = aioredis.ConnectionPool(  # type: ignore
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                username=self.config.username,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval,
                decode_responses=self.config.decode_responses
            )
            
            self._redis_client = aioredis.Redis(connection_pool=self._pool)  # type: ignore
            
            # Test connection
            await self._redis_client.ping()  # type: ignore
            self._initialized = True
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            if not self.config.fallback_enabled:
                raise
            logger.info("Running in fallback mode due to Redis connection failure")
    
    @property
    def client(self) -> Optional[Any]:
        """Get Redis client instance"""
        if not self._initialized or not self._redis_client:
            if not self.config.fallback_enabled:
                raise RuntimeError("Redis connection not initialized")
            return None
        return self._redis_client
    
    async def close(self) -> None:
        """Close Redis connection and pool"""
        if self._redis_client:
            await self._redis_client.aclose()
        if self._pool:
            await self._pool.aclose()
        logger.info("Redis connections closed")
    
    async def safe_redis_operation(
        self,
        operation: Callable,
        fallback_value: Any = None,
        operation_name: str = "redis_operation"
    ) -> Any:
        """Execute Redis operation with circuit breaker and fallback"""
        
        if not await self.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker open for {operation_name}")
            return fallback_value
        
        if not self.client:
            logger.warning(f"No Redis client available for {operation_name}")
            return fallback_value
        
        try:
            result = await operation()
            await self.circuit_breaker.record_success()
            return result
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error in {operation_name}: {e}")
            await self.circuit_breaker.record_failure()
            return fallback_value
            
        except RedisError as e:
            logger.error(f"Redis error in {operation_name}: {e}")
            return fallback_value
            
        except Exception as e:
            logger.error(f"Unexpected error in {operation_name}: {e}")
            return fallback_value


class CacheKeyGenerator:
    """Generates consistent cache keys for different data types"""
    
    @staticmethod
    def indicator_metadata(series_id: str) -> str:
        return f"indicator:metadata:{series_id}"
    
    @staticmethod
    def indicator_data(series_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        date_part = ""
        if start_date or end_date:
            date_part = f":{start_date or 'start'}:{end_date or 'end'}"
        return f"indicator:data:{series_id}{date_part}"
    
    @staticmethod
    def gdpr_deletion_request(request_id: str) -> str:
        return f"gdpr:deletion:{request_id}"
    
    @staticmethod
    def gdpr_consent_record(user_id: str, consent_type: str) -> str:
        return f"gdpr:consent:{user_id}:{consent_type}"
    
    @staticmethod
    def gdpr_data_export(export_id: str) -> str:
        return f"gdpr:export:{export_id}"


class IndicatorCache:
    """Specialized caching for economic indicator data"""
    
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
        self.key_gen = CacheKeyGenerator()
    
    async def store_indicator_metadata(self, series_id: str, metadata: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store indicator metadata in cache"""
        key = self.key_gen.indicator_metadata(series_id)
        ttl = ttl or self.redis.config.metadata_ttl
        
        async def store_operation():
            if self.redis.client:
                await self.redis.client.hset(key, mapping={
                    "series_id": series_id,
                    "title": metadata.get("title", ""),
                    "source": metadata.get("source", ""),
                    "indicator_type": metadata.get("indicator_type", ""),
                    "frequency": metadata.get("frequency", ""),
                    "seasonal_adjustment": metadata.get("seasonal_adjustment", ""),
                    "geography_level": metadata.get("geography_level", ""),
                    "units": metadata.get("units", ""),
                    "cached_at": datetime.now(timezone.utc).isoformat()
                })
                await self.redis.client.expire(key, ttl)
                return True
            return False
        
        return await self.redis.safe_redis_operation(store_operation, fallback_value=False)
    
    async def get_indicator_metadata(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Get indicator metadata from cache"""
        key = self.key_gen.indicator_metadata(series_id)
        
        async def get_operation():
            if self.redis.client:
                metadata = await self.redis.client.hgetall(key)
                if metadata:
                    # Convert cached_at back to datetime for TTL checking
                    cached_at_str = metadata.get("cached_at")
                    if cached_at_str:
                        try:
                            cached_at = datetime.fromisoformat(cached_at_str)
                            # Check if cache is still valid (additional safety)
                            if datetime.now(timezone.utc) - cached_at < timedelta(seconds=self.redis.config.metadata_ttl):
                                return dict(metadata)
                        except ValueError:
                            pass
                return None
            return None
        
        return await self.redis.safe_redis_operation(get_operation, fallback_value=None)
    
    async def store_indicator_data(self, series_id: str, data_points: List[Dict[str, Any]], 
                                  start_date: Optional[str] = None, end_date: Optional[str] = None,
                                  ttl: Optional[int] = None) -> bool:
        """Store indicator time series data in cache"""
        key = self.key_gen.indicator_data(series_id, start_date, end_date)
        ttl = ttl or self.redis.config.indicator_data_ttl
        
        async def store_operation():
            if self.redis.client:
                # Store as JSON string for complex data
                data_json = json.dumps({
                    "series_id": series_id,
                    "data_points": data_points,
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                    "count": len(data_points)
                })
                await self.redis.client.setex(key, ttl, data_json)
                return True
            return False
        
        return await self.redis.safe_redis_operation(store_operation, fallback_value=False)
    
    async def get_indicator_data(self, series_id: str, start_date: Optional[str] = None, 
                                end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get indicator time series data from cache"""
        key = self.key_gen.indicator_data(series_id, start_date, end_date)
        
        async def get_operation():
            if self.redis.client:
                data_json = await self.redis.client.get(key)
                if data_json:
                    try:
                        data = json.loads(data_json)
                        # Validate cache freshness
                        cached_at_str = data.get("cached_at")
                        if cached_at_str:
                            cached_at = datetime.fromisoformat(cached_at_str)
                            ttl = self.redis.config.indicator_data_ttl
                            if datetime.now(timezone.utc) - cached_at < timedelta(seconds=ttl):
                                return data
                    except (json.JSONDecodeError, ValueError):
                        pass
                return None
            return None
        
        return await self.redis.safe_redis_operation(get_operation, fallback_value=None)


class GDPRCache:
    """Specialized caching for GDPR compliance data"""
    
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
        self.key_gen = CacheKeyGenerator()
    
    async def store_deletion_request(self, request_id: str, request_data: Dict[str, Any]) -> bool:
        """Store GDPR deletion request"""
        key = self.key_gen.gdpr_deletion_request(request_id)
        
        async def store_operation():
            if self.redis.client:
                request_json = json.dumps(request_data)
                await self.redis.client.setex(key, self.redis.config.gdpr_data_ttl, request_json)
                return True
            return False
        
        return await self.redis.safe_redis_operation(store_operation, fallback_value=False)
    
    async def get_deletion_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get GDPR deletion request"""
        key = self.key_gen.gdpr_deletion_request(request_id)
        
        async def get_operation():
            if self.redis.client:
                request_json = await self.redis.client.get(key)
                if request_json:
                    try:
                        return json.loads(request_json)
                    except json.JSONDecodeError:
                        pass
                return None
            return None
        
        return await self.redis.safe_redis_operation(get_operation, fallback_value=None)
    
    async def store_consent_record(self, user_id: str, consent_type: str, consent_data: Dict[str, Any]) -> bool:
        """Store GDPR consent record"""
        key = self.key_gen.gdpr_consent_record(user_id, consent_type)
        
        async def store_operation():
            if self.redis.client:
                consent_json = json.dumps(consent_data)
                await self.redis.client.setex(key, self.redis.config.gdpr_data_ttl, consent_json)
                return True
            return False
        
        return await self.redis.safe_redis_operation(store_operation, fallback_value=False)
    
    async def get_consent_record(self, user_id: str, consent_type: str) -> Optional[Dict[str, Any]]:
        """Get GDPR consent record"""
        key = self.key_gen.gdpr_consent_record(user_id, consent_type)
        
        async def get_operation():
            if self.redis.client:
                consent_json = await self.redis.client.get(key)
                if consent_json:
                    try:
                        return json.loads(consent_json)
                    except json.JSONDecodeError:
                        pass
                return None
            return None
        
        return await self.redis.safe_redis_operation(get_operation, fallback_value=None)
    
    async def store_data_export(self, export_id: str, export_data: Dict[str, Any]) -> bool:
        """Store GDPR data export"""
        key = self.key_gen.gdpr_data_export(export_id)
        
        async def store_operation():
            if self.redis.client:
                export_json = json.dumps(export_data)
                await self.redis.client.setex(key, self.redis.config.gdpr_data_ttl, export_json)
                return True
            return False
        
        return await self.redis.safe_redis_operation(store_operation, fallback_value=False)
    
    async def get_data_export(self, export_id: str) -> Optional[Dict[str, Any]]:
        """Get GDPR data export"""
        key = self.key_gen.gdpr_data_export(export_id)
        
        async def get_operation():
            if self.redis.client:
                export_json = await self.redis.client.get(key)
                if export_json:
                    try:
                        return json.loads(export_json)
                    except json.JSONDecodeError:
                        pass
                return None
            return None
        
        return await self.redis.safe_redis_operation(get_operation, fallback_value=None)


# Global Redis manager instance
redis_manager = RedisManager()
indicator_cache = IndicatorCache(redis_manager)
gdpr_cache = GDPRCache(redis_manager)


# Decorator for caching function results
def cache_result(
    ttl: int = 3600,
    key_prefix: str = "api",
    fallback_on_error: bool = True
):
    """Decorator for caching function results with Redis fallback"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract Redis client from kwargs if available
            redis_client = kwargs.get('redis')
            if not redis_client:
                redis_client = redis_manager.client
            
            if not redis_client:
                # If no Redis client, execute function normally
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{hashlib.md5(str(args).encode() + str(sorted(kwargs.items())).encode()).hexdigest()}"
            
            try:
                # Try to get from cache
                cached_result = await redis_client.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for {cache_key}")
                    try:
                        return json.loads(cached_result)
                    except json.JSONDecodeError:
                        return cached_result
                
            except Exception as e:
                if fallback_on_error:
                    logger.warning(f"Cache read error for {cache_key}: {e}")
                else:
                    raise
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            if result is not None:
                try:
                    # Cache the result
                    serialized_result = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                    await redis_client.setex(cache_key, ttl, serialized_result)
                    logger.debug(f"Cached result for {cache_key}")
                except Exception as e:
                    if fallback_on_error:
                        logger.warning(f"Cache write error for {cache_key}: {e}")
                    else:
                        raise
            
            return result
        return wrapper
    return decorator


