"""
Environment Configuration Management for EconoVault API

This module provides centralized configuration management for different environments
(development, staging, production) with proper validation and security.
"""

import os
import json
from typing import Dict, Any, Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class BaseConfig(BaseSettings):
    """Base configuration class with common settings"""
    
    # Application Settings
    app_name: str = "EconoVault API"
    app_version: str = "1.0.0"
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    # Security Settings
    secret_key: str = Field(default="default-secret-key-change-in-production")
    master_encryption_key: str = Field(default="default-encryption-key-change-in-production")
    
    # Database Settings
    database_url: str = Field(default="sqlite:///./default.db")
    database_pool_size: int = Field(default=10)
    database_max_overflow: int = Field(default=20)
    database_pool_timeout: int = Field(default=30)
    
    # Redis Settings (for caching and sessions)
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_password: Optional[str] = Field(default=None)
    redis_db: int = Field(default=0)
    
    # API Settings
    api_title: str = "EconoVault API"
    api_version: str = "v1"
    api_prefix: str = "/v1"
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60)
    rate_limit_per_hour: int = Field(default=3600)
    rate_limit_per_day: int = Field(default=86400)
    
    # Logging Settings
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file: Optional[str] = Field(default=None)
    
    # Monitoring Settings
    enable_metrics: bool = Field(default=True)
    metrics_retention_days: int = Field(default=30)
    
    # Security Headers
    secure_headers: bool = Field(default=True)
    cors_origins: List[str] = Field(default=["*"])
    
    # External API Settings
    bls_api_key: Optional[str] = Field(default=None)
    bea_api_key: Optional[str] = Field(default=None)
    fred_api_key: Optional[str] = Field(default=None)
    
    # GDPR Settings
    gdpr_enabled: bool = Field(default=True)
    data_retention_days: int = Field(default=2555)  # 7 years
    consent_expiry_days: int = Field(default=365)
    
    # File Storage Settings
    data_storage_path: str = Field(default="/app/data")
    max_file_size_mb: int = Field(default=100)
    
    # Health Check Settings
    health_check_timeout: int = Field(default=30)
    
    # Webhook Settings
    webhook_timeout: int = Field(default=30)
    webhook_retry_attempts: int = Field(default=3)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting"""
        valid_environments = ["development", "staging", "production", "testing"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


class DevelopmentConfig(BaseConfig):
    """Development environment configuration"""
    
    debug: bool = True
    log_level: str = "DEBUG"
    rate_limit_per_minute: int = 1000  # Relaxed limits for development
    rate_limit_per_hour: int = 10000
    rate_limit_per_day: int = 100000
    
    # Development database (SQLite)
    database_url: str = Field(default="sqlite:///./dev.db")
    
    # Development Redis (optional)
    redis_url: str = Field(default="redis://localhost:6379/0")
    
    # Relaxed security for development
    secure_headers: bool = Field(default=False)
    cors_origins: List[str] = Field(default=["*"])
    
    # Enable detailed error messages
    show_error_details: bool = Field(default=True)


class StagingConfig(BaseConfig):
    """Staging environment configuration"""
    
    debug: bool = False
    log_level: str = "INFO"
    
    # Staging database (PostgreSQL)
    database_url: str = Field(default="postgresql://localhost:5432/econovault_staging")
    
    # Production-like settings with relaxed limits
    rate_limit_per_minute: int = Field(default=120)
    rate_limit_per_hour: int = Field(default=7200)
    rate_limit_per_day: int = Field(default=86400)
    
    # Enable security headers
    secure_headers: bool = Field(default=True)
    cors_origins: List[str] = Field(
        default=["https://staging.econovault.com", "https://staging-api.econovault.com"]
    )
    
    # Hide error details in staging
    show_error_details: bool = Field(default=False)


class ProductionConfig(BaseConfig):
    """Production environment configuration"""
    
    debug: bool = False
    log_level: str = "WARNING"
    
    # Production database (PostgreSQL with connection pooling)
    database_url: str = Field(default="postgresql://localhost:5432/econovault_production")
    database_pool_size: int = Field(default=20)
    database_max_overflow: int = Field(default=30)
    
    # Production Redis
    redis_url: str = Field(default="redis://localhost:6379/0")
    
    # Strict rate limiting for production
    rate_limit_per_minute: int = Field(default=60)
    rate_limit_per_hour: int = Field(default=3600)
    rate_limit_per_day: int = Field(default=86400)
    
    # Strict security settings
    secure_headers: bool = Field(default=True)
    cors_origins: List[str] = Field(
        default=["https://econovault.com", "https://api.econovault.com"]
    )
    
    # Production logging
    log_file: str = Field(default="/app/logs/app.log")
    
    # Enhanced monitoring
    enable_metrics: bool = Field(default=True)
    metrics_retention_days: int = Field(default=90)
    
    # Hide error details in production
    show_error_details: bool = Field(default=False)
    
    # Production-specific settings
    webhook_timeout: int = Field(default=60)
    health_check_timeout: int = Field(default=60)


class TestingConfig(BaseConfig):
    """Testing environment configuration"""
    
    debug: bool = True
    log_level: str = "DEBUG"
    
    # In-memory database for testing
    database_url: str = Field(default="sqlite:///:memory:")
    
    # Disable external API calls in tests
    bls_api_key: Optional[str] = None
    bea_api_key: Optional[str] = None
    fred_api_key: Optional[str] = None
    
    # Relaxed rate limiting for tests
    rate_limit_per_minute: int = 10000
    rate_limit_per_hour: int = 100000
    rate_limit_per_day: int = 1000000
    
    # Enable detailed error messages for debugging
    show_error_details: bool = True


class ConfigManager:
    """Configuration manager with environment-specific settings"""
    
    def __init__(self):
        self._config_cache = {}
        self._current_config = None
    
    def get_config(self, environment: Optional[str] = None) -> BaseConfig:
        """Get configuration for specified environment"""
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "development")
        
        # Return cached config if available
        if environment in self._config_cache:
            return self._config_cache[environment]
        
        # Create new config instance
        config_class = self._get_config_class(environment)
        config = config_class()
        
        # Cache the config
        self._config_cache[environment] = config
        self._current_config = config
        
        logger.info(f"Loaded {environment} configuration")
        return config
    
    def _get_config_class(self, environment: str):
        """Get configuration class for environment"""
        config_classes = {
            "development": DevelopmentConfig,
            "staging": StagingConfig,
            "production": ProductionConfig,
            "testing": TestingConfig
        }
        
        if environment not in config_classes:
            logger.warning(f"Unknown environment '{environment}', using development config")
            return DevelopmentConfig
        
        return config_classes[environment]
    
    def reload_config(self):
        """Reload configuration from environment"""
        self._config_cache.clear()
        self._current_config = None
        logger.info("Configuration reloaded")
    
    def get_current_config(self) -> BaseConfig:
        """Get current active configuration"""
        if self._current_config is None:
            self._current_config = self.get_config()
        return self._current_config
    
    def validate_secrets(self) -> Dict[str, bool]:
        """Validate that all required secrets are configured"""
        config = self.get_current_config()
        
        required_secrets = {
            "SECRET_KEY": bool(config.secret_key),
            "MASTER_ENCRYPTION_KEY": bool(config.master_encryption_key),
            "DATABASE_URL": bool(config.database_url),
        }
        
        # Optional secrets that should be validated in production
        if config.environment == "production":
            optional_secrets = {
                "BLS_API_KEY": bool(config.bls_api_key),
                "BEA_API_KEY": bool(config.bea_api_key),
                "FRED_API_KEY": bool(config.fred_api_key),
            }
            required_secrets.update(optional_secrets)
        
        return required_secrets
    
    def export_config(self, mask_secrets: bool = True) -> Dict[str, Any]:
        """Export configuration as dictionary (for debugging)"""
        config = self.get_current_config()
        config_dict = config.dict()
        
        if mask_secrets:
            # Mask sensitive information
            secret_fields = [
                "secret_key", "master_encryption_key", "database_url",
                "redis_password", "bls_api_key", "bea_api_key", "fred_api_key"
            ]
            
            for field in secret_fields:
                if field in config_dict and config_dict[field]:
                    config_dict[field] = "***MASKED***"
        
        return config_dict
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database-specific configuration"""
        config = self.get_current_config()
        
        return {
            "url": config.database_url,
            "pool_size": config.database_pool_size,
            "max_overflow": config.database_max_overflow,
            "pool_timeout": config.database_pool_timeout,
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis-specific configuration"""
        config = self.get_current_config()
        
        return {
            "url": config.redis_url,
            "password": config.redis_password,
            "db": config.redis_db,
        }
    
    def get_rate_limit_config(self) -> Dict[str, int]:
        """Get rate limiting configuration"""
        config = self.get_current_config()
        
        return {
            "per_minute": config.rate_limit_per_minute,
            "per_hour": config.rate_limit_per_hour,
            "per_day": config.rate_limit_per_day,
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        config = self.get_current_config()
        
        return {
            "level": config.log_level,
            "format": config.log_format,
            "file": config.log_file,
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        config = self.get_current_config()
        
        return {
            "secure_headers": config.secure_headers,
            "cors_origins": config.cors_origins,
            "gdpr_enabled": config.gdpr_enabled,
            "data_retention_days": config.data_retention_days,
            "consent_expiry_days": config.consent_expiry_days,
        }


# Global configuration manager instance
config_manager = ConfigManager()


# Convenience functions for accessing configuration
@lru_cache()
def get_settings() -> BaseConfig:
    """Get current application settings (cached)"""
    return config_manager.get_current_config()


def get_config() -> BaseConfig:
    """Get current configuration"""
    return config_manager.get_current_config()


def reload_config():
    """Reload configuration from environment"""
    config_manager.reload_config()
    # Clear the cached settings
    get_settings.cache_clear()


def validate_configuration() -> Dict[str, bool]:
    """Validate current configuration"""
    return config_manager.validate_secrets()


# Example usage
if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print(f"Environment: {config.environment}")
    print(f"Debug: {config.debug}")
    print(f"Log Level: {config.log_level}")
    
    # Validate secrets
    secrets_status = validate_configuration()
    print("\nSecrets validation:")
    for secret, status in secrets_status.items():
        print(f"  {secret}: {'✓' if status else '✗'}")
    
    # Export configuration (with secrets masked)
    config_export = config_manager.export_config(mask_secrets=True)
    print(f"\nConfiguration export (secrets masked):")
    print(json.dumps(config_export, indent=2))