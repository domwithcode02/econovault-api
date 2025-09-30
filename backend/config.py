"""
Environment Configuration Management for EconoVault API

This module provides centralized configuration management for different environments
(development, staging, production) with proper validation and security.
"""

import os
import json
from typing import Dict, Any, Optional, List
from pydantic import Field, validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class BaseConfig(BaseSettings):
    """Base configuration class with common settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra environment variables
    )
    
    # Application Settings
    app_name: str = "EconoVault API"
    app_version: str = "1.0.0"
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    # Security Settings - Loaded from environment variables
    secret_key: str = Field(default="", description="Secret key for JWT signing - loaded from environment")
    master_encryption_key: str = Field(default="", description="Master encryption key - loaded from environment")
    
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
    cors_origins: str = Field(
        default="https://econovault.com,https://api.econovault.com,https://econovault-api-2.onrender.com"
    )
    
    # External API Keys (Optional - can be None for development)
    bls_api_key: Optional[str] = Field(default=None, description="Bureau of Labor Statistics API key")
    bea_api_key: Optional[str] = Field(default=None, description="Bureau of Economic Analysis API key") 
    fred_api_key: Optional[str] = Field(default=None, description="Federal Reserve Economic Data API key")
    
    # GDPR Compliance Settings
    gdpr_enabled: bool = Field(default=True, description="Enable GDPR compliance features")
    data_retention_days: int = Field(default=2555, description="Number of days to retain user data (7 years)")
    consent_expiry_days: int = Field(default=365, description="Number of days before consent expires")
    
    # Alerting Configuration
    alerting_enabled: bool = Field(default=False, description="Enable alerting system")
    slack_token: Optional[str] = Field(default=None, description="Slack bot token for alerts")
    slack_channel: str = Field(default="#alerts", description="Slack channel for alerts")
    pagerduty_token: Optional[str] = Field(default=None, description="PagerDuty API token")
    pagerduty_routing_key: Optional[str] = Field(default=None, description="PagerDuty routing key")
    alert_email_enabled: bool = Field(default=False, description="Enable email alerts")
    alert_email_from: str = Field(default="noreply@econovault.com", description="From email address for alerts")
    alert_email_to: str = Field(default="admin@econovault.com", description="To email address for alerts")
    alert_smtp_host: str = Field(default="smtp.gmail.com", description="SMTP host for email alerts")
    alert_smtp_port: int = Field(default=587, description="SMTP port for email alerts")
    alert_smtp_username: Optional[str] = Field(default=None, description="SMTP username")
    alert_smtp_password: Optional[str] = Field(default=None, description="SMTP password")
    alert_smtp_use_tls: bool = Field(default=True, description="Use TLS for SMTP connections")
    
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
    
    @field_validator("secret_key", "master_encryption_key")
    def validate_security_keys(cls, v, info):
        """Validate that security keys are not using default/insecure values"""
        if not v or len(v) < 32:
            raise ValueError(f"{info.field_name} must be at least 32 characters long")
        if "default" in v.lower() or "change-in-production" in v.lower():
            raise ValueError(f"{info.field_name} cannot use default or placeholder values")
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or JSON array"""
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:  # Handle empty or whitespace-only string
                logger.warning(f"CORS_ORIGINS environment variable is empty, using default '*'")
                return "*"
            
            # Try to parse as JSON first (for JSON array format)
            try:
                # Only try JSON parsing if it looks like JSON (starts with [ or {)
                if stripped.startswith('[') or stripped.startswith('{'):
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        # Convert JSON array back to comma-separated string
                        result = ",".join(parsed)
                        logger.info(f"Parsed CORS_ORIGINS JSON array: {stripped} -> {result}")
                        return result
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse CORS_ORIGINS as JSON: {stripped}, error: {e}")
                pass  # Not valid JSON, continue with comma parsing
            
            # Return as-is (comma-separated string)
            logger.info(f"Using CORS_ORIGINS as comma-separated string: {stripped}")
            return stripped
        return v
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list"""
        if not self.cors_origins or not self.cors_origins.strip():
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


class DevelopmentConfig(BaseConfig):
    """Development environment configuration"""
    
    debug: bool = True
    log_level: str = "DEBUG"
    rate_limit_per_minute: int = 1000  # Relaxed limits for development
    rate_limit_per_hour: int = 10000
    rate_limit_per_day: int = 100000
    
    # Development database (SQLite for local dev)
    database_url: str = Field(default="sqlite:///./dev.db")
    
    # Development Redis (disabled by default for local dev)
    redis_url: str = Field(default="redis://localhost:6379/0")
    
    # Relaxed security for development
    secure_headers: bool = Field(default=False)
    cors_origins: str = Field(default="*")
    
    # Enable detailed error messages
    show_error_details: bool = Field(default=True)
    
    # Override Redis settings for development
    @validator("redis_url", pre=True)
    def set_redis_url(cls, v):
        """Set Redis URL for development"""
        return v or "redis://localhost:6379/0"


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
    cors_origins: str = Field(
        default="https://econovault.com,https://api.econovault.com,https://econovault-api-2.onrender.com,*"
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
    cors_origins: str = Field(
        default="https://econovault.com,https://api.econovault.com,https://econovault-api-2.onrender.com"
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


class RenderConfig(BaseConfig):
    """Render-specific configuration that detects Render environment variables"""
    
    debug: bool = Field(default=False)
    
    # Database will be configured via Render's environment variables
    database_url: str = Field(default="")
    
    # Redis will be configured via Render's environment variables or disabled
    redis_url: str = Field(default="redis://localhost:6379/0")
    
    # Security settings for Render
    secure_headers: bool = Field(default=True)
    cors_origins: str = Field(default="*")
    
    # Override database URL from Render environment if available
    @validator("database_url", pre=True)
    def set_database_url(cls, v):
        """Use Render's DATABASE_URL if available, otherwise use provided value"""
        render_db_url = os.getenv("DATABASE_URL")
        if render_db_url:
            return render_db_url
        return v or "sqlite:///./dev.db"
    
    # Override Redis URL from Render environment if available
    @validator("redis_url", pre=True)
    def set_redis_url(cls, v):
        """Use Render's REDIS_URL if available, otherwise use provided value"""
        render_redis_url = os.getenv("REDIS_URL")
        if render_redis_url:
            return render_redis_url
        return v or "redis://localhost:6379/0"
    
    # Auto-detect environment based on Render variables
    @validator("environment", pre=True)
    def detect_environment(cls, v):
        """Detect if we're running on Render and set appropriate environment"""
        if is_render_environment():
            # Check if it's explicitly set to staging
            if os.getenv("RENDER_SERVICE_NAME", "").lower().find("staging") != -1:
                return "staging"
            return "production"
        return v or "development"


class ConfigManager:
    """Configuration manager with environment-specific settings"""
    
    def __init__(self):
        self._config_cache = {}
        self._current_config = None
    
    def get_config(self, environment: Optional[str] = None) -> BaseConfig:
        """Get configuration for specified environment"""
        if environment is None:
            environment = get_environment_for_render()
        
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
            "testing": TestingConfig,
            "render": RenderConfig
        }
        
        # If we're on Render but not explicitly set to production/staging, use RenderConfig
        if environment == "production" and is_render_environment():
            # Check if this is actually a Render deployment that needs specific handling
            if not os.getenv("DATABASE_URL"):  # If no DATABASE_URL is explicitly set
                return RenderConfig
        
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
            "cors_origins": config.cors_origins_list,
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


def is_render_environment() -> bool:
    """Check if we're running on Render platform"""
    return (
        os.getenv("RENDER") == "true" or 
        os.getenv("RENDER_SERVICE_URL") is not None or
        os.getenv("RENDER_EXTERNAL_URL") is not None
    )


def get_environment_for_render() -> str:
    """Get appropriate environment for Render deployment"""
    # Check if specific environment is set
    env = os.getenv("ENVIRONMENT", "").lower()
    
    # Map short environment names to full names
    env_mapping = {
        "prod": "production",
        "dev": "development",
        "test": "testing",
        "stage": "staging"
    }
    
    # Return mapped environment or original if valid
    if env in env_mapping:
        return env_mapping[env]
    elif env in ["production", "staging", "development", "testing"]:
        return env
    
    # If no environment specified but we're on Render, default to production
    if is_render_environment():
        return "production"
    
    # Otherwise, default to development
    return "development"


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