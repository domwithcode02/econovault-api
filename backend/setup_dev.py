#!/usr/bin/env python3
"""
Development environment setup script for EconoVault API
Creates proper .env file with development configuration
"""

import os
import secrets
import string

def generate_secure_key(length=64):
    """Generate a cryptographically secure random key"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-="
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def setup_development_env():
    """Create development environment configuration"""
    
    print("🚀 Setting up EconoVault API development environment...")
    
    # Generate secure keys
    secret_key = generate_secure_key(64)
    master_encryption_key = generate_secure_key(64)
    
    # Create .env file content
    env_content = f"""# EconoVault API Development Environment Configuration
# Generated on: {import datetime; datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Security Settings (Generated - DO NOT SHARE)
SECRET_KEY={secret_key}
MASTER_ENCRYPTION_KEY={master_encryption_key}

# Database Settings (SQLite for development)
DATABASE_URL=sqlite:///./dev.db

# Redis Settings (disabled for development)
REDIS_URL=redis://localhost:6379/0
REDIS_ENABLED=false

# External API Keys (get your own keys from respective services)
BLS_API_KEY=your-bls-api-key-here
BEA_API_KEY=your-bea-api-key-here
FRED_API_KEY=your-fred-api-key-here

# Rate Limiting (relaxed for development)
RATE_LIMIT_PER_MINUTE=1000
RATE_LIMIT_PER_HOUR=10000
RATE_LIMIT_PER_DAY=100000

# CORS Settings (permissive for development)
CORS_ORIGINS=["http://localhost:3000","http://localhost:8080","http://127.0.0.1:3000"]

# Monitoring (enabled for development)
ENABLE_METRICS=true
METRICS_PORT=9090

# GDPR Settings
GDPR_ENABLED=true

# Development Features
HOT_RELOAD=true
DEBUG_SQL=false
SHOW_ERROR_DETAILS=true
"""

    # Write .env file
    env_path = ".env"
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Created {env_path} with development configuration")
    print(f"✅ Generated secure 64-character keys")
    print(f"✅ Configured SQLite database for development")
    print(f"✅ Set relaxed rate limits for development")
    print(f"✅ Enabled detailed error messages")
    
    print("\n📋 Next steps:")
    print("1. Get API keys from:")
    print("   - BLS: https://www.bls.gov/developers/")
    print("   - BEA: https://apps.bea.gov/API/signup/")
    print("   - FRED: https://research.stlouisfed.org/fred2/")
    print("2. Update the API keys in .env file")
    print("3. Run: python main.py")
    print("\n🌐 The API will be available at: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")
    print("📊 Metrics: http://localhost:8000/metrics")

if __name__ == "__main__":
    setup_development_env()