# EconoVault API Docker Deployment Guide

## Required Environment Variables

The following environment variables **MUST** be provided at runtime (not hardcoded in the Docker image):

### Critical Security Variables (Required)
```bash
# Required for JWT token signing and encryption
SECRET_KEY=your-secret-key-here-minimum-32-characters
MASTER_ENCRYPTION_KEY=your-encryption-key-here-minimum-32-characters

# Required for external API access
BLS_API_KEY=your-bls-api-key
BEA_API_KEY=your-bea-api-key  
FRED_API_KEY=your-fred-api-key
```

### Database Configuration (Required)
```bash
# Database connection string
DATABASE_URL=postgresql://user:password@host:port/database

# Database pool settings (optional, defaults will be used if not set)
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DATABASE_POOL_TIMEOUT=30
```

### Redis Configuration (Optional)
```bash
# Redis connection (optional, will use fallback if not set)
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your-redis-password  # optional
REDIS_DB=0  # optional
```

### Rate Limiting (Optional)
```bash
# Rate limiting settings (optional, sensible defaults will be used)
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=3600
RATE_LIMIT_PER_DAY=86400
```

### CORS Configuration (Optional)
```bash
# CORS origins (optional, defaults to production domains)
CORS_ORIGINS=["https://econovault.com","https://api.econovault.com"]
CORS_ALLOW_CREDENTIALS=true
```

### Alerting Configuration (Optional)
```bash
# Slack alerting (optional)
SLACK_TOKEN=xoxb-your-slack-token
SLACK_CHANNEL=#alerts

# PagerDuty alerting (optional)
PAGERDUTY_TOKEN=your-pagerduty-token
PAGERDUTY_ROUTING_KEY=your-routing-key

# Email alerting (optional)
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_FROM=alerts@econovault.com
ALERT_EMAIL_TO=["admin@econovault.com"]
ALERT_SMTP_HOST=smtp.gmail.com
ALERT_SMTP_PORT=587
ALERT_SMTP_USERNAME=your-email@gmail.com
ALERT_SMTP_PASSWORD=your-app-password
ALERT_SMTP_USE_TLS=true
```

### GDPR Configuration (Optional)
```bash
# GDPR settings (optional, defaults will be used)
GDPR_ENABLED=true
DATA_RETENTION_DAYS=2555
CONSENT_EXPIRY_DAYS=365
```

### File Storage (Optional)
```bash
# File storage settings (optional)
DATA_STORAGE_PATH=/app/data
MAX_FILE_SIZE_MB=100
```

## Docker Run Commands

### Basic Development Run
```bash
docker run -d \
  -p 8000:8000 \
  -e SECRET_KEY=dev-secret-key-min-32-chars \
  -e MASTER_ENCRYPTION_KEY=dev-encryption-key-min-32-chars \
  -e DATABASE_URL=sqlite:///app/data/dev.db \
  -e BLS_API_KEY=your-bls-key \
  -e BEA_API_KEY=your-bea-key \
  -e FRED_API_KEY=your-fred-key \
  -e ENVIRONMENT=development \
  --name econovault-dev \
  econovault-api:latest
```

### Production Run with PostgreSQL
```bash
docker run -d \
  -p 8000:8000 \
  -e SECRET_KEY=your-production-secret-key-min-32-chars \
  -e MASTER_ENCRYPTION_KEY=your-production-encryption-key-min-32-chars \
  -e DATABASE_URL=postgresql://user:pass@postgres:5432/econovault \
  -e REDIS_URL=redis://redis:6379/0 \
  -e BLS_API_KEY=your-bls-key \
  -e BEA_API_KEY=your-bea-key \
  -e FRED_API_KEY=your-fred-key \
  -e ENVIRONMENT=production \
  -e CORS_ORIGINS='["https://econovault.com","https://api.econovault.com"]' \
  -e SLACK_TOKEN=xoxb-your-slack-token \
  -e SLACK_CHANNEL=#production-alerts \
  --name econovault-prod \
  econovault-api:latest
```

### Docker Compose Example
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: econovault
      POSTGRES_USER: econovault
      POSTGRES_PASSWORD: your-postgres-password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  econovault:
    image: econovault-api:latest
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://econovault:your-postgres-password@postgres:5432/econovault
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=your-production-secret-key-min-32-chars
      - MASTER_ENCRYPTION_KEY=your-production-encryption-key-min-32-chars
      - BLS_API_KEY=your-bls-api-key
      - BEA_API_KEY=your-bea-api-key
      - FRED_API_KEY=your-fred-api-key
      - CORS_ORIGINS=["https://econovault.com","https://api.econovault.com"]
      - SLACK_TOKEN=xoxb-your-slack-token
      - SLACK_CHANNEL=#alerts
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs

volumes:
  postgres_data:
```

## Security Notes

1. **Never hardcode sensitive values** in the Dockerfile
2. **Use Docker secrets** or environment variable management systems in production
3. **Rotate API keys** regularly using the built-in rotation system
4. **Use strong, unique secrets** for SECRET_KEY and MASTER_ENCRYPTION_KEY
5. **Enable SSL/TLS** in production with proper certificates
6. **Configure firewall rules** to restrict access to necessary ports only

## Health Checks

The container includes health checks that monitor:
- Application responsiveness
- Database connectivity
- Redis connectivity (if configured)
- API endpoint availability

Health check endpoint: `http://localhost:8000/health`

## Monitoring

The application exposes Prometheus metrics at:
`http://localhost:9090/metrics`

Enable alerting by configuring Slack, PagerDuty, or email settings in the environment variables above.