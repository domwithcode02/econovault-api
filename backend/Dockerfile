# syntax=docker/dockerfile:1

# Multi-stage build for production optimization
FROM python:3.11-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    postgresql-dev \
    python3-dev \
    linux-headers

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-alpine

# Create non-root user for security
RUN addgroup -g 1001 -S econovault && \
    adduser -S econovault -u 1001 -G econovault \
    --home /app \
    --shell /sbin/nologin \
    --disabled-password

# Install runtime dependencies only
RUN apk add --no-cache \
    libpq \
    libffi \
    curl \
    && rm -rf /var/cache/apk/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create logs directory with proper permissions
RUN mkdir -p /app/logs && chown -R econovault:econovault /app

# Copy application code with proper ownership (excluding .dockerignore patterns)
COPY --chown=econovault:econovault . .

# Make entrypoint script executable
USER root
RUN chmod +x /app/entrypoint.sh

# Switch to non-root user
USER econovault

# Security: Set minimal environment variables (sensitive ones must be provided at runtime)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    ENVIRONMENT=production \
    DEBUG=false \
    LOG_LEVEL=INFO \
    API_TITLE="EconoVault API" \
    API_VERSION=1.0.0 \
    API_DOCS_URL=/docs \
    API_REDOC_URL=/redoc \
    CORS_ALLOW_CREDENTIALS=true \
    CORS_ALLOW_METHODS='["GET", "POST", "PUT", "DELETE", "OPTIONS"]' \
    CORS_ALLOW_HEADERS='["*"]' \
    LOG_FORMAT=json \
    LOG_FILE=/app/logs/app.log \
    LOG_MAX_BYTES=10485760 \
    LOG_BACKUP_COUNT=5 \
    ENABLE_METRICS=true \
    METRICS_PORT=9090 \
    HEALTH_CHECK_PATH=/health \
    CACHE_TTL_SECONDS=300 \
    CACHE_PREFIX=econovault:prod \
    RUN_MIGRATIONS=true \
    WAIT_FOR_DB=true \
    ENABLE_REAL_DATA=true \
    ENABLE_STREAMING=true \
    ENABLE_GDPR_COMPLIANCE=true \
    ENABLE_AUDIT_LOGGING=true \
    HOT_RELOAD=false \
    DEBUG_SQL=false \
    PROFILING_ENABLED=false

# Expose port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use entrypoint script for security checks and initialization
ENTRYPOINT ["/app/entrypoint.sh"]

# Run with uvicorn for production (single worker for stability)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]