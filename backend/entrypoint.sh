#!/bin/sh
set -e

# EconoVault API Entrypoint Script
# Provides security checks and initialization for production deployment

echo "Starting EconoVault API container..."

# Security: Check if running as root (should not happen)
if [ "$(id -u)" -eq 0 ]; then
    echo "ERROR: Container is running as root. This is a security violation."
    exit 1
fi

# Security: Validate required environment variables
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL is not set"
    exit 1
fi

if [ -z "$SECRET_KEY" ]; then
    echo "ERROR: SECRET_KEY is not set"
    exit 1
fi

# Security: Set umask for file creation
umask 077

# Create logs directory if it doesn't exist
mkdir -p /app/logs

# Set proper permissions on log files
if [ -f /app/logs/app.log ]; then
    chmod 600 /app/logs/app.log
fi

# Wait for database to be ready (if waiting is enabled)
if [ "${WAIT_FOR_DB:-false}" = "true" ]; then
    echo "Waiting for database to be ready..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if python -c "
import urllib.parse
import psycopg2
try:
    result = urllib.parse.urlparse('$DATABASE_URL')
    conn = psycopg2.connect(
        host=result.hostname,
        port=result.port or 5432,
        database=result.path[1:],
        user=result.username,
        password=result.password
    )
    conn.close()
    print('Database connection successful')
    exit(0)
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
" 2>/dev/null; then
            echo "Database is ready!"
            break
        fi
        echo "Waiting for database... ($timeout seconds remaining)"
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -eq 0 ]; then
        echo "ERROR: Database connection timeout"
        exit 1
    fi
fi

# Run database migrations if enabled
if [ "${RUN_MIGRATIONS:-false}" = "true" ]; then
    echo "Running database migrations..."
    python -c "
import asyncio
import sys
sys.path.insert(0, '/app')
try:
    from database import init_db
    asyncio.run(init_db())
    print('Database migrations completed successfully')
except Exception as e:
    print(f'Migration failed: {e}')
    sys.exit(1)
"
fi

echo "Security checks completed successfully"
echo "Starting EconoVault API service..."

# Start the application
exec "$@"