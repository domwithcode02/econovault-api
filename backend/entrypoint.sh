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
import sys
import urllib.parse
try:
    result = urllib.parse.urlparse('$DATABASE_URL')
    # Simple connection test without psycopg2 dependency
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex((result.hostname, result.port or 5432))
    sock.close()
    if result == 0:
        print('Database host is reachable')
        exit(0)
    else:
        print('Database host unreachable')
        exit(1)
except Exception as e:
    print(f'Connection test failed: {e}')
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
import sys
import os
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')
try:
    from database import create_tables
    create_tables()
    print('Database tables created successfully')
except Exception as e:
    print(f'Migration failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
fi

echo "Security checks completed successfully"
echo "Starting EconoVault API service..."

# Start the application with error handling
echo "Starting application with command: $@"
echo "Current working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Environment check:"
echo "  DATABASE_URL: ${DATABASE_URL:0:20}..."
echo "  SECRET_KEY: ${SECRET_KEY:0:10}..."
echo "  ENVIRONMENT: ${ENVIRONMENT:-unknown}"

# Test if main module can be imported
echo "Testing application import..."
if python -c "import sys; sys.path.insert(0, '/app'); from main import app; print('Import successful')"; then
    echo "Application import test passed"
else
    echo "ERROR: Application import test failed"
    exit 1
fi

exec "$@"