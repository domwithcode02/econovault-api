#!/usr/bin/env python3
"""
Quick start script for EconoVault API development
Handles environment setup and port conflicts automatically
"""

import os
import sys
import subprocess
import secrets
import string

def generate_secure_key(length=64):
    """Generate a cryptographically secure random key"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-="
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def setup_minimal_env():
    """Create minimal .env file for quick testing"""
    if os.path.exists('.env'):
        print(".env file already exists")
        return
    
    print("Creating minimal development environment...")
    
    # Generate secure keys
    secret_key = generate_secure_key(64)
    master_encryption_key = generate_secure_key(64)
    
    # Create minimal .env file
    env_content = f"""# Minimal EconoVault API Configuration
SECRET_KEY={secret_key}
MASTER_ENCRYPTION_KEY={master_encryption_key}
DATABASE_URL=sqlite:///./dev.db
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Disable external APIs for now (optional)
BLS_API_KEY=
BEA_API_KEY=
FRED_API_KEY=

# Redis configuration (handled by RedisManager, not config.py)
# Redis will be auto-detected as unavailable in development
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("Created minimal .env file with secure keys")

def find_available_port(start_port=8000, max_port=8010):
    """Find an available port"""
    import socket
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    return None

def main():
    """Main function to start development server"""
    print("EconoVault API Quick Start")
    print("=" * 40)
    
    # Setup environment
    setup_minimal_env()
    
    # Find available port
    port = find_available_port()
    if not port:
        print("No available ports found between 8000-8010")
        print("Please free up a port or check what's running")
        sys.exit(1)
    
    print(f"Found available port: {port}")
    print(f"API will be available at: http://127.0.0.1:{port}")
    print(f"Documentation: http://127.0.0.1:{port}/docs")
    print("\nPress Ctrl+C to stop\n")
    
    # Set environment and run
    env = os.environ.copy()
    env['PORT'] = str(port)
    
    try:
        # Run with uvicorn directly
        subprocess.run([
            sys.executable, '-m', 'uvicorn',
            'main:app',
            '--host', '127.0.0.1',
            '--port', str(port),
            '--reload'
        ], env=env, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped")
    except subprocess.CalledProcessError as e:
        print(f"\nServer failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()