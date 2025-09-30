#!/usr/bin/env python3
"""
Port checker and development runner for EconoVault API
Checks if port 8000 is available and suggests alternatives
"""

import socket
import sys
import os
import subprocess

def check_port_available(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def find_available_port(start_port=8000, max_port=8010):
    """Find an available port starting from start_port"""
    for port in range(start_port, max_port + 1):
        if check_port_available(port):
            return port
    return None

def run_development_server():
    """Run the development server with proper configuration"""
    
    print("🔍 Checking development environment...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Running setup...")
        subprocess.run([sys.executable, 'setup_dev.py'], check=True)
    
    # Check port availability
    target_port = 8000
    if not check_port_available(target_port):
        print(f"⚠️  Port {target_port} is already in use.")
        available_port = find_available_port(target_port + 1, target_port + 10)
        if available_port:
            print(f"✅ Found available port: {available_port}")
            target_port = available_port
        else:
            print("❌ No available ports found between 8000-8010")
            print("💡 Please free up a port or modify the port range in this script")
            sys.exit(1)
    
    # Set environment variables
    env = os.environ.copy()
    env['PORT'] = str(target_port)
    
    print(f"🚀 Starting EconoVault API development server on port {target_port}...")
    print(f"📚 API Documentation: http://localhost:{target_port}/docs")
    print(f"📊 Metrics: http://localhost:{target_port}/metrics")
    print(f"❤️  Health Check: http://localhost:{target_port}/health")
    print("\n🛑 Press Ctrl+C to stop the server\n")
    
    try:
        # Run the application with the selected port
        subprocess.run([
            sys.executable, 'main.py',
            '--host', '127.0.0.1',
            '--port', str(target_port),
            '--reload'  # Enable auto-reload for development
        ], env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_development_server()