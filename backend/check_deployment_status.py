#!/usr/bin/env python3
"""
Multi-instance deployment analysis for EconoVault API
Checks for potential issues with multiple Render instances
"""

import os
import requests
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your production API URL
API_BASE_URL = "https://econovault-api-2.onrender.com"

def check_health_endpoint():
    """Check the health endpoint status"""
    logger.info("Checking health endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=30)
        logger.info(f"Health check status: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"Health data: {json.dumps(health_data, indent=2)}")
            
            # Check for system health issues
            if 'system_health' in health_data:
                system_health = health_data['system_health']
                logger.info(f"System health: {system_health.get('status', 'unknown')}")
                if system_health.get('unhealthy_services'):
                    logger.warning(f"Unhealthy services: {system_health.get('unhealthy_services')}")
            
            return health_data
        else:
            logger.error(f"Health check failed with status {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Health check request failed: {e}")
        return None

def check_api_endpoints():
    """Test basic API endpoints"""
    logger.info("Testing basic API endpoints...")
    
    endpoints_to_test = [
        "/",
        "/docs",
        "/health",
        "/metrics"
    ]
    
    results = {}
    for endpoint in endpoints_to_test:
        try:
            logger.info(f"Testing {endpoint}...")
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30)
            results[endpoint] = {
                "status": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
            logger.info(f"  {endpoint}: {response.status_code} ({response.elapsed.total_seconds():.2f}s)")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"  {endpoint}: Failed - {e}")
            results[endpoint] = {
                "status": "error",
                "error": str(e)
            }
    
    return results

def check_deployment_status():
    """Check deployment and instance status"""
    logger.info("Checking deployment status...")
    
    # Check if we can detect multiple instances by making multiple requests
    instance_responses = []
    
    for i in range(3):
        try:
            logger.info(f"Request {i+1} to detect instance differences...")
            response = requests.get(f"{API_BASE_URL}/health", timeout=30)
            
            # Extract any instance-specific headers or data
            instance_info = {
                "request_num": i+1,
                "status": response.status_code,
                "headers": dict(response.headers),
                "timestamp": datetime.now().isoformat()
            }
            
            if response.status_code == 200:
                try:
                    health_data = response.json()
                    instance_info["health_data"] = health_data
                except:
                    pass
            
            instance_responses.append(instance_info)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request {i+1} failed: {e}")
            instance_responses.append({
                "request_num": i+1,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    # Analyze responses for inconsistencies
    logger.info("Analyzing instance responses...")
    
    # Check for different response headers (might indicate different instances)
    server_headers = [resp.get("headers", {}).get("Server", "unknown") for resp in instance_responses if resp.get("status") == 200]
    if len(set(server_headers)) > 1:
        logger.warning(f"Detected different server headers: {set(server_headers)}")
    
    # Check for different response times (might indicate different instances)
    response_times = [resp.get("headers", {}).get("X-Response-Time", "unknown") for resp in instance_responses if resp.get("status") == 200]
    if len(set(response_times)) > 1:
        logger.info(f"Detected different response times: {set(response_times)}")
    
    return instance_responses

def simulate_rotation_task():
    """Simulate the API key rotation task that's failing"""
    logger.info("Simulating API key rotation task...")
    
    # This would normally be done internally, but we can check if there are
    # any endpoints that might trigger this
    
    try:
        # Check if there's an admin endpoint for rotation
        response = requests.get(f"{API_BASE_URL}/admin/health", timeout=30)
        logger.info(f"Admin health check: {response.status_code}")
        
    except requests.exceptions.RequestException as e:
        logger.info(f"Admin endpoint not available or failed: {e}")

def main():
    """Main analysis function"""
    logger.info("=== EconoVault Multi-Instance Deployment Analysis ===")
    logger.info(f"API Base URL: {API_BASE_URL}")
    logger.info(f"Analysis Time: {datetime.now().isoformat()}")
    
    # Run all checks
    health_data = check_health_endpoint()
    endpoint_results = check_api_endpoints()
    instance_responses = check_deployment_status()
    simulate_rotation_task()
    
    logger.info("\n=== Analysis Summary ===")
    
    if health_data:
        logger.info("✓ Health endpoint is responding")
        if health_data.get('status') == 'healthy':
            logger.info("✓ System reports healthy status")
        else:
            logger.warning("⚠ System reports non-healthy status")
    else:
        logger.error("✗ Health endpoint is not responding")
    
    # Check endpoint results
    failed_endpoints = [ep for ep, result in endpoint_results.items() if result.get('status') != 200]
    if failed_endpoints:
        logger.warning(f"⚠ Failed endpoints: {failed_endpoints}")
    else:
        logger.info("✓ All tested endpoints are responding")
    
    # Check for instance inconsistencies
    if len(instance_responses) > 1:
        logger.info(f"✓ Made {len(instance_responses)} requests to detect instance differences")
        
        # Look for patterns that might indicate multiple instances
        successful_responses = [r for r in instance_responses if r.get('status') == 200]
        if successful_responses:
            logger.info(f"✓ {len(successful_responses)} requests succeeded")
            
            # Check for different response signatures
            response_signatures = []
            for resp in successful_responses:
                sig = {
                    'server': resp.get('headers', {}).get('Server', 'unknown'),
                    'date': resp.get('headers', {}).get('Date', 'unknown'),
                    'content_type': resp.get('headers', {}).get('Content-Type', 'unknown')
                }
                response_signatures.append(str(sig))
            
            unique_signatures = set(response_signatures)
            if len(unique_signatures) > 1:
                logger.warning(f"⚠ Detected {len(unique_signatures)} different response signatures (possible multiple instances)")
            else:
                logger.info("✓ Consistent response signatures (likely single instance or load-balanced)")
    
    logger.info("\n=== Recommendations ===")
    logger.info("1. If health checks show unhealthy status, restart the service")
    logger.info("2. If multiple instances detected, ensure all instances are synchronized")
    logger.info("3. Check Render dashboard for deployment status and instance health")
    logger.info("4. Monitor logs for rotation task errors after restart")

if __name__ == "__main__":
    main()