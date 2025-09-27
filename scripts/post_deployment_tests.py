#!/usr/bin/env python3
"""
Post-deployment verification script for EconoVault API
Tests critical endpoints after production deployment
"""

import requests
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any

class PostDeploymentTester:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.results = []
        
    def log_result(self, test_name: str, success: bool, message: str = "", response_time: float = 0.0):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        status = "✓" if success else "✗"
        print(f"{status} {test_name}: {message} ({response_time:.2f}s)")
        
    def test_health_endpoint(self) -> bool:
        """Test health check endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_result("Health Check", True, "Service is healthy", response_time)
                    return True
                else:
                    self.log_result("Health Check", False, f"Unexpected status: {data}", response_time)
                    return False
            else:
                self.log_result("Health Check", False, f"HTTP {response.status_code}", response_time)
                return False
                
        except Exception as e:
            self.log_result("Health Check", False, str(e), 0.0)
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test root endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/", timeout=self.timeout)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "EconoVault API" in data.get("message", ""):
                    self.log_result("Root Endpoint", True, "Service info available", response_time)
                    return True
                else:
                    self.log_result("Root Endpoint", False, "Unexpected response format", response_time)
                    return False
            else:
                self.log_result("Root Endpoint", False, f"HTTP {response.status_code}", response_time)
                return False
                
        except Exception as e:
            self.log_result("Root Endpoint", False, str(e), 0.0)
            return False
    
    def test_api_documentation(self) -> bool:
        """Test API documentation endpoints"""
        try:
            # Test Swagger UI
            start_time = time.time()
            response = requests.get(f"{self.base_url}/docs", timeout=self.timeout)
            response_time = time.time() - start_time
            
            if response.status_code == 200 and "swagger" in response.text.lower():
                self.log_result("API Documentation", True, "Swagger UI accessible", response_time)
                return True
            else:
                self.log_result("API Documentation", False, "Swagger UI not accessible", response_time)
                return False
                
        except Exception as e:
            self.log_result("API Documentation", False, str(e), 0.0)
            return False
    
    def test_openapi_schema(self) -> bool:
        """Test OpenAPI schema endpoint"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/openapi.json", timeout=self.timeout)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                schema = response.json()
                if "openapi" in schema and "paths" in schema:
                    path_count = len(schema["paths"])
                    self.log_result("OpenAPI Schema", True, f"Schema loaded with {path_count} paths", response_time)
                    return True
                else:
                    self.log_result("OpenAPI Schema", False, "Invalid schema format", response_time)
                    return False
            else:
                self.log_result("OpenAPI Schema", False, f"HTTP {response.status_code}", response_time)
                return False
                
        except Exception as e:
            self.log_result("OpenAPI Schema", False, str(e), 0.0)
            return False
    
    def test_response_times(self) -> bool:
        """Test response times for critical endpoints"""
        endpoints = [
            ("/health", "Health Check"),
            ("/", "Root Endpoint"),
            ("/docs", "API Documentation"),
            ("/openapi.json", "OpenAPI Schema")
        ]
        
        all_passed = True
        for endpoint, name in endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=self.timeout)
                response_time = time.time() - start_time
                
                # Check if response time is under 100ms for cached/static content
                if endpoint in ["/health", "/", "/docs", "/openapi.json"]:
                    if response_time <= 0.1:  # 100ms threshold
                        self.log_result(f"{name} Response Time", True, f"{response_time*1000:.1f}ms", response_time)
                    else:
                        self.log_result(f"{name} Response Time", False, f"{response_time*1000:.1f}ms (target: <100ms)", response_time)
                        all_passed = False
                else:
                    self.log_result(f"{name} Response Time", True, f"{response_time*1000:.1f}ms", response_time)
                    
            except Exception as e:
                self.log_result(f"{name} Response Time", False, str(e), 0.0)
                all_passed = False
        
        return all_passed
    
    def test_security_headers(self) -> bool:
        """Test security headers are present"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            
            security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options", 
                "X-XSS-Protection",
                "Strict-Transport-Security"
            ]
            
            missing_headers = []
            for header in security_headers:
                if header not in response.headers:
                    missing_headers.append(header)
            
            if not missing_headers:
                self.log_result("Security Headers", True, "All security headers present")
                return True
            else:
                self.log_result("Security Headers", False, f"Missing: {', '.join(missing_headers)}")
                return False
                
        except Exception as e:
            self.log_result("Security Headers", False, str(e))
            return False
    
    def run_all_tests(self) -> bool:
        """Run all post-deployment tests"""
        print(f"Running post-deployment tests for {self.base_url}")
        print("=" * 60)
        
        tests = [
            self.test_health_endpoint,
            self.test_root_endpoint,
            self.test_api_documentation,
            self.test_openapi_schema,
            self.test_response_times,
            self.test_security_headers
        ]
        
        all_passed = True
        for test in tests:
            if not test():
                all_passed = False
        
        print("=" * 60)
        
        # Summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"\nTest Summary:")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        
        if all_passed:
            print("\n✓ All tests passed! Deployment is healthy.")
        else:
            print(f"\n✗ {failed_tests} tests failed. Check the results above.")
        
        return all_passed
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate detailed test report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["success"])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "summary": {
                "total": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "results": self.results
        }

def main():
    """Main function"""
    import os
    
    # Get URL from environment variable
    base_url = os.getenv("PRODUCTION_URL", "https://econovault-api.onrender.com")
    
    print(f"EconoVault API Post-Deployment Testing")
    print(f"Testing URL: {base_url}")
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    tester = PostDeploymentTester(base_url)
    
    try:
        success = tester.run_all_tests()
        
        # Generate and save report
        report = tester.generate_report()
        
        # Save report to file
        with open("deployment-test-report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: deployment-test-report.json")
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()