#!/usr/bin/env python3
"""
EconoVault API Sandbox Testing Suite

Comprehensive testing suite for the sandbox environment with mock data,
test endpoints, and validation scripts.
"""

import requests
import json
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections.abc import Callable
import concurrent.futures
import argparse
import sys
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResult(Enum):
    """Test result status"""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class TestCase:
    """Test case definition"""
    name: str
    description: str
    endpoint: str
    method: str = "GET"
    params: Optional[Dict] = None
    data: Optional[Dict] = None
    expected_status: int = 200
    expected_fields: Optional[List[str]] = None
    validation_func: Optional[Callable[[Dict], bool]] = None


class SandboxTester:
    """Comprehensive sandbox testing suite"""
    
    def __init__(self, base_url: str, api_key: str = "sandbox-test-key"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'User-Agent': 'EconoVault-Sandbox-Tester/1.0.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        self.test_results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
        logger.info(f"Sandbox tester initialized for {base_url}")
    
    def run_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Run a single test case"""
        logger.info(f"Running test: {test_case.name}")
        
        result = {
            "name": test_case.name,
            "description": test_case.description,
            "status": TestResult.FAIL,
            "response_time": 0,
            "error": None,
            "response_data": None
        }
        
        try:
            start_time = time.time()
            
            # Make request
            url = f"{self.base_url}/{test_case.endpoint.lstrip('/')}"
            response = self.session.request(
                method=test_case.method,
                url=url,
                params=test_case.params,
                json=test_case.data,
                timeout=30
            )
            
            response_time = time.time() - start_time
            result["response_time"] = response_time
            
            # Check status code
            if response.status_code != test_case.expected_status:
                result["error"] = f"Expected status {test_case.expected_status}, got {response.status_code}"
                logger.error(f"Test {test_case.name} failed: {result['error']}")
                return result
            
            # Parse response
            try:
                response_data = response.json()
                result["response_data"] = response_data
            except json.JSONDecodeError as e:
                result["error"] = f"Failed to parse JSON response: {e}"
                logger.error(f"Test {test_case.name} failed: {result['error']}")
                return result
            
            # Validate expected fields
            if test_case.expected_fields:
                missing_fields = []
                for field in test_case.expected_fields:
                    if field not in response_data:
                        missing_fields.append(field)
                
                if missing_fields:
                    result["error"] = f"Missing expected fields: {missing_fields}"
                    logger.error(f"Test {test_case.name} failed: {result['error']}")
                    return result
            
            # Run custom validation
            if test_case.validation_func:
                try:
                    validation_result = test_case.validation_func(response_data)
                    if not validation_result:
                        result["error"] = "Custom validation failed"
                        logger.error(f"Test {test_case.name} failed: {result['error']}")
                        return result
                except Exception as e:
                    result["error"] = f"Validation function error: {e}"
                    logger.error(f"Test {test_case.name} failed: {result['error']}")
                    return result
            
            # Test passed
            result["status"] = TestResult.PASS
            logger.info(f"Test {test_case.name} passed in {response_time:.3f}s")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Test {test_case.name} failed with exception: {e}")
        
        return result
    
    def run_test_suite(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Run a complete test suite"""
        logger.info(f"Running test suite with {len(test_cases)} test cases")
        
        results = []
        passed = 0
        failed = 0
        skipped = 0
        
        for test_case in test_cases:
            try:
                result = self.run_test(test_case)
                results.append(result)
                
                if result["status"] == TestResult.PASS:
                    passed += 1
                elif result["status"] == TestResult.FAIL:
                    failed += 1
                elif result["status"] == TestResult.SKIP:
                    skipped += 1
                    
            except Exception as e:
                logger.error(f"Test suite error for {test_case.name}: {e}")
                failed += 1
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        summary = {
            "total_tests": len(test_cases),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "success_rate": (passed / len(test_cases)) * 100 if test_cases else 0,
            "total_time": total_time,
            "results": results
        }
        
        logger.info(f"Test suite completed: {passed}/{len(test_cases)} passed ({summary['success_rate']:.1f}%)")
        return summary
    
    def generate_test_report(self, summary: Dict[str, Any]) -> str:
        """Generate a detailed test report"""
        report = []
        report.append("=" * 80)
        report.append("ECONOVAULT SANDBOX TEST REPORT")
        report.append("=" * 80)
        report.append(f"Test Date: {datetime.now().isoformat()}")
        report.append(f"API URL: {self.base_url}")
        report.append(f"Total Tests: {summary['total_tests']}")
        report.append(f"Passed: {summary['passed']}")
        report.append(f"Failed: {summary['failed']}")
        report.append(f"Skipped: {summary['skipped']}")
        report.append(f"Success Rate: {summary['success_rate']:.1f}%")
        report.append(f"Total Time: {summary['total_time']:.2f}s")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 80)
        
        for result in summary["results"]:
            status_symbol = "✓" if result["status"] == TestResult.PASS else "✗"
            report.append(f"{status_symbol} {result['name']}")
            report.append(f"  Description: {result['description']}")
            report.append(f"  Status: {result['status'].value}")
            report.append(f"  Response Time: {result['response_time']:.3f}s")
            
            if result["error"]:
                report.append(f"  Error: {result['error']}")
            
            report.append("")
        
        return "\n".join(report)
    
    def save_report(self, summary: Dict[str, Any], filename: str | None = None):
        """Save test report to file"""
        if filename is None:
            filename = f"sandbox_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save JSON report
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save text report
        text_filename = filename.replace('.json', '.txt')
        with open(text_filename, 'w') as f:
            f.write(self.generate_test_report(summary))
        
        logger.info(f"Test reports saved: {filename}, {text_filename}")


def create_test_suite() -> List[TestCase]:
    """Create comprehensive test suite for sandbox"""
    
    def validate_indicator_data(data: Dict) -> bool:
        """Validate indicator data structure"""
        required_fields = ["series_id", "title", "source", "indicator_type"]
        return all(field in data for field in required_fields)
    
    def validate_time_series_data(data: Dict) -> bool:
        """Validate time series data structure"""
        required_fields = ["series_id", "title", "source", "data", "count"]
        if not all(field in data for field in required_fields):
            return False
        
        # Check data points
        if not isinstance(data["data"], list):
            return False
        
        for point in data["data"]:
            if not all(field in point for field in ["date", "value", "period", "year"]):
                return False
        
        return True
    
    def validate_health_response(data: Dict) -> bool:
        """Validate health check response"""
        required_fields = ["status", "timestamp", "service", "version"]
        return all(field in data for field in required_fields)
    
    test_cases = [
        # Health and basic endpoints
        TestCase(
            name="Health Check",
            description="Test health check endpoint",
            endpoint="health",
            expected_fields=["status", "timestamp", "service", "version"],
            validation_func=validate_health_response
        ),
        
        TestCase(
            name="Root Endpoint",
            description="Test root API endpoint",
            endpoint="",
            expected_fields=["message", "docs", "health"]
        ),
        
        TestCase(
            name="API Documentation",
            description="Test OpenAPI documentation",
            endpoint="docs",
            expected_status=200
        ),
        
        TestCase(
            name="OpenAPI Schema",
            description="Test OpenAPI schema endpoint",
            endpoint="openapi.json",
            expected_fields=["openapi", "info", "paths"]
        ),
        
        # Indicators endpoints
        TestCase(
            name="Get All Indicators",
            description="Test getting all economic indicators",
            endpoint="v1/indicators",
            expected_fields=None,  # Should return array
            validation_func=lambda data: isinstance(data, list) and len(data) > 0
        ),
        
        TestCase(
            name="Get BLS Indicators",
            description="Test filtering indicators by source",
            endpoint="v1/indicators",
            params={"source": "BLS", "limit": 5},
            validation_func=lambda data: isinstance(data, list) and len(data) <= 5
        ),
        
        TestCase(
            name="Get CPI Indicators",
            description="Test filtering indicators by type",
            endpoint="v1/indicators",
            params={"indicator_type": "CPI", "limit": 3},
            validation_func=lambda data: isinstance(data, list) and len(data) <= 3
        ),
        
TestCase(
            name="Get Consumer Price Index",
            description="Test getting CPI indicator",
            endpoint="v1/indicators/consumer-price-index",
            expected_fields=["series_id", "title", "source", "indicator_type"],
            validation_func=validate_indicator_data
        ),
        
        TestCase(
            name="Get Non-existent Indicator",
            description="Test error handling for non-existent indicator",
            endpoint="v1/indicators/INVALID_SERIES",
            expected_status=404
        ),
        
# Data endpoints
        TestCase(
            name="Get CPI Data",
            description="Test getting CPI time series data",
            endpoint="v1/indicators/consumer-price-index/data",
            expected_fields=["series_id", "title", "source", "data", "count"],
            validation_func=validate_time_series_data
        ),
        
        TestCase(
            name="Get Unemployment Data with Limit",
            description="Test unemployment data endpoint with limit parameter",
            endpoint="v1/indicators/unemployment-rate/data",
            params={"limit": 5},
            validation_func=lambda data: isinstance(data["data"], list) and len(data["data"]) <= 5
        ),
        
        TestCase(
            name="Get Nonfarm Payrolls Data with Date Range",
            description="Test nonfarm payrolls data endpoint with date range",
            endpoint="v1/indicators/nonfarm-payrolls/data",
            params={"start_date": "2023-01-01", "end_date": "2023-12-31"},
            validation_func=validate_time_series_data
        ),
        
        # Authentication endpoints (if available)
        TestCase(
            name="Get API Health",
            description="Test API health status",
            endpoint="health",
            expected_fields=["status", "timestamp", "service", "version"]
        ),
    ]
    
    return test_cases


def run_load_test(base_url: str, api_key: str, duration_seconds: int = 60, concurrent_users: int = 5):
    """Run load testing against sandbox"""
    
    def make_request():
        """Make a single request"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{base_url}/v1/indicators",
                headers={'X-API-Key': api_key},
                params={'limit': 10},
                timeout=10
            )
            response_time = time.time() - start_time
            
            return {
                'status_code': response.status_code,
                'response_time': response_time,
                'success': response.status_code == 200
            }
        except Exception as e:
            return {
                'status_code': 0,
                'response_time': 0,
                'success': False,
                'error': str(e)
            }
    
    logger.info(f"Starting load test: {duration_seconds}s with {concurrent_users} concurrent users")
    
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        # Submit initial batch of requests
        futures = []
        for _ in range(concurrent_users * 10):  # 10 requests per user initially
            futures.append(executor.submit(make_request))
        
        # Keep submitting requests during the test duration
        while time.time() - start_time < duration_seconds:
            # Process completed futures
            completed = []
            for future in concurrent.futures.as_completed(futures, timeout=1):
                try:
                    result = future.result()
                    results.append(result)
                    completed.append(future)
                except Exception as e:
                    logger.error(f"Request failed: {e}")
            
            # Remove completed futures
            for future in completed:
                futures.remove(future)
            
            # Submit new requests to maintain load
            while len(futures) < concurrent_users * 10:
                futures.append(executor.submit(make_request))
            
            time.sleep(0.1)  # Small delay to prevent overwhelming
    
    # Process remaining futures
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            logger.error(f"Request failed: {e}")
    
    # Calculate statistics
    if results:
        successful_requests = sum(1 for r in results if r['success'])
        total_requests = len(results)
        avg_response_time = sum(r['response_time'] for r in results if r['response_time'] > 0) / len([r for r in results if r['response_time'] > 0])
        min_response_time = min(r['response_time'] for r in results if r['response_time'] > 0)
        max_response_time = max(r['response_time'] for r in results if r['response_time'] > 0)
        
        load_test_summary = {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': total_requests - successful_requests,
            'success_rate': (successful_requests / total_requests) * 100,
            'avg_response_time': avg_response_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time,
            'requests_per_second': total_requests / duration_seconds,
            'test_duration': duration_seconds,
            'concurrent_users': concurrent_users
        }
        
        logger.info(f"Load test completed: {successful_requests}/{total_requests} successful ({load_test_summary['success_rate']:.1f}%)")
        logger.info(f"Average response time: {avg_response_time:.3f}s")
        
        return load_test_summary
    else:
        logger.error("No results collected from load test")
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="EconoVault Sandbox Testing Suite")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for sandbox API")
    parser.add_argument("--api-key", default="sandbox-test-key", help="API key for testing")
    parser.add_argument("--load-test", action="store_true", help="Run load testing")
    parser.add_argument("--load-duration", type=int, default=60, help="Load test duration in seconds")
    parser.add_argument("--concurrent-users", type=int, default=5, help="Number of concurrent users for load test")
    parser.add_argument("--output", help="Output file for test results")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = SandboxTester(args.url, args.api_key)
    
    try:
        # Create and run test suite
        test_cases = create_test_suite()
        summary = tester.run_test_suite(test_cases)
        
        # Run load test if requested
        load_test_summary = None
        if args.load_test:
            load_test_summary = run_load_test(
                args.url, 
                args.api_key, 
                args.load_duration, 
                args.concurrent_users
            )
            
            # Add load test results to summary
            summary["load_test"] = load_test_summary
        
        # Print report
        print("\n" + tester.generate_test_report(summary))
        
        # Save results
        output_file = args.output or f"sandbox_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tester.save_report(summary, output_file)
        
        # Exit with appropriate code
        if summary['failed'] > 0:
            logger.error(f"Test suite failed with {summary['failed']} failures")
            sys.exit(1)
        else:
            logger.info("All tests passed!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()