#!/usr/bin/env python3
"""
Test script to verify circuit breaker functionality in the BLS client.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.bls_client import BLSClient, BLSAPIException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_circuit_breaker():
    """Test circuit breaker functionality"""
    
    # Create BLS client with circuit breaker
    client = BLSClient(api_key="invalid_key")  # Invalid key to trigger failures
    
    logger.info("Testing circuit breaker with invalid API key...")
    
    # Test multiple failed requests to trigger circuit breaker
    for i in range(7):
        try:
            logger.info(f"Attempt {i+1}: Fetching CPI data...")
            data = client.get_series_data("CUUR0000SA0", start_year=2023, end_year=2023)
            logger.info(f"Attempt {i+1}: Success - Got {len(data)} data points")
        except BLSAPIException as e:
            logger.warning(f"Attempt {i+1}: Failed - {e}")
        except Exception as e:
            logger.error(f"Attempt {i+1}: Unexpected error - {e}")
    
    logger.info("Circuit breaker test completed.")
    logger.info("Check the logs above to see if the circuit breaker opened after 5 failures.")

if __name__ == "__main__":
    test_circuit_breaker()