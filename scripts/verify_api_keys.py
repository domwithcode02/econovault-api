#!/usr/bin/env python3
"""
EconoVault API Key Verification Script

This script tests the external API keys to ensure they are working correctly
before deployment to production.
"""

import os
import sys
import requests
import json
from datetime import datetime
from typing import Dict, Any, Tuple

def test_bls_api_key(api_key: str) -> Tuple[bool, str]:
    """Test BLS API key with a simple request"""
    try:
        url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        payload = {
            "seriesid": ["CUUR0000SA0"],  # CPI series
            "startyear": "2023",
            "endyear": "2023",
            "registrationkey": api_key
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "REQUEST_SUCCEEDED":
                return True, "BLS API key is working"
            else:
                return False, f"BLS API error: {data.get('message', 'Unknown error')}"
        else:
            return False, f"BLS API HTTP error: {response.status_code}"
            
    except Exception as e:
        return False, f"BLS API test failed: {str(e)}"

def test_bea_api_key(api_key: str) -> Tuple[bool, str]:
    """Test BEA API key with a simple request"""
    try:
        url = "https://apps.bea.gov/api/data/"
        params = {
            "UserID": api_key,
            "method": "GETDATASETLIST",
            "datasetname": "NIPA",
            "ResultFormat": "JSON"
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if "BEAAPI" in data and "Results" in data["BEAAPI"]:
                return True, "BEA API key is working"
            else:
                return False, f"BEA API error: {data.get('BEAAPI', {}).get('Error', 'Unknown error')}"
        else:
            return False, f"BEA API HTTP error: {response.status_code}"
            
    except Exception as e:
        return False, f"BEA API test failed: {str(e)}"

def test_fred_api_key(api_key: str) -> Tuple[bool, str]:
    """Test FRED API key with a simple request"""
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "CPIAUCSL",  # CPI series
            "api_key": api_key,
            "file_type": "json",
            "limit": 1
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if "observations" in data:
                return True, "FRED API key is working"
            else:
                return False, f"FRED API error: {data.get('error_message', 'Unknown error')}"
        else:
            return False, f"FRED API HTTP error: {response.status_code}"
            
    except Exception as e:
        return False, f"FRED API test failed: {str(e)}"

def load_environment_variables():
    """Load environment variables from .env file if it exists"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Environment variables loaded from .env file")
    except ImportError:
        print("⚠ python-dotenv not installed, using existing environment variables")

def main():
    """Main verification function"""
    print("🔑 EconoVault API Key Verification")
    print("=" * 50)
    
    # Load environment variables
    load_environment_variables()
    
    # Get API keys from environment
    api_keys = {
        "BLS": os.getenv("BLS_API_KEY"),
        "BEA": os.getenv("BEA_API_KEY"), 
        "FRED": os.getenv("FRED_API_KEY")
    }
    
    print("\n📋 Checking API Key Configuration:")
    print("-" * 30)
    
    all_keys_present = True
    for source, key in api_keys.items():
        if key and key not in ["SET_YOUR_" + source + "_API_KEY_HERE", ""]:
            print(f"✓ {source} API key: Present")
        else:
            print(f"❌ {source} API key: Missing or placeholder")
            all_keys_present = False
    
    if not all_keys_present:
        print("\n❌ Some API keys are missing or set to placeholder values.")
        print("Please set the following environment variables:")
        for source in api_keys.keys():
            print(f"  - {source}_API_KEY")
        print("\nSee DEPLOYMENT_SETUP.md for instructions on obtaining API keys.")
        return False
    
    print("\n🧪 Testing API Keys:")
    print("-" * 30)
    
    results = {}
    all_tests_passed = True
    
    # Test each API key
    test_functions = {
        "BLS": test_bls_api_key,
        "BEA": test_bea_api_key,
        "FRED": test_fred_api_key
    }
    
    for source, test_func in test_functions.items():
        print(f"\nTesting {source} API key...")
        success, message = test_func(api_keys[source])
        
        if success:
            print(f"✅ {message}")
            results[source] = "PASS"
        else:
            print(f"❌ {message}")
            results[source] = "FAIL"
            all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    for source, result in results.items():
        status = "✅ PASS" if result == "PASS" else "❌ FAIL"
        print(f"{source:5} API: {status}")
    
    if all_tests_passed:
        print("\n🎉 All API keys are working correctly!")
        print("✅ Ready for deployment to Render")
        return True
    else:
        print("\n⚠️  Some API keys are not working.")
        print("❌ Please check your API keys and try again.")
        print("💡 See DEPLOYMENT_SETUP.md for troubleshooting tips.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)