#!/usr/bin/env python3
"""
Test script to verify security validation in config.py
"""

import os
import sys
import traceback

# Add backend to path
sys.path.insert(0, 'backend')

def test_security_validation():
    """Test that security validation works correctly"""
    
    print("Testing security validation...")
    
    # Test 1: Missing secret_key should fail
    print("\n1. Testing missing secret_key...")
    try:
        os.environ.pop('SECRET_KEY', None)
        os.environ.pop('MASTER_ENCRYPTION_KEY', None)
        
        from config import get_config
        config = get_config()
        print("❌ FAILED: Should have raised validation error for missing secret_key")
        return False
    except Exception as e:
        print(f"✅ PASSED: Correctly rejected missing secret_key: {e}")
    
    # Test 2: Weak/short secret_key should fail
    print("\n2. Testing weak secret_key (too short)...")
    try:
        os.environ['SECRET_KEY'] = 'short'
        os.environ['MASTER_ENCRYPTION_KEY'] = 'short'
        
        from config import get_config
        config = get_config()
        print("❌ FAILED: Should have raised validation error for short secret_key")
        return False
    except Exception as e:
        print(f"✅ PASSED: Correctly rejected short secret_key: {e}")
    
    # Test 3: Default placeholder should fail
    print("\n3. Testing default placeholder secret_key...")
    try:
        os.environ['SECRET_KEY'] = 'default-secret-key-change-in-production'
        os.environ['MASTER_ENCRYPTION_KEY'] = 'default-encryption-key-change-in-production'
        
        from config import get_config
        config = get_config()
        print("❌ FAILED: Should have raised validation error for default placeholder")
        return False
    except Exception as e:
        print(f"✅ PASSED: Correctly rejected default placeholder: {e}")
    
    # Test 4: Valid strong keys should pass
    print("\n4. Testing valid strong secret_key...")
    try:
        os.environ['SECRET_KEY'] = 'a' * 32  # 32 character strong key
        os.environ['MASTER_ENCRYPTION_KEY'] = 'b' * 32  # 32 character strong key
        
        from config import get_config
        config = get_config()
        print(f"✅ PASSED: Accepted valid strong secret_key (length: {len(config.secret_key)})")
        print(f"✅ PASSED: Accepted valid strong master_encryption_key (length: {len(config.master_encryption_key)})")
    except Exception as e:
        print(f"❌ FAILED: Should have accepted valid strong keys: {e}")
        return False
    
    # Test 5: Test with actual production-like keys
    print("\n5. Testing with production-like keys...")
    try:
        os.environ['SECRET_KEY'] = 'EconoVaultProdSecretKey2024SecureRandomString1234567890ABCDEF'
        os.environ['MASTER_ENCRYPTION_KEY'] = 'EconoVaultMasterEncryptionKey2024SecureRandomABCDEF1234567890'
        
        from config import get_config
        config = get_config()
        print(f"✅ PASSED: Accepted production-like keys")
        print(f"   Secret key length: {len(config.secret_key)}")
        print(f"   Encryption key length: {len(config.master_encryption_key)}")
    except Exception as e:
        print(f"❌ FAILED: Should have accepted production-like keys: {e}")
        return False
    
    print("\n🎉 All security validation tests passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_security_validation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)