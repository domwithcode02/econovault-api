#!/usr/bin/env python3
"""
Connection pool analysis and reset script for EconoVault API
Checks connection pool settings and provides solutions for connection caching issues
"""

import os
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection details
DATABASE_URL = "postgresql://econovault_db_user:ZAVQLwpbCEr9pEKnYMugHOvsQ1764yHe@dpg-d3bmeur7mgec739r7af0-a.oregon-postgres.render.com:5432/econovault_db"

def analyze_connection_pool():
    """Analyze current connection pool settings and issues"""
    logger.info("Analyzing connection pool settings...")
    
    # Set environment variable first
    os.environ['DATABASE_URL'] = DATABASE_URL
    
    # Test with current engine settings
    from database import engine
    
    logger.info(f"Current engine pool settings:")
    logger.info(f"  Pool size: {engine.pool.size()}")
    logger.info(f"  Max overflow: {getattr(engine.pool, 'max_overflow', 'N/A')}")
    logger.info(f"  Pool timeout: {getattr(engine.pool, 'timeout', 'N/A')}")
    logger.info(f"  Pool recycle: {getattr(engine.pool, 'recycle', 'N/A')}")
    logger.info(f"  Pool pre-ping: {getattr(engine.pool, 'pre_ping', 'N/A')}")
    
    # Test connection
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("✓ Current engine connection test successful")
            
            # Check PostgreSQL settings
            result = conn.execute(text("SHOW max_connections;"))
            max_connections = result.scalar()
            logger.info(f"PostgreSQL max_connections: {max_connections}")
            
            result = conn.execute(text("SELECT count(*) FROM pg_stat_activity;"))
            active_connections = result.scalar()
            logger.info(f"Active connections: {active_connections}")
            
    except Exception as e:
        logger.error(f"✗ Current engine connection test failed: {e}")

def test_fresh_connection():
    """Test with a completely fresh connection"""
    logger.info("Testing with fresh connection settings...")
    
    # Create a new engine with fresh settings
    fresh_engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=300,  # 5 minutes instead of 1 hour
        echo=True  # Enable SQL logging for debugging
    )
    
    try:
        with fresh_engine.connect() as conn:
            # Test the problematic query
            result = conn.execute(text("""
                SELECT key_id, rotation_policy 
                FROM api_keys 
                WHERE rotation_policy != 'manual' 
                LIMIT 5;
            """))
            rows = result.fetchall()
            logger.info(f"✓ Fresh connection query successful, found {len(rows)} rows")
            
            for row in rows:
                logger.info(f"  Key: {row.key_id}, Policy: {row.rotation_policy}")
                
    except Exception as e:
        logger.error(f"✗ Fresh connection query failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        fresh_engine.dispose()

def check_table_schema():
    """Double-check the table schema"""
    logger.info("Double-checking table schema...")
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Check exact column names (case sensitive)
        cursor.execute("""
            SELECT column_name, data_type, column_default, is_nullable
            FROM information_schema.columns 
            WHERE table_name = 'api_keys' 
            ORDER BY ordinal_position;
        """)
        
        columns = cursor.fetchall()
        logger.info("API Keys table schema:")
        for col in columns:
            logger.info(f"  {col[0]}: {col[1]} (nullable: {col[3]}, default: {col[2]})")
        
        # Check if there are any case sensitivity issues
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'api_keys' 
            AND lower(column_name) = 'rotation_policy';
        """)
        
        result = cursor.fetchone()
        if result:
            logger.info(f"✓ rotation_policy column found (case-insensitive): {result[0]}")
        else:
            logger.error("✗ rotation_policy column not found (case-insensitive)")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"✗ Schema check failed: {e}")

def main():
    """Main analysis function"""
    logger.info("=== EconoVault Connection Pool Analysis ===")
    
    analyze_connection_pool()
    check_table_schema()
    test_fresh_connection()
    
    logger.info("\n=== Analysis Complete ===")
    logger.info("If the fresh connection works but current engine fails, the issue is connection pool caching.")
    logger.info("Solution: Restart the application to reset the connection pool.")

if __name__ == "__main__":
    main()