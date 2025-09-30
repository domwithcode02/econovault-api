#!/usr/bin/env python3
"""
Manual migration script to add rotation_policy column to api_keys table
Run this script to update the production database schema
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

def migrate_add_rotation_policy():
    """Add rotation_policy column to api_keys table"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable must be set")
        sys.exit(1)
    
    try:
        # Create engine
        engine = create_engine(database_url)
        
        # Check if column already exists
        with engine.connect() as conn:
            # PostgreSQL specific query to check if column exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'api_keys' 
                AND column_name = 'rotation_policy'
            """))
            
            if result.fetchone():
                print("✓ rotation_policy column already exists in api_keys table")
                return
            
            # Add the column
            print("Adding rotation_policy column to api_keys table...")
            conn.execute(text("""
                ALTER TABLE api_keys 
                ADD COLUMN rotation_policy VARCHAR(50) DEFAULT 'manual'
            """))
            
            # Verify the column was added
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'api_keys' 
                AND column_name = 'rotation_policy'
            """))
            
            if result.fetchone():
                print("✓ rotation_policy column successfully added to api_keys table")
            else:
                print("✗ Failed to add rotation_policy column")
                sys.exit(1)
                
    except SQLAlchemyError as e:
        print(f"✗ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    migrate_add_rotation_policy()