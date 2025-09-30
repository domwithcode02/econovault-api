-- PostgreSQL migration script to add rotation_policy and expiry_notification_sent columns to api_keys table
-- This script uses PostgreSQL-specific syntax that works in Render

-- Add rotation_policy column if it doesn't exist
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS rotation_policy VARCHAR(50) DEFAULT 'manual';

-- Add expiry_notification_sent column if it doesn't exist
ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS expiry_notification_sent BOOLEAN DEFAULT FALSE;