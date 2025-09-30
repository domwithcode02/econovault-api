-- Migration script to add rotation_policy and expiry_notification_sent columns to api_keys table
-- Run this SQL command on your SQL Server database

BEGIN TRY
    DECLARE @column_exists BIT;
    DECLARE @message NVARCHAR(MAX);
    
    -- Check if rotation_policy column exists
    SELECT @column_exists = CASE 
        WHEN EXISTS (
            SELECT 1 FROM sys.columns 
            WHERE object_id = OBJECT_ID('dbo.api_keys') 
            AND name = 'rotation_policy'
        ) THEN 1 ELSE 0 END;
    
    IF @column_exists = 0
    BEGIN
        -- Add the rotation_policy column with DEFAULT value
        ALTER TABLE dbo.api_keys
        ADD rotation_policy VARCHAR(50) 
        CONSTRAINT DF_api_keys_rotation_policy 
        DEFAULT 'manual' WITH VALUES;
        
        SET @message = 'SUCCESS: rotation_policy column successfully added to api_keys table';
        PRINT @message;
    END
    ELSE
    BEGIN
        SET @message = 'INFO: rotation_policy column already exists in api_keys table';
        PRINT @message;
    END;
    
    -- Check if expiry_notification_sent column exists
    SELECT @column_exists = CASE 
        WHEN EXISTS (
            SELECT 1 FROM sys.columns 
            WHERE object_id = OBJECT_ID('dbo.api_keys') 
            AND name = 'expiry_notification_sent'
        ) THEN 1 ELSE 0 END;
    
    IF @column_exists = 0
    BEGIN
        -- Add the expiry_notification_sent column with DEFAULT value
        ALTER TABLE dbo.api_keys
        ADD expiry_notification_sent BIT 
        CONSTRAINT DF_api_keys_expiry_notification_sent 
        DEFAULT 0 WITH VALUES;
        
        SET @message = 'SUCCESS: expiry_notification_sent column successfully added to api_keys table';
        PRINT @message;
    END
    ELSE
    BEGIN
        SET @message = 'INFO: expiry_notification_sent column already exists in api_keys table';
        PRINT @message;
    END;
END TRY
BEGIN CATCH
    -- Error handling
    SELECT 
        ERROR_NUMBER() AS ErrorNumber,
        ERROR_SEVERITY() AS ErrorSeverity,
        ERROR_STATE() AS ErrorState,
        ERROR_PROCEDURE() AS ErrorProcedure,
        ERROR_LINE() AS ErrorLine,
        ERROR_MESSAGE() AS ErrorMessage;
    
    SET @message = 'ERROR: Failed to add columns to api_keys table: ' + ERROR_MESSAGE();
    PRINT @message;
    
    -- Re-throw the error to ensure the transaction fails
    THROW;
END CATCH;
GO