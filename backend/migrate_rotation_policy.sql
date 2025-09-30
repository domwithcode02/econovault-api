-- Migration script to add rotation_policy column to api_keys table
-- Run this SQL command on your SQL Server database

BEGIN TRY
    DECLARE @column_exists BIT;
    DECLARE @message NVARCHAR(MAX);
    
    -- Check if column exists
    SELECT @column_exists = CASE 
        WHEN EXISTS (
            SELECT 1 FROM sys.columns 
            WHERE object_id = OBJECT_ID('dbo.api_keys') 
            AND name = 'rotation_policy'
        ) THEN 1 ELSE 0 END;
    
    IF @column_exists = 0
    BEGIN
        -- Add the column with DEFAULT value
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
    
    SET @message = 'ERROR: Failed to add rotation_policy column: ' + ERROR_MESSAGE();
    PRINT @message;
    
    -- Re-throw the error to ensure the transaction fails
    THROW;
END CATCH;
GO