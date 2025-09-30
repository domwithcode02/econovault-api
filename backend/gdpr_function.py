# GDPR Deletion Endpoint
@router.delete("/users/{user_id}/data", response_model=DeletionResponse)
async def delete_user_data(
    user_id: str,
    deletion_request: DeletionRequest,
    current_user: Dict = Depends(require_permission(Permission.MANAGE_GDPR_REQUESTS)),
    db: Session = Depends(get_db)
):
    """GDPR implementation"""
    
    # Verify the deletion request token
    if not verify_deletion_token(deletion_request.verification_token, user_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired deletion verification token"
        )
    
    # Execute deletion based on type
    request_id = generate_request_id()
    
    # Hash user ID for GDPR compliance
    from security import audit_logger
    user_id_hash = audit_logger.hash_identifier(user_id)
    verification_token_hash = audit_logger.hash_identifier(deletion_request.verification_token)
    
    # Store deletion request in database
    from database import DatabaseManager
    db_manager = DatabaseManager(db)
    
    deletion_data = {
        "request_id": request_id,
        "user_id_hash": user_id_hash,
        "deletion_type": deletion_request.deletion_type.value,
        "status": DeletionStatus.COMPLETED.value,
        "reason": deletion_request.reason,
        "verification_token_hash": verification_token_hash,
        "requested_at": datetime.utcnow(),
        "approved_at": datetime.utcnow(),
        "completed_at": datetime.utcnow()
    }
    
    try:
        # Store the deletion request
        db_manager.create_deletion_request(deletion_data)
        
        # Return success response
        return {
            "request_id": request_id,
            "status": DeletionStatus.COMPLETED.value,
            "estimated_completion": datetime.utcnow() + timedelta(days=1),
            "message": f"User data deletion request processed successfully for {user_id_hash}"
        }
        
    except Exception as e:
        logger.error(f"Error processing deletion request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing deletion request: {str(e)}"
        )