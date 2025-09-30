# Test file to check syntax
class DeletionType(str, Enum):
    SOFT = "soft"
    HARD = "hard"
    ANONYMIZE = "anonymize"

class DeletionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    COMPLETED = "completed"
    REJECTED = "rejected"

class DeletionRequest(BaseModel):
    user_id: str
    deletion_type: DeletionType
    reason: Optional[str] = None
    verification_token: str

class DeletionResponse(BaseModel):
    request_id: str
    status: DeletionStatus
    estimated_completion: datetime
    legal_retention_notice: Optional[str] = None

class ConsentRequest(BaseModel):
    consent_type: ConsentType
    status: ConsentStatus
    consent_version: str = "1.0"
    additional_data: Optional[Dict[str, Any]] = None

class DataExportRequest(BaseModel):
    user_id: str
    data_categories: Optional[List[str]] = None
    format: str = "json"
    verification_token: str

# GDPR Deletion Endpoint
@router.delete("/users/{user_id}/data", response_model=DeletionResponse)
async def delete_user_data(
    user_id: str,
    deletion_request: DeletionRequest,
    current_user: Dict = Depends(require_permission(Permission.MANAGE_GDPR_REQUESTS)),
    db: Session = Depends(get_db)
):
    """GDPR Article 17 - Right to erasure implementation (requires GDPR management permission)"""
    
    # Verify the deletion request token
    if not verify_deletion_token(deletion_request.verification_token, user_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired deletion verification token"
        )