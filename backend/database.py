from sqlalchemy.engine import create_engine
from sqlalchemy import Column, Integer, String, DateTime, Date, DECIMAL, ForeignKey, Text, Boolean, insert, Index, func
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import Generator, Optional, List
from datetime import date, datetime, timedelta
import os

# Database configuration - require DATABASE_URL to be set
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable must be set")

# Create data directory for SQLite if needed
if DATABASE_URL.startswith("sqlite:///"):
    import os
    import pathlib
    db_path = DATABASE_URL.replace("sqlite:///", "")
    db_dir = pathlib.Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Set to True for debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


class EconomicIndicator(Base):
    """Economic indicator metadata"""
    __tablename__ = "economic_indicators"
    
    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String(50), unique=True, index=True, nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    source = Column(String(20), nullable=False)
    indicator_type = Column(String(50), nullable=False)
    frequency = Column(String(10), nullable=False)
    seasonal_adjustment = Column(String(10), nullable=False)
    geography = Column(String(100))
    geography_level = Column(String(20), nullable=False)
    units = Column(String(100), nullable=False)
    start_date = Column(Date)
    end_date = Column(Date)
    last_updated = Column(DateTime, default=func.now())
    registration_key_required = Column(Boolean, default=False)
    
    # Relationship to data points
    data_points = relationship("DataPoint", back_populates="indicator", cascade="all, delete-orphan")


class DataPoint(Base):
    """Individual data points for economic indicators"""
    __tablename__ = "data_points"
    
    id = Column(Integer, primary_key=True, index=True)
    indicator_id = Column(Integer, ForeignKey("economic_indicators.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    value = Column(DECIMAL(20, 6))
    period = Column(String(10))
    period_name = Column(String(50))
    footnote_code = Column(String(10))
    footnote_text = Column(Text)
    realtime_start = Column(Date)
    realtime_end = Column(Date)
    latest = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relationship to indicator
    indicator = relationship("EconomicIndicator", back_populates="data_points")
    
    # Composite index for efficient queries
    __table_args__ = (
        # Index for time series queries with covering index for value and period
        Index('idx_datapoint_indicator_date', 'indicator_id', 'date', postgresql_include=['value', 'period']),
    )


class AuditLog(Base):
    """GDPR compliance audit log"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)
    user_id_hash = Column(String(64), index=True)
    data_subject_id_hash = Column(String(64), index=True)
    session_id = Column(String(100))
    ip_address_hash = Column(String(64))
    user_agent_hash = Column(String(64))
    resource_type = Column(String(50))
    resource_id = Column(String(100))
    action = Column(String(100), nullable=False)
    result = Column(String(20), nullable=False)
    reason = Column(Text)
    metadata_json = Column(Text)
    gdpr_basis = Column(String(50))
    data_categories_json = Column(Text)
    retention_period = Column(String(20))
    hash_chain = Column(String(64))


class ConsentRecord(Base):
    """GDPR consent management"""
    __tablename__ = "consent_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id_hash = Column(String(64), nullable=False, index=True)
    consent_type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)
    ip_address_encrypted = Column(Text)
    user_agent_hash = Column(String(64))
    consent_version = Column(String(20), nullable=False)
    expires_at = Column(DateTime)
    additional_data_json = Column(Text)


class APIKey(Base):
    """API key management"""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    user_id_hash = Column(String(64), nullable=False, index=True)
    key_hash = Column(String(200), nullable=False)
    scopes_json = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, index=True)
    status = Column(String(20), default="active", index=True)
    created_at = Column(DateTime, default=func.now())
    last_used = Column(DateTime)
    rate_limit_per_minute = Column(Integer, default=60)
    rate_limit_per_hour = Column(Integer, default=3600)
    expires_at = Column(DateTime)
    next_rotation_date = Column(DateTime)
    usage_count = Column(Integer, default=0)
    rotation_policy = Column(String(50), default="manual")
    expiry_notification_sent = Column(Boolean, default=False)


class DeletionRequest(Base):
    """GDPR Article 17 - Right to erasure (deletion requests)"""
    __tablename__ = "deletion_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id_hash = Column(String(64), nullable=False, index=True)
    deletion_type = Column(String(20), nullable=False)  # soft, hard, anonymize
    status = Column(String(20), nullable=False, index=True)  # pending, approved, completed, rejected
    reason = Column(Text)
    verification_token_hash = Column(String(64), nullable=False)
    requested_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    approved_at = Column(DateTime)
    completed_at = Column(DateTime)
    legal_retention_notice = Column(Text)
    metadata_json = Column(Text)  # Additional data for audit trail
    
    # Index for efficient querying by user and status
    __table_args__ = (
        Index('idx_deletion_user_status', 'user_id_hash', 'status'),
        Index('idx_deletion_created', 'requested_at'),
    )


class DataExportRequest(Base):
    """GDPR Article 20 - Right to data portability (export requests)"""
    __tablename__ = "data_export_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    export_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id_hash = Column(String(64), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)  # pending, processing, completed, failed, expired
    format = Column(String(20), nullable=False, default="json")  # json, csv, xml
    data_categories_json = Column(Text)  # List of data categories to export
    verification_token_hash = Column(String(64), nullable=False)
    file_path = Column(String(500))  # Path to exported file (if stored on disk)
    file_size_bytes = Column(Integer)  # Size of exported file
    requested_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    processing_started_at = Column(DateTime)
    completed_at = Column(DateTime)
    expires_at = Column(DateTime, nullable=False, index=True)
    download_count = Column(Integer, default=0)
    metadata_json = Column(Text)  # Additional metadata for audit trail
    
    # Index for efficient querying
    __table_args__ = (
        Index('idx_export_user_status', 'user_id_hash', 'status'),
        Index('idx_export_expires', 'expires_at'),
    )


# Create all tables
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


# Initialize database (async function for migrations)
async def init_db():
    """Initialize database tables"""
    create_tables()


# Dependency to get database session
def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database utilities
class DatabaseManager:
    """Database management utilities"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_indicator_by_series_id(self, series_id: str) -> Optional[EconomicIndicator]:
        """Get economic indicator by series ID"""
        return self.session.query(EconomicIndicator).filter(
            EconomicIndicator.series_id == series_id
        ).first()
    
    def create_indicator(self, indicator_data: dict) -> EconomicIndicator:
        """Create new economic indicator"""
        indicator = EconomicIndicator(**indicator_data)
        self.session.add(indicator)
        self.session.commit()
        self.session.refresh(indicator)
        return indicator
    
    def update_indicator(self, series_id: str, update_data: dict) -> Optional[EconomicIndicator]:
        """Update existing economic indicator"""
        indicator = self.get_indicator_by_series_id(series_id)
        if not indicator:
            return None
        
        # Update fields
        for key, value in update_data.items():
            if hasattr(indicator, key) and value is not None:
                setattr(indicator, key, value)
        
        # Update last_updated timestamp
        indicator.last_updated = datetime.utcnow()  # type: ignore
        
        self.session.commit()
        self.session.refresh(indicator)
        return indicator
    
    def delete_indicator(self, series_id: str) -> bool:
        """Delete economic indicator and all associated data points"""
        indicator = self.get_indicator_by_series_id(series_id)
        if not indicator:
            return False
        
        # Delete all associated data points first (cascade might handle this)
        self.session.query(DataPoint).filter(DataPoint.indicator_id == indicator.id).delete()
        
        # Delete the indicator
        self.session.delete(indicator)
        self.session.commit()
        return True
    
    def bulk_insert_data_points(self, data_points: List[dict]) -> int:
        """Bulk insert data points for better performance"""
        if not data_points:
            return 0
        
        # Use modern insert approach for better performance and type safety
        self.session.execute(insert(DataPoint), data_points)
        self.session.commit()
        return len(data_points)
    
    def get_data_points(self, series_id: str, start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[DataPoint]:
        """Get data points for a series with optional date filtering"""
        query = self.session.query(DataPoint).join(EconomicIndicator).filter(
            EconomicIndicator.series_id == series_id
        )
        
        if start_date:
            query = query.filter(DataPoint.date >= start_date)
        if end_date:
            query = query.filter(DataPoint.date <= end_date)
        
        return query.order_by(DataPoint.date).all()
    
    def get_latest_data_point(self, series_id: str) -> Optional[DataPoint]:
        """Get the latest data point for a series"""
        return self.session.query(DataPoint).join(EconomicIndicator).filter(
            EconomicIndicator.series_id == series_id
        ).order_by(DataPoint.date.desc()).first()
    
    def create_deletion_request(self, request_data: dict) -> DeletionRequest:
        """Create a new GDPR deletion request"""
        deletion_request = DeletionRequest(**request_data)
        self.session.add(deletion_request)
        self.session.commit()
        self.session.refresh(deletion_request)
        return deletion_request
    
    def get_deletion_request(self, request_id: str) -> Optional[DeletionRequest]:
        """Get deletion request by request ID"""
        return self.session.query(DeletionRequest).filter(
            DeletionRequest.request_id == request_id
        ).first()
    
    def get_user_deletion_requests(self, user_id_hash: str, status: Optional[str] = None) -> List[DeletionRequest]:
        """Get deletion requests for a user"""
        query = self.session.query(DeletionRequest).filter(
            DeletionRequest.user_id_hash == user_id_hash
        )
        
        if status:
            query = query.filter(DeletionRequest.status == status)
        
        return query.order_by(DeletionRequest.requested_at.desc()).all()
    
    def update_deletion_request_status(self, request_id: str, status: str, completed_at: Optional[datetime] = None) -> bool:
        """Update deletion request status"""
        deletion_request = self.get_deletion_request(request_id)
        if not deletion_request:
            return False
        
        deletion_request.status = status  # type: ignore
        if completed_at:
            deletion_request.completed_at = completed_at  # type: ignore
        elif status == "completed":
            deletion_request.completed_at = datetime.utcnow()  # type: ignore
        
        self.session.commit()
        return True
    
    def create_data_export_request(self, export_data: dict) -> DataExportRequest:
        """Create a new GDPR data export request"""
        export_request = DataExportRequest(**export_data)
        self.session.add(export_request)
        self.session.commit()
        self.session.refresh(export_request)
        return export_request
    
    def get_data_export_request(self, export_id: str) -> Optional[DataExportRequest]:
        """Get data export request by export ID"""
        return self.session.query(DataExportRequest).filter(
            DataExportRequest.export_id == export_id
        ).first()
    
    def get_user_data_export_requests(self, user_id_hash: str, status: Optional[str] = None) -> List[DataExportRequest]:
        """Get data export requests for a user"""
        query = self.session.query(DataExportRequest).filter(
            DataExportRequest.user_id_hash == user_id_hash
        )
        
        if status:
            query = query.filter(DataExportRequest.status == status)
        
        return query.order_by(DataExportRequest.requested_at.desc()).all()
    
    def update_data_export_request_status(self, export_id: str, status: str, file_path: Optional[str] = None, 
                                        file_size: Optional[int] = None, completed_at: Optional[datetime] = None) -> bool:
        """Update data export request status"""
        export_request = self.get_data_export_request(export_id)
        if not export_request:
            return False
        
        export_request.status = status  # type: ignore
        if file_path:
            export_request.file_path = file_path  # type: ignore
        if file_size:
            export_request.file_size_bytes = file_size  # type: ignore
        if completed_at:
            export_request.completed_at = completed_at  # type: ignore
        elif status == "completed":
            export_request.completed_at = datetime.utcnow()  # type: ignore
        
        self.session.commit()
        return True
    
    def increment_export_download_count(self, export_id: str) -> bool:
        """Increment download count for export request"""
        export_request = self.get_data_export_request(export_id)
        if not export_request:
            return False
        
        export_request.download_count = (export_request.download_count or 0) + 1  # type: ignore
        self.session.commit()
        return True
    
    def create_consent_record(self, consent_data: dict) -> ConsentRecord:
        """Create a new GDPR consent record"""
        consent_record = ConsentRecord(**consent_data)
        self.session.add(consent_record)
        self.session.commit()
        self.session.refresh(consent_record)
        return consent_record
    
    def get_consent_record(self, user_id_hash: str, consent_type: str) -> Optional[ConsentRecord]:
        """Get consent record by user ID and consent type"""
        return self.session.query(ConsentRecord).filter(
            ConsentRecord.user_id_hash == user_id_hash,
            ConsentRecord.consent_type == consent_type
        ).order_by(ConsentRecord.timestamp.desc()).first()
    
    def get_user_consent_records(self, user_id_hash: str, consent_type: Optional[str] = None) -> List[ConsentRecord]:
        """Get all consent records for a user"""
        query = self.session.query(ConsentRecord).filter(
            ConsentRecord.user_id_hash == user_id_hash
        )
        
        if consent_type:
            query = query.filter(ConsentRecord.consent_type == consent_type)
        
        return query.order_by(ConsentRecord.timestamp.desc()).all()
    
    def update_consent_record_status(self, user_id_hash: str, consent_type: str, status: str, 
                                   expires_at: Optional[datetime] = None) -> bool:
        """Update consent record status (creates new record for audit trail)"""
        # Get the latest consent record
        latest_record = self.get_consent_record(user_id_hash, consent_type)
        
        # Create new record with updated status
        new_record_data = {
            "user_id_hash": user_id_hash,
            "consent_type": consent_type,
            "status": status,
            "consent_version": latest_record.consent_version if latest_record else "1.0",
            "expires_at": expires_at
        }
        
        if latest_record:
            new_record_data["ip_address_encrypted"] = latest_record.ip_address_encrypted
            new_record_data["user_agent_hash"] = latest_record.user_agent_hash
            new_record_data["additional_data_json"] = latest_record.additional_data_json
        
        self.create_consent_record(new_record_data)
        return True
    
    def get_expired_consents(self) -> List[ConsentRecord]:
        """Get all expired consent records"""
        return self.session.query(ConsentRecord).filter(
            ConsentRecord.expires_at < datetime.utcnow(),
            ConsentRecord.status == "granted"
        ).all()
    
    def cleanup_expired_consents(self, max_age_days: int = 365) -> int:
        """Clean up very old consent records (keep for audit trail)"""
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        
        old_consents = self.session.query(ConsentRecord).filter(
            ConsentRecord.timestamp < cutoff_date
        ).all()
        
        count = 0
        for consent in old_consents:
            # In production, you might anonymize rather than delete for audit purposes
            # For now, we'll delete very old records
            self.session.delete(consent)
            count += 1
        
        self.session.commit()
        return count
    
    def cleanup_expired_exports(self, max_age_days: int = 7) -> int:
        """Clean up expired export requests and their files"""
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        
        expired_exports = self.session.query(DataExportRequest).filter(
            DataExportRequest.expires_at < cutoff_date,
            DataExportRequest.status.in_(["expired", "completed"])
        ).all()
        
        count = 0
        for export in expired_exports:
            # In production, you would also delete the actual files here
            if export.file_path and os.path.exists(str(export.file_path)):  # type: ignore
                try:
                    os.remove(str(export.file_path))  # type: ignore
                except OSError:
                    pass
            
            # Delete the database record
            self.session.delete(export)
            count += 1
        
        self.session.commit()
        return count


# Initialize database if running as main module
if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully")