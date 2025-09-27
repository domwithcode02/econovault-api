from sqlalchemy.engine import create_engine
from sqlalchemy import Column, Integer, String, DateTime, Date, DECIMAL, ForeignKey, Text, Boolean, insert, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
from typing import Generator, Optional, List
from datetime import date
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://econovault_db_user:ZAVQLwpbCEr9pEKnYMugHOvsQ1764yHe@dpg-d3bmeur7mgec739r7af0-a/econovault_db")

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
    created_at = Column(DateTime, default=func.now())
    last_used = Column(DateTime)
    rate_limit_per_minute = Column(Integer, default=60)
    rate_limit_per_hour = Column(Integer, default=3600)


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


# Initialize database if running as main module
if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully")