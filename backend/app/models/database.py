"""
SQLAlchemy database models and session management.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.config import get_settings

Base = declarative_base()


# ── Models ─────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    organization = Column(String(200), nullable=True)
    role = Column(String(20), default="analyst", nullable=False)
    is_active = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    analyses = relationship("AnalysisRecord", back_populates="user", lazy="dynamic")
    audit_logs = relationship("AuditLog", back_populates="user", lazy="dynamic")


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)

    # Analysis results (stored as JSON)
    summary = Column(Text, nullable=True)
    overall_risk_score = Column(Integer, nullable=True)
    overall_risk_level = Column(String(20), nullable=True)
    recommendation = Column(String(50), nullable=True)
    risk_flags = Column(JSON, nullable=True)  # List of risk flag dicts
    detailed_analysis = Column(Text, nullable=True)
    guidelines_checked = Column(Integer, default=0)
    processing_time_seconds = Column(Float, default=0.0)

    # Meta
    llm_provider = Column(String(50), nullable=True)
    llm_model = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="analyses")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)  # e.g., "login", "analyze", "register"
    resource = Column(String(255), nullable=True)  # e.g., filename
    details = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="audit_logs")


# ── Phase 3: Enterprise Features ───────────────────────────────────

class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    key_hash = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    scopes = Column(String(255), default="read,write", nullable=False)  # Comma-separated list
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Integer, default=1, nullable=False)

    user = relationship("User")


class BackgroundTask(Base):
    __tablename__ = "background_tasks"

    id = Column(String(50), primary_key=True, index=True)  # UUID string
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    task_type = Column(String(50), nullable=False)  # e.g., "analysis"
    status = Column(String(20), default="pending", nullable=False)  # pending, running, complete, failed
    progress = Column(Float, default=0.0)
    result_json = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    agent_trace_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")


class WebhookEndpointModel(Base):
    __tablename__ = "webhook_endpoints"

    id = Column(String(50), primary_key=True, index=True)  # UUID string
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    url = Column(String(500), nullable=False)
    events = Column(String(500), nullable=False)  # Comma-separated list of events
    secret = Column(String(255), nullable=False)
    description = Column(String(255), nullable=True)
    is_active = Column(Integer, default=1, nullable=False)
    failure_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


# ── Database Session ───────────────────────────────────────────────

_engine = None
_async_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = settings.DATABASE_URL
        
        # Ensure aiosqlite is used for SQLite async support
        if db_url.startswith("sqlite"):
            if "+aiosqlite" not in db_url:
                db_url = db_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
            _engine = create_async_engine(
                db_url, 
                echo=settings.DEBUG,
                connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
            )
        else:
            # PostgreSQL or other async DB
            if "postgresql://" in db_url and "+asyncpg" not in db_url:
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            _engine = create_async_engine(
                db_url,
                echo=settings.DEBUG,
            )
    return _engine


def get_session_factory():
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


async def get_db():
    """FastAPI dependency: yields an async database session."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables():
    """Create all tables (for development — use Alembic for production)."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
