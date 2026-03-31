from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Department(Base):
    __tablename__ = "Departments"
    dept_id = Column(Integer, primary_key=True, index=True)
    dept_name = Column(String(100), unique=True)

class User(Base):
    __tablename__ = "Users"
    user_id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100))
    email = Column(String(255), unique=True, index=True)
    password_hash = Column(String)
    role = Column(String(50), default="operator")
    dept_id = Column(Integer, ForeignKey("Departments.dept_id"))
    clearance_level = Column(Integer, default=1)
    is_active = Column(Boolean, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

class AnalysisHistory(Base):
    __tablename__ = "AnalysisHistory"
    analysis_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("Users.user_id"))
    filename = Column(String(255))
    verdict = Column(String(50))
    confidence_score = Column(Float)
    spatial_score = Column(Float)
    physics_score = Column(Float)
    forensics_score = Column(Float)
    audio_score = Column(Float)
    processing_time_sec = Column(Float)
    client_ip = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    metadata_rel = relationship("VideoMetadata", back_populates="analysis", uselist=False)
    verifications = relationship("ResultVerification", back_populates="analysis")

class VideoMetadata(Base):
    __tablename__ = "VideoMetadata"
    meta_id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("AnalysisHistory.analysis_id", ondelete="CASCADE"))
    file_size_mb = Column(Float)
    resolution = Column(String(50))
    codec = Column(String(50))
    fps = Column(Integer, default=30)

    analysis = relationship("AnalysisHistory", back_populates="metadata_rel")

class ResultVerification(Base):
    __tablename__ = "ResultVerifications"
    verify_id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("AnalysisHistory.analysis_id"))
    auditor_id = Column(Integer, ForeignKey("Users.user_id"), nullable=True)
    verification_status = Column(String(50), default="PENDING")
    comments = Column(String)

    analysis = relationship("AnalysisHistory", back_populates="verifications")

class AuditLog(Base):
    __tablename__ = "AuditLogs"
    log_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("Users.user_id"))
    action_type = Column(String(100))
    description = Column(String)
    ip_address = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)