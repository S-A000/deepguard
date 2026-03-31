from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from sqlalchemy.orm import Session
import shutil
import os
import bcrypt
from datetime import datetime

# ==========================================
# 🔗 IMPORTS FROM OTHER FILES
# ==========================================
from .database import get_db  # File 1 se database connection mangwaya
from . import models          # Models file (jahan tables define hain)

# AI Pipeline
from core_ai.inference_pipeline import analyze_video

router = APIRouter()

# Uploads folder automatically ban banane ka logic
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../storage/uploads/"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==========================================
# 1. AUTHENTICATION (Login & Signup)
# ==========================================

@router.post("/api/login")
def login_user(login_data: dict, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == login_data['email']).first()
    if not user or not bcrypt.checkpw(login_data['password'].encode('utf-8'), user.password_hash.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Access Denied: Invalid Identity.")
    
    return {
        "user_id": user.user_id, 
        "full_name": user.full_name, 
        "role": user.role 
    }

@router.post("/api/signup")
def register_user(user_data: dict, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == user_data['email']).first()
    if existing: raise HTTPException(status_code=400, detail="Identity exists.")
    
    hashed = bcrypt.hashpw(user_data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = models.User(full_name=user_data['full_name'], email=user_data['email'], password_hash=hashed, role="operator", dept_id=1)
    db.add(new_user)
    db.commit()
    return {"status": "success"}

# ==========================================
# 2. ADMIN POWER (User & Record Management)
# ==========================================

@router.get("/api/admin/users")
def get_all_users(db: Session = Depends(get_db)):
    return db.query(models.User).all()

@router.post("/api/admin/create-member")
def create_member(data: dict, db: Session = Depends(get_db)):
    hashed = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_member = models.User(
        full_name=data['full_name'], email=data['email'], 
        password_hash=hashed, role=data['role'], dept_id=1
    )
    db.add(new_member)
    db.commit()
    return {"status": "success"}

@router.delete("/api/admin/delete-record/{analysis_id}")
def delete_record(analysis_id: int, db: Session = Depends(get_db)):
    db.query(models.AnalysisHistory).filter(models.AnalysisHistory.analysis_id == analysis_id).delete()
    db.commit()
    return {"status": "success"}

# ==========================================
# 3. FORENSIC SCAN ENGINE
# ==========================================

@router.post("/api/analyze")
async def analyze_uploaded_video(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    form_data = await request.form()
    user_id = int(form_data.get("user_id", 1))
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        ai_result = analyze_video(file_path)
        
        new_analysis = models.AnalysisHistory(
            user_id=user_id, filename=file.filename,
            verdict=ai_result.get("verdict", "UNKNOWN"),
            confidence_score=ai_result.get("confidence", 0.0),
            spatial_score=ai_result.get("branch_scores", {}).get("spatial", 0.0),
            physics_score=ai_result.get("branch_scores", {}).get("physics", 0.0),
            forensics_score=ai_result.get("branch_scores", {}).get("forensics", 0.0),
            audio_score=ai_result.get("branch_scores", {}).get("audio", 0.0),
            processing_time_sec=1.5, client_ip=request.client.host
        )
        db.add(new_analysis)
        db.flush() 

        db.add(models.VideoMetadata(analysis_id=new_analysis.analysis_id, file_size_mb=2.5, resolution="1080p", codec="H.264"))
        db.commit()

        return {"status": "success", "verdict": new_analysis.verdict, "confidence": new_analysis.confidence_score, "branch_scores": ai_result.get("branch_scores", {})}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/history")
def get_history(db: Session = Depends(get_db)):
    return db.query(models.AnalysisHistory).order_by(models.AnalysisHistory.timestamp.desc()).all()