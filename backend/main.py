from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uvicorn

# Database aur Models import karein
import app.models as models
from app.database import engine

# Routes import
from app.api_routes import router as api_router

app = FastAPI(title="DeepGuard Enterprise API", version="1.0")

# --- DATABASE INITIALIZATION ---
print("📡 Connecting to SQL Server and Initializing Governance Tables...")
models.Base.metadata.create_all(bind=engine)

# Security Bypass for React (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect API Routes
app.include_router(api_router)

# Storage folders creation
UPLOAD_DIR = "../storage/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("../storage/reports", exist_ok=True)

@app.get("/")
def read_root():
    return {
        "status": "Online",
        "system": "DeepGuard Enterprise Governance",
        "database": "SQL Server Connected ✅"
    }

# ==========================================
# 🧠 AI INFERENCE ENDPOINT (Mobile App ke liye)
# ==========================================
@app.post("/api/analyze")
async def analyze_mobile_video(file: UploadFile = File(...)):
    print(f"\n📥 Video Received from Mobile App: {file.filename}")
    
    # 1. Video ko aapke storage/uploads folder mein save karna
    temp_video_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. Asal Jadoo (Yahan aapka 1GB wala dimaagh chalega)
        print("🧠 DeepGuard AI is scanning the video for threats...")
        
        # ⚠️ Abhi hum App connection check karne ke liye Dummy Result bhej rahe hain.
        # Jab App connect ho jayegi, toh hum yahan asal inference() function laga denge.
        result = {
            "status": "success",
            "filename": file.filename,
            "verdict": "FAKE", 
            "confidence_score": 98.5,
            "details": "Unnatural facial movement detected by TimeSformer."
        }
        
        print("✅ Scan Complete. Sending result to Mobile App.")
        
    finally:
        # 3. Scan ke baad server ka bojh halka karne ke liye video delete karna
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
    return result

if __name__ == "__main__":
    print("🚀 Starting DeepGuard Backend Server...")
    # Mobile device se connect karne ke liye host ko '0.0.0.0' karna behtar hai,
    # lekin abhi testing ke liye hum 127.0.0.1 hi use kar rahe hain.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)