import requests
import os

# 1. API ka address (FastAPI default port 8000 use karta hai)
URL = "http://127.0.0.1:8000/api/analyze"

# 2. Video ka rasta (Path)
# Aapki tasveer ke mutabiq uploads folder mein 'WhatsApp.mp4' mojood hai
VIDEO_NAME = "WhatsApp.mp4" 
VIDEO_PATH = os.path.abspath(f"storage/uploads/{VIDEO_NAME}")

def test_deepguard_api():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: {VIDEO_NAME} nahi mili! Pehle check karein ke file 'storage/uploads' mein hai.")
        return

    print(f"🚀 Sending '{VIDEO_NAME}' to DeepGuard AI Engine...")
    
    # Video file ko binary mode mein open karna
    with open(VIDEO_PATH, "rb") as f:
        files = {"file": (VIDEO_NAME, f, "video/mp4")}
        
        try:
            # API ko Request bhejna
            response = requests.post(URL, files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("\n✅ --- AI SCAN RESULT ---")
                print(f"🎬 Filename: {result.get('filename')}")
                print(f"⚠️ Verdict:  {result.get('verdict')}")
                print(f"🎯 Confidence: {result.get('confidence')}%")
                print(f"📊 Branch Scores: {result.get('branch_scores')}")
                print("--------------------------\n")
            else:
                print(f"❌ Server Error: {response.status_code}")
                print(response.text)
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Kya aapka FastAPI server (main.py) chal raha hai?")

if __name__ == "__main__":
    test_deepguard_api()