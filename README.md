# 🛡️ DeepGuard Enterprise System - AI Video Forensics

Welcome to the **DeepGuard Training Repository**! 
Yeh repository DeepGuard AI Model (A Multi-branch Neural Network) ko train karne ke liye banayi gayi hai. Agar aap is model ko apne GPU (jaise RTX 4050) par train karne wale hain, toh neechay diye gaye **Baby Steps** ko follow karein.

---

## 🚀 Training Shuru Karne Ke "Baby Steps"

### Step 1: Project Ko Apne PC Mein Laana (Clone)
Sabse pehle apne PC mein VS Code kholein, terminal open karein aur yeh command lagayen taake saara code aapke PC mein aa jaye:

```bash
git clone <YAHAN_APNE_GITHUB_REPO_KA_LINK_DALNA>
cd DeepGuard_Enterprise_System
```

### Step 2: Zaroori Software (Libraries) Install Karna
AI ko chalane ke liye PyTorch, OpenCV aur baqi dependencies chahiye hoti hain. Terminal mein bas yeh ek line likhein aur enter dabayen (Internet on rakhna):

```bash
pip install -r requirements.txt
```

### Step 3: Datasets (Videos) Ko Sahi Folders Mein Rakhna ⚠️ (SABSE ZAROORI)
Model ko properly train karne ke liye aapko neeche diye gaye folders mein apni `.mp4` ya `.avi` videos rakhni hongi. (GitHub par heavy videos nahi hain, isliye aapko manually daalni hongi). Agar folders nahi bane hue, toh khud create kar lein:

**Phase 1: Pretraining (Motion & Physics Sikhane Ke Liye)**
* 👉 `datasets/01_pretraining/kinetics_700/` (Asli insaani actions sikhane ke liye)
* 👉 `datasets/01_pretraining/ucf_101/` (Optical flow aur baseline motion videos)

**Phase 2: Core Deepfake Training (Real vs Fake)**
* 👉 `datasets/02_forensics_training/FaceForensics++/original/` (100% Asli videos)
* 👉 `datasets/02_forensics_training/FaceForensics++/manipulated/` (Deepfake/Altered videos)
* 👉 `datasets/02_forensics_training/Celeb-DF/` (High-quality deepfakes for advanced testing)

**Phase 3: Modern Generative AI Threats (Zero-Day Attacks)**
* 👉 `datasets/03_modern_threats/sora_clips/` (OpenAI Sora ki generated videos)
* 👉 `datasets/03_modern_threats/runway_clips/` (Runway Gen-3 ki high-fidelity videos)

*(Note: Training start karne se pehle **FaceForensics++** ke `original/` aur `manipulated/` folders mein kam az kam kuch videos lazmi honi chahiye, warna data loader error dega).*

### Step 4: AI Engine Ko Start Karna (The Magic)
Jab sari videos apni sahi jagah par aa jayen, toh terminal mein yeh final command lagayen:

```bash
python backend/core_ai/training_loop.py
```
* **Kya hoga?** Aapko screen par aapke GPU ka naam (e.g., RTX 4050) likha aayega aur ek progress bar chalna shuru ho jayegi. Iska matlab AI seekh raha hai! Is process mein videos ki tadad ke hisaab se kuch ghante lag sakte hain.

### Step 5: Dimaagh (.pth file) Nikaalna aur Dena
Jab terminal par **"✅ Training Complete!"** likha aa jaye, toh aapka kaam khatam! 
AI ka trained dimaagh ek file ki shakal mein yahan save ho jayega:
👉 `saved_models/production/deepguard_fusion_v1.pth`

Is **`.pth`** file ko USB mein copy karein aur apne Frontend/Backend developer ko de dein taake wo isay live website par chala sake! 🎉

---

### 🐛 Troubleshooting (Agar Koi Masla Aaye)

* **Error: "No MP4 videos found in the dataset folders!"** * **Solution:** Aapne Step 3 mein videos sahi folder mein nahi rakhi hain. `FaceForensics++` ke andar `original` aur `manipulated` folders ka path dobara check karein.
* **Error: "CUDA Out of Memory"** * **Solution:** Aapka GPU memory (VRAM) full ho gaya hai. `backend/core_ai/training_loop.py` file mein jayen aur dataloader mein `batch_size=2` ko kam karke `batch_size=1` kar dein.
* **Error: "ModuleNotFoundError"**
  * **Solution:** Aapne Step 2 sahi se run nahi kiya. Terminal mein dobara `pip install -r requirements.txt` chalayen.
