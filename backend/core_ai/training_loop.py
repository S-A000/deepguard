import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==========================================
# 📂 SYSTEM PATHS
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

# ==========================================
# 🧠 DEEPGUARD EXPERTS IMPORT
# ==========================================
from backend.core_ai.models.fusion_net import DeepGuardFusionModel 
from backend.core_ai.models.branch_b_physics import calculate_physics_penalty 
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset

# 🚀 THE MAGIC KAGGLE PATH FINDER (CTO HACK)
def find_dir(target_folder):
    """Kaggle ke nested folders mein ghus kar path dhoondta hai"""
    paths = glob.glob(f"/kaggle/input/*/{target_folder}/")
    if not paths:
        paths = glob.glob(f"/kaggle/input/*/*/{target_folder}/")
    
    if paths:
        return paths[0]
    else:
        print(f"⚠️ Warning: Yeh folder Kaggle ko nahi mila -> {target_folder}")
        return ""

def train_model():
    # GPU Setup (Kaggle T4 x2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Enterprise System Online! Training on: {device}")

    # ==========================================
    # 🗂️ AUTO-DETECT MEGA DATASET CONFIGURATION
    # ==========================================
    print("\n[*] Kaggle GPS: Auto-Detecting Dataset Paths...")
    
    # 🟢 REAL VIDEOS (Tasweer ke hisaab se exact inner folders)
    raw_real_dirs = [
        find_dir("kinetics400_5per"),
        find_dir("kinetics600_5per"),
        find_dir("TrainValVideo"),             # MSRVTT
        find_dir("UCF101/UCF-101"),            # UCF101 (Double nested)
        find_dir("FF++/real"),                 # FF++ Real
        find_dir("dfdc_equal_split_part_14/real") # DFDC 14 Real
    ]
    
    # 🔴 FAKE VIDEOS (Tasweer ke hisaab se exact inner folders)
    raw_fake_dirs = [
        find_dir("dfdc_train_part_1"),         # DFDC 1
        find_dir("dfdc_train_part_18"),        # DFDC 18
        find_dir("FF++/fake"),                 # FF++ Fake
        find_dir("dfdc_equal_split_part_14/fake") # DFDC 14 Fake
    ]

    # Khali (Empty) paths ko filter kar dein
    REAL_DIRS = [d for d in raw_real_dirs if d != ""]
    FAKE_DIRS = [d for d in raw_fake_dirs if d != ""]
    
    print(f"✅ Found {len(REAL_DIRS)} Real Folders and {len(FAKE_DIRS)} Fake Folders!")

    # Model Save Location 
    SAVE_MODEL_DIR = "/kaggle/working/saved_models/production/"
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    final_model_path = os.path.join(SAVE_MODEL_DIR, "deepguard_fusion_v1.pth")

    # ==========================================
    # 📦 FLEXIBLE DATA LOADING
    # ==========================================
    print("\n[*] Loading The MEGA Enterprise Dataset...")
    dataset = DeepGuardDataset(
        real_dirs=REAL_DIRS, 
        fake_dirs=FAKE_DIRS, 
        num_frames=16, 
        max_samples=5000 
    )
    
    if len(dataset) == 0:
        print("❌ Error: Dataset abhi bhi 0 hai. Dataloader mein koi issue hai.")
        return

    # DataLoader 
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    
    # ==========================================
    # 🧠 MODEL INITIALIZATION & RESUME LOGIC
    # ==========================================
    model = DeepGuardFusionModel(embed_dim=256, num_heads=8).to(device)
    
    if os.path.exists(final_model_path):
        print(f"\n🔄 Purana Model Mil Gaya! Loading Weights from: {final_model_path}")
        model.load_state_dict(torch.load(final_model_path, map_location=device), strict=False)
        print("✅ Resume Successful. Continuing training...")
    else:
        print("\n🆕 Phase 1 Model ban raha hai (First Time Training with Dynamic Attention)...")

    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # ==========================================
    # 🔥 THE SOTA TRAINING LOOP
    # ==========================================
    EPOCHS = 35 
    print(f"\n🔥 INITIATING SOTA TRAINING FOR {EPOCHS} EPOCHS 🔥\n")

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        for video_rgb, flow, fft, audio, labels in loop:
            video_rgb = video_rgb.to(device)
            flow = flow.to(device)
            fft = fft.to(device)
            audio = audio.to(device)
            
            labels = labels.float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            predictions = model(video_rgb, flow, fft, audio)
            bce_loss = criterion(predictions, labels) 
            pinn_loss = calculate_physics_penalty(flow, alpha=0.1, beta=0.1) 
            loss = bce_loss + pinn_loss 
            
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(Total_Loss=f"{loss.item():.4f}", BCE=f"{bce_loss.item():.4f}", PINN=f"{pinn_loss.item():.4f}")

    # ==========================================
    # 💾 SAVE UPDATED BRAIN
    # ==========================================
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Training Complete! Model Updated & Saved at: {final_model_path}")

if __name__ == "__main__":
    train_model()