import os
import sys
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
from custom_datasets.loaders.multi_modal_loader import DeepGuardDatase

def train_model():
    # GPU Setup (Kaggle T4 x2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Enterprise System Online! Training on: {device}")

    # ==========================================
    # 🗂️ MASTER DATASET CONFIGURATION (KAGGLE)
    # ==========================================
    # CTO Note: Kaggle par aapke dataset ka exact folder name yahan aayega
    KAGGLE_DATASET_ROOT = "/kaggle/input/deepguard-master-data" 
    
    # 🟢 REAL VIDEOS
    REAL_DIRS = [
        os.path.join(KAGGLE_DATASET_ROOT, 'kinetics400_5per/'),
        os.path.join(KAGGLE_DATASET_ROOT, 'kinetics600_5per/'),
        os.path.join(KAGGLE_DATASET_ROOT, 'UCF-101/'),
        os.path.join(KAGGLE_DATASET_ROOT, 'MSR_VTT/'),
        os.path.join(KAGGLE_DATASET_ROOT, 'FaceForensics++/original/') # FF++ ke real
    ]
    
    # 🔴 FAKE VIDEOS
    FAKE_DIRS = [
        os.path.join(KAGGLE_DATASET_ROOT, 'dfdc_train_part_0/'),
        os.path.join(KAGGLE_DATASET_ROOT, 'dfdc_train_part_1/'),
        os.path.join(KAGGLE_DATASET_ROOT, 'FaceForensics++/manipulated/') # FF++ ke fakes
    ]
    
    # Model Save Location (Kaggle requires /kaggle/working/)
    SAVE_MODEL_DIR = "/kaggle/working/saved_models/production/"
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    final_model_path = os.path.join(SAVE_MODEL_DIR, "deepguard_fusion_v1.pth")

    # ==========================================
    # 📦 FLEXIBLE DATA LOADING
    # ==========================================
    print("[*] Loading Enterprise Dataset...")
    dataset = DeepGuardDataset(
        real_dirs=REAL_DIRS, 
        fake_dirs=FAKE_DIRS, 
        num_frames=16, 
        max_samples=2000 
    )
    
    if len(dataset) == 0:
        print("❌ Error: Koi bhi video nahi mili! Kaggle Input paths verify karein.")
        return

    # DataLoader (Batch size 4 for Dual GPU stability)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    
    # ==========================================
    # 🧠 MODEL INITIALIZATION & RESUME LOGIC
    # ==========================================
    # Initialize SOTA Model with Dynamic Cross-Attention (num_heads=8)
    model = DeepGuardFusionModel(embed_dim=256, num_heads=8).to(device)
    
    # Resume Logic for Incremental Training
    if os.path.exists(final_model_path):
        print(f"\n🔄 Purana Model Mil Gaya! Loading Weights from: {final_model_path}")
        model.load_state_dict(torch.load(final_model_path, map_location=device), strict=False)
        print("✅ Resume Successful. Continuing training...")
    else:
        print("\n🆕 Phase 1 Model ban raha hai (First Time Training with Dynamic Attention)...")

    # Optimizer & Loss Function
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    # ==========================================
    # 🔥 THE SOTA TRAINING LOOP
    # ==========================================
    EPOCHS = 35 
    print(f"\n🔥 INITIATING SOTA TRAINING FOR {EPOCHS} EPOCHS WITH EXPLICIT PHYSICS & DYNAMIC ATTENTION 🔥\n")

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        for video_rgb, flow, fft, audio, labels in loop:
            # Send data to GPU
            video_rgb = video_rgb.to(device)
            flow = flow.to(device)
            fft = fft.to(device)
            audio = audio.to(device)
            
            # 🚨 Shape Fix: Labels ko (batch, 1) banana zaroori hai
            labels = labels.float().to(device).unsqueeze(1)
            
            # Reset Gradients
            optimizer.zero_grad()
            
            # 1. Forward Pass (Cross-Attention processes all 4 streams)
            predictions = model(video_rgb, flow, fft, audio)
            
            # 2. Calculate Base BCE Loss
            bce_loss = criterion(predictions, labels) 
            
            # 3. Calculate Explicit Physics Penalty (Divergence + Laplacian)
            pinn_loss = calculate_physics_penalty(flow, alpha=0.1, beta=0.1) 
            
            # 4. Total Loss Calculation
            loss = bce_loss + pinn_loss 
            
            # Backpropagation & Weight Update
            loss.backward()
            optimizer.step()
            
            # Update Progress Bar
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(Total_Loss=f"{loss.item():.4f}", BCE=f"{bce_loss.item():.4f}", PINN=f"{pinn_loss.item():.4f}")

    # ==========================================
    # 💾 SAVE UPDATED BRAIN
    # ==========================================
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Training Complete! Model Updated & Saved at: {final_model_path}")

if __name__ == "__main__":
    train_model()