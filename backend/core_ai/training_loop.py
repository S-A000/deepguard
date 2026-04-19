import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

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

def train_model():
    # GPU Setup (Kaggle T4 x2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Enterprise System Online! Training on: {device}")

    # ==========================================
    # 🗂️ ABSOLUTE DATASET PATHS (DIRECT KAGGLE GPS)
    # ==========================================
    print("\n[*] Loading EXACT Kaggle Dataset Paths...")
    
    # 🟢 REAL VIDEOS
    raw_real_dirs = [
        "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train",
        "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics600_5per/kinetics600_5per/train",
        "/kaggle/input/datasets/abdullahpy/msrvtt/TrainValVideo",
        "/kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101",
        "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
        "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real"
    ]
    
    # 🔴 FAKE VIDEOS
    raw_fake_dirs = [
        "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
        "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18",
        "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
        "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake"
    ]

    # Verify which paths actually exist right now to prevent crashes
    REAL_DIRS = [d for d in raw_real_dirs if os.path.exists(d)]
    FAKE_DIRS = [d for d in raw_fake_dirs if os.path.exists(d)]
    
    for missing in set(raw_real_dirs) - set(REAL_DIRS):
        print(f"⚠️ Missing Real Path: {missing}")
    for missing in set(raw_fake_dirs) - set(FAKE_DIRS):
        print(f"⚠️ Missing Fake Path: {missing}")
        
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
        print("❌ Error: Dataset abhi bhi 0 hai. Dataloader mein koi issue hai ya folders khali hain.")
        return

    # 🚀 FIX: Batch size increased to 8 for Dual GPU processing
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    
    # ==========================================
    # 🧠 MODEL INITIALIZATION & RESUME LOGIC
    # ==========================================
    model = DeepGuardFusionModel(embed_dim=256, num_heads=8).float()
    
    # 🚀 DUAL GPU ENABLER
    if torch.cuda.device_count() > 1:
        print(f"🔥 Dual GPU Activated! Using {torch.cuda.device_count()} GPUs in Parallel! 🔥")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    if os.path.exists(final_model_path):
        print(f"\n🔄 Purana Model Mil Gaya! Loading Weights from: {final_model_path}")
        model.load_state_dict(torch.load(final_model_path, map_location=device), strict=False)
        print("✅ Resume Successful. Continuing training...")
    else:
        print("\n🆕 Phase 1 Model ban raha hai (First Time Training with Unified Attention)...")

    criterion = nn.BCEWithLogitsLoss() 
    
    # Lower Learning Rate to prevent NaN (2e-5 is standard for Vision Transformers)
    optimizer = optim.AdamW(model.parameters(), lr=0.00002)

    # Modern AMP Scaler Syntax
    scaler = torch.amp.GradScaler('cuda')

    # ==========================================
    # 🔥 THE SOTA TRAINING LOOP
    # ==========================================
    EPOCHS = 6 
    print(f"\n🔥 INITIATING SOTA TRAINING FOR {EPOCHS} EPOCHS 🔥\n")

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        for video_rgb, flow, fft, audio, labels in loop:
            
            # 🚀 DATA SANITIZER: Replace NaN/Inf from corrupt media files with 0s
            video_rgb = torch.nan_to_num(video_rgb.to(device))
            flow = torch.nan_to_num(flow.to(device))
            fft = torch.nan_to_num(fft.to(device))
            audio = torch.nan_to_num(audio.to(device))
            
            labels = labels.float().to(device).view(-1, 1)
            
            optimizer.zero_grad()
            
            # Modern Autocast Syntax
            with torch.amp.autocast('cuda'):
                predictions = model(video_rgb, flow, fft, audio)
                
                # 🚀 BATCH SKIPPER: If predictions still output NaN, skip this specific batch safely
                if torch.isnan(predictions).any():
                    continue
                
                # Force predictions to float32 before calculating loss
                bce_loss = criterion(predictions.float(), labels.float()) 
                
                pinn_loss = calculate_physics_penalty(flow, alpha=0.1, beta=0.1) 
                loss = bce_loss + pinn_loss 
            
            # SOTA GRADIENT CLIPPING
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(Total_Loss=f"{loss.item():.4f}", BCE=f"{bce_loss.item():.4f}", PINN=f"{pinn_loss.item():.4f}")

    # ==========================================
    # 💾 SAVE UPDATED BRAIN
    # ==========================================
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Training Complete! Model Updated & Saved at: {final_model_path}")

if __name__ == "__main__":
    train_model()