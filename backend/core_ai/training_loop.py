import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 🚀 FIX 1: Memory fragmentation bypass
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        "/kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101",
        "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real"
    ]
    
    # 🔴 FAKE VIDEOS
    raw_fake_dirs = [
        "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake"
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
    # ⚖️ THE CURE: FORCED BALANCED DATA LOADING
    # ==========================================
    print("\n[*] Loading The BALANCED Enterprise Dataset (50% Real, 50% Fake)...")
    
    # Yahan model barabar data uthayega taake lazy na ho sake!
    SAMPLES_PER_CLASS = 100 
    
    real_dataset = DeepGuardDataset(
        real_dirs=REAL_DIRS, 
        fake_dirs=[], 
        num_frames=16, 
        max_samples=SAMPLES_PER_CLASS 
    )
    
    fake_dataset = DeepGuardDataset(
        real_dirs=[], 
        fake_dirs=FAKE_DIRS, 
        num_frames=16, 
        max_samples=SAMPLES_PER_CLASS 
    )

    if len(real_dataset) == 0 or len(fake_dataset) == 0:
        print("❌ Error: Koi ek dataset khali hai. Paths check karein.")
        return

    balanced_dataset = ConcatDataset([real_dataset, fake_dataset])
    print(f"✅ Total Balanced Videos Ready: {len(balanced_dataset)}")

    # 🚀 FIX 2: num_workers=0 to prevent Kaggle deadlocks! Batch size 2 for safety.
    dataloader = DataLoader(balanced_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    
    # ==========================================
    # 🧠 MODEL INITIALIZATION & RESUME LOGIC
    # ==========================================
    model = DeepGuardFusionModel(embed_dim=256, num_heads=8).float()
    
    if torch.cuda.device_count() > 1:
        print(f"🔥 Dual GPU Activated! Using {torch.cuda.device_count()} GPUs in Parallel! 🔥")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # 🚀 FIX 3: Auto-delete old model for a guaranteed fresh start
    if os.path.exists(final_model_path):
        os.remove(final_model_path)
        print(f"\n🗑️ Purana model delete kar diya gaya! Fresh training shuru ho rahi hai...")
    
    print("\n🆕 Phase 1 Model ban raha hai (Fresh Training with Balanced Data)...")

    criterion = nn.BCEWithLogitsLoss() 
    
    # 🚀 FIX 4: Higher learning rate to force learning visibility
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scaler = torch.amp.GradScaler('cuda')

    # ==========================================
    # 🔥 THE DEBUG TRAINING LOOP
    # ==========================================
    EPOCHS = 5
    print(f"\n🔥 INITIATING DEBUG TRAINING FOR {EPOCHS} EPOCHS 🔥\n")

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        # Enumerate added to get exact batch index
        for batch_idx, (video_rgb, flow, fft, audio, labels) in enumerate(loop):
            
            video_rgb = torch.nan_to_num(video_rgb.to(device))
            flow = torch.nan_to_num(flow.to(device))
            fft = torch.nan_to_num(fft.to(device))
            audio = torch.nan_to_num(audio.to(device))
            
            labels = labels.float().to(device).view(-1, 1)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                predictions = model(video_rgb, flow, fft, audio)
                
                if torch.isnan(predictions).any():
                    print(f"\n⚠️ WARNING: NaN in predictions at batch {batch_idx+1}")
                    continue
                
                bce_loss = criterion(predictions.float(), labels.float()) 
                # Temporary: Ignoring PINN to isolate BCE classification
                loss = bce_loss 
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            # Stricter gradient clipping to avoid explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
            
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(BCE=f"{bce_loss.item():.4f}")
            
            # 🚀 FIX 5: Manual Terminal Print for absolute visibility
            if batch_idx % 2 == 0: 
                print(f"   >>> Batch {batch_idx+1} Done | BCE Loss: {bce_loss.item():.4f}")
            
            del predictions, loss, bce_loss
            torch.cuda.empty_cache()

    # ==========================================
    # 💾 SAVE UPDATED BRAIN
    # ==========================================
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Training Complete! Model Updated & Saved at: {final_model_path}")

if __name__ == "__main__":
    train_model()