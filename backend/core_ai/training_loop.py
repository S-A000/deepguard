import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

from backend.core_ai.models.fusion_net import DeepGuardFusionModel 
from backend.core_ai.models.branch_b_physics import calculate_physics_penalty 
from datasets.loaders.multi_modal_loader import DeepGuardDataset 

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Enterprise System Online! Training on: {device}")

    # ==========================================
    # 🗂️ MASTER DATASET CONFIGURATION
    # ==========================================
    REAL_DIRS = [
        os.path.join(root_dir, 'datasets/Kinetics700/'),
        os.path.join(root_dir, 'datasets/UCF-101/'),
        os.path.join(root_dir, 'datasets/FaceForensics++/original/'),
        os.path.join(root_dir, 'datasets/MSR_VTT/'),
        os.path.join(root_dir, 'datasets/custom_3k/real/')
    ]
    
    FAKE_DIRS = [
        os.path.join(root_dir, 'datasets/DFDC/'),
        os.path.join(root_dir, 'datasets/GenVidBench/'),
        os.path.join(root_dir, 'datasets/FaceForensics++/manipulated/'),
        os.path.join(root_dir, 'datasets/custom_3k/fake/')
    ]
    
    SAVE_MODEL_DIR = os.path.join(root_dir, 'saved_models/production/')
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    final_model_path = os.path.join(SAVE_MODEL_DIR, "deepguard_fusion_v1.pth")

    # ==========================================
    # 📦 FLEXIBLE DATA LOADING
    # ==========================================
    print("[*] Loading a Chunk of Enterprise Dataset...")
    # Har baar jab run karenge, ye random 2000 naye videos uthayega (1000 real, 1000 fake)
    dataset = DeepGuardDataset(
        real_dirs=REAL_DIRS, 
        fake_dirs=FAKE_DIRS, 
        num_frames=16, 
        max_samples=2000 # Aap isay apni marzi se kam/zyada kar sakte hain
    )
    
    if len(dataset) == 0:
        print("❌ Error: Koi bhi video nahi mili! Folder paths check karein.")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    
    # ==========================================
    # 🧠 MODEL INITIALIZATION & RESUME LOGIC
    # ==========================================
    model = DeepGuardFusionModel().to(device)
    
    # 🌟 MAGIC HAPPENS HERE: INCREMENTAL TRAINING CHECK
    if os.path.exists(final_model_path):
        print(f"\n🔄 Purana Model Mil Gaya! Loading Weights from: {final_model_path}")
        # Purane model ka dimaagh naye model mein daal do
        model.load_state_dict(torch.load(final_model_path, map_location=device))
        print("✅ Resume Successful. Naye data par aage ki training shuru ho rahi hai...")
    else:
        print("\n🆕 Naya Model ban raha hai (First Time Training)...")

    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    EPOCHS = 5 # Chunk chota hai toh Epochs bhi kam rakhein (e.g., 5)
    print("\n🔥 INITIATING INCREMENTAL TRAINING WITH PINN NOVELTY 🔥\n")

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        for video_rgb, flow, fft, audio, labels in loop:
            video_rgb = video_rgb.to(device)
            flow = flow.to(device)
            fft = fft.to(device)
            audio = audio.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            predictions = model(video_rgb, flow, fft, audio)
            
            bce_loss = criterion(predictions, labels) 
            pinn_loss = calculate_physics_penalty(flow) 
            
            loss = bce_loss + (0.1 * pinn_loss) 
            
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(Total_Loss=f"{loss.item():.4f}", BCE=f"{bce_loss.item():.4f}", PINN=f"{pinn_loss.item():.4f}")

    # ==========================================
    # 💾 SAVE UPDATED BRAIN
    # ==========================================
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Training Chunk Complete! Model Updated & Saved at: {final_model_path}")

if __name__ == "__main__":
    train_model()