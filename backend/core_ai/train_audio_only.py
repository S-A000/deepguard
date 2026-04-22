import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 📂 SYSTEM PATHS
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

# 🧠 SIRF AUDIO EXPERT IMPORT KAREIN
from backend.core_ai.models.branch_d_audio import AudioExpert
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset

# ==========================================
# 🛠️ TEMPORARY AUDIO MODEL (Sirf seekhne ke liye)
# ==========================================
class AudioOnlyDeepGuard(nn.Module):
    def __init__(self, embed_dim=256):
        super(AudioOnlyDeepGuard, self).__init__()
        self.expert = AudioExpert(embed_dim=embed_dim)
        
        # Ek chota sa dimaagh jo sirf awaaz sun kar Fake/Real batayega
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, audio_waveforms):
        features = self.expert(audio_waveforms)
        return self.classifier(features)

# ==========================================
# 🚀 TRAINING FUNCTION
# ==========================================
def train_audio_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎧 Audio-Only System Online! Training on: {device}")

    # 1. PATHS (Phase 1: Sirf FaceForensics++)
    REAL_DIRS = ["/kaggle/input/datasets/hungle3401/faceforensics/FF++/real"]
    FAKE_DIRS = ["/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake"]
    
    # 2. DATASET (Batch size bara rakh sakte hain kyunke sirf audio hai!)
    SAMPLES_PER_CLASS = 1000 
    
    print("\n[*] Loading Audio Dataset...")
    real_dataset = DeepGuardDataset(real_dirs=REAL_DIRS, fake_dirs=[], max_samples=SAMPLES_PER_CLASS)
    fake_dataset = DeepGuardDataset(real_dirs=[], fake_dirs=FAKE_DIRS, max_samples=SAMPLES_PER_CLASS)
    
    balanced_dataset = ConcatDataset([real_dataset, fake_dataset])
    dataloader = DataLoader(balanced_dataset, batch_size=16, shuffle=True, num_workers=2) # Batch 16 is safe here

    # 3. MODEL SETUP
    model = AudioOnlyDeepGuard().float().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    SAVE_PATH = "/kaggle/working/saved_models/production/audio_expert_pretrained.pth"
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # 4. TRAINING LOOP
    EPOCHS = 30
    print(f"\n🔥 INITIATING AUDIO TRAINING FOR {EPOCHS} EPOCHS 🔥\n")

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        for batch_idx, (video_rgb, flow, fft, audio, labels) in enumerate(loop):
            # Humein video, flow, fft se koi matlab nahi. Sirf Audio lenge!
            audio = torch.nan_to_num(audio.to(device))
            labels = labels.float().to(device).view(-1, 1)

            optimizer.zero_grad()
            
            # Seedha Float32 mein training (Kyunke audio halke hote hain, memory masla nahi hoga)
            predictions = model(audio)
            
            if torch.isnan(predictions).any():
                print(f"⚠️ NaN detected at batch {batch_idx}! Audio pipeline has a bug.")
                continue
                
            loss = criterion(predictions, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(BCE_Loss=f"{loss.item():.4f}")

    # 5. SAVE SIRF EXPERT KO KARENGE (Classifier humein nahi chahiye baad mein)
    torch.save(model.expert.state_dict(), SAVE_PATH)
    print(f"\n✅ Audio Pre-training Complete! Brain Saved at: {SAVE_PATH}")

if __name__ == "__main__":
    train_audio_model()