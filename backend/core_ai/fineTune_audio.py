import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score
import numpy as np

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 🚀 POINT 9: Faster Training Stability
torch.backends.cudnn.benchmark = True 

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

from backend.core_ai.models.branch_d_audio import AudioExpert
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset

CURRENT_PHASE = 1  

# ==========================================
# 🚀 POINT 5: 1D Waveform Augmentation (Time Masking)
# ==========================================
def augment_waveform(audio):
    # Sirf 50% chance hai augment hone ka taake data natural bhi rahe
    if torch.rand(1).item() > 0.5:
        # Time masking (Zeroing out random chunks of audio)
        time_len = audio.shape[-1]
        t = torch.randint(0, time_len // 10, (1,)).item() # Max 10% mask
        t0 = torch.randint(0, time_len - t, (1,)).item()
        audio[:, t0:t0+t] = 0
    return audio

# ==========================================
# 🚀 POINT 1: Deeper Classifier with GELU
# ==========================================
class AudioOnlyDeepGuard(nn.Module):
    def __init__(self, embed_dim=256):
        super(AudioOnlyDeepGuard, self).__init__()
        self.expert = AudioExpert(embed_dim=embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

    def forward(self, audio_waveforms):
        features = self.expert(audio_waveforms)
        return self.classifier(features)

def train_audio_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎧 V2.0 Enterprise Engine Online! Phase: {CURRENT_PHASE} on {device}")

    if CURRENT_PHASE == 1:
        REAL_DIRS = [
             "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Real/Real",
            "/kaggle/input/datasets/kynthesis/vctk-corpus/VCTK-Corpus/wav48"
        ]
        FAKE_DIRS = [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Fake/Fake",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_melgan",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_parallel_wavegan"
        ]
        
        # 🚀 POINT 3: Better LRs (Freezing with extra steps fixed)
        LR_BACKBONE, LR_CLASSIFIER = 1e-6, 1e-5 
        
        PREV_MODEL_PATH = "/kaggle/input/deepguard-weights-v1/audio_phase1.pth" 
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase1_v2.pth"
    else: return

    valid_real = [d for d in REAL_DIRS if os.path.exists(d)]
    valid_fake = [d for d in FAKE_DIRS if os.path.exists(d)]

    fake_ds = DeepGuardDataset(real_dirs=[], fake_dirs=valid_fake, max_samples=None, mode="audio_only")
    num_fake = len(fake_ds)
    if num_fake == 0: return

    real_ds = DeepGuardDataset(real_dirs=valid_real, fake_dirs=[], max_samples=num_fake, mode="audio_only")
    full_dataset = ConcatDataset([real_ds, fake_ds])
    
    # ==========================================
    # 🚀 POINT 7: Validation Split (80% Train, 20% Val)
    # ==========================================
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    print(f"⚖️ Dataset Split: Train ({train_size}) | Validation ({val_size})")
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = AudioOnlyDeepGuard().float().to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    if PREV_MODEL_PATH and os.path.exists(PREV_MODEL_PATH):
        try:
            # 🚀 Point 10 Compatibility: Load expert weights safely if previous model only saved expert
            target = model.module.expert if isinstance(model, nn.DataParallel) else model.expert
            target.load_state_dict(torch.load(PREV_MODEL_PATH, map_location=device), strict=False)
            print(f"✅ Loaded Previous Backbone from {PREV_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Memory load warning: {e}")

    # ==========================================
    # 🚀 POINT 2: Mid-Level Unfreezing (Layers 6 to 11)
    # ==========================================
    print("🔓 Unfreezing Layers 6 to 11 for Mid-Level deepfake features...")
    for name, param in model.named_parameters():
        if any(f"encoder.layers.{i}" in name for i in range(6, 12)) or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': model.module.expert.parameters() if isinstance(model, nn.DataParallel) else model.expert.parameters(), 'lr': LR_BACKBONE},
        {'params': model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters(), 'lr': LR_CLASSIFIER}
    ], weight_decay=0.01)

    # 🚀 POINT 6: Back to Baseline Loss since data is perfectly balanced
    criterion = nn.BCEWithLogitsLoss()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    EPOCHS = 5 
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_f1 = 0.0

    # ==========================================
    # 🔥 TRAINING & VALIDATION LOOP
    # ==========================================
    for epoch in range(EPOCHS):
        model.train()
        train_loop = tqdm(train_loader, total=len(train_loader), leave=True)
        epoch_loss = 0
        
        for batch_idx, (_, _, _, audio, labels) in enumerate(train_loop):
            if torch.isnan(audio).any(): continue

            audio, labels = audio.to(device), labels.to(device).view(-1, 1)
            
            # 🚀 POINT 8: Sample-level Normalization (Crucial Fix)
            audio = (audio - audio.mean(dim=-1, keepdim=True)) / (audio.std(dim=-1, keepdim=True) + 1e-8)
            audio = torch.clamp(audio, -1.0, 1.0)

            # Apply Augmentation
            audio = augment_waveform(audio)

            optimizer.zero_grad()
            preds = model(audio)
            
            if torch.isnan(preds).any(): continue
            loss = criterion(preds, labels)
            if torch.isnan(loss): continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1) 
            optimizer.step()
            
            train_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}] Train")
            train_loop.set_postfix(Loss=f"{loss.item():.4f}")
            
        scheduler.step()

        # ==========================================
        # 🧪 POINT 7: VALIDATION PHASE
        # ==========================================
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for _, _, _, audio, labels in val_loader:
                audio, labels = audio.to(device), labels.to(device).view(-1, 1)
                
                # Sample-level norm (No augmentation in Validation)
                audio = (audio - audio.mean(dim=-1, keepdim=True)) / (audio.std(dim=-1, keepdim=True) + 1e-8)
                audio = torch.clamp(audio, -1.0, 1.0)

                outputs = model(audio)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds)
        
        print(f"📊 Epoch [{epoch+1}/{EPOCHS}] Stats:")
        print(f"   -> Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Save Best Model Only
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # 🚀 POINT 10: Saving Complete Model State
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), SAVE_PATH)
            print(f"   🌟 New Best Model Saved! F1: {val_f1:.4f}")

    print(f"\n✅ Training Complete! Best Val F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    train_audio_model()
