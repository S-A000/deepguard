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
# P100 GPU ke liye memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True 

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

from backend.core_ai.models.branch_d_audio import AudioExpert
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset

CURRENT_PHASE = 1  

# ==========================================
# 🛡️ BULLETPROOF 1D WAVEFORM AUGMENTATION
# ==========================================
def augment_waveform(audio):
    if torch.rand(1).item() > 0.5:
        time_len = audio.shape[-1]
        # Sirf tab augment karein jab audio reasonable length ka ho (Crash Prevention)
        if time_len > 100: 
            max_t = max(1, time_len // 10)
            t = torch.randint(1, max_t + 1, (1,)).item()
            t0 = torch.randint(0, max(1, time_len - t), (1,)).item()
            audio[..., t0:t0+t] = 0 
    return audio

# ==========================================
# 🧠 DEEPER CLASSIFIER (GELU)
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
    print(f"\n🎧 Single-GPU (P100) V2.0 Engine Online! Phase: {CURRENT_PHASE} on {device}")

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
        
        LR_BACKBONE, LR_CLASSIFIER = 1e-6, 1e-5 
        
        # ⚠️ Yahan apni 0.67 wali file ka Asal Path check kar lein
        PREV_MODEL_PATH = "/kaggle/input/datasets/abdullahpy/audioweights/audio_phase1.pth" 
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase1_v2.pth"
    else: return

    valid_real = [d for d in REAL_DIRS if os.path.exists(d)]
    valid_fake = [d for d in FAKE_DIRS if os.path.exists(d)]

    fake_ds = DeepGuardDataset(real_dirs=[], fake_dirs=valid_fake, max_samples=None, mode="audio_only")
    num_fake = len(fake_ds)
    if num_fake == 0: return

    real_ds = DeepGuardDataset(real_dirs=valid_real, fake_dirs=[], max_samples=num_fake, mode="audio_only")
    full_dataset = ConcatDataset([real_ds, fake_ds])
    
    # 80/20 Validation Split
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    print(f"⚖️ Dataset Split: Train ({train_size}) | Validation ({val_size})")
    
    # P100 ke liye Batch Size 8 bilkul perfect hai
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    # 🚀 DATA-PARALLEL REMOVED (Strictly Single GPU)
    model = AudioOnlyDeepGuard().float().to(device)

    # Load Previous Backbone
    if PREV_MODEL_PATH and os.path.exists(PREV_MODEL_PATH):
        try:
            # Single GPU loading (No .module needed)
            model.expert.load_state_dict(torch.load(PREV_MODEL_PATH, map_location=device), strict=False)
            print(f"✅ Loaded Previous Backbone from {PREV_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Memory load error: {e}")

    # ==========================================
    # 🔓 Mid-Level Unfreezing (Layers 6 to 11)
    # ==========================================
    print("🔓 Unfreezing Layers 6 to 11 for Mid-Level deepfake features...")
    for name, param in model.named_parameters():
        if any(f"encoder.layers.{i}" in name for i in range(6, 12)) or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Optimizer (No .module needed)
    optimizer = optim.AdamW([
        {'params': model.expert.parameters(), 'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_CLASSIFIER}
    ], weight_decay=0.01)

    criterion = nn.BCEWithLogitsLoss()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    EPOCHS = 5 
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_f1 = 0.0

    # ==========================================
    # 🔥 TRAINING LOOP
    # ==========================================
    for epoch in range(EPOCHS):
        model.train()
        train_loop = tqdm(train_loader, total=len(train_loader), leave=True)
        
        for batch_idx, (_, _, _, audio, labels) in enumerate(train_loop):
            if torch.isnan(audio).any(): continue

            audio, labels = audio.to(device), labels.to(device).view(-1, 1)
            
            # Sample-level Normalization
            audio = (audio - audio.mean(dim=-1, keepdim=True)) / (audio.std(dim=-1, keepdim=True) + 1e-8)
            audio = torch.clamp(audio, -1.0, 1.0)

            # Bulletproof Augmentation & Contiguous Memory Fix
            audio = augment_waveform(audio)
            audio = audio.contiguous() # 👈 YEH CRASH SE BACHAYEGA

            optimizer.zero_grad()
            
            # Forward Pass
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
        # 🧪 VALIDATION LOOP
        # ==========================================
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for _, _, _, audio, labels in val_loader:
                audio, labels = audio.to(device), labels.to(device).view(-1, 1)
                
                # Normalization (No augmentation in Validation)
                audio = (audio - audio.mean(dim=-1, keepdim=True)) / (audio.std(dim=-1, keepdim=True) + 1e-8)
                audio = torch.clamp(audio, -1.0, 1.0)
                audio = audio.contiguous()

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

        # Save Full Model State (Expert + Classifier)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH) # Single GPU save format
            print(f"   🌟 New Best Model Saved! F1: {val_f1:.4f}")

    print(f"\n✅ Training Complete! Best Val F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    train_audio_model()
