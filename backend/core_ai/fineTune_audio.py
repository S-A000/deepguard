import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

from backend.core_ai.models.branch_d_audio import AudioExpert
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset

# ==========================================
# 🎛️ PHASE CONTROLLER (MASTER SWITCH)
# ==========================================
CURRENT_PHASE = 1  

class AudioOnlyDeepGuard(nn.Module):
    def __init__(self, embed_dim=256):
        super(AudioOnlyDeepGuard, self).__init__()
        self.expert = AudioExpert(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, audio_waveforms):
        features = self.expert(audio_waveforms)
        return self.classifier(features)

def train_audio_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎧 Audio-Only System Online! Training Phase: {CURRENT_PHASE} on {device}")

    # ==========================================
    # 🗺️ 4-PHASE DATASET ROUTING
    # ==========================================
    if CURRENT_PHASE == 1:
        print("🟢 PHASE 1: RECOVERY & SELECTIVE FINE-TUNING")
        REAL_DIRS = [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Real/Real",
            "/kaggle/input/datasets/kynthesis/vctk-corpus/VCTK-Corpus/wav48"
        ]
        FAKE_DIRS = [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Fake/Fake",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_parallel_wavegan",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_melgan"
        ]
        # 🚀 Path Fix: Points to your uploaded Kaggle Dataset
        # Note: Change 'deepguard-weights-v1' to your actual dataset name
        PREV_MODEL_PATH = "/kaggle/input/datasets/abdullahpy/audioweights/audio_phase1.pth" 
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase1.pth"
        
        # 🐢 Ultra-Slow LRs for selective fine-tuning
        LR_BACKBONE, LR_CLASSIFIER = 1e-7, 1e-6 

    # ... (Other phases stay the same structure)

    valid_real = [d for d in REAL_DIRS if os.path.exists(d)]
    valid_fake = [d for d in FAKE_DIRS if os.path.exists(d)]
    print(f"✅ Found Folders - Real: {len(valid_real)} | Fake: {len(valid_fake)}")

    fake_ds = DeepGuardDataset(real_dirs=[], fake_dirs=valid_fake, max_samples=None, mode="audio_only")
    num_fake = len(fake_ds)
    real_ds = DeepGuardDataset(real_dirs=valid_real, fake_dirs=[], max_samples=num_fake, mode="audio_only")

    print(f"⚖️ Final Balance: Real ({len(real_ds)}) | Fake ({len(fake_ds)})")
    dataloader = DataLoader(ConcatDataset([real_ds, fake_ds]), batch_size=8, shuffle=True, num_workers=2)

    model = AudioOnlyDeepGuard().float().to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    # ==========================================
    # 🧠 RECOVERY LOADING
    # ==========================================
    if PREV_MODEL_PATH and os.path.exists(PREV_MODEL_PATH):
        try:
            target = model.module.expert if isinstance(model, nn.DataParallel) else model.expert
            target.load_state_dict(torch.load(PREV_MODEL_PATH, map_location=device))
            print(f"✅ 0.67 F1 Memory Recovered from Kaggle Input!")
        except Exception as e:
            print(f"⚠️ Memory load error: {e}")

    # ==========================================
    # 🔓 SELECTIVE UNFREEZE (Stability Fix)
    # ==========================================
    print("🔓 Unfreezing ONLY top Transformer layers (10 & 11)...")
    for name, param in model.named_parameters():
        # Sirf aakhri layers aur classifier ko train hone denge
        if "encoder.layers.11" in name or "encoder.layers.10" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': model.module.expert.parameters() if isinstance(model, nn.DataParallel) else model.expert.parameters(), 'lr': LR_BACKBONE},
        {'params': model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters(), 'lr': LR_CLASSIFIER}
    ], weight_decay=0.01)

    criterion = nn.BCEWithLogitsLoss()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # ==========================================
    # 🔥 STABLE TRAINING LOOP
    # ==========================================
    EPOCHS = 5 # Selective fine-tuning ke liye 5 epochs kafi hain
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        for batch_idx, (_, _, _, audio, labels) in enumerate(loop):
            if torch.isnan(audio).any(): continue

            audio, labels = audio.to(device), labels.to(device).view(-1, 1)
            audio = (audio - audio.mean()) / (audio.std() + 1e-8)
            audio = torch.clamp(audio, -1.0, 1.0)

            optimizer.zero_grad()
            preds = model(audio)
            
            if torch.isnan(preds).any(): continue
            loss = criterion(preds, labels)
            if torch.isnan(loss): continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1) 
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(BCE=f"{loss.item():.4f}")

    torch.save(model.module.expert.state_dict() if isinstance(model, nn.DataParallel) else model.expert.state_dict(), SAVE_PATH)
    print(f"\n✅ Phase {CURRENT_PHASE} Selective Fine-tuning Complete! Saved: {SAVE_PATH}")

if __name__ == "__main__":
    train_audio_model()
