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
    print(f"\n🎧 Audio-Only System Online! Training on: {device}")

    # ==========================================
    # 🗺️ 4-PHASE DATASET ROUTING (Wahi original structure!)
    # ==========================================
    if CURRENT_PHASE == 1:
        print("🟢 PHASE 1: WARM-UP (Pure Audio - Basic AI Voices)")
        REAL_DIRS = [
            "/kaggle/input/speech-dataset-of-human-and-ai-generated-voices/Real/Real",
            "/kaggle/input/vctk-corpus/VCTK-Corpus/wav48"
        ]
        # 🚀 Path Fix: Removed 'datasets/' and added exact Kaggle slugs
        FAKE_DIRS = [
            "/kaggle/input/speech-dataset-of-human-and-ai-generated-voices/Fake/Fake",
            "/kaggle/input/wavefake-test/generated_audio/ljspeech_melgan",
            "/kaggle/input/wavefake-test/generated_audio/ljspeech_parallel_wavegan"
        ]
        LR_BACKBONE = 0.000001 # Micro LR for Backbone (NaN protection)
        LR_CLASSIFIER = 0.00005 
        PREV_MODEL_PATH = None
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase1.pth"

    elif CURRENT_PHASE == 2:
        print("🟡 PHASE 2: ADVANCED AUDIO (Adding WaveFake Hard Architectures)")
        REAL_DIRS = ["/kaggle/input/speech-dataset-of-human-and-ai-generated-voices/Real/Real"]
        FAKE_DIRS = ["/kaggle/input/wavefake-test/generated_audio/ljspeech_multi_band_melgan"]
        LR_BACKBONE = 0.000001
        LR_CLASSIFIER = 0.00005
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase1.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase2.pth"

    elif CURRENT_PHASE == 3:
        print("🟠 PHASE 3: CROSS-DOMAIN FAKES (FF++ & DFDC)")
        REAL_DIRS = ["/kaggle/input/faceforensics/FF++/real"]
        FAKE_DIRS = ["/kaggle/input/dfdc-part-14/fake"]
        LR_BACKBONE = 0.000001
        LR_CLASSIFIER = 0.00001
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase2.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase3.pth"

    elif CURRENT_PHASE == 4:
        print("🔴 PHASE 4: THE FUTURE THREATS")
        REAL_DIRS = ["/kaggle/input/faceforensics/FF++/real"]
        FAKE_DIRS = ["/kaggle/input/soragenvid/fake"]
        LR_BACKBONE = 0.000001
        LR_CLASSIFIER = 0.00001
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase3.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_FINAL.pth"

    else:
        print("❌ Invalid Phase Selected!")
        return

    # Path check (Kaggle slug verification)
    valid_real = [d for d in REAL_DIRS if os.path.exists(d)]
    valid_fake = [d for d in FAKE_DIRS if os.path.exists(d)]
    print(f"✅ Active Folders - Real: {len(valid_real)} | Fake: {len(valid_fake)}")

    # ==========================================
    # 🧠 DATASET LOADING (No Limit)
    # ==========================================
    # 🚀 FIX: SAMPLES_PER_CLASS hatadiye taake saari files load hon
    real_ds = DeepGuardDataset(real_dirs=valid_real, fake_dirs=[], max_samples=None, mode="audio_only")
    fake_ds = DeepGuardDataset(real_dirs=[], fake_dirs=valid_fake, max_samples=None, mode="audio_only")
    
    # Balance check: Agar fakes ab bhi kam hain toh repeat karo
    if len(fake_ds) < len(real_ds) and len(fake_ds) > 0:
        multiplier = (len(real_ds) // len(fake_ds)) + 1
        fake_ds = torch.utils.data.ConcatDataset([fake_ds] * multiplier)
        print(f"🔄 Oversampling: Fake data repeated {multiplier}x.")

    dataloader = DataLoader(ConcatDataset([real_ds, fake_ds]), batch_size=16, shuffle=True, num_workers=2)

    model = AudioOnlyDeepGuard().float().to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    # Differential Optimizer: Backbone 🐢, Classifier 🐆
    optimizer = optim.AdamW([
        {'params': model.module.expert.parameters() if isinstance(model, nn.DataParallel) else model.expert.parameters(), 'lr': LR_BACKBONE},
        {'params': model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters(), 'lr': LR_CLASSIFIER}
    ])

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # ==========================================
    # 🔥 SOTA TRAINING LOOP
    # ==========================================
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        for batch_idx, (_, _, _, audio, labels) in enumerate(loop):
            if torch.isnan(audio).any(): continue

            audio, labels = audio.to(device), labels.to(device).view(-1, 1)
            audio = torch.clamp(audio, -1.0, 1.0)

            optimizer.zero_grad()
            preds = model(audio)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(BCE=f"{loss.item():.4f}")

    torch.save(model.module.expert.state_dict() if isinstance(model, nn.DataParallel) else model.expert.state_dict(), SAVE_PATH)
    print(f"\n✅ Phase {CURRENT_PHASE} Complete! Saved at: {SAVE_PATH}")

if __name__ == "__main__":
    train_audio_model()
