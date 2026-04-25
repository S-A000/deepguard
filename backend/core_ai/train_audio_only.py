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
# Phase badalney ke liye sirf is number ko change karein (1, 2, 3, or 4)
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
    # 🗺️ 4-PHASE DATASET ROUTING (FULL DYNAMIC)
    # ==========================================
    if CURRENT_PHASE == 1:
        print("🟢 PHASE 1: WARM-UP (Basic AI Voices)")
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
        PREV_MODEL_PATH = None
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase1.pth"

    elif CURRENT_PHASE == 2:
        print("🟡 PHASE 2: ADVANCED AUDIO (Multi-band MelGAN)")
        REAL_DIRS = ["/kaggle/input/speech-dataset-of-human-and-ai-generated-voices/Real/Real"]
        FAKE_DIRS = ["/kaggle/input/wavefake-test/generated_audio/ljspeech_multi_band_melgan"]
        LR_BACKBONE, LR_CLASSIFIER = 1e-6, 1e-5
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase1.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase2.pth"

    elif CURRENT_PHASE == 3:
        print("🟠 PHASE 3: CROSS-DOMAIN (FF++ & DFDC Video Audio)")
        REAL_DIRS = ["/kaggle/input/faceforensics/FF++/real"]
        FAKE_DIRS = [
            "/kaggle/input/faceforensics/FF++/fake",
            "/kaggle/input/dfdc-part-14/fake"
        ]
        LR_BACKBONE, LR_CLASSIFIER = 5e-7, 5e-6 # 🚀 Lower LR for fine-tuning
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase2.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase3.pth"

    elif CURRENT_PHASE == 4:
        print("🔴 PHASE 4: FUTURE THREATS (Custom & SoraGen)")
        REAL_DIRS = ["/kaggle/input/faceforensics/FF++/real"]
        FAKE_DIRS = ["/kaggle/input/soragenvid/fake"]
        LR_BACKBONE, LR_CLASSIFIER = 1e-7, 1e-6 # 🚀 Ultra-low LR
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase3.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_FINAL.pth"

    else:
        print("❌ Invalid Phase!")
        return

    # Path Verification
    valid_real = [d for d in REAL_DIRS if os.path.exists(d)]
    valid_fake = [d for d in FAKE_DIRS if os.path.exists(d)]
    print(f"✅ Found Folders - Real: {len(valid_real)} | Fake: {len(valid_fake)}")

    # ==========================================
    # 🧠 BALANCED LOADING (UNDERSAMPLING)
    # ==========================================
    # First, load all available fakes for this phase
    fake_ds = DeepGuardDataset(real_dirs=[], fake_dirs=valid_fake, max_samples=None, mode="audio_only")
    num_fake = len(fake_ds)

    if num_fake == 0:
        print("❌ No fake files found in this phase!")
        return

    # 🚀 Undersample Real data to match Fake count
    real_ds = DeepGuardDataset(real_dirs=valid_real, fake_dirs=[], max_samples=num_fake, mode="audio_only")

    print(f"⚖️ Final Balance: Real ({len(real_ds)}) | Fake ({len(fake_ds)})")
    dataloader = DataLoader(ConcatDataset([real_ds, fake_ds]), batch_size=8, shuffle=True, num_workers=2)

    # ==========================================
    # 🧠 MODEL & MEMORY LOADING
    # ==========================================
    model = AudioOnlyDeepGuard().float().to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    # Load Previous Phase Memory
    if PREV_MODEL_PATH and os.path.exists(PREV_MODEL_PATH):
        try:
            target = model.module.expert if isinstance(model, nn.DataParallel) else model.expert
            target.load_state_dict(torch.load(PREV_MODEL_PATH, map_location=device))
            print(f"✅ Memory Loaded from previous phase!")
        except Exception as e:
            print(f"⚠️ Memory load error (Architecture might differ): {e}")

    # Freeze Backbone CNN layers for stability
    for name, param in model.named_parameters():
        if "feature_extractor" in name or "conv" in name:
            param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': model.module.expert.parameters() if isinstance(model, nn.DataParallel) else model.expert.parameters(), 'lr': LR_BACKBONE},
        {'params': model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters(), 'lr': LR_CLASSIFIER}
    ], weight_decay=0.01)

    criterion = nn.BCEWithLogitsLoss()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # ==========================================
    # 🔥 STABLE SOTA TRAINING LOOP
    # ==========================================
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        for batch_idx, (_, _, _, audio, labels) in enumerate(loop):
            if torch.isnan(audio).any(): continue

            audio, labels = audio.to(device), labels.to(device).view(-1, 1)
            
            # Normalization
            audio = (audio - audio.mean()) / (audio.std() + 1e-8)
            audio = torch.clamp(audio, -1.0, 1.0)

            optimizer.zero_grad()
            preds = model(audio)
            
            loss = criterion(preds, labels)
            if torch.isnan(loss): continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1) # 🛡️ NAN PREVENTION
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(BCE=f"{loss.item():.4f}")

    # Final Save
    torch.save(model.module.expert.state_dict() if isinstance(model, nn.DataParallel) else model.expert.state_dict(), SAVE_PATH)
    print(f"\n✅ Phase {CURRENT_PHASE} Complete! Saved: {SAVE_PATH}")

if __name__ == "__main__":
    train_audio_model()
