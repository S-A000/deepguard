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
    # 🗺️ 4-PHASE DATASET ROUTING (AUDIO SOTA STRATEGY)
    # ==========================================
    if CURRENT_PHASE == 1:
        print("🟢 PHASE 1: WARM-UP (Pure Audio - Basic AI Voices)")
        # Phase 1 mein hum thora aasaan aur clean data denge
        REAL_DIRS = [
            "/kaggle/input/speech-dataset-of-human-and-ai-generated/Real",
            "/kaggle/input/vctk-corpus-version-0-92/VCTK-Corpus/wav48"
        ]
        FAKE_DIRS = [
            "/kaggle/input/speech-dataset-of-human-and-ai-generated/Fake"
        ]
        LR = 0.0001
        PREV_MODEL_PATH = None
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase1.pth"

    elif CURRENT_PHASE == 2:
        print("🟡 PHASE 2: ADVANCED AUDIO (Adding WaveFake Hard Architectures)")
        # Phase 2 mein WaveFake ki mushkil AI aawazein add kar denge
        REAL_DIRS = [
            "/kaggle/input/speech-dataset-of-human-and-ai-generated/Real",
            "/kaggle/input/vctk-corpus-version-0-92/VCTK-Corpus/wav48"
        ]
        FAKE_DIRS = [
            "/kaggle/input/speech-dataset-of-human-and-ai-generated/Fake",
            "/kaggle/input/wavefake/generated_audio/common_voices_prompts_from_conform/generated"
        ]
        LR = 0.00005
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase1.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase2.pth"

    elif CURRENT_PHASE == 3:
        print("🟠 PHASE 3: CROSS-DOMAIN FAKES (Adding Video Datasets - FF++ & DFDC)")
        # Phase 3 mein woh video datasets ayenge jin mein audio hai
        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real"
        ]
        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
            "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake"
        ]
        LR = 0.00001
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase2.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase3.pth"

    elif CURRENT_PHASE == 4:
        print("🔴 PHASE 4: THE FUTURE THREATS (Custom + SoraGenVid Placeholder)")
        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/custom_dataset/real"
        ]
        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/soragenvid/fake"
        ]
        LR = 0.00001
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase3.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_FINAL.pth"

    else:
        print("❌ Invalid Phase Selected!")
        return

    REAL_DIRS = [d for d in REAL_DIRS if os.path.exists(d)]
    FAKE_DIRS = [d for d in FAKE_DIRS if os.path.exists(d)]
    print(f"✅ Active Folders - Real: {len(REAL_DIRS)} | Fake: {len(FAKE_DIRS)}")

    # ==========================================
    # 🧠 MODEL INITIALIZATION & LOADING
    # ==========================================
    SAMPLES_PER_CLASS = 1000 
    real_dataset = DeepGuardDataset(real_dirs=REAL_DIRS, fake_dirs=[], max_samples=SAMPLES_PER_CLASS)
    fake_dataset = DeepGuardDataset(real_dirs=[], fake_dirs=FAKE_DIRS, max_samples=SAMPLES_PER_CLASS)
    
    balanced_dataset = ConcatDataset([real_dataset, fake_dataset])
    # Batch size 16 is safe here because audio tensors are small
    dataloader = DataLoader(balanced_dataset, batch_size=16, shuffle=True, num_workers=2)

    model = AudioOnlyDeepGuard().float().to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    if PREV_MODEL_PATH and os.path.exists(PREV_MODEL_PATH):
        try:
            target = model.module.expert if isinstance(model, nn.DataParallel) else model.expert
            target.load_state_dict(torch.load(PREV_MODEL_PATH, map_location=device))
            print(f"✅ Memory Loaded from Phase {CURRENT_PHASE - 1}! Continuing evolution...")
        except Exception as e:
            print(f"⚠️ Error loading previous phase memory: {e}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # ==========================================
    # 🔥 PURE FLOAT32 TRAINING LOOP (ANTI-NaN)
    # ==========================================
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        for batch_idx, (video_rgb, flow, fft, audio, labels) in enumerate(loop):
            
            # 🚀 FLOAT32 FORCE: No Autocast/AMP used here to prevent Wav2Vec crashing
            audio = audio.float().to(device)
            labels = labels.float().to(device).view(-1, 1)

            # EXTRA SANITIZER: Fallback safety against corrupt librosa outputs
            audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

            optimizer.zero_grad()
            
            # Direct forward pass without autocast
            predictions = model(audio)
            
            if torch.isnan(predictions).any():
                print(f"⚠️ NaN detected at batch {batch_idx}! (Skipping batch to save SOTA model)")
                continue
                
            loss = criterion(predictions, labels)
            
            # Standard backward pass (No Scaler required for pure Float32)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(BCE=f"{loss.item():.4f}")

    torch.save(model.module.expert.state_dict() if isinstance(model, nn.DataParallel) else model.expert.state_dict(), SAVE_PATH)
    print(f"\n✅ Phase {CURRENT_PHASE} Complete! Model Saved at: {SAVE_PATH}")

if __name__ == "__main__":
    train_audio_model()