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
    # 🗺️ 4-PHASE DATASET ROUTING
    # ==========================================
    if CURRENT_PHASE == 1:
        print("🟢 PHASE 1: WARM-UP (Pure Audio - Basic AI Voices)")
        REAL_DIRS = [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Real/Real",
            "/kaggle/input/datasets/kynthesis/vctk-corpus/VCTK-Corpus/wav48"
        ]
        FAKE_DIRS = [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Fake/Fake"
        ]
        LR = 0.00005
        PREV_MODEL_PATH = None
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase1.pth"

    elif CURRENT_PHASE == 2:
        print("🟡 PHASE 2: ADVANCED AUDIO (Adding WaveFake Hard Architectures)")
        REAL_DIRS = [
            "/kaggle/input/speech-dataset-of-human-and-ai-generated/Real/Real",
            "/kaggle/input/vctk-corpus-version-0-92/VCTK-Corpus/wav48"
        ]
        FAKE_DIRS = [
            "/kaggle/input/speech-dataset-of-human-and-ai-generated/Fake/Fake",
            "/kaggle/input/wavefake/generated_audio/common_voices_prompts_from_conform/generated"
        ]
        LR = 0.00005
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase1.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase2.pth"

    elif CURRENT_PHASE == 3:
        print("🟠 PHASE 3: CROSS-DOMAIN FAKES (Adding Video Datasets - FF++ & DFDC)")
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

    valid_real = [d for d in REAL_DIRS if os.path.exists(d)]
    valid_fake = [d for d in FAKE_DIRS if os.path.exists(d)]
    
    if len(valid_real) == 0 or len(valid_fake) == 0:
        print("\n⚠️ ALERT: Kaggle URL Slug Mismatch Detected!")
        return

    print(f"✅ Active Folders - Real: {len(valid_real)} | Fake: {len(valid_fake)}")

    # ==========================================
    # 🧠 DATASET LOADING
    # ==========================================
    SAMPLES_PER_CLASS = 1000 
    real_dataset = DeepGuardDataset(real_dirs=valid_real, fake_dirs=[], max_samples=SAMPLES_PER_CLASS, mode="audio_only")
    fake_dataset = DeepGuardDataset(real_dirs=[], fake_dirs=valid_fake, max_samples=SAMPLES_PER_CLASS, mode="audio_only")
    
    balanced_dataset = ConcatDataset([real_dataset, fake_dataset])
    dataloader = DataLoader(balanced_dataset, batch_size=16, shuffle=True, num_workers=2)

    model = AudioOnlyDeepGuard().float().to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    if PREV_MODEL_PATH and os.path.exists(PREV_MODEL_PATH):
        try:
            target = model.module.expert if isinstance(model, nn.DataParallel) else model.expert
            target.load_state_dict(torch.load(PREV_MODEL_PATH, map_location=device))
            print(f"✅ Memory Loaded from Phase {CURRENT_PHASE - 1}!")
        except Exception as e:
            print(f"⚠️ Error loading memory: {e}")

    # ==========================================
    # ❄️ THE SOTA FREEZER
    # ==========================================
    print("\n❄️ Applying Fix Stack: Freezing Wav2Vec2 Feature Extractor...")
    frozen_layers = 0
    for name, param in model.named_parameters():
        if "feature_extractor" in name or "feature_projection" in name or "conv" in name:
            param.requires_grad = False
            frozen_layers += 1
    print(f"✅ Locked {frozen_layers} layers!\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # ==========================================
    # 🔥 SOTA TRAINING LOOP
    # ==========================================
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        for batch_idx, (_, _, _, audio, labels) in enumerate(loop):
            
            # 🚀 FIX: Removed the redundant `if audio.shape[-1] != 16000: continue` 
            
            # 🛡️ GATEKEEPER: Purity Check ONLY
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                continue

            audio = audio.float().to(device)
            labels = labels.float().to(device).view(-1, 1)

            # 🛡️ THE ONLY AUDIO FIX: Strict Clamping
            audio = torch.clamp(audio, min=-1.0, max=1.0)

            optimizer.zero_grad()
            predictions = model(audio)
            
            # Final output safety
            if torch.isnan(predictions).any():
                continue
                
            loss = criterion(predictions, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(BCE=f"{loss.item():.4f}")

    torch.save(model.module.expert.state_dict() if isinstance(model, nn.DataParallel) else model.expert.state_dict(), SAVE_PATH)
    print(f"\n✅ Phase {CURRENT_PHASE} Complete! Model Saved at: {SAVE_PATH}")

if __name__ == "__main__":
    train_audio_model()