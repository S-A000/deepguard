import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score

# System Performance & Safety Configurations
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_num_threads(4) 

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from backend.core_ai.models.branch_d_audio import AudioExpert
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset

# ==========================================
# 🎛️ PHASE CONTROLLER (MASTER SWITCH)
# ==========================================
# 1: Scratch Base | 2: Advanced Audio | 3: Cross-Domain | 4: Future Threats
CURRENT_PHASE = 1  

def augment_waveform(audio):
    """SOTA Augmentation for robust learning"""
    if torch.rand(1).item() > 0.5:
        time_len = audio.shape[-1]
        if time_len > 500:
            max_t = max(1, time_len // 8)
            t = torch.randint(1, max_t + 1, (1,)).item()
            t0 = torch.randint(0, max(1, time_len - t), (1,)).item()
            audio[..., t0:t0+t] = 0 
    return audio

class AudioOnlyDeepGuard(nn.Module):
    """Standard DeepGuard GELU Architecture"""
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
    
    # ==========================================================
    # 🗺️ DYNAMIC PHASE ROUTING (THE BRAIN OF THE SCRIPT)
    # ==========================================================
    
    # Common Master Real Base (Har phase mein Real data ka balance hona zaroori hai)
    MASTER_REAL = [
        "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Real/Real",
        "/kaggle/input/datasets/kynthesis/vctk-corpus/VCTK-Corpus/wav48"
    ]

    if CURRENT_PHASE == 1:
        print("🟢 PHASE 1: BUILDING MASTER BASE (Scratch Training)")
        REAL_DIRS = MASTER_REAL
        FAKE_DIRS = [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Fake/Fake",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_melgan",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_parallel_wavegan"
        ]
        LR_BACKBONE, LR_CLASSIFIER = 2e-5, 2e-4
        PREV_MODEL_PATH = None
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase1.pth"
        WARMUP_EPOCHS = 2

    elif CURRENT_PHASE == 2:
        print("🟡 PHASE 2: ADVANCED AUDIO (HiFi-GAN & WaveGlow)")
        REAL_DIRS = MASTER_REAL
        FAKE_DIRS = [
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_hifiGAN",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/jsut_parallel_wavegan",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_waveglow"
        ]
        LR_BACKBONE, LR_CLASSIFIER = 5e-6, 5e-5 # Lower LR for retention
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase1.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase2.pth"
        WARMUP_EPOCHS = 1

    elif CURRENT_PHASE == 3:
        print("🟠 PHASE 3: CROSS-DOMAIN (Video-extracted Audio)")
        REAL_DIRS = ["/kaggle/input/faceforensics/FF++/real"] + MASTER_REAL
        FAKE_DIRS = [
            "/kaggle/input/faceforensics/FF++/fake",
            "/kaggle/input/dfdc-part-14/fake"
        ]
        LR_BACKBONE, LR_CLASSIFIER = 2e-6, 2e-5
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase2.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase3.pth"
        WARMUP_EPOCHS = 1

    elif CURRENT_PHASE == 4:
        print("🔴 PHASE 4: FUTURE THREATS (SoraGen & Custom)")
        REAL_DIRS = MASTER_REAL
        FAKE_DIRS = ["/kaggle/input/soragenvid/fake"]
        LR_BACKBONE, LR_CLASSIFIER = 1e-6, 1e-5
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase3.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_FINAL.pth"
        WARMUP_EPOCHS = 1

    else:
        print("❌ Invalid Phase Selection!")
        return

    # ==========================================
    # 🧠 DATA & MODEL INITIALIZATION
    # ==========================================
    valid_real = [d for d in REAL_DIRS if os.path.exists(d)]
    valid_fake = [d for d in FAKE_DIRS if os.path.exists(d)]
    
    fake_ds = DeepGuardDataset(real_dirs=[], fake_dirs=valid_fake, max_samples=None, mode="audio_only")
    num_fake = len(fake_ds)
    real_ds = DeepGuardDataset(real_dirs=valid_real, fake_dirs=[], max_samples=num_fake, mode="audio_only")
    
    full_dataset = ConcatDataset([real_ds, fake_ds])
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    model = AudioOnlyDeepGuard().float().to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    target_model = model.module if isinstance(model, nn.DataParallel) else model

    # Load Memory from Previous Phase
    if PREV_MODEL_PATH and os.path.exists(PREV_MODEL_PATH):
        try:
            target_model.load_state_dict(torch.load(PREV_MODEL_PATH, map_location=device), strict=False)
            print(f"✅ Memory Loaded from: {PREV_MODEL_PATH}")
        except:
            print("⚠️ Memory loading failed, starting fresh for this phase.")

    optimizer = optim.AdamW([
        {'params': target_model.expert.parameters(), 'lr': LR_BACKBONE},
        {'params': target_model.classifier.parameters(), 'lr': LR_CLASSIFIER}
    ], weight_decay=0.02)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    best_val_f1 = 0.0
    EPOCHS = 10

    # ==========================================
    # 🔥 STABLE TRAINING LOOP
    # ==========================================
    for epoch in range(EPOCHS):
        
        # Protective Warm-up Strategy
        if epoch < WARMUP_EPOCHS:
            print(f"\n🥶 Warming up GELU Head...")
            for name, param in target_model.named_parameters():
                param.requires_grad = True if "classifier" in name else False
        else:
            print(f"\n🔓 Fine-tuning Full Model (Retention Mode)...")
            for name, param in target_model.named_parameters():
                if any(f"encoder.layers.{i}" in name for i in range(4, 12)) or "classifier" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        model.train()
        train_loop = tqdm(train_loader, total=len(train_loader), leave=True)
        
        for _, _, _, audio, labels in train_loop:
            audio = audio.detach().cpu()
            audio = (audio - audio.mean(dim=-1, keepdim=True)) / (audio.std(dim=-1, keepdim=True) + 1e-8)
            audio = torch.clamp(audio, -1.0, 1.0)
            audio = augment_waveform(audio.clone())
            
            audio = audio.contiguous().to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(audio)
                loss = criterion(preds, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            
            train_loop.set_description(f"Phase {CURRENT_PHASE} | Epoch [{epoch+1}/{EPOCHS}]")
            train_loop.set_postfix(Loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for _, _, _, audio, labels in val_loader:
                audio = audio.detach().cpu()
                audio = (audio - audio.mean(dim=-1, keepdim=True)) / (audio.std(dim=-1, keepdim=True) + 1e-8)
                audio = torch.clamp(audio, -1.0, 1.0).to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(audio)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        val_f1 = f1_score(all_labels, all_preds)
        print(f"📊 Validation F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(target_model.state_dict(), SAVE_PATH)
            print(f"🌟 Phase {CURRENT_PHASE} Model Updated!")

    print(f"\n✅ Phase {CURRENT_PHASE} Complete! Best F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    train_audio_model()
