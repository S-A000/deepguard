import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_num_threads(2) # CPU thread optimization

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

from backend.core_ai.models.branch_d_audio import AudioExpert
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset

# ==========================================
# 🎛️ PHASE CONTROLLER (MASTER SWITCH)
# ==========================================
# Phase badalney ke liye sirf is number ko change karein (1, 2, 3, or 4)
CURRENT_PHASE = 2 

# ✅ UPDATE 3: Restored augmentation function
def augment_waveform(audio):
    if torch.rand(1).item() > 0.5:
        time_len = audio.shape[-1]
        if time_len > 100: 
            max_t = max(1, time_len // 10)
            t = torch.randint(1, max_t + 1, (1,)).item()
            t0 = torch.randint(0, max(1, time_len - t), (1,)).item()
            audio[..., t0:t0+t] = 0 
    return audio

# ✅ UPDATE 2: Restored deeper classifier (GELU)
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
    print(f"\n🎧 Audio-Only System Online! Training Phase: {CURRENT_PHASE} on {device}")

    # ==========================================
    # 🗺️ 4-PHASE DATASET ROUTING (UNCHANGED STRUCTURE)
    # ==========================================
    if CURRENT_PHASE == 1:
        print("🟢 PHASE 1: WARM-UP (Basic AI Voices) - TRAINING FROM SCRATCH")
        REAL_DIRS = [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Real/Real",
            "/kaggle/input/datasets/kynthesis/vctk-corpus/VCTK-Corpus/wav48"
        ]
        FAKE_DIRS = [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Fake/Fake",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_melgan",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_parallel_wavegan"
        ]
        # 🚀 ZERO-STATE FIX: Increased LR for training from scratch
        LR_BACKBONE, LR_CLASSIFIER = 1e-5, 1e-4
        PREV_MODEL_PATH = None
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase1.pth"

    elif CURRENT_PHASE == 2:
        print("🟡 PHASE 2: ADVANCED AUDIO (Multi-band MelGAN)")
        REAL_DIRS = ["/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Real/Real"]
        FAKE_DIRS = ["/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_hifiGAN",
                     "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/jsut_parallel_wavegan",
                     "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_waveglow"]
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
        LR_BACKBONE, LR_CLASSIFIER = 5e-7, 5e-6 
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase2.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/audio_phase3.pth"

    elif CURRENT_PHASE == 4:
        print("🔴 PHASE 4: FUTURE THREATS (Custom & SoraGen)")
        REAL_DIRS = ["/kaggle/input/faceforensics/FF++/real"]
        FAKE_DIRS = ["/kaggle/input/soragenvid/fake"]
        LR_BACKBONE, LR_CLASSIFIER = 1e-7, 1e-6 
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
    fake_ds = DeepGuardDataset(real_dirs=[], fake_dirs=valid_fake, max_samples=None, mode="audio_only")
    num_fake = len(fake_ds)

    if num_fake == 0:
        print("❌ No fake files found in this phase!")
        return

    real_ds = DeepGuardDataset(real_dirs=valid_real, fake_dirs=[], max_samples=num_fake, mode="audio_only")
    full_dataset = ConcatDataset([real_ds, fake_ds])

    # ✅ UPDATE 4: Added Validation Split
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    print(f"⚖️ Dataset Split: Train ({train_size}) | Validation ({val_size})")
    
    # ✅ UPDATE: Increased batch size to 16 for better throughput with AMP
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    # ==========================================
    # 🧠 MODEL & MEMORY LOADING
    # ==========================================
    model = AudioOnlyDeepGuard().float().to(device)
    
    # Keeping DataParallel as requested
    if torch.cuda.device_count() > 1: 
        print(f"🔥 Dual-GPU Mode Activated! Utilizing {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    target_model = model.module if isinstance(model, nn.DataParallel) else model

    if PREV_MODEL_PATH and os.path.exists(PREV_MODEL_PATH):
        try:
            target_model.load_state_dict(torch.load(PREV_MODEL_PATH, map_location=device), strict=False)
            print(f"✅ Memory Loaded from previous phase!")
        except Exception as e:
            print(f"⚠️ Memory load error: {e}")

    optimizer = optim.AdamW([
        {'params': target_model.expert.parameters(), 'lr': LR_BACKBONE},
        {'params': target_model.classifier.parameters(), 'lr': LR_CLASSIFIER}
    ], weight_decay=0.01)

    criterion = nn.BCEWithLogitsLoss()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # ✅ NEW: AMP Scaler initialized
    scaler = torch.cuda.amp.GradScaler()
    best_val_f1 = 0.0
    EPOCHS = 10

    # ==========================================
    # 🔥 STABLE SOTA TRAINING LOOP
    # ==========================================
    for epoch in range(EPOCHS):
        
        # 🚀 ZERO-STATE FIX: 2-Epoch Head Warm-Up
        if epoch < 2:
            print(f"\n🥶 EPOCH {epoch+1}: WARM-UP PHASE - Freezing Backbone, Training Classifier Only...")
            for name, param in target_model.named_parameters():
                if "classifier" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            print(f"\n🔓 EPOCH {epoch+1}: WARM-UP COMPLETE - Unfreezing Layers 6 to 11...")
            for name, param in target_model.named_parameters():
                if any(f"encoder.layers.{i}" in name for i in range(6, 12)) or "classifier" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        model.train()
        train_loop = tqdm(train_loader, total=len(train_loader), leave=True)
        
        for batch_idx, (_, _, _, audio, labels) in enumerate(train_loop):
            if torch.isnan(audio).any(): continue

            # ✅ UPDATE 1: CPU Math & Normalization Fixes
            audio = audio.detach().cpu()
            audio = (audio - audio.mean(dim=-1, keepdim=True)) / (audio.std(dim=-1, keepdim=True) + 1e-8)
            audio = torch.clamp(audio, -1.0, 1.0)
            audio = augment_waveform(audio.clone())
            
            # Non-blocking transfer to GPU
            audio = audio.contiguous().to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad()
            
            # ✅ NEW: AMP Context for forward pass
            with torch.cuda.amp.autocast():
                preds = model(audio)
                loss = criterion(preds, labels)
            
            if torch.isnan(preds).any() or torch.isnan(loss): continue

            # ✅ NEW: AMP Backward pass and scaling
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}] Train")
            train_loop.set_postfix(Loss=f"{loss.item():.4f}")

        # ==========================================
        # 🧪 VALIDATION LOOP + F1 TRACKING
        # ==========================================
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for _, _, _, audio, labels in val_loader:
                
                audio = audio.detach().cpu()
                audio = (audio - audio.mean(dim=-1, keepdim=True)) / (audio.std(dim=-1, keepdim=True) + 1e-8)
                audio = torch.clamp(audio, -1.0, 1.0)
                
                audio = audio.contiguous().to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).view(-1, 1)

                # ✅ NEW: AMP in validation for speed
                with torch.cuda.amp.autocast():
                    outputs = model(audio)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds)
        
        print(f"📊 Epoch [{epoch+1}/{EPOCHS}] Stats: Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

        # ✅ UPDATE 5: Save FULL MODEL if F1 improves
        if val_f1 > best_val_f1 or epoch == 0: # Ensure at least one save
            best_val_f1 = val_f1
            save_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(save_state, SAVE_PATH)
            print(f"   🌟 New Best Model Saved! F1: {val_f1:.4f}")

    print(f"\n✅ Phase {CURRENT_PHASE} Complete! Best F1: {best_val_f1:.4f} Saved at: {SAVE_PATH}")

if __name__ == "__main__":
    train_audio_model()
