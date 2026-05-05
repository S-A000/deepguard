"""
DeepGuard - Audio Expert Training Script  (v2.1 — Path Fix & Kaggle Optimized)
=============================================================================
Strategy:
  Phase 1 → Audio-only datasets (wavefake, Speech Dataset, VCTK)
  Phase 2 → DFDC video datasets (audio extracted from videos)

Fixes applied:
  [FIX 1] AMP removed for Wav2Vec2 stability
  [FIX 2] Per-sample audio normalization
  [FIX 3] NaN/Inf guards on audio, logits, and loss
  [FIX 4] BCEWithLogitsLoss pos_weight=1.5
  [FIX 5] Decision threshold = 0.4
  [FIX 6] Phase-aware layer freezing
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import numpy as np
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# 🛠️ SYSTEM PATH INJECTION (The "ModuleNotFoundError" Fix)
# ══════════════════════════════════════════════════════════════════════════════

# Script ki apni location maloom karein
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Project ki main root (deepguard folder) tak pahuchein
# Agar aap backend/core_ai mein hain, toh 3 levels oopar root hai
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✅ Project Root Added to System Path: {project_root}")

# Ab Dataset loader ko import karein
try:
    from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset
    print("✅ Successfully imported DeepGuardDataset from custom_datasets")
except ImportError:
    try:
        from dataset import DeepGuardDataset
        print("✅ Successfully imported DeepGuardDataset from local dataset.py")
    except ImportError:
        print("❌ CRITICAL ERROR: Could not find DeepGuardDataset. Check your folder structure!")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

class AudioExpert(nn.Module):
    def __init__(self, embed_dim: int = 256, phase: int = 1):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        # FIX 6: Phase-aware layer freezing
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False

        if phase == 1:
            for param in self.wav2vec.encoder.parameters():
                param.requires_grad = False

            total_layers  = len(self.wav2vec.encoder.layers)
            unfreeze_from = total_layers - 4 
            for i in range(unfreeze_from, total_layers):
                for param in self.wav2vec.encoder.layers[i].parameters():
                    param.requires_grad = True

            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  🧊 Phase 1: Last 4/{total_layers} transformer layers trainable")
        else:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  🔥 Phase 2: All transformer layers trainable")

        self.fc = nn.Linear(self.wav2vec.config.hidden_size, embed_dim)

    def forward(self, audio_waveforms: torch.Tensor) -> torch.Tensor:
        outputs = self.wav2vec(audio_waveforms)
        pooled  = torch.mean(outputs.last_hidden_state, dim=1)
        return self.fc(pooled)


class AudioClassifier(nn.Module):
    def __init__(self, embed_dim: int = 256, phase: int = 1):
        super().__init__()
        self.expert = AudioExpert(embed_dim=embed_dim, phase=phase)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, audio_waveforms: torch.Tensor) -> torch.Tensor:
        emb    = self.expert(audio_waveforms)
        logits = self.classifier(emb)
        return logits.squeeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET PHASE CONFIGS (KAGGLE UPDATED PATHS)
# ══════════════════════════════════════════════════════════════════════════════

PHASE_CONFIGS = {
    1: {
        "name": "Audio-Only Pre-Training",
        "description": "Pure audio datasets — teaches real vs fake speech",
        "real_dirs": [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Real/Real",
            "/kaggle/input/datasets/kynthesis/vctk-corpus/VCTK-Corpus/wav48"
        ],
        "fake_dirs": [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Fake/Fake",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_melgan",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_parallel_wavegan"
        ],
        "max_samples": 20000,
        "epochs": 10,
        "lr": 2e-5,
    },
    2: {
        "name": "DFDC Audio Fine-Tuning",
        "description": "Audio from deepfake videos — domain shift training",
        "real_dirs": [
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real"
        ],
        "fake_dirs": [
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake"
        ],
        "max_samples": 30000,
        "epochs": 7,
        "lr": 5e-6,
    },
}

DECISION_THRESHOLD = 0.4


# ══════════════════════════════════════════════════════════════════════════════
# 3.  AUDIO NORMALIZATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def normalize_audio(audio: torch.Tensor) -> torch.Tensor:
    mean  = audio.mean(dim=-1, keepdim=True)
    std   = audio.std(dim=-1, keepdim=True)
    std   = torch.where(std < 1e-6, torch.ones_like(std), std)
    audio = (audio - mean) / std
    audio = torch.clamp(audio, -3.0, 3.0)
    return audio


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING + VALIDATION LOOPS
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    skipped    = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  Training", leave=False):
        # audio_only mode → (_, _, _, audio, labels)
        _, _, _, audio, labels = batch

        bad_mask   = torch.isnan(audio).any(dim=1) | torch.isinf(audio).any(dim=1)
        valid_mask = ~bad_mask
        if valid_mask.sum() == 0:
            skipped += len(audio)
            continue

        audio  = audio[valid_mask].to(device)
        labels = labels[valid_mask].to(device)
        audio  = normalize_audio(audio)

        optimizer.zero_grad()
        logits = model(audio)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            skipped += len(audio)
            continue

        loss = criterion(logits, labels)
        if torch.isnan(loss) or torch.isinf(loss):
            skipped += len(audio)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    binary_preds = (np.array(all_preds) > DECISION_THRESHOLD).astype(int)
    acc = (binary_preds == np.array(all_labels)).mean()
    return avg_loss, acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    skipped    = 0
    all_probs, all_labels = [], []

    for batch in tqdm(loader, desc="  Validating", leave=False):
        _, _, _, audio, labels = batch
        bad_mask   = torch.isnan(audio).any(dim=1) | torch.isinf(audio).any(dim=1)
        valid_mask = ~bad_mask
        if valid_mask.sum() == 0:
            skipped += len(audio)
            continue

        audio  = audio[valid_mask].to(device)
        labels = labels[valid_mask].to(device)
        audio  = normalize_audio(audio)

        logits = model(audio)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            skipped += len(audio)
            continue

        loss = criterion(logits, labels)
        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss   = total_loss / max(len(loader), 1)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    binary_preds = (all_probs > DECISION_THRESHOLD).astype(int)
    acc = (binary_preds == all_labels).mean()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    return avg_loss, acc, auc, f1, all_labels, binary_preds


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train_phase(phase: int, checkpoint_path: str = None):
    cfg    = PHASE_CONFIGS[phase]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  DeepGuard Audio Training — Phase {phase}: {cfg['name']}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # Filter paths that actually exist
    real_paths = [d for d in cfg["real_dirs"] if os.path.exists(d)]
    fake_paths = [d for d in cfg["fake_dirs"] if os.path.exists(d)]

    full_dataset = DeepGuardDataset(
        real_dirs   = real_paths,
        fake_dirs   = fake_paths,
        num_frames  = 16,
        max_samples = cfg["max_samples"],
        mode        = "audio_only",
    )

    val_size   = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = AudioClassifier(embed_dim=256, phase=phase).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"⚡ Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-7)

    # FIX 4: Class imbalance guard
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))

    best_auc  = 0.0
    save_path = f"/kaggle/working/audio_expert_phase{phase}.pth"

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"Epoch [{epoch}/{cfg['epochs']}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc, val_f1, true_labels, pred_labels = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train → Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   → Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_auc": val_auc, "phase": phase}, save_path)
            print(f"  ✅ Best model saved → {save_path}\n")

    return save_path


if __name__ == "__main__":
    # ⚠️ MANUALLY SET PHASE HERE FOR KAGGLE
    PHASE_TO_RUN = 1 
    
    ckpt_in = None
    if PHASE_TO_RUN == 2:
        if os.path.exists("/kaggle/working/audio_expert_phase1.pth"):
            ckpt_in = "/kaggle/working/audio_expert_phase1.pth"

    train_phase(phase=PHASE_TO_RUN, checkpoint_path=ckpt_in)
