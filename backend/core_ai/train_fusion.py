# ==========================================
# 🔥 DEEPGUARD FUSION TRAINING SCRIPT
# ✅ Loads 4 trained expert models
# ✅ Freezes experts initially
# ✅ Trains only fusion attention + fusion classifier
# ✅ Uses Visual + Physics + Forensic + Audio embeddings
# ✅ Saves final fusion model only once at the end
# ==========================================

import os
import sys
import copy
import random
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ==========================================
# ✅ PROJECT ROOT SETUP
# ==========================================
PROJECT_ROOT = "/kaggle/working/deepguard"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("✅ Project root added:", PROJECT_ROOT)
print("✅ Current working directory:", os.getcwd())


# ==========================================
# ✅ IMPORTS
# ==========================================
from backend.core_ai.models.fusion_net import DeepGuardFusionModel
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset


# ==========================================
# ⚙️ CONFIG
# ==========================================
SEED = 42

EPOCHS = 10
BATCH_SIZE = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0
DECISION_THRESHOLD = 0.5

EMBED_DIM = 256
NUM_HEADS = 8

SAMPLES_PER_CLASS = 1000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==========================================
# 📁 EXPERT CHECKPOINT PATHS
# ==========================================
VISUAL_EXPERT_PATH = "/kaggle/input/models/abdullahpy/audiophase2/pytorch/default/1/visual_FINAL_expert.pth"
PHYSICS_EXPERT_PATH = "/kaggle/input/models/abdullahpy/audiophase2/pytorch/default/1/physics_FINAL_expert.pth"
FORENSIC_EXPERT_PATH = "/kaggle/input/models/abdullahpy/audiophase2/pytorch/default/1/forensic_FINAL.pth"

# Audio Phase 2 best tha, isi liye yeh use karna hai
AUDIO_EXPERT_PATH = "/kaggle/input/models/abdullahpy/audiophase2/pytorch/default/1/audio_phase2_expert.pth"


# ==========================================
# 💾 SAVE PATHS
# ==========================================
SAVE_DIR = "/kaggle/working/saved_models/production"
SAVE_FUSION_FULL_PATH = os.path.join(SAVE_DIR, "fusion_FINAL_full.pth")
SAVE_FUSION_BEST_PATH = os.path.join(SAVE_DIR, "fusion_FINAL_best.pth")


# ==========================================
# 📁 DATASET PATHS FOR FUSION TRAINING
# ==========================================
# Fusion training ke liye woh datasets use karo jahan RGB, flow, FFT aur audio sab available/usable hon.
# Audio ke liye kuch videos silent ho sakti hain; SafeDataset unko skip karega.

REAL_DIRS = [
    "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
    "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real",
    "/kaggle/input/datasets/abdullahpy/msrvtt/TrainValVideo",
    "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics600_5per/kinetics600_5per/train",
    "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train"
]

FAKE_DIRS = [
    "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
    "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake",
    "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
    "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18",
    "/kaggle/input/datasets/abdullahpy/ai-generated-video/Fake",
    "/kaggle/input/datasets/abdullahpy/raw-fake-ai/Raw_reel",
]


# ==========================================
# ✅ REPRODUCIBILITY
# ==========================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ==========================================
# 🧹 PATH CLEANER
# ==========================================
def clean_existing_dirs(dir_list):
    valid_dirs = []

    for d in dir_list:
        if os.path.exists(d):
            valid_dirs.append(d)
        else:
            print(f"⚠️ Missing path skipped: {d}")

    return valid_dirs


# ==========================================
# 🛡️ SAFE DATASET WRAPPER
# ==========================================
class SafeDataset(Dataset):
    """
    Fusion ke liye corrupted videos/audio ko skip karega.
    Important:
    - video_rgb valid hona chahiye
    - optical_flow valid hona chahiye
    - fft valid hona chahiye
    - audio valid hona chahiye
    """

    def __init__(self, base_dataset, max_retries=25, name="dataset"):
        self.base_dataset = base_dataset
        self.max_retries = max_retries
        self.name = name

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        dataset_len = len(self.base_dataset)

        for attempt in range(self.max_retries):
            try:
                if attempt == 0:
                    safe_idx = idx
                else:
                    safe_idx = random.randint(0, dataset_len - 1)

                sample = self.base_dataset[safe_idx]

                if sample is None:
                    raise RuntimeError("Dataset returned None")

                video_rgb, optical_flow, fft_images, audio_waveforms, label = sample

                # ------------------------------
                # Validate RGB
                # ------------------------------
                if not isinstance(video_rgb, torch.Tensor) or video_rgb.numel() == 0:
                    raise RuntimeError("Invalid RGB tensor")

                if torch.isnan(video_rgb).any() or torch.isinf(video_rgb).any():
                    raise RuntimeError("RGB has NaN/Inf")

                # ------------------------------
                # Validate Optical Flow
                # ------------------------------
                if not isinstance(optical_flow, torch.Tensor) or optical_flow.numel() == 0:
                    raise RuntimeError("Invalid optical flow tensor")

                if torch.isnan(optical_flow).any() or torch.isinf(optical_flow).any():
                    raise RuntimeError("Optical flow has NaN/Inf")

                # ------------------------------
                # Validate FFT
                # ------------------------------
                if not isinstance(fft_images, torch.Tensor) or fft_images.numel() == 0:
                    raise RuntimeError("Invalid FFT tensor")

                if torch.isnan(fft_images).any() or torch.isinf(fft_images).any():
                    raise RuntimeError("FFT has NaN/Inf")

                # ------------------------------
                # Validate Audio
                # ------------------------------
                if not isinstance(audio_waveforms, torch.Tensor) or audio_waveforms.numel() == 0:
                    raise RuntimeError("Invalid audio tensor")

                if torch.isnan(audio_waveforms).any() or torch.isinf(audio_waveforms).any():
                    raise RuntimeError("Audio has NaN/Inf")

                if audio_waveforms.abs().sum() < 1e-6:
                    raise RuntimeError("Silent audio")

                return sample

            except KeyboardInterrupt:
                raise

            except Exception as e:
                print(
                    f"\n⚠️ Bad fusion sample skipped | {self.name} | "
                    f"idx={idx} | attempt={attempt + 1}/{self.max_retries} | "
                    f"error={str(e)[:180]}"
                )
                continue

        raise RuntimeError(
            f"❌ Too many corrupt samples in {self.name} near index {idx}"
        )


# ==========================================
# 🎧 AUDIO SHAPE FIX
# ==========================================
def fix_audio_batch(audio):
    """
    Expected by Wav2Vec:
    (B, L)

    Handles:
    (B, L)
    (B, 1, L)
    (B, C, L)
    """

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    if audio.dim() == 3:
        audio = audio.mean(dim=1)

    if audio.dim() != 2:
        raise ValueError(f"Invalid audio batch shape: {audio.shape}")

    return audio


# ==========================================
# 📊 METRICS
# ==========================================
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0

    return acc, precision, recall, f1, auc, y_pred


# ==========================================
# 🔥 TRAIN ONE EPOCH
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    valid_steps = 0

    all_labels = []
    all_probs = []

    loop = tqdm(loader, desc="Training", leave=True)

    for batch_idx, batch in enumerate(loop):
        video_rgb, optical_flow, fft_images, audio_waveforms, labels = batch

        video_rgb = video_rgb.to(device, non_blocking=True).float()
        optical_flow = optical_flow.to(device, non_blocking=True).float()
        fft_images = fft_images.to(device, non_blocking=True).float()

        audio_waveforms = fix_audio_batch(audio_waveforms)
        audio_waveforms = audio_waveforms.to(device, non_blocking=True).float()

        labels = labels.to(device, non_blocking=True).float().view(-1, 1)

        video_rgb = torch.nan_to_num(video_rgb, nan=0.0, posinf=1.0, neginf=-1.0)
        optical_flow = torch.nan_to_num(optical_flow, nan=0.0, posinf=1.0, neginf=-1.0)
        fft_images = torch.nan_to_num(fft_images, nan=0.0, posinf=1.0, neginf=-1.0)
        audio_waveforms = torch.nan_to_num(audio_waveforms, nan=0.0, posinf=1.0, neginf=-1.0)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(
                    video_frames=video_rgb,
                    optical_flow=optical_flow,
                    fft_images=fft_images,
                    audio_waveforms=audio_waveforms
                )

                loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        else:
            logits = model(
                video_frames=video_rgb,
                optical_flow=optical_flow,
                fft_images=fft_images,
                audio_waveforms=audio_waveforms
            )

            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ Invalid loss skipped")
            continue

        total_loss += loss.item()
        valid_steps += 1

        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        labs = labels.detach().cpu().numpy().flatten()

        all_probs.extend(probs.tolist())
        all_labels.extend(labs.tolist())

        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(valid_steps, 1)

    if len(all_labels) == 0:
        return avg_loss, 0.0, 0.0, 0.0, 0.0, 0.0

    acc, precision, recall, f1, auc, _ = compute_metrics(
        all_labels,
        all_probs,
        threshold=DECISION_THRESHOLD
    )

    return avg_loss, acc, precision, recall, f1, auc


# ==========================================
# 🔍 VALIDATE ONE EPOCH
# ==========================================
@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    valid_steps = 0

    all_labels = []
    all_probs = []

    loop = tqdm(loader, desc="Validating", leave=True)

    for batch_idx, batch in enumerate(loop):
        video_rgb, optical_flow, fft_images, audio_waveforms, labels = batch

        video_rgb = video_rgb.to(device, non_blocking=True).float()
        optical_flow = optical_flow.to(device, non_blocking=True).float()
        fft_images = fft_images.to(device, non_blocking=True).float()

        audio_waveforms = fix_audio_batch(audio_waveforms)
        audio_waveforms = audio_waveforms.to(device, non_blocking=True).float()

        labels = labels.to(device, non_blocking=True).float().view(-1, 1)

        video_rgb = torch.nan_to_num(video_rgb, nan=0.0, posinf=1.0, neginf=-1.0)
        optical_flow = torch.nan_to_num(optical_flow, nan=0.0, posinf=1.0, neginf=-1.0)
        fft_images = torch.nan_to_num(fft_images, nan=0.0, posinf=1.0, neginf=-1.0)
        audio_waveforms = torch.nan_to_num(audio_waveforms, nan=0.0, posinf=1.0, neginf=-1.0)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(
                    video_frames=video_rgb,
                    optical_flow=optical_flow,
                    fft_images=fft_images,
                    audio_waveforms=audio_waveforms
                )

                loss = criterion(logits, labels)

        else:
            logits = model(
                video_frames=video_rgb,
                optical_flow=optical_flow,
                fft_images=fft_images,
                audio_waveforms=audio_waveforms
            )

            loss = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        total_loss += loss.item()
        valid_steps += 1

        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        labs = labels.detach().cpu().numpy().flatten()

        all_probs.extend(probs.tolist())
        all_labels.extend(labs.tolist())

        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(valid_steps, 1)

    if len(all_labels) == 0:
        return avg_loss, 0.0, 0.0, 0.0, 0.0, 0.0, [], [], []

    acc, precision, recall, f1, auc, y_pred = compute_metrics(
        all_labels,
        all_probs,
        threshold=DECISION_THRESHOLD
    )

    return avg_loss, acc, precision, recall, f1, auc, all_labels, all_probs, y_pred


# ==========================================
# 🚀 MAIN TRAINING FUNCTION
# ==========================================
def train_fusion_model():
    print("\n" + "=" * 80)
    print("🔥 DEEPGUARD FUSION TRAINING STARTED")
    print("=" * 80)
    print("CUDA available     :", torch.cuda.is_available())
    print("CUDA device count  :", torch.cuda.device_count())
    print("Selected device    :", DEVICE)

    if torch.cuda.is_available():
        print("GPU name           :", torch.cuda.get_device_name(0))
    else:
        print("❌ GPU not detected. Training will run on CPU.")

    # ==========================================
    # ✅ Check expert files
    # ==========================================
    expert_paths = {
        "Visual": VISUAL_EXPERT_PATH,
        "Physics": PHYSICS_EXPERT_PATH,
        "Forensic": FORENSIC_EXPERT_PATH,
        "Audio": AUDIO_EXPERT_PATH,
    }

    print("\n🔍 Checking expert checkpoint files")
    for name, path in expert_paths.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ {name}: {path} | {size_mb:.2f} MB")
        else:
            raise FileNotFoundError(f"❌ {name} expert not found: {path}")

    # ==========================================
    # ✅ Check dataset paths
    # ==========================================
    real_dirs = clean_existing_dirs(REAL_DIRS)
    fake_dirs = clean_existing_dirs(FAKE_DIRS)

    print("\n✅ Active REAL folders:")
    for d in real_dirs:
        print("REAL →", d)

    print("\n✅ Active FAKE folders:")
    for d in fake_dirs:
        print("FAKE →", d)

    if len(real_dirs) == 0:
        raise RuntimeError("❌ No REAL folders found.")

    if len(fake_dirs) == 0:
        raise RuntimeError("❌ No FAKE folders found.")

    # ==========================================
    # 📦 Dataset loading
    # ==========================================
    real_dataset_raw = DeepGuardDataset(
        real_dirs=real_dirs,
        fake_dirs=[],
        num_frames=16,
        max_samples=SAMPLES_PER_CLASS,
        mode="multi"
    )

    fake_dataset_raw = DeepGuardDataset(
        real_dirs=[],
        fake_dirs=fake_dirs,
        num_frames=16,
        max_samples=SAMPLES_PER_CLASS,
        mode="multi"
    )

    real_dataset = SafeDataset(
        base_dataset=real_dataset_raw,
        max_retries=25,
        name="REAL_FUSION"
    )

    fake_dataset = SafeDataset(
        base_dataset=fake_dataset_raw,
        max_retries=25,
        name="FAKE_FUSION"
    )

    # Class-balanced split
    real_val_size = max(1, int(0.2 * len(real_dataset)))
    real_train_size = len(real_dataset) - real_val_size

    fake_val_size = max(1, int(0.2 * len(fake_dataset)))
    fake_train_size = len(fake_dataset) - fake_val_size

    real_train, real_val = random_split(
        real_dataset,
        [real_train_size, real_val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    fake_train, fake_val = random_split(
        fake_dataset,
        [fake_train_size, fake_val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_dataset = ConcatDataset([real_train, fake_train])
    val_dataset = ConcatDataset([real_val, fake_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == "cuda" else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == "cuda" else False,
        drop_last=False
    )

    print("\n📊 Dataset Info")
    print(f"Real total    : {len(real_dataset)}")
    print(f"Fake total    : {len(fake_dataset)}")
    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    # ==========================================
    # 🧠 Model initialize
    # ==========================================
    model = DeepGuardFusionModel(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        freeze_experts=True
    )

    model.load_expert_weights(
        visual_path=VISUAL_EXPERT_PATH,
        physics_path=PHYSICS_EXPERT_PATH,
        forensic_path=FORENSIC_EXPERT_PATH,
        audio_path=AUDIO_EXPERT_PATH,
        map_location=DEVICE,
        strict=True
    )

    model = model.to(DEVICE)

    print("\n✅ Fusion model initialized")
    print("✅ Model device:", next(model.parameters()).device)

    # Trainable params check
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")

    # ==========================================
    # 🎯 Loss + Optimizer + Scheduler
    # ==========================================
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ==========================================
    # 🔥 Training loop
    # ==========================================
    best_auc = -1.0
    best_f1 = -1.0
    best_state_dict = None
    best_epoch = 0
    best_report_data = None

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 80)
        print(f"🔥 FUSION EPOCH {epoch}/{EPOCHS}")
        print("=" * 80)

        train_loss, train_acc, train_precision, train_recall, train_f1, train_auc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE
        )

        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, y_true, y_prob, y_pred = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE
        )

        scheduler.step()

        print("\n📊 Epoch Summary")
        print("Training")
        print(f"Loss      : {train_loss:.6f}")
        print(f"Accuracy  : {train_acc:.4f}")
        print(f"Precision : {train_precision:.4f}")
        print(f"Recall    : {train_recall:.4f}")
        print(f"F1        : {train_f1:.4f}")
        print(f"ROC-AUC   : {train_auc:.4f}")

        print("\nValidation")
        print(f"Loss      : {val_loss:.6f}")
        print(f"Accuracy  : {val_acc:.4f}")
        print(f"Precision : {val_precision:.4f}")
        print(f"Recall    : {val_recall:.4f}")
        print(f"F1        : {val_f1:.4f}")
        print(f"ROC-AUC   : {val_auc:.4f}")

        print("\nLabel/Prediction Count")
        print("Val REAL count :", np.sum(np.array(y_true) == 0))
        print("Val FAKE count :", np.sum(np.array(y_true) == 1))
        print("Pred REAL count:", np.sum(np.array(y_pred) == 0))
        print("Pred FAKE count:", np.sum(np.array(y_pred) == 1))

        # Save best in memory only
        if (val_auc > best_auc) or (val_auc == best_auc and val_f1 > best_f1):
            best_auc = val_auc
            best_f1 = val_f1
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

            best_report_data = {
                "y_true": y_true,
                "y_prob": y_prob,
                "y_pred": y_pred,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "val_auc": val_auc,
            }

            print(f"\n✅ New best fusion model in memory | Epoch {epoch} | AUC={val_auc:.4f} | F1={val_f1:.4f}")

    # ==========================================
    # 💾 Save final best fusion model once
    # ==========================================
    if best_state_dict is None:
        raise RuntimeError("❌ No valid fusion model state found.")

    torch.save(
        {
            "best_epoch": best_epoch,
            "best_auc": best_auc,
            "best_f1": best_f1,
            "model_state_dict": best_state_dict,
            "visual_expert_path": VISUAL_EXPERT_PATH,
            "physics_expert_path": PHYSICS_EXPERT_PATH,
            "forensic_expert_path": FORENSIC_EXPERT_PATH,
            "audio_expert_path": AUDIO_EXPERT_PATH,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
        },
        SAVE_FUSION_BEST_PATH
    )

    torch.save(
        best_state_dict,
        SAVE_FUSION_FULL_PATH
    )

    print("\n" + "=" * 80)
    print("✅ FUSION TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Epoch   : {best_epoch}")
    print(f"Best Val AUC : {best_auc:.4f}")
    print(f"Best Val F1  : {best_f1:.4f}")

    print("\n✅ Best fusion checkpoint saved at:")
    print(SAVE_FUSION_BEST_PATH)

    print("\n✅ Fusion full state_dict saved at:")
    print(SAVE_FUSION_FULL_PATH)

    if best_report_data is not None:
        print("\n" + "-" * 80)
        print("🧾 Best Validation Confusion Matrix")
        print(confusion_matrix(best_report_data["y_true"], best_report_data["y_pred"]))

        print("\nClassification Report")
        print(
            classification_report(
                best_report_data["y_true"],
                best_report_data["y_pred"],
                target_names=["REAL", "FAKE"],
                zero_division=0
            )
        )

    print("=" * 80)


# ==========================================
# ▶️ RUN
# ==========================================
if __name__ == "__main__":
    train_fusion_model()
