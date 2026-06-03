# ==========================================
# 🔍 DEEPGUARD FORENSIC / FFT BRANCH TRAINING
# ✅ Dual-GPU supported via nn.DataParallel
# ✅ Multi-frame FFT enabled through DeepGuardDataset
# ✅ Phase-wise training
# ✅ WeightedRandomSampler for balanced batches
# ✅ Focal Loss
# ✅ AMP mixed precision for CUDA
# ✅ Gradient clipping
# ✅ Saves best model by validation F1
# ==========================================

import os
import sys
import random
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    random_split,
    WeightedRandomSampler
)

# ==========================================
# SYSTEM CONFIG
# ==========================================

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==========================================
# PATH SETUP
# ==========================================

current_dir = os.path.dirname(
    os.path.abspath(__file__)
) if "__file__" in globals() else os.getcwd()

root_dir = os.path.abspath(
    os.path.join(current_dir, "../../")
)

if root_dir not in sys.path:
    sys.path.append(root_dir)

print("✅ Root directory added:", root_dir)
print("✅ Current working directory:", os.getcwd())

# ==========================================
# IMPORTS
# ==========================================

from backend.core_ai.models.branch_c_forensics import ForensicExpert
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset

# ==========================================
# MASTER PHASE SWITCH
# ==========================================

CURRENT_PHASE = 4

# ==========================================
# MULTI-FRAME FFT CONFIG
# ==========================================

FFT_MODE = "multi_avg"       # options: "single_center", "multi_avg"
FFT_NUM_FRAMES = 8           # 8 frames FFT average
USE_MULTIFRAME_FFT = True

# ==========================================
# MODEL
# ==========================================

class ForensicsOnlyDeepGuard(nn.Module):
    def __init__(self, embed_dim=256):
        super(ForensicsOnlyDeepGuard, self).__init__()

        self.expert = ForensicExpert(
            embed_dim=embed_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

    def forward(self, fft_images):
        features = self.expert(fft_images)
        logits = self.classifier(features)
        return logits


# ==========================================
# FOCAL LOSS
# ==========================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.80, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction="none"
        )

        pt = torch.exp(-bce_loss)

        focal_loss = (
            self.alpha *
            ((1 - pt) ** self.gamma) *
            bce_loss
        )

        return focal_loss.mean()


# ==========================================
# SAFE CHECKPOINT LOADER
# ==========================================

def load_previous_phase_model(model, prev_model_path, device):
    if prev_model_path is None:
        print("\n🆕 No previous phase model required.")
        return model

    if not os.path.exists(prev_model_path):
        print("\n⚠️ Previous phase model not found:")
        print(prev_model_path)
        print("🆕 Training from current initialization.")
        return model

    print("\n📂 Loading previous phase model:")
    print(prev_model_path)

    try:
        state_dict = torch.load(
            prev_model_path,
            map_location=device
        )

        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        new_state_dict = OrderedDict()

        curr_is_dp = isinstance(model, nn.DataParallel)
        file_is_dp = any(k.startswith("module.") for k in state_dict.keys())

        for k, v in state_dict.items():
            if curr_is_dp and not file_is_dp:
                name = "module." + k
            elif not curr_is_dp and file_is_dp:
                name = k.replace("module.", "", 1)
            else:
                name = k

            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)

        print("✅ Previous checkpoint loaded successfully.")

    except Exception as e:
        print(f"⚠️ Checkpoint loading error: {e}")
        print("🆕 Continuing without previous checkpoint.")

    return model


# ==========================================
# VALIDATION FUNCTION
# ==========================================

@torch.no_grad()
def validate_model(
    model,
    val_loader,
    device,
    threshold=0.35
):
    model.eval()

    all_probs = []
    all_labels = []

    valid_steps = 0
    skipped_steps = 0

    loop = tqdm(
        val_loader,
        desc="Validating",
        leave=True
    )

    for batch in loop:
        try:
            video_rgb, flow, fft, audio, labels = batch

            fft = fft.to(device, non_blocking=True).float()
            fft = torch.nan_to_num(
                fft,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0
            )

            labels = labels.float().to(device, non_blocking=True).view(-1, 1)

            outputs = model(fft)

            outputs = torch.nan_to_num(
                outputs,
                nan=0.0,
                posinf=10.0,
                neginf=-10.0
            )

            probs = torch.sigmoid(outputs)

            all_probs.extend(
                probs.detach().cpu().numpy().flatten().tolist()
            )

            all_labels.extend(
                labels.detach().cpu().numpy().flatten().tolist()
            )

            valid_steps += 1

        except Exception as e:
            skipped_steps += 1
            print(f"⚠️ Validation batch skipped: {str(e)[:160]}")
            continue

    if len(all_labels) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
            "cm": None,
            "report": None,
            "valid_steps": valid_steps,
            "skipped_steps": skipped_steps
        }

    preds = [
        1 if p > threshold else 0
        for p in all_probs
    ]

    accuracy = accuracy_score(
        all_labels,
        preds
    )

    precision = precision_score(
        all_labels,
        preds,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        preds,
        zero_division=0
    )

    try:
        auc = roc_auc_score(
            all_labels,
            all_probs
        )
    except Exception:
        auc = 0.0

    cm = confusion_matrix(
        all_labels,
        preds
    )

    report = classification_report(
        all_labels,
        preds,
        target_names=["REAL", "FAKE"],
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "report": report,
        "valid_steps": valid_steps,
        "skipped_steps": skipped_steps
    }


# ==========================================
# TRAINING FUNCTION
# ==========================================

def train_forensics_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 80)
    print("🔍 DEEPGUARD FORENSICS TRAINING")
    print("=" * 80)
    print(f"📍 CURRENT PHASE      : {CURRENT_PHASE}")
    print(f"⚡ DEVICE             : {device}")
    print(f"CUDA available        : {torch.cuda.is_available()}")
    print(f"CUDA device count     : {torch.cuda.device_count()}")
    print(f"FFT mode              : {FFT_MODE}")
    print(f"FFT num frames        : {FFT_NUM_FRAMES}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("❌ GPU not detected. Training will run on CPU.")

    # ==========================================
    # PHASE DATASET ROUTING
    # ==========================================

    if CURRENT_PHASE == 1:

        print("\n🟢 PHASE 1 → FF++ BASE")

        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real"
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake"
        ]

        LR = 1e-4
        SAMPLES_PER_CLASS = 1500
        PREV_MODEL_PATH = None

        SAVE_PATH = (
            "/kaggle/working/"
            "saved_models/production/"
            "forensic_phase1_multiframe_fft.pth"
        )

    elif CURRENT_PHASE == 2:

        print("\n🟡 PHASE 2 → DFDC INTEGRATION")

        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real"
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1"
        ]

        LR = 5e-5
        SAMPLES_PER_CLASS = 2500

        PREV_MODEL_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_phase1_multiframe_fft.pth"
        )

        SAVE_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_phase2_multiframe_fft.pth"
        )

    elif CURRENT_PHASE == 3:

        print("\n🟠 PHASE 3 → HARD FAKES")

        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real"
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake",
            "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18"
        ]

        LR = 1e-5
        SAMPLES_PER_CLASS = 3500

        PREV_MODEL_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_phase2_multiframe_fft.pth"
        )

        SAVE_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_phase3_multiframe_fft.pth"
        )

    elif CURRENT_PHASE == 4:

        print("\n🔴 PHASE 4 → FUTURE THREATS")

        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real",
            "/kaggle/input/datasets/abdullahpy/msrvtt/TrainValVideo"
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake",
            "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18",
            "/kaggle/input/datasets/abdullahpy/ai-generated-video/Fake",
            "/kaggle/input/datasets/abdullahpy/raw-fake-ai/Raw_reel"
        ]

        LR = 1e-5
        SAMPLES_PER_CLASS = 4500

        PREV_MODEL_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_phase3_multiframe_fft.pth"
        )

        SAVE_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_FINAL_multiframe_fft.pth"
        )

    else:
        raise ValueError("❌ INVALID PHASE")

    # ==========================================
    # VERIFY PATHS
    # ==========================================

    REAL_DIRS = [
        d for d in REAL_DIRS
        if os.path.exists(d)
    ]

    FAKE_DIRS = [
        d for d in FAKE_DIRS
        if os.path.exists(d)
    ]

    print(f"\n✅ REAL DATASETS : {len(REAL_DIRS)}")
    for d in REAL_DIRS:
        print("REAL →", d)

    print(f"\n✅ FAKE DATASETS : {len(FAKE_DIRS)}")
    for d in FAKE_DIRS:
        print("FAKE →", d)

    if len(REAL_DIRS) == 0:
        raise RuntimeError("❌ No REAL datasets found.")

    if len(FAKE_DIRS) == 0:
        raise RuntimeError("❌ No FAKE datasets found.")

    # ==========================================
    # DATASETS WITH MULTI-FRAME FFT
    # ==========================================

    real_dataset = DeepGuardDataset(
        real_dirs=REAL_DIRS,
        fake_dirs=[],
        max_samples=SAMPLES_PER_CLASS,
        mode="multi",
        fft_mode=FFT_MODE,
        fft_num_frames=FFT_NUM_FRAMES
    )

    fake_dataset = DeepGuardDataset(
        real_dirs=[],
        fake_dirs=FAKE_DIRS,
        max_samples=SAMPLES_PER_CLASS,
        mode="multi",
        fft_mode=FFT_MODE,
        fft_num_frames=FFT_NUM_FRAMES
    )

    full_dataset = ConcatDataset([
        real_dataset,
        fake_dataset
    ])

    # ==========================================
    # TRAIN / VAL SPLIT
    # ==========================================

    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print("\n📊 Dataset Info")
    print(f"Total samples : {total_size}")
    print(f"Train samples : {train_size}")
    print(f"Val samples   : {val_size}")

    # ==========================================
    # BALANCED SAMPLER
    # ==========================================

    print("\n⚖️ Implementing 50/50 batch balancing...")

    full_labels = np.concatenate([
        np.zeros(len(real_dataset)),
        np.ones(len(fake_dataset))
    ])

    train_labels = full_labels[train_dataset.indices]

    class_sample_count = np.array([
        len(np.where(train_labels == t)[0])
        for t in np.unique(train_labels)
    ])

    weight = 1.0 / class_sample_count

    samples_weight = np.array([
        weight[int(t)]
        for t in train_labels
    ])

    samples_weight = torch.from_numpy(samples_weight).double()

    sampler = WeightedRandomSampler(
        weights=samples_weight,
        num_samples=len(samples_weight),
        replacement=True
    )

    print("✅ Balanced sampler ready.")

    # ==========================================
    # DATALOADERS
    # ==========================================

    BATCH_SIZE = 32
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        drop_last=False
    )

    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")

    # ==========================================
    # MODEL INIT
    # ==========================================

    model = ForensicsOnlyDeepGuard().float().to(device)

    if torch.cuda.device_count() > 1:
        print(f"\n🚀 Dual/Multi-GPU detected: {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    else:
        print("\nℹ️ Single GPU/CPU mode.")

    # ==========================================
    # LOAD PREVIOUS PHASE
    # ==========================================

    model = load_previous_phase_model(
        model=model,
        prev_model_path=PREV_MODEL_PATH,
        device=device
    )

    # ==========================================
    # LOSS / OPTIMIZER / SCHEDULER
    # ==========================================

    criterion = FocalLoss(
        alpha=0.80,
        gamma=2
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    EPOCHS = 5 if CURRENT_PHASE >= 4 else 5

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    use_amp = torch.cuda.is_available()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    best_f1 = 0.0
    best_metrics = None

    os.makedirs(
        os.path.dirname(SAVE_PATH),
        exist_ok=True
    )

    # ==========================================
    # TRAIN LOOP
    # ==========================================

    for epoch in range(EPOCHS):

        print("\n" + "=" * 80)
        print(f"🔥 FORENSIC PHASE {CURRENT_PHASE} | EPOCH {epoch + 1}/{EPOCHS}")
        print("=" * 80)

        model.train()

        running_loss = 0.0
        valid_steps = 0
        skipped_steps = 0

        loop = tqdm(
            train_loader,
            total=len(train_loader),
            leave=True,
            desc="Training"
        )

        for batch_idx, batch in enumerate(loop):

            try:
                video_rgb, flow, fft, audio, labels = batch

                fft = fft.to(device, non_blocking=True).float()
                fft = torch.nan_to_num(
                    fft,
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0
                )

                labels = labels.float().to(device, non_blocking=True).view(-1, 1)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    "cuda",
                    enabled=use_amp
                ):
                    predictions = model(fft)

                    predictions = torch.nan_to_num(
                        predictions,
                        nan=0.0,
                        posinf=10.0,
                        neginf=-10.0
                    )

                    loss = criterion(
                        predictions,
                        labels
                    )

                if torch.isnan(loss) or torch.isinf(loss):
                    skipped_steps += 1
                    print("⚠️ Invalid loss skipped.")
                    continue

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                valid_steps += 1

                loop.set_postfix(
                    loss=f"{loss.item():.4f}",
                    valid=valid_steps,
                    skipped=skipped_steps
                )

            except RuntimeError as e:
                skipped_steps += 1

                if "out of memory" in str(e).lower():
                    print("⚠️ CUDA OOM. Clearing cache and skipping batch.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    print(f"⚠️ Runtime batch skipped: {str(e)[:180]}")

                continue

            except Exception as e:
                skipped_steps += 1
                print(f"⚠️ Batch skipped: {str(e)[:180]}")
                continue

        scheduler.step()

        avg_loss = running_loss / max(valid_steps, 1)

        # ==========================================
        # VALIDATION
        # ==========================================

        metrics = validate_model(
            model=model,
            val_loader=val_loader,
            device=device,
            threshold=0.35
        )

        print("\n" + "-" * 80)
        print(f"📊 EPOCH {epoch + 1} VALIDATION")
        print("-" * 80)

        print(f"📉 LOSS      : {avg_loss:.6f}")
        print(f"✅ ACCURACY  : {metrics['accuracy']:.4f}")
        print(f"🎯 PRECISION : {metrics['precision']:.4f}")
        print(f"🎯 RECALL    : {metrics['recall']:.4f}")
        print(f"🎯 F1 SCORE  : {metrics['f1']:.4f}")
        print(f"⭐ ROC-AUC   : {metrics['auc']:.4f}")
        print(f"✅ Train valid steps   : {valid_steps}")
        print(f"⚠️ Train skipped steps : {skipped_steps}")
        print(f"✅ Val valid steps     : {metrics['valid_steps']}")
        print(f"⚠️ Val skipped steps   : {metrics['skipped_steps']}")

        if metrics["cm"] is not None:
            print("\n🧾 Confusion Matrix")
            print(metrics["cm"])

        # ==========================================
        # SAVE BEST MODEL
        # ==========================================

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]
            best_metrics = metrics

            final_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )

            torch.save(
                final_dict,
                SAVE_PATH
            )

            print("\n🔥 NEW BEST MULTI-FRAME FFT FORENSIC MODEL SAVED!")
            print(f"📁 PATH    : {SAVE_PATH}")
            print(f"🏆 BEST F1 : {best_f1:.4f}")

    print("\n" + "=" * 80)
    print("✅ FORENSIC MULTI-FRAME FFT TRAINING COMPLETED")
    print("=" * 80)
    print(f"🏆 BEST F1 : {best_f1:.4f}")
    print(f"📁 MODEL   : {SAVE_PATH}")

    if best_metrics is not None:
        print("\n🧾 Best Confusion Matrix")
        print(best_metrics["cm"])

        print("\nClassification Report")
        print(best_metrics["report"])


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    train_forensics_model()
