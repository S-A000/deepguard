# ==========================================
# 🔍 DEEPGUARD FORENSIC TRAINING - TPU/XLA VERSION
# ✅ For Kaggle TPU
# ✅ No CUDA AMP
# ✅ No GradScaler
# ✅ No DataParallel
# ✅ Uses torch_xla xm.xla_device()
# ✅ Uses xm.optimizer_step()
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
    roc_auc_score
)

from torch.utils.data import (
    DataLoader,
    ConcatDataset,
    random_split,
    WeightedRandomSampler
)

# ==========================================
# TPU / XLA IMPORTS
# ==========================================
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
except Exception as e:
    TPU_AVAILABLE = False
    print("⚠️ torch_xla not available. Error:", e)

# ==========================================
# SYSTEM CONFIG
# ==========================================
warnings.filterwarnings("ignore")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
CURRENT_PHASE = 1


# ==========================================
# DEVICE FUNCTION
# ==========================================
def get_device():
    """
    Priority:
    1. TPU/XLA
    2. CUDA GPU
    3. CPU
    """

    if TPU_AVAILABLE:
        device = xm.xla_device()
        print("✅ TPU/XLA device selected:", device)
        return device, "tpu"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ CUDA GPU selected:", device)
        return device, "cuda"

    device = torch.device("cpu")
    print("⚠️ CPU selected:", device)
    return device, "cpu"


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
        return self.classifier(features)


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
# VALIDATION FUNCTION
# ==========================================
def validate_model(
    model,
    val_loader,
    device,
    device_type,
    threshold=0.55
):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for batch in tqdm(val_loader, desc="Validating", leave=True):

            video_rgb, flow, fft, audio, labels = batch

            fft = torch.nan_to_num(fft).float().to(device)
            labels = labels.float().view(-1, 1).to(device)

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

            if device_type == "tpu":
                xm.mark_step()

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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


# ==========================================
# CHECKPOINT LOADER
# ==========================================
def load_checkpoint_safely(model, checkpoint_path, device):
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("🆕 No previous checkpoint loaded.")
        return model

    print("\n📂 Loading previous phase model:")
    print(checkpoint_path)

    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k.replace("module.", "", 1)
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)

        print("✅ Checkpoint loaded successfully.")

    except Exception as e:
        print("⚠️ Checkpoint load failed:", e)

    return model


# ==========================================
# SAVE CHECKPOINT
# ==========================================
def save_model(model, save_path, device_type):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    state_dict = model.state_dict()

    if device_type == "tpu":
        xm.save(state_dict, save_path)
    else:
        torch.save(state_dict, save_path)


# ==========================================
# TRAINING FUNCTION
# ==========================================
def train_forensics_model():

    device, device_type = get_device()

    print("\n========================================")
    print("🔍 DEEPGUARD FORENSICS TRAINING")
    print("========================================")
    print(f"📍 CURRENT PHASE : {CURRENT_PHASE}")
    print(f"⚡ DEVICE TYPE   : {device_type}")
    print(f"⚡ DEVICE        : {device}")

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
            "forensic_phase1_tpu.pth"
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
            "production/forensic_phase1_tpu.pth"
        )

        SAVE_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_phase2_tpu.pth"
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
            "production/forensic_phase2_tpu.pth"
        )

        SAVE_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_phase3_tpu.pth"
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
            "/kaggle/input/datasets/abdullahpy/ai-generated-video/Fake"
        ]

        LR = 1e-5
        SAMPLES_PER_CLASS = 4500

        PREV_MODEL_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_phase3_tpu.pth"
        )

        SAVE_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_FINAL_tpu.pth"
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
    print(f"✅ FAKE DATASETS : {len(FAKE_DIRS)}")

    if len(REAL_DIRS) == 0:
        raise RuntimeError("❌ No real dataset path found.")

    if len(FAKE_DIRS) == 0:
        raise RuntimeError("❌ No fake dataset path found.")

    # ==========================================
    # DATASETS
    # ==========================================

    real_dataset = DeepGuardDataset(
        real_dirs=REAL_DIRS,
        fake_dirs=[],
        max_samples=SAMPLES_PER_CLASS,
        mode="multi"
    )

    fake_dataset = DeepGuardDataset(
        real_dirs=[],
        fake_dirs=FAKE_DIRS,
        max_samples=SAMPLES_PER_CLASS,
        mode="multi"
    )

    full_dataset = ConcatDataset([
        real_dataset,
        fake_dataset
    ])

    total_size = len(full_dataset)

    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print(f"\n📦 TOTAL SAMPLES : {total_size}")
    print(f"🚂 TRAIN SAMPLES : {train_size}")
    print(f"🧪 VAL SAMPLES   : {val_size}")

    # ==========================================
    # BALANCED SAMPLER
    # ==========================================

    print("\n⚖️ Implementing balanced sampler...")

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
    # TPU note:
    # num_workers=0 is safer with OpenCV/librosa/ffmpeg preprocessing.
    # pin_memory=False because TPU is not CUDA.

    BATCH_SIZE = 32
    NUM_WORKERS = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False
    )

    # ==========================================
    # MODEL INIT
    # ==========================================

    model = ForensicsOnlyDeepGuard().float()
    model = load_checkpoint_safely(
        model=model,
        checkpoint_path=PREV_MODEL_PATH,
        device=device
    )
    model = model.to(device)

    print("\n✅ Model moved to:", device)

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

    EPOCHS = 20 if CURRENT_PHASE >= 4 else 5

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_f1 = 0.0

    # ==========================================
    # TRAIN LOOP
    # ==========================================

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        valid_steps = 0
        skipped_steps = 0

        loop = tqdm(
            train_loader,
            total=len(train_loader),
            leave=True
        )

        for batch_idx, batch in enumerate(loop):

            video_rgb, flow, fft, audio, labels = batch

            fft = torch.nan_to_num(
                fft,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0
            ).float().to(device)

            labels = labels.float().view(-1, 1).to(device)

            optimizer.zero_grad(set_to_none=True)

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
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            if device_type == "tpu":
                xm.optimizer_step(optimizer)
                xm.mark_step()
            else:
                optimizer.step()

            running_loss += loss.item()
            valid_steps += 1

            loop.set_description(
                f"PH {CURRENT_PHASE} | EP [{epoch+1}/{EPOCHS}]"
            )

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                valid=valid_steps,
                skipped=skipped_steps
            )

        scheduler.step()

        avg_loss = running_loss / max(valid_steps, 1)

        # ==========================================
        # VALIDATION
        # ==========================================

        metrics = validate_model(
            model=model,
            val_loader=val_loader,
            device=device,
            device_type=device_type,
            threshold=0.35
        )

        print("\n========================================")
        print(f"📊 EPOCH {epoch+1} VALIDATION")
        print("========================================")

        print(f"📉 LOSS      : {avg_loss:.4f}")
        print(f"✅ ACCURACY  : {metrics['accuracy']:.4f}")
        print(f"🎯 PRECISION : {metrics['precision']:.4f}")
        print(f"🎯 RECALL    : {metrics['recall']:.4f}")
        print(f"🎯 F1 SCORE  : {metrics['f1']:.4f}")
        print(f"⭐ ROC-AUC   : {metrics['auc']:.4f}")
        print(f"✅ Valid steps   : {valid_steps}")
        print(f"⚠️ Skipped steps : {skipped_steps}")

        # ==========================================
        # SAVE BEST MODEL
        # ==========================================

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]

            save_model(
                model=model,
                save_path=SAVE_PATH,
                device_type=device_type
            )

            print("\n🔥 NEW BEST MODEL SAVED!")
            print(f"📁 PATH    : {SAVE_PATH}")
            print(f"🏆 BEST F1 : {best_f1:.4f}")

    print("\n========================================")
    print("✅ TRAINING COMPLETED")
    print("========================================")
    print(f"🏆 BEST F1 : {best_f1:.4f}")
    print(f"📁 MODEL   : {SAVE_PATH}")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    train_forensics_model()
