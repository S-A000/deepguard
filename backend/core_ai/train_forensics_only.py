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
    WeightedRandomSampler # <-- Imported Sampler
)

# ==========================================
# SYSTEM CONFIG
# ==========================================

warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==========================================
# PATH SETUP
# ==========================================

current_dir = os.path.dirname(
    os.path.abspath(__file__)
) if '__file__' in globals() else os.getcwd()

root_dir = os.path.abspath(
    os.path.join(current_dir, "../../")
)

sys.path.append(root_dir)

# ==========================================
# IMPORTS
# ==========================================

from backend.core_ai.models.branch_c_forensics import ForensicExpert
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset

# ==========================================
# MASTER PHASE SWITCH
# ==========================================

CURRENT_PHASE = 3


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

    def __init__(self, alpha=0.60, gamma=2):

        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction='none'
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
    threshold=0.55
):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for batch in val_loader:

            video_rgb, flow, fft, audio, labels = batch

            fft = torch.nan_to_num(
                fft.to(device, non_blocking=True)
            )

            labels = labels.float().to(
                device
            ).view(-1, 1)

            outputs = model(fft)

            probs = torch.sigmoid(outputs)

            all_probs.extend(
                probs.cpu().numpy().flatten()
            )

            all_labels.extend(
                labels.cpu().numpy().flatten()
            )

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
        preds
    )

    recall = recall_score(
        all_labels,
        preds
    )

    f1 = f1_score(
        all_labels,
        preds
    )

    try:

        auc = roc_auc_score(
            all_labels,
            all_probs
        )

    except:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

# ==========================================
# TRAINING FUNCTION
# ==========================================

def train_forensics_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n========================================")
    print("🔍 DEEPGUARD FORENSICS TRAINING")
    print("========================================")

    print(f"📍 CURRENT PHASE: {CURRENT_PHASE}")
    print(f"⚡ DEVICE: {device}")

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
            "forensic_phase1.pth"
        )

    elif CURRENT_PHASE == 2:

        print("\n🟡 PHASE 2 → DFDC INTEGRATION")

        REAL_DIRS = [

            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",

            "/kaggle/input/datasets/krishna191919/"
            "dfdc-part-14/dfdc_equal_split_part_14/real"
        ]

        FAKE_DIRS = [

            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",

            "/kaggle/input/datasets/zz14423/"
            "dfdc-part-01/dfdc_train_part_1"
        ]

        LR = 5e-5

        SAMPLES_PER_CLASS = 2500

        PREV_MODEL_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_phase1.pth"
        )

        SAVE_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_phase2.pth"
        )

    elif CURRENT_PHASE == 3:

        print("\n🟠 PHASE 3 → HARD FAKES")

        REAL_DIRS = [

            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",

            "/kaggle/input/datasets/krishna191919/"
            "dfdc-part-14/dfdc_equal_split_part_14/real"
        ]

        FAKE_DIRS = [

            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",

            "/kaggle/input/datasets/zz14423/"
            "dfdc-part-01/dfdc_train_part_1",

            "/kaggle/input/datasets/krishna191919/"
            "dfdc-part-14/dfdc_equal_split_part_14/fake",

            "/kaggle/input/datasets/aknirala/"
            "dfdc-train-part-18/dfdc_train_part_18"
        ]

        LR = 1e-5

        SAMPLES_PER_CLASS = 3500

        PREV_MODEL_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_phase2.pth"
        )

        SAVE_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_phase3.pth"
        )

    elif CURRENT_PHASE == 4:

        print("\n🔴 PHASE 4 → FUTURE THREATS")

        REAL_DIRS = [

            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",

            "/kaggle/input/datasets/krishna191919/"
            "dfdc-part-14/dfdc_equal_split_part_14/real",
            "/kaggle/input/datasets/abdullahpy/msrvtt/TrainValVideo"
        ]

        FAKE_DIRS = [

            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",

            "/kaggle/input/datasets/zz14423/"
            "dfdc-part-01/dfdc_train_part_1",

            "/kaggle/input/datasets/krishna191919/"
            "dfdc-part-14/dfdc_equal_split_part_14/fake",

            "/kaggle/input/datasets/aknirala/"
            "dfdc-train-part-18/dfdc_train_part_18",

            "/kaggle/input/datasets/abdullahpy/"
            "ai-generated-video/Fake"
        ]

        LR = 1e-5

        SAMPLES_PER_CLASS = 4500

        PREV_MODEL_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_phase3.pth"
        )

        SAVE_PATH = (
            "/kaggle/working/saved_models/"
            "production/forensic_FINAL.pth"
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

    # ==========================================
    # DATASETS
    # ==========================================

    real_dataset = DeepGuardDataset(
        real_dirs=REAL_DIRS,
        fake_dirs=[],
        max_samples=SAMPLES_PER_CLASS
    )

    fake_dataset = DeepGuardDataset(
        real_dirs=[],
        fake_dirs=FAKE_DIRS,
        max_samples=SAMPLES_PER_CLASS
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

    print(f"\n📦 TOTAL SAMPLES : {total_size}")
    print(f"🚂 TRAIN SAMPLES : {train_size}")
    print(f"🧪 VAL SAMPLES   : {val_size}")

    # ==========================================
    # 🎯 SMART BALANCED SAMPLER (THE FIX)
    # ==========================================
    
    print("\n⚖️ Implementing Strict 50/50 Batch Balancing...")
    
    # 1. Create a logical array of all labels in full_dataset
    # ConcatDataset puts all real (0s) first, then all fake (1s)
    full_labels = np.concatenate([
        np.zeros(len(real_dataset)), 
        np.ones(len(fake_dataset))
    ])
    
    # 2. Extract only the labels for the train split using indices
    train_labels = full_labels[train_dataset.indices]
    
    # 3. Calculate weights instantly
    class_sample_count = np.array([
        len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)
    ])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in train_labels])
    
    samples_weight = torch.from_numpy(samples_weight)
    
    # 4. Initialize the Sampler
    sampler = WeightedRandomSampler(
        weights=samples_weight.type('torch.DoubleTensor'),
        num_samples=len(samples_weight),
        replacement=True
    )
    print("✅ Sampler Ready! Every batch will be perfectly balanced.")

    # ==========================================
    # DATALOADERS
    # ==========================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler, # <-- SAMPLER ADDED HERE
        # shuffle=True,  <-- REMOVED SHUFFLE (Sampler handles it)
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # ==========================================
    # MODEL INIT
    # ==========================================

    model = ForensicsOnlyDeepGuard().float().to(device)

    if torch.cuda.device_count() > 1:

        model = nn.DataParallel(model)

    # ==========================================
    # LOAD PREVIOUS PHASE
    # ==========================================

    if PREV_MODEL_PATH and os.path.exists(PREV_MODEL_PATH):

        try:

            print("\n📂 LOADING PREVIOUS PHASE MODEL...")

            state_dict = torch.load(
                PREV_MODEL_PATH,
                map_location=device
            )

            new_state_dict = OrderedDict()

            curr_is_dp = isinstance(
                model,
                nn.DataParallel
            )

            file_is_dp = any(
                k.startswith('module.')
                for k in state_dict.keys()
            )

            for k, v in state_dict.items():

                if curr_is_dp and not file_is_dp:

                    name = 'module.' + k

                elif not curr_is_dp and file_is_dp:

                    name = k[7:]

                else:

                    name = k

                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)

            print("✅ CHECKPOINT LOADED SUCCESSFULLY")

        except Exception as e:

            print(f"⚠️ CHECKPOINT ERROR: {e}")

    # ==========================================
    # LOSS FUNCTION
    # ==========================================

    criterion = FocalLoss(
        alpha=0.80,
        gamma=2
    )

    # ==========================================
    # OPTIMIZER
    # ==========================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    # ==========================================
    # LR SCHEDULER
    # ==========================================

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10
    )

    # ==========================================
    # AMP SCALER
    # ==========================================

    scaler = torch.amp.GradScaler('cuda')

    # ==========================================
    # TRAINING SETUP
    # ==========================================

    EPOCHS = 20 if CURRENT_PHASE >= 4 else 5

    best_f1 = 0.0

    os.makedirs(
        os.path.dirname(SAVE_PATH),
        exist_ok=True
    )

    # ==========================================
    # TRAIN LOOP
    # ==========================================

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        loop = tqdm(
            train_loader,
            total=len(train_loader),
            leave=True
        )

        for batch_idx, batch in enumerate(loop):

            video_rgb, flow, fft, audio, labels = batch

            fft = torch.nan_to_num(
                fft.to(device, non_blocking=True)
            )

            labels = labels.float().to(
                device,
                non_blocking=True
            ).view(-1, 1)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):

                predictions = model(fft)

                loss = criterion(
                    predictions,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)

            scaler.update()

            running_loss += loss.item()

            loop.set_description(
                f"PH {CURRENT_PHASE} | "
                f"EP [{epoch+1}/{EPOCHS}]"
            )

            loop.set_postfix(
                LOSS=f"{loss.item():.4f}"
            )

        scheduler.step()

        avg_loss = running_loss / len(train_loader)

        # ==========================================
        # VALIDATION
        # ==========================================

        metrics = validate_model(
            model,
            val_loader,
            device,
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

        # ==========================================
        # SAVE BEST MODEL
        # ==========================================

        if metrics['f1'] > best_f1:

            best_f1 = metrics['f1']

            final_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )

            torch.save(
                final_dict,
                SAVE_PATH
            )

            print("\n🔥 NEW BEST MODEL SAVED!")
            print(f"📁 PATH : {SAVE_PATH}")
            print(f"🏆 BEST F1 : {best_f1:.4f}")

    print("\n========================================")
    print("✅ TRAINING COMPLETED")
    print("========================================")

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    train_forensics_model()
