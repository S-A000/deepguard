# ==========================================
# 🔬 DEEPGUARD FUSION ABLATION TRAINING
# ✅ Trains multiple ablation variants
# ✅ Uses same frozen experts
# ✅ Multi-frame FFT enabled
# ✅ Saves CSV table for paper
# ✅ Saves best checkpoint per variant
# ==========================================

import os
import sys
import csv
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

# For ablation, 3 epochs usually enough.
# For final paper run, you can increase to 5.
EPOCHS = 3

# 30GB GPU: 4 is usually safe.
# If OOM, set 2.
BATCH_SIZE = 4

LR = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0
DECISION_THRESHOLD = 0.5

EMBED_DIM = 256
NUM_HEADS = 8

# For ablation, do not make it too huge first.
# You can change to 10000 after stable run.
SAMPLES_PER_CLASS = 4000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==========================================
# 🔬 MULTI-FRAME FFT CONFIG
# ==========================================
FFT_MODE = "multi_avg"
FFT_NUM_FRAMES = 8


# ==========================================
# 📁 EXPERT CHECKPOINT PATHS
# ==========================================
VISUAL_EXPERT_PATH = "/kaggle/input/models/abdullahpy/audiophase2/pytorch/default/1/visual_FINAL_expert.pth"

PHYSICS_EXPERT_PATH = "/kaggle/input/models/abdullahpy/audiophase2/pytorch/default/1/physics_FINAL_expert.pth"

# IMPORTANT:
# Use your actual multi-frame FFT forensic checkpoint.
# If this file is in Kaggle working directory:
FORENSIC_EXPERT_PATH = "/kaggle/input/models/abdullahpy/final-forensic/pytorch/default/1/forensic_FINAL_multiframe_fft.pth"

# If uploaded as Kaggle model/input, replace above with something like:
# FORENSIC_EXPERT_PATH = "/kaggle/input/models/abdullahpy/forensic-multiframe/pytorch/default/1/forensic_FINAL_multiframe_fft.pth"

AUDIO_EXPERT_PATH = "/kaggle/input/models/abdullahpy/audiophase2/pytorch/default/1/audio_phase2_expert.pth"


# ==========================================
# 💾 SAVE PATHS
# ==========================================
SAVE_DIR = "/kaggle/working/saved_models/production"
ABLATION_DIR = os.path.join(SAVE_DIR, "ablation")
RESULT_CSV_PATH = os.path.join(SAVE_DIR, "ablation_results.csv")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(ABLATION_DIR, exist_ok=True)


# ==========================================
# 📁 DATASET PATHS
# ==========================================
REAL_DIRS = [
    "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
    "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real",
    "/kaggle/input/datasets/abdullahpy/msrvtt/TrainValVideo",
    "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics600_5per/kinetics600_5per/train",
    "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train",
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
# 🔬 ABLATION VARIANTS
# ==========================================
# You can comment out variants if training time is too long.
ABLATION_VARIANTS = {
    "FULL": ["visual", "physics", "forensic", "audio"],

    "NO_PHYSICS": ["visual", "forensic", "audio"],
    "NO_AUDIO": ["visual", "physics", "forensic"],
    "NO_FORENSIC": ["visual", "physics", "audio"],

    "VISUAL_ONLY": ["visual"],
    "PHYSICS_ONLY": ["physics"],
    "FORENSIC_ONLY": ["forensic"],
    "AUDIO_ONLY": ["audio"],
}


# ==========================================
# ✅ REPRODUCIBILITY
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


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
                safe_idx = idx if attempt == 0 else random.randint(0, dataset_len - 1)
                sample = self.base_dataset[safe_idx]

                if sample is None:
                    raise RuntimeError("Dataset returned None")

                video_rgb, optical_flow, fft_images, audio_waveforms, label = sample

                if not isinstance(video_rgb, torch.Tensor) or video_rgb.numel() == 0:
                    raise RuntimeError("Invalid RGB tensor")

                if torch.isnan(video_rgb).any() or torch.isinf(video_rgb).any():
                    raise RuntimeError("RGB has NaN/Inf")

                if not isinstance(optical_flow, torch.Tensor) or optical_flow.numel() == 0:
                    raise RuntimeError("Invalid optical flow tensor")

                if torch.isnan(optical_flow).any() or torch.isinf(optical_flow).any():
                    raise RuntimeError("Optical flow has NaN/Inf")

                if not isinstance(fft_images, torch.Tensor) or fft_images.numel() == 0:
                    raise RuntimeError("Invalid FFT tensor")

                if torch.isnan(fft_images).any() or torch.isinf(fft_images).any():
                    raise RuntimeError("FFT has NaN/Inf")

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
                    f"\n⚠️ Bad sample skipped | {self.name} | "
                    f"idx={idx} | attempt={attempt + 1}/{self.max_retries} | "
                    f"error={str(e)[:160]}"
                )
                continue

        raise RuntimeError(f"❌ Too many corrupt samples in {self.name} near index {idx}")


# ==========================================
# 🎧 AUDIO SHAPE FIX
# ==========================================
def fix_audio_batch(audio):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    if audio.dim() == 3:
        audio = audio.mean(dim=1)

    if audio.dim() != 2:
        raise ValueError(f"Invalid audio batch shape: {audio.shape}")

    return audio


# ==========================================
# 🧠 ABLATION MODEL
# ==========================================
class AblationDeepGuardFusionModel(DeepGuardFusionModel):
    """
    This subclass uses the same expert modules and fusion layers,
    but masks inactive modalities for ablation.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        active_modalities=None,
        freeze_experts=True
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            freeze_experts=freeze_experts
        )

        if active_modalities is None:
            active_modalities = ["visual", "physics", "forensic", "audio"]

        self.active_modalities = set(active_modalities)
        self.embed_dim = embed_dim

    def _apply_modality_embedding_if_exists(self, stacked_features):
        """
        Supports different fusion_net.py versions:
        - with self.modality_embedding
        - without modality embedding
        """

        if hasattr(self, "modality_embedding"):
            try:
                emb = self.modality_embedding

                # Common case: nn.Parameter shape (1, 4, D)
                if isinstance(emb, torch.nn.Parameter):
                    if emb.dim() == 3:
                        return stacked_features + emb
                    if emb.dim() == 2:
                        return stacked_features + emb.unsqueeze(0)

                # If modality_embedding is nn.Embedding
                if isinstance(emb, nn.Embedding):
                    idx = torch.arange(
                        stacked_features.shape[1],
                        device=stacked_features.device
                    )
                    return stacked_features + emb(idx).unsqueeze(0)

            except Exception:
                return stacked_features

        return stacked_features

    def _apply_pre_norm(self, x):
        if hasattr(self, "pre_attn_norm"):
            return self.pre_attn_norm(x)

        if hasattr(self, "norm"):
            return self.norm(x)

        return x

    def _apply_post_norm(self, attn_out, residual):
        if hasattr(self, "post_attn_norm"):
            return self.post_attn_norm(attn_out + residual)

        return attn_out

    def forward(self, video_frames, optical_flow, fft_images, audio_waveforms):
        batch_size = video_frames.shape[0]
        device = video_frames.device

        zero_feat = torch.zeros(
            batch_size,
            self.embed_dim,
            device=device,
            dtype=torch.float32
        )

        # Experts are frozen, so no need to store gradients for them.
        with torch.no_grad():
            if "visual" in self.active_modalities:
                vis_feat = self.visual_expert(video_frames)
            else:
                vis_feat = zero_feat

            if "physics" in self.active_modalities:
                phys_feat = self.physics_expert(optical_flow)
            else:
                phys_feat = zero_feat

            if "forensic" in self.active_modalities:
                for_feat = self.forensic_expert(fft_images)
            else:
                for_feat = zero_feat

            if "audio" in self.active_modalities:
                aud_feat = self.audio_expert(audio_waveforms)
            else:
                aud_feat = zero_feat

        vis_feat = torch.nan_to_num(vis_feat, nan=0.0, posinf=1.0, neginf=-1.0)
        phys_feat = torch.nan_to_num(phys_feat, nan=0.0, posinf=1.0, neginf=-1.0)
        for_feat = torch.nan_to_num(for_feat, nan=0.0, posinf=1.0, neginf=-1.0)
        aud_feat = torch.nan_to_num(aud_feat, nan=0.0, posinf=1.0, neginf=-1.0)

        # Keep fixed 4-token layout:
        # [visual, physics, forensic, audio]
        stacked_features = torch.stack(
            (vis_feat, phys_feat, for_feat, aud_feat),
            dim=1
        )

        stacked_features = self._apply_modality_embedding_if_exists(stacked_features)

        residual = stacked_features
        stacked_features = self._apply_pre_norm(stacked_features)

        attn_out, _ = self.attention_fusion(
            stacked_features,
            stacked_features,
            stacked_features
        )

        attn_out = self._apply_post_norm(attn_out, residual)

        pooled_features = torch.mean(attn_out, dim=1)

        output_logit = self.fusion_mlp(pooled_features)

        return output_logit


# ==========================================
# 🧊 KEEP FROZEN EXPERTS IN EVAL MODE
# ==========================================
def set_frozen_experts_eval(model):
    base_model = model.module if isinstance(model, nn.DataParallel) else model

    base_model.visual_expert.eval()
    base_model.physics_expert.eval()
    base_model.forensic_expert.eval()
    base_model.audio_expert.eval()


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
def train_one_epoch(model, loader, optimizer, criterion, device, variant_name):
    model.train()
    set_frozen_experts_eval(model)

    total_loss = 0.0
    valid_steps = 0
    skipped_steps = 0

    all_labels = []
    all_probs = []

    loop = tqdm(loader, desc=f"Training {variant_name}", leave=True)

    for batch_idx, batch in enumerate(loop):
        try:
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

            logits = model(
                video_frames=video_rgb,
                optical_flow=optical_flow,
                fft_images=fft_images,
                audio_waveforms=audio_waveforms
            )

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                skipped_steps += 1
                continue

            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                skipped_steps += 1
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()
            valid_steps += 1

            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            labs = labels.detach().cpu().numpy().flatten()

            all_probs.extend(probs.tolist())
            all_labels.extend(labs.tolist())

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
                print(f"⚠️ Runtime skipped: {str(e)[:160]}")

            continue

        except Exception as e:
            skipped_steps += 1
            print(f"⚠️ Batch skipped: {str(e)[:160]}")
            continue

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
def validate_one_epoch(model, loader, criterion, device, variant_name):
    model.eval()

    total_loss = 0.0
    valid_steps = 0
    skipped_steps = 0

    all_labels = []
    all_probs = []

    loop = tqdm(loader, desc=f"Validating {variant_name}", leave=True)

    for batch_idx, batch in enumerate(loop):
        try:
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

            logits = model(
                video_frames=video_rgb,
                optical_flow=optical_flow,
                fft_images=fft_images,
                audio_waveforms=audio_waveforms
            )

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                skipped_steps += 1
                continue

            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                skipped_steps += 1
                continue

            total_loss += loss.item()
            valid_steps += 1

            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            labs = labels.detach().cpu().numpy().flatten()

            all_probs.extend(probs.tolist())
            all_labels.extend(labs.tolist())

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                valid=valid_steps,
                skipped=skipped_steps
            )

        except Exception as e:
            skipped_steps += 1
            print(f"⚠️ Val batch skipped: {str(e)[:160]}")
            continue

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
# 🧾 SAVE RESULTS CSV
# ==========================================
def save_results_csv(results, csv_path):
    fieldnames = [
        "variant",
        "active_modalities",
        "best_epoch",
        "val_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "checkpoint_path"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print("\n✅ Ablation CSV saved at:")
    print(csv_path)


# ==========================================
# 📦 PREPARE DATA
# ==========================================
def prepare_dataloaders():
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

    real_dataset_raw = DeepGuardDataset(
        real_dirs=real_dirs,
        fake_dirs=[],
        num_frames=16,
        max_samples=SAMPLES_PER_CLASS,
        mode="multi",
        fft_mode=FFT_MODE,
        fft_num_frames=FFT_NUM_FRAMES
    )

    fake_dataset_raw = DeepGuardDataset(
        real_dirs=[],
        fake_dirs=fake_dirs,
        num_frames=16,
        max_samples=SAMPLES_PER_CLASS,
        mode="multi",
        fft_mode=FFT_MODE,
        fft_num_frames=FFT_NUM_FRAMES
    )

    real_dataset = SafeDataset(
        base_dataset=real_dataset_raw,
        max_retries=25,
        name="REAL_ABLATION"
    )

    fake_dataset = SafeDataset(
        base_dataset=fake_dataset_raw,
        max_retries=25,
        name="FAKE_ABLATION"
    )

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

    return train_loader, val_loader


# ==========================================
# 🧠 BUILD MODEL FOR VARIANT
# ==========================================
def build_model_for_variant(variant_name, active_modalities):
    print("\n🔧 Building model for variant:", variant_name)
    print("Active modalities:", active_modalities)

    model = AblationDeepGuardFusionModel(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        active_modalities=active_modalities,
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

    if torch.cuda.device_count() > 1:
        print(f"🚀 Multi-GPU detected: {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")

    return model


# ==========================================
# 🚀 TRAIN ONE ABLATION VARIANT
# ==========================================
def run_single_variant(variant_name, active_modalities, train_loader, val_loader):
    set_seed(SEED)

    model = build_model_for_variant(
        variant_name=variant_name,
        active_modalities=active_modalities
    )

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

    best_auc = -1.0
    best_f1 = -1.0
    best_epoch = 0
    best_state_dict = None
    best_data = None

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 80)
        print(f"🔬 VARIANT: {variant_name} | EPOCH {epoch}/{EPOCHS}")
        print("=" * 80)

        train_loss, train_acc, train_precision, train_recall, train_f1, train_auc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            variant_name=variant_name
        )

        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, y_true, y_prob, y_pred = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE,
            variant_name=variant_name
        )

        scheduler.step()

        print("\n📊 Epoch Summary")

        print("\nTraining")
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

        if (val_auc > best_auc) or (val_auc == best_auc and val_f1 > best_f1):
            best_auc = val_auc
            best_f1 = val_f1
            best_epoch = epoch

            base_model = model.module if isinstance(model, nn.DataParallel) else model
            best_state_dict = copy.deepcopy(base_model.state_dict())

            best_data = {
                "val_loss": val_loss,
                "accuracy": val_acc,
                "precision": val_precision,
                "recall": val_recall,
                "f1": val_f1,
                "roc_auc": val_auc,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob,
            }

            print(
                f"\n✅ New best for {variant_name} | "
                f"Epoch {epoch} | AUC={val_auc:.4f} | F1={val_f1:.4f}"
            )

    if best_state_dict is None:
        raise RuntimeError(f"❌ No valid model state for variant {variant_name}")

    checkpoint_path = os.path.join(
        ABLATION_DIR,
        f"fusion_ablation_{variant_name}.pth"
    )

    torch.save(
        {
            "variant": variant_name,
            "active_modalities": active_modalities,
            "best_epoch": best_epoch,
            "best_auc": best_auc,
            "best_f1": best_f1,
            "model_state_dict": best_state_dict,
            "visual_expert_path": VISUAL_EXPERT_PATH,
            "physics_expert_path": PHYSICS_EXPERT_PATH,
            "forensic_expert_path": FORENSIC_EXPERT_PATH,
            "audio_expert_path": AUDIO_EXPERT_PATH,
            "fft_mode": FFT_MODE,
            "fft_num_frames": FFT_NUM_FRAMES,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
        },
        checkpoint_path
    )

    cm = confusion_matrix(best_data["y_true"], best_data["y_pred"])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    print("\n" + "-" * 80)
    print(f"🧾 BEST RESULT FOR {variant_name}")
    print("-" * 80)
    print("Best Epoch:", best_epoch)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(
        classification_report(
            best_data["y_true"],
            best_data["y_pred"],
            target_names=["REAL", "FAKE"],
            zero_division=0
        )
    )
    print("Checkpoint:", checkpoint_path)

    return {
        "variant": variant_name,
        "active_modalities": "+".join(active_modalities),
        "best_epoch": best_epoch,
        "val_loss": best_data["val_loss"],
        "accuracy": best_data["accuracy"],
        "precision": best_data["precision"],
        "recall": best_data["recall"],
        "f1": best_data["f1"],
        "roc_auc": best_data["roc_auc"],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "checkpoint_path": checkpoint_path
    }


# ==========================================
# 🚀 MAIN
# ==========================================
def main():
    print("\n" + "=" * 80)
    print("🔬 DEEPGUARD FUSION ABLATION STARTED")
    print("=" * 80)

    print("CUDA available     :", torch.cuda.is_available())
    print("CUDA device count  :", torch.cuda.device_count())
    print("Selected device    :", DEVICE)

    if torch.cuda.is_available():
        print("GPU name           :", torch.cuda.get_device_name(0))
    else:
        print("❌ GPU not detected. Training will run on CPU.")

    print("FFT mode           :", FFT_MODE)
    print("FFT num frames     :", FFT_NUM_FRAMES)
    print("Samples/class      :", SAMPLES_PER_CLASS)
    print("Epochs/variant     :", EPOCHS)
    print("Batch size         :", BATCH_SIZE)

    # Check expert files once
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

    train_loader, val_loader = prepare_dataloaders()

    all_results = []

    for variant_name, active_modalities in ABLATION_VARIANTS.items():
        result = run_single_variant(
            variant_name=variant_name,
            active_modalities=active_modalities,
            train_loader=train_loader,
            val_loader=val_loader
        )

        all_results.append(result)

        save_results_csv(
            results=all_results,
            csv_path=RESULT_CSV_PATH
        )

        # Clean memory between variants
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("✅ ALL ABLATION RUNS COMPLETE")
    print("=" * 80)

    print("\nFinal Ablation Summary")
    for r in all_results:
        print(
            f"{r['variant']:15s} | "
            f"Acc={r['accuracy']:.4f} | "
            f"F1={r['f1']:.4f} | "
            f"AUC={r['roc_auc']:.4f} | "
            f"CM=[[{r['tn']},{r['fp']}],[{r['fn']},{r['tp']}]]"
        )

    print("\nCSV saved at:")
    print(RESULT_CSV_PATH)


if __name__ == "__main__":
    main()
