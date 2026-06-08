# ============================================================
# 🔍 DEEPGUARD FUSION CROSS-DATASET EVALUATION - BALANCED
# ✅ Fixes imbalanced real/fake evaluation
# ✅ Fixes PyTorch 2.6 weights_only checkpoint error
# ✅ Loads final fusion checkpoint
# ✅ Saves predictions, misclassified videos, summary
# ✅ Multi-frame FFT enabled
# ============================================================

import os
import sys
import json
import csv
import random
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
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


# ============================================================
# ✅ PROJECT ROOT SETUP
# ============================================================

PROJECT_ROOT = "/kaggle/working/deepguard"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("✅ Project root added:", PROJECT_ROOT)
print("✅ Current working directory:", os.getcwd())


# ============================================================
# ✅ IMPORTS FROM YOUR PROJECT
# ============================================================

from backend.core_ai.models.fusion_net import DeepGuardFusionModel
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset


# ============================================================
# ⚙️ CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 2
NUM_WORKERS = 0
DECISION_THRESHOLD = 0.5

EMBED_DIM = 256
NUM_HEADS = 8

FFT_MODE = "multi_avg"
FFT_NUM_FRAMES = 8

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ============================================================
# ✅ BALANCED EVALUATION CONFIG
# ============================================================
# True  = 18290 real / 116 fake ko balance karke 116 real + 116 fake karega
# False = full imbalanced dataset evaluate karega

BALANCE_EVAL = True

# None = automatically min(real_count, fake_count)
# Example: 100 means 100 real + 100 fake
BALANCE_TARGET_PER_CLASS = None

# Random balanced subset reproducible banane ke liye
BALANCE_SEED = 42


# ============================================================
# 📌 FINAL FUSION CHECKPOINT PATH
# ============================================================

FUSION_CHECKPOINT_PATH = "/kaggle/input/models/abdullahpy/fusion-best/pytorch/default/1/fusion_FINAL_best_multiframe_fft.pth"

# Agar best checkpoint issue de aur tumhare paas working dir mein full state dict ho:
# FUSION_CHECKPOINT_PATH = "/kaggle/working/saved_models/production/fusion_FINAL_full_multiframe_fft.pth"


# ============================================================
# 📌 EXPERT CHECKPOINT PATHS
# ============================================================

VISUAL_EXPERT_PATH = "/kaggle/input/models/abdullahpy/audiophase2/pytorch/default/1/visual_FINAL_expert.pth"

PHYSICS_EXPERT_PATH = "/kaggle/input/models/abdullahpy/audiophase2/pytorch/default/1/physics_FINAL_expert.pth"

FORENSIC_EXPERT_PATH = "/kaggle/input/models/abdullahpy/final-forensic/pytorch/default/1/forensic_FINAL_multiframe_fft.pth"

AUDIO_EXPERT_PATH = "/kaggle/input/models/abdullahpy/audiophase2/pytorch/default/1/audio_phase2_expert.pth"


# ============================================================
# 📌 CROSS-DATASET TEST PATHS
# IMPORTANT:
# Yahan woh dataset do jo training mein use nahi hua
# ============================================================

TEST_REAL_DIRS = [
    "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics600_5per/kinetics600_5per/train",
]

TEST_FAKE_DIRS = [
    "/kaggle/input/datasets/abdullahpy/ai-generated-video/Fake",
]


# ============================================================
# 💾 OUTPUT PATHS
# ============================================================

OUTPUT_DIR = "/kaggle/working/saved_models/production/crossdataset_eval_balanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "crossdataset_predictions.csv")
MISCLASSIFIED_CSV = os.path.join(OUTPUT_DIR, "crossdataset_misclassified.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "crossdataset_summary.json")
SELECTED_INDICES_JSON = os.path.join(OUTPUT_DIR, "selected_balanced_indices.json")


# ============================================================
# ✅ SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 🧹 PATH CLEANER
# ============================================================

def clean_existing_dirs(dir_list):
    valid_dirs = []

    for d in dir_list:
        if os.path.exists(d):
            valid_dirs.append(d)
        else:
            print(f"⚠️ Missing path skipped: {d}")

    return valid_dirs


# ============================================================
# 🎧 AUDIO SHAPE FIX
# ============================================================

def fix_audio_batch(audio):
    """
    Expected final audio batch shape: (B, samples)
    """

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    if audio.dim() == 3:
        audio = audio.mean(dim=1)

    if audio.dim() != 2:
        raise ValueError(f"Invalid audio batch shape: {audio.shape}")

    return audio


# ============================================================
# 🛡️ SAFE DATASET WRAPPER WITH PATH TRACKING
# ============================================================

class SafeEvalDataset(Dataset):
    def __init__(self, base_dataset, label_name="DATASET", max_retries=25):
        self.base_dataset = base_dataset
        self.label_name = label_name
        self.max_retries = max_retries

    def __len__(self):
        return len(self.base_dataset)

    def _extract_path_from_item(self, item):
        video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")

        if isinstance(item, str):
            return item

        if isinstance(item, dict):
            for key in ["path", "video_path", "file_path", "filename", "video"]:
                if key in item and isinstance(item[key], str):
                    return item[key]

        if isinstance(item, (list, tuple)):
            for x in item:
                if isinstance(x, str) and x.lower().endswith(video_exts):
                    return x

        return None

    def get_sample_path(self, idx):
        possible_attrs = [
            "video_paths",
            "file_paths",
            "paths",
            "video_files",
            "files",
            "samples",
            "data",
            "items",
            "all_videos",
            "video_list"
        ]

        for attr in possible_attrs:
            if hasattr(self.base_dataset, attr):
                obj = getattr(self.base_dataset, attr)

                try:
                    if isinstance(obj, (list, tuple)) and idx < len(obj):
                        path = self._extract_path_from_item(obj[idx])
                        if path is not None:
                            return path
                except Exception:
                    pass

        return f"UNKNOWN_PATH | {self.label_name} | idx={idx}"

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

                if not isinstance(optical_flow, torch.Tensor) or optical_flow.numel() == 0:
                    raise RuntimeError("Invalid optical flow tensor")

                if not isinstance(fft_images, torch.Tensor) or fft_images.numel() == 0:
                    raise RuntimeError("Invalid FFT tensor")

                if not isinstance(audio_waveforms, torch.Tensor) or audio_waveforms.numel() == 0:
                    raise RuntimeError("Invalid audio tensor")

                video_rgb = torch.nan_to_num(video_rgb, nan=0.0, posinf=1.0, neginf=-1.0)
                optical_flow = torch.nan_to_num(optical_flow, nan=0.0, posinf=1.0, neginf=-1.0)
                fft_images = torch.nan_to_num(fft_images, nan=0.0, posinf=1.0, neginf=-1.0)
                audio_waveforms = torch.nan_to_num(audio_waveforms, nan=0.0, posinf=1.0, neginf=-1.0)

                video_path = self.get_sample_path(safe_idx)

                return (
                    video_rgb,
                    optical_flow,
                    fft_images,
                    audio_waveforms,
                    label,
                    video_path
                )

            except KeyboardInterrupt:
                raise

            except Exception as e:
                print(
                    f"\n⚠️ Bad eval sample skipped | {self.label_name} | "
                    f"idx={idx} | attempt={attempt + 1}/{self.max_retries} | "
                    f"error={str(e)[:160]}"
                )
                continue

        raise RuntimeError(f"❌ Too many corrupt samples in {self.label_name} near index {idx}")


# ============================================================
# ⚖️ BALANCE DATASET
# ============================================================

def make_balanced_subsets(real_dataset, fake_dataset):
    real_count = len(real_dataset)
    fake_count = len(fake_dataset)

    if not BALANCE_EVAL:
        print("\n⚠️ BALANCE_EVAL=False")
        print("Evaluation will use full imbalanced dataset.")
        return real_dataset, fake_dataset, real_count, fake_count, None

    if real_count == 0 or fake_count == 0:
        raise RuntimeError("❌ Cannot balance because one class has zero samples.")

    if BALANCE_TARGET_PER_CLASS is None:
        target = min(real_count, fake_count)
    else:
        target = int(BALANCE_TARGET_PER_CLASS)
        target = min(target, real_count, fake_count)

    rng = random.Random(BALANCE_SEED)

    real_indices = list(range(real_count))
    fake_indices = list(range(fake_count))

    rng.shuffle(real_indices)
    rng.shuffle(fake_indices)

    selected_real_indices = sorted(real_indices[:target])
    selected_fake_indices = sorted(fake_indices[:target])

    balanced_real = Subset(real_dataset, selected_real_indices)
    balanced_fake = Subset(fake_dataset, selected_fake_indices)

    selected_info = {
        "balance_eval": True,
        "balance_seed": BALANCE_SEED,
        "original_real_count": real_count,
        "original_fake_count": fake_count,
        "target_per_class": target,
        "selected_real_count": len(selected_real_indices),
        "selected_fake_count": len(selected_fake_indices),
        "selected_real_indices": selected_real_indices,
        "selected_fake_indices": selected_fake_indices,
    }

    with open(SELECTED_INDICES_JSON, "w") as f:
        json.dump(selected_info, f, indent=4)

    print("\n⚖️ Balanced Evaluation Enabled")
    print(f"Original REAL : {real_count}")
    print(f"Original FAKE : {fake_count}")
    print(f"Using REAL    : {len(selected_real_indices)}")
    print(f"Using FAKE    : {len(selected_fake_indices)}")
    print(f"Saved indices : {SELECTED_INDICES_JSON}")

    return balanced_real, balanced_fake, len(selected_real_indices), len(selected_fake_indices), selected_info


# ============================================================
# 📦 PREPARE CROSS-DATASET LOADER
# ============================================================

def prepare_crossdataset_loader():
    real_dirs = clean_existing_dirs(TEST_REAL_DIRS)
    fake_dirs = clean_existing_dirs(TEST_FAKE_DIRS)

    print("\n✅ CROSS-DATASET REAL folders:")
    for d in real_dirs:
        print("REAL →", d)

    print("\n✅ CROSS-DATASET FAKE folders:")
    for d in fake_dirs:
        print("FAKE →", d)

    if len(real_dirs) == 0:
        raise RuntimeError("❌ No cross-dataset REAL folders found.")

    if len(fake_dirs) == 0:
        raise RuntimeError("❌ No cross-dataset FAKE folders found.")

    real_dataset_raw = DeepGuardDataset(
        real_dirs=real_dirs,
        fake_dirs=[],
        num_frames=16,
        max_samples=None,
        mode="multi",
        fft_mode=FFT_MODE,
        fft_num_frames=FFT_NUM_FRAMES
    )

    fake_dataset_raw = DeepGuardDataset(
        real_dirs=[],
        fake_dirs=fake_dirs,
        num_frames=16,
        max_samples=None,
        mode="multi",
        fft_mode=FFT_MODE,
        fft_num_frames=FFT_NUM_FRAMES
    )

    real_dataset = SafeEvalDataset(
        base_dataset=real_dataset_raw,
        label_name="CROSS_REAL"
    )

    fake_dataset = SafeEvalDataset(
        base_dataset=fake_dataset_raw,
        label_name="CROSS_FAKE"
    )

    real_dataset, fake_dataset, real_count, fake_count, selected_info = make_balanced_subsets(
        real_dataset=real_dataset,
        fake_dataset=fake_dataset
    )

    test_dataset = ConcatDataset([real_dataset, fake_dataset])

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == "cuda" else False,
        drop_last=False
    )

    print("\n📊 Cross-Dataset Info")
    print(f"Real total  : {real_count}")
    print(f"Fake total  : {fake_count}")
    print(f"Total test  : {len(test_dataset)}")
    print(f"Batches     : {len(test_loader)}")

    return test_loader, real_count, fake_count, selected_info


# ============================================================
# 🧠 CHECKPOINT LOADING HELPERS
# ============================================================

def remove_module_prefix(state_dict):
    cleaned = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k.replace("module.", "", 1)] = v
        else:
            cleaned[k] = v

    return cleaned


def extract_state_dict(checkpoint):
    """
    Supports:
    - direct state_dict
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    - {"fusion_state_dict": ...}
    - {"fusion_model_state_dict": ...}
    - {"model": ...}
    """

    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "fusion_state_dict",
            "fusion_model_state_dict",
            "model"
        ]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        if all(isinstance(k, str) for k in checkpoint.keys()):
            tensor_values = [v for v in checkpoint.values() if torch.is_tensor(v)]
            if len(tensor_values) > 0:
                return checkpoint

    raise RuntimeError("❌ Could not extract model state_dict from checkpoint.")


def safe_torch_load(path, map_location):
    """
    Fix for PyTorch 2.6:
    torch.load default weights_only=True can fail on checkpoints containing
    numpy scalar metadata. Since this is your own trusted checkpoint, we load
    with weights_only=False.
    """

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False
        )
    except TypeError:
        # Older PyTorch versions do not support weights_only argument
        return torch.load(
            path,
            map_location=map_location
        )


# ============================================================
# 🧠 LOAD FUSION MODEL
# ============================================================

def load_fusion_model():
    print("\n🔧 Building DeepGuardFusionModel")

    model = DeepGuardFusionModel(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        freeze_experts=True
    )

    expert_paths_exist = all([
        os.path.exists(VISUAL_EXPERT_PATH),
        os.path.exists(PHYSICS_EXPERT_PATH),
        os.path.exists(FORENSIC_EXPERT_PATH),
        os.path.exists(AUDIO_EXPERT_PATH),
    ])

    if expert_paths_exist:
        print("\n🔁 Loading expert weights before fusion checkpoint")
        model.load_expert_weights(
            visual_path=VISUAL_EXPERT_PATH,
            physics_path=PHYSICS_EXPERT_PATH,
            forensic_path=FORENSIC_EXPERT_PATH,
            audio_path=AUDIO_EXPERT_PATH,
            map_location=DEVICE,
            strict=True
        )
    else:
        print("\n⚠️ Expert paths not all found. Will rely on fusion checkpoint state_dict.")
        print("Visual exists  :", os.path.exists(VISUAL_EXPERT_PATH))
        print("Physics exists :", os.path.exists(PHYSICS_EXPERT_PATH))
        print("Forensic exists:", os.path.exists(FORENSIC_EXPERT_PATH))
        print("Audio exists   :", os.path.exists(AUDIO_EXPERT_PATH))

    if not os.path.exists(FUSION_CHECKPOINT_PATH):
        raise FileNotFoundError(f"❌ Fusion checkpoint not found: {FUSION_CHECKPOINT_PATH}")

    print("\n🔁 Loading fusion checkpoint:")
    print(FUSION_CHECKPOINT_PATH)

    checkpoint = safe_torch_load(
        path=FUSION_CHECKPOINT_PATH,
        map_location=DEVICE
    )

    state_dict = extract_state_dict(checkpoint)
    state_dict = remove_module_prefix(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("\n✅ Fusion checkpoint loaded with strict=False")

    if len(missing) > 0:
        print(f"⚠️ Missing keys: {len(missing)}")
        print(missing[:20])

    if len(unexpected) > 0:
        print(f"⚠️ Unexpected keys: {len(unexpected)}")
        print(unexpected[:20])

    model = model.to(DEVICE)
    model.eval()

    if torch.cuda.device_count() > 1:
        print(f"🚀 Multi-GPU detected: {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    return model


# ============================================================
# 🔍 EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_crossdataset(model, test_loader):
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []
    rows = []

    loop = tqdm(test_loader, desc="Cross-Dataset Evaluation", leave=True)

    for batch_idx, batch in enumerate(loop):
        try:
            video_rgb, optical_flow, fft_images, audio_waveforms, labels, paths = batch

            video_rgb = video_rgb.to(DEVICE, non_blocking=True).float()
            optical_flow = optical_flow.to(DEVICE, non_blocking=True).float()
            fft_images = fft_images.to(DEVICE, non_blocking=True).float()

            audio_waveforms = fix_audio_batch(audio_waveforms)
            audio_waveforms = audio_waveforms.to(DEVICE, non_blocking=True).float()

            labels = labels.to(DEVICE, non_blocking=True).float().view(-1, 1)

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

            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            labs = labels.detach().cpu().numpy().flatten().astype(int)
            preds = (probs >= DECISION_THRESHOLD).astype(int)

            if isinstance(paths, (list, tuple)):
                path_list = list(paths)
            else:
                path_list = [str(paths)] * len(labs)

            for i in range(len(labs)):
                true_label = int(labs[i])
                pred_label = int(preds[i])
                fake_prob = float(probs[i])

                rows.append({
                    "video_path": path_list[i],
                    "true_label": true_label,
                    "true_class": "FAKE" if true_label == 1 else "REAL",
                    "pred_label": pred_label,
                    "pred_class": "FAKE" if pred_label == 1 else "REAL",
                    "fake_probability": fake_prob,
                    "real_probability": 1.0 - fake_prob,
                    "correct": int(true_label == pred_label),
                    "error_type": "NONE" if true_label == pred_label else (
                        "REAL_PREDICTED_AS_FAKE"
                        if true_label == 0 and pred_label == 1
                        else "FAKE_PREDICTED_AS_REAL"
                    )
                })

            all_labels.extend(labs.tolist())
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

            loop.set_postfix(
                batch=batch_idx,
                total=len(all_labels)
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("⚠️ CUDA OOM during eval. Clearing cache and skipping batch.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"⚠️ Runtime eval batch skipped: {str(e)[:160]}")
            continue

        except Exception as e:
            print(f"⚠️ Eval batch skipped: {str(e)[:160]}")
            continue

    return all_labels, all_probs, all_preds, rows


# ============================================================
# 📊 METRICS + SAVE
# ============================================================

def compute_and_save_results(y_true, y_prob, y_pred, rows, real_count, fake_count, selected_info):
    y_true_np = np.array(y_true).astype(int)
    y_prob_np = np.array(y_prob)
    y_pred_np = np.array(y_pred).astype(int)

    acc = accuracy_score(y_true_np, y_pred_np)
    precision = precision_score(y_true_np, y_pred_np, zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_np, zero_division=0)

    try:
        auc = roc_auc_score(y_true_np, y_prob_np)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true_np, y_pred_np)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    report = classification_report(
        y_true_np,
        y_pred_np,
        target_names=["REAL", "FAKE"],
        zero_division=0
    )

    fieldnames = [
        "video_path",
        "true_label",
        "true_class",
        "pred_label",
        "pred_class",
        "fake_probability",
        "real_probability",
        "correct",
        "error_type"
    ]

    with open(PREDICTIONS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    misclassified = [r for r in rows if r["correct"] == 0]

    with open(MISCLASSIFIED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in misclassified:
            writer.writerow(row)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "fusion_checkpoint": FUSION_CHECKPOINT_PATH,
        "balance_eval": BALANCE_EVAL,
        "selected_info": selected_info,
        "real_count": real_count,
        "fake_count": fake_count,
        "total_samples": len(y_true),
        "threshold": DECISION_THRESHOLD,
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        },
        "pred_real_count": int(np.sum(y_pred_np == 0)),
        "pred_fake_count": int(np.sum(y_pred_np == 1)),
        "predictions_csv": PREDICTIONS_CSV,
        "misclassified_csv": MISCLASSIFIED_CSV,
        "selected_indices_json": SELECTED_INDICES_JSON if BALANCE_EVAL else None
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 80)
    print("✅ CROSS-DATASET EVALUATION COMPLETE")
    print("=" * 80)

    print("\n📊 Dataset")
    print(f"Real samples : {real_count}")
    print(f"Fake samples : {fake_count}")
    print(f"Total        : {len(y_true)}")
    print(f"Balanced     : {BALANCE_EVAL}")

    print("\n📊 Metrics")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

    print("\n🧾 Confusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(report)

    print("\nLabel/Prediction Count")
    print("True REAL count :", np.sum(y_true_np == 0))
    print("True FAKE count :", np.sum(y_true_np == 1))
    print("Pred REAL count :", np.sum(y_pred_np == 0))
    print("Pred FAKE count :", np.sum(y_pred_np == 1))

    print("\n💾 Saved files:")
    print("Predictions   :", PREDICTIONS_CSV)
    print("Misclassified :", MISCLASSIFIED_CSV)
    print("Summary       :", SUMMARY_JSON)

    if BALANCE_EVAL:
        print("Selected idx  :", SELECTED_INDICES_JSON)


# ============================================================
# 🚀 MAIN
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("🔍 DEEPGUARD BALANCED CROSS-DATASET FUSION EVALUATION")
    print("=" * 80)

    print("CUDA available    :", torch.cuda.is_available())
    print("CUDA device count :", torch.cuda.device_count())
    print("Selected device   :", DEVICE)

    if torch.cuda.is_available():
        print("GPU name          :", torch.cuda.get_device_name(0))
    else:
        print("❌ GPU not detected. Evaluation will run on CPU.")

    print("Fusion checkpoint :", FUSION_CHECKPOINT_PATH)
    print("FFT mode          :", FFT_MODE)
    print("FFT num frames    :", FFT_NUM_FRAMES)
    print("Batch size        :", BATCH_SIZE)
    print("Balance eval      :", BALANCE_EVAL)
    print("Balance target    :", BALANCE_TARGET_PER_CLASS)

    test_loader, real_count, fake_count, selected_info = prepare_crossdataset_loader()

    model = load_fusion_model()

    y_true, y_prob, y_pred, rows = evaluate_crossdataset(
        model=model,
        test_loader=test_loader
    )

    if len(y_true) == 0:
        raise RuntimeError("❌ No valid samples evaluated.")

    compute_and_save_results(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        rows=rows,
        real_count=real_count,
        fake_count=fake_count,
        selected_info=selected_info
    )


if __name__ == "__main__":
    main()
