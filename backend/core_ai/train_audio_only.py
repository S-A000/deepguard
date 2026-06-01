# ==========================================
# 🎧 DEEPGUARD AUDIO BRANCH TRAINING CODE
# ✅ Wav2Vec2 AudioExpert
# ✅ 4 Phase Strategy:
#    Phase 1-2: Audio-only datasets
#    Phase 3-4: Audio extracted from videos
# ✅ Corrupt audio/video safe
# ✅ No epoch-wise checkpoint saving
# ✅ Saves final FULL model + final EXPERT model
# ✅ Expert file is used later in Fusion Net
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

from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from tqdm import tqdm

from transformers import Wav2Vec2Model

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
# ✅ KAGGLE / SCRIPT SAFE PATH SETUP
# ==========================================
PROJECT_ROOT = "/kaggle/working/deepguard"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("✅ Project root added:", PROJECT_ROOT)
print("✅ Current working directory:", os.getcwd())


# ==========================================
# ✅ IMPORT DATASET LOADER
# ==========================================
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset


# ==========================================
# 🎛️ PHASE CONTROLLER
# ==========================================
# 1 = Audio-only warm-up
# 2 = Audio-only stronger training
# 3 = Video-audio domain adaptation
# 4 = Final video-audio generalization

CURRENT_PHASE = 1


# ==========================================
# ⚙️ GLOBAL SETTINGS
# ==========================================
SEED = 42
EMBED_DIM = 256
SAMPLE_RATE = 16000

# 4 seconds audio length for Wav2Vec2 stability
TARGET_AUDIO_SECONDS = 4
TARGET_AUDIO_LEN = SAMPLE_RATE * TARGET_AUDIO_SECONDS

DECISION_THRESHOLD = 0.5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ==========================================
# 🛡️ CORRUPT AUDIO / VIDEO SAFE DATASET WRAPPER
# ==========================================
class SafeDataset(Dataset):
    """
    Agar koi audio/video corrupt ho:
    - training crash nahi hogi
    - random valid sample try karega
    - max_retries ke baad error throw karega
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
                    raise RuntimeError("Dataset returned None sample")

                return sample

            except KeyboardInterrupt:
                raise

            except Exception as e:
                print(
                    f"\n⚠️ Bad audio/video skipped | {self.name} | "
                    f"original_idx={idx} | attempt={attempt + 1}/{self.max_retries} | "
                    f"error={str(e)[:200]}"
                )
                continue

        raise RuntimeError(
            f"❌ Too many corrupt/unreadable samples in {self.name} near index {idx}"
        )


# ==========================================
# 🎧 AUDIO HELPER FUNCTIONS
# ==========================================
def fix_audio_shape(audio):
    """
    Input possible:
    - (B, L)
    - (B, 1, L)
    - (B, C, L)

    Output:
    - (B, TARGET_AUDIO_LEN)
    """

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    if audio.dim() == 3:
        # If stereo/multi-channel, average channels
        audio = audio.mean(dim=1)

    if audio.dim() != 2:
        raise ValueError(f"Invalid audio shape: {audio.shape}")

    batch_size, audio_len = audio.shape

    if audio_len > TARGET_AUDIO_LEN:
        audio = audio[:, :TARGET_AUDIO_LEN]

    elif audio_len < TARGET_AUDIO_LEN:
        pad_len = TARGET_AUDIO_LEN - audio_len
        audio = torch.nn.functional.pad(audio, (0, pad_len))

    return audio


def normalize_audio(audio):
    """
    Per-sample normalization for Wav2Vec2 stability.
    """
    audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

    mean = audio.mean(dim=-1, keepdim=True)
    std = audio.std(dim=-1, keepdim=True)

    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    audio = (audio - mean) / std
    audio = torch.clamp(audio, -3.0, 3.0)

    return audio


def filter_bad_audio(audio, labels):
    """
    Batch ke andar bad audio remove karega.
    """
    audio = fix_audio_shape(audio)

    bad_mask = (
        torch.isnan(audio).any(dim=1)
        | torch.isinf(audio).any(dim=1)
        | (audio.abs().sum(dim=1) < 1e-6)
    )

    valid_mask = ~bad_mask

    if valid_mask.sum() == 0:
        return None, None

    return audio[valid_mask], labels[valid_mask]


# ==========================================
# 🎧 AUDIO EXPERT MODEL
# ==========================================
class AudioExpert(nn.Module):
    def __init__(self, embed_dim=256, phase=1):
        super(AudioExpert, self).__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

        hidden_size = self.wav2vec.config.hidden_size

        # Feature extractor mostly frozen for stability
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False

        total_layers = len(self.wav2vec.encoder.layers)

        # ==========================================
        # Phase-wise freezing strategy
        # ==========================================
        if phase == 1:
            # Phase 1: only last 4 transformer layers train
            for param in self.wav2vec.encoder.parameters():
                param.requires_grad = False

            unfreeze_from = max(0, total_layers - 4)

            for i in range(unfreeze_from, total_layers):
                for param in self.wav2vec.encoder.layers[i].parameters():
                    param.requires_grad = True

            print(f"🧊 Phase 1 freezing: last 4/{total_layers} Wav2Vec layers trainable")

        elif phase == 2:
            # Phase 2: last 6 transformer layers train
            for param in self.wav2vec.encoder.parameters():
                param.requires_grad = False

            unfreeze_from = max(0, total_layers - 6)

            for i in range(unfreeze_from, total_layers):
                for param in self.wav2vec.encoder.layers[i].parameters():
                    param.requires_grad = True

            print(f"🧊 Phase 2 freezing: last 6/{total_layers} Wav2Vec layers trainable")

        else:
            # Phase 3/4: full encoder trainable, feature extractor frozen
            for param in self.wav2vec.encoder.parameters():
                param.requires_grad = True

            print(f"🔥 Phase {phase}: full Wav2Vec encoder trainable")

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, audio_waveforms):
        """
        audio_waveforms shape:
        (batch, audio_length)
        """

        outputs = self.wav2vec(audio_waveforms)

        # Mean pooling over time
        pooled = outputs.last_hidden_state.mean(dim=1)

        embedding = self.projection(pooled)

        return embedding


# ==========================================
# 🎧 AUDIO-ONLY CLASSIFIER
# ==========================================
class AudioOnlyDeepGuard(nn.Module):
    def __init__(self, embed_dim=256, phase=1):
        super(AudioOnlyDeepGuard, self).__init__()

        self.expert = AudioExpert(
            embed_dim=embed_dim,
            phase=phase
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, audio_waveforms):
        features = self.expert(audio_waveforms)
        logits = self.classifier(features)
        return logits


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
# 🔁 CHECKPOINT LOADER
# ==========================================
def extract_state_dict(checkpoint):
    """
    Supports:
    - raw state_dict
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    """
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

    return checkpoint


def remove_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict

    clean_state_dict = {}

    for k, v in state_dict.items():
        clean_state_dict[k.replace("module.", "")] = v

    return clean_state_dict


def load_previous_phase_model(model, full_model_path, expert_only_path, device):
    """
    Priority:
    1. Full previous model load
    2. Expert-only previous model load
    """

    target_model = model.module if isinstance(model, nn.DataParallel) else model

    if full_model_path is not None and os.path.exists(full_model_path):
        print("\n🔁 Loading FULL previous phase model:")
        print(full_model_path)

        checkpoint = torch.load(full_model_path, map_location=device)
        state_dict = extract_state_dict(checkpoint)
        state_dict = remove_module_prefix(state_dict)

        missing, unexpected = target_model.load_state_dict(
            state_dict,
            strict=False
        )

        print("✅ Full previous phase model loaded.")
        print(f"Missing keys    : {len(missing)}")
        print(f"Unexpected keys : {len(unexpected)}")
        return

    if expert_only_path is not None and os.path.exists(expert_only_path):
        print("\n⚠️ Full previous phase model not found.")
        print("Loading EXPERT-ONLY previous checkpoint:")
        print(expert_only_path)

        checkpoint = torch.load(expert_only_path, map_location=device)
        state_dict = extract_state_dict(checkpoint)
        state_dict = remove_module_prefix(state_dict)

        missing, unexpected = target_model.expert.load_state_dict(
            state_dict,
            strict=False
        )

        print("✅ Expert-only checkpoint loaded.")
        print("Classifier will train fresh.")
        print(f"Missing keys    : {len(missing)}")
        print(f"Unexpected keys : {len(unexpected)}")
        return

    print("\n🆕 No previous checkpoint loaded. Training from fresh initialization.")


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
        video_rgb, flow, fft, audio, labels = batch

        labels = labels.float().view(-1, 1)

        audio, labels = filter_bad_audio(audio, labels)

        if audio is None:
            continue

        audio = audio.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).float()

        audio = normalize_audio(audio)

        optimizer.zero_grad(set_to_none=True)

        # AMP intentionally not used for Wav2Vec stability
        logits = model(audio)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

        loss = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ Invalid loss skipped.")
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

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
        video_rgb, flow, fft, audio, labels = batch

        labels = labels.float().view(-1, 1)

        audio, labels = filter_bad_audio(audio, labels)

        if audio is None:
            continue

        audio = audio.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).float()

        audio = normalize_audio(audio)

        logits = model(audio)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

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
def train_audio_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print("🎧 DeepGuard Audio Branch Training Started")
    print("=" * 80)
    print("CUDA available     :", torch.cuda.is_available())
    print("CUDA device count  :", torch.cuda.device_count())
    print("Selected device    :", device)

    if torch.cuda.is_available():
        print("GPU name           :", torch.cuda.get_device_name(0))
    else:
        print("❌ GPU not detected. Training will run on CPU.")

    # ==========================================
    # 🗺️ PHASE-WISE DATASET ROUTING
    # ==========================================

    if CURRENT_PHASE == 1:
        print("\n🟢 PHASE 1: AUDIO-ONLY WARM-UP")
        print("Datasets: Speech Human/AI + VCTK + WaveFake")

        REAL_DIRS = [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Real/Real",
            "/kaggle/input/datasets/kynthesis/vctk-corpus/VCTK-Corpus/wav48",
            "/kaggle/input/datasets/kynthesis/vctk-corpus/VCTK-Corpus/wav48_silence_trimmed",
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Fake/Fake",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_melgan",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_parallel_wavegan",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_waveglow",
        ]

        LR = 2e-5
        EPOCHS = 10
        BATCH_SIZE = 8
        SAMPLES_PER_CLASS = 8000
        NUM_WORKERS = 0

        PREV_FULL_MODEL_PATH = None
        PREV_EXPERT_ONLY_PATH = None

        SAVE_FULL_PATH = "/kaggle/working/saved_models/production/audio_phase1_full.pth"
        SAVE_EXPERT_PATH = "/kaggle/working/saved_models/production/audio_phase1_expert.pth"

    elif CURRENT_PHASE == 2:
        print("\n🟡 PHASE 2: AUDIO-ONLY STRONGER TRAINING")
        print("Datasets: Expanded audio-only real/fake speech")

        REAL_DIRS = [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Real/Real",
            "/kaggle/input/datasets/kynthesis/vctk-corpus/VCTK-Corpus/wav48",
            "/kaggle/input/datasets/kynthesis/vctk-corpus/VCTK-Corpus/wav48_silence_trimmed",
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Fake/Fake",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_melgan",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_parallel_wavegan",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_waveglow",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_full_band_melgan",
            "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_hifiGAN",
        ]

        LR = 1e-5
        EPOCHS = 8
        BATCH_SIZE = 8
        SAMPLES_PER_CLASS = 12000
        NUM_WORKERS = 0

        PREV_FULL_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase1_full.pth"
        PREV_EXPERT_ONLY_PATH = "/kaggle/working/saved_models/production/audio_phase1_expert.pth"

        SAVE_FULL_PATH = "/kaggle/working/saved_models/production/audio_phase2_full.pth"
        SAVE_EXPERT_PATH = "/kaggle/working/saved_models/production/audio_phase2_expert.pth"

    elif CURRENT_PHASE == 3:
        print("\n🟠 PHASE 3: VIDEO-AUDIO DOMAIN ADAPTATION")
        print("Datasets: FF++ + DFDC audio extracted from videos")

        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real",
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
            "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18",
        ]

        LR = 5e-6
        EPOCHS = 7
        BATCH_SIZE = 6
        SAMPLES_PER_CLASS = 3000
        NUM_WORKERS = 0

        PREV_FULL_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase2_full.pth"
        PREV_EXPERT_ONLY_PATH = "/kaggle/working/saved_models/production/audio_phase2_expert.pth"

        SAVE_FULL_PATH = "/kaggle/working/saved_models/production/audio_phase3_full.pth"
        SAVE_EXPERT_PATH = "/kaggle/working/saved_models/production/audio_phase3_expert.pth"

    elif CURRENT_PHASE == 4:
        print("\n🔴 PHASE 4: FINAL VIDEO-AUDIO GENERALIZATION")
        print("Datasets: DFDC + FF++ + AI Generated Video + Raw Fake AI")

        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real",
            "/kaggle/input/datasets/abdullahpy/msrvtt/TrainValVideo",
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
            "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18",

            # AI Generated Video / Raw Fake AI exact paths from your Kaggle datasets
            "/kaggle/input/datasets/abdullahpy/ai-generated-video/Fake",
            "/kaggle/input/datasets/abdullahpy/raw-fake-ai/Raw_reel",

            # fallback possible paths
            "/kaggle/input/ai-generated-video",
            "/kaggle/input/ai-generated-videos",
            "/kaggle/input/raw-fake-ai",
        ]

        LR = 3e-6
        EPOCHS = 7
        BATCH_SIZE = 6
        SAMPLES_PER_CLASS = 4000
        NUM_WORKERS = 0

        PREV_FULL_MODEL_PATH = "/kaggle/working/saved_models/production/audio_phase3_full.pth"
        PREV_EXPERT_ONLY_PATH = "/kaggle/working/saved_models/production/audio_phase3_expert.pth"

        SAVE_FULL_PATH = "/kaggle/working/saved_models/production/audio_FINAL_full.pth"
        SAVE_EXPERT_PATH = "/kaggle/working/saved_models/production/audio_FINAL_expert.pth"

    else:
        raise ValueError("❌ Invalid CURRENT_PHASE. Use 1, 2, 3, or 4.")

    # ==========================================
    # ✅ CHECK ACTIVE DATASET PATHS
    # ==========================================
    REAL_DIRS = clean_existing_dirs(REAL_DIRS)
    FAKE_DIRS = clean_existing_dirs(FAKE_DIRS)

    print("\n✅ Active Dataset Folders")

    print(f"\nReal folders: {len(REAL_DIRS)}")
    for d in REAL_DIRS:
        print(f"   REAL → {d}")

    print(f"\nFake folders: {len(FAKE_DIRS)}")
    for d in FAKE_DIRS:
        print(f"   FAKE → {d}")

    if len(REAL_DIRS) == 0:
        raise RuntimeError("❌ No REAL dataset folder found. Please check Kaggle paths.")

    if len(FAKE_DIRS) == 0:
        raise RuntimeError("❌ No FAKE dataset folder found. Please check Kaggle paths.")

    print("\n⚙️ Training Config")
    print(f"Phase             : {CURRENT_PHASE}")
    print(f"Epochs            : {EPOCHS}")
    print(f"Batch Size        : {BATCH_SIZE}")
    print(f"Learning Rate     : {LR}")
    print(f"Samples/Class     : {SAMPLES_PER_CLASS}")
    print(f"Num Workers       : {NUM_WORKERS}")
    print(f"Decision Threshold: {DECISION_THRESHOLD}")
    print(f"Save Full Model   : {SAVE_FULL_PATH}")
    print(f"Save Expert Only  : {SAVE_EXPERT_PATH}")

    # ==========================================
    # 📦 DATASET LOADING
    # ==========================================
    real_dataset_raw = DeepGuardDataset(
        real_dirs=REAL_DIRS,
        fake_dirs=[],
        num_frames=16,
        max_samples=SAMPLES_PER_CLASS,
        mode="audio_only"
    )

    fake_dataset_raw = DeepGuardDataset(
        real_dirs=[],
        fake_dirs=FAKE_DIRS,
        num_frames=16,
        max_samples=SAMPLES_PER_CLASS,
        mode="audio_only"
    )

    real_dataset = SafeDataset(
        base_dataset=real_dataset_raw,
        max_retries=25,
        name="REAL_AUDIO"
    )

    fake_dataset = SafeDataset(
        base_dataset=fake_dataset_raw,
        max_retries=25,
        name="FAKE_AUDIO"
    )

    # Split real/fake separately to preserve class balance
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
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=False
    )

    print("\n📊 Dataset Info")
    print(f"Real total        : {len(real_dataset)}")
    print(f"Fake total        : {len(fake_dataset)}")
    print(f"Train samples     : {len(train_dataset)}")
    print(f"Val samples       : {len(val_dataset)}")
    print(f"Train batches     : {len(train_loader)}")
    print(f"Val batches       : {len(val_loader)}")

    # ==========================================
    # 🧠 MODEL INITIALIZATION
    # ==========================================
    model = AudioOnlyDeepGuard(
        embed_dim=EMBED_DIM,
        phase=CURRENT_PHASE
    ).float().to(device)

    if torch.cuda.device_count() > 1:
        print(f"\n🚀 Multi-GPU detected: {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    if CURRENT_PHASE > 1:
        load_previous_phase_model(
            model=model,
            full_model_path=PREV_FULL_MODEL_PATH,
            expert_only_path=PREV_EXPERT_ONLY_PATH,
            device=device
        )
    else:
        print("\n🆕 Phase 1 fresh audio training started.")

    first_param = next(model.parameters())
    print("✅ Model dtype :", first_param.dtype)
    print("✅ Model device:", first_param.device)

    # ==========================================
    # 🎯 LOSS + OPTIMIZER + SCHEDULER
    # ==========================================
    # positive class = fake = 1
    # pos_weight = num_real / num_fake
    pos_weight_value = len(real_train) / max(len(fake_train), 1)
    pos_weight_value = max(0.5, min(pos_weight_value, 3.0))

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], device=device)
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-7
    )

    print(f"✅ BCE pos_weight: {pos_weight_value:.4f}")

    os.makedirs(os.path.dirname(SAVE_FULL_PATH), exist_ok=True)

    # ==========================================
    # 🔥 TRAINING LOOP
    # ==========================================
    best_auc = -1.0
    best_f1 = -1.0
    best_state_dict = None
    best_epoch = 0
    best_report_data = None

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 80)
        print(f"🎧 AUDIO PHASE {CURRENT_PHASE} | EPOCH {epoch}/{EPOCHS}")
        print("=" * 80)

        train_loss, train_acc, train_precision, train_recall, train_f1, train_auc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, y_true, y_prob, y_pred = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step()

        print("\n📊 Epoch Summary")
        print(f"Train Loss      : {train_loss:.6f}")
        print(f"Train Accuracy  : {train_acc:.4f}")
        print(f"Train Precision : {train_precision:.4f}")
        print(f"Train Recall    : {train_recall:.4f}")
        print(f"Train F1        : {train_f1:.4f}")
        print(f"Train ROC-AUC   : {train_auc:.4f}")

        print("\nValidation")
        print(f"Val Loss        : {val_loss:.6f}")
        print(f"Val Accuracy    : {val_acc:.4f}")
        print(f"Val Precision   : {val_precision:.4f}")
        print(f"Val Recall      : {val_recall:.4f}")
        print(f"Val F1          : {val_f1:.4f}")
        print(f"Val ROC-AUC     : {val_auc:.4f}")

        # Best model selection
        # Primary: AUC, Secondary: F1
        if (val_auc > best_auc) or (val_auc == best_auc and val_f1 > best_f1):
            best_auc = val_auc
            best_f1 = val_f1
            best_epoch = epoch

            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            best_state_dict = copy.deepcopy(model_to_save.state_dict())

            best_report_data = {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "val_auc": val_auc,
            }

            print(f"\n✅ New best model found in memory | Epoch {epoch} | AUC={val_auc:.4f} | F1={val_f1:.4f}")

        # Important:
        # No checkpoint is saved here.
        # Final files are saved only once after all epochs.

    # ==========================================
    # 💾 SAVE FINAL BEST FULL MODEL + FINAL BEST EXPERT ONLY
    # ==========================================
    if best_state_dict is None:
        raise RuntimeError("❌ No valid best model state found. Training failed.")

    final_model = AudioOnlyDeepGuard(
        embed_dim=EMBED_DIM,
        phase=CURRENT_PHASE
    ).float()

    final_model.load_state_dict(best_state_dict, strict=True)

    # For standalone audio branch evaluation
    torch.save(
        final_model.state_dict(),
        SAVE_FULL_PATH
    )

    # For final multimodal fusion
    torch.save(
        final_model.expert.state_dict(),
        SAVE_EXPERT_PATH
    )

    print("\n" + "=" * 80)
    print("✅ AUDIO TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Epoch      : {best_epoch}")
    print(f"Best Val AUC    : {best_auc:.4f}")
    print(f"Best Val F1     : {best_f1:.4f}")

    print("\n✅ FINAL full audio model saved at:")
    print(SAVE_FULL_PATH)

    print("\n✅ FINAL audio expert-only model saved for fusion at:")
    print(SAVE_EXPERT_PATH)

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
    train_audio_model()
