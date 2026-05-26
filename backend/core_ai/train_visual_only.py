# ==========================================
# 👁️ VISUAL / SPATIAL-TEMPORAL BRANCH TRAINING CODE
# ✅ TimeSformer VisualExpert
# ✅ Saves FULL model = VisualExpert + Classifier
# ✅ Saves Expert-only model for final fusion
# ✅ Kaggle / script safe
# ✅ Phase-wise dataset routing
# ✅ Corrupt video safe
# ==========================================

import os
import sys
import random
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from tqdm import tqdm

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
# 🧠 IMPORTS
# ==========================================
from backend.core_ai.models.branch_a_spatial import VisualExpert
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset


# ==========================================
# 🎛️ PHASE CONTROLLER
# ==========================================
# 1 = FaceForensics++ warm-up
# 2 = Real-world action/background stability
# 3 = DFDC hard fake training
# 4 = AI generated future fake training

CURRENT_PHASE = 1


# ==========================================
# 🛡️ CORRUPT VIDEO SAFE DATASET WRAPPER
# ==========================================
class SafeDataset(Dataset):
    def __init__(self, base_dataset, max_retries=20, name="dataset"):
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
                    f"\n⚠️ Bad sample skipped | {self.name} | "
                    f"original_idx={idx} | attempt={attempt + 1}/{self.max_retries} | "
                    f"error={str(e)[:200]}"
                )
                continue

        raise RuntimeError(
            f"❌ Too many corrupt/unreadable samples in {self.name} near index {idx}"
        )


# ==========================================
# 👁️ VISUAL-ONLY MODEL
# ==========================================
class VisualOnlyDeepGuard(nn.Module):
    def __init__(self, embed_dim=256):
        super(VisualOnlyDeepGuard, self).__init__()

        self.expert = VisualExpert(embed_dim=embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, video_rgb):
        features = self.expert(video_rgb)
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
def load_previous_phase_model(model, full_model_path, expert_only_path, device):
    """
    Priority:
    1. Full previous model load.
    2. Expert-only previous model load.
    """

    target_model = model.module if isinstance(model, nn.DataParallel) else model

    if full_model_path is not None and os.path.exists(full_model_path):
        print("\n🔁 Loading FULL previous phase model:")
        print(full_model_path)

        state_dict = torch.load(full_model_path, map_location=device)

        if any(k.startswith("module.") for k in state_dict.keys()):
            clean_state_dict = {}
            for k, v in state_dict.items():
                clean_state_dict[k.replace("module.", "")] = v
            state_dict = clean_state_dict

        target_model.load_state_dict(state_dict, strict=True)

        print("✅ Full previous phase model loaded successfully.")
        return

    if expert_only_path is not None and os.path.exists(expert_only_path):
        print("\n⚠️ Full previous phase model not found.")
        print("Loading EXPERT-ONLY previous checkpoint:")
        print(expert_only_path)

        state_dict = torch.load(expert_only_path, map_location=device)

        if any(k.startswith("module.") for k in state_dict.keys()):
            clean_state_dict = {}
            for k, v in state_dict.items():
                clean_state_dict[k.replace("module.", "")] = v
            state_dict = clean_state_dict

        target_model.expert.load_state_dict(state_dict, strict=True)

        print("✅ Expert-only checkpoint loaded. Classifier will train fresh.")
        return

    print("\n🆕 No previous checkpoint loaded. Training from fresh initialization.")


# ==========================================
# 🚀 TRAINING FUNCTION
# ==========================================
def train_visual_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("👁️ Visual-Only TimeSformer System Online")
    print("=" * 70)
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
        print("\n🟢 PHASE 1: WARM-UP PHASE")
        print("Dataset: FaceForensics++ only")
        print("Goal: Basic visual deepfake learning")

        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
        ]

        LR = 1e-4
        EPOCHS = 10
        BATCH_SIZE = 4
        SAMPLES_PER_CLASS = 1000
        NUM_WORKERS = 0

        PREV_FULL_MODEL_PATH = None
        PREV_EXPERT_ONLY_PATH = None

        SAVE_FULL_PATH = "/kaggle/working/saved_models/production/visual_phase1_full.pth"
        SAVE_EXPERT_PATH = "/kaggle/working/saved_models/production/visual_phase1_expert.pth"

    elif CURRENT_PHASE == 2:
        print("\n🟡 PHASE 2: BACKGROUND / NATURAL MOTION STABILITY")
        print("Datasets: FaceForensics++ + Kinetics + UCF101 + MSRVTT")
        print("Goal: Reduce false positives on natural real videos")

        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train",
            "/kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101",
            "/kaggle/input/datasets/abdullahpy/msrvtt/TrainValVideo",
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake",
        ]

        LR = 5e-5
        EPOCHS = 10
        BATCH_SIZE = 2
        SAMPLES_PER_CLASS = 2000
        NUM_WORKERS = 0

        PREV_FULL_MODEL_PATH = "/kaggle/working/saved_models/production/visual_phase1_full.pth"
        PREV_EXPERT_ONLY_PATH = "/kaggle/working/saved_models/production/visual_phase1_expert.pth"

        SAVE_FULL_PATH = "/kaggle/working/saved_models/production/visual_phase2_full.pth"
        SAVE_EXPERT_PATH = "/kaggle/working/saved_models/production/visual_phase2_expert.pth"

    elif CURRENT_PHASE == 3:
        print("\n🟠 PHASE 3: HARD FAKE PHASE")
        print("Datasets: FaceForensics++ + DFDC")
        print("Goal: Learn harder visual fake artifacts")

        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real",
            "/kaggle/input/datasets/abdullahpy/msrvtt/TrainValVideo",
            "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics600_5per/kinetics600_5per"
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
            "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake",
        ]

        LR = 1e-5
        EPOCHS = 10
        BATCH_SIZE = 4
        SAMPLES_PER_CLASS = 2000
        NUM_WORKERS = 0

        PREV_FULL_MODEL_PATH = "/kaggle/working/saved_models/production/visual_phase2_full.pth"
        PREV_EXPERT_ONLY_PATH = "/kaggle/working/saved_models/production/visual_phase2_expert.pth"

        SAVE_FULL_PATH = "/kaggle/working/saved_models/production/visual_phase3_full.pth"
        SAVE_EXPERT_PATH = "/kaggle/working/saved_models/production/visual_phase3_expert.pth"

    elif CURRENT_PHASE == 4:
        print("\n🔴 PHASE 4: AI-GENERATED FUTURE THREAT PHASE")
        print("Datasets: FaceForensics++ + DFDC + real action datasets + AI generated videos")
        print("Goal: Generalization against modern generative video")

        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train",
            "/kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101",
            "/kaggle/input/datasets/abdullahpy/msrvtt/TrainValVideo",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real",
            "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics600_5per/kinetics600_5per"
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
            "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18",
            "/kaggle/input/datasets/abdullahpy/ai-generated-video/Fake",
            "/kaggle/input/datasets/abdullahpy/raw-fake-ai/Raw_reel"

            # AI Generated Video possible paths
            "/kaggle/input/ai-generated-video",
            "/kaggle/input/ai-generated-videos",
            "/kaggle/input/ai-generated-video-dataset",

            # Raw fake AI possible paths
            "/kaggle/input/raw-fake-ai",
            "/kaggle/input/raw-fake-ai-video",
            "/kaggle/input/raw-fake-ai-dataset",
        ]

        LR = 1e-5
        EPOCHS = 10
        BATCH_SIZE = 4
        SAMPLES_PER_CLASS = 4000
        NUM_WORKERS = 0

        PREV_FULL_MODEL_PATH = "/kaggle/working/saved_models/production/visual_phase3_full.pth"
        PREV_EXPERT_ONLY_PATH = "/kaggle/working/saved_models/production/visual_phase3_expert.pth"

        SAVE_FULL_PATH = "/kaggle/working/saved_models/production/visual_FINAL_full.pth"
        SAVE_EXPERT_PATH = "/kaggle/working/saved_models/production/visual_FINAL_expert.pth"

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
    print(f"Save Full Model   : {SAVE_FULL_PATH}")
    print(f"Save Expert Only  : {SAVE_EXPERT_PATH}")

    # ==========================================
    # 📦 DATASET LOADING
    # ==========================================
    real_dataset_raw = DeepGuardDataset(
        real_dirs=REAL_DIRS,
        fake_dirs=[],
        max_samples=SAMPLES_PER_CLASS
    )

    fake_dataset_raw = DeepGuardDataset(
        real_dirs=[],
        fake_dirs=FAKE_DIRS,
        max_samples=SAMPLES_PER_CLASS
    )

    real_dataset = SafeDataset(
        base_dataset=real_dataset_raw,
        max_retries=20,
        name="REAL"
    )

    fake_dataset = SafeDataset(
        base_dataset=fake_dataset_raw,
        max_retries=20,
        name="FAKE"
    )

    balanced_dataset = ConcatDataset([real_dataset, fake_dataset])

    dataloader = DataLoader(
        balanced_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True
    )

    print("\n📊 Dataset Info")
    print(f"Real samples      : {len(real_dataset)}")
    print(f"Fake samples      : {len(fake_dataset)}")
    print(f"Total samples     : {len(balanced_dataset)}")
    print(f"Total batches     : {len(dataloader)}")

    # ==========================================
    # 🧠 MODEL INITIALIZATION
    # ==========================================
    model = VisualOnlyDeepGuard(embed_dim=256).float().to(device)

    if torch.cuda.device_count() > 1:
        print(f"\n🚀 Multi-GPU detected: {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ==========================================
    # 🔁 LOAD PREVIOUS PHASE
    # ==========================================
    if CURRENT_PHASE > 1:
        load_previous_phase_model(
            model=model,
            full_model_path=PREV_FULL_MODEL_PATH,
            expert_only_path=PREV_EXPERT_ONLY_PATH,
            device=device
        )
    else:
        print("\n🆕 Phase 1 fresh training started.")

    # ==========================================
    # 🎯 LOSS + OPTIMIZER
    # ==========================================
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", init_scale=128)
    else:
        scaler = None

    os.makedirs(os.path.dirname(SAVE_FULL_PATH), exist_ok=True)

    # ==========================================
    # 🔥 TRAINING LOOP
    # ==========================================
    for epoch in range(EPOCHS):
        model.train()

        epoch_bce_loss = 0.0

        loop = tqdm(dataloader, total=len(dataloader), leave=True)

        for batch_idx, batch in enumerate(loop):
            video_rgb, flow, fft, audio, labels = batch

            video_rgb = video_rgb.to(device, non_blocking=True).float()
            video_rgb = torch.nan_to_num(video_rgb, nan=0.0, posinf=1.0, neginf=-1.0)

            labels = labels.float().to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    predictions = model(video_rgb)
                    loss = criterion(predictions, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

            else:
                predictions = model(video_rgb)
                loss = criterion(predictions, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            epoch_bce_loss += loss.item()

            loop.set_description(f"Phase {CURRENT_PHASE} | Epoch [{epoch + 1}/{EPOCHS}]")
            loop.set_postfix(BCE=f"{loss.item():.4f}")

        avg_bce = epoch_bce_loss / len(dataloader)

        print("\n" + "=" * 70)
        print(f"📊 VISUAL PHASE {CURRENT_PHASE} | EPOCH {epoch + 1}/{EPOCHS} SUMMARY")
        print("=" * 70)
        print(f"Classification BCE Loss : {avg_bce:.6f}")
        print("=" * 70)

        # Epoch safety checkpoint
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        epoch_ckpt_path = SAVE_FULL_PATH.replace(".pth", f"_epoch{epoch + 1}.pth")

        torch.save(
            model_to_save.state_dict(),
            epoch_ckpt_path
        )

        print(f"💾 Epoch checkpoint saved: {epoch_ckpt_path}")

    # ==========================================
    # 💾 SAVE FULL MODEL + EXPERT ONLY
    # ==========================================
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model

    # For standalone visual branch evaluation
    torch.save(
        model_to_save.state_dict(),
        SAVE_FULL_PATH
    )

    # For final multimodal fusion
    torch.save(
        model_to_save.expert.state_dict(),
        SAVE_EXPERT_PATH
    )

    print("\n" + "=" * 70)
    print("✅ Visual Training Complete!")
    print("=" * 70)
    print("✅ FULL visual model saved at:")
    print(SAVE_FULL_PATH)
    print("\n✅ Visual expert-only model also saved at:")
    print(SAVE_EXPERT_PATH)
    print("=" * 70)


# ==========================================
# ▶️ RUN
# ==========================================
if __name__ == "__main__":
    train_visual_model()
