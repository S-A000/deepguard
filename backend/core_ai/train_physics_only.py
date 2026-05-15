# ==========================================
# 🌊 PHYSICS/PINN BRANCH TRAINING CODE
# ✅ Updated: Saves FULL model = PhysicsExpert + Classifier
# ✅ Kaggle/Jupyter safe
# ✅ Phase-wise dataset routing included
# ==========================================

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ==========================================
# ✅ KAGGLE / NOTEBOOK SAFE PATH SETUP
# ==========================================
PROJECT_ROOT = "/kaggle/working/deepguard"

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print("✅ Project root added:", PROJECT_ROOT)


# ==========================================
# 🧠 IMPORTS
# ==========================================
from backend.core_ai.models.branch_b_physics import PhysicsExpert, calculate_physics_penalty
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
# 🧠 PHYSICS-ONLY MODEL
# ==========================================
class PhysicsOnlyDeepGuard(nn.Module):
    def __init__(self, embed_dim=256):
        super(PhysicsOnlyDeepGuard, self).__init__()

        self.expert = PhysicsExpert(embed_dim=embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, flow_frames):
        features = self.expert(flow_frames)
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
    1. Full model checkpoint load karo.
    2. Agar full model nahi hai, expert-only checkpoint load karo.
       Is case mein classifier fresh rahega.
    """

    target_model = model.module if isinstance(model, nn.DataParallel) else model

    if full_model_path is not None and os.path.exists(full_model_path):
        print(f"\n🔁 Loading FULL previous phase model:")
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
        print(f"\n⚠️ Full previous phase model not found.")
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
def train_physics_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("🌊 Physics-Only PINN System Online")
    print("=" * 70)
    print(f"Training on: {device}")

    # ==========================================
    # 🗺️ PHASE-WISE DATASET ROUTING
    # ==========================================

    if CURRENT_PHASE == 1:
        print("\n🟢 PHASE 1: WARM-UP PHASE")
        print("Dataset: FaceForensics++ only")
        print("Goal: Basic real/fake optical-flow learning")

        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
        ]

        LR = 1e-4
        EPOCHS = 10
        BATCH_SIZE = 8
        SAMPLES_PER_CLASS = 1000

        ALPHA_DIV = 0.1
        BETA_SMOOTH = 0.1
        LAMBDA_PINN = 0.01

        PREV_FULL_MODEL_PATH = None
        PREV_EXPERT_ONLY_PATH = None

        SAVE_FULL_PATH = "/kaggle/working/saved_models/production/physics_phase1_full.pth"
        SAVE_EXPERT_PATH = "/kaggle/working/saved_models/production/physics_phase1_expert.pth"

    elif CURRENT_PHASE == 2:
        print("\n🟡 PHASE 2: BACKGROUND / NATURAL MOTION STABILITY")
        print("Datasets: FaceForensics++ + Kinetics + UCF101 + MSRVTT")
        print("Goal: Reduce false positives on natural real motion")

        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train",
            "/kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101",
            "/kaggle/input/datasets/abdullahpy/msrvtt/TrainValVideo",
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
        ]

        LR = 5e-5
        EPOCHS = 10
        BATCH_SIZE = 8
        SAMPLES_PER_CLASS = 1000

        ALPHA_DIV = 0.1
        BETA_SMOOTH = 0.1
        LAMBDA_PINN = 0.01

        PREV_FULL_MODEL_PATH = "/kaggle/working/saved_models/production/physics_phase1_full.pth"
        PREV_EXPERT_ONLY_PATH = "/kaggle/working/saved_models/production/physics_phase1_expert.pth"

        SAVE_FULL_PATH = "/kaggle/working/saved_models/production/physics_phase2_full.pth"
        SAVE_EXPERT_PATH = "/kaggle/working/saved_models/production/physics_phase2_expert.pth"

    elif CURRENT_PHASE == 3:
        print("\n🟠 PHASE 3: HARD FAKE PHASE")
        print("Datasets: FaceForensics++ + DFDC")
        print("Goal: Learn stronger fake motion inconsistencies")

        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real",
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
            "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake",
        ]

        LR = 1e-5
        EPOCHS = 10
        BATCH_SIZE = 8
        SAMPLES_PER_CLASS = 1000

        ALPHA_DIV = 0.1
        BETA_SMOOTH = 0.1
        LAMBDA_PINN = 0.01

        PREV_FULL_MODEL_PATH = "/kaggle/working/saved_models/production/physics_phase2_full.pth"
        PREV_EXPERT_ONLY_PATH = "/kaggle/working/saved_models/production/physics_phase2_expert.pth"

        SAVE_FULL_PATH = "/kaggle/working/saved_models/production/physics_phase3_full.pth"
        SAVE_EXPERT_PATH = "/kaggle/working/saved_models/production/physics_phase3_expert.pth"

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
        ]

        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
            "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18",

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
        BATCH_SIZE = 8
        SAMPLES_PER_CLASS = 1000

        ALPHA_DIV = 0.1
        BETA_SMOOTH = 0.1
        LAMBDA_PINN = 0.01

        PREV_FULL_MODEL_PATH = "/kaggle/working/saved_models/production/physics_phase3_full.pth"
        PREV_EXPERT_ONLY_PATH = "/kaggle/working/saved_models/production/physics_phase3_expert.pth"

        SAVE_FULL_PATH = "/kaggle/working/saved_models/production/physics_FINAL_full.pth"
        SAVE_EXPERT_PATH = "/kaggle/working/saved_models/production/physics_FINAL_expert.pth"

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
    print(f"Alpha Div         : {ALPHA_DIV}")
    print(f"Beta Smooth       : {BETA_SMOOTH}")
    print(f"Lambda PINN       : {LAMBDA_PINN}")
    print(f"Save Full Model   : {SAVE_FULL_PATH}")
    print(f"Save Expert Only  : {SAVE_EXPERT_PATH}")

    # ==========================================
    # 📦 DATASET LOADING
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

    balanced_dataset = ConcatDataset([real_dataset, fake_dataset])

    dataloader = DataLoader(
        balanced_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
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
    model = PhysicsOnlyDeepGuard(embed_dim=256).float().to(device)

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
    # 🔥 TRAINING LOOP WITH PINN LOSS
    # ==========================================
    for epoch in range(EPOCHS):
        model.train()

        epoch_total_loss = 0.0
        epoch_bce_loss = 0.0
        epoch_pinn_loss = 0.0

        loop = tqdm(dataloader, total=len(dataloader), leave=True)

        for batch_idx, (video_rgb, flow, fft, audio, labels) in enumerate(loop):

            flow = flow.to(device, non_blocking=True).float()
            flow = torch.nan_to_num(flow, nan=0.0, posinf=1.0, neginf=-1.0)

            labels = labels.float().to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    predictions = model(flow)

                    bce_loss = criterion(predictions, labels)

                    pinn_loss = calculate_physics_penalty(
                        optical_flow=flow,
                        alpha=ALPHA_DIV,
                        beta=BETA_SMOOTH
                    )

                    loss = bce_loss + (LAMBDA_PINN * pinn_loss)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

            else:
                predictions = model(flow)

                bce_loss = criterion(predictions, labels)

                pinn_loss = calculate_physics_penalty(
                    optical_flow=flow,
                    alpha=ALPHA_DIV,
                    beta=BETA_SMOOTH
                )

                loss = bce_loss + (LAMBDA_PINN * pinn_loss)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            epoch_total_loss += loss.item()
            epoch_bce_loss += bce_loss.item()
            epoch_pinn_loss += pinn_loss.item()

            loop.set_description(f"Phase {CURRENT_PHASE} | Epoch [{epoch + 1}/{EPOCHS}]")
            loop.set_postfix(
                BCE=f"{bce_loss.item():.4f}",
                PINN=f"{pinn_loss.item():.4f}",
                TOTAL=f"{loss.item():.4f}"
            )

        avg_total = epoch_total_loss / len(dataloader)
        avg_bce = epoch_bce_loss / len(dataloader)
        avg_pinn = epoch_pinn_loss / len(dataloader)

        print("\n" + "=" * 70)
        print(f"📊 PHASE {CURRENT_PHASE} | EPOCH {epoch + 1}/{EPOCHS} SUMMARY")
        print("=" * 70)
        print(f"Classification BCE Loss : {avg_bce:.6f}")
        print(f"PINN Kinematic Loss     : {avg_pinn:.6f}")
        print(f"Total Loss              : {avg_total:.6f}")
        print("=" * 70)

    # ==========================================
    # 💾 SAVE FULL MODEL + EXPERT ONLY
    # ==========================================
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model

    # ✅ This is the important one for evaluation
    torch.save(
        model_to_save.state_dict(),
        SAVE_FULL_PATH
    )

    # Optional: expert-only for final multimodal fusion use
    torch.save(
        model_to_save.expert.state_dict(),
        SAVE_EXPERT_PATH
    )

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print("✅ FULL model saved at:")
    print(SAVE_FULL_PATH)
    print("\n✅ Expert-only model also saved at:")
    print(SAVE_EXPERT_PATH)
    print("=" * 70)


# ==========================================
# ▶️ RUN
# ==========================================
train_physics_model()
