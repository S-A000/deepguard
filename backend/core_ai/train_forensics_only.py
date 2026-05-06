import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import warnings
from collections import OrderedDict

# System Configuration
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

from backend.core_ai.models.branch_c_forensics import ForensicExpert
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset

# ==========================================
# 🎛️ PHASE CONTROLLER (MASTER SWITCH)
# ==========================================
# Set this to 1, 2, 3, or 4 based on your training stage
CURRENT_PHASE = 2 

class ForensicsOnlyDeepGuard(nn.Module):
    def __init__(self, embed_dim=256):
        super(ForensicsOnlyDeepGuard, self).__init__()
        self.expert = ForensicExpert(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, fft_images):
        features = self.expert(fft_images)
        return self.classifier(features)

def train_forensics_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔍 DeepGuard Forensics Online! Mode: Phase {CURRENT_PHASE} | Device: {device}")

    # ==========================================
    # 🗺️ MULTI-PHASE DATASET ROUTING
    # ==========================================
    if CURRENT_PHASE == 1:
        print("🟢 PHASE 1: WARM-UP (FaceForensics++ Base)")
        REAL_DIRS = ["/kaggle/input/datasets/hungle3401/faceforensics/FF++/real"]
        FAKE_DIRS = ["/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake"]
        LR = 0.0001
        SAMPLES_PER_CLASS = 1000
        PREV_MODEL_PATH = None
        SAVE_PATH = "/kaggle/working/saved_models/production/forensic_phase1.pth"

    elif CURRENT_PHASE == 2:
        print("🟡 PHASE 2: BACKGROUND STABILITY (FF++ & Action Diversity)")
        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train",
            "/kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101",
            "/kaggle/input/datasets/abdullahpy/msrvtt/TrainValVideo"
        ]
        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1"
        ]
        LR = 0.00005
        SAMPLES_PER_CLASS = 2000
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/forensic_phase1.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/forensic_phase2.pth"

    elif CURRENT_PHASE == 3:
        print("🟠 PHASE 3: THE HARD FAKES (Enterprise DFDC Integration)")
        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real",
            "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics600_5per/kinetics600_5per/train"
        ]
        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake",
            "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18"
        ]
        LR = 0.00001
        SAMPLES_PER_CLASS = 3000
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/forensic_phase2.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/forensic_phase3.pth"

    elif CURRENT_PHASE == 4:
        print("🔴 PHASE 4: THE FUTURE THREATS (Custom + SoraGenVid + All)")
        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics600_5per/kinetics600_5per/train",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real",
            "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train",
            "/kaggle/input/datasets/abdullahpy/msrvtt/TrainValVideo",
            "/kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101"
        ]
        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/soragenvid/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake",
            "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18",
            "/kaggle/input/datasets/abdullahpy/ai-generated-video/Fake"
        ]
        LR = 0.00001
        SAMPLES_PER_CLASS = 4000
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/forensic_phase3.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/forensic_FINAL.pth"

    # Validation
    REAL_DIRS = [d for d in REAL_DIRS if os.path.exists(d)]
    FAKE_DIRS = [d for d in FAKE_DIRS if os.path.exists(d)]
    
    # Dataset Preparation
    real_dataset = DeepGuardDataset(real_dirs=REAL_DIRS, fake_dirs=[], max_samples=SAMPLES_PER_CLASS)
    fake_dataset = DeepGuardDataset(real_dirs=[], fake_dirs=FAKE_DIRS, max_samples=SAMPLES_PER_CLASS)
    
    balanced_dataset = ConcatDataset([real_dataset, fake_dataset])
    dataloader = DataLoader(balanced_dataset, batch_size=16, shuffle=True, 
                            num_workers=4, pin_memory=True, persistent_workers=True)

    # Model Init
    model = ForensicsOnlyDeepGuard().float().to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    # ==========================================
    # 🛠️ SMART CHECKPOINT LOADER
    # ==========================================
    if PREV_MODEL_PATH and os.path.exists(PREV_MODEL_PATH):
        try:
            state_dict = torch.load(PREV_MODEL_PATH, map_location=device)
            new_state_dict = OrderedDict()
            
            # Auto-detect DataParallel prefix mismatches
            curr_is_dp = isinstance(model, nn.DataParallel)
            file_is_dp = any(k.startswith('module.') for k in state_dict.keys())
            
            for k, v in state_dict.items():
                if curr_is_dp and not file_is_dp:
                    name = 'module.' + k
                elif not curr_is_dp and file_is_dp:
                    name = k[7:]
                else:
                    name = k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)
            print(f"✅ State Loaded! Continuing evolution from Phase {CURRENT_PHASE - 1}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")

    # Training Components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    # Loop Setup
    EPOCHS = 20 if CURRENT_PHASE > 2 else 10
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        for batch_idx, (video_rgb, flow, fft, audio, labels) in enumerate(loop):
            fft = torch.nan_to_num(fft.to(device, non_blocking=True))
            labels = labels.float().to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                predictions = model(fft)
                loss = criterion(predictions, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_description(f"Ph {CURRENT_PHASE} | Ep [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(BCE=f"{loss.item():.4f}")

    # Final Save (Always save clean dict without module. prefix)
    final_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(final_dict, SAVE_PATH)
    print(f"\n✅ Phase {CURRENT_PHASE} Done. Model saved at: {SAVE_PATH}")

if __name__ == "__main__":
    train_forensics_model()
