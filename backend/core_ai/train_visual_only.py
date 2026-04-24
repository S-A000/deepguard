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

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

from backend.core_ai.models.branch_a_spatial import VisualExpert
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset

# ==========================================
# 🎛️ PHASE CONTROLLER (MASTER SWITCH)
# ==========================================
# 1 = FF++ | 2 = Kinetics/UCF | 3 = DFDC | 4 = Custom + Sora
CURRENT_PHASE = 1  

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
        return self.classifier(features)

def train_visual_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n👁️ Visual-Only System Online! Training on: {device}")

    # ==========================================
    # 🗺️ 4-PHASE DATASET ROUTING
    # ==========================================
    if CURRENT_PHASE == 1:
        print("🟢 PHASE 1: WARM-UP (FaceForensics++ Only)")
        REAL_DIRS = ["/kaggle/input/datasets/hungle3401/faceforensics/FF++/real"]
        FAKE_DIRS = ["/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake"]
        LR = 0.0001
        PREV_MODEL_PATH = None
        SAVE_PATH = "/kaggle/working/saved_models/production/visual_phase1.pth"

    elif CURRENT_PHASE == 2:
        print("🟡 PHASE 2: BACKGROUND STABILITY (FF++ & Action Datasets)")
        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per/kinetics400_5per/train",
            "/kaggle/input/datasets/pevogam/ucf101/UCF101/UCF-101",
            "/kaggle/input/datasets/abdullahpy/msrvtt/TrainValVideo"
        ]
        FAKE_DIRS = ["/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake"]
        LR = 0.00005  # Thora kam taake purani memory erase na ho
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/visual_phase1.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/visual_phase2.pth"

    elif CURRENT_PHASE == 3:
        print("🟠 PHASE 3: THE HARD FAKES (Adding DFDC Enterprise Data)")
        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/real"
        ]
        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/datasets/zz14423/dfdc-part-01/dfdc_train_part_1",
            "/kaggle/input/datasets/aknirala/dfdc-train-part-18/dfdc_train_part_18",
            "/kaggle/input/datasets/krishna191919/dfdc-part-14/dfdc_equal_split_part_14/fake"
        ]
        LR = 0.00001
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/visual_phase2.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/visual_phase3.pth"

    elif CURRENT_PHASE == 4:
        print("🔴 PHASE 4: THE FUTURE THREATS (Custom + SoraGenVid Placeholder)")
        # Baaki purane paths bhi add kar sakte hain, abhi sirf placeholders hain
        REAL_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/real",
            "/kaggle/input/custom_dataset/real" # 🚧 Empty Placeholder
        ]
        FAKE_DIRS = [
            "/kaggle/input/datasets/hungle3401/faceforensics/FF++/fake",
            "/kaggle/input/soragenvid/fake"     # 🚧 Empty Placeholder
        ]
        LR = 0.00001
        PREV_MODEL_PATH = "/kaggle/working/saved_models/production/visual_phase3.pth"
        SAVE_PATH = "/kaggle/working/saved_models/production/visual_FINAL.pth"

    else:
        print("❌ Invalid Phase Selected!")
        return

    # Filter out empty paths dynamically
    REAL_DIRS = [d for d in REAL_DIRS if os.path.exists(d)]
    FAKE_DIRS = [d for d in FAKE_DIRS if os.path.exists(d)]
    print(f"✅ Active Folders - Real: {len(REAL_DIRS)} | Fake: {len(FAKE_DIRS)}")

    # ==========================================
    # 🧠 MODEL INITIALIZATION & LOADING
    # ==========================================
    SAMPLES_PER_CLASS = 1000 
    real_dataset = DeepGuardDataset(real_dirs=REAL_DIRS, fake_dirs=[], max_samples=SAMPLES_PER_CLASS)
    fake_dataset = DeepGuardDataset(real_dirs=[], fake_dirs=FAKE_DIRS, max_samples=SAMPLES_PER_CLASS)
    
    balanced_dataset = ConcatDataset([real_dataset, fake_dataset])
    dataloader = DataLoader(balanced_dataset, batch_size=4, shuffle=True, num_workers=2)

    model = VisualOnlyDeepGuard().float().to(device)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)

    # 🚀 Load previous phase memory if moving to phase 2, 3, or 4
    if PREV_MODEL_PATH and os.path.exists(PREV_MODEL_PATH):
        try:
            target = model.module.expert if isinstance(model, nn.DataParallel) else model.expert
            target.load_state_dict(torch.load(PREV_MODEL_PATH, map_location=device))
            print(f"✅ Memory Loaded from Phase {CURRENT_PHASE - 1}! Continuing evolution...")
        except Exception as e:
            print(f"⚠️ Error loading previous phase memory: {e}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda', init_scale=128)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # ==========================================
    # 🔥 TRAINING LOOP
    # ==========================================
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        
        for batch_idx, (video_rgb, flow, fft, audio, labels) in enumerate(loop):
            video_rgb = torch.nan_to_num(video_rgb.to(device))
            labels = labels.float().to(device).view(-1, 1)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                predictions = model(video_rgb)
                loss = criterion(predictions, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(BCE=f"{loss.item():.4f}")

    # Save only the Expert weights
    torch.save(model.module.expert.state_dict() if isinstance(model, nn.DataParallel) else model.expert.state_dict(), SAVE_PATH)
    print(f"\n✅ Phase {CURRENT_PHASE} Complete! Model Saved at: {SAVE_PATH}")

if __name__ == "__main__":
    train_visual_model()