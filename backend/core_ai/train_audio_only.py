import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_num_threads(2)

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)

from backend.core_ai.models.branch_d_audio import AudioExpert
from custom_datasets.loaders.multi_modal_loader import DeepGuardDataset

CURRENT_PHASE = 1  # 🔥 START FROM PHASE 1

# ==========================================
# 🔥 STRONG AUGMENTATION (FIXED)
# ==========================================
def augment_waveform(audio):
    if torch.rand(1).item() > 0.5:
        # Time Mask
        t = int(audio.shape[-1] * 0.1)
        t0 = torch.randint(0, audio.shape[-1] - t, (1,)).item()
        audio[..., t0:t0+t] = 0

    if torch.rand(1).item() > 0.5:
        # Noise Injection
        noise = torch.randn_like(audio) * 0.005
        audio = audio + noise

    return audio


# ==========================================
# 🧠 MODEL
# ==========================================
class AudioOnlyDeepGuard(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.expert = AudioExpert(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.classifier(self.expert(x))


def train_audio_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎧 Training Phase {CURRENT_PHASE} on {device}")

    # ==========================================
    # DATASET
    # ==========================================
    REAL_DIRS = [
        "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Real/Real",
        "/kaggle/input/datasets/kynthesis/vctk-corpus/VCTK-Corpus/wav48"
    ]

    FAKE_DIRS = [
        "/kaggle/input/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices/Fake/Fake",
        "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_melgan",
        "/kaggle/input/datasets/andreadiubaldo/wavefake-test/generated_audio/ljspeech_parallel_wavegan"
    ]

    LR_BACKBONE = 5e-6
    LR_CLASSIFIER = 5e-5
    SAVE_PATH = "/kaggle/working/audio_phase1.pth"

    real_ds = DeepGuardDataset(real_dirs=REAL_DIRS, fake_dirs=[], mode="audio_only")
    fake_ds = DeepGuardDataset(real_dirs=[], fake_dirs=FAKE_DIRS, mode="audio_only")

    min_len = min(len(real_ds), len(fake_ds))
    real_ds.max_samples = min_len

    dataset = ConcatDataset([real_ds, fake_ds])

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    # ==========================================
    # MODEL
    # ==========================================
    model = AudioOnlyDeepGuard().to(device)

    if torch.cuda.device_count() > 1:
        print(f"🔥 Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    target_model = model.module if isinstance(model, nn.DataParallel) else model

    optimizer = optim.AdamW([
        {'params': target_model.expert.parameters(), 'lr': LR_BACKBONE},
        {'params': target_model.classifier.parameters(), 'lr': LR_CLASSIFIER}
    ], weight_decay=0.01)

    # 🔥 IMPORTANT FIX: Class imbalance handling
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))

    scaler = torch.cuda.amp.GradScaler()
    best_f1 = 0

    EPOCHS = 10

    # ==========================================
    # TRAIN LOOP
    # ==========================================
    for epoch in range(EPOCHS):

        # 🔥 Gradual Unfreezing
        if epoch < 2:
            for name, p in target_model.named_parameters():
                p.requires_grad = "classifier" in name
        else:
            for name, p in target_model.named_parameters():
                if any(f"encoder.layers.{i}" in name for i in range(8, 12)) or "classifier" in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        model.train()
        loop = tqdm(train_loader)

        for _, _, _, audio, labels in loop:

            audio = audio.cpu()
            audio = (audio - audio.mean(dim=-1, keepdim=True)) / (audio.std(dim=-1, keepdim=True) + 1e-8)
            audio = augment_waveform(audio)

            audio = audio.to(device)
            labels = labels.to(device).view(-1, 1)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                preds = model(audio)
                loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())

        # ==========================================
        # VALIDATION
        # ==========================================
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for _, _, _, audio, labels in val_loader:
                audio = audio.cpu()
                audio = (audio - audio.mean(dim=-1, keepdim=True)) / (audio.std(dim=-1, keepdim=True) + 1e-8)

                audio = audio.to(device)
                labels = labels.to(device).view(-1, 1)

                outputs = model(audio)
                probs = torch.sigmoid(outputs)

                # 🔥 THRESHOLD FIX (IMPORTANT)
                preds = (probs > 0.4).float()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds)
        print(f"\n📊 Epoch {epoch+1} F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(target_model.state_dict(), SAVE_PATH)
            print("🌟 Best model saved!")

    print(f"\n✅ Training Complete | Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    train_audio_model()
