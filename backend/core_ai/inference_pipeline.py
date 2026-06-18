import os
import sys
import torch
import numpy as np

# Path setup taake scripts folder mil jaye
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from backend.core_ai.models.fusion_net import DeepGuardFusionModel
from scripts.video_to_frames import process_video_for_ai

MODEL_WEIGHTS_PATH = os.path.abspath(
    os.path.join(
        current_dir,
        "../../saved_models/production/fusion_FINAL_best_multiframe_fft.pth"
    )
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_deepguard_model = None
_is_mock_mode = True


def clean_state_dict(state_dict):
    """
    Checkpoint ko model.load_state_dict ke qabil banata hai.
    DataParallel ka module. prefix bhi remove karta hai.
    """
    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        raise ValueError("Invalid checkpoint format. State dict dictionary nahi mila.")

    # DataParallel prefix remove
    if any(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    return state_dict


def load_model():
    global _deepguard_model, _is_mock_mode

    if _deepguard_model is not None and _is_mock_mode is False:
        return

    try:
        print(f"🧠 Loading DeepGuard model on device: {device}")

        _deepguard_model = DeepGuardFusionModel().to(device)

        if not os.path.exists(MODEL_WEIGHTS_PATH):
            print(f"[X] Model .pth file nahi mili at: {MODEL_WEIGHTS_PATH}")
            _deepguard_model = None
            _is_mock_mode = True
            return

        print(f"[✓] Model file found: {MODEL_WEIGHTS_PATH}")

        checkpoint = torch.load(
            MODEL_WEIGHTS_PATH,
            map_location=device,
            weights_only=False
        )

        state_dict = clean_state_dict(checkpoint)

        missing_keys, unexpected_keys = _deepguard_model.load_state_dict(
            state_dict,
            strict=False
        )

        if missing_keys:
            print(f"⚠️ Missing keys count: {len(missing_keys)}")
            print("⚠️ First missing keys:", missing_keys[:10])

        if unexpected_keys:
            print(f"⚠️ Unexpected keys count: {len(unexpected_keys)}")
            print("⚠️ First unexpected keys:", unexpected_keys[:10])

        _deepguard_model.eval()
        _is_mock_mode = False

        print("[✓] Asli Trained Model (.pth) loaded successfully!")
        print("[✓] Mock/random mode disabled.")

    except Exception as e:
        print(f"[X] Error loading model structure: {e}")
        print("[X] Model load nahi hua. Mock/random result disabled rahega.")
        _deepguard_model = None
        _is_mock_mode = True


def analyze_video(video_path: str) -> dict:
    """
    AI Model se asal video scan karwa kar result deta hai.
    REAL = 0, FAKE = 1
    Isliye sigmoid output ko fake_probability treat karna hai.
    """
    load_model()

    if _is_mock_mode or _deepguard_model is None:
        return {
            "status": "error",
            "verdict": "MODEL_NOT_LOADED",
            "confidence": 0.0,
            "fake_probability": 0.0,
            "real_probability": 0.0,
            "details": "Model load nahi hua. Terminal error check karo. Random/mock result disabled hai.",
            "branch_scores": {
                "spatial": 0.0,
                "physics": 0.0,
                "forensics": 0.0,
                "audio": 0.0
            }
        }

    audio_path = None

    try:
        print(f"🎬 Processing Asli Video: {video_path}")

        # 1. Asal Video se Frames, Optical Flow, FFT aur Audio nikalna
        frames_np, flow_np, fft_np, audio_np, audio_path = process_video_for_ai(
            video_path,
            num_frames=16
        )

        # 2. Visual Tensor
        # Expected by TimeSformer: (B, T, C, H, W)
        # Shape: (16, 224, 224, 3) -> (1, 16, 3, 224, 224)
        video_tensor = torch.from_numpy(frames_np).float() / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
        video_tensor = torch.nan_to_num(
            video_tensor,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        ).to(device)

        # 3. Physics Tensor
        # Expected: (B, 2, H, W)
        # Shape: (2, 224, 224) -> (1, 2, 224, 224)
        flow_tensor = torch.from_numpy(flow_np).float().unsqueeze(0)
        flow_tensor = torch.nan_to_num(
            flow_tensor,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        ).to(device)

        # 4. Forensic FFT Tensor
# Training loader output: (1, 224, 224)
# Model expects: (B, 1, 224, 224)
        fft_tensor = torch.from_numpy(fft_np).float().unsqueeze(0)
        fft_tensor = torch.nan_to_num(
           fft_tensor,
           nan=0.0,
           posinf=1.0,
           neginf=0.0
        ).to(device)

        # 5. Audio Tensor
        # AudioExpert/Wav2Vec expects: (B, L)
        audio_tensor = torch.from_numpy(audio_np).float()

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        audio_tensor = torch.nan_to_num(
            audio_tensor,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        ).to(device)

        # 6. Inference
        with torch.no_grad():
            probability_tensor = _deepguard_model(
                video_tensor,
                flow_tensor,
                fft_tensor,
                audio_tensor
            )

            prob = torch.sigmoid(probability_tensor).mean().item()

        # Safety clamp
        prob = float(np.clip(prob, 0.0, 1.0))

        # Training code ke hisaab se:
        # REAL = 0
        # FAKE = 1
        fake_probability = round(prob * 100, 2)
        real_probability = round((1.0 - prob) * 100, 2)

        is_fake = bool(prob >= 0.5)
        verdict = "FAKE" if is_fake else "REAL"
        confidence = fake_probability if is_fake else real_probability

        print(
            f"✅ Result: {verdict} | "
            f"Fake={fake_probability}% | Real={real_probability}% | RawProb={round(prob, 6)}"
        )

        return {
            "status": "success",
            "verdict": verdict,
            "confidence": confidence,
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "raw_probability": round(prob, 6),

            # Ye exact branch-wise classifier scores nahi hain.
            # Abhi same final fake probability de rahe hain taake random na ho.
            "branch_scores": {
                "spatial": fake_probability,
                "physics": fake_probability,
                "forensics": fake_probability,
                "audio": fake_probability
            }
        }

    except Exception as e:
        print(f"[X] Error during inference: {e}")

        return {
            "status": "error",
            "verdict": "INFERENCE_FAILED",
            "confidence": 0.0,
            "fake_probability": 0.0,
            "real_probability": 0.0,
            "details": str(e),
            "branch_scores": {
                "spatial": 0.0,
                "physics": 0.0,
                "forensics": 0.0,
                "audio": 0.0
            }
        }

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)