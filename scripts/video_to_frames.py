import os
import cv2
import numpy as np
import librosa
import tempfile
import subprocess


def extract_frames(video_path, num_frames=16):
    """
    Training loader jaisa:
    Video ke first continuous 16 frames nikalta hai.
    Output: (16, 224, 224, 3)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    while len(frames) < num_frames:
        frames.append(
            frames[-1] if len(frames) > 0
            else np.zeros((224, 224, 3), dtype=np.uint8)
        )

    return np.array(frames, dtype=np.uint8)


def extract_optical_flow(frames_np):
    """
    Training loader jaisa:
    Sirf first 2 frames ke darmiyan optical flow.
    Output: (2, 224, 224)
    """
    if frames_np is None or len(frames_np) < 2:
        return np.zeros((2, 224, 224), dtype=np.float32)

    gray1 = cv2.cvtColor(frames_np[0], cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frames_np[1], cv2.COLOR_RGB2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    flow = np.nan_to_num(flow, nan=0.0, posinf=1.0, neginf=-1.0)
    flow = np.transpose(flow, (2, 0, 1)).astype(np.float32)

    return flow


def single_frame_fft(frame_rgb):
    """
    Training loader ka same FFT helper.
    Input: (224, 224, 3)
    Output: (224, 224)
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32)

    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)

    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)

    max_val = np.max(magnitude_spectrum)

    if max_val < 1e-8:
        max_val = 1.0

    magnitude_spectrum = magnitude_spectrum / (max_val + 1e-8)

    magnitude_spectrum = np.nan_to_num(
        magnitude_spectrum,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    magnitude_spectrum = np.clip(magnitude_spectrum, 0.0, 1.0)

    return magnitude_spectrum.astype(np.float32)


def extract_fft(frames_np, fft_num_frames=8):
    """
    Training loader jaisa multi_avg FFT.
    Output: (1, 224, 224)
    """
    if frames_np is None or len(frames_np) == 0:
        return np.zeros((1, 224, 224), dtype=np.float32)

    total_frames = len(frames_np)
    k = min(fft_num_frames, total_frames)

    indices = np.linspace(0, total_frames - 1, k).astype(int)

    fft_maps = []

    for idx in indices:
        try:
            fft_map = single_frame_fft(frames_np[idx])
            fft_maps.append(fft_map)
        except Exception:
            continue

    if len(fft_maps) == 0:
        return np.zeros((1, 224, 224), dtype=np.float32)

    avg_fft = np.mean(np.stack(fft_maps, axis=0), axis=0)

    avg_fft = np.nan_to_num(
        avg_fft,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    avg_fft = np.clip(avg_fft, 0.0, 1.0)

    return np.expand_dims(avg_fft.astype(np.float32), axis=0)


def extract_audio(video_path):
    """
    Training loader jaisa audio:
    4 seconds, 16kHz, mono, mean/std normalize, clip -3 to 3.
    Output: (64000,)
    """
    sample_rate = 16000
    duration = 4.0
    target_len = int(sample_rate * duration)

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_wav = tmp.name

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-vn",
                "-ac", "1",
                "-ar", str(sample_rate),
                "-t", str(duration),
                temp_wav
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            if result.returncode != 0 or not os.path.exists(temp_wav):
                raise RuntimeError("ffmpeg audio extraction failed")

            y, sr = librosa.load(
                temp_wav,
                sr=sample_rate,
                mono=True,
                duration=duration
            )

        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        if y is None or len(y) == 0:
            raise RuntimeError("Empty audio")

        if np.isnan(y).any() or np.isinf(y).any():
            raise RuntimeError("NaN/Inf audio")

        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mean = np.mean(y)
        std = np.std(y)

        if std < 1e-6:
            std = 1.0

        y = (y - mean) / std
        y = np.clip(y, -3.0, 3.0)

        return y.astype(np.float32)

    except Exception as e:
        print(f"⚠️ Audio extraction failed: {e}")
        return np.zeros((target_len,), dtype=np.float32)


def process_video_for_ai(video_path, num_frames=16):
    """
    Training-compatible preprocessing.
    Returns:
        frames_np: (16, 224, 224, 3)
        flow_np:   (2, 224, 224)
        fft_np:    (1, 224, 224)
        audio_np:  (64000,)
        audio_path: None
    """
    print(f"🔪 Training-compatible extraction from: {video_path}")

    frames_np = extract_frames(video_path, num_frames=num_frames)
    flow_np = extract_optical_flow(frames_np)
    fft_np = extract_fft(frames_np, fft_num_frames=8)
    audio_np = extract_audio(video_path)

    print("✅ Extraction Complete!")

    return frames_np, flow_np, fft_np, audio_np, None