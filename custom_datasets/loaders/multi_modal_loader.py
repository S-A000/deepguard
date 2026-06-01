import os
import cv2
import torch
import numpy as np
import librosa
import random
import tempfile
import subprocess
from torch.utils.data import Dataset


class DeepGuardDataset(Dataset):
    # 🚀 FIX: Added 'mode' parameter for ultra-fast conditional loading
    def __init__(self, real_dirs, fake_dirs, num_frames=16, max_samples=None, mode="multi"):
        self.num_frames = num_frames
        self.mode = mode
        self.video_paths = []
        self.labels = []

        real_videos = []
        fake_videos = []

        # ✅ Updated supported extensions
        supported_exts = (
            '.mp4', '.avi', '.mov', '.mkv', '.webm',
            '.wav', '.flac', '.mp3', '.ogg', '.m4a'
        )

        for d in real_dirs:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for file in files:
                        if file.lower().endswith(supported_exts):
                            real_videos.append(os.path.join(root, file))

        for d in fake_dirs:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for file in files:
                        if file.lower().endswith(supported_exts):
                            fake_videos.append(os.path.join(root, file))

        if max_samples is not None:
            half_sample = max_samples // 2

            if len(real_videos) > half_sample:
                real_videos = random.sample(real_videos, half_sample)

            if len(fake_videos) > half_sample:
                fake_videos = random.sample(fake_videos, half_sample)

        for vid in real_videos:
            self.video_paths.append(vid)
            self.labels.append(0.0)  # REAL

        for vid in fake_videos:
            self.video_paths.append(vid)
            self.labels.append(1.0)  # FAKE

        print(f"📦 Total Enterprise Dataset Loaded: {len(self.video_paths)} Files (Mode: {self.mode})")

    def __len__(self):
        return len(self.video_paths)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < self.num_frames:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        while len(frames) < self.num_frames:
            frames.append(
                frames[-1] if len(frames) > 0
                else np.zeros((224, 224, 3), dtype=np.uint8)
            )

        return np.array(frames)

    def extract_optical_flow(self, frames_np):
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

        return torch.tensor(flow).permute(2, 0, 1).float()

    def extract_fft(self, frames_np):
        gray = cv2.cvtColor(frames_np[self.num_frames // 2], cv2.COLOR_RGB2GRAY)

        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)

        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
        magnitude_spectrum = magnitude_spectrum / (np.max(magnitude_spectrum) + 1e-8)

        return torch.tensor(magnitude_spectrum).unsqueeze(0).float()

    def extract_audio(self, file_path):
        """
        Robust audio extractor for both audio files and video files.

        Output:
            torch.Tensor shape: (64000,)
            4 seconds audio at 16kHz

        If audio is corrupt, missing, silent, or unreadable:
            returns NaN tensor so training script can skip it.
        """

        sample_rate = 16000
        duration = 4.0
        target_len = int(sample_rate * duration)

        audio_exts = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')
        video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

        ext = os.path.splitext(file_path)[1].lower()

        try:
            # ==========================================
            # Case 1: Direct audio file
            # ==========================================
            if ext in audio_exts:
                y, sr = librosa.load(
                    file_path,
                    sr=sample_rate,
                    mono=True,
                    duration=duration
                )

            # ==========================================
            # Case 2: Video file, extract audio by ffmpeg
            # ==========================================
            elif ext in video_exts:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_wav = tmp.name

                try:
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i", file_path,
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

            else:
                raise RuntimeError(f"Unsupported file extension: {ext}")

            # ==========================================
            # Safety checks
            # ==========================================
            if y is None or len(y) == 0:
                raise RuntimeError("Empty audio")

            if np.isnan(y).any() or np.isinf(y).any():
                raise RuntimeError("NaN/Inf audio")

            if np.abs(y).sum() < 1e-6:
                raise RuntimeError("Silent audio")

            # ==========================================
            # Fixed length: 4 seconds
            # ==========================================
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]

            # ==========================================
            # Normalize audio
            # ==========================================
            mean = np.mean(y)
            std = np.std(y)

            if std < 1e-6:
                std = 1.0

            y = (y - mean) / std
            y = np.clip(y, -3.0, 3.0)

            return torch.tensor(y, dtype=torch.float32)

        except Exception:
            # Training script / SafeDataset can skip this sample
            return torch.full((target_len,), float('nan'), dtype=torch.float32)

    def __getitem__(self, idx):
        file_path = self.video_paths[idx]
        label = self.labels[idx]

        # 🚀 FIX: THE ULTRA-FAST AUDIO-ONLY BYPASS
        if self.mode == "audio_only":
            audio_features = self.extract_audio(file_path)

            # Empty tensors taake memory aur bandwidth zaya na ho
            return (
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                audio_features,
                torch.tensor(label, dtype=torch.float32)
            )

        # 🎬 NORMAL VIDEO PIPELINE (For Multi-Modal Mode)
        else:
            audio_exts = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')

            if file_path.lower().endswith(audio_exts):
                # Agar multi mode mein audio file aajaye toh dummy frames dein
                video_rgb = torch.zeros((self.num_frames, 3, 224, 224)).float()
                flow_frames = torch.zeros((2, 224, 224)).float()
                forensics_frames = torch.zeros((1, 224, 224)).float()
                audio_features = self.extract_audio(file_path)

            else:
                frames_np = self.extract_frames(file_path)

                video_rgb = torch.tensor(frames_np).permute(0, 3, 1, 2).float() / 255.0
                flow_frames = self.extract_optical_flow(frames_np)
                forensics_frames = self.extract_fft(frames_np)
                audio_features = self.extract_audio(file_path)

            return (
                video_rgb,
                flow_frames,
                forensics_frames,
                audio_features,
                torch.tensor(label, dtype=torch.float32)
            )
