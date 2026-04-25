import os
import cv2
import torch
import numpy as np
import librosa
import random
from torch.utils.data import Dataset

class DeepGuardDataset(Dataset):
    def __init__(self, real_dirs, fake_dirs, num_frames=16, max_samples=None):
        self.num_frames = num_frames
        self.video_paths = []
        self.labels = []
        
        real_videos = []
        fake_videos = []

        # 1. Collect Real Videos & Audio
        for d in real_dirs:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for file in files:
                        # 🚀 CHANGE 1: Added Audio Formats (.wav, .flac)
                        if file.endswith(('.mp4', '.avi', '.wav', '.flac')):
                            real_videos.append(os.path.join(root, file))

        # 2. Collect Fake Videos & Audio
        for d in fake_dirs:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for file in files:
                        # 🚀 CHANGE 1: Added Audio Formats (.wav, .flac)
                        if file.endswith(('.mp4', '.avi', '.wav', '.flac')):
                            fake_videos.append(os.path.join(root, file))

        # 3. Balancing / Sampling Logic
        if max_samples is not None:
            half_sample = max_samples // 2
            if len(real_videos) > half_sample:
                real_videos = random.sample(real_videos, half_sample)
            if len(fake_videos) > half_sample:
                fake_videos = random.sample(fake_videos, half_sample)

        # 4. Final Merge with Float32 labels
        for vid in real_videos:
            self.video_paths.append(vid)
            self.labels.append(0.0) # REAL

        for vid in fake_videos:
            self.video_paths.append(vid)
            self.labels.append(1.0) # FAKE

        print(f"📦 Total Enterprise Dataset Loaded: {len(self.video_paths)} Files")

    def __len__(self):
        return len(self.video_paths)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if len(frames) > 0 else np.zeros((224, 224, 3), dtype=np.uint8))
        return np.array(frames)

    def extract_optical_flow(self, frames_np):
        # Convert first two frames to gray for simple flow
        gray1 = cv2.cvtColor(frames_np[0], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frames_np[1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return torch.tensor(flow).permute(2, 0, 1).float()

    def extract_fft(self, frames_np):
        gray = cv2.cvtColor(frames_np[self.num_frames // 2], cv2.COLOR_RGB2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
        magnitude_spectrum = magnitude_spectrum / (np.max(magnitude_spectrum) + 1e-8) 
        return torch.tensor(magnitude_spectrum).unsqueeze(0).float()

    def extract_audio(self, video_path):
        try:
            # 1. Load Audio
            y, sr = librosa.load(video_path, sr=16000, duration=1.0)
            
            # 2. 🚨 THE "SILENT KILLER" CHECK: Librosa kabhi NaN de deta hai corrupt video par
            if np.isnan(y).any() or np.isinf(y).any():
                print(f"\n⚠️ CRITICAL WARNING: Librosa found NaN/Inf in file: {video_path}")
                # Fauran NaN ko zero se replace karo aur noise daalo
                y = np.nan_to_num(y) + (1e-5 * np.random.randn(len(y)))

            # 3. Padding agar audio 1 second se kam ho
            if len(y) < 16000:
                y = np.pad(y, (0, 16000 - len(y)))

            # 4. Final Cleanup: Ek bar aur confirm karein ke output saaf hai
            tensor_audio = torch.tensor(y[:16000]).float()
            tensor_audio = torch.nan_to_num(tensor_audio) # Final PyTorch level check
            
            return tensor_audio
            
        except Exception as e:
            # Fallback agar file open hi na ho
            # print(f"⚠️ Librosa failed on {video_path}: {e}") # Isko comment kiya hai taake warnings na ayen
            noise = 1e-5 * np.random.randn(16000)
            return torch.tensor(noise).float()

    def __getitem__(self, idx):
        file_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # 🚀 CHANGE 2: AUDIO-ONLY BYPASS ROUTE
        if file_path.endswith(('.wav', '.flac')):
            # Dummy tensors for Visual components
            video_rgb = torch.zeros((self.num_frames, 3, 224, 224)).float()
            flow_frames = torch.zeros((2, 224, 224)).float()
            forensics_frames = torch.zeros((1, 224, 224)).float()
            
            # Asal Audio Extract
            audio_features = self.extract_audio(file_path)
            
            return (video_rgb, 
                    flow_frames, 
                    forensics_frames, 
                    audio_features, 
                    torch.tensor(label, dtype=torch.float32))

        # 🎬 NORMAL VIDEO PIPELINE
        else:
            frames_np = self.extract_frames(file_path)
            
            # RGB Video Normalization
            video_rgb = torch.tensor(frames_np).permute(0, 3, 1, 2).float() / 255.0
            
            # Experts Features
            flow_frames = self.extract_optical_flow(frames_np)
            forensics_frames = self.extract_fft(frames_np)
            audio_features = self.extract_audio(file_path)

            # 🚀 Return with explicit float32 label tensor
            return (video_rgb, 
                    flow_frames, 
                    forensics_frames, 
                    audio_features, 
                    torch.tensor(label, dtype=torch.float32))