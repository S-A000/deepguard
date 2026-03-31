import cv2
import numpy as np
import os
import subprocess

def process_video_for_ai(video_path, num_frames=16):
    """
    Yeh function MP4 video ko AI ke samajhne qabil frames aur audio mein torta hai.
    """
    print(f"🔪 Extracting frames and audio from: {video_path}")
    
    # ==========================================
    # 1. EXTRACT FRAMES (Visual Branch ke liye)
    # ==========================================
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Har video mein se barabar faasle par 16 frames nikalne ka math
    step = max(1, total_frames // num_frames)
    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            # OpenCV BGR mein read karta hai, humein RGB chahiye
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # TimeSformer ka standard size 224x224 hai
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
            
    cap.release()
    
    # Agar video bohut choti thi aur 16 frames nahi nikle, toh aakhri frame copy kar do
    while len(frames) < num_frames:
        frames.append(frames[-1] if len(frames) > 0 else np.zeros((224, 224, 3)))

    frames_np = np.array(frames) # Shape: (16, 224, 224, 3)

    # ==========================================
    # 2. EXTRACT AUDIO (Audio Branch ke liye)
    # ==========================================
    audio_path = "temp_audio.wav"
    # FFmpeg use karke video se awaz alag kar rahe hain
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y -loglevel error"
    subprocess.call(command, shell=True)
    
    print("✅ Extraction Complete!")
    return frames_np, audio_path