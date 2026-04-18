import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tempfile
import urllib.request
import json

# --- CONFIG & STYLES ---
st.set_page_config(page_title="Coach Nikki | Stereo Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #020617; color: #f8fafc; }
    .glass-card { background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 20px; border: 1px solid rgba(255,255,255,0.1); }
    .highlight { color: #ccff00; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- ENGINE ---
def get_model():
    model_path = "pose_landmarker_heavy.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        with st.spinner("Downloading Pro AI Model..."):
            urllib.request.urlretrieve(url, model_path)
    return model_path

class StereoEngine:
    def __init__(self, model_path):
        self.model_path = model_path

    def analyze_impact(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO)
        
        impact_frame, peak_vel = 0, 0
        prev_wrist = None
        
        with vision.PoseLandmarker.create_from_options(options) as detector:
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = detector.detect_for_video(mp_image, int((i * 1000) / fps))
                
                if res.pose_world_landmarks:
                    wrist = res.pose_world_landmarks[0][15] # Left wrist as proxy
                    if prev_wrist:
                        vel = np.linalg.norm(np.array([wrist.x, wrist.y, wrist.z]) - np.array([prev_wrist.x, prev_wrist.y, prev_wrist.z]))
                        if vel > peak_vel:
                            peak_vel, impact_frame = vel, i
                    prev_wrist = wrist
        cap.release()
        return impact_frame, total_frames, fps

# --- UI LOGIC ---
st.title("🎾 AI COACH | <span class='highlight'>STEREO SYNC</span>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
v1 = col1.file_uploader("Lead Angle (Video A)", type=["mp4", "mov"])
v2 = col2.file_uploader("Side Angle (Video B)", type=["mp4", "mov"])

if v1 and v2:
    # Save to temp
    t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    t1.write(v1.read()); t2.write(v2.read())

    # 1. RUN AI INITIAL SYNC
    if 'offset' not in st.session_state:
        if st.button("🚀 RUN AI IMPACT DETECTION"):
            engine = StereoEngine(get_model())
            impact1, total1, fps1 = engine.analyze_impact(t1.name)
            impact2, total2, fps2 = engine.analyze_impact(t2.name)
            st.session_state.sync_data = {'i1': impact1, 'i2': impact2, 't1': total1, 't2': total2, 'fps': fps1}
            st.session_state.offset = impact1 - impact2
            st.rerun()

    # 2. FINE TUNING UI
    if 'offset' in st.session_state:
        st.markdown("---")
        st.subheader("🛠️ FINE-TUNE IMPACT SYNC")
        s = st.session_state.sync_data
        
        # UI for manual adjustment
        c1, c2, c3 = st.columns([1, 1, 2])
        f1 = c1.slider("Video A Impact Frame", 0, s['t1'], s['i1'])
        f2 = c2.slider("Video B Impact Frame", 0, s['t2'], s['i2'])
        
        # Display 20% Preview Window (Impact +/- 10%)
        # Here we show the frames side-by-side as a "Sanity Check"
        cap1 = cv2.VideoCapture(t1.name)
        cap2 = cv2.VideoCapture(t2.name)
        
        cap1.set(cv2.CAP_PROP_POS_FRAMES, f1)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, f2)
        
        _, img1 = cap1.read()
        _, img2 = cap2.read()
        
        if img1 is not None and img2 is not None:
            preview = np.hstack((cv2.resize(img1, (640, 360)), cv2.resize(img2, (640, 360))))
            st.image(preview, caption="Sync Check: Are the impacts aligned? Adjust sliders above.", use_container_width=True)

        if st.button("✅ GENERATE FINAL PRODUCTION VIDEO"):
            # This is where you'd call your pipeline_stereo.py render logic
            # using (f1 - f2) as the final offset.
            st.success(f"Rendering Video with Offset: {f1 - f2} frames...")
            # [Rendering logic here...]
            
            # Dummy JSON for example
            analytics = {"impact_sync": {"a": f1, "b": f2, "offset": f1-f2}}
            st.json(analytics)
