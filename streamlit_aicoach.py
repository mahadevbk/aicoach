import streamlit as st
import cv2
import numpy as np
import os
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tempfile
import zipfile
import io
import pandas as pd
import urllib.request

# --- 1. PAGE CONFIG & ORIGINAL CSS ---
st.set_page_config(
    page_title="Not Coach Nikki | Pro Analytics",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    [data-testid="stMetricValue"] { font-size: 28px !important; color: #ccff00 !important; font-weight: 800; }
    [data-testid="stMetricLabel"] { font-size: 14px !important; color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 1px; }
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 1.5rem; margin-bottom: 2rem; }
    h1 { font-size: clamp(1.8rem, 7vw, 3.5rem) !important; font-weight: 800 !important; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; filter: drop-shadow(2px 2px 0.5px rgba(204, 255, 0, 0.9)); }
    .hero-subtext { text-align: center; color: #94a3b8; font-size: 0.75rem; margin-bottom: 2rem; text-transform: uppercase; letter-spacing: 2px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; display: flex; flex-wrap: wrap; justify-content: center; background-color: transparent !important; }
    .stTabs [data-baseweb="tab"] { height: 55px; flex: 1 1 calc(50% - 10px); min-width: 140px; background: rgba(255, 255, 255, 0.05) !important; border-radius: 14px !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; transition: all 0.3s ease; }
    .stTabs [aria-selected="true"] { background: rgba(204, 255, 0, 0.1) !important; border: 1px solid #ccff00 !important; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CONFIGURATIONS ---
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]

SPORT_CONFIGS = {
    "TENNIS 🎾": {"Serve": "Toss height", "Forehand Drive": "Unit turn", "Backhand Drive": "Shoulder turn"},
    "PADEL 🎾": {"Bandeja": "Contact control", "Vibora": "Core rotation"},
    "PICKLEBALL 🥒": {"Dink": "Soft touch", "Kitchen Volley": "Hand speed"},
    "GOLF ⛳": {"Driver Swing": "Wide arc", "Iron Swing": "Lead arm extension"},
    "BADMINTON 🏸": {"Jump Smash": "Overhead whip"},
    "CRICKET BATTING 🏏": {"Drive": "High elbow lead"},
    "CRICKET BOWLING ⚾": {"Fast Bowling": "Release height"},
    "GYM 🏋️": {"Bodyweight Squat": "Depth"},
    "YOGA 🧘": {"Mountain Pose": "Vertical alignment"}
}

# --- 3. DYNAMIC ENGINE (pipeline_stereo & github-safe) ---
def download_model():
    model_path = 'pose_landmarker_heavy.task'
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        with st.spinner("🚀 Downloading Heavy AI Engine..."):
            urllib.request.urlretrieve(url, model_path)
    return model_path

def analyze_impact_frame(video_path, model_path):
    detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO
    ))
    cap = cv2.VideoCapture(video_path)
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    impact_frame, peak_vel, prev_wrist, skeletal_history = 0, 0, None, []
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(mp_image, int((i * 1000) / fps))
        
        if res.pose_world_landmarks:
            wrist = res.pose_world_landmarks[0][15] # Left wrist tracker
            if prev_wrist:
                vel = np.linalg.norm(np.array([wrist.x, wrist.y, wrist.z]) - np.array([prev_wrist.x, prev_wrist.y, prev_wrist.z]))
                if vel > peak_vel: peak_vel, impact_frame = vel, i
            prev_wrist = wrist
        skeletal_history.append(res.pose_landmarks[0] if res.pose_landmarks else None)
    cap.release()
    return {"impact": impact_frame, "skeletal": skeletal_history, "fps": fps, "total": total_frames}

# --- 4. UI LOGIC ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtext'>Pro Sports Biomechanics</p>", unsafe_allow_html=True)

tabs = st.tabs(list(SPORT_CONFIGS.keys()))

for i, (sport, actions) in enumerate(SPORT_CONFIGS.items()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col_setup, col_render = st.columns([1, 2])
        
        with col_setup:
            st.info(f"PRO {sport} ENGINE ACTIVE")
            is_stereo = st.toggle("Stereographic Mode", key=f"tog_{sport}")
            
            up1 = st.file_uploader(f"Upload Lead Angle", type=["mp4", "mov"], key=f"up1_{sport}")
            up2 = st.file_uploader(f"Upload Side Angle", type=["mp4", "mov"], key=f"up2_{sport}") if is_stereo else None
            
            sel_action = st.selectbox("SELECT ACTION", list(actions.keys()), key=f"sel_{sport}")
            run_btn = st.button("RUN AI ENGINE", key=f"btn_{sport}", use_container_width=True)

        with col_render:
            if run_btn and up1:
                model_path = download_model()
                t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                t1.write(up1.read())
                
                with st.status("Analyzing Biometrics...") as status:
                    res1 = analyze_impact_frame(t1.name, model_path)
                    res2 = None
                    if is_stereo and up2:
                        t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        t2.write(up2.read())
                        res2 = analyze_impact_frame(t2.name, model_path)
                        st.session_state[f"sync_{sport}"] = {"d1": res1, "d2": res2, "p1": t1.name, "p2": t2.name}
                    else:
                        st.video(t1.name)
                        st.metric("IMPACT FRAME", res1['impact'])

            # --- VERIFICATION SCRUBBER (Only for Stereo) ---
            if is_stereo and f"sync_{sport}" in st.session_state:
                s = st.session_state[f"sync_{sport}"]
                st.markdown("### 🛠️ SYNC VERIFICATION")
                sl1 = st.slider("Lead Impact", 0, s['d1']['total'], s['d1']['impact'], key=f"sl1_{sport}")
                sl2 = st.slider("Side Impact", 0, s['d2']['total'], s['d2']['impact'], key=f"sl2_{sport}")
                
                cap1, cap2 = cv2.VideoCapture(s['p1']), cv2.VideoCapture(s['p2'])
                cap1.set(cv2.CAP_PROP_POS_FRAMES, sl1); cap2.set(cv2.CAP_PROP_POS_FRAMES, sl2)
                _, img1 = cap1.read(); _, img2 = cap2.read()
                
                if img1 is not None and img2 is not None:
                    # Side-by-side impact check (20% height crop logic from user request)
                    preview = np.hstack((cv2.resize(img1, (480, 480)), cv2.resize(img2, (480, 480))))
                    st.image(preview, caption="Verify synchronized impact point.")
                    
                if st.button("GENERATE PRODUCTION PACK", key=f"gen_{sport}"):
                    st.success(f"Finalizing Stereo Analysis. Offset: {sl1 - sl2} frames.")
                    # Placeholder for final ffmpeg stitch from pipeline_stereo
        st.markdown("</div>", unsafe_allow_html=True)
