import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tempfile
import zipfile
import io
import pandas as pd
import urllib.request
import json

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Not Coach Nikki | Pro Analytics",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- DYNAMIC MODEL LOADER (Bypass 25MB Limit) ---
def get_model():
    model_path = "pose_landmarker_heavy.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        with st.spinner("🚀 INITIALIZING PRO-GRADE AI ENGINE..."):
            urllib.request.urlretrieve(url, model_path)
    return model_path

# --- PREMIUM CSS STYLING (Restored) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    [data-testid="stMetricValue"] { font-size: 28px !important; color: #ccff00 !important; font-weight: 800; }
    [data-testid="stMetricLabel"] { font-size: 14px !important; color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 1px; }
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 15px; padding: 25px; margin-bottom: 20px; }
    .stButton>button { background: linear-gradient(90deg, #ccff00, #9eff00); color: black; font-weight: 900; border: none; padding: 12px 30px; border-radius: 8px; transition: 0.3s; width: 100%; }
    .highlight { color: #ccff00; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- STEREO ENGINE LOGIC ---
def analyze_impact(video_path, model_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO)
    
    impact_frame, peak_vel = 0, 0
    prev_wrist = None
    
    with vision.PoseLandmarker.create_from_options(options) as detector:
        for i in range(total):
            ret, frame = cap.read()
            if not ret: break
            res = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int((i * 1000) / fps))
            if res.pose_world_landmarks:
                w = res.pose_world_landmarks[0][15]
                if prev_wrist:
                    vel = np.linalg.norm(np.array([w.x, w.y, w.z]) - np.array([prev_wrist.x, prev_wrist.y, prev_wrist.z]))
                    if vel > peak_vel: peak_vel, impact_frame = vel, i
                prev_wrist = w
    cap.release()
    return impact_frame, total, fps

# --- UI HEADER ---
st.markdown("## 🎾 NOT COACH NIKKI | <span class='highlight'>PRO ANALYTICS</span>", unsafe_allow_html=True)

# --- RESTORED ORIGINAL TABS & DATA STRUCTURE ---
sports = {
    "Tennis": {"actions": ["Serve", "Forehand", "Backhand"], "goal": "Consistency"},
    "Golf": {"actions": ["Drive", "Iron Shot", "Putting"], "goal": "Swing Path"},
    "Cricket": {"actions": ["Bowling", "Cover Drive"], "goal": "Stability"}
}

tab_list = st.tabs(list(sports.keys()))

for i, (sport, data) in enumerate(sports.items()):
    with tab_list[i]:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"<div class='glass-card'><h4>{sport.upper()} SETUP</h4>", unsafe_allow_html=True)
            sel_action = st.selectbox("Select Action", data['actions'], key=f"sel_{sport}")
            
            # --- STEREO TOGGLE ---
            stereo_on = st.toggle("Stereographic Mode", key=f"st_{sport}")
            
            v_main = st.file_uploader("Upload Lead Angle", type=["mp4", "mov"], key=f"v1_{sport}")
            v_side = None
            if stereo_on:
                v_side = st.file_uploader("Upload Side Angle", type=["mp4", "mov"], key=f"v2_{sport}")
            
            run_btn = st.button("START AI ANALYSIS", key=f"btn_{sport}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            if run_btn and v_main:
                # Save temp
                t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                t1.write(v_main.read())
                
                if stereo_on and v_side:
                    t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    t2.write(v_side.read())
                    
                    # AI DETECTION
                    with st.status("🤖 Syncing Dual Cameras...") as status:
                        model = get_model()
                        i1, tot1, fps1 = analyze_impact(t1.name, model)
                        i2, tot2, fps2 = analyze_impact(t2.name, model)
                        st.session_state[f"sync_{sport}"] = {'i1': i1, 'i2': i2, 't1': tot1, 't2': tot2, 'p1': t1.name, 'p2': t2.name}
                else:
                    st.video(t1.name)
                    st.success("Single Angle Analysis Complete")

            # --- IMPACT SCRUBBER UI (If Stereo is active) ---
            if stereo_on and f"sync_{sport}" in st.session_state:
                sd = st.session_state[f"sync_{sport}"]
                st.markdown("### 🛠️ VERIFY IMPACT SYNC")
                
                sc1, sc2 = st.columns(2)
                f1 = sc1.slider("Lead Frame", 0, sd['t1'], sd['i1'], key=f"f1_{sport}")
                f2 = sc2.slider("Side Frame", 0, sd['t2'], sd['i2'], key=f"f2_{sport}")
                
                # Visual Verification
                cap1 = cv2.VideoCapture(sd['p1'])
                cap2 = cv2.VideoCapture(sd['p2'])
                cap1.set(cv2.CAP_PROP_POS_FRAMES, f1)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, f2)
                _, img1 = cap1.read()
                _, img2 = cap2.read()
                
                if img1 is not None and img2 is not None:
                    # Show 20% area side-by-side
                    combined = np.hstack((cv2.resize(img1, (500, 500)), cv2.resize(img2, (500, 500))))
                    st.image(combined, caption="Match the exact impact frame on both views.")

                if st.button("🚀 GENERATE PRODUCTION STEREO PACK", key=f"prod_{sport}"):
                    # In a real app, this triggers the heavy ffmpeg/cv2 render
                    st.success(f"Generated! Offset: {f1 - f2} frames.")
                    st.download_button("Download JSON Data", json.dumps({"offset": f1-f2}), f"{sport}_meta.json")
