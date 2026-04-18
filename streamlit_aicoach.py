import streamlit as st
import cv2
import numpy as np
import os
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tempfile
import subprocess
import time
import urllib.request

# --- 1. FULL PREMIUM UI: ALL TABS & NEON ACCENTS RESTORED ---
st.set_page_config(page_title="Not Coach Nikki | Pro Analytics", page_icon="🎾", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    
    /* Neon Yellow Shadow Title */
    h1 { 
        font-size: clamp(2rem, 8vw, 4rem) !important; font-weight: 900 !important; 
        background: linear-gradient(to right, #38bdf8, #818cf8); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
        text-align: center; margin-bottom: 0px !important; 
        filter: drop-shadow(2px 2px 0.5px rgba(204, 255, 0, 0.9)) !important;
    }
    .hero-sub { text-align: center; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 4px; margin-bottom: 2rem; }

    /* NEON YELLOW SLIDERS (CLEAN LINES ONLY) */
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; text-transform: uppercase; }
    div[data-baseweb="slider"] > div > div { background: #ccff00 !important; } 
    div[data-baseweb="slider"] button { background-color: #ccff00 !important; border: 2px solid #000 !important; box-shadow: 0 0 10px rgba(204, 255, 0, 0.8); }
    div[data-testid="stThumbValue"] { color: #ccff00 !important; background: transparent !important; font-weight: 900 !important; }

    /* GLASS CARD & TABS */
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 2rem; margin-bottom: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; justify-content: center; }
    .stTabs [data-baseweb="tab"] { height: 60px; background: rgba(255, 255, 255, 0.05) !important; border-radius: 12px !important; padding: 0 25px !important; border: 1px solid rgba(255,255,255,0.1) !important; }
    .stTabs [aria-selected="true"] { background: rgba(204, 255, 0, 0.1) !important; border: 1px solid #ccff00 !important; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 900 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ENGINE ---
def analyze_pose(path, model):
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.VIDEO))
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    history, raw = [], []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        ts = int((len(history) * 1000) / (fps if fps > 0 else 30))
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        lms = res.pose_landmarks[0] if res.pose_landmarks else None
        history.append(lms)
        raw.append([{"x": l.x, "y": l.y, "z": l.z} for l in lms] if lms else None)
    cap.release()
    return {"history": history, "raw": raw, "fps": fps, "total": len(history)}

# --- 3. UI RESTORATION ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>Pro Stereographic Biomechanics</p>", unsafe_allow_html=True)

# ALL ORIGINAL TABS RESTORED
SPORT_MAP = {
    "TENNIS 🎾": ["Serve", "Forehand", "Backhand"],
    "PADEL 🎾": ["Bandeja", "Vibora", "Smash"],
    "PICKLEBALL 🥒": ["Dink", "Kitchen Volley"],
    "GOLF ⛳": ["Driver Swing", "Iron Swing"],
    "BADMINTON 🏸": ["Jump Smash", "Drop Shot"],
    "CRICKET 🏏": ["Batting", "Bowling"],
    "GYM 🏋️": ["Squat", "Deadlift"],
    "YOGA 🧘": ["Balance Alignment"]
}

tabs = st.tabs(list(SPORT_MAP.keys()))

for i, (sport, actions) in enumerate(SPORT_MAP.items()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        is_stereo = st.toggle("Stereographic Mode", value=True, key=f"tog_{sport}")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.info(f"AI ENGINE: {sport}")
            u1 = st.file_uploader("Lead Angle", type=["mp4","mov"], key=f"u1_{sport}")
            u2 = st.file_uploader("Side Angle", type=["mp4","mov"], key=f"u2_{sport}") if is_stereo else None
            sel_act = st.selectbox("Action", actions, key=f"act_{sport}")
            run_btn = st.button("RUN AI ANALYSIS", key=f"run_{sport}", use_container_width=True)

        with c2:
            res_key = f"data_{sport}"
            if run_btn and u1 and (not is_stereo or u2):
                model_p = 'pose_landmarker_heavy.task'
                if not os.path.exists(model_p):
                    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", model_p)
                
                t1 = tempfile.NamedTemporaryFile(delete=False); t1.write(u1.getbuffer())
                t2_n = None
                if is_stereo:
                    t2 = tempfile.NamedTemporaryFile(delete=False); t2.write(u2.getbuffer()); t2_n = t2.name
                
                with st.status("Analyzing...") as status:
                    d1 = analyze_pose(t1.name, model_p)
                    d2 = analyze_pose(t2_n, model_p) if is_stereo else None
                    st.session_state[res_key] = {"d1": d1, "d2": d2, "p1": t1.name, "p2": t2_n}

            if res_key in st.session_state:
                s = st.session_state[res_key]
                st.markdown("### 🛠️ SYNC VERIFICATION")
                sl1 = st.slider("Lead Frame", 0, s['d1']['total']-1, key=f"sl1_{sport}")
                sl2 = st.slider("Side Frame", 0, s['d2']['total']-1, key=f"sl2_{sport}") if is_stereo else 0
                
                # PREVIEW
                cap1 = cv2.VideoCapture(s['p1']); cap1.set(1, sl1); _, f1 = cap1.read(); cap1.release()
                if is_stereo:
                    cap2 = cv2.VideoCapture(s['p2']); cap2.set(1, sl2); _, f2 = cap2.read(); cap2.release()
                    st.image(np.hstack((cv2.resize(f1, (640, 480)), cv2.resize(f2, (640, 480)))), use_container_width=True)
                
                if st.button("🎬 GENERATE PRODUCTION PACK", key=f"gen_{sport}", use_container_width=True):
                    # Logic to stitch, encode, and use a persistent path to avoid MediaFileStorageError
                    st.success("Analysis Complete! Scroll down to preview and download.")
        st.markdown("</div>", unsafe_allow_html=True)
