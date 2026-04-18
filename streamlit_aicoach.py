import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import subprocess
import time
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. PREMIUM UI (8 TABS, MOBILE GRID, NEON YELLOW PRECISION) ---
st.set_page_config(page_title="Not Coach Nikki | Pro Analytics", page_icon="🎾", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    h1 { font-size: clamp(2rem, 8vw, 4rem) !important; font-weight: 900 !important; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; filter: drop-shadow(2px 2px 0.5px rgba(204, 255, 0, 0.9)) !important; }
    
    /* RESTORED: 2-Column Mobile Tabs */
    .stTabs [data-baseweb="tab-list"] { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .stTabs [data-baseweb="tab"] { width: 100%; height: 55px; background: rgba(255, 255, 255, 0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important; }
    .stTabs [aria-selected="true"] { background: rgba(204, 255, 0, 0.1) !important; border: 1px solid #ccff00 !important; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 900 !important; }

    /* FIXED: Neon Yellow Slider Text & Tracks */
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; text-transform: uppercase; }
    div[data-baseweb="slider"] > div > div { background: #ccff00 !important; } 
    div[data-baseweb="slider"] button { background-color: #ccff00 !important; border: 2px solid #000 !important; box-shadow: 0 0 10px rgba(204, 255, 0, 0.8); }
    div[data-testid="stThumbValue"] { color: #ccff00 !important; background: transparent !important; font-weight: 900 !important; }

    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 1.5rem; margin-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ENGINE LOGIC ---
def analyze_vid(p, m):
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=m), running_mode=vision.RunningMode.VIDEO))
    cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS); hist = []; cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        ts = int((cnt * 1000) / fps)
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        hist.append(res.pose_landmarks[0] if res.pose_landmarks else None)
        cnt += 1
    cap.release(); return {"history": hist, "fps": fps, "total": cnt}

# --- 3. UI TAB RESTORATION ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)

SPORT_MAP = {
    "TENNIS 🎾": ["Serve"], "PADEL 🎾": ["Bandeja"], "PICKLEBALL 🥒": ["Dink"],
    "GOLF ⛳": ["Swing"], "BADMINTON 🏸": ["Smash"], "CRICKET 🏏": ["Batting"],
    "GYM 🏋️": ["Squat"], "YOGA 🧘": ["Balance"]
}

tabs = st.tabs(list(SPORT_MAP.keys()))

for i, (sport, actions) in enumerate(SPORT_MAP.items()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        res_key = f"final_res_{sport}"
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write(f"**ENGINE: {sport}**")
            u1 = st.file_uploader("Lead Angle", type=["mp4","mov"], key=f"u1_{sport}")
            u2 = st.file_uploader("Side Angle", type=["mp4","mov"], key=f"u2_{sport}")
            run_btn = st.button("RUN PRO ANALYSIS", key=f"run_{sport}", use_container_width=True)

        if run_btn and u1:
            model_p = 'pose_landmarker_heavy.task'
            if not os.path.exists(model_p):
                urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", model_p)
            
            t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); t1.write(u1.getbuffer())
            t2_path = None
            if u2:
                t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); t2.write(u2.getbuffer()); t2_path = t2.name
            
            with st.spinner("Decoding Biometrics..."):
                d1 = analyze_vid(t1.name, model_p)
                d2 = analyze_vid(t2_path, model_p) if t2_path else None
                st.session_state[res_key] = {"d1": d1, "d2": d2, "p1": t1.name, "p2": t2_path}

        # PERSISTENT VIEW LOGIC: This ensures the sliders stay even if you change tabs
        with c2:
            if res_key in st.session_state:
                r = st.session_state[res_key]
                st.markdown("### 🛠️ SYNC VERIFICATION")
                sl1 = st.slider("Lead Impact Frame", 0, r['d1']['total']-1, key=f"sl1_{sport}")
                sl2 = st.slider("Side Impact Frame", 0, (r['d2']['total']-1 if r['d2'] else 0), key=f"sl2_{sport}")
                
                # RGB COLOR FIX PREVIEW
                cap1 = cv2.VideoCapture(r['p1']); cap1.set(1, sl1); _, f1 = cap1.read(); cap1.release()
                if r['p2']:
                    cap2 = cv2.VideoCapture(r['p2']); cap2.set(1, sl2); _, f2 = cap2.read(); cap2.release()
                    img = np.hstack((cv2.resize(f1, (640, 480)), cv2.resize(f2, (640, 480))))
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.image(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB), width=640)

                if st.button("🎬 GENERATE WHATSAPP-READY PACK", key=f"gen_{sport}", use_container_width=True):
                    st.info("Encoding... (Ensure ffmpeg is in packages.txt)")
        st.markdown("</div>", unsafe_allow_html=True)
