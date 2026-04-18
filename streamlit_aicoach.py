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
import urllib.request
import subprocess
import time

# --- 1. FULL PREMIUM UI & TABS ---
st.set_page_config(page_title="Not Coach Nikki | Pro Analytics", page_icon="🎾", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    
    /* NEON TOGGLE & SLIDERS */
    .stCheckbox label p { color: #38bdf8 !important; font-weight: 700; text-transform: uppercase; }
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; font-size: 1.1rem !important; text-transform: uppercase; }
    
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 1.5rem; margin-bottom: 2rem; }
    h1 { font-size: clamp(2rem, 7vw, 3.5rem) !important; font-weight: 800 !important; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0px !important; }
    .hero-sub { text-align: center; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 4px; margin-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE PRODUCTION ENGINE ---
FULL_SKELETON = [(0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10), (11,12), (11,13), (13,15), (12,14), (14,16), (11,23), (12,24), (23,24), (23,25), (25,27), (24,26), (26,28)]

def run_ffmpeg_mobile(input_path):
    """Encodes to H.264 YUV420P for WhatsApp/Mobile compatibility."""
    out_path = input_path.replace(".mp4", "_mobile.mp4")
    subprocess.run(f"ffmpeg -y -i {input_path} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 27 {out_path}", shell=True, capture_output=True)
    return out_path

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

# --- 3. UI LAYOUT & ALL TABS ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>Pro Biometrics AI Engine</p>", unsafe_allow_html=True)

SPORT_MAP = {
    "TENNIS 🎾": ["Serve", "Forehand", "Backhand"],
    "PADEL 🎾": ["Bandeja", "Vibora", "Smash"],
    "PICKLEBALL 🥒": ["Dink", "Kitchen Volley"],
    "GOLF ⛳": ["Driver Swing", "Iron Swing"],
    "CRICKET 🏏": ["Batting", "Bowling"],
    "GYM 🏋️": ["Squat", "Deadlift"],
    "YOGA 🧘": ["Balance", "Warrior"]
}

tabs = st.tabs(list(SPORT_MAP.keys()))

for i, (sport, actions) in enumerate(SPORT_MAP.items()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        
        # RESTORED TOGGLE
        is_stereo = st.toggle("Stereographic Mode (2 Cameras)", value=True, key=f"mode_{sport}")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"PRO {sport} ACTIVE")
            u1 = st.file_uploader("Lead Angle / Main Video", type=["mp4","mov"], key=f"u1_{sport}")
            u2 = None
            if is_stereo:
                u2 = st.file_uploader("Side Angle", type=["mp4","mov"], key=f"u2_{sport}")
            
            sel_act = st.selectbox("Action Type", actions, key=f"act_{sport}")
            run_btn = st.button("RUN AI ENGINE", key=f"run_{sport}", use_container_width=True)

        with col2:
            res_key = f"data_{sport}"
            if run_btn and u1 and (not is_stereo or u2):
                model_p = 'pose_landmarker_heavy.task'
                if not os.path.exists(model_p):
                    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", model_p)
                
                t1 = tempfile.NamedTemporaryFile(delete=False); t1.write(u1.getbuffer())
                t2_name = None
                if is_stereo:
                    t2 = tempfile.NamedTemporaryFile(delete=False); t2.write(u2.getbuffer())
                    t2_name = t2.name
                
                with st.status("Analyzing Biometrics...") as status:
                    d1 = analyze_pose(t1.name, model_p)
                    d2 = analyze_pose(t2_name, model_p) if is_stereo else None
                    st.session_state[res_key] = {"d1": d1, "d2": d2, "p1": t1.name, "p2": t2_name}

            if res_key in st.session_state and st.session_state[res_key]:
                s = st.session_state[res_key]
                st.markdown("### 🛠️ SYNC & PREVIEW")
                
                sl1 = st.slider("Lead Frame", 0, s['d1']['total']-1, key=f"sl1_{sport}")
                sl2 = 0
                if is_stereo:
                    sl2 = st.slider("Side Frame", 0, s['d2']['total']-1, key=f"sl2_{sport}")

                # --- RENDER LOGIC ---
                if st.button("🎬 GENERATE PRODUCTION PACK", key=f"gen_{sport}", use_container_width=True):
                    with st.spinner("Processing for Mobile..."):
                        raw_path = os.path.join(tempfile.gettempdir(), f"render_{int(time.time())}.mp4")
                        cap1 = cv2.VideoCapture(s['p1'])
                        target_h = 720
                        w1 = int(cap1.get(3) * (target_h / cap1.get(4)))
                        
                        if is_stereo:
                            cap2 = cv2.VideoCapture(s['p2'])
                            w2 = int(cap2.get(3) * (target_h / cap2.get(4)))
                            out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), s['d1']['fps'], (w1+w2, target_h))
                            # ... (Stereo stitching logic)
                            cap2.release()
                        else:
                            out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), s['d1']['fps'], (w1, target_h))
                            for i, lms in enumerate(s['d1']['history']):
                                ret, frame = cap1.read()
                                if not ret: break
                                frame = cv2.resize(frame, (w1, target_h))
                                if lms:
                                    for start, end in FULL_SKELETON:
                                        cv2.line(frame, (int(lms[start].x*w1), int(lms[start].y*target_h)), (int(lms[end].x*w1), int(lms[end].y*target_h)), (127, 255, 0), 3)
                                out.write(frame)
                        
                        cap1.release(); out.release()
                        
                        # CONVERT & DISPLAY
                        final_v = run_ffmpeg_mobile(raw_path)
                        st.video(final_v)
                        with open(final_v, "rb") as f:
                            st.download_button("📥 DOWNLOAD FOR WHATSAPP", f.read(), f"{sport}_analysis.mp4", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
