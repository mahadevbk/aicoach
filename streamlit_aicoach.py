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

# --- 1. FULL PREMIUM UI RESTORATION (LOCKED) ---
st.set_page_config(page_title="Not Coach Nikki | Pro Analytics", page_icon="🎾", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; font-size: 1.2rem !important; text-transform: uppercase; }
    div[data-testid="stThumbValue"] { color: #000 !important; background-color: #ccff00 !important; font-weight: 900 !important; }
    div[data-baseweb="slider"] > div { background: #ccff00 !important; }
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 1.5rem; margin-bottom: 2rem; }
    h1 { font-size: clamp(2rem, 7vw, 3.5rem) !important; font-weight: 800 !important; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0px !important; }
    .hero-sub { text-align: center; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 4px; margin-bottom: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; justify-content: center; }
    .stTabs [data-baseweb="tab"] { height: 60px; background: rgba(255, 255, 255, 0.05) !important; border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.1) !important; padding: 0 25px !important; }
    .stTabs [aria-selected="true"] { background: rgba(204, 255, 0, 0.1) !important; border: 1px solid #ccff00 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE PRODUCTION ENGINE (WITH STORAGE FIX) ---
FULL_SKELETON = [(0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10), (11,12), (11,13), (13,15), (12,14), (14,16), (11,23), (12,24), (23,24), (23,25), (25,27), (24,26), (26,28)]

def safe_render_and_encode(p1, p2, h1, h2, f1, f2, fps):
    # Create permanent-style temporary names to avoid MediaFileStorageError
    raw_out = os.path.join(tempfile.gettempdir(), f"raw_{int(time.time())}.mp4")
    final_out = os.path.join(tempfile.gettempdir(), f"final_{int(time.time())}.mp4")
    
    cap1, cap2 = cv2.VideoCapture(p1), cv2.VideoCapture(p2)
    off, target_h = f1 - f2, 720
    w1 = int(cap1.get(3) * (target_h / cap1.get(4)))
    w2 = int(cap2.get(3) * (target_h / cap2.get(4)))
    
    out = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w1+w2, target_h))
    for i in range(len(h1)):
        ret1, img1 = cap1.read()
        if not ret1: break
        idx2 = i - off
        if 0 <= idx2 < len(h2):
            cap2.set(1, idx2); _, img2 = cap2.read()
            lm2 = h2[idx2]
        else: img2, lm2 = np.zeros((720, w2, 3), dtype=np.uint8), None

        for img, lms in [(img1, h1[i]), (img2, lm2)]:
            if lms:
                for s, e in FULL_SKELETON:
                    cv2.line(img, (int(lms[s].x*img.shape[1]), int(lms[s].y*img.shape[0])), 
                             (int(lms[e].x*img.shape[1]), int(lms[e].y*img.shape[0])), (127, 255, 0), 3)
        out.write(np.hstack((cv2.resize(img1, (w1, target_h)), cv2.resize(img2, (w2, target_h)))))
    
    cap1.release(); cap2.release(); out.release()
    
    # Critical WhatsApp Fix: Re-encode to H.264 YUV420P
    subprocess.run(f"ffmpeg -y -i {raw_out} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 28 {final_out}", shell=True, capture_output=True)
    return final_out

def analyze_video(path, model):
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.VIDEO))
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    history, raw_xyz = [], []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        ts = int((len(history) * 1000) / (fps if fps > 0 else 30))
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        lms = res.pose_landmarks[0] if res.pose_landmarks else None
        history.append(lms)
        raw_xyz.append([{"x": l.x, "y": l.y, "z": l.z} for l in lms] if lms else None)
    cap.release()
    return {"history": history, "raw": raw_xyz, "fps": fps, "total": len(history)}

# --- 3. ALL ORIGINAL TABS RESTORED ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>Pro Biometrics Engine</p>", unsafe_allow_html=True)

SPORT_MAP = {
    "TENNIS 🎾": ["Serve", "Forehand", "Backhand"],
    "PADEL 🎾": ["Bandeja", "Vibora", "Smash"],
    "PICKLEBALL 🥒": ["Dink", "Kitchen Volley"],
    "GOLF ⛳": ["Driver Swing", "Iron Swing"],
    "BADMINTON 🏸": ["Jump Smash", "Drop Shot"],
    "CRICKET BATTING 🏏": ["Drive", "Pull Shot"],
    "CRICKET BOWLING ⚾": ["Fast Bowling", "Spin"],
    "GYM 🏋️": ["Squat", "Deadlift"],
    "YOGA 🧘": ["Warrior", "Balance"]
}

tabs = st.tabs(list(SPORT_MAP.keys()))

for i, (sport, actions) in enumerate(SPORT_MAP.items()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"AI READY: {sport}")
            u1 = st.file_uploader("Lead Angle", type=["mp4","mov"], key=f"u1_{sport}")
            u2 = st.file_uploader("Side Angle", type=["mp4","mov"], key=f"u2_{sport}")
            sel_act = st.selectbox("Action", actions, key=f"act_{sport}")
            run_btn = st.button("RUN PRO ANALYSIS", key=f"run_{sport}", use_container_width=True)
        
        with col2:
            res_key = f"data_{sport}"
            if run_btn and u1 and u2:
                model_path = 'pose_landmarker_heavy.task'
                if not os.path.exists(model_path):
                    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", model_path)
                
                t1, t2 = tempfile.NamedTemporaryFile(delete=False), tempfile.NamedTemporaryFile(delete=False)
                t1.write(u1.getbuffer()); t2.write(u2.getbuffer())
                
                with st.status("Extracting Deep Telemetry...") as status:
                    d1, d2 = analyze_video(t1.name, model_path), analyze_video(t2.name, model_path)
                    st.session_state[res_key] = {"d1": d1, "d2": d2, "p1": t1.name, "p2": t2.name}

            if res_key in st.session_state and st.session_state[res_key]:
                s = st.session_state[res_key]
                st.markdown("### 🛠️ SYNC VERIFICATION")
                sl1 = st.slider("Lead Frame", 0, s['d1']['total']-1, key=f"sl1_{sport}")
                sl2 = st.slider("Side Frame", 0, s['d2']['total']-1, key=f"sl2_{sport}")
                
                # Preview Verification
                cap1, cap2 = cv2.VideoCapture(s['p1']), cv2.VideoCapture(s['p2'])
                cap1.set(1, sl1); cap2.set(1, sl2); _, f1 = cap1.read(); _, f2 = cap2.read()
                cap1.release(); cap2.release()
                if f1 is not None and f2 is not None:
                    st.image(np.hstack((cv2.resize(cv2.cvtColor(f1, 4), (640, 480)), cv2.resize(cv2.cvtColor(f2, 4), (640, 480)))))

                if st.button("🎬 GENERATE WHATSAPP PACK", key=f"gen_{sport}", use_container_width=True):
                    final_v = safe_render_and_encode(s['p1'], s['p2'], s['d1']['history'], s['d2']['history'], sl1, sl2, s['d1']['fps'])
                    if os.path.exists(final_v):
                        st.video(final_v)
                        with open(final_v, "rb") as f:
                            st.download_button("📥 DOWNLOAD MOBILE PACK", f.read(), f"{sport}_mobile.mp4", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
