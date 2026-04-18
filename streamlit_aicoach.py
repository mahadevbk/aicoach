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
import subprocess
import time

# --- 1. YOUR ORIGINAL PREMIUM UI (RESTORED FROM FILE) ---
st.set_page_config(
    page_title="Not Coach Nikki | Pro Analytics",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    
    .stApp { 
        background: radial-gradient(circle at top right, #1e293b, #020617); 
        color: #f8fafc; 
        font-family: 'Roboto Flex', sans-serif; 
    }
    
    /* Premium Glass Cards */
    .glass-card { 
        background: rgba(255, 255, 255, 0.03); 
        backdrop-filter: blur(12px); 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        border-radius: 24px; 
        padding: 2rem; 
        margin-bottom: 2rem; 
    }
    
    /* Neon Custom Components */
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; text-transform: uppercase; }
    div[data-testid="stThumbValue"] { background-color: #ccff00 !important; color: black !important; font-weight: 900 !important; }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; justify-content: center; }
    .stTabs [data-baseweb="tab"] { 
        height: 60px; background: rgba(255, 255, 255, 0.05) !important; 
        border-radius: 12px !important; padding: 0 25px !important; border: 1px solid rgba(255,255,255,0.1) !important;
    }
    .stTabs [aria-selected="true"] { background: rgba(204, 255, 0, 0.1) !important; border: 1px solid #ccff00 !important; }
    
    h1 { font-size: clamp(2rem, 8vw, 4rem) !important; font-weight: 900 !important; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0px !important; }
    .hero-sub { text-align: center; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 4px; margin-bottom: 3rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE PRO ENGINE (STEREOGRAPHIC & WHATSAPP) ---
FULL_SKELETON = [(0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10), (11,12), (11,13), (13,15), (12,14), (14,16), (11,23), (12,24), (23,24), (23,25), (25,27), (24,26), (26,28)]

def render_and_encode(p1, p2, h1, h2, f1, f2, fps, is_stereo):
    target_h = 720
    ts = int(time.time())
    raw_p = os.path.join(tempfile.gettempdir(), f"raw_{ts}.mp4")
    final_p = os.path.join(tempfile.gettempdir(), f"final_{ts}.mp4")
    
    cap1 = cv2.VideoCapture(p1)
    w1 = int(cap1.get(3) * (target_h / cap1.get(4)))
    
    if is_stereo:
        cap2 = cv2.VideoCapture(p2)
        w2 = int(cap2.get(3) * (target_h / cap2.get(4)))
        out = cv2.VideoWriter(raw_p, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w1+w2, target_h))
        off = f1 - f2
        for i in range(len(h1)):
            ret1, img1 = cap1.read()
            if not ret1: break
            idx2 = i - off
            if 0 <= idx2 < len(h2):
                cap2.set(1, idx2); _, img2 = cap2.read()
                lms2 = h2[idx2]
            else: img2, lms2 = np.zeros((target_h, w2, 3), dtype=np.uint8), None
            
            for img, lms in [(img1, h1[i]), (img2, lms2)]:
                if lms:
                    for s, e in FULL_SKELETON:
                        cv2.line(img, (int(lms[s].x*img.shape[1]), int(lms[s].y*img.shape[0])), (int(lms[e].x*img.shape[1]), int(lms[e].y*img.shape[0])), (127, 255, 0), 3)
            out.write(np.hstack((cv2.resize(img1, (w1, target_h)), cv2.resize(img2, (w2, target_h)))))
        cap2.release()
    else:
        out = cv2.VideoWriter(raw_p, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w1, target_h))
        for i, lms in enumerate(h1):
            ret, img = cap1.read()
            if not ret: break
            if lms:
                for s, e in FULL_SKELETON:
                    cv2.line(img, (int(lms[s].x*img.shape[1]), int(lms[s].y*img.shape[0])), (int(lms[e].x*img.shape[1]), int(lms[e].y*img.shape[0])), (127, 255, 0), 3)
            out.write(cv2.resize(img, (w1, target_h)))
            
    cap1.release(); out.release()
    # WhatsApp Mobile Encoding Fix
    subprocess.run(f"ffmpeg -y -i {raw_p} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 25 {final_p}", shell=True, capture_output=True)
    return final_p

def analyze_full(path, model):
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

# --- 3. THE UI: ALL TABS + DUAL MODE ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>Professional Biomechanics AI Engine</p>", unsafe_allow_html=True)

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
        is_stereo = st.toggle("Stereographic Mode (Dual Camera)", value=True, key=f"tog_{sport}")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            u1 = st.file_uploader("Lead Video", type=["mp4","mov"], key=f"u1_{sport}")
            u2 = st.file_uploader("Side Video", type=["mp4","mov"], key=f"u2_{sport}") if is_stereo else None
            sel_act = st.selectbox("Action", actions, key=f"act_{sport}")
            run_btn = st.button("RUN PRO ANALYSIS", key=f"run_{sport}", use_container_width=True)

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
                
                with st.status("Analyzing Biometrics...") as status:
                    d1 = analyze_full(t1.name, model_p)
                    d2 = analyze_full(t2_n, model_p) if is_stereo else None
                    st.session_state[res_key] = {"d1": d1, "d2": d2, "p1": t1.name, "p2": t2_n}

            if res_key in st.session_state:
                s = st.session_state[res_key]
                st.markdown("### 🛠️ SYNC & EXPORT")
                sl1 = st.slider("Lead Frame", 0, s['d1']['total']-1, key=f"sl1_{sport}")
                sl2 = st.slider("Side Frame", 0, s['d2']['total']-1, key=f"sl2_{sport}") if is_stereo else 0
                
                if st.button("🎬 GENERATE WHATSAPP PACK", key=f"gen_{sport}", use_container_width=True):
                    final_v = render_and_encode(s['p1'], s['p2'], s['d1']['history'], s['d2']['history'] if is_stereo else None, sl1, sl2, s['d1']['fps'], is_stereo)
                    st.video(final_v)
                    with open(final_v, "rb") as f:
                        st.download_button("📥 DOWNLOAD MOBILE PACK", f.read(), f"{sport}_mobile.mp4", use_container_width=True)
                        
                    # Dense JSON
                    telemetry = {"metadata": {"sport": sport, "stereo": is_stereo}, "lead_raw": s['d1']['raw'], "side_raw": s['d2']['raw'] if is_stereo else None}
                    st.download_button("📊 DOWNLOAD FULL DATA (JSON)", json.dumps(telemetry), f"{sport}_telemetry.json", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
