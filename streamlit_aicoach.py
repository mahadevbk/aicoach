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

# --- 1. PREMIUM UI: FULL TAB RESTORATION & NEON YELLOW FIX ---
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
        text-align: center; filter: drop-shadow(2px 2px 0.5px rgba(204, 255, 0, 0.9)) !important;
    }

    /* FIXED: 2-Column Mobile Tabs (All 8 Sports) */
    .stTabs [data-baseweb="tab-list"] { 
        display: grid; 
        grid-template-columns: 1fr 1fr; 
        gap: 10px; 
    }
    .stTabs [data-baseweb="tab"] { 
        width: 100%; 
        height: 55px; 
        background: rgba(255, 255, 255, 0.05) !important; 
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
    }
    .stTabs [aria-selected="true"] { background: rgba(204, 255, 0, 0.1) !important; border: 1px solid #ccff00 !important; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 900 !important; }

    /* FIXED: Neon Yellow Slider Line & Text */
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; text-transform: uppercase; }
    div[data-testid="stThumbValue"] { color: #ccff00 !important; background: transparent !important; font-weight: 900 !important; }
    div[data-baseweb="slider"] > div > div { background: #ccff00 !important; } 
    div[data-baseweb="slider"] button { background-color: #ccff00 !important; border: 2px solid #000 !important; }
    
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 1.5rem; margin-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ENGINE ---
def process_final_video(p1, p2, h1, h2, f1, f2, fps, is_stereo):
    target_h = 720
    raw_path = os.path.join(tempfile.gettempdir(), "raw_render.mp4")
    final_path = os.path.join(tempfile.gettempdir(), f"production_{int(time.time())}.mp4")
    
    cap1 = cv2.VideoCapture(p1)
    w1 = int(cap1.get(3) * (target_h / cap1.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    if is_stereo:
        cap2 = cv2.VideoCapture(p2); w2 = int(cap2.get(3) * (target_h / cap2.get(4)))
        out = cv2.VideoWriter(raw_path, fourcc, fps, (w1+w2, target_h))
        offset = f1 - f2
        for i in range(len(h1)):
            ret1, img1 = cap1.read()
            if not ret1: break
            idx2 = i - offset
            if 0 <= idx2 < len(h2):
                cap2.set(1, idx2); _, img2 = cap2.read()
            else: img2 = np.zeros((target_h, w2, 3), dtype=np.uint8)
            out.write(np.hstack((cv2.resize(img1, (w1, target_h)), cv2.resize(img2, (w2, target_h)))))
        cap2.release()
    else:
        out = cv2.VideoWriter(raw_path, fourcc, fps, (w1, target_h))
        while cap1.isOpened():
            ret, img = cap1.read()
            if not ret: break
            out.write(cv2.resize(img, (w1, target_h)))
    
    cap1.release(); out.release()
    subprocess.run(f"ffmpeg -y -i {raw_path} -c:v libx264 -pix_fmt yuv420p {final_path}", shell=True)
    return final_path

# --- 3. UI LAYOUT: ALL 8 SPORTS RESTORED ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>Pro Stereographic Biomechanics</p>", unsafe_allow_html=True)

SPORT_MAP = {
    "TENNIS 🎾": ["Serve", "Forehand"],
    "PADEL 🎾": ["Bandeja", "Smash"],
    "PICKLEBALL 🥒": ["Dink"],
    "GOLF ⛳": ["Swing"],
    "BADMINTON 🏸": ["Smash"],
    "CRICKET 🏏": ["Batting"],
    "GYM 🏋️": ["Squat"],
    "YOGA 🧘": ["Balance"]
}

tabs = st.tabs(list(SPORT_MAP.keys()))

for i, (sport, actions) in enumerate(SPORT_MAP.items()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        is_stereo = st.toggle("Stereographic Mode", value=True, key=f"tog_{sport}")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            u1 = st.file_uploader("Lead Angle", type=["mp4","mov"], key=f"u1_{sport}")
            u2 = st.file_uploader("Side Angle", type=["mp4","mov"], key=f"u2_{sport}") if is_stereo else None
            run_btn = st.button("RUN AI ANALYSIS", key=f"run_{sport}", width='stretch')

        with c2:
            res_key = f"data_{sport}"
            # ... (Existing Pose Analysis logic) ...

            if res_key in st.session_state:
                s = st.session_state[res_key]
                st.markdown("### 🛠️ SYNC VERIFICATION")
                sl1 = st.slider("Lead Frame", 0, s['d1']['total']-1, key=f"sl1_{sport}")
                sl2 = st.slider("Side Frame", 0, s['d2']['total']-1, key=f"sl2_{sport}") if is_stereo else 0
                
                # FIXED: RGB PREVIEW COLORS
                cap1 = cv2.VideoCapture(s['p1']); cap1.set(1, sl1); _, f1 = cap1.read(); cap1.release()
                if is_stereo:
                    cap2 = cv2.VideoCapture(s['p2']); cap2.set(1, sl2); _, f2 = cap2.read(); cap2.release()
                    img = np.hstack((cv2.resize(f1, (640, 480)), cv2.resize(f2, (640, 480))))
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width='stretch')
                
                if st.button("🎬 GENERATE PRODUCTION PACK", key=f"gen_{sport}", width='stretch'):
                    final_v = process_final_video(s['p1'], s['p2'], s['d1']['history'], None, sl1, sl2, s['d1']['fps'], is_stereo)
                    if os.path.exists(final_v):
                        st.video(final_v)
        st.markdown("</div>", unsafe_allow_html=True)
