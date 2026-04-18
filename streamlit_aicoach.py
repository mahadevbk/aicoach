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

# --- 1. PREMIUM UI: NEON YELLOW FIXED & RGB PREVIEW ---
st.set_page_config(page_title="Not Coach Nikki | Pro Analytics", page_icon="🎾", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    
    /* Title & Hero */
    h1 { 
        font-size: clamp(2rem, 8vw, 4rem) !important; font-weight: 900 !important; 
        background: linear-gradient(to right, #38bdf8, #818cf8); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
        text-align: center; filter: drop-shadow(2px 2px 0.5px rgba(204, 255, 0, 0.9)) !important;
    }
    .hero-sub { text-align: center; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 4px; margin-bottom: 2rem; }

    /* FIXED: NEON YELLOW SLIDER TEXT & TRACK */
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; }
    div[data-testid="stThumbValue"] { color: #ccff00 !important; background: transparent !important; font-weight: 900 !important; }
    div[data-baseweb="slider"] > div > div { background: #ccff00 !important; } 
    div[data-baseweb="slider"] button { background-color: #ccff00 !important; border: 2px solid #000 !important; }

    /* Glass Cards & Tabs */
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 2rem; margin-bottom: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; justify-content: center; }
    .stTabs [data-baseweb="tab"] { height: 60px; background: rgba(255, 255, 255, 0.05) !important; border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.1) !important; }
    .stTabs [aria-selected="true"] { background: rgba(204, 255, 0, 0.1) !important; border: 1px solid #ccff00 !important; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 900 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CORE PRODUCTION ENGINE ---
FULL_SKELETON = [(0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10), (11,12), (11,13), (13,15), (12,14), (14,16), (11,23), (12,24), (23,24), (23,25), (25,27), (24,26), (26,28)]

def process_and_stitch(p1, p2, h1, h2, f1, f2, fps, is_stereo):
    target_h = 720
    out_name = f"final_{int(time.time())}.mp4"
    temp_raw = os.path.join(tempfile.gettempdir(), "raw_render.mp4")
    final_path = os.path.join(tempfile.gettempdir(), out_name)
    
    cap1 = cv2.VideoCapture(p1)
    w1 = int(cap1.get(3) * (target_h / cap1.get(4)))
    
    if is_stereo:
        cap2 = cv2.VideoCapture(p2)
        w2 = int(cap2.get(3) * (target_h / cap2.get(4)))
        out = cv2.VideoWriter(temp_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w1+w2, target_h))
        offset = f1 - f2
        for i in range(len(h1)):
            ret1, img1 = cap1.read()
            if not ret1: break
            idx2 = i - offset
            if 0 <= idx2 < len(h2):
                cap2.set(1, idx2); _, img2 = cap2.read()
                lms2 = h2[idx2]
            else: img2, lms2 = np.zeros((target_h, w2, 3), dtype=np.uint8), None
            
            for img, lms in [(img1, h1[i]), (img2, lms2)]:
                if img is not None and lms:
                    img_res = cv2.resize(img, (w1 if img is img1 else w2, target_h))
                    for s, e in FULL_SKELETON:
                        cv2.line(img_res, (int(lms[s].x*img_res.shape[1]), int(lms[s].y*img_res.shape[0])), (int(lms[e].x*img_res.shape[1]), int(lms[e].y*img_res.shape[0])), (127, 255, 0), 3)
                    if img is img1: img1 = img_res
                    else: img2 = img_res
            out.write(np.hstack((img1, img2)))
        cap2.release()
    else:
        out = cv2.VideoWriter(temp_raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w1, target_h))
        for i, lms in enumerate(h1):
            ret, img = cap1.read()
            if not ret: break
            img = cv2.resize(img, (w1, target_h))
            if lms:
                for s, e in FULL_SKELETON:
                    cv2.line(img, (int(lms[s].x*w1), int(lms[s].y*target_h)), (int(lms[e].x*w1), int(lms[e].y*target_h)), (127, 255, 0), 3)
            out.write(img)
    
    cap1.release(); out.release()
    # RE-ENCODE FOR WHATSAPP/STREMLIT PLAYER
    subprocess.run(f"ffmpeg -y -i {temp_raw} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 28 {final_path}", shell=True)
    return final_path

# --- 3. UI TAB RESTORATION ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>Pro Stereographic Biomechanics</p>", unsafe_allow_html=True)

SPORT_MAP = {
    "TENNIS 🎾": ["Serve", "Forehand"], "PADEL 🎾": ["Bandeja"], "PICKLEBALL 🥒": ["Dink"],
    "GOLF ⛳": ["Swing"], "BADMINTON 🏸": ["Smash"], "CRICKET 🏏": ["Batting"],
    "GYM 🏋️": ["Squat"], "YOGA 🧘": ["Alignment"]
}

tabs = st.tabs(list(SPORT_MAP.keys()))

for i, (sport, actions) in enumerate(SPORT_MAP.items()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        is_stereo = st.toggle("Stereographic Mode", value=True, key=f"tog_{sport}")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            u1 = st.file_uploader("Lead Video", type=["mp4","mov"], key=f"u1_{sport}")
            u2 = st.file_uploader("Side Video", type=["mp4","mov"], key=f"u2_{sport}") if is_stereo else None
            run_btn = st.button("RUN AI ANALYSIS", key=f"run_{sport}", use_container_width=True)

        with c2:
            res_key = f"data_{sport}"
            # ... [Pose Analysis logic to fill s_state[res_key]] ...

            if res_key in st.session_state:
                s = st.session_state[res_key]
                st.markdown("### 🛠️ SYNC VERIFICATION")
                sl1 = st.slider("Lead Frame", 0, s['d1']['total']-1, key=f"sl1_{sport}")
                sl2 = st.slider("Side Frame", 0, s['d2']['total']-1, key=f"sl2_{sport}") if is_stereo else 0
                
                # FIXED: RGB PREVIEW (No more blue skin/sky)
                cap1 = cv2.VideoCapture(s['p1']); cap1.set(1, sl1); _, f1 = cap1.read(); cap1.release()
                if is_stereo:
                    cap2 = cv2.VideoCapture(s['p2']); cap2.set(1, sl2); _, f2 = cap2.read(); cap2.release()
                    combined = np.hstack((cv2.resize(f1, (640, 480)), cv2.resize(f2, (640, 480))))
                    st.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                if st.button("🎬 GENERATE PRODUCTION PACK", key=f"gen_{sport}", use_container_width=True):
                    with st.spinner("Processing Final Video..."):
                        final_v = process_and_stitch(s['p1'], s['p2'], s['d1']['history'], 
                                                     s['d2']['history'] if is_stereo else None, 
                                                     sl1, sl2, s['d1']['fps'], is_stereo)
                        if os.path.exists(final_v):
                            st.video(final_v)
                            with open(final_v, "rb") as f:
                                st.download_button("📥 DOWNLOAD MOBILE PACK", f.read(), f"{sport}_PRO.mp4")
        st.markdown("</div>", unsafe_allow_html=True)
