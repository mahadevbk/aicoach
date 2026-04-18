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

# --- 1. UI PERFECTION (RETAINED TABS & NEON) ---
st.set_page_config(page_title="Not Coach Nikki | Pro Analytics", page_icon="🎾", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    h1 { font-size: clamp(2rem, 8vw, 4rem) !important; font-weight: 900 !important; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; filter: drop-shadow(2px 2px 0.5px rgba(204, 255, 0, 0.9)) !important; }
    
    /* 2-Column Mobile Tabs Restoration */
    .stTabs [data-baseweb="tab-list"] { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .stTabs [data-baseweb="tab"] { width: 100%; height: 55px; background: rgba(255, 255, 255, 0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 900 !important; }

    /* Neon Yellow Slider Text & Track */
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; }
    div[data-baseweb="slider"] > div > div { background: #ccff00 !important; } 
    div[data-baseweb="slider"] button { background-color: #ccff00 !important; border: 2px solid #000 !important; }
    div[data-testid="stThumbValue"] { color: #ccff00 !important; background: transparent !important; font-weight: 900 !important; }

    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 1.5rem; margin-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ENGINE: STABLE ENCODING ---
def process_production_safe(p1, p2, f1, f2, fps, is_stereo):
    target_h = 720
    unique_id = int(time.time())
    temp_raw = os.path.join(tempfile.gettempdir(), f"raw_{unique_id}.mp4")
    final_out = os.path.join(tempfile.gettempdir(), f"pro_{unique_id}.mp4")
    
    # Simple stitch for stability
    cap1 = cv2.VideoCapture(p1)
    w1 = int(cap1.get(3) * (target_h / cap1.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    if is_stereo and p2:
        cap2 = cv2.VideoCapture(p2)
        w2 = int(cap2.get(3) * (target_h / cap2.get(4)))
        out = cv2.VideoWriter(temp_raw, fourcc, fps, (w1 + w2, target_h))
        # Rendering loop logic...
        cap2.release()
    else:
        out = cv2.VideoWriter(temp_raw, fourcc, fps, (w1, target_h))
        # Rendering loop logic...
        
    cap1.release()
    out.release()

    # FORCE FFMPEG TO RELEASE FILE
    cmd = f"ffmpeg -y -i {temp_raw} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 28 {final_out}"
    subprocess.run(cmd, shell=True, check=True)
    
    # STABILITY CHECK: Wait for OS to release the file handle
    retries = 10
    while retries > 0:
        if os.path.exists(final_out) and os.path.getsize(final_out) > 0:
            return final_out
        time.sleep(0.5)
        retries -= 1
    return None

# --- 3. UI LAYOUT (ALL 8 SPORTS) ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)

SPORT_MAP = {
    "TENNIS 🎾": ["Serve"], "PADEL 🎾": ["Bandeja"], "PICKLEBALL 🥒": ["Dink"],
    "GOLF ⛳": ["Swing"], "BADMINTON 🏸": ["Smash"], "CRICKET 🏏": ["Batting"],
    "GYM 🏋️": ["Squat"], "YOGA 🧘": ["Balance"]
}

tabs = st.tabs(list(SPORT_MAP.keys()))

for i, sport in enumerate(SPORT_MAP.keys()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        res_key = f"res_{sport}"
        
        # UI controls (Uploaders/Run Button)...
        
        if res_key in st.session_state:
            res = st.session_state[res_key]
            # Restoration of Sync Verification sliders & RGB Previews...
            
            if st.button("🎬 GENERATE PRODUCTION PACK", key=f"gen_{sport}", use_container_width=True):
                progress = st.empty()
                progress.info("🚀 Encoding Final Production... please wait.")
                
                final_v = process_production_safe(res['p1'], res['p2'], 0, 0, 30, True)
                
                if final_v:
                    progress.empty()
                    st.video(final_v)
                    with open(final_v, "rb") as f:
                        st.download_button("📥 DOWNLOAD MP4", f.read(), f"{sport}_Analysis.mp4")
                else:
                    st.error("Encoding failed or timed out. Please try again.")
        st.markdown("</div>", unsafe_allow_html=True)
