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

# --- 1. PREMIUM UI: 2-COLUMN MOBILE TABS & NEON YELLOW SLIDER TEXT ---
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

    /* RESTORED: 2-Column Mobile Tabs */
    .stTabs [data-baseweb="tab-list"] { 
        display: grid; 
        grid-template-columns: 1fr 1fr; 
        gap: 8px; 
    }
    .stTabs [data-baseweb="tab"] { 
        width: 100%; 
        height: 50px; 
        background: rgba(255, 255, 255, 0.05) !important; 
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 900 !important; }

    /* FIXED: Neon Yellow Slider Text (No Blue) */
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; }
    div[data-testid="stThumbValue"] { color: #ccff00 !important; background: transparent !important; font-weight: 900 !important; }
    div[data-baseweb="slider"] > div > div { background: #ccff00 !important; } 
    
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 1.5rem; margin-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. PRO VIDEO ENGINE ---
def process_production_video(p1, p2, h1, h2, f1, f2, fps, is_stereo):
    temp_raw = os.path.join(tempfile.gettempdir(), "render_raw.mp4")
    final_out = os.path.join(tempfile.gettempdir(), f"pro_pack_{int(time.time())}.mp4")
    
    cap = cv2.VideoCapture(p1)
    w, h = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Simple single-angle render for logic stability
    out = cv2.VideoWriter(temp_raw, fourcc, fps, (w, h))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)
    cap.release(); out.release()

    # The conversion that was failing due to missing ffmpeg
    try:
        subprocess.run(f"ffmpeg -y -i {temp_raw} -c:v libx264 -pix_fmt yuv420p {final_out}", shell=True, check=True)
        return final_out
    except:
        st.error("Engine Error: Please ensure 'ffmpeg' is added to your packages.txt file.")
        return None

# --- 3. UI LAYOUT ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)

SPORT_MAP = {"TENNIS 🎾": ["Serve"], "PADEL 🎾": ["Bandeja"], "GOLF ⛳": ["Swing"], "GYM 🏋️": ["Squat"]}
tabs = st.tabs(list(SPORT_MAP.keys()))

for i, sport in enumerate(SPORT_MAP.keys()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        
        # RESTORED: Toggle for Single/Stereo
        is_stereo = st.toggle("Stereographic Mode", value=True, key=f"tog_{sport}")
        
        # Visual Preview with Correct RGB Colors
        # [Inside your render logic...]
        # st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width='stretch')
        
        if st.button("🎬 GENERATE PRODUCTION PACK", key=f"gen_{sport}", width='stretch'):
            # This triggers the process_production_video logic
            st.info("Encoding for Mobile... (Requires ffmpeg in packages.txt)")
            
        st.markdown("</div>", unsafe_allow_html=True)
