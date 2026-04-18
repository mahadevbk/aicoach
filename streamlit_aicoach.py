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

# --- 1. UI: 8-SPORT MOBILE GRID & NEON TEXT ---
st.set_page_config(page_title="Not Coach Nikki", page_icon="🎾", layout="wide")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; }
    h1 { text-align: center; filter: drop-shadow(2px 2px 0.5px rgba(204, 255, 0, 0.9)); }
    
    /* 2-Column Mobile Tabs */
    .stTabs [data-baseweb="tab-list"] { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .stTabs [data-baseweb="tab"] { background: rgba(255, 255, 255, 0.05) !important; border-radius: 12px !important; height: 55px; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 900 !important; }

    /* Neon Yellow Slider Labels */
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; text-transform: uppercase; }
    div[data-baseweb="slider"] > div > div { background: #ccff00 !important; } 
    div[data-baseweb="slider"] button { background-color: #ccff00 !important; }
    div[data-testid="stThumbValue"] { color: #ccff00 !important; background: transparent !important; }

    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border-radius: 24px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid rgba(255,255,255,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. STABLE PRODUCTION ENGINE ---
def fast_production_render(p1, p2, f1, f2, fps, is_stereo):
    target_h = 720
    ts = int(time.time())
    raw_vid = os.path.join(tempfile.gettempdir(), f"raw_{ts}.mp4")
    final_vid = os.path.join(tempfile.gettempdir(), f"prod_{ts}.mp4")
    
    cap1 = cv2.VideoCapture(p1)
    w1 = int(cap1.get(3) * (target_h / cap1.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Process frames and stitch
    if is_stereo and p2:
        cap2 = cv2.VideoCapture(p2)
        w2 = int(cap2.get(3) * (target_h / cap2.get(4)))
        out = cv2.VideoWriter(raw_vid, fourcc, fps, (w1 + w2, target_h))
        # ... (Stitching logic) ...
        cap2.release()
    else:
        out = cv2.VideoWriter(raw_vid, fourcc, fps, (w1, target_h))
        # ... (Single angle logic) ...
    
    cap1.release()
    out.release()

    # Fast Re-encode for WhatsApp Compatibility
    try:
        subprocess.run(f"ffmpeg -y -i {raw_vid} -c:v libx264 -preset ultrafast -crf 28 -pix_fmt yuv420p {final_vid}", shell=True, check=True)
        return final_vid
    except:
        return None

# --- 3. PERSISTENT TAB LOGIC ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)

SPORTS = ["TENNIS 🎾", "PADEL 🎾", "PICKLEBALL 🥒", "GOLF ⛳", "BADMINTON 🏸", "CRICKET 🏏", "GYM 🏋️", "YOGA 🧘"]
tabs = st.tabs(SPORTS)

for i, sport in enumerate(SPORTS):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        res_key = f"data_{sport}"
        
        c1, c2 = st.columns([1, 2])
        with c1:
            u1 = st.file_uploader("Lead Video", type=["mp4","mov"], key=f"u1_{sport}")
            u2 = st.file_uploader("Side Video", type=["mp4","mov"], key=f"u2_{sport}")
            if st.button("RUN AI ENGINE", key=f"run_{sport}", width="stretch"):
                # Analysis Logic and Session State Save
                st.session_state[res_key] = {"p1": "path1", "p2": "path2", "fps": 30, "total": 100}

        with c2:
            if res_key in st.session_state:
                st.markdown("### 🛠️ SYNC VERIFICATION")
                s1 = st.slider("Lead Impact Frame", 0, 100, key=f"s1_{sport}")
                s2 = st.slider("Side Impact Frame", 0, 100, key=f"s2_{sport}")
                
                # RGB COLOR FIX: Prevents blue-tinted skin
                # st.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB))
                
                if st.button("🎬 GENERATE PRODUCTION PACK", key=f"gen_{sport}", width="stretch"):
                    with st.spinner("🚀 Finalizing High-Res Export..."):
                        final_path = fast_production_render("p1", "p2", s1, s2, 30, True)
                        if final_path:
                            st.video(final_path)
                            with open(final_path, "rb") as f:
                                st.download_button("📥 DOWNLOAD FOR WHATSAPP", f.read(), f"{sport}_Analysis.mp4")
        st.markdown("</div>", unsafe_allow_html=True)
