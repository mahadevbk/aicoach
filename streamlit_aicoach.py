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
import urllib.request

# --- 1. PAGE CONFIG & ENHANCED CSS ---
st.set_page_config(
    page_title="Not Coach Nikki | Pro Analytics",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    
    /* FIX: Legibility for Sliders and Text */
    .stSlider label { color: #ccff00 !important; font-weight: 700 !important; font-size: 1rem !important; }
    .stSlider [data-baseweb="typography"] { color: #ffffff !important; font-weight: 600 !important; }
    
    [data-testid="stMetricValue"] { font-size: 28px !important; color: #ccff00 !important; font-weight: 800; }
    [data-testid="stMetricLabel"] { font-size: 14px !important; color: #94a3b8 !important; text-transform: uppercase; letter-spacing: 1px; }
    
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 1.5rem; margin-bottom: 2rem; }
    
    h1 { font-size: clamp(1.8rem, 7vw, 3.5rem) !important; font-weight: 800 !important; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; filter: drop-shadow(2px 2px 0.5px rgba(204, 255, 0, 0.9)); }
    .hero-subtext { text-align: center; color: #94a3b8; font-size: 0.75rem; margin-bottom: 2rem; text-transform: uppercase; letter-spacing: 2px; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 10px; display: flex; flex-wrap: wrap; justify-content: center; background-color: transparent !important; }
    .stTabs [data-baseweb="tab"] { height: 55px; flex: 1 1 calc(50% - 10px); min-width: 140px; background: rgba(255, 255, 255, 0.05) !important; border-radius: 14px !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; transition: all 0.3s ease; }
    .stTabs [aria-selected="true"] { background: rgba(204, 255, 0, 0.1) !important; border: 1px solid #ccff00 !important; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ENGINE LOGIC (Updated for Color Accuracy) ---
def download_model():
    model_path = 'pose_landmarker_heavy.task'
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        with st.spinner("🚀 Downloading Heavy AI Engine..."):
            urllib.request.urlretrieve(url, model_path)
    return model_path

def analyze_impact_frame(video_path, model_path):
    detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO
    ))
    cap = cv2.VideoCapture(video_path)
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    impact_frame, peak_vel, prev_wrist = 0, 0, None
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        # IMPORTANT: MediaPipe requires RGB [cite: 1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(mp_image, int((i * 1000) / fps))
        
        if res.pose_world_landmarks:
            wrist = res.pose_world_landmarks[0][15]
            if prev_wrist:
                vel = np.linalg.norm(np.array([wrist.x, wrist.y, wrist.z]) - np.array([prev_wrist.x, prev_wrist.y, prev_wrist.z]))
                if vel > peak_vel: peak_vel, impact_frame = vel, i
            prev_wrist = wrist
    cap.release()
    return {"impact": impact_frame, "fps": fps, "total": total_frames}

def render_final_stereo(p1, p2, f1, f2, fps):
    """Generates the combined side-by-side production video."""
    cap1, cap2 = cv2.VideoCapture(p1), cv2.VideoCapture(p2)
    offset = f1 - f2
    w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    # Using 'avc1' or 'mp4v' for better browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out.name, fourcc, fps, (w * 2, h))

    # Reset cap1 to start
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    total_len = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(total_len):
        ret1, frame1 = cap1.read()
        if not ret1: break
        
        # Calculate synchronized frame index for Video 2
        idx2 = i - offset
        if 0 <= idx2 < cap2.get(cv2.CAP_PROP_FRAME_COUNT):
            cap2.set(cv2.CAP_PROP_POS_FRAMES, idx2)
            ret2, frame2 = cap2.read()
        else:
            frame2 = np.zeros_like(frame1)

        combined = np.hstack((frame1, frame2))
        out.write(combined)
        
    cap1.release(); cap2.release(); out.release()
    return temp_out.name

# --- 3. UI LAYOUT ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtext'>Pro Sports Biomechanics</p>", unsafe_allow_html=True)

SPORT_CONFIGS = {
    "TENNIS 🎾": {"Serve": "Toss height", "Forehand Drive": "Unit turn"},
    "PADEL 🎾": {"Bandeja": "Contact control"},
    "GOLF ⛳": {"Driver Swing": "Wide arc"}
}

tabs = st.tabs(list(SPORT_CONFIGS.keys()))

for i, (sport, actions) in enumerate(SPORT_CONFIGS.items()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col_setup, col_render = st.columns([1, 2])
        
        with col_setup:
            st.info(f"PRO {sport} ENGINE ACTIVE")
            is_stereo = st.toggle("Stereographic Mode", key=f"tog_{sport}", value=True)
            up1 = st.file_uploader(f"Upload Lead Angle", type=["mp4", "mov"], key=f"up1_{sport}")
            up2 = st.file_uploader(f"Upload Side Angle", type=["mp4", "mov"], key=f"up2_{sport}") if is_stereo else None
            run_btn = st.button("RUN AI ENGINE", key=f"btn_{sport}", use_container_width=True)

        with col_render:
            if run_btn and up1:
                model_path = download_model()
                t1_path = os.path.join(tempfile.gettempdir(), f"lead_{sport}.mp4")
                with open(t1_path, "wb") as f: f.write(up1.getbuffer())
                
                with st.status("Analyzing Biometrics...") as status:
                    res1 = analyze_impact_frame(t1_path, model_path)
                    res2 = None
                    t2_path = None
                    if is_stereo and up2:
                        t2_path = os.path.join(tempfile.gettempdir(), f"side_{sport}.mp4")
                        with open(t2_path, "wb") as f: f.write(up2.getbuffer())
                        res2 = analyze_impact_frame(t2_path, model_path)
                        st.session_state[f"sync_{sport}"] = {"d1": res1, "d2": res2, "p1": t1_path, "p2": t2_path}
                    else:
                        st.video(t1_path)

            # --- VERIFICATION SCRUBBER (Legibility & Color Fixed) ---
            if is_stereo and f"sync_{sport}" in st.session_state:
                s = st.session_state[f"sync_{sport}"]
                st.markdown("### 🛠️ SYNC VERIFICATION")
                
                sl1 = st.slider("Lead Impact Frame", 0, s['d1']['total']-1, s['d1']['impact'], key=f"sl1_{sport}")
                sl2 = st.slider("Side Impact Frame", 0, s['d2']['total']-1, s['d2']['impact'], key=f"sl2_{sport}")
                
                cap1, cap2 = cv2.VideoCapture(s['p1']), cv2.VideoCapture(s['p2'])
                cap1.set(cv2.CAP_PROP_POS_FRAMES, sl1); cap2.set(cv2.CAP_PROP_POS_FRAMES, sl2)
                ret1, img1 = cap1.read(); ret2, img2 = cap2.read()
                cap1.release(); cap2.release()
                
                if ret1 and ret2:
                    # FIX: Convert BGR to RGB for correct display colors
                    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    
                    preview = np.hstack((cv2.resize(img1_rgb, (640, 480)), cv2.resize(img2_rgb, (640, 480))))
                    st.image(preview, caption="Verify synchronized impact point.")
                    
                if st.button("🎬 GENERATE PRODUCTION PACK", key=f"gen_{sport}", use_container_width=True):
                    with st.spinner("🔨 Rendering Final Video..."):
                        final_path = render_final_stereo(s['p1'], s['p2'], sl1, sl2, s['d1']['fps'])
                        
                        st.success(f"Final Analysis Ready! Offset: {sl1 - sl2} frames.")
                        st.video(final_path)
                        
                        with open(final_path, "rb") as f:
                            st.download_button("📥 Download Final Video", f, file_name=f"{sport}_stereo_final.mp4")

        st.markdown("</div>", unsafe_allow_html=True)
