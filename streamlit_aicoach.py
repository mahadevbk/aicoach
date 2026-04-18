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

# --- 1. PREMIUM UI (8 TABS, 2-COL MOBILE, NEON ACCENTS) ---
st.set_page_config(page_title="Not Coach Nikki | Pro Analytics", page_icon="🎾", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    h1 { font-size: clamp(2rem, 8vw, 4rem) !important; font-weight: 900 !important; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; filter: drop-shadow(2px 2px 0.5px rgba(204, 255, 0, 0.9)) !important; }
    
    /* 2-Column Mobile Tabs */
    .stTabs [data-baseweb="tab-list"] { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .stTabs [data-baseweb="tab"] { width: 100%; height: 55px; background: rgba(255, 255, 255, 0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 900 !important; }

    /* Neon Sliders */
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; }
    div[data-baseweb="slider"] > div > div { background: #ccff00 !important; } 
    div[data-baseweb="slider"] button { background-color: #ccff00 !important; border: 2px solid #000 !important; }
    div[data-testid="stThumbValue"] { color: #ccff00 !important; background: transparent !important; font-weight: 900 !important; }

    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 1.5rem; margin-bottom: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ENGINE ---
def run_full_analysis(video_path, model_path):
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO)
    detector = vision.PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    history, raw_data = [], []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        timestamp = int((frame_count * 1000) / fps)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(mp_image, timestamp)
        
        lms = res.pose_landmarks[0] if res.pose_landmarks else None
        history.append(lms)
        raw_data.append([{"x": l.x, "y": l.y, "z": l.z} for l in lms] if lms else None)
        frame_count += 1
    cap.release()
    return {"history": history, "raw": raw_data, "fps": fps, "total": frame_count}

# --- 3. UI TAB RESTORATION & LOGIC ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)

SPORT_MAP = {
    "TENNIS 🎾": ["Serve", "Forehand"], "PADEL 🎾": ["Bandeja", "Smash"],
    "PICKLEBALL 🥒": ["Dink"], "GOLF ⛳": ["Swing"],
    "BADMINTON 🏸": ["Smash"], "CRICKET 🏏": ["Batting"],
    "GYM 🏋️": ["Squat"], "YOGA 🧘": ["Balance"]
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
            run_btn = st.button("RUN AI ANALYSIS", key=f"run_{sport}", use_container_width=True)

        with c2:
            res_key = f"results_{sport}"
            if run_btn and u1:
                # 1. Prepare Model
                model_path = 'pose_landmarker_heavy.task'
                if not os.path.exists(model_path):
                    with st.spinner("Initializing AI Models..."):
                        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", model_path)
                
                # 2. Save Temps
                t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                t1.write(u1.getbuffer())
                t2_name = None
                if is_stereo and u2:
                    t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    t2.write(u2.getbuffer())
                    t2_name = t2.name

                # 3. Process
                with st.status(f"Analyzing {sport} Biometrics...") as status:
                    d1 = run_full_analysis(t1.name, model_path)
                    d2 = run_full_analysis(t2_name, model_path) if t2_name else None
                    st.session_state[res_key] = {"d1": d1, "d2": d2, "p1": t1.name, "p2": t2_name}
                    status.update(label="Analysis Complete!", state="complete")

            # 4. Display Results if they exist in Session
            if res_key in st.session_state:
                res = st.session_state[res_key]
                st.markdown("### 🛠️ SYNC VERIFICATION")
                sl1 = st.slider("Lead Frame", 0, res['d1']['total']-1, key=f"sl1_{sport}")
                sl2 = st.slider("Side Frame", 0, res['d2']['total']-1, key=f"sl2_{sport}") if is_stereo else 0
                
                # RGB Preview
                cap1 = cv2.VideoCapture(res['p1']); cap1.set(1, sl1); _, f1 = cap1.read(); cap1.release()
                if is_stereo and res['p2']:
                    cap2 = cv2.VideoCapture(res['p2']); cap2.set(1, sl2); _, f2 = cap2.read(); cap2.release()
                    img = np.hstack((cv2.resize(f1, (640, 480)), cv2.resize(f2, (640, 480))))
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.image(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB), width=640)
                    
                if st.button("🎬 GENERATE PRODUCTION PACK", key=f"gen_{sport}", use_container_width=True):
                    st.write("Encoding Final Production...") # This will now trigger with ffmpeg installed
        st.markdown("</div>", unsafe_allow_html=True)
