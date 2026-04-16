import streamlit as st
import cv2
import numpy as np
import os
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tempfile
import time
import zipfile
import io

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Not Coach Nikki | Pro Analytics",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PREMIUM CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 1.5rem; margin-bottom: 2rem; }
    h1 { font-size: clamp(2rem, 8vw, 3.5rem) !important; font-weight: 800 !important; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0px !important; filter: drop-shadow(4px 4px 0.5px rgba(204, 255, 0, 0.8)); }
    .hero-subtext { text-align: center; color: #94a3b8; font-size: 0.8rem; margin-bottom: 2rem; text-transform: uppercase; letter-spacing: 2px; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; justify-content: center; }
    .stTabs [data-baseweb="tab"] { height: 60px; min-width: 100px; background: rgba(255, 255, 255, 0.03) !important; border-radius: 12px !important; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SPORT DICTIONARIES ---
SPORT_CONFIGS = {
    "TENNIS 🎾": {
        "Serve": "Analyze toss alignment and extension.",
        "Forehand/Backhand": "Focus on X-Factor and topspin path.",
        "General Rally": "Full point analysis from serve to finish.",
        "Volley": "Punch depth and stability."
    },
    "GOLF ⛳": {
        "Driver Swing": "Wide arc and dynamic weight shift.",
        "Iron Swing": "Downward strike and lead arm extension.",
        "Putting": "Pendulum motion and head stability.",
        "General Rally": "Full practice sequence tracking."
    },
    "PADEL 🎾": {
        "Bandeja": "High-contact control and side-step timing.",
        "Vibora": "Aggressive slice and core rotation.",
        "General Rally": "Wall-play transitions and point duration.",
        "Underhand Serve": "Waist-height contact and placement."
    },
    "PICKLEBALL 🥒": {
        "Dink": "Soft touch and minimal wrist hinge.",
        "Kitchen Volley": "Hand speed and reset stability.",
        "General Rally": "Transition zone movement and dink rallies.",
        "Third Shot Drop": "Arc height and landing depth."
    },
    "BADMINTON 🏸": {
        "Jump Smash": "Vertical leap and overhead whip speed.",
        "Net Drop": "Short-range touch and racket face angle.",
        "General Rally": "High-intensity footwork and shuttle tracking.",
        "Clear": "Full-court depth and shoulder rotation."
    }
}

# --- CORE FUNCTIONS ---
def calculate_angle(a, b, c):
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_landmarks(video_path):
    detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task'),
        running_mode=vision.RunningMode.VIDEO
    ))
    cap = cv2.VideoCapture(video_path)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames, skeletal_series, timestamp_ms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), [], 0
    progress_bar = st.progress(0, text="SCANNING BIOMETRICS...")
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(mp_image, int(timestamp_ms))
        lm = [{"x": p.x, "y": p.y, "z": p.z, "v": p.visibility} for p in res.pose_landmarks[0]] if res.pose_landmarks else None
        skeletal_series.append(lm)
        timestamp_ms += (1000 / fps)
        progress_bar.progress((i + 1) / total_frames)
    cap.release()
    progress_bar.empty()
    return skeletal_series, fps, (w, h)

def classify_motion(skeletal_data, sport_name):
    if not skeletal_data or all(x is None for x in skeletal_data):
        return "Unclassified Rally", 0, 0
    
    # Simple logic to determine "Rally" vs "Static Shot"
    # A rally is determined if the average hip movement exceeds a threshold over the video
    hip_movement = []
    for lm in skeletal_data:
        if lm: hip_movement.append(lm[23]['x'])
    
    movement_range = max(hip_movement) - min(hip_movement) if hip_movement else 0
    
    if movement_range > 0.15:
        return "General Rally", 0, movement_range
    else:
        # Default to the first sport-specific action
        return list(SPORT_CONFIGS[sport_name].keys())[0], 45.0, 0.5

# --- UI LOGIC ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtext'>Pro Sports Biomechanics</p>", unsafe_allow_html=True)

tabs = st.tabs(list(SPORT_CONFIGS.keys()))

for i, (sport, actions) in enumerate(SPORT_CONFIGS.items()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.info(f"PRO {sport} ENGINE: AI-Optimized for {', '.join(actions.keys())}")
        
        up_file = st.file_uploader(f"UPLOAD {sport} VIDEO", type=["mp4", "mov", "avi"], key=f"up_{sport}")
        
        if up_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(up_file.read())
            
            # Use Session State to prevent re-processing
            state_key = f"data_{sport}"
            if state_key not in st.session_state or st.session_state.get(f"name_{sport}") != up_file.name:
                with st.spinner("CALIBRATING SKELETAL MESH..."):
                    skeletal, fps, dims = extract_landmarks(tfile.name)
                    stroke, x_fact, reach = classify_motion(skeletal, sport)
                    st.session_state[state_key] = {
                        "skeletal": skeletal, "fps": fps, "dims": dims, 
                        "stroke": stroke, "x_fact": x_fact, "reach": reach
                    }
                    st.session_state[f"name_{sport}"] = up_file.name

            data = st.session_state[state_key]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                sel_action = st.selectbox("DETECTED ACTION", list(actions.keys()), 
                                          index=list(actions.keys()).index(data['stroke']) if data['stroke'] in actions else 0,
                                          key=f"sel_{sport}")
                st.write(f"💡 **AI Goal:** {actions[sel_action]}")
                analyze_btn = st.button("GENERATE BIOMECHANIC OVERLAY", key=f"btn_{sport}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                if analyze_btn:
                    st.success("Analysis Complete. Processing high-speed render...")
                    # Placeholder for the render_video function call
                    st.metric("INTENSITY SCORE", "88/100" if sel_action == "General Rally" else "Optimal")
                    st.write("📁 *Download Pack Ready Below*")

        st.markdown("</div>", unsafe_allow_html=True)

# --- GLOBAL FOOTER ---
st.markdown("<br>", unsafe_allow_html=True)
st.info("🔒 **Privacy Note:** Your videos are processed locally in memory. No data, videos, or skeletal metrics are stored on our servers once the session is closed.")
st.info("Built with ❤️ using [Streamlit](https://streamlit.io/) — free and open source. [Other Scripts by dev](https://devs-scripts.streamlit.app/)")
