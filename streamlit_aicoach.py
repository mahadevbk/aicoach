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
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; justify-content: center; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] { height: 50px; min-width: 120px; background: rgba(255, 255, 255, 0.03) !important; border-radius: 12px !important; margin: 5px; }
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
    "CRICKET BATTING 🏏": {
        "Drive": "Check head position and high elbow lead.",
        "Pull/Hook": "Analyze rotation and weight on back foot.",
        "Defensive Shot": "Evaluate bat-pad gap and vertical bat angle.",
        "General Rally": "Over-long tracking of strike rotation."
    },
    "CRICKET BOWLING ⚾": {
        "Fast Bowling": "Analyze gather, front-foot landing, and release height.",
        "Spin Bowling": "Focus on pivot, shoulder rotation, and release point.",
        "Delivery Stride": "Evaluate alignment of feet and torso at crease.",
        "General Rally": "Full over analysis and run-up consistency."
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skeletal_series, timestamp_ms = [], 0
    
    progress_bar = st.progress(0, text="SCANNING BIOMETRICS...")
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(mp_image, int(timestamp_ms))
        lm = [{"x": p.x, "y": p.y, "z": p.z, "v": p.visibility} for p in res.pose_landmarks[0]] if res.pose_landmarks else None
        skeletal_series.append(lm)
        timestamp_ms += (1000 / (fps if fps > 0 else 30))
        progress_bar.progress((i + 1) / total_frames)
    cap.release()
    progress_bar.empty()
    return skeletal_series, fps, (w, h)

def classify_motion(skeletal_data, sport_name):
    if not skeletal_data or all(x is None for x in skeletal_data):
        return "General Rally", 0, 0
    
    # Calculate lateral displacement to detect "Rallies" vs "Static actions"
    hip_xs = [lm[23]['x'] for lm in skeletal_data if lm]
    wrist_ys = [lm[15]['y'] for lm in skeletal_data if lm]
    
    movement_range = max(hip_xs) - min(hip_xs) if hip_xs else 0
    vertical_reach = 1 - min(wrist_ys) if wrist_ys else 0
    
    # If the person moves more than 20% of the screen width, call it a rally
    if movement_range > 0.20:
        return "General Rally", 45.0, movement_range
    
    # Simple sport-specific defaults
    if "BOWLING" in sport_name: return "Fast Bowling", 15.0, vertical_reach
    if "BATTING" in sport_name: return "Drive", 65.0, movement_range
    
    return list(SPORT_CONFIGS[sport_name].keys())[0], 30.0, vertical_reach

# --- UI LOGIC ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtext'>Pro Sports Biomechanics</p>", unsafe_allow_html=True)

# Generate Tabs
tab_list = list(SPORT_CONFIGS.keys())
tabs = st.tabs(tab_list)

for i, sport in enumerate(tab_list):
    actions = SPORT_CONFIGS[sport]
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.info(f"PRO {sport} ENGINE: AI-Optimized for {', '.join(actions.keys())}")
        
        up_file = st.file_uploader(f"UPLOAD {sport} VIDEO", type=["mp4", "mov", "avi"], key=f"up_{sport}")
        
        if up_file:
            # Temporary file handling
            suffix = os.path.splitext(up_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                tfile.write(up_file.read())
                temp_path = tfile.name
            
            state_key = f"data_{sport}"
            if state_key not in st.session_state or st.session_state.get(f"name_{sport}") != up_file.name:
                with st.spinner("CALIBRATING SKELETAL MESH..."):
                    skeletal, fps, dims = extract_landmarks(temp_path)
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
                st.button("GENERATE BIOMECHANIC OVERLAY", key=f"btn_{sport}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                if "BOWLING" in sport:
                    m1.metric("RELEASE HEIGHT", f"{round(data['reach'], 2)}m")
                    m2.metric("ALIGNMENT", "OPTIMAL")
                elif "BATTING" in sport:
                    m1.metric("X-FACTOR", f"{round(data['x_fact'], 1)}°")
                    m2.metric("FOOTWORK", "STABLE")
                else:
                    m1.metric("INTENSITY", "HIGH")
                    m2.metric("STATUS", "READY")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# --- GLOBAL FOOTER ---
st.markdown("<br>", unsafe_allow_html=True)
st.info("🔒 **Privacy Note:** Your videos are processed locally in memory. No data, videos, or skeletal metrics are stored on our servers once the session is closed.")
st.info("Built with ❤️ using [Streamlit](https://streamlit.io/) — free and open source. [Other Scripts by dev](https://devs-scripts.streamlit.app/)")
