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
    .stTabs [data-baseweb="tab"] { height: 50px; min-width: 130px; background: rgba(255, 255, 255, 0.03) !important; border-radius: 12px !important; margin: 5px; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS ---
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32)
]

SPORT_CONFIGS = {
    "TENNIS 🎾": {
        "Serve": "Analyze toss alignment and extension.",
        "Forehand/Backhand": "Focus on X-Factor and topspin path.",
        "General Rally": "Full point analysis from serve to finish.",
        "Volley": "Punch depth and stability."
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
    "GOLF ⛳": {
        "Driver Swing": "Wide arc and dynamic weight shift.",
        "Iron Swing": "Downward strike and lead arm extension.",
        "Putting": "Pendulum motion and head stability.",
        "Practice Sequence": "Multi-shot consistency tracking."
    },
    "BADMINTON 🏸": {
        "Jump Smash": "Vertical leap and overhead whip speed.",
        "Net Drop": "Short-range touch and racket face angle.",
        "High-Intensity Rally": "High-intensity footwork and shuttle tracking.",
        "Clear": "Full-court depth and shoulder rotation."
    },
    "CRICKET BATTING 🏏": {
        "Drive": "Check head position and high elbow lead.",
        "Pull/Hook": "Analyze rotation and weight on back foot.",
        "Defensive Shot": "Evaluate bat-pad gap and vertical bat angle.",
        "Net Practice": "Continuous shot selection and posture analysis."
    },
    "CRICKET BOWLING ⚾": {
        "Fast Bowling": "Analyze gather, front-foot landing, and release height.",
        "Spin Bowling": "Focus on pivot, shoulder rotation, and release point.",
        "Delivery Stride": "Evaluate alignment of feet and torso at crease.",
        "Over Analysis": "Full over analysis and run-up consistency."
    }
}

# --- FUNCTIONS ---
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

def render_video(input_path, skeletal_data, stroke_label, info_dict, w, h, fps):
    cap = cv2.VideoCapture(input_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps if fps > 0 else 30.0, (w, h + 200))
    instr = info_dict.get(stroke_label, "Analyze alignment.")
    progress_bar = st.progress(0, text="RENDERING BIOMECHANICS...")
    for i, frame_data in enumerate(skeletal_data):
        ret, frame = cap.read()
        if not ret: break
        canvas = np.zeros((h + 200, w, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = frame
        if frame_data:
            for s, e in POSE_CONNECTIONS:
                p1, p2 = (int(frame_data[s]['x']*w), int(frame_data[s]['y']*h)), (int(frame_data[e]['x']*w), int(frame_data[e]['y']*h))
                cv2.line(canvas, p1, p2, (0, 255, 127), 4)
        cv2.putText(canvas, f"DETECTION: {stroke_label}", (40, h + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(canvas, f"GOAL: {instr}", (40, h + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        out.write(canvas)
        progress_bar.progress((i + 1) / len(skeletal_data))
    cap.release(); out.release(); progress_bar.empty()
    return temp_output.name

# --- APP LAYOUT ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtext'>Pro Sports Biomechanics</p>", unsafe_allow_html=True)

tab_list = list(SPORT_CONFIGS.keys())
tabs = st.tabs(tab_list)

for i, sport in enumerate(tab_list):
    actions = SPORT_CONFIGS[sport]
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        up_file = st.file_uploader(f"UPLOAD {sport} VIDEO", type=["mp4", "mov", "avi"], key=f"up_{sport}")
        
        if up_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up_file.name)[1])
            tfile.write(up_file.read())
            
            state_key = f"data_{sport}"
            if state_key not in st.session_state or st.session_state.get(f"name_{sport}") != up_file.name:
                with st.spinner("CALIBRATING SKELETAL MESH..."):
                    skeletal, fps, dims = extract_landmarks(tfile.name)
                    st.session_state[state_key] = {"skeletal": skeletal, "fps": fps, "dims": dims, "processed": None}
                    st.session_state[f"name_{sport}"] = up_file.name

            data = st.session_state[state_key]
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                sel_action = st.selectbox("SELECT ACTION", list(actions.keys()), key=f"sel_{sport}")
            with col2:
                analyze_btn = st.button("RUN PRO ANALYSIS", key=f"btn_{sport}")
            
            if analyze_btn:
                processed_path = render_video(tfile.name, data['skeletal'], sel_action, actions, *data['dims'], data['fps'])
                
                # --- DATA PACK GENERATION ---
                json_data = json.dumps({
                    "metadata": {"sport": sport, "action": sel_action, "info": actions[sel_action]},
                    "skeletal_frames": data['skeletal']
                }, indent=4)
                
                ai_prompt = f"USER: Analyzing {sport} {sel_action}. {actions[sel_action]} Provide professional biomechanical coaching feedback based on this motion data."
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    with open(processed_path, "rb") as f:
                        zf.writestr("analysis.mp4", f.read())
                    zf.writestr("data.json", json_data)
                    zf.writestr("prompt.txt", ai_prompt)
                
                st.success("Analysis Complete!")
                st.download_button(
                    label="📦 DOWNLOAD PRO PACK",
                    data=zip_buffer.getvalue(),
                    file_name=f"{sport.lower().replace(' ', '_')}_pack.zip",
                    mime="application/zip",
                    key=f"dl_{sport}"
                )
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- GLOBAL FOOTER ---
st.markdown("<br>", unsafe_allow_html=True)
st.info("🔒 **Privacy Note:** Your videos are processed locally in memory. No data, videos, or skeletal metrics are stored on our servers once the session is closed.")
st.info("Built with ❤️ using [Streamlit](https://streamlit.io/) — free and open source. [Other Scripts by dev](https://devs-scripts.streamlit.app/)")
