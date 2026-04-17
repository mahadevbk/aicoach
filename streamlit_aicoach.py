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
import pandas as pd  # Added for data visualization

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
    
    .stApp { 
        background: radial-gradient(circle at top right, #1e293b, #020617); 
        color: #f8fafc; 
        font-family: 'Roboto Flex', sans-serif; 
    }
    
    .glass-card { 
        background: rgba(255, 255, 255, 0.03); 
        backdrop-filter: blur(12px); 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        border-radius: 24px; 
        padding: 1.5rem; 
        margin-bottom: 2rem; 
    }
    
    h1 { 
        font-size: clamp(1.8rem, 7vw, 3.5rem) !important; 
        font-weight: 800 !important; 
        background: linear-gradient(to right, #38bdf8, #818cf8); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        text-align: center; 
        margin-bottom: 0px !important; 
        filter: drop-shadow(2px 2px 0.5px rgba(204, 255, 0, 0.9));
    }
    
    .hero-subtext { 
        text-align: center; 
        color: #94a3b8; 
        font-size: 0.75rem; 
        margin-bottom: 2rem; 
        text-transform: uppercase; 
        letter-spacing: 2px; 
    }

    .stTabs [data-baseweb="tab-list"] { 
        gap: 10px; 
        display: flex; 
        flex-wrap: wrap; 
        justify-content: center;
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"] { 
        height: 55px; 
        flex: 1 1 calc(50% - 10px); 
        min-width: 140px;
        background: rgba(255, 255, 255, 0.05) !important; 
        border-radius: 14px !important; 
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        margin: 0px !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(204, 255, 0, 0.1) !important;
        border: 1px solid #ccff00 !important;
    }

    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 700 !important; }

    @media (min-width: 1024px) { .stTabs [data-baseweb="tab"] { flex: 0 1 180px; } }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATIONS ---
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]

SPORT_CONFIGS = {
    "TENNIS 🎾": {"Serve": "Toss height and kinetic chain.", "Forehand Drive": "Unit turn and X-Factor.", "Backhand Drive": "Shoulder turn.", "Forehand Slice": "High-to-low path.", "Backhand Slice": "Knife motion.", "Volley": "Head stability."},
    "PADEL 🎾": {"Bandeja": "Contact control.", "Vibora": "Core rotation.", "General Rally": "Wall-play.", "Underhand Serve": "Waist height."},
    "PICKLEBALL 🥒": {"Dink": "Soft touch.", "Kitchen Volley": "Hand speed.", "Third Shot Drop": "Arc height."},
    "GOLF ⛳": {"Driver Swing": "Wide arc and weight shift.", "Iron Swing": "Lead arm extension.", "Putting": "Pendulum motion."},
    "BADMINTON 🏸": {"Jump Smash": "Overhead whip.", "Net Drop": "Racket face angle.", "Clear": "Court depth."},
    "CRICKET BATTING 🏏": {"Drive": "High elbow lead.", "Pull/Hook": "Weight transfer.", "Defensive Shot": "Bat-pad gap."},
    "CRICKET BOWLING ⚾": {"Fast Bowling": "Release height.", "Spin Bowling": "Pivot consistency.", "Delivery Stride": "Alignment."},
    "GYM 🏋️": {"Bodyweight Squat": "Depth.", "Walking Lunges": "Knee alignment.", "Push-Ups": "Elbow flare.", "Deadlift": "Hip hinge.", "Pull-Ups": "Scapular retraction."},
    "YOGA 🧘": {"Mountain Pose": "Vertical alignment.", "Downward Dog": "Spine length.", "Tree Pose": "Balance stability.", "Warrior 2": "Hip opening.", "Crow Pose": "Center of gravity."}
}

# --- BIOMECHANICAL ANALYTICS ENGINE ---
def calculate_angle(a, b, c):
    """Computes the angle (degrees) between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1 = a - b
    v2 = c - b
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_sport_metrics(skeletal_frames, sport_type):
    """Extracts relevant biometric data based on the sport."""
    metrics = []
    for frame in skeletal_frames:
        if not frame:
            metrics.append({})
            continue
        
        # Helper to get x,y for a specific landmark index
        gp = lambda i: [frame[i]['x'], frame[i]['y']]
        
        # Base metrics for all sports
        m = {
            "Trunk Tilt": calculate_angle(gp(11), gp(23), [frame[23]['x'], 0]), # Angle against vertical
            "Knee Flexion": calculate_angle(gp(23), gp(25), gp(27))
        }
        
        # Sport Specific Focus
        if any(s in sport_type for s in ["TENNIS", "PADEL", "GOLF", "CRICKET"]):
            m["Lead Arm Extension"] = calculate_angle(gp(11), gp(13), gp(15))
        if "GYM" in sport_type or "YOGA" in sport_type:
            m["Hip Hinge"] = calculate_angle(gp(11), gp(23), gp(25))
            m["Ankle Stability"] = calculate_angle(gp(25), gp(27), gp(31))
            
        metrics.append(m)
    return pd.DataFrame(metrics).interpolate().fillna(method='bfill').fillna(0)

# --- CORE FUNCTIONS ---
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
        timestamp_ms += (1000 / (fps if fps > 0 else 30))
        progress_bar.progress((i + 1) / total_frames)
    cap.release(); progress_bar.empty()
    return skeletal_series, fps, (w, h)

def render_video(input_path, skeletal_data, stroke_label, info_dict, w, h, fps):
    cap = cv2.VideoCapture(input_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'mp4v'), fps if fps > 0 else 30.0, (w, h + 200))
    progress_bar = st.progress(0, text="RENDERING...")
    for i, frame_data in enumerate(skeletal_data):
        ret, frame = cap.read()
        if not ret: break
        canvas = np.zeros((h + 200, w, 3), dtype=np.uint8); canvas[0:h, 0:w] = frame
        if frame_data:
            for s, e in POSE_CONNECTIONS:
                p1, p2 = (int(frame_data[s]['x']*w), int(frame_data[s]['y']*h)), (int(frame_data[e]['x']*w), int(frame_data[e]['y']*h))
                cv2.line(canvas, p1, p2, (0, 255, 127), 4)
        cv2.putText(canvas, f"DETECTION: {stroke_label}", (40, h + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(canvas); progress_bar.progress((i + 1) / len(skeletal_data))
    cap.release(); out.release(); progress_bar.empty()
    return temp_output.name

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
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up_file.name)[1])
            tfile.write(up_file.read())
            
            state_key = f"data_{sport}"
            if state_key not in st.session_state or st.session_state.get(f"name_{sport}") != up_file.name:
                with st.spinner("CALIBRATING..."):
                    skeletal, fps, dims = extract_landmarks(tfile.name)
                    st.session_state[state_key] = {"skeletal": skeletal, "fps": fps, "dims": dims}
                    st.session_state[f"name_{sport}"] = up_file.name

            data = st.session_state[state_key]
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            sel_action = st.selectbox("SELECT ACTION", list(actions.keys()), key=f"sel_{sport}")
            analyze_btn = st.button("RUN PRO ANALYSIS", key=f"btn_{sport}", use_container_width=True)
            
            if analyze_btn:
                processed_path = render_video(tfile.name, data['skeletal'], sel_action, actions, *data['dims'], data['fps'])
                
                json_data = json.dumps({"metadata": {"sport": sport, "action": sel_action}, "skeletal_frames": data['skeletal']}, indent=4)
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    with open(processed_path, "rb") as f: zf.writestr("analysis.mp4", f.read())
                    zf.writestr("data.json", json_data)
                
                st.success("Analysis Complete!")
                st.download_button(label="📦 DOWNLOAD PRO PACK", data=zip_buffer.getvalue(), file_name=f"{sport.lower().replace(' ', '_')}_pack.zip", use_container_width=True)
                
                # --- NEW DATA VISUALIZATION SECTION ---
                st.divider()
                st.subheader("📊 Performance Telemetry")
                df = get_sport_metrics(data['skeletal'], sport)
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.line_chart(df, height=300)
                with c2:
                    st.write("**Range of Motion (Deg)**")
                    st.dataframe(df.describe().loc[['min', 'max', 'mean']], use_container_width=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
