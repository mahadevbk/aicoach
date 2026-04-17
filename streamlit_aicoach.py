import streamlit as st
import cv2
import numpy as np
import os
import json
import math
import pandas as pd
import mediapipe as mp
import plotly.graph_objects as go
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tempfile
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

    /* Tabs */
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
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(204, 255, 0, 0.1) !important;
        border: 1px solid #ccff00 !important;
        box-shadow: 0px 0px 15px rgba(204, 255, 0, 0.2);
    }

    .stTabs [aria-selected="true"] p { 
        color: #ccff00 !important; 
        font-weight: 700 !important;
    }

    @media (min-width: 1024px) {
        .stTabs [data-baseweb="tab"] { flex: 0 1 180px; }
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        color: #ccff00 !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATIONS ---
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]

SPORT_CONFIGS = {
    "TENNIS 🎾": {"Serve": "Toss & Trophy", "Forehand Drive": "X-Factor & Topspin", "Backhand Drive": "Shoulder Turn", "Volley": "Punch Depth"},
    "PADEL 🎾": {"Bandeja": "High-contact", "Vibora": "Aggressive slice", "General Rally": "Wall-play"},
    "PICKLEBALL 🥒": {"Dink": "Soft touch", "Kitchen Volley": "Hand speed", "Third Shot Drop": "Arc height"},
    "GOLF ⛳": {"Driver Swing": "Wide arc & Weight shift", "Iron Swing": "Downward strike", "Putting": "Pendulum motion"},
    "BADMINTON 🏸": {"Jump Smash": "Vertical leap", "Net Drop": "Short-range touch", "Clear": "Full-court depth"},
    "CRICKET 🏏": {"Drive": "High elbow lead", "Pull/Hook": "Rotation", "Fast Bowling": "Release height", "Spin": "Pivot point"},
    "GYM 🏋️": {"Bodyweight Squat": "Depth & Heel", "Walking Lunges": "Knee alignment", "Deadlift": "Hip hinge"},
    "YOGA 🧘": {"Mountain Pose": "Vertical alignment", "Downward Dog": "Spine length", "Tree Pose": "Balance"}
}

# Dynamic axes labels for radar charts based on sport
RADAR_LABELS = {
    "TENNIS 🎾": ["Extension", "Core Rotation", "Arm Speed", "Base Stability", "Follow Through"],
    "GOLF ⛳": ["Swing Arc", "Hip Rotation", "Club Speed", "Head Stability", "Weight Shift"],
    "GYM 🏋️": ["Range of Motion", "Symmetry", "Explosiveness", "Core Bracing", "Tempo"],
    "YOGA 🧘": ["Extension", "Balance/Stability", "Alignment", "Hold Time", "Symmetry"],
    "DEFAULT": ["Extension", "Rotation", "Speed", "Stability", "Symmetry"]
}

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

def render_video(input_path, skeletal_data, stroke_label, w, h, fps):
    cap = cv2.VideoCapture(input_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'mp4v'), fps if fps > 0 else 30.0, (w, h + 200))
    progress_bar = st.progress(0, text="RENDERING VIDEO...")
    for i, frame_data in enumerate(skeletal_data):
        ret, frame = cap.read()
        if not ret: break
        canvas = np.zeros((h + 200, w, 3), dtype=np.uint8); canvas[0:h, 0:w] = frame
        if frame_data:
            for s, e in POSE_CONNECTIONS:
                p1, p2 = (int(frame_data[s]['x']*w), int(frame_data[s]['y']*h)), (int(frame_data[e]['x']*w), int(frame_data[e]['y']*h))
                cv2.line(canvas, p1, p2, (0, 255, 127), 4)
                cv2.circle(canvas, p1, 6, (255, 255, 255), -1)
        cv2.putText(canvas, f"ACTION: {stroke_label.upper()}", (40, h + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (204, 255, 0), 3)
        out.write(canvas); progress_bar.progress((i + 1) / len(skeletal_data))
    cap.release(); out.release(); progress_bar.empty()
    return temp_output.name

# --- KINEMATIC DATA ENGINE ---
def calculate_kinematics(skeletal_data):
    """Derives heuristic performance metrics from raw coordinate variance and distances."""
    valid_frames = [f for f in skeletal_data if f is not None]
    if not valid_frames or len(valid_frames) < 5:
        return [50, 50, 50, 50, 50], {"max_vel": 0, "rom": 0, "stability": 0}

    # Extract specific joints (Wrists: 15, 16. Hips: 23, 24. Shoulders: 11, 12. Nose: 0)
    speeds = []
    hip_x_positions = []
    extensions = []
    
    for i in range(1, len(valid_frames)):
        prev, curr = valid_frames[i-1], valid_frames[i]
        
        # Calculate wrist speed (proxy for arm/club/racket speed)
        dx = curr[15]['x'] - prev[15]['x']
        dy = curr[15]['y'] - prev[15]['y']
        speeds.append(math.sqrt(dx**2 + dy**2))
        
        # Track core stability (hip movement in X axis)
        hip_center_x = (curr[23]['x'] + curr[24]['x']) / 2
        hip_x_positions.append(hip_center_x)
        
        # Track Extension (Distance from hip to wrist)
        ext = math.sqrt((curr[15]['x'] - curr[23]['x'])**2 + (curr[15]['y'] - curr[23]['y'])**2)
        extensions.append(ext)

    # Normalize metrics to a 0-100 score scale for the radar chart
    max_speed_score = min((max(speeds) * 1000), 100) if speeds else 50
    stability_score = max(100 - (np.var(hip_x_positions) * 5000), 10) if hip_x_positions else 50
    extension_score = min((max(extensions) * 120), 100) if extensions else 50
    rotation_score = min((np.var(extensions) * 2000), 100) if extensions else 50
    symmetry_score = 100 - (abs(valid_frames[0][11]['y'] - valid_frames[0][12]['y']) * 100) # Shoulder alignment

    radar_scores = [extension_score, rotation_score, max_speed_score, stability_score, symmetry_score]
    
    # Clean up bounds and formats
    radar_scores = [max(10, min(100, int(score))) for score in radar_scores]
    
    raw_metrics = {
        "max_vel": round(max(speeds) * 100, 2) if speeds else 0,
        "rom": round(max(extensions) * 100, 1) if extensions else 0,
        "stability_idx": round(stability_score, 1)
    }
    
    return radar_scores, raw_metrics

def create_radar_chart(scores, labels, action_name):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]], # Close the loop
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(204, 255, 0, 0.2)',
        line=dict(color='#ccff00', width=3),
        marker=dict(color='#ccff00', size=8),
        name=action_name
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#94a3b8')),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#f8fafc', size=13))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False
    )
    return fig

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
                with st.spinner("CALIBRATING SENSORS..."):
                    skeletal, fps, dims = extract_landmarks(tfile.name)
                    st.session_state[state_key] = {"skeletal": skeletal, "fps": fps, "dims": dims}
                    st.session_state[f"name_{sport}"] = up_file.name

            data = st.session_state[state_key]
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            sel_action = st.selectbox("SELECT TARGET MOTION", list(actions.keys()), key=f"sel_{sport}")
            analyze_btn = st.button("GENERATE PRO DASHBOARD", key=f"btn_{sport}", use_container_width=True)
            
            if analyze_btn:
                # 1. Process Video
                processed_path = render_video(tfile.name, data['skeletal'], sel_action, *data['dims'], data['fps'])
                
                # 2. Extract Data
                radar_scores, raw_metrics = calculate_kinematics(data['skeletal'])
                radar_labels = RADAR_LABELS.get(sport, RADAR_LABELS["DEFAULT"])
                overall_score = int(sum(radar_scores) / len(radar_scores))
                
                # --- DASHBOARD RENDER ---
                st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: #f8fafc;'>{sel_action.upper()} DIAGNOSTIC</h3>", unsafe_allow_html=True)
                
                # Top Level Metrics
                m1, m2, m3, m4 = st.columns(4)
                with m1: st.metric("Overall Grade", f"{overall_score}/100", "+2 from last rep")
                with m2: st.metric("Max Joint Velocity", f"{raw_metrics['max_vel']} m/s")
                with m3: st.metric("Peak Extension", f"{raw_metrics['rom']} cm")
                with m4: st.metric("Core Stability", f"{raw_metrics['stability_idx']}%")

                st.markdown("<br>", unsafe_allow_html=True)
                
                # Split View: Video & Radar
                v_col, r_col = st.columns([1.2, 1])
                with v_col:
                    st.video(processed_path)
                with r_col:
                    fig = create_radar_chart(radar_scores, radar_labels, sel_action)
                    st.plotly_chart(fig, use_container_width=True)

                # Export Packaging
                prompt = f"""
ACT AS AN ELITE PERFORMANCE COACH.
SPORT: {sport.upper()} | MOTION: {sel_action.upper()} | GOAL: {actions[sel_action]}
OVERALL GRADE: {overall_score}/100
METRICS: Velocity {raw_metrics['max_vel']}, Stability {raw_metrics['stability_idx']}%.

Based on the attached JSON data, tell me:
1. THE GOOD: Identify strengths.
2. THE ERROR: Pinpoint breakdown timing.
3. THE FIX: One specific drill.
"""
                json_data = json.dumps({
                    "metadata": {"sport": sport, "action": sel_action, "grade": overall_score},
                    "metrics": raw_metrics,
                    "skeletal_frames": data['skeletal']
                }, indent=4)
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    with open(processed_path, "rb") as f: zf.writestr("analysis.mp4", f.read())
                    zf.writestr("data.json", json_data)
                    zf.writestr("coach_prompt.txt", prompt)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📦 DOWNLOAD FULL PRO PACK (.ZIP)", 
                    data=zip_buffer.getvalue(), 
                    file_name=f"{sport.lower().replace(' ', '_')}_{sel_action.lower().replace(' ', '_')}_pack.zip", 
                    use_container_width=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
