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

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Not Coach Nikki | Pro Analytics",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PREMIUM CSS STYLING (Mobile Grid Optimized) ---
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
        filter: drop-shadow(0px 4px 10px rgba(0,0,0,0.3));
    }
    
    .hero-subtext { 
        text-align: center; 
        color: #94a3b8; 
        font-size: 0.75rem; 
        margin-bottom: 2rem; 
        text-transform: uppercase; 
        letter-spacing: 2px; 
    }

    /* --- MOBILE 2-COLUMN GRID SYSTEM --- */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 10px; 
        display: flex; 
        flex-wrap: wrap; 
        justify-content: center;
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"] { 
        height: 55px; 
        /* This creates the 2-column layout on mobile (accounting for gap) */
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

    /* Desktop View: Keep them more compact */
    @media (min-width: 1024px) {
        .stTabs [data-baseweb="tab"] {
            flex: 0 1 180px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATIONS ---
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]

SPORT_CONFIGS = {
    "TENNIS 🎾": {
        "Serve": "Analyze toss height, trophy position, and kinetic chain.",
        "Forehand Drive": "Focus on unit turn, X-Factor rotation, and topspin follow-through.",
        "Backhand Drive": "Evaluate shoulder turn and contact point out front.",
        "Forehand Slice": "Analyze high-to-low path.",
        "Backhand Slice": "Evaluate the knife-like motion.",
        "Volley": "Punch depth and head stability."
    },
    "PADEL 🎾": {
        "Bandeja": "High-contact control and side-step timing.",
        "Vibora": "Aggressive slice and core rotation.",
        "General Rally": "Wall-play transitions.",
        "Underhand Serve": "Waist-height contact."
    },
    "PICKLEBALL 🥒": {
        "Dink": "Soft touch and minimal wrist hinge.",
        "Kitchen Volley": "Hand speed and reset stability.",
        "Third Shot Drop": "Arc height and landing depth."
    },
    "GOLF ⛳": {
        "Driver Swing": "Wide arc, spine angle, and dynamic weight shift.",
        "Iron Swing": "Downward strike and lead arm extension.",
        "Putting": "Pendulum motion and head stability."
    },
    "BADMINTON 🏸": {
        "Jump Smash": "Vertical leap and overhead whip speed.",
        "Net Drop": "Short-range touch and racket face angle.",
        "Clear": "Full-court depth."
    },
    "CRICKET BATTING 🏏": {
        "Drive": "Check head position, high elbow lead, and front foot stride.",
        "Pull/Hook": "Analyze rotation and back-foot weight transfer.",
        "Defensive Shot": "Evaluate bat-pad gap and bat angle."
    },
    "CRICKET BOWLING ⚾": {
        "Fast Bowling": "Analyze gather, release height, and follow-through.",
        "Spin Bowling": "Focus on pivot and release point consistency.",
        "Delivery Stride": "Evaluate alignment of feet and torso."
    },
    "GYM 🏋️": {
        "Bodyweight Squat": "Analyze depth and heel contact.",
        "Walking Lunges": "Check knee alignment and stride length.",
        "Push-Ups": "Evaluate core bracing and elbow flare.",
        "Deadlift": "Focus on lumbar neutrality and hip hinge.",
        "Pull-Ups": "Analyze range of motion and scapular retraction."
    },
    "YOGA 🧘": {
        "Mountain Pose (Tadasana)": "Analyze vertical alignment and weight distribution.",
        "Downward Dog (Adho Mukha Svanasana)": "Focus on spine length and hip elevation.",
        "Tree Pose (Vrikshasana)": "Evaluate unilateral balance stability.",
        "Warrior 2 (Virabhadrasana II)": "Check hip opening and arm alignment.",
        "Crow Pose (Bakasana)": "Focus on center of gravity and arm balance."
    }
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
                
                # HYPER-DETAILED AI PROMPT
                detailed_prompt = f"""
ACT AS AN ELITE PERFORMANCE COACH AND BIOMECHANIST. 
SESSION: {sport.upper()} - {sel_action.upper()}
TECHNICAL GOAL: {actions[sel_action]}

Using the attached JSON skeletal coordinates, provide:
1. THE GOOD: Identify foundational biomechanical strengths.
2. THE ERROR: Pinpoint the exact joint or timing breakdown using the frame data.
3. THE FIX: Provide one clinical cue or drill for immediate improvement.
"""
                json_data = json.dumps({"metadata": {"sport": sport, "action": sel_action}, "skeletal_frames": data['skeletal']}, indent=4)
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    with open(processed_path, "rb") as f: zf.writestr("analysis.mp4", f.read())
                    zf.writestr("data.json", json_data)
                    zf.writestr("prompt.txt", detailed_prompt)
                
                st.success("Analysis Complete!")
                st.download_button(label="📦 DOWNLOAD PRO PACK", data=zip_buffer.getvalue(), file_name=f"{sport.lower().replace(' ', '_')}_pack.zip", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
