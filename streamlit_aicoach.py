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

    /* Main Background and Text */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #020617);
        color: #f8fafc;
        font-family: 'Roboto Flex', sans-serif;
    }

    /* Top Bar Gradient */
    .stHeader {
        background: linear-gradient(90deg, #38bdf8 0%, #6366f1 100%) !important;
        height: 3rem !important;
    }
    .stHeader * {
        color: white !important;
    }

    /* Glassmorphic Card Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }

    /* Typography */
    h1 {
        font-size: clamp(2rem, 8vw, 3.5rem) !important;
        font-weight: 800 !important;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px !important;
        filter: drop-shadow(4px 4px 12px rgba(204, 255, 0, 0.1)); /* DIFFUSED GRADIENT SHADOW */
    }
    
    .hero-subtext {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-bottom: 2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* STEP TRACKER - FORCED HORIZONTAL ROW */
    .process-wrapper {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        justify-content: space-around !important;
        align-items: flex-start !important;
        width: 100% !important;
        margin: 1rem 0 3rem 0 !important;
        gap: 5px !important;
    }

    .process-step {
        flex: 1 1 0px !important;
        text-align: center !important;
        min-width: 0 !important;
    }

    .process-icon-circle {
        width: 40px;
        height: 40px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 8px auto;
        font-size: 1.2rem;
    }

    .process-label {
        font-size: 0.65rem !important;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        display: block;
    }

    .process-desc {
        font-size: 0.55rem !important;
        color: #64748b;
        display: block;
        margin-top: 2px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        min-width: 140px;
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
    }

    .stTabs [data-baseweb="tab"] p {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #94a3b8 !important; /* MEDIUM GREY */
    }

    .stTabs [aria-selected="true"] p {
        color: #ccff00 !important; /* OPTIC YELLOW */
        text-shadow: 0 0 10px rgba(204, 255, 0, 0.2);
    }

    .stTabs [aria-selected="true"] {
        border: 1px solid #ccff00 !important;
        background: rgba(204, 255, 0, 0.05) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #38bdf8 0%, #6366f1 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        width: 100%;
        height: 50px;
    }

    /* Metrics */
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #38bdf8 !important; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS ---
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32)
]

TENNIS_INFO = {
    "Serve": "Analyze toss alignment, extension, and kinetic chain.",
    "Overhead Smash": "Evaluate balance, reach, and power transfer.",
    "Lob": "Analyze knee bend and follow-through verticality.",
    "Forehand Drive": "Focus on X-Factor rotation and topspin path.",
    "Backhand Drive": "Evaluate shoulder turn and weight transfer.",
    "Forehand Slice": "Analyze high-to-low path and carving motion.",
    "Backhand Slice": "Check core stability and high-to-low slice path.",
    "Volley": "Analyze punch depth and head stability.",
    "Unclassified Movement": "General biomechanical feedback."
}

GOLF_INFO = {
    "Driver Swing": "Wide arc, spine angle, and dynamic weight shift.",
    "Iron Swing": "Downward strike, lead arm extension, and hip clearance.",
    "Pitch / Chip": "Short game precision, wrist hinge, and center of gravity.",
    "Putting Stroke": "Pendulum motion, head stability, and consistent tempo.",
    "Unclassified Swing": "General postural and rotation analysis."
}

# --- FUNCTIONS ---
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
    progress_bar = st.progress(0, text="ANALYZING FRAMES")
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

def classify_motion(skeletal_data, mode="Tennis"):
    wrist_trajectory, max_x_factor, peak_y, spine_angles = [], 0, 1.0, []
    for lm in skeletal_data:
        if lm:
            active_wrist = lm[15] if lm[15]['v'] > lm[16]['v'] else lm[16]
            wrist_trajectory.append(active_wrist)
            peak_y = min(peak_y, active_wrist['y'])
            s_vec = np.array([lm[11]['x']-lm[12]['x'], lm[11]['z']-lm[12]['z']])
            h_vec = np.array([lm[23]['x']-lm[24]['x'], lm[23]['z']-lm[24]['z']])
            if np.linalg.norm(s_vec) > 0 and np.linalg.norm(h_vec) > 0:
                dot = np.dot(s_vec, h_vec) / (np.linalg.norm(s_vec) * np.linalg.norm(h_vec))
                max_x_factor = max(max_x_factor, np.degrees(np.arccos(np.clip(dot, -1, 1))))
            mid_s = np.array([(lm[11]['x']+lm[12]['x'])/2, (lm[11]['y']+lm[12]['y'])/2])
            mid_h = np.array([(lm[23]['x']+lm[24]['x'])/2, (lm[23]['y']+lm[24]['y'])/2])
            spine_angles.append(calculate_angle(mid_s, mid_h, mid_h + np.array([0, -1])))
    
    stroke = "Unclassified Movement"
    if wrist_trajectory:
        xs, ys = [p['x'] for p in wrist_trajectory], [p['y'] for p in wrist_trajectory]
        l_span, v_span = max(xs) - min(xs), max(ys) - min(ys)
        if mode == "Tennis":
            v_delta = ys[0] - ys[-1]
            max_up_vel = abs(min(np.diff(ys))) if len(ys) > 1 else 0
            if peak_y < 0.35 or max_up_vel > 0.04: stroke = "Serve" if l_span < 0.25 else "Overhead Smash"
            elif v_delta > 0.4: stroke = "Lob"
            elif l_span > 0.35: 
                base = "Forehand" if xs[-1] > xs[0] else "Backhand"
                stroke = f"{base} Slice" if ys[-1] > ys[0] + 0.15 else f"{base} Drive"
            else: stroke = "Volley"
        else:
            if v_span < 0.15: stroke = "Putting Stroke"
            elif v_span < 0.35: stroke = "Pitch / Chip"
            elif l_span > 0.45: stroke = "Driver Swing"
            else: stroke = "Iron Swing"
    return stroke, max_x_factor, peak_y, np.std(spine_angles) if spine_angles else 0

def render_video(input_path, skeletal_data, stroke_label, info_dict, w, h, fps):
    cap = cv2.VideoCapture(input_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h + 250))
    instr = info_dict.get(stroke_label, "General Analysis")
    progress_bar = st.progress(0, text="GENERATING REPORT")
    for i, frame_data in enumerate(skeletal_data):
        ret, frame = cap.read()
        if not ret: break
        canvas = np.zeros((h + 250, w, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = frame
        if frame_data:
            for s, e in POSE_CONNECTIONS:
                p1, p2 = (int(frame_data[s]['x']*w), int(frame_data[s]['y']*h)), (int(frame_data[e]['x']*w), int(frame_data[e]['y']*h))
                cv2.line(canvas, p1, p2, (0, 255, 127), 6)
        cv2.putText(canvas, f"MOTION: {stroke_label}", (50, h + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(canvas, f"GOAL: {instr}", (50, h + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        out.write(canvas)
        progress_bar.progress((i + 1) / len(skeletal_data))
    cap.release()
    out.release()
    progress_bar.empty()
    return temp_output.name

# --- APP LAYOUT ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtext'>Pro Sports Biomechanics</p>", unsafe_allow_html=True)

# FIXED ROW OF ICONS
st.markdown("""
    <div class="process-wrapper">
        <div class="process-step">
            <div class="process-icon-circle">📤</div>
            <span class="process-label">Upload</span>
            <span class="process-desc">Video</span>
        </div>
        <div class="process-step">
            <div class="process-icon-circle">⚙️</div>
            <span class="process-label">Detect</span>
            <span class="process-desc">Motion</span>
        </div>
        <div class="process-step">
            <div class="process-icon-circle">📐</div>
            <span class="process-label">Analyze</span>
            <span class="process-desc">Angles</span>
        </div>
        <div class="process-step">
            <div class="process-icon-circle">🚀</div>
            <span class="process-label">Ready</span>
            <span class="process-desc">Download</span>
        </div>
    </div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["TENNIS 🎾", "GOLF ⛳"])

with tab1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.info("PRO TENNIS ENGINE: Supports Serve, Smash, Lob, Drive, Slice, and Volleys.")
    up_t = st.file_uploader("UPLOAD TENNIS VIDEO", type=["mp4", "mov", "avi"], key="t_up")
    if up_t:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(up_t.read())
        if 't_skeletal' not in st.session_state or st.session_state.get('t_vid_name') != up_t.name:
            with st.spinner("SYNCING KINETIC DATA..."):
                skeletal, fps, dims = extract_landmarks(tfile.name)
                stroke, max_x, peak, spine_var = classify_motion(skeletal, "Tennis")
                st.session_state.update({'t_skeletal': skeletal, 't_fps': fps, 't_dims': dims, 't_stroke': stroke, 't_max_x': max_x, 't_peak': peak, 't_vid_name': up_t.name, 't_processed': None})
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("SETTINGS")
            st.warning("⚠️ Verify motion below.")
            sel_t = st.selectbox("STROKE", list(TENNIS_INFO.keys()), index=list(TENNIS_INFO.keys()).index(st.session_state['t_stroke']))
            if st.button("RUN ANALYSIS", key="t_btn"):
                st.session_state['t_processed'] = render_video(tfile.name, st.session_state['t_skeletal'], sel_t, TENNIS_INFO, *st.session_state['t_dims'], st.session_state['t_fps'])
                st.session_state['t_final_stroke'] = sel_t
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.get('t_processed'):
            with col2:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("RESULTS")
                m1, m2, m3 = st.columns(3)
                m1.metric("X-FACTOR", f"{round(st.session_state['t_max_x'], 1)}°")
                m2.metric("REACH", f"{round(1 - st.session_state['t_peak'], 2)}m")
                m3.metric("STATUS", "DONE")
                t_json = json.dumps({"metadata": {"stroke": st.session_state['t_final_stroke'], "max_x": st.session_state['t_max_x']}, "data": st.session_state['t_skeletal']}, indent=4)
                t_prompt = f"USER: Analyzing Tennis {st.session_state['t_final_stroke']}. {TENNIS_INFO[st.session_state['t_final_stroke']]}"
                zip_t = io.BytesIO()
                with zipfile.ZipFile(zip_t, "w") as zf:
                    with open(st.session_state['t_processed'], "rb") as f: zf.writestr("analysis.mp4", f.read())
                    zf.writestr("data.json", t_json)
                    zf.writestr("prompt.txt", t_prompt)
                st.download_button("📦 DOWNLOAD PACK", zip_t.getvalue(), "tennis_pack.zip", "application/zip")
                st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.info("PRO GOLF ENGINE: Supports Driver, Irons, Pitching, and Putting Analysis.")
    up_g = st.file_uploader("UPLOAD GOLF SWING", type=["mp4", "mov", "avi"], key="g_up")
    if up_g:
        gfile = tempfile.NamedTemporaryFile(delete=False)
        gfile.write(up_g.read())
        if 'g_skeletal' not in st.session_state or st.session_state.get('g_vid_name') != up_g.name:
            with st.spinner("SYNCING KINETIC DATA..."):
                skeletal, fps, dims = extract_landmarks(gfile.name)
                stroke, max_x, peak, spine_var = classify_motion(skeletal, "Golf")
                st.session_state.update({'g_skeletal': skeletal, 'g_fps': fps, 'g_dims': dims, 'g_stroke': stroke, 'g_max_x': max_x, 'g_spine_var': spine_var, 'g_vid_name': up_g.name, 'g_processed': None})
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("SETTINGS")
            st.warning("⚠️ Verify swing type.")
            sel_g = st.selectbox("SWING", list(GOLF_INFO.keys()), index=list(GOLF_INFO.keys()).index(st.session_state['g_stroke']))
            if st.button("RUN ANALYSIS", key="g_btn"):
                st.session_state['g_processed'] = render_video(gfile.name, st.session_state['g_skeletal'], sel_g, GOLF_INFO, *st.session_state['g_dims'], st.session_state['g_fps'])
                st.session_state['g_final_stroke'] = sel_g
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.get('g_processed'):
            with col2:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("RESULTS")
                m1, m2, m3 = st.columns(3)
                m1.metric("X-FACTOR", f"{round(st.session_state['g_max_x'], 1)}°")
                stability = round(100 - min(st.session_state['g_spine_var'] * 10, 100), 1)
                m2.metric("STABILITY", f"{stability}%")
                m3.metric("STATUS", "DONE")
                g_json = json.dumps({"metadata": {"swing": st.session_state['g_final_stroke'], "x_factor": st.session_state['g_max_x'], "stability": stability}, "data": st.session_state['g_skeletal']}, indent=4)
                g_prompt = f"USER: Analyzing Golf {st.session_state['g_final_stroke']}. Stability: {stability}%. {GOLF_INFO[st.session_state['g_final_stroke']]}"
                zip_g = io.BytesIO()
                with zipfile.ZipFile(zip_g, "w") as zf:
                    with open(st.session_state['g_processed'], "rb") as f: zf.writestr("analysis.mp4", f.read())
                    zf.writestr("data.json", g_json)
                    zf.writestr("prompt.txt", g_prompt)
                st.download_button("📦 DOWNLOAD PACK", zip_g.getvalue(), "golf_pack.zip", "application/zip")
                st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
