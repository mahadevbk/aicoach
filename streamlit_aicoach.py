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
import urllib.request
import subprocess
import time

# --- 1. UI & CONFIG (UNCHANGED) ---
st.set_page_config(page_title="Not Coach Nikki | Pro Analytics", page_icon="🎾", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; font-size: 1.1rem !important; text-transform: uppercase; }
    div[data-testid="stThumbValue"] { color: #000 !important; background-color: #ccff00 !important; font-weight: 900 !important; }
    div[data-baseweb="slider"] > div { background: #ccff00 !important; }
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 2rem; margin-bottom: 2rem; }
    h1 { font-size: clamp(2rem, 8vw, 4rem) !important; font-weight: 900 !important; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .hero-sub { text-align: center; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 4px; margin-bottom: 3rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; justify-content: center; }
    .stTabs [data-baseweb="tab"] { height: 60px; background: rgba(255, 255, 255, 0.05) !important; border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.1) !important; }
    .stTabs [aria-selected="true"] { background: rgba(204, 255, 0, 0.1) !important; border: 1px solid #ccff00 !important; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. OPTIMIZATION LOGIC ---
SPORT_LANDMARKS = {
    "TENNIS 🎾": [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
    "PADEL 🎾": [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
    "PICKLEBALL 🥒": [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
    "GOLF ⛳": [11, 12, 13, 14, 15, 16, 23, 24, 25, 26],
    "CRICKET 🏏": [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
    "GYM 🏋️": [11, 12, 23, 24, 25, 26, 27, 28],
    "YOGA 🧘": list(range(33)) # Keep all for balance/alignment
}

def get_optimised_data(raw_frames, sport):
    if not raw_frames: return None
    indices = SPORT_LANDMARKS.get(sport, list(range(33)))
    optimised = []
    for frame in raw_frames:
        if frame is None:
            optimised.append(None)
            continue
        # Only keep indices for the specific sport and round to 4 decimals
        optimised.append([
            {
                "id": j, 
                "x": round(frame[j]['x'], 4), 
                "y": round(frame[j]['y'], 4), 
                "z": round(frame[j]['z'], 4)
            } for j in indices
        ])
    return optimised

# --- 3. THE CORE ENGINE (RESTORED FROM YOUR WORKING FILE) ---
FULL_SKELETON = [(0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10), (11,12), (11,13), (13,15), (12,14), (14,16), (11,23), (12,24), (23,24), (23,25), (25,27), (24,26), (26,28)]

def download_model():
    p = 'pose_landmarker_heavy.task'
    if not os.path.exists(p):
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", p)
    return p

def analyze_full_data(path, model):
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.VIDEO))
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    history, raw, impact_f, peak_v, prev_w = [], [], 0, 0, None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        ts = int((len(history) * 1000) / (fps if fps > 0 else 30))
        m_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = det.detect_for_video(m_img, ts)
        lms = res.pose_landmarks[0] if res.pose_landmarks else None
        history.append(lms)
        if lms:
            raw.append([{"id": j, "x": l.x, "y": l.y, "z": l.z} for j, l in enumerate(lms)])
            if res.pose_world_landmarks:
                w = res.pose_world_landmarks[0][15]
                if prev_w:
                    v = np.linalg.norm(np.array([w.x, w.y, w.z]) - np.array([prev_w.x, prev_w.y, prev_w.z]))
                    if v > peak_v: peak_v, impact_f = v, len(history)-1
                prev_w = w
        else: raw.append(None)
    cap.release()
    return {"history": history, "raw": raw, "fps": fps, "total": len(history), "impact": impact_f}

def render_pro_stereo(p1, p2, h1, h2, f1, f2, fps):
    cap1, cap2 = cv2.VideoCapture(p1), cv2.VideoCapture(p2)
    off, target_h = f1 - f2, 720
    w1 = int(cap1.get(3) * (target_h / cap1.get(4)))
    w2 = int(cap2.get(3) * (target_h / cap2.get(4)))
    combined_width = (w1 + w2) // 2 * 2
    raw_path = os.path.join(tempfile.gettempdir(), f"raw_{int(time.time())}.mp4")
    final_path = os.path.join(tempfile.gettempdir(), f"prod_{int(time.time())}.mp4")
    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w1+w2, target_h))
    for i in range(len(h1)):
        ret1, f1_img = cap1.read()
        if not ret1: break
        idx2 = i - off
        if 0 <= idx2 < len(h2):
            cap2.set(1, idx2); _, f2_img = cap2.read()
            lm2 = h2[idx2]
        else: f2_img, lm2 = np.zeros((720, w2, 3), dtype=np.uint8), None
        for img, lms in [(f1_img, h1[i]), (f2_img, lm2)]:
            if lms:
                for s, e in FULL_SKELETON:
                    cv2.line(img, (int(lms[s].x*img.shape[1]), int(lms[s].y*img.shape[0])), (int(lms[e].x*img.shape[1]), int(lms[e].y*img.shape[0])), (127, 255, 0), 3)
        out.write(np.hstack((cv2.resize(f1_img, (w1, target_h)), cv2.resize(f2_img, (w2, target_h)))))
    cap1.release(); cap2.release(); out.release()
    subprocess.run(f'ffmpeg -y -i "{raw_path}" -c:v libx264 -pix_fmt yuv420p -preset ultrafast "{final_path}"', shell=True)
    return final_path

# --- 4. UI AND EXPORT LOGIC ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>Professional Biomechanics AI</p>", unsafe_allow_html=True)

SPORT_MAP = {
    "TENNIS 🎾": {"Serve": "Toss height"}, "PADEL 🎾": {"Bandeja": "Contact point"},
    "PICKLEBALL 🥒": {"Dink": "Pace control"}, "GOLF ⛳": {"Driver": "Swing arc"},
    "CRICKET 🏏": {"Drive": "Elbow position"}, "GYM 🏋️": {"Squat": "Depth"},
    "YOGA 🧘": {"Warrior": "Alignment"}
}

tabs = st.tabs(list(SPORT_MAP.keys()))

for i, (sport, actions) in enumerate(SPORT_MAP.items()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"AI ENGINE: {sport}")
            is_stereo = st.toggle("Stereographic Mode", key=f"is_st_{sport}", value=True)
            u1 = st.file_uploader("Lead Angle", type=["mp4","mov"], key=f"u1_{sport}")
            u2 = st.file_uploader("Side Angle", type=["mp4","mov"] if is_stereo else None, key=f"u2_{sport}")
            sel_act = st.selectbox("Action Type", list(actions.keys()), key=f"act_{sport}")
            run_btn = st.button("RUN PRO ANALYSIS", key=f"run_{sport}", use_container_width=True)

        with col2:
            res_key = f"data_{sport}"
            if run_btn and u1:
                model = download_model()
                t1_p = os.path.join(tempfile.gettempdir(), f"l_{sport}.mp4")
                with open(t1_p, "wb") as f: f.write(u1.getbuffer())
                with st.status("Analyzing...") as status:
                    d1 = analyze_full_data(t1_p, model)
                    d2, t2_p = None, None
                    if is_stereo and u2:
                        t2_p = os.path.join(tempfile.gettempdir(), f"s_{sport}.mp4")
                        with open(t2_p, "wb") as f: f.write(u2.getbuffer())
                        d2 = analyze_full_data(t2_p, model)
                    st.session_state[res_key] = {"d1": d1, "d2": d2, "p1": t1_p, "p2": t2_p}

            if res_key in st.session_state:
                s = st.session_state[res_key]
                st.markdown("### 🛠️ SYNC VERIFICATION")
                sl1 = st.slider("Lead Impact Frame", 0, s['d1']['total']-1, s['d1']['impact'], key=f"sl1_{sport}")
                sl2 = st.slider("Side Impact Frame", 0, (s['d2']['total']-1 if s['d2'] else 0), (s['d2']['impact'] if s['d2'] else 0), key=f"sl2_{sport}")
                
                # Preview frames
                c1 = cv2.VideoCapture(s['p1']); c1.set(1, sl1); _, i1 = c1.read(); c1.release()
                if is_stereo and s['p2']:
                    c2 = cv2.VideoCapture(s['p2']); c2.set(1, sl2); _, i2 = c2.read(); c2.release()
                    st.image(np.hstack((cv2.resize(cv2.cvtColor(i1, 4), (640, 480)), cv2.resize(cv2.cvtColor(i2, 4), (640, 480)))))
                
                if st.button("🎬 GENERATE PRODUCTION PACK", key=f"gen_{sport}", use_container_width=True):
                    final_v = render_pro_stereo(s['p1'], s['p2'], s['d1']['history'], s['d2']['history'], sl1, sl2, s['d1']['fps'])
                    st.video(final_v)
                    
                    # --- CREATING BOTH JSON VERSIONS ---
                    tele_heavy = {
                        "sport": sport, "type": "heavy_full_data",
                        "lead_xyz": s['d1']['raw'], "side_xyz": s['d2']['raw'] if s['d2'] else None
                    }
                    
                    tele_opt = {
                        "sport": sport, "type": "ai_optimized_report",
                        "action": sel_act, "fps": s['d1']['fps'],
                        "lead_xyz": get_optimised_data(s['d1']['raw'], sport),
                        "side_xyz": get_optimised_data(s['d2']['raw'], sport) if s['d2'] else None
                    }
                    
                    z_buf = io.BytesIO()
                    with zipfile.ZipFile(z_buf, "w") as zf:
                        zf.write(final_v, "analysis_video.mp4")
                        zf.writestr("telemetry_dense_FULL.json", json.dumps(tele_heavy))
                        zf.writestr("telemetry_OPTIMIZED.json", json.dumps(tele_opt))
                        zf.writestr("ai_prompt_hint.txt", f"Sport: {sport}. Focus: {actions[sel_act]}. Target: Report Generation.")
                    
                    st.download_button("📥 DOWNLOAD COMPARISON PACK", z_buf.getvalue(), f"{sport}_Report_Files.zip", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
