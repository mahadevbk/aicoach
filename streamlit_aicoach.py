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

# --- 1. PAGE CONFIG & AGGRESSIVE CSS FIXES ---
st.set_page_config(page_title="Not Coach Nikki | Pro Analytics", page_icon="🎾", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    
    /* SLIDER COLOR FIXES: Forcing High Contrast Neon/White */
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 800 !important; font-size: 1.2rem !important; text-shadow: 1px 1px 2px black; }
    div[data-testid="stThumbValue"] { color: #000000 !important; background-color: #ccff00 !important; font-weight: 900 !important; }
    div[data-baseweb="slider"] > div { background-image: linear-gradient(to right, #ccff00, #38bdf8) !important; }
    
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 1.5rem; margin-bottom: 2rem; }
    h1 { font-size: clamp(1.8rem, 7vw, 3.5rem) !important; font-weight: 800 !important; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; filter: drop-shadow(2px 2px 0.5px rgba(204, 255, 0, 0.9)); }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; display: flex; flex-wrap: wrap; justify-content: center; background-color: transparent !important; }
    .stTabs [data-baseweb="tab"] { height: 55px; flex: 1 1 calc(50% - 10px); min-width: 140px; background: rgba(255, 255, 255, 0.05) !important; border-radius: 14px !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; transition: all 0.3s ease; }
    .stTabs [aria-selected="true"] { background: rgba(204, 255, 0, 0.1) !important; border: 1px solid #ccff00 !important; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE STEREO SKELETAL ENGINE ---
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]

def download_model():
    model_path = 'pose_landmarker_heavy.task'
    if not os.path.exists(model_path):
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", model_path)
    return model_path

def analyze_video_data(video_path, model_path):
    detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path), running_mode=vision.RunningMode.VIDEO))
    cap = cv2.VideoCapture(video_path)
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    history, impact_frame, peak_vel, prev_w = [], 0, 0, None
    
    for i in range(total):
        ret, frame = cap.read()
        if not ret: break
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(mp_img, int((i * 1000) / fps))
        lms = res.pose_landmarks[0] if res.pose_landmarks else None
        if res.pose_world_landmarks:
            w = res.pose_world_landmarks[0][15]
            if prev_w:
                vel = np.linalg.norm(np.array([w.x, w.y, w.z]) - np.array([prev_w.x, prev_w.y, prev_w.z]))
                if vel > peak_vel: peak_vel, impact_frame = vel, i
            prev_w = w
        history.append(lms)
    cap.release()
    return {"history": history, "impact": impact_frame, "fps": fps, "total": total}

def render_stereo_production(p1, p2, h1, h2, f1, f2, fps):
    cap1, cap2 = cv2.VideoCapture(p1), cv2.VideoCapture(p2)
    offset, target_h = f1 - f2, 720
    # Calculate widths to maintain aspect ratio at 720p height
    w1 = int(cap1.get(3) * (target_h / cap1.get(4)))
    w2 = int(cap2.get(3) * (target_h / cap2.get(4)))
    
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_out.name, cv2.VideoWriter_fourcc(*'avc1'), fps, (w1 + w2, target_h))

    for i in range(len(h1)):
        ret1, frame1 = cap1.read()
        if not ret1: break
        idx2 = i - offset
        if 0 <= idx2 < len(h2):
            cap2.set(cv2.CAP_PROP_POS_FRAMES, idx2)
            ret2, frame2 = cap2.read()
            lm2 = h2[idx2]
        else:
            frame2, lm2 = np.zeros((int(cap2.get(4)), int(cap2.get(3)), 3), dtype=np.uint8), None

        # Draw Skeletons on both frames
        for f, lms, w, h in [(frame1, h1[i], w1, target_h), (frame2, lm2, w2, target_h)]:
            if lms:
                for s, e in POSE_CONNECTIONS:
                    p1_pt = (int(lms[s].x * f.shape[1]), int(lms[s].y * f.shape[0]))
                    p2_pt = (int(lms[e].x * f.shape[1]), int(lms[e].y * f.shape[0]))
                    cv2.line(f, p1_pt, p2_pt, (127, 255, 0), 3)

        res1, res2 = cv2.resize(frame1, (w1, target_h)), cv2.resize(frame2, (w2, target_h))
        out.write(np.hstack((res1, res2)))
        
    cap1.release(); cap2.release(); out.release()
    return temp_out.name

# --- 3. UI LAYOUT (RESTORING ALL 9+ TABS) ---
SPORT_CONFIGS = {
    "TENNIS 🎾": {"Serve": "Toss height", "Forehand": "Unit turn"},
    "PADEL 🎾": {"Bandeja": "Contact control"},
    "PICKLEBALL 🥒": {"Dink": "Soft touch"},
    "GOLF ⛳": {"Driver": "Wide arc"},
    "CRICKET 🏏": {"Drive": "High elbow"},
    "GYM 🏋️": {"Squat": "Depth"},
    "YOGA 🧘": {"Warrior": "Balance"}
}

st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
tabs = st.tabs(list(SPORT_CONFIGS.keys()))

for i, (sport, actions) in enumerate(SPORT_CONFIGS.items()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.info(f"PRO {sport} ENGINE ACTIVE")
            is_stereo = st.toggle("Stereographic Mode", key=f"t_{sport}", value=True)
            u1 = st.file_uploader("Lead Angle", type=["mp4", "mov"], key=f"u1_{sport}")
            u2 = st.file_uploader("Side Angle", type=["mp4", "mov"], key=f"u2_{sport}") if is_stereo else None
            sel_act = st.selectbox("ACTION", list(actions.keys()), key=f"s_{sport}")
            run_btn = st.button("RUN AI ENGINE", key=f"b_{sport}", use_container_width=True)

        with c2:
            if run_btn and u1:
                model = download_model()
                t1_path = os.path.join(tempfile.gettempdir(), f"l_{sport}.mp4")
                with open(t1_path, "wb") as f: f.write(u1.getbuffer())
                with st.status("Analyzing...") as status:
                    res1 = analyze_video_data(t1_path, model)
                    res2, t2_path = None, None
                    if is_stereo and u2:
                        t2_path = os.path.join(tempfile.gettempdir(), f"s_{sport}.mp4")
                        with open(t2_path, "wb") as f: f.write(u2.getbuffer())
                        res2 = analyze_video_data(t2_path, model)
                        st.session_state[f"sync_{sport}"] = {"d1": res1, "d2": res2, "p1": t1_path, "p2": t2_path}
                    else: st.video(t1_path)

            if is_stereo and f"sync_{sport}" in st.session_state:
                s = st.session_state[f"sync_{sport}"]
                st.markdown("### 🛠️ SYNC VERIFICATION")
                sl1 = st.slider("Lead Impact Frame", 0, s['d1']['total']-1, s['d1']['impact'], key=f"sl1_{sport}")
                sl2 = st.slider("Side Impact Frame", 0, s['d2']['total']-1, s['d2']['impact'], key=f"sl2_{sport}")
                
                cap1, cap2 = cv2.VideoCapture(s['p1']), cv2.VideoCapture(s['p2'])
                cap1.set(1, sl1); cap2.set(1, sl2)
                _, i1 = cap1.read(); _, i2 = cap2.read()
                cap1.release(); cap2.release()
                if i1 is not None and i2 is not None:
                    st.image(np.hstack((cv2.resize(cv2.cvtColor(i1, cv2.COLOR_BGR2RGB), (640, 480)), 
                                        cv2.resize(cv2.cvtColor(i2, cv2.COLOR_BGR2RGB), (640, 480)))))
                
                if st.button("🎬 GENERATE PRODUCTION PACK", key=f"g_{sport}", use_container_width=True):
                    final = render_stereo_production(s['p1'], s['p2'], s['d1']['history'], s['d2']['history'], sl1, sl2, s['d1']['fps'])
                    st.success("Production Ready!")
                    st.video(final)
                    
                    # --- DOWNLOAD PACK: VIDEO + JSON + LLM BRIEF ---
                    brief = f"ACT AS ELITE COACH. SPORT: {sport}. ACTION: {sel_act}. GOAL: {actions[sel_act]}. ANALYZE SYNCED OFFSET OF {sl1-sl2} FRAMES."
                    telemetry = {"offset": sl1-sl2, "lead_impact": sl1, "side_impact": sl2, "sport": sport}
                    
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        zf.write(final, "analysis.mp4")
                        zf.writestr("coach_brief.txt", brief)
                        zf.writestr("telemetry.json", json.dumps(telemetry))
                    st.download_button("📥 DOWNLOAD PRO PACK", zip_buf.getvalue(), f"{sport}_analysis.zip", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
