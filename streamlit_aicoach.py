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
import subprocess  # Added for WhatsApp compatibility
import time        # Added for file management

# --- 1. FULL PREMIUM UI RESTORATION ---
st.set_page_config(
    page_title="Not Coach Nikki | Pro Analytics",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    
    /* FIX: High-Contrast Neon Sliders */
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; font-size: 1.1rem !important; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stThumbValue"] { color: #000 !important; background-color: #ccff00 !important; font-weight: 900 !important; }
    div[data-baseweb="slider"] > div { background: #ccff00 !important; }

    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 2rem; margin-bottom: 2rem; }
    h1 { font-size: clamp(2rem, 8vw, 4rem) !important; font-weight: 900 !important; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0px !important; }
    .hero-sub { text-align: center; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 4px; margin-bottom: 3rem; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 10px; justify-content: center; }
    .stTabs [data-baseweb="tab"] { height: 60px; background: rgba(255, 255, 255, 0.05) !important; border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.1) !important; padding: 0 25px !important; }
    .stTabs [aria-selected="true"] { background: rgba(204, 255, 0, 0.1) !important; border: 1px solid #ccff00 !important; }
    .stTabs [aria-selected="true"] p { color: #ccff00 !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE HIGH-DENSITY ENGINE ---
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
        ts = int((len(history) * 1000) / fps)
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

# FIXED: Now Uses FFmpeg for WhatsApp (H.264 + YUV420p)
def render_pro_stereo(p1, p2, h1, h2, f1, f2, fps):
    cap1, cap2 = cv2.VideoCapture(p1), cv2.VideoCapture(p2)
    off, target_h = f1 - f2, 720
    w1 = int(cap1.get(3) * (target_h / cap1.get(4)))
    w2 = int(cap2.get(3) * (target_h / cap2.get(4)))
    
    # Ensure total width is divisible by 2 for H.264
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
                    cv2.line(img, (int(lms[s].x*img.shape[1]), int(lms[s].y*img.shape[0])), 
                             (int(lms[e].x*img.shape[1]), int(lms[e].y*img.shape[0])), (127, 255, 0), 3)

        out.write(np.hstack((cv2.resize(f1_img, (w1, target_h)), cv2.resize(f2_img, (w2, target_h)))))
    
    cap1.release(); cap2.release(); out.release()
    
    # Convert for WhatsApp compatibility
    subprocess.run(f'ffmpeg -y -i "{raw_path}" -c:v libx264 -pix_fmt yuv420p -preset ultrafast "{final_path}"', shell=True)
    return final_path

# --- 3. UI LAYOUT ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>Professional Biomechanics AI</p>", unsafe_allow_html=True)

SPORT_MAP = {
    "TENNIS 🎾": {"Serve": "Toss height", "Forehand": "Unit turn", "Backhand": "Shoulder turn"},
    "PADEL 🎾": {"Bandeja": "Contact point", "Vibora": "Side spin"},
    "PICKLEBALL 🥒": {"Dink": "Pace control", "Volley": "Reset position"},
    "GOLF ⛳": {"Driver": "Swing arc", "Iron": "Compression"},
    "CRICKET 🏏": {"Drive": "Elbow position", "Bowling": "Release"},
    "GYM 🏋️": {"Squat": "Depth", "Deadlift": "Spine angle"},
    "YOGA 🧘": {"Warrior": "Alignment", "Balance": "Stability"}
}

tabs = st.tabs(list(SPORT_MAP.keys()))

for i, (sport, actions) in enumerate(SPORT_MAP.items()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.info(f"AI ENGINE: {sport}")
            is_stereo = st.toggle("Stereographic Mode", key=f"is_st_{sport}", value=True)
            u1 = st.file_uploader("Lead Angle (MP4/MOV)", type=["mp4","mov"], key=f"u1_{sport}")
            u2 = st.file_uploader("Side Angle (MP4/MOV)", type=["mp4","mov"], key=f"u2_{sport}") if is_stereo else None
            sel_act = st.selectbox("Action Type", list(actions.keys()), key=f"act_{sport}")
            run_btn = st.button("RUN PRO ANALYSIS", key=f"run_{sport}", use_container_width=True)

        with col2:
            res_key = f"data_{sport}"
            if run_btn and u1:
                model = download_model()
                t1_p = os.path.join(tempfile.gettempdir(), f"l_{sport}.mp4")
                with open(t1_p, "wb") as f: f.write(u1.getbuffer())
                
                with st.status("Performing High-Density Analysis...") as status:
                    d1 = analyze_full_data(t1_p, model)
                    d2, t2_p = None, None
                    if is_stereo and u2:
                        t2_p = os.path.join(tempfile.gettempdir(), f"s_{sport}.mp4")
                        with open(t2_p, "wb") as f: f.write(u2.getbuffer())
                        d2 = analyze_full_data(t2_p, model)
                    st.session_state[res_key] = {"d1": d1, "d2": d2, "p1": t1_p, "p2": t2_p}

            if is_stereo and res_key in st.session_state and st.session_state[res_key] is not None:
                s = st.session_state[res_key]
                st.markdown("### 🛠️ SYNC VERIFICATION")
                sl1 = st.slider("Lead Impact Frame", 0, s['d1']['total']-1, s['d1']['impact'], key=f"sl1_{sport}")
                sl2 = st.slider("Side Impact Frame", 0, s['d2']['total']-1, s['d2']['impact'], key=f"sl2_{sport}")
                
                # Preview sync frames
                c1, c2 = cv2.VideoCapture(s['p1']), cv2.VideoCapture(s['p2'])
                c1.set(1, sl1); c2.set(1, sl2)
                _, i1 = c1.read(); _, i2 = c2.read()
                c1.release(); c2.release()
                if i1 is not None and i2 is not None:
                    st.image(np.hstack((cv2.resize(cv2.cvtColor(i1, 4), (640, 480)), cv2.resize(cv2.cvtColor(i2, 4), (640, 480)))))
                
                if st.button("🎬 GENERATE PRODUCTION PACK", key=f"gen_{sport}", use_container_width=True):
                    final_v = render_pro_stereo(s['p1'], s['p2'], s['d1']['history'], s['d2']['history'], sl1, sl2, s['d1']['fps'])
                    st.success("Analysis Optimized. Finalizing Pack...")
                    st.video(final_v)
                    
                    # PRO PACK EXPORT
                    telemetry = {
                        "sport": sport, "action": sel_act, "offset": int(sl1-sl2),
                        "lead_xyz": s['d1']['raw'], "side_xyz": s['d2']['raw']
                    }
                    
                    z_buf = io.BytesIO()
                    with zipfile.ZipFile(z_buf, "w") as zf:
                        zf.write(final_v, "full_skeletal_analysis.mp4")
                        zf.writestr("telemetry_dense.json", json.dumps(telemetry))
                        zf.writestr("ai_coach_brief.txt", f"Sport: {sport}. Goal: {actions[sel_act]}. Sync Offset: {sl1-sl2}")
                    
                    st.download_button("📥 DOWNLOAD PRO PACK", z_buf.getvalue(), f"{sport}_pro_pack.zip", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
