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

# --- 1. PREMIUM CSS & MOBILE OPTIMIZATION ---
st.set_page_config(page_title="Not Coach Nikki | Pro Analytics", page_icon="🎾", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    .stApp { background: radial-gradient(circle at top right, #1e293b, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    
    /* NEON MOBILE SLIDERS */
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; font-size: 1.2rem !important; text-transform: uppercase; }
    div[data-testid="stThumbValue"] { color: #000 !important; background-color: #ccff00 !important; font-weight: 900 !important; }
    div[data-baseweb="slider"] > div { background: #ccff00 !important; }

    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 1.5rem; margin-bottom: 2rem; }
    h1 { font-size: clamp(2rem, 7vw, 3.5rem) !important; font-weight: 800 !important; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HIGH-DENSITY TRACKING ENGINE ---
# Full 33-point landmark map including Head/Face
FULL_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), # Head
    (9,10), (11,12), (11,13), (13,15), (12,14), (14,16),   # Upper
    (11,23), (12,24), (23,24), (23,25), (25,27), (24,26), (26,28) # Lower
]

def get_mobile_ready_video(input_path):
    """Re-encodes video for WhatsApp/Mobile compatibility using FFmpeg."""
    output_path = input_path.replace(".mp4", "_mobile.mp4")
    try:
        # Forces H.264, YUV420p (required for mobile playback), and faststart for streaming
        cmd = f"ffmpeg -y -i {input_path} -c:v libx264 -pix_fmt yuv420p -preset fast -crf 23 {output_path}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        return output_path
    except:
        return input_path # Fallback to original if ffmpeg fails

def analyze_high_density(path, model):
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.VIDEO))
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    history, raw_xyz, impact_f, peak_v, prev_w = [], [], 0, 0, None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        ts = int((len(history) * 1000) / fps)
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        
        lms = res.pose_landmarks[0] if res.pose_landmarks else None
        history.append(lms)
        
        if lms:
            # Maximum Data Extraction for JSON
            raw_xyz.append([{"id": j, "x": l.x, "y": l.y, "z": l.z, "vis": l.visibility} for j, l in enumerate(lms)])
            # Impact Logic
            if res.pose_world_landmarks:
                w = res.pose_world_landmarks[0][15]
                if prev_w:
                    v = np.linalg.norm(np.array([w.x, w.y, w.z]) - np.array([prev_w.x, prev_w.y, prev_w.z]))
                    if v > peak_v: peak_v, impact_f = v, len(history)-1
                prev_w = w
        else: raw_xyz.append(None)
    cap.release()
    return {"history": history, "raw": raw_xyz, "fps": fps, "total": len(history), "impact": impact_f}

def render_pro_stereo(p1, p2, h1, h2, f1, f2, fps):
    cap1, cap2 = cv2.VideoCapture(p1), cv2.VideoCapture(p2)
    off, target_h = f1 - f2, 720
    w1 = int(cap1.get(3) * (target_h / cap1.get(4)))
    w2 = int(cap2.get(3) * (target_h / cap2.get(4)))
    
    temp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_raw.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w1+w2, target_h))

    for i in range(len(h1)):
        ret1, img1 = cap1.read()
        if not ret1: break
        idx2 = i - off
        if 0 <= idx2 < len(h2):
            cap2.set(1, idx2); _, img2 = cap2.read()
            lm2 = h2[idx2]
        else: img2, lm2 = np.zeros((720, w2, 3), dtype=np.uint8), None

        # SKELETAL OVERLAY (Including Head)
        for img, lms in [(img1, h1[i]), (img2, lm2)]:
            if lms:
                for s, e in FULL_CONNECTIONS:
                    cv2.line(img, (int(lms[s].x*img.shape[1]), int(lms[s].y*img.shape[0])), 
                             (int(lms[e].x*img.shape[1]), int(lms[e].y*img.shape[0])), (204, 255, 0), 3)
                for pt in lms: # Draw joint nodes
                    cv2.circle(img, (int(pt.x*img.shape[1]), int(pt.y*img.shape[0])), 3, (255,255,255), -1)

        out.write(np.hstack((cv2.resize(img1, (w1, target_h)), cv2.resize(img2, (w2, target_h)))))
    
    cap1.release(); cap2.release(); out.release()
    # CONVERT TO MOBILE COMPATIBLE FORMAT
    return get_mobile_ready_video(temp_raw.name)

# --- 3. UI TABS & WORKFLOW ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)

SPORT_CONFIGS = {
    "TENNIS 🎾": ["Serve", "Forehand", "Backhand"],
    "PADEL 🎾": ["Bandeja", "Vibora", "Smash"],
    "GOLF ⛳": ["Driver", "Iron"],
    "GYM 🏋️": ["Squat", "Deadlift"]
}

tabs = st.tabs(list(SPORT_CONFIGS.keys()))

for i, sport in enumerate(SPORT_CONFIGS.keys()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col_setup, col_viz = st.columns([1, 2])
        
        with col_setup:
            st.info(f"PRO ANALYTICS: {sport}")
            u1 = st.file_uploader("Lead Angle (Phone 1)", type=["mp4","mov"], key=f"u1_{sport}")
            u2 = st.file_uploader("Side Angle (Phone 2)", type=["mp4","mov"], key=f"u2_{sport}")
            sel_act = st.selectbox("Action Type", SPORT_CONFIGS[sport], key=f"act_{sport}")
            run_btn = st.button("PROCESS FOR MOBILE", key=f"run_{sport}", use_container_width=True)

        with col_viz:
            res_key = f"sync_data_{sport}"
            if run_btn and u1 and u2:
                # Pre-download check for model
                model_p = 'pose_landmarker_heavy.task'
                if not os.path.exists(model_p):
                    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", model_p)
                
                t1_p = os.path.join(tempfile.gettempdir(), f"l_{sport}.mp4")
                t2_p = os.path.join(tempfile.gettempdir(), f"s_{sport}.mp4")
                with open(t1_p, "wb") as f: f.write(u1.getbuffer())
                with open(t2_p, "wb") as f: f.write(u2.getbuffer())
                
                with st.status("Extracting High-Density Biometrics...") as status:
                    d1 = analyze_high_density(t1_p, model_p)
                    d2 = analyze_high_density(t2_p, model_p)
                    st.session_state[res_key] = {"d1": d1, "d2": d2, "p1": t1_p, "p2": t2_p}

            if res_key in st.session_state and st.session_state[res_key]:
                s = st.session_state[res_key]
                st.markdown("### 🛠️ SYNC & EXPORT")
                sl1 = st.slider("Lead Impact", 0, s['d1']['total']-1, s['d1']['impact'], key=f"sl1_{sport}")
                sl2 = st.slider("Side Impact", 0, s['d2']['total']-1, s['d2']['impact'], key=f"sl2_{sport}")
                
                if st.button("🎬 GENERATE WHATSAPP-READY PACK", key=f"gen_{sport}", use_container_width=True):
                    with st.spinner("Encoding for Mobile Playback..."):
                        final_v = render_pro_stereo(s['p1'], s['p2'], s['d1']['history'], s['d2']['history'], sl1, sl2, s['d1']['fps'])
                        st.video(final_v) # Preview will work now!
                        
                        # Pack maximum data into JSON
                        telemetry = {
                            "metadata": {"sport": sport, "action": sel_act, "offset_frames": int(sl1-sl2)},
                            "lead_angle_raw": s['d1']['raw'],
                            "side_angle_raw": s['d2']['raw']
                        }
                        
                        zip_buf = io.BytesIO()
                        with zipfile.ZipFile(zip_buf, "w") as zf:
                            zf.write(final_v, "WhatsApp_Analysis.mp4")
                            zf.writestr("biometrics_full_dump.json", json.dumps(telemetry))
                            zf.writestr("coach_brief.txt", f"PRO BRIEF: {sport} {sel_act}. Sync achieved at offset {sl1-sl2}.")
                        
                        st.download_button("📥 DOWNLOAD MOBILE PACK", zip_buf.getvalue(), f"{sport}_mobile_pack.zip", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
