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

# --- 1. PAGE CONFIG & CSS ---
st.set_page_config(page_title="Not Coach Nikki | Pro Analytics", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #020617; color: #f8fafc; }
    div[data-testid="stSlider"] label p { color: #ccff00 !important; font-weight: 900 !important; font-size: 1.2rem !important; }
    div[data-testid="stThumbValue"] { color: #000 !important; background-color: #ccff00 !important; }
    .glass-card { background: rgba(255, 255, 255, 0.05); border-radius: 20px; padding: 20px; border: 1px solid rgba(255,255,255,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SKELETAL DEFINITIONS ---
FULL_CONNECTIONS = [(0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10), (11,12), (11,13), (13,15), (12,14), (14,16), (11,23), (12,24), (23,24), (23,25), (25,27), (24,26), (26,28)]

def download_model():
    model_path = 'pose_landmarker_heavy.task'
    if not os.path.exists(model_path):
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", model_path)
    return model_path

def analyze_video_data(video_path, model_path):
    detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path), running_mode=vision.RunningMode.VIDEO))
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    history, raw_coords = [], []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        timestamp = int((len(history) * 1000) / fps)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(mp_img, timestamp)
        
        lms = res.pose_landmarks[0] if res.pose_landmarks else None
        history.append(lms)
        
        # Extract Full Raw XYZ Data for JSON
        if lms:
            frame_data = [{"id": j, "x": lm.x, "y": lm.y, "z": lm.z, "v": lm.visibility} for j, lm in enumerate(lms)]
            raw_coords.append(frame_data)
        else:
            raw_coords.append(None)
            
    cap.release()
    return {"history": history, "raw": raw_coords, "fps": fps, "total": len(history)}

def render_stereo_production(p1, p2, h1, h2, f1, f2, fps):
    cap1, cap2 = cv2.VideoCapture(p1), cv2.VideoCapture(p2)
    offset, target_h = f1 - f2, 720
    w1 = int(cap1.get(3) * (target_h / cap1.get(4)))
    w2 = int(cap2.get(3) * (target_h / cap2.get(4)))
    
    temp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_raw.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w1 + w2, target_h))

    for i in range(len(h1)):
        ret1, frame1 = cap1.read()
        if not ret1: break
        idx2 = i - offset
        
        if 0 <= idx2 < len(h2):
            cap2.set(cv2.CAP_PROP_POS_FRAMES, idx2)
            ret2, frame2 = cap2.read()
            lm2 = h2[idx2]
        else:
            frame2, lm2 = np.zeros((720, w2, 3), dtype=np.uint8), None

        # Draw Skels
        for f, lms in [(frame1, h1[i]), (frame2, lm2)]:
            if lms:
                for s, e in FULL_CONNECTIONS:
                    p1_pt = (int(lms[s].x * f.shape[1]), int(lms[s].y * f.shape[0]))
                    p2_pt = (int(lms[e].x * f.shape[1]), int(lms[e].y * f.shape[0]))
                    cv2.line(f, p1_pt, p2_pt, (204, 255, 0), 3)

        res1, res2 = cv2.resize(frame1, (w1, target_h)), cv2.resize(frame2, (w2, target_h))
        out.write(np.hstack((res1, res2)))
        
    cap1.release(); cap2.release(); out.release()
    return temp_raw.name

# --- 3. UI ---
st.title("NOT COACH NIKKI | STEREOGRAPHIC")
sport = st.selectbox("Sport", ["Tennis", "Padel", "Golf"])
c1, c2 = st.columns([1, 2])

with c1:
    u1 = st.file_uploader("Lead Angle", type=["mp4", "mov"])
    u2 = st.file_uploader("Side Angle", type=["mp4", "mov"])
    run = st.button("PROCESS ANGLES")

if run and u1 and u2:
    model = download_model()
    t1, t2 = tempfile.NamedTemporaryFile(delete=False), tempfile.NamedTemporaryFile(delete=False)
    t1.write(u1.getbuffer()); t2.write(u2.getbuffer())
    
    with st.status("High-Density Analysis...") as status:
        st.session_state.res1 = analyze_video_data(t1.name, model)
        st.session_state.res2 = analyze_video_data(t2.name, model)
        st.session_state.paths = (t1.name, t2.name)

if "res1" in st.session_state:
    s1, s2 = st.session_state.res1, st.session_state.res2
    p1, p2 = st.session_state.paths
    
    sl1 = st.slider("Lead Sync", 0, s1['total']-1, s1['total']//2)
    sl2 = st.slider("Side Sync", 0, s2['total']-1, s2['total']//2)
    
    if st.button("🎬 GENERATE PRO PACK"):
        with st.spinner("Stitching & Encoding..."):
            final_path = render_stereo_production(p1, p2, s1['history'], s2['history'], sl1, sl2, s1['fps'])
            
            # THE PREVIEW FIX: Use Streamlit-safe encoding
            st.success("Final Render Preview:")
            st.video(final_path)
            
            # THE HIGH-DENSITY JSON
            telemetry = {
                "sync_offset": sl1 - sl2,
                "fps": s1['fps'],
                "data_lead": s1['raw'],
                "data_side": s2['raw']
            }
            
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                zf.write(final_path, "synced_video.mp4")
                zf.writestr("telemetry_full_xyz.json", json.dumps(telemetry))
                zf.writestr("coach_brief.txt", f"Sport: {sport}. Offset: {sl1-sl2}")
            
            st.download_button("📥 DOWNLOAD COMPLETE DATASET", zip_buf.getvalue(), "pro_analytics_pack.zip")
