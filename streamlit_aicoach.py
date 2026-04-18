import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tempfile
import urllib.request
import json
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Coach Nikki | Pro AI", layout="wide")

# --- CSS STYLING (Preserving your original look) ---
st.markdown("""
    <style>
    .stApp { background: #020617; color: #f8fafc; }
    [data-testid="stMetricValue"] { color: #ccff00 !important; font-weight: 800; }
    .glass-card { background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 20px; border: 1px solid rgba(255,255,255,0.1); }
    .highlight { color: #ccff00; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- ENGINE UTILITIES ---
def get_model():
    """Downloads the heavy model on the fly to bypass GitHub 25MB limit."""
    model_path = "pose_landmarker_heavy.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        with st.spinner("🚀 Initializing Heavy AI Engine..."):
            urllib.request.urlretrieve(url, model_path)
    return model_path

def analyze_impact(video_path, model_path):
    """The pipeline_stereo engine for impact detection."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO)
    
    impact_frame, peak_vel = 0, 0
    prev_wrist = None
    
    with vision.PoseLandmarker.create_from_options(options) as detector:
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = detector.detect_for_video(mp_image, int((i * 1000) / fps))
            
            if res.pose_world_landmarks:
                # Track wrist velocity (Index 15/16)
                wrist = res.pose_world_landmarks[0][15]
                if prev_wrist:
                    vel = np.linalg.norm(np.array([wrist.x, wrist.y, wrist.z]) - np.array([prev_wrist.x, prev_wrist.y, prev_wrist.z]))
                    if vel > peak_vel:
                        peak_vel, impact_frame = vel, i
                prev_wrist = wrist
    cap.release()
    return impact_frame, total_frames, fps

# --- UI HEADER ---
st.markdown("## 🎾 AI COACH | <span class='highlight'>PRO ANALYTICS</span>", unsafe_allow_html=True)

# --- MODE TOGGLE ---
stereo_mode = st.toggle("Enable Stereographic Mode (Dual Camera Sync)", value=False)

if not stereo_mode:
    # --- SINGLE FILE MODE ---
    uploaded_file = st.file_uploader("Upload Action Video", type=["mp4", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        st.video(tfile.name)
        st.info("Single video mode: Normal analysis pipeline active.")
        # [Add your existing single-video analysis calls here]

else:
    # --- STEREOGRAPHIC MODE ---
    col1, col2 = st.columns(2)
    v1 = col1.file_uploader("Lead Angle (Camera A)", type=["mp4", "mov"])
    v2 = col2.file_uploader("Side Angle (Camera B)", type=["mp4", "mov"])

    if v1 and v2:
        # Save to temp
        t1_path = os.path.join(tempfile.gettempdir(), "lead.mp4")
        t2_path = os.path.join(tempfile.gettempdir(), "side.mp4")
        with open(t1_path, "wb") as f: f.write(v1.getbuffer())
        with open(t2_path, "wb") as f: f.write(v2.getbuffer())

        if 'sync_data' not in st.session_state:
            if st.button("🔍 DETECT IMPACT & SYNC"):
                model = get_model()
                i1, tot1, fps1 = analyze_impact(t1_path, model)
                i2, tot2, fps2 = analyze_impact(t2_path, model)
                st.session_state.sync_data = {'i1': i1, 'i2': i2, 't1': tot1, 't2': tot2, 'fps': fps1}
                st.rerun()

        if 'sync_data' in st.session_state:
            st.markdown("### 🛠️ VERIFY & FINE-TUNE SYNC")
            sd = st.session_state.sync_data
            
            # Scrubber UI
            c1, c2 = st.columns(2)
            f1 = c1.slider("Lead Impact Frame", 0, sd['t1'], sd['i1'])
            f2 = c2.slider("Side Impact Frame", 0, sd['t2'], sd['i2'])
            
            # Frame Preview (Verification Logic)
            cap1 = cv2.VideoCapture(t1_path)
            cap2 = cv2.VideoCapture(t2_path)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, f1)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, f2)
            _, img1 = cap1.read()
            _, img2 = cap2.read()
            
            if img1 is not None and img2 is not None:
                # Show 20% of frame area centered on athlete or just full resized side-by-side
                preview = np.hstack((cv2.resize(img1, (640, 360)), cv2.resize(img2, (640, 360))))
                st.image(preview, caption="Sync Preview: Adjust sliders until both racket impacts match visually.")

            if st.button("🎬 GENERATE FINAL STEREO PACK"):
                # Final Offset Calculation
                offset = f1 - f2
                st.success(f"Finalizing with {offset} frame offset...")
                
                # Here you would trigger the pipeline_stereo.py render logic
                # For brevity, creating a placeholder JSON result
                result_json = {
                    "lead_impact": f1,
                    "side_impact": f2,
                    "sync_offset": offset,
                    "engine": "Pose Landmarker Heavy"
                }
                
                st.download_button(
                    label="📥 Download JSON Analytics",
                    data=json.dumps(result_json),
                    file_name="stereo_metadata.json",
                    mime="application/json"
                )
