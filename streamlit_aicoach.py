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

# --- CONSTANTS & CONFIG ---
st.set_page_config(page_title="Not Coach Nikki", layout="wide", page_icon="🎾")

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32)
]

TENNIS_INFO = {
    "Serve": "Analyze the toss alignment, arm extension at contact, and leg explosion.",
    "Overhead Smash": "Analyze movement to the ball, core stability, and weight transfer.",
    "Lob": "Check if the lift comes from the knees. Evaluate the vertical follow-through angle.",
    "Forehand Slice": "Analyze the high-to-low carving motion and backspin control.",
    "Backhand Slice": "Analyze the high-to-low carving motion and backspin control.",
    "Forehand Drive": "Analyze the X-Factor coil release and the low-to-high topspin path.",
    "Backhand Drive": "Analyze the X-Factor coil release and the low-to-high topspin path.",
    "Volley": "Analyze the short punch motion. Check for head stability and keeping the ball in front.",
    "Unclassified Movement": "Provide general biomechanic feedback."
}

GOLF_INFO = {
    "Driver Swing": "Analyze the wide arc, spine angle maintenance, and weight shift through the ball.",
    "Iron Swing": "Focus on the downward strike, lead arm straightness at impact, and hip rotation.",
    "Pitch / Chip": "Analyze the quiet lower body, wrist hinge control, and consistent low point.",
    "Putting Stroke": "Check for head stability, pendulum motion from shoulders, and minimal wrist breakdown.",
    "Unclassified Swing": "Provide general golf biomechanics and posture feedback."
}

# --- HELPER FUNCTIONS ---

def calculate_angle(a, b, c):
    """Calculates the 2D angle between three points."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_landmarks(video_path):
    """Processes video with MediaPipe and returns skeletal data and metadata."""
    detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='pose_landmarker_lite.task'),
        running_mode=vision.RunningMode.VIDEO
    ))
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    full_skeletal_series = []
    frame_timestamp_ms = 0
    
    progress_bar = st.progress(0, text="Extracting skeletal data...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(mp_image, int(frame_timestamp_ms))
        
        frame_data = None
        if res.pose_landmarks:
            lm = [{"x": p.x, "y": p.y, "z": p.z, "v": p.visibility} for p in res.pose_landmarks[0]]
            frame_data = lm
        
        full_skeletal_series.append(frame_data)
        
        frame_idx += 1
        frame_timestamp_ms += (1000 / fps)
        progress_bar.progress(frame_idx / total_frames, text=f"Processing frame {frame_idx}/{total_frames}...")

    cap.release()
    progress_bar.empty()
    return full_skeletal_series, fps, (w, h)

def classify_motion(skeletal_data, mode="Tennis"):
    """Classifies the motion based on wrist trajectory and other metrics."""
    wrist_trajectory = []
    max_x_factor = 0
    peak_y = 1.0
    spine_angles = []
    
    for lm in skeletal_data:
        if lm:
            # Track active wrist for trajectory
            active_wrist = lm[15] if lm[15]['v'] > lm[16]['v'] else lm[16]
            wrist_trajectory.append(active_wrist)
            peak_y = min(peak_y, active_wrist['y'])
            
            # X-Factor
            s_vec = np.array([lm[11]['x']-lm[12]['x'], lm[11]['z']-lm[12]['z']])
            h_vec = np.array([lm[23]['x']-lm[24]['x'], lm[23]['z']-lm[24]['z']])
            if np.linalg.norm(s_vec) > 0 and np.linalg.norm(h_vec) > 0:
                dot = np.dot(s_vec, h_vec) / (np.linalg.norm(s_vec) * np.linalg.norm(h_vec))
                max_x_factor = max(max_x_factor, np.degrees(np.arccos(np.clip(dot, -1, 1))))
            
            # Spine Angle (Neck to Mid-Hip vs Vertical)
            mid_shoulder = np.array([(lm[11]['x']+lm[12]['x'])/2, (lm[11]['y']+lm[12]['y'])/2])
            mid_hip = np.array([(lm[23]['x']+lm[24]['x'])/2, (lm[23]['y']+lm[24]['y'])/2])
            spine_vec = mid_shoulder - mid_hip
            vertical_vec = np.array([0, -1])
            spine_angles.append(calculate_angle(mid_shoulder, mid_hip, mid_hip + vertical_vec))

    stroke = "Unclassified Movement"
    
    if wrist_trajectory:
        xs = [p['x'] for p in wrist_trajectory]
        ys = [p['y'] for p in wrist_trajectory]
        lateral_span = max(xs) - min(xs)
        vertical_span = max(ys) - min(ys)
        
        if mode == "Tennis":
            vertical_delta = ys[0] - ys[-1] 
            y_velocities = np.diff(ys)
            max_up_vel = abs(min(y_velocities)) if len(y_velocities) > 0 else 0

            if peak_y < 0.35 or max_up_vel > 0.04: 
                stroke = "Serve" if lateral_span < 0.25 else "Overhead Smash"
            elif vertical_delta > 0.4:
                stroke = "Lob"
            elif lateral_span > 0.35:
                base = "Forehand" if xs[-1] > xs[0] else "Backhand"
                stroke = f"{base} Slice" if ys[-1] > ys[0] + 0.15 else f"{base} Drive"
            else:
                stroke = "Volley"
        
        else: # Golf Mode
            if vertical_span < 0.15 and lateral_span > 0.15:
                stroke = "Putting Stroke"
            elif vertical_span < 0.35:
                stroke = "Pitch / Chip"
            elif lateral_span > 0.45:
                stroke = "Driver Swing"
            else:
                stroke = "Iron Swing"

    return stroke, max_x_factor, peak_y, np.std(spine_angles) if spine_angles else 0

def render_video(input_path, skeletal_data, stroke_label, info_dict, w, h, fps):
    """Generates a video with skeletal overlay and black bar."""
    cap = cv2.VideoCapture(input_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (w, h + 250))
    
    instructions = info_dict.get(stroke_label, "Provide general biomechanic feedback.")
    
    progress_bar = st.progress(0, text="Rendering video...")
    total_frames = len(skeletal_data)
    
    for i, frame_data in enumerate(skeletal_data):
        ret, frame = cap.read()
        if not ret: break
        
        canvas = np.zeros((h + 250, w, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = frame
        
        if frame_data:
            for s, e in POSE_CONNECTIONS:
                p1 = (int(frame_data[s]['x']*w), int(frame_data[s]['y']*h))
                p2 = (int(frame_data[e]['x']*w), int(frame_data[e]['y']*h))
                cv2.line(canvas, p1, p2, (0, 255, 127), 6) 
        
        cv2.putText(canvas, f"MOTION: {stroke_label}", (50, h + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(canvas, f"GOAL: {instructions}", (50, h + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        out.write(canvas)
        progress_bar.progress((i + 1) / total_frames, text=f"Rendering frame {i+1}/{total_frames}...")

    cap.release()
    out.release()
    progress_bar.empty()
    return temp_output.name

# --- APP UI ---

st.title("🎾 Not Coach Nikki")
st.markdown("""
### **How it works:**
1.  **Choose your Sport** (Tennis or Golf).
2.  **Upload** your swing video.
3.  **Confirm** the detected motion (Autodetection is in **Beta**).
4.  **Download** your ZIP package and use the **Prompt + JSON** in an AI for coaching!
""")

tab1, tab2 = st.tabs(["Tennis 🎾", "Golf ⛳"])

with tab1:
    st.header("Tennis Analysis")
    st.info("📊 **Supported Strokes:** Serve, Overhead Smash, Lob, Forehand/Backhand Drive, Forehand/Backhand Slice, and Volleys.")
    uploaded_tennis = st.file_uploader("Upload Tennis Video", type=["mp4", "mov", "avi"], key="tennis_up")
    
    if uploaded_tennis:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_tennis.read())
        
        if 't_skeletal' not in st.session_state or st.session_state.get('t_vid_name') != uploaded_tennis.name:
            with st.spinner("Analyzing Tennis Swing..."):
                skeletal, fps, dims = extract_landmarks(tfile.name)
                stroke, max_x, peak, spine_var = classify_motion(skeletal, "Tennis")
                st.session_state.update({
                    't_skeletal': skeletal, 't_fps': fps, 't_dims': dims,
                    't_stroke': stroke, 't_max_x': max_x, 't_peak': peak,
                    't_vid_name': uploaded_tennis.name, 't_processed': None
                })

        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.warning("⚠️ **Beta Detection.** Verify the motion.")
            sel_stroke = st.selectbox("Confirm Motion:", list(TENNIS_INFO.keys()), 
                                      index=list(TENNIS_INFO.keys()).index(st.session_state['t_stroke']), key="t_sel")
            if st.button("Generate Tennis Analysis", type="primary", key="t_btn"):
                st.session_state['t_processed'] = render_video(tfile.name, st.session_state['t_skeletal'], sel_stroke, TENNIS_INFO, *st.session_state['t_dims'], st.session_state['t_fps'])
                st.session_state['t_final_stroke'] = sel_stroke

        if st.session_state.get('t_processed'):
            st.success("Tennis Analysis Complete!")
            # Export logic (same as before but using t_ session states)
            final_json = json.dumps({"metadata": {"sport": "Tennis", "stroke": st.session_state['t_final_stroke'], "max_x": st.session_state['t_max_x']}, "data": st.session_state['t_skeletal']}, indent=4)
            final_prompt = f"Act as a Tennis Coach. Analyze this {st.session_state['t_final_stroke']}. {TENNIS_INFO[st.session_state['t_final_stroke']]}"
            
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                with open(st.session_state['t_processed'], "rb") as f: zf.writestr("tennis_video.mp4", f.read())
                zf.writestr("motion_data.json", final_json)
                zf.writestr("coach_prompt.txt", final_prompt)
            
            st.download_button("📦 Download Tennis Pack", zip_buf.getvalue(), "tennis_analysis.zip", "application/zip", use_container_width=True)

with tab2:
    st.header("Golf Analysis")
    st.info("🏌️ **Supported Swings:** Driver, Iron, Pitch / Chip, and Putting Stroke.")
    st.info("📐 **Key Metrics:** X-Factor (Hip/Shoulder Rotation), Spine Stability (Angle Maintenance), and Lead Arm Straightness.")
    uploaded_golf = st.file_uploader("Upload Golf Swing", type=["mp4", "mov", "avi"], key="golf_up")
    
    if uploaded_golf:
        gfile = tempfile.NamedTemporaryFile(delete=False)
        gfile.write(uploaded_golf.read())
        
        if 'g_skeletal' not in st.session_state or st.session_state.get('g_vid_name') != uploaded_golf.name:
            with st.spinner("Analyzing Golf Swing..."):
                skeletal, fps, dims = extract_landmarks(gfile.name)
                stroke, max_x, peak, spine_var = classify_motion(skeletal, "Golf")
                st.session_state.update({
                    'g_skeletal': skeletal, 'g_fps': fps, 'g_dims': dims,
                    'g_stroke': stroke, 'g_max_x': max_x, 'g_spine_var': spine_var,
                    'g_vid_name': uploaded_golf.name, 'g_processed': None
                })

        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.warning("⚠️ **Beta Detection.** Verify the swing type.")
            sel_stroke_g = st.selectbox("Confirm Swing:", list(GOLF_INFO.keys()), 
                                        index=list(GOLF_INFO.keys()).index(st.session_state['g_stroke']), key="g_sel")
            if st.button("Generate Golf Analysis", type="primary", key="g_btn"):
                st.session_state['g_processed'] = render_video(gfile.name, st.session_state['g_skeletal'], sel_stroke_g, GOLF_INFO, *st.session_state['g_dims'], st.session_state['g_fps'])
                st.session_state['g_final_stroke'] = sel_stroke_g

        if st.session_state.get('g_processed'):
            st.success("Golf Analysis Complete!")
            # Specialized Golf Export
            g_json = json.dumps({
                "metadata": {
                    "sport": "Golf",
                    "swing_type": st.session_state['g_final_stroke'],
                    "x_factor_max": round(st.session_state['g_max_x'], 1),
                    "spine_stability_score": round(100 - min(st.session_state['g_spine_var'] * 10, 100), 1)
                },
                "skeletal_data": st.session_state['g_skeletal']
            }, indent=4)
            
            g_prompt = (
                f"USER: Analyzing a Golf {st.session_state['g_final_stroke']}.\n"
                f"METRICS: X-Factor: {round(st.session_state['g_max_x'], 1)}°, Spine Stability: {round(100 - min(st.session_state['g_spine_var'] * 10, 100), 1)}/100.\n"
                f"COACHING GOAL: {GOLF_INFO[st.session_state['g_final_stroke']]}\n"
                "Review the JSON for 'early extension' (hip movement towards ball) and lead arm breakdown at the top of the backswing."
            )
            
            zip_buf_g = io.BytesIO()
            with zipfile.ZipFile(zip_buf_g, "w") as zf:
                with open(st.session_state['g_processed'], "rb") as f: zf.writestr("golf_analysis_video.mp4", f.read())
                zf.writestr("golf_data.json", g_json)
                zf.writestr("golf_coach_prompt.txt", g_prompt)
            
            st.download_button("📦 Download Golf Pack", zip_buf_g.getvalue(), "golf_analysis.zip", "application/zip", use_container_width=True)
