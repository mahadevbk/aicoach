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

STROKE_INFO = {
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

# --- HELPER FUNCTIONS ---

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
        # Need timestamp in ms for video mode
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

def classify_stroke_logic(skeletal_data):
    """Classifies the motion based on wrist trajectory and other metrics."""
    wrist_trajectory = []
    max_x_factor = 0
    peak_y = 1.0
    
    for lm in skeletal_data:
        if lm:
            # Track active wrist for trajectory (15=L, 16=R)
            active_wrist = lm[15] if lm[15]['v'] > lm[16]['v'] else lm[16]
            wrist_trajectory.append(active_wrist)
            peak_y = min(peak_y, active_wrist['y'])
            
            # Calculate X-Factor (Shoulder vs Hip line rotation)
            s_vec = np.array([lm[11]['x']-lm[12]['x'], lm[11]['z']-lm[12]['z']])
            h_vec = np.array([lm[23]['x']-lm[24]['x'], lm[23]['z']-lm[24]['z']])
            if np.linalg.norm(s_vec) > 0 and np.linalg.norm(h_vec) > 0:
                dot = np.dot(s_vec, h_vec) / (np.linalg.norm(s_vec) * np.linalg.norm(h_vec))
                max_x_factor = max(max_x_factor, np.degrees(np.arccos(np.clip(dot, -1, 1))))

    stroke = "Unclassified Movement"
    
    if wrist_trajectory:
        xs = [p['x'] for p in wrist_trajectory]
        ys = [p['y'] for p in wrist_trajectory]
        
        lateral_span = max(xs) - min(xs)
        vertical_delta = ys[0] - ys[-1] 
        
        y_velocities = np.diff(ys)
        max_up_vel = abs(min(y_velocities)) if len(y_velocities) > 0 else 0

        if peak_y < 0.35 or max_up_vel > 0.04: 
            if lateral_span < 0.25:
                stroke = "Serve"
            else:
                stroke = "Overhead Smash"
        elif vertical_delta > 0.4:
            stroke = "Lob"
        elif lateral_span > 0.35:
            base = "Forehand" if xs[-1] > xs[0] else "Backhand"
            if ys[-1] > ys[0] + 0.15: 
                stroke = f"{base} Slice"
            else:
                stroke = f"{base} Drive"
        else:
            stroke = "Volley"

    return stroke, max_x_factor, peak_y

def render_video(input_path, skeletal_data, stroke_label, w, h, fps):
    """Generates a video with skeletal overlay and black bar."""
    cap = cv2.VideoCapture(input_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    # Using 'avc1' or 'mp4v' for streamlit compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (w, h + 250))
    
    instructions = STROKE_INFO.get(stroke_label, "Provide general biomechanic feedback.")
    
    progress_bar = st.progress(0, text="Rendering video...")
    total_frames = len(skeletal_data)
    
    for i, frame_data in enumerate(skeletal_data):
        ret, frame = cap.read()
        if not ret: break
        
        canvas = np.zeros((h + 250, w, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = frame
        
        # Draw Skeleton
        if frame_data:
            for s, e in POSE_CONNECTIONS:
                p1 = (int(frame_data[s]['x']*w), int(frame_data[s]['y']*h))
                p2 = (int(frame_data[e]['x']*w), int(frame_data[e]['y']*h))
                cv2.line(canvas, p1, p2, (0, 255, 127), 6) 
        
        # Add Text Overlay in the black bar
        cv2.putText(canvas, f"MOTION: {stroke_label}", (50, h + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(canvas, f"GOAL: {instructions}", (50, h + 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
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
1.  **Upload** your tennis swing video (.mp4, .mov, or .avi).
2.  **Confirm** the detected motion (or manually select it). 
    *⚠️ **Note:** Autodetection is currently in **Beta**. Please cross-check and manually select the correct movement if needed!*
3.  **Generate** and **Download** your analysis ZIP package.
4.  **Get Coached:** Upload the **JSON** and paste the **Prompt** from the ZIP into an AI (like **ChatGPT**, **Claude**, or **Gemini**) for your technical breakdown!
""")

uploaded_file = st.file_uploader("Step 1: Upload Video", type=["mp4", "mov", "avi"])

if uploaded_file:
    # 1. Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # 2. Extract Data (Cache in session state)
    if 'skeletal_data' not in st.session_state or st.session_state.get('video_name') != uploaded_file.name:
        with st.spinner("Analyzing video..."):
            skeletal_data, fps, (w, h) = extract_landmarks(video_path)
            detected_stroke, max_x, peak_reach = classify_stroke_logic(skeletal_data)
            
            st.session_state['skeletal_data'] = skeletal_data
            st.session_state['fps'] = fps
            st.session_state['dims'] = (w, h)
            st.session_state['detected_stroke'] = detected_stroke
            st.session_state['max_x'] = max_x
            st.session_state['peak_reach'] = peak_reach
            st.session_state['video_name'] = uploaded_file.name
            st.session_state['processed_video'] = None # Clear old results

    # 3. Confirmation UI
    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Analysis Settings")
        st.warning("⚠️ **Autodetection is in Beta.** Please verify the motion below.")
        st.info(f"Detected: **{st.session_state['detected_stroke']}**")
        
        selected_stroke = st.selectbox(
            "Confirm or manually choose motion:",
            options=list(STROKE_INFO.keys()),
            index=list(STROKE_INFO.keys()).index(st.session_state['detected_stroke'])
        )
        
        generate_btn = st.button("Generate Final Analysis", type="primary", use_container_width=True)

    # 4. Final Generation
    if generate_btn or st.session_state.get('processed_video'):
        if generate_btn:
            with st.spinner("Rendering final video and files..."):
                w, h = st.session_state['dims']
                fps = st.session_state['fps']
                skeletal_data = st.session_state['skeletal_data']
                
                output_video_path = render_video(video_path, skeletal_data, selected_stroke, w, h, fps)
                
                # Store results in session state
                st.session_state['processed_video'] = output_video_path
                st.session_state['final_stroke'] = selected_stroke

        # Show Results
        st.divider()
        st.subheader("Results")
        
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            st.video(st.session_state['processed_video'])
        
        with res_col2:
            st.success("Analysis Complete!")
            
            # Prepare JSON
            final_json = {
                "metadata": {
                    "stroke": st.session_state['final_stroke'],
                    "max_x_factor": round(st.session_state['max_x'], 1),
                    "peak_reach": round(1 - st.session_state['peak_reach'], 3),
                    "fps": st.session_state['fps']
                },
                "skeletal_data": st.session_state['skeletal_data']
            }
            json_str = json.dumps(final_json, indent=4)
            
            # Prepare Prompt
            instr = STROKE_INFO[st.session_state['final_stroke']]
            final_prompt = (
                f"USER: I am uploading a JSON of my tennis skeletal data.\n"
                f"DETECTION: The app identified this as a {st.session_state['final_stroke']}.\n"
                f"INSTRUCTIONS: Act as a World-Class Tennis Biomechanics Coach. {instr}\n"
                "Use the provided skeletal time-series data to find power leaks in my kinetic chain.\n"
                "Examine Landmark 0 (Head) for balance and Landmarks 27-32 (Feet) for weight transfer."
            )
            
            # Create ZIP in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                # Add video
                with open(st.session_state['processed_video'], "rb") as f:
                    zip_file.writestr("analysed_video.mp4", f.read())
                # Add JSON
                zip_file.writestr("motion_data.json", json_str)
                # Add Prompt
                zip_file.writestr("coach_prompt.txt", final_prompt)
            
            # Download Buttons
            st.info("💡 **Next Step:** Upload the **JSON** and paste the **Prompt** into an AI (like ChatGPT, Claude, or Gemini) to get your personalized coaching report!")
            
            st.download_button(
                label="📦 Download All (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="tennis_analysis_pack.zip",
                mime="application/zip",
                use_container_width=True
            )
            
            st.divider()

            with open(st.session_state['processed_video'], "rb") as f:
                st.download_button("Download Video Only", f, "analysed_video.mp4", "video/mp4", use_container_width=True)
            
            st.download_button("Download JSON Only", json_str, "motion_data.json", "application/json", use_container_width=True)
            
            st.download_button("Download Prompt Only", final_prompt, "coach_prompt.txt", "text/plain", use_container_width=True)
            
            with st.expander("Preview AI Prompt Brief"):
                st.code(final_prompt)
            
            with st.expander("Preview JSON Metadata"):
                st.json(final_json["metadata"])
