import streamlit as st
import os
import subprocess
import json
from src.pipeline import run_pipeline

st.set_page_config(page_title="AI Tennis Coach", layout="centered")

st.title("🎾 AI Pro: Progressive Slow-Mo")
st.markdown("""
Analysis now features **Dynamic Speed Ramping**:
* **Real-time** start/end.
* **0.5x** during the approach and follow-through.
* **0.25x (Ultra Slow)** at the exact point of impact.
""")

uploaded_file = st.file_uploader("Upload Swing", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    if not os.path.exists("data"): os.makedirs("data")
    if not os.path.exists("output"): os.makedirs("output")
    
    video_path = "data/input_video.mp4"
    raw_output = "output/raw_analysis.mp4"
    final_output = "output/pro_analysis.mp4"

    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("Extracting biometrics and detecting impact...")
    
    try:
        # 1. Run Pipeline
        run_pipeline(video_path, raw_output)
        
        # 2. Read Metadata to find the Impact Moment
        with open("output/motion_data.json", "r") as f:
            data = json.load(f)
            impact_f = data["impact_frame"]
            fps = data["fps"]
            total_f = data["total_frames"]

        # 3. Calculate Time Segments (Seconds)
        t_impact = impact_f / fps
        t_total = total_f / fps
        
        # Define slow-mo boundaries around impact
        t1 = max(0, t_impact - 1.0) # Start slowing to 0.5x
        t2 = max(0, t_impact - 0.3) # Start slowing to 0.25x
        t3 = min(t_total, t_impact + 0.3) # End 0.25x
        t4 = min(t_total, t_impact + 1.0) # End 0.5x

        # 4. FFmpeg Complex Speed Ramp
        # setpts: 1.0=1x, 2.0=0.5x, 4.0=0.25x
        filter_cmd = (
            f"[0:v]trim=0:{t1},setpts=PTS[v1]; "
            f"[0:v]trim={t1}:{t2},setpts=2.0*(PTS-STARTPTS)[v2]; "
            f"[0:v]trim={t2}:{t3},setpts=4.0*(PTS-STARTPTS)[v3]; "
            f"[0:v]trim={t3}:{t4},setpts=2.0*(PTS-STARTPTS)[v4]; "
            f"[0:v]trim={t4}:{t_total},setpts=PTS-STARTPTS[v5]; "
            f"[v1][v2][v3][v4][v5]concat=n=5:v=1:a=0[outv]"
        )

        st.info("Applying dynamic speed ramp...")
        subprocess.call([
            'ffmpeg', '-i', raw_output, 
            '-filter_complex', filter_cmd, 
            '-map', '[outv]',
            '-vcodec', 'libx264', '-crf', '25', '-y', final_output
        ])

        st.video(final_output)
        st.download_button("Download Pro Analysis", open(final_output, "rb"), "pro_slowmo.mp4")

    except Exception as e:
        st.error(f"Analysis Error: {e}")
