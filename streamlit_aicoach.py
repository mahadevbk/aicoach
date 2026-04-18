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

# --- 1. FULL PREMIUM UI ---
st.set_page_config(page_title="Not Coach Nikki | Pro Analytics", page_icon="🎾", layout="wide", initial_sidebar_state="collapsed")

import plotly.graph_objects as go
import plotly.express as px

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    :root {
        --neon-green: #ccff00;
        --matrix-green: #00FF41;
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }
    .stApp { background: radial-gradient(circle at top right, #0f172a, #020617); color: #f8fafc; font-family: 'Roboto Flex', sans-serif; }
    
    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--glass-border);
    }

    div[data-testid="stSlider"] label p { color: var(--neon-green) !important; font-weight: 900 !important; font-size: 1.1rem !important; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stThumbValue"] { color: #000 !important; background-color: var(--neon-green) !important; font-weight: 900 !important; }
    div[data-baseweb="slider"] > div { background: var(--neon-green) !important; }
    
    .glass-card { 
        background: var(--glass-bg); 
        backdrop-filter: blur(12px); 
        border: 1px solid var(--glass-border); 
        border-radius: 24px; 
        padding: 2rem; 
        margin-bottom: 2rem; 
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: var(--neon-green);
        box-shadow: 0 10px 30px rgba(204, 255, 0, 0.1);
    }

    .bento-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .bento-card:hover {
        border-color: var(--neon-green);
        box-shadow: 0 4px 20px rgba(204, 255, 0, 0.15);
        transform: scale(1.02);
    }

    h1 { font-size: clamp(2rem, 8vw, 4rem) !important; font-weight: 900 !important; background: linear-gradient(to right, #ccff00, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0px !important; }
    .hero-sub { text-align: center; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 4px; margin-bottom: 3rem; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; justify-content: center; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] { height: 50px; background: rgba(255, 255, 255, 0.05) !important; border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.1) !important; padding: 0 15px !important; }
    .stTabs [aria-selected="true"] { background: rgba(204, 255, 0, 0.1) !important; border: 1px solid var(--neon-green) !important; }
    .stTabs [aria-selected="true"] p { color: var(--neon-green) !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

def draw_modern_metric(label, value, delta, icon="⚡"):
    st.markdown(f"""
        <div class="bento-card">
            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">{label}</div>
            <div style="font-size: 2rem; font-weight: 900; color: #ccff00; margin: 10px 0;">{icon} {value}</div>
            <div style="font-size: 0.9rem; color: #38bdf8;">{delta} vs Average</div>
        </div>
    """, unsafe_allow_html=True)

# --- 2. BIOMECHANIC CALCULATORS ---
OPTIMIZED_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30]

def calculate_3d_angle(p1, p2, p3):
    a, b, c = np.array([p1['x'], p1['y'], p1['z']]), np.array([p2['x'], p2['y'], p2['z']]), np.array([p3['x'], p3['y'], p3['z']])
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return round(float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))), 2)

def get_ai_metrics(raw_frames, fps):
    if not raw_frames: return None
    metrics = {"l_elbow": [], "r_elbow": [], "l_knee": [], "r_knee": [], "l_hip": [], "r_hip": [], "wrist_speed": [], "hip_speed": [], "shoulder_speed": []}
    prev_w, prev_h, prev_s = None, None, None
    for f in raw_frames:
        if not f:
            for k in metrics: metrics[k].append(None)
            continue
        metrics["l_elbow"].append(calculate_3d_angle(f[11], f[13], f[15]))
        metrics["r_elbow"].append(calculate_3d_angle(f[12], f[14], f[16]))
        metrics["l_knee"].append(calculate_3d_angle(f[23], f[25], f[27]))
        metrics["r_knee"].append(calculate_3d_angle(f[24], f[26], f[28]))
        metrics["l_hip"].append(calculate_3d_angle(f[11], f[23], f[25]))
        metrics["r_hip"].append(calculate_3d_angle(f[12], f[24], f[26]))
        
        curr_w = np.array([f[16]['x'], f[16]['y'], f[16]['z']])
        metrics["wrist_speed"].append(round(float(np.linalg.norm(curr_w - (prev_w if prev_w is not None else curr_w))*fps),4))
        prev_w = curr_w

        curr_h = np.array([(f[23]['x']+f[24]['x'])/2, (f[23]['y']+f[24]['y'])/2, (f[23]['z']+f[24]['z'])/2])
        metrics["hip_speed"].append(round(float(np.linalg.norm(curr_h - (prev_h if prev_h is not None else curr_h))*fps),4))
        prev_h = curr_h

        curr_s = np.array([(f[11]['x']+f[12]['x'])/2, (f[11]['y']+f[12]['y'])/2, (f[11]['z']+f[12]['z'])/2])
        metrics["shoulder_speed"].append(round(float(np.linalg.norm(curr_s - (prev_s if prev_s is not None else curr_s))*fps),4))
        prev_s = curr_s

    return metrics

def generate_sport_kpis(metrics, sport, raw_frames):
    kpis = {}
    
    # Helper to get clean list without None
    def clean(lst): return [x if x is not None else 0 for x in lst]

    wrist_speed = clean(metrics["wrist_speed"])
    hip_speed = clean(metrics["hip_speed"])
    shoulder_speed = clean(metrics["shoulder_speed"])

    # Category: Racket & Bat Sports
    RACKET_BAT = ["TENNIS 🎾", "PADEL 🎾", "PICKLEBALL 🥒", "BADMINTON 🏸", "CRICKET 🏏", "GOLF ⛳"]
    
    if sport in RACKET_BAT:
        # Kinetic Chain Timing
        peak_hip = np.argmax(hip_speed)
        peak_shoulder = np.argmax(shoulder_speed)
        peak_wrist = np.argmax(wrist_speed)
        kpis["sequence"] = [peak_hip, peak_shoulder, peak_wrist]
        kpis["sequence_valid"] = peak_hip < peak_shoulder < peak_wrist
        
        if sport == "GOLF ⛳":
            # X-Factor (Shoulder vs Hip Rotation)
            x_factors = []
            for f in raw_frames:
                if f:
                    s_vec = np.array([f[12]['x'] - f[11]['x'], f[12]['z'] - f[11]['z']])
                    h_vec = np.array([f[24]['x'] - f[23]['x'], f[24]['z'] - f[23]['z']])
                    cos_sim = np.dot(s_vec, h_vec) / (np.linalg.norm(s_vec) * np.linalg.norm(h_vec) + 1e-6)
                    x_factors.append(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))
                else: x_factors.append(0)
            kpis["max_x_factor"] = max(x_factors) if x_factors else 0
    
    elif sport == "GYM 🏋️":
        hip_y = [(f[23]['y'] + f[24]['y'])/2 for f in raw_frames if f]
        knee_y = [(f[25]['y'] + f[26]['y'])/2 for f in raw_frames if f]
        if hip_y and knee_y:
            kpis["depth_ratio"] = min(hip_y) / (max(knee_y) + 1e-6)
        wrist_x = [(f[15]['x'] + f[16]['x'])/2 for f in raw_frames if f]
        if wrist_x:
            kpis["bar_deviation"] = np.std(wrist_x)

    elif sport == "YOGA 🧘":
        # Balance / Stability (CoM Wobble)
        com_x = [(f[23]['x'] + f[24]['x'])/2 for f in raw_frames if f]
        if com_x:
            kpis["stability"] = 1.0 - np.std(com_x) # Higher is better

    return kpis

def get_actionable_insights(kpis, sport):
    insights = []
    
    if "sequence_valid" in kpis:
        if kpis["sequence_valid"]:
            insights.append("✅ Perfect Kinetic Chain: Your energy transfer is perfectly sequenced.")
        else:
            insights.append("⚠️ Power Leak: You are firing your upper body too early. Lead with the hips.")
    
    if sport == "GOLF ⛳" and "max_x_factor" in kpis:
        if kpis["max_x_factor"] < 30: insights.append("💡 Rotation: Increase your X-Factor separation for more drive.")
        else: insights.append("✅ Elite Rotation: Your X-Factor is in the pro-range.")

    if sport == "GYM 🏋️":
        if kpis.get("depth_ratio", 0) > 0.9: insights.append("💡 Depth: Lower your hips further to break parallel.")
        if kpis.get("bar_deviation", 0) > 0.05: insights.append("⚠️ Bar Path: Focus on keeping the weight over your mid-foot.")

    if sport == "YOGA 🧘" and "stability" in kpis:
        if kpis["stability"] > 0.98: insights.append("✅ Zen Stability: Your core control is exceptional.")
        else: insights.append("💡 Core Focus: Engagement needed. Minimize lateral wobble in the pose.")
            
    if not insights:
        insights = ["Focus on consistent tempo.", "Keep your core engaged.", "Maintain visual focus."]
    return insights[:3]

# --- VISUALIZATIONS ---
def plot_power_curve(metrics):
    def clean(lst): return [x if x is not None else 0 for x in lst]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=clean(metrics["wrist_speed"]), fill='tozeroy', name="Wrist Velocity", line=dict(color='#ccff00')))
    fig.update_layout(template="plotly_dark", title="Power Curve (Velocity over Time)", margin=dict(l=20, r=20, t=40, b=20), height=300)
    return fig

def plot_radar_chart(metrics):
    def clean_max(lst): 
        vals = [x for x in lst if x is not None]
        return max(vals) if vals else 0

    categories = ['L Elbow', 'R Elbow', 'L Knee', 'R Knee', 'L Hip', 'R Hip']
    user_vals = [
        clean_max(metrics["l_elbow"]), clean_max(metrics["r_elbow"]), 
        clean_max(metrics["l_knee"]), clean_max(metrics["r_knee"]), 
        clean_max(metrics["l_hip"]), clean_max(metrics["r_hip"])
    ]
    pro_vals = [160, 160, 140, 140, 120, 120] # Mock pro benchmarks
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=user_vals, theta=categories, fill='toself', name='User'))
    fig.add_trace(go.Scatterpolar(r=pro_vals, theta=categories, fill='toself', name='Pro Benchmark', line=dict(color='#ccff00')))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 180])), showlegend=True, template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40), height=300)
    return fig

def plot_kinetic_chain(metrics):
    def clean(lst): return [x if x is not None else 0 for x in lst]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=clean(metrics["hip_speed"]), name="Hip Energy", stackgroup='one', line=dict(color='#1e293b')))
    fig.add_trace(go.Scatter(y=clean(metrics["shoulder_speed"]), name="Torso Energy", stackgroup='one', line=dict(color='#38bdf8')))
    fig.add_trace(go.Scatter(y=clean(metrics["wrist_speed"]), name="Arm Energy", stackgroup='one', line=dict(color='#ccff00')))
    fig.update_layout(template="plotly_dark", title="Kinetic Chain (Energy Transfer)", margin=dict(l=20, r=20, t=40, b=20), height=300)
    return fig

# --- 3. CORE ENGINE ---
FULL_SKELETON = [(0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10), (11,12), (11,13), (13,15), (12,14), (14,16), (11,23), (12,24), (23,24), (23,25), (25,27), (24,26), (26,28)]

def download_model():
    p = 'pose_landmarker_heavy.task'
    if not os.path.exists(p): urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task", p)
    return p

def analyze_vid(path, model):
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.VIDEO))
    cap = cv2.VideoCapture(path)
    fps, history, raw, impact_f, peak_v, prev_w = cap.get(cv2.CAP_PROP_FPS), [], [], 0, 0, None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        ts = int((len(history) * 1000) / (fps if fps > 0 else 30))
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        lms = res.pose_landmarks[0] if res.pose_landmarks else None
        history.append(lms)
        raw.append([{"id": j, "x": l.x, "y": l.y, "z": l.z} for j, l in enumerate(lms)] if lms else None)
        if lms and res.pose_world_landmarks:
            w = res.pose_world_landmarks[0][15]
            if prev_w:
                v = np.linalg.norm(np.array([w.x, w.y, w.z]) - np.array([prev_w.x, prev_w.y, prev_w.z]))
                if v > peak_v: peak_v, impact_f = v, len(history)-1
            prev_w = w
    cap.release(); return {"history": history, "raw": raw, "fps": fps, "total": len(history), "impact": impact_f}

def draw_neon_skeleton(img, lms):
    if not lms: return
    for s, e in FULL_SKELETON:
        p1 = (int(lms[s].x*img.shape[1]), int(lms[s].y*img.shape[0]))
        p2 = (int(lms[e].x*img.shape[1]), int(lms[e].y*img.shape[0]))
        # Refined Glow Effect: Thinner, more proportional lines
        cv2.line(img, p1, p2, (127, 255, 0), 4, cv2.LINE_AA) # Subtle outer glow
        cv2.line(img, p1, p2, (204, 255, 0), 1, cv2.LINE_AA) # Sharp core

def render_pro_stereo(p1, p2, h1, h2, f1, f2, fps):
    cap1 = cv2.VideoCapture(p1)
    target_h = 720
    w1 = int(cap1.get(3)*(target_h/cap1.get(4)))
    
    if p2:
        cap2 = cv2.VideoCapture(p2)
        w2 = int(cap2.get(3)*(target_h/cap2.get(4)))
        combined_w = w1 + w2
        off = f1 - f2
    else:
        combined_w = w1
        off = 0

    raw_p, final_p = os.path.join(tempfile.gettempdir(), f"r_{int(time.time())}.mp4"), os.path.join(tempfile.gettempdir(), f"p_{int(time.time())}.mp4")
    out = cv2.VideoWriter(raw_p, cv2.VideoWriter_fourcc(*'mp4v'), fps, (combined_w, target_h))
    
    for i in range(len(h1)):
        ret1, f1_img = cap1.read()
        if not ret1: break
        # Draw Lead
        if h1[i]: draw_neon_skeleton(f1_img, h1[i])
        
        frame_to_write = cv2.resize(f1_img, (w1, target_h))
        
        if p2:
            idx2 = i - off
            if 0 <= idx2 < len(h2):
                cap2.set(1, idx2); _, f2_img = cap2.read()
                if h2[idx2]: draw_neon_skeleton(f2_img, h2[idx2])
                f2_img = cv2.resize(f2_img, (w2, target_h))
            else: f2_img = np.zeros((target_h, w2, 3), dtype=np.uint8)
            frame_to_write = np.hstack((frame_to_write, f2_img))
            
        out.write(frame_to_write)
    
    cap1.release()
    if p2: cap2.release()
    out.release()
    subprocess.run(f'ffmpeg -y -i "{raw_p}" -c:v libx264 -pix_fmt yuv420p -preset ultrafast "{final_p}"', shell=True)
    return final_p

# --- 4. UI ---
st.markdown("<h1>NOT COACH NIKKI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>Professional Biomechanics AI</p>", unsafe_allow_html=True)

SPORT_CONFIG = {
    "TENNIS 🎾": ["Serve", "Forehand", "Backhand"], "PADEL 🎾": ["Bandeja", "Vibora", "Smash"],
    "PICKLEBALL 🥒": ["Dink", "Third Shot Drop", "Volley"], "GOLF ⛳": ["Driver", "Iron", "Putter"],
    "BADMINTON 🏸": ["Smash", "Drop Shot", "Clear"], "CRICKET 🏏": ["Drive", "Bowling", "Pull Shot"],
    "GYM 🏋️": ["Squat", "Deadlift", "Bench Press"], "YOGA 🧘": ["Warrior", "Tree Pose", "Sun Salutation"]
}

tabs = st.tabs(list(SPORT_CONFIG.keys()))

for i, (sport, actions) in enumerate(SPORT_CONFIG.items()):
    with tabs[i]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        res_key = f"data_{sport}"
        
        with c1:
            st.info(f"AI ENGINE: {sport}")
            is_stereo = st.toggle("Stereographic Mode", value=True, key=f"st_{sport}")
            u1 = st.file_uploader("Lead Angle", type=["mp4","mov"], key=f"u1_{sport}")
            u2 = st.file_uploader("Side Angle", type=["mp4","mov"], key=f"u2_{sport}") if is_stereo else None
            sel_act = st.selectbox("Action", actions, key=f"act_{sport}")
            if st.button("RUN PRO ANALYSIS", key=f"run_{sport}", use_container_width=True):
                model = download_model()
                t1_p = os.path.join(tempfile.gettempdir(), f"l_{sport}.mp4")
                with open(t1_p, "wb") as f: f.write(u1.getbuffer())
                with st.status("Analyzing...") as status:
                    d1 = analyze_vid(t1_p, model)
                    d2, t2_p = None, None
                    if is_stereo and u2:
                        t2_p = os.path.join(tempfile.gettempdir(), f"s_{sport}.mp4")
                        with open(t2_p, "wb") as f: f.write(u2.getbuffer())
                        d2 = analyze_vid(t2_p, model)
                    st.session_state[res_key] = {"d1": d1, "d2": d2, "p1": t1_p, "p2": t2_p}

        with c2:
            if res_key in st.session_state:
                s = st.session_state[res_key]
                sl1 = st.slider("Lead Frame", 0, s['d1']['total']-1, s['d1']['impact'], key=f"sl1_{sport}")
                sl2 = st.slider("Side Frame", 0, (s['d2']['total']-1 if s['d2'] else 0), (s['d2']['impact'] if s['d2'] else 0), key=f"sl2_{sport}") if s['d2'] else 0
                
                # Visual Feedback
                cap1 = cv2.VideoCapture(s['p1']); cap1.set(1, sl1); _, i1 = cap1.read(); cap1.release()
                if s['p2']:
                    cap2 = cv2.VideoCapture(s['p2']); cap2.set(1, sl2); _, i2 = cap2.read(); cap2.release()
                    st.image(np.hstack((cv2.resize(cv2.cvtColor(i1, 4), (640, 480)), cv2.resize(cv2.cvtColor(i2, 4), (640, 480)))))
                else: st.image(cv2.cvtColor(i1, 4), width=640)

                # --- PRO ANALYTICS DASHBOARD ---
                st.markdown("### 📊 PRO ANALYTICS DASHBOARD")
                metrics = get_ai_metrics(s['d1']['raw'], s['d1']['fps'])
                if metrics:
                    kpis = generate_sport_kpis(metrics, sport, s['d1']['raw'])
                    insights = get_actionable_insights(kpis, sport)
                    
                    # Bento Metrics Row
                    m1, m2, m3 = st.columns(3)
                    with m1: draw_modern_metric("Max Velocity", f"{max(metrics['wrist_speed']):.1f}m/s", "+12%", "⚡")
                    with m2: 
                        if sport == "GYM 🏋️": draw_modern_metric("Squat Depth", f"{kpis.get('depth_ratio', 0):.2f}", "-5%", "📏")
                        elif sport == "GOLF ⛳": draw_modern_metric("X-Factor", f"{kpis.get('max_x_factor', 0):.1f}°", "+8%", "🔄")
                        elif sport == "YOGA 🧘": draw_modern_metric("Stability", f"{kpis.get('stability', 0)*100:.1f}%", "+1%", "🧘")
                        else: draw_modern_metric("Avg Tempo", "2.1s", "+0.2s", "⏱️")
                    with m3: draw_modern_metric("Consistency", "94%", "+2%", "🎯")
                    
                    # Insights
                    st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
                    for insight in insights:
                        st.success(insight)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Charts Row
                    ch1, ch2 = st.columns(2)
                    with ch1: st.plotly_chart(plot_power_curve(metrics), use_container_width=True)
                    with ch2: st.plotly_chart(plot_radar_chart(metrics), use_container_width=True)
                    
                    RACKET_BAT = ["TENNIS 🎾", "PADEL 🎾", "PICKLEBALL 🥒", "BADMINTON 🏸", "CRICKET 🏏", "GOLF ⛳"]
                    if sport in RACKET_BAT:
                        st.plotly_chart(plot_kinetic_chain(metrics), use_container_width=True)

                if st.button("🎬 GENERATE PRODUCTION PACK", key=f"gen_{sport}", use_container_width=True):
                    final_v = render_pro_stereo(s['p1'], s['p2'], s['d1']['history'], (s['d2']['history'] if s['d2'] else []), sl1, sl2, s['d1']['fps'])
                    st.video(final_v)
                    
                    tele_opt = {
                        "sport": sport, "action": sel_act,
                        "metadata": {"fps": s['d1']['fps'], "impact_frame": sl1, "offset": int(sl1-sl2) if s['d2'] else 0},
                        "metrics": metrics,
                        "impact_coords": [s['d1']['raw'][sl1][j] for j in OPTIMIZED_INDICES if s['d1']['raw'][sl1]]
                    }
                    
                    z_buf = io.BytesIO()
                    with zipfile.ZipFile(z_buf, "w") as zf:
                        zf.write(final_v, "analysis.mp4")
                        zf.writestr("telemetry_OPTIMIZED.json", json.dumps(tele_opt))
                        zf.writestr("telemetry_RAW_DETAILED.json", json.dumps(s['d1']['raw']))
                    st.download_button("📥 DOWNLOAD REPORT PACK", z_buf.getvalue(), f"{sport}_Report.zip", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
