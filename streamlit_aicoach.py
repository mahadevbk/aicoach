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
from generate_brief import generate_brief

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# --- 1. FULL PREMIUM UI ---
st.set_page_config(page_title="Vector Victor AI | Pro Analytics", page_icon="🎾", layout="wide", initial_sidebar_state="collapsed")

import plotly.graph_objects as go
import plotly.express as px

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Flex:wght@100..1000&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Bitcount+Prop+Single&display=swap');
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
        padding: 1.5rem; 
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

    h1 { font-family: 'Bitcount Prop Single', sans-serif !important; font-size: clamp(2rem, 8vw, 4rem) !important; font-weight: 900 !important; background: linear-gradient(to right, #00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0px !important; }
    .hero-sub { font-family: 'Bitcount Prop Single', sans-serif !important; text-align: center; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 4px; margin-bottom: 3rem; }

    
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
    return round(float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))), 1)

def get_dist(p1, p2):
    return np.linalg.norm(np.array([p1['x'], p1['y'], p1['z']]) - np.array([p2['x'], p2['y'], p2['z']]))

def get_midpoint(p1, p2):
    return {'x': (p1['x']+p2['x'])/2, 'y': (p1['y']+p2['y'])/2, 'z': (p1['z']+p2['z'])/2}

def calculate_lean(p1, p2, plane='sagittal'):
    v = np.array([p2['x']-p1['x'], p2['y']-p1['y'], p2['z']-p1['z']])
    if plane == 'sagittal': # Y-Z plane
        angle = np.degrees(np.arctan2(v[2], v[1]))
    else: # Coronal: X-Y plane
        angle = np.degrees(np.arctan2(v[0], v[1]))
    return round(float(angle), 1)

def interpolate_landmarks(raw_frames):
    total = len(raw_frames)
    if total == 0: return []
    frames = [f.copy() if f else None for f in raw_frames]
    for j in range(33):
        missing = [i for i in range(total) if not frames[i] or frames[i][j] is None]
        if not missing or len(missing) == total: continue
        for m_idx in missing:
            prev_idx = next((i for i in range(m_idx-1, -1, -1) if frames[i] and frames[i][j]), None)
            next_idx = next((i for i in range(m_idx+1, total) if frames[i] and frames[i][j]), None)
            if prev_idx is not None and next_idx is not None:
                p, n = frames[prev_idx][j], frames[next_idx][j]
                t = (m_idx - prev_idx) / (next_idx - prev_idx)
                val = {'x': p['x'] + t*(n['x']-p['x']), 'y': p['y'] + t*(n['y']-p['y']), 'z': p['z'] + t*(n['z']-p['z'])}
            elif prev_idx is not None: val = frames[prev_idx][j]
            elif next_idx is not None: val = frames[next_idx][j]
            else: val = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            if not frames[m_idx]: frames[m_idx] = [{'x':0.0, 'y':0.0, 'z':0.0} for _ in range(33)]
            frames[m_idx][j] = val
    return frames

def build_pro_telemetry(raw_frames, sport_raw, action, event_frame, fps, camera_mode):
    total_frames = len(raw_frames)
    sport_clean = "".join([c for c in sport_raw if ord(c) < 128]).strip().upper()
    event_frame = max(0, min(event_frame, total_frames - 1))
    offset = int(event_frame - total_frames)
    
    metrics = {
        "r_elbow": [], "l_elbow": [], "r_knee": [], "l_knee": [], "r_hip": [], "l_hip": [],
        "r_shoulder_abduction": [], "l_shoulder_abduction": [], "r_ankle": [], "l_ankle": [],
        "r_wrist_speed": [], "l_wrist_speed": [], "r_ankle_speed": [], "l_ankle_speed": [],
        "shoulder_z_diff": [], "hip_z_diff": [], "trunk_forward_lean": [], "trunk_lateral_lean": []
    }
    
    validation_warnings = []
    
    # Fix: 3-frame median filter for keypoint positions to eliminate tracking dropouts
    filtered_pos = []
    for i in range(total_frames):
        f_filt = {}
        for pt_idx in [15, 16, 27, 28]:
            win_x, win_y, win_z = [], [], []
            for d in [-1, 0, 1]:
                idx = max(0, min(total_frames - 1, i + d))
                if raw_frames[idx]:
                    win_x.append(raw_frames[idx][pt_idx]['x'])
                    win_y.append(raw_frames[idx][pt_idx]['y'])
                    win_z.append(raw_frames[idx][pt_idx]['z'])
            if win_x:
                f_filt[pt_idx] = {'x': float(np.median(win_x)), 'y': float(np.median(win_y)), 'z': float(np.median(win_z))}
            else:
                f_filt[pt_idx] = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        filtered_pos.append(f_filt)

    last_valid_scale = 0.5
    prev_pts = [None] * 4 # RW, LW, RA, LA
    
    for idx_f, f in enumerate(raw_frames):
        if not f:
            for k in metrics: metrics[k].append(None)
            continue
            
        mid_s, mid_h = get_midpoint(f[11], f[12]), get_midpoint(f[23], f[24])
        current_scale = get_dist(mid_s, mid_h)
        if current_scale < 0.05:
            scale = last_valid_scale
        else:
            scale = current_scale
            last_valid_scale = current_scale
            
        metrics["r_elbow"].append(calculate_3d_angle(f[12], f[14], f[16]))
        metrics["l_elbow"].append(calculate_3d_angle(f[11], f[13], f[15]))
        metrics["r_knee"].append(calculate_3d_angle(f[24], f[26], f[28]))
        metrics["l_knee"].append(calculate_3d_angle(f[23], f[25], f[27]))
        metrics["r_hip"].append(calculate_3d_angle(f[12], f[24], f[26]))
        metrics["l_hip"].append(calculate_3d_angle(f[11], f[23], f[25]))
        metrics["r_shoulder_abduction"].append(calculate_3d_angle(f[14], f[12], f[24]))
        metrics["l_shoulder_abduction"].append(calculate_3d_angle(f[13], f[11], f[23]))
        metrics["r_ankle"].append(calculate_3d_angle(f[26], f[28], f[30]))
        metrics["l_ankle"].append(calculate_3d_angle(f[25], f[27], f[29]))
        
        for i, pt_idx in enumerate([16, 15, 28, 27]):
            curr_pt = filtered_pos[idx_f][pt_idx]
            if prev_pts[i]:
                raw_disp = get_dist(curr_pt, prev_pts[i])
                norm_speed = round(raw_disp / scale, 4)
                metrics[list(metrics.keys())[10+i]].append(norm_speed)
            else:
                metrics[list(metrics.keys())[10+i]].append(0.0)
            prev_pts[i] = curr_pt
        
        metrics["shoulder_z_diff"].append(round(f[12]['z'] - f[11]['z'], 3))
        metrics["hip_z_diff"].append(round(f[24]['z'] - f[23]['z'], 3))
        metrics["trunk_forward_lean"].append(calculate_lean(mid_s, mid_h, 'sagittal'))
        metrics["trunk_lateral_lean"].append(calculate_lean(mid_s, mid_h, 'coronal'))

    # Fix: Confirm and clamp remaining speed glitches exceeding 1.5
    for key in ["r_wrist_speed", "l_wrist_speed", "r_ankle_speed", "l_ankle_speed"]:
        if key in metrics:
            for i in range(len(metrics[key])):
                if metrics[key][i] is not None and metrics[key][i] > 1.5:
                    validation_warnings.append(f"Confirmed tracker glitch in {key} at frame {i}. Speed {metrics[key][i]} clamped.")
                    prev_v = metrics[key][max(0, i-1)] or 0.0
                    next_v = metrics[key][min(total_frames-1, i+1)] or 0.0
                    metrics[key][i] = round((prev_v + next_v) / 2, 4)

    racket_sports = ["TENNIS", "PADEL", "PICKLEBALL", "BADMINTON", "SQUASH"]
    if sport_clean in racket_sports:
        for k in ["r_ankle", "l_ankle", "r_ankle_speed", "l_ankle_speed"]: del metrics[k]

    def get_snapshot(idx):
        idx = max(0, min(idx, total_frames - 1))
        f = raw_frames[idx]
        if not f: return {}
        mid_s, mid_h = get_midpoint(f[11], f[12]), get_midpoint(f[23], f[24])
        hip_dist = get_dist(f[23], f[24])
        
        # Fix 3: Shoulder tilt angle wrap
        tilt = np.degrees(np.arctan2(f[12]['y'] - f[11]['y'], f[12]['x'] - f[11]['x']))
        if abs(tilt) > 90:
            tilt = tilt - 180 if tilt > 0 else tilt + 180
        
        # Fix 4: Stance width ratio guard
        sw_ratio = round(get_dist(f[27], f[28]) / (hip_dist + 1e-6), 4)
        sw_note = None
        if sw_ratio > 2.5 or hip_dist < 0.05:
            sw_ratio = None
            sw_note = "keypoint_unreliable"

        snap = {
            "r_elbow_angle": calculate_3d_angle(f[12], f[14], f[16]), "l_elbow_angle": calculate_3d_angle(f[11], f[13], f[15]),
            "r_knee_angle": calculate_3d_angle(f[24], f[26], f[28]), "l_knee_angle": calculate_3d_angle(f[23], f[25], f[27]),
            "r_hip_angle": calculate_3d_angle(f[12], f[24], f[26]), "l_hip_angle": calculate_3d_angle(f[11], f[23], f[25]),
            "r_shoulder_abduction_angle": calculate_3d_angle(f[14], f[12], f[24]), "l_shoulder_abduction_angle": calculate_3d_angle(f[13], f[11], f[23]),
            "shoulder_tilt_deg": round(tilt, 1),
            "trunk_forward_lean": calculate_lean(mid_s, mid_h, 'sagittal'), "trunk_lateral_lean": calculate_lean(mid_s, mid_h, 'coronal'),
            "shoulder_z_diff": round(f[12]['z'] - f[11]['z'], 3), "hip_z_diff": round(f[24]['z'] - f[23]['z'], 3),
            "hip_shoulder_separation": round((f[12]['z'] - f[11]['z']) - (f[24]['z'] - f[23]['z']), 3),
            "r_wrist_above_r_shoulder": f[16]['y'] < f[12]['y'], "l_wrist_above_l_shoulder": f[15]['y'] < f[11]['y'],
            "feet_grounded": f[28]['y'] > 0.80 and f[27]['y'] > 0.80, "stance_width_ratio": sw_ratio
        }
        if sw_note: snap["stance_width_note"] = sw_note
        return snap

    output = {
        "sport": sport_clean, "action": action, "camera": camera_mode,
        "metadata": {
            "fps": fps, "total_frames": total_frames, "offset": offset,
            "dominant_side": "right" if np.max([s for s in metrics["r_wrist_speed"] if s is not None]) > np.max([s for s in metrics["l_wrist_speed"] if s is not None]) else "left",
            "coordinate_system": {"y_axis": "increases_downward", "z_axis": "depth_into_camera", "normalisation": "mediapipe_image_fraction_0_to_1"},
            "validation_warnings": validation_warnings
        },
        "metrics": {k: [v for v in metrics[k] if v is not None] for k in metrics if any(v is not None for v in metrics[k])},
        "event_snapshot": get_snapshot(event_frame), "phase_snapshots": {}, "speed_analysis": {}, "rotation_analysis": {}, "balance_stability": {}
    }
    output["event_snapshot"]["keypoints"] = [{"id": j, "x": round(raw_frames[event_frame][j]['x'], 4), "y": round(raw_frames[event_frame][j]['y'], 4), "z": round(raw_frames[event_frame][j]['z'], 4)} for j in [0,11,12,13,14,15,16,23,24,25,26,27,28,29,30] if raw_frames[event_frame]]

    phases = []
    if sport_clean in racket_sports: phases = [("trophy", -40), ("swing_start", -15), ("follow_through", 20)]
    elif sport_clean == "GOLF": phases = [("address", -80), ("top", -30), ("downswing", -12), ("follow", 25)]
    elif sport_clean == "GYM": phases = [("start", -45), ("midpoint", -22), ("finish", 30)]
    elif sport_clean == "YOGA": phases = [("approach", -30), ("exit", 30)]
    for name, p_off in phases: output["phase_snapshots"][name] = get_snapshot(event_frame + p_off)

    def analyze_speed(speed_series):
        clean_s = [s if s is not None else 0 for s in speed_series]
        peak_idx = np.argmax(clean_s)
        series_around = [{"offset": o, "speed": round(clean_s[max(0, min(total_frames-1, event_frame+o))], 4)} for o in range(-45, 11) if o == 0 or abs(o) <= 3 or o % 5 == 0]
        decel = 0
        for i in range(event_frame-1, 0, -1):
            if i+1 < total_frames and clean_s[i] > clean_s[i+1]: decel += 1
            else: break
        return {"peak_speed": round(float(np.max(clean_s)), 4), "peak_frame_offset": int(peak_idx - event_frame), "speed_at_event": round(clean_s[event_frame], 4), "frames_decelerating_before_event": decel, "speed_series_around_event": series_around}

    output["speed_analysis"]["r_wrist"] = analyze_speed(metrics["r_wrist_speed"])
    output["speed_analysis"]["l_wrist"] = analyze_speed(metrics["l_wrist_speed"])

    # Fix 2: Rotation Peak Detection by Velocity (rate of change)
    def find_velocity_peak(series):
        # Calculate velocity (rate of change)
        vel = [abs(series[i] - series[i-1]) for i in range(1, len(series))]
        vel = [0] + vel # pad to match length
        # Search window: event_frame-60 to event_frame+5
        start, end = max(0, event_frame-60), min(total_frames, event_frame+6)
        window = vel[start:end]
        if not window: return event_frame
        return start + np.argmax(window)

    shoulder_p = find_velocity_peak(metrics["shoulder_z_diff"])
    hip_p = find_velocity_peak(metrics["hip_z_diff"])
    
    # Fix 5: x_factor_at_event and trophy
    s_z_ev = metrics["shoulder_z_diff"][event_frame]
    h_z_ev = metrics["hip_z_diff"][event_frame]
    
    output["rotation_analysis"] = {
        "hip_leads_shoulder": hip_p < shoulder_p, "hip_peak_offset": int(hip_p - event_frame), "shoulder_peak_offset": int(shoulder_p - event_frame),
        "x_factor_at_event": round(s_z_ev - h_z_ev, 3),
        "rotation_series": [{"offset": o, "hip_z": metrics["hip_z_diff"][max(0, min(total_frames-1, event_frame+o))], "shoulder_z": metrics["shoulder_z_diff"][max(0, min(total_frames-1, event_frame+o))]} for o in range(-45, 11, 5)]
    }
    if sport_clean in racket_sports:
        tr_idx = max(0, min(total_frames-1, event_frame-40))
        output["rotation_analysis"]["x_factor_at_trophy"] = round(metrics["shoulder_z_diff"][tr_idx] - metrics["hip_z_diff"][tr_idx], 3)

    if sport_clean == "GOLF": output["rotation_analysis"]["x_factor_at_top"] = round(metrics["hip_z_diff"][max(0, min(total_frames-1, event_frame-30))] - metrics["shoulder_z_diff"][max(0, min(total_frames-1, event_frame-30))], 3)

    win = [f[0]['x'] for f in raw_frames[max(0, event_frame-5):min(total_frames, event_frame+5)] if f]
    win_y = [f[0]['y'] for f in raw_frames[max(0, event_frame-5):min(total_frames, event_frame+5)] if f]
    output["balance_stability"] = {
        "nose_x_variance": round(float(np.var(win)), 4) if win else 0, "nose_y_variance": round(float(np.var(win_y)), 4) if win_y else 0,
        "head_drift_before_event": "stable", "feet_grounded_at_event": output["event_snapshot"].get("feet_grounded", False), "heel_rise_detected": False
    }
    return output

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

def draw_neon_skeleton(img, lms, alpha=0.5):
    if not lms: return
    overlay = img.copy()
    # Draw Vectors (Grey Lines)
    for s, e in FULL_SKELETON:
        p1 = (int(lms[s].x*img.shape[1]), int(lms[s].y*img.shape[0]))
        p2 = (int(lms[e].x*img.shape[1]), int(lms[e].y*img.shape[0]))
        cv2.line(overlay, p1, p2, (128, 128, 128), 2, cv2.LINE_AA)
    
    # Draw Points (Red Dots)
    for i in range(len(lms)):
        pt = (int(lms[i].x*img.shape[1]), int(lms[i].y*img.shape[0]))
        cv2.circle(overlay, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)
    
    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

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
st.markdown("<h1>Vector Victor AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-sub'>Deep form Vector and Bio mechanics AI engine</p>", unsafe_allow_html=True)

SPORT_CONFIG = {
    "TENNIS 🎾": ["Serve", "Forehand", "Backhand"], "PADEL 🎾": ["Bandeja", "Vibora", "Smash"],
    "PICKLEBALL 🥒": ["Dink", "Third Shot Drop", "Volley"], "GOLF ⛳": ["Driver", "Iron", "Putter"],
    "BADMINTON 🏸": ["Smash", "Drop Shot", "Clear"], "CRICKET 🏏": ["Drive", "Bowling", "Pull Shot"],
    "GYM 🏋️": ["Squat", "Deadlift", "Bench Press"], "YOGA 🧘": ["Warrior", "Tree Pose", "Sun Salutation"]
}

tabs = st.tabs(list(SPORT_CONFIG.keys()))

for i, (sport, actions) in enumerate(SPORT_CONFIG.items()):
    with tabs[i]:
        # How to Use Section
        st.markdown("""
            <div class="glass-card">
                <div style="color: #ccff00; font-weight: 900; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">🚀 Quick Start Guide</div>
                <div style="font-size: 0.9rem; color: #94a3b8; display: flex; gap: 20px; flex-wrap: wrap;">
                    <span><b>1.</b> Upload Video(s)</span>
                    <span><b>2.</b> Sync Impact Frame (if needed)</span>
                    <span><b>3.</b> Align Stereographic Sliders</span>
                    <span><b>4.</b> Generate Analysis Pack</span>
                    <span><b>5.</b> Upload JSON to Claude/GPT/Gemini for analysis</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 2])
        res_key = f"data_{sport}"
        
        with c1:
            st.info(f"AI ENGINE: {sport}")
            is_stereo = st.toggle("Stereographic Mode", value=False, key=f"st_{sport}")
            u1 = st.file_uploader("Source 1", type=["mp4","mov"], key=f"u1_{sport}")
            u2 = st.file_uploader("Source 2", type=["mp4","mov"], key=f"u2_{sport}") if is_stereo else None
            sel_act = st.selectbox("Action", actions, key=f"act_{sport}")
            if st.button("RUN PRO ANALYSIS", key=f"run_{sport}", width="stretch"):
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
                
                if s['p2']:
                    st.warning("⚠️ **STEREOGRAPHIC SYNC:** Use the sliders below to ensure both views are perfectly aligned on the **Impact Frame** before generating the pack.")
                
                sl1 = st.slider("Source 1 Frame", 0, s['d1']['total']-1, s['d1']['impact'], key=f"sl1_{sport}")
                sl2 = st.slider("Source 2 Frame", 0, (s['d2']['total']-1 if s['d2'] else 0), (s['d2']['impact'] if s['d2'] else 0), key=f"sl2_{sport}") if s['d2'] else 0
                
                # Visual Feedback
                cap1 = cv2.VideoCapture(s['p1']); cap1.set(1, sl1); _, i1 = cap1.read(); cap1.release()
                i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
                
                if s['p2']:
                    cap2 = cv2.VideoCapture(s['p2']); cap2.set(1, sl2); _, i2 = cap2.read(); cap2.release()
                    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2RGB)
                    
                    # 50% Scale of Original Dimensions
                    h1, w1 = i1.shape[:2]
                    h2, w2 = i2.shape[:2]
                    h_target = int(min(h1, h2) * 0.5)
                    
                    w1_new = int(w1 * (h_target / h1))
                    w2_new = int(w2 * (h_target / h2))
                    
                    st.image(np.hstack((cv2.resize(i1, (w1_new, h_target)), cv2.resize(i2, (w2_new, h_target)))), width="stretch")
                else: 
                    h, w = i1.shape[:2]
                    st.image(cv2.resize(i1, (int(w*0.5), int(h*0.5))), width="stretch")

                if st.button("🎬 GENERATE FINAL ANALYSIS PACK", key=f"gen_{sport}", width="stretch"):
                    final_v = render_pro_stereo(s['p1'], s['p2'], s['d1']['history'], (s['d2']['history'] if s['d2'] else []), sl1, sl2, s['d1']['fps'])
                    st.video(final_v)
                    
                    # New Pro Telemetry Generation
                    raw_interp = interpolate_landmarks(s['d1']['raw'])
                    tele_opt = build_pro_telemetry(
                        raw_interp, sport, sel_act, sl1, 
                        s['d1']['fps'], "dual" if s['p2'] else "lead"
                    )
                    
                    # Generate Coaching Brief
                    coaching_brief = generate_brief(tele_opt)
                    
                    z_buf = io.BytesIO()
                    with zipfile.ZipFile(z_buf, "w") as zf:
                        zf.write(final_v, "analysis.mp4")
                        zf.writestr("SHARE_FILE_WITH_AI.txt", coaching_brief)
                    st.download_button("📥 DOWNLOAD REPORT PACK", z_buf.getvalue(), f"{sport}_Report.zip", width="stretch")

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
                    with ch1: st.plotly_chart(plot_power_curve(metrics), width="stretch")
                    with ch2: st.plotly_chart(plot_radar_chart(metrics), width="stretch")
                    
                    RACKET_BAT = ["TENNIS 🎾", "PADEL 🎾", "PICKLEBALL 🥒", "BADMINTON 🏸", "CRICKET 🏏", "GOLF ⛳"]
                    if sport in RACKET_BAT:
                        st.plotly_chart(plot_kinetic_chain(metrics), width="stretch")
