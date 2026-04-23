# action_aware_generate_brief.py
# Enhanced brief generation with full sport and action awareness

import numpy as np

def get_action_phases(sport, action):
    """
    Returns phase definitions based on sport and specific action.
    This replaces hardcoded phase logic.
    """
    
    sport = sport.upper() if sport else "GENERAL"
    action = action.upper() if action else "MOVEMENT"
    
    # ===== TENNIS =====
    if sport == "TENNIS":
        if action in ["SERVE", "FIRST_SERVE", "SECOND_SERVE"]:
            return {
                "phases": [("trophy", -40), ("swing_start", -15), ("contact", 0), ("follow_through", 20)],
                "event_description": "ball contact during serve motion"
            }
        elif action in ["RALLY", "FOREHAND", "BACKHAND", "GROUNDSTROKE"]:
            return {
                "phases": [("ready", -30), ("loading", -15), ("contact", 0), ("recovery", 15)],
                "event_description": "ball contact during rally shot"
            }
        elif action in ["VOLLEY", "NET_PLAY"]:
            return {
                "phases": [("split_step", -8), ("contact", 0), ("recovery", 5)],
                "event_description": "ball contact at net"
            }
        elif action in ["OVERHEAD", "SMASH"]:
            return {
                "phases": [("preparation", -25), ("peak_height", -10), ("contact", 0), ("recovery", 12)],
                "event_description": "overhead shot contact"
            }
        elif action == "RETURN":
            return {
                "phases": [("ready", -20), ("loading", -8), ("contact", 0), ("follow", 10)],
                "event_description": "serve return contact"
            }
    
    # ===== GOLF =====
    elif sport == "GOLF":
        if action in ["DRIVE", "DRIVER"]:
            return {
                "phases": [("address", -80), ("top", -30), ("downswing", -12), ("impact", 0), ("follow", 25)],
                "event_description": "ball impact during drive"
            }
        elif action in ["IRON", "MID_IRON", "LONG_IRON"]:
            return {
                "phases": [("address", -60), ("top", -25), ("downswing", -8), ("impact", 0), ("follow", 20)],
                "event_description": "ball impact during iron shot"
            }
        elif action in ["CHIP", "PITCH"]:
            return {
                "phases": [("address", -20), ("peak", -8), ("impact", 0), ("finish", 10)],
                "event_description": "ball impact during chip/pitch"
            }
        elif action in ["PUTT"]:
            return {
                "phases": [("address", -10), ("backswing", -5), ("impact", 0), ("follow", 3)],
                "event_description": "ball impact during putt"
            }
    
    # ===== RACKET SPORTS (Generic) =====
    elif sport in ["BADMINTON", "SQUASH", "PADEL", "PICKLEBALL"]:
        if action in ["SERVE", "SERVICE"]:
            return {
                "phases": [("ready", -30), ("loading", -15), ("contact", 0), ("follow", 15)],
                "event_description": "racket contact during serve"
            }
        elif action in ["RALLY", "STROKE", "SHOT"]:
            return {
                "phases": [("ready", -25), ("loading", -12), ("contact", 0), ("recovery", 12)],
                "event_description": "racket contact during rally"
            }
    
    # ===== SOCCER/FOOTBALL =====
    elif sport == "SOCCER":
        if action in ["KICK", "SHOT", "PASS"]:
            return {
                "phases": [("approach", -15), ("plant_foot", -5), ("contact", 0), ("follow_through", 10)],
                "event_description": "ball contact during kick"
            }
        elif action in ["HEADER"]:
            return {
                "phases": [("approach", -20), ("timing", -5), ("contact", 0), ("recovery", 8)],
                "event_description": "head contact with ball"
            }
    
    # ===== BASEBALL =====
    elif sport == "BASEBALL":
        if action in ["PITCH", "THROW"]:
            return {
                "phases": [("leg_lift", -40), ("stride", -15), ("release", 0), ("follow_through", 20)],
                "event_description": "ball release during pitch"
            }
        elif action in ["SWING", "HIT", "BATTING"]:
            return {
                "phases": [("ready", -30), ("load", -15), ("contact", 0), ("follow_through", 20)],
                "event_description": "bat contact with ball"
            }
    
    # ===== BASKETBALL =====
    elif sport == "BASKETBALL":
        if action in ["SHOOT", "THREE_POINTER", "LAY_UP"]:
            return {
                "phases": [("ready", -30), ("loading", -15), ("release", 0), ("follow_through", 8)],
                "event_description": "ball release during shot"
            }
        elif action in ["PASS"]:
            return {
                "phases": [("ready", -10), ("loading", -5), ("release", 0), ("recovery", 3)],
                "event_description": "ball release during pass"
            }
    
    # ===== GYMNASTICS/GYM =====
    elif sport == "GYM":
        if action in ["SQUAT", "BACK_SQUAT", "FRONT_SQUAT"]:
            return {
                "phases": [("start", -45), ("descent", -25), ("bottom", -5), ("ascent", 15), ("lockout", 30)],
                "event_description": "deepest squat position"
            }
        elif action in ["DEADLIFT"]:
            return {
                "phases": [("setup", -40), ("pull", -15), ("lockout", 0), ("descent", 20)],
                "event_description": "complete lockout position"
            }
        elif action in ["BENCH_PRESS"]:
            return {
                "phases": [("start", -35), ("descent", -15), ("bottom", -5), ("ascent", 20), ("lockout", 30)],
                "event_description": "chest depth position"
            }
    
    # ===== ATHLETICS/RUNNING =====
    elif sport == "ATHLETICS/RUNNING":
        if action in ["SPRINT", "RUNNING", "ACCELERATION"]:
            return {
                "phases": [("drive", -5), ("extension", 0), ("recovery", 8), ("swing", 15)],
                "event_description": "ground contact during running"
            }
        elif action in ["JUMP", "LONG_JUMP", "HIGH_JUMP"]:
            return {
                "phases": [("approach", -20), ("takeoff", -5), ("flight", 5), ("landing", 15)],
                "event_description": "takeoff moment"
            }
    
    # ===== BOXING/MMA =====
    elif sport == "BOXING/MMA":
        if action in ["JAB", "CROSS", "PUNCH", "STRIKE"]:
            return {
                "phases": [("load", -10), ("acceleration", -5), ("impact", 0), ("recoil", 8)],
                "event_description": "punch impact"
            }
        elif action in ["KICK"]:
            return {
                "phases": [("chamber", -15), ("extension", -5), ("impact", 0), ("recovery", 12)],
                "event_description": "kick impact"
            }
    
    # ===== HOCKEY =====
    elif sport in ["ICE_HOCKEY", "FIELD_HOCKEY"]:
        if action in ["SHOT", "SLAPSHOT", "WRISTSHOT"]:
            return {
                "phases": [("backswing", -12), ("contact", 0), ("follow_through", 10)],
                "event_description": "stick contact with puck/ball"
            }
    
    # ===== YOGA =====
    elif sport == "YOGA":
        if action in ["POSE", "ASANA", "HOLD"]:
            return {
                "phases": [("approach", -30), ("entry", -10), ("hold", 10), ("exit", 30)],
                "event_description": "peak pose position"
            }
    
    # ===== MARTIAL ARTS =====
    elif sport == "MARTIAL_ARTS":
        if action in ["KICK", "STRIKE", "ATTACK"]:
            return {
                "phases": [("chamber", -12), ("extension", -3), ("impact", 0), ("reset", 8)],
                "event_description": "strike impact"
            }
        elif action in ["BLOCK", "DEFENSE"]:
            return {
                "phases": [("ready", -8), ("response", -2), ("contact", 0), ("recovery", 5)],
                "event_description": "defensive block"
            }
    
    # ===== FALLBACK =====
    else:
        return {
            "phases": [("start", -30), ("midpoint", -10), ("contact", 0), ("finish", 20)],
            "event_description": "movement event"
        }


def get_action_benchmarks(sport, action):
    """
    Returns action-specific performance benchmarks.
    """
    
    sport = sport.upper() if sport else "GENERAL"
    action = action.upper() if action else "MOVEMENT"
    
    # ===== TENNIS BENCHMARKS =====
    if sport == "TENNIS":
        if action in ["SERVE", "FIRST_SERVE", "SECOND_SERVE"]:
            return {
                "dominant_elbow_at_contact": (140, 160, "degrees"),
                "dominant_knee_at_contact": (100, 120, "degrees"),
                "trunk_rotation_at_trophy": (45, 65, "degrees"),
                "hip_shoulder_separation_at_trophy": (15, 35, "degrees"),
                "dominant_wrist_speed": (80, 130, "normalized"),
                "feet_grounded_at_contact": (False, "flag"),
                "x_factor_at_trophy": (0.15, 0.35, "ratio"),
            }
        elif action in ["RALLY", "FOREHAND", "BACKHAND"]:
            return {
                "dominant_elbow_at_contact": (90, 110, "degrees"),
                "dominant_knee_at_contact": (120, 140, "degrees"),
                "trunk_rotation_at_contact": (30, 50, "degrees"),
                "hip_shoulder_separation_at_contact": (5, 20, "degrees"),
                "dominant_wrist_speed": (40, 80, "normalized"),
                "stability_at_contact": (0.7, 1.0, "score"),
                "balance_maintained": (True, "flag"),
            }
        elif action in ["VOLLEY"]:
            return {
                "dominant_elbow_at_contact": (60, 90, "degrees"),
                "reaction_time": (0.3, 0.5, "seconds"),
                "stability_score": (0.8, 1.0, "score"),
                "compact_swing": (True, "flag"),
                "dominant_wrist_speed": (20, 50, "normalized"),
            }
        elif action in ["OVERHEAD"]:
            return {
                "shoulder_extension": (140, 170, "degrees"),
                "dominant_elbow_at_contact": (160, 180, "degrees"),
                "trunk_forward_lean": (30, 50, "degrees"),
                "dominant_wrist_speed": (60, 100, "normalized"),
                "jump_detected": (True, "flag"),
            }
    
    # ===== GOLF BENCHMARKS =====
    elif sport == "GOLF":
        if action in ["DRIVE", "DRIVER"]:
            return {
                "hip_rotation_at_top": (45, 65, "degrees"),
                "shoulder_rotation_at_top": (85, 105, "degrees"),
                "hip_shoulder_separation": (30, 50, "degrees"),
                "weight_transfer": (0.7, 1.0, "ratio"),
                "swing_tempo": (1.5, 2.5, "seconds"),
            }
        elif action in ["CHIP", "PITCH"]:
            return {
                "swing_arc": (30, 60, "degrees"),
                "rhythm_consistency": (0.8, 1.0, "score"),
                "distance_control": (True, "flag"),
            }
    
    # ===== GYM BENCHMARKS =====
    elif sport == "GYM":
        if action in ["SQUAT"]:
            return {
                "bilateral_symmetry": (0.85, 1.0, "ratio"),
                "knee_angle_at_bottom": (80, 110, "degrees"),
                "trunk_forward_lean": (10, 30, "degrees"),
                "depth_consistency": (0.9, 1.0, "ratio"),
                "stability_score": (0.8, 1.0, "score"),
            }
        elif action in ["DEADLIFT"]:
            return {
                "back_alignment": (0.85, 1.0, "score"),
                "hip_shoulder_alignment": (-5, 5, "degrees"),
                "weight_distribution": (0.9, 1.0, "ratio"),
                "lockout_stability": (0.9, 1.0, "score"),
            }
    
    # ===== FALLBACK =====
    else:
        return {}


def get_event_description(sport, action):
    """Returns a description of what the event moment represents."""
    
    sport = sport.upper() if sport else "GENERAL"
    action = action.upper() if action else "MOVEMENT"
    
    descriptions = {
        ("TENNIS", "SERVE"): "ball contact during serve motion",
        ("TENNIS", "RALLY"): "ball contact during rally shot",
        ("TENNIS", "VOLLEY"): "ball contact at net",
        ("TENNIS", "OVERHEAD"): "overhead shot contact",
        ("GOLF", "DRIVE"): "ball impact during drive",
        ("GOLF", "CHIP"): "ball impact during chip shot",
        ("GYM", "SQUAT"): "deepest squat position",
        ("GYM", "DEADLIFT"): "complete lockout position",
        ("SOCCER", "KICK"): "ball contact during kick",
        ("BASEBALL", "PITCH"): "ball release during pitch",
        ("BASKETBALL", "SHOOT"): "ball release during shot",
    }
    
    return descriptions.get((sport, action), f"{action.lower()} event moment")


def generate_brief(tele_opt, sport="GENERAL", action="MOVEMENT"):
    """
    Generate an intelligent, action-aware brief for AI coaching analysis.
    
    Args:
        tele_opt: Telemetry object with computed metrics
        sport: Sport name (TENNIS, GOLF, GYM, etc)
        action: Action type (SERVE, RALLY, SQUAT, etc)
    
    Returns:
        Formatted brief string with sport and action context
    """
    
    # Normalize inputs
    sport = sport.upper() if sport else "GENERAL"
    action = action.upper() if action else "MOVEMENT"
    
    # Get action-aware phase definitions
    phase_info = get_action_phases(sport, action)
    phases = phase_info["phases"]
    event_description = phase_info["event_description"]
    
    # Get action-aware benchmarks
    benchmarks = get_action_benchmarks(sport, action)
    
    # Extract telemetry data
    metadata = tele_opt.get("metadata", {})
    detected_actions = metadata.get("detected_actions", [])
    action_frames_str = ", ".join([f"{a['action']}@f{a['frame']}" for a in detected_actions]) if detected_actions else "none"

    metrics = tele_opt.get("metrics", {})
    event_snapshot = tele_opt.get("event_snapshot", {})
    phase_snapshots = tele_opt.get("phase_snapshots", {})
    speed_analysis = tele_opt.get("speed_analysis", {})
    rotation_analysis = tele_opt.get("rotation_analysis", {})
    bilateral = tele_opt.get("bilateral_analysis", {})
    rom = tele_opt.get("rom_analysis", {})
    smoothness = tele_opt.get("smoothness_analysis", {})
    
    # Build brief
    brief = f"""=== VECTOR VICTOR AI - BIOMECHANICAL ANALYSIS ===

## SECTION 1 — IDENTITY
Sport:          {sport}
Action:         {action}
Dominant side:  {metadata.get('dominant_side', 'unknown')}
Camera:         {tele_opt.get('camera', 'unknown')}
Capture:        {metadata.get('total_frames', 0)} frames @ {metadata.get('fps', 0)} fps
Event frame:    {metadata.get('total_frames', 0)}
Data quality:   clean

## SECTION 2 — ACTION CONTEXT
Sport context:  {sport} {action}
Event moment represents: {event_description}
Detected key actions: {action_frames_str}
Dominant side is the primary limb (arm/leg)
Phase vocabulary: {', '.join([f'{name}={offset}fr' for name, offset in phases])}
Benchmarks: {', '.join([f'{k} {v[0]}-{v[1]}{v[2]}' for k, v in list(benchmarks.items())[:4]]) if benchmarks else 'standard'}

## SECTION 3 — EVENT MOMENT (frame {metadata.get('total_frames', 0)})

Joint angles at event:
  Dominant elbow:       {event_snapshot.get('r_elbow_angle', 'N/A')}°
  Non-dominant elbow:   {event_snapshot.get('l_elbow_angle', 'N/A')}°
  Dominant knee:        {event_snapshot.get('r_knee_angle', 'N/A')}°
  Non-dominant knee:    {event_snapshot.get('l_knee_angle', 'N/A')}°
  Dominant hip:         {event_snapshot.get('r_hip_angle', 'N/A')}°
  Non-dominant hip:     {event_snapshot.get('l_hip_angle', 'N/A')}°

Trunk at event:
  Forward lean:         {event_snapshot.get('trunk_forward_lean', 'N/A')}°
  Lateral lean:         {event_snapshot.get('trunk_lateral_lean', 'N/A')}°
  Shoulder tilt:        {event_snapshot.get('shoulder_tilt_deg', 'N/A')}°

Rotation at event:
  Shoulder z-diff:      {event_snapshot.get('shoulder_z_diff', 'N/A')}
  Hip z-diff:           {event_snapshot.get('hip_z_diff', 'N/A')}
  X-factor:             {rotation_analysis.get('x_factor_at_event', 'N/A')}

Position flags at event:
  Dom wrist above dom shoulder:    {event_snapshot.get('r_wrist_above_r_shoulder', 'N/A')}
  Feet grounded:                   {event_snapshot.get('feet_grounded', 'N/A')}
  Stance width ratio:              {event_snapshot.get('stance_width_ratio', 'N/A')}

## SECTION 4 — PHASE SNAPSHOTS

"""
    
    for phase_name, phase_offset in phases:
        if phase_name in phase_snapshots:
            snap = phase_snapshots[phase_name]
            brief += f"""### Phase: {phase_name}  (frame offset {phase_offset})
    Dominant elbow:       {snap.get('r_elbow_angle', 'N/A')}°
    Non-dominant elbow:   {snap.get('l_elbow_angle', 'N/A')}°
    Dominant knee:        {snap.get('r_knee_angle', 'N/A')}°
    Trunk forward lean:   {snap.get('trunk_forward_lean', 'N/A')}°
    Trunk lateral lean:   {snap.get('trunk_lateral_lean', 'N/A')}°

"""
    
    brief += f"""## SECTION 5 — KINETIC CHAIN
Rotation analysis:
  Hip leads shoulder:     {rotation_analysis.get('hip_leads_shoulder', 'N/A')}
  Hip rotation peak:      frame offset {rotation_analysis.get('hip_peak_offset', 'N/A')}
  Shoulder rotation peak: frame offset {rotation_analysis.get('shoulder_peak_offset', 'N/A')}
  X-factor at event:      {rotation_analysis.get('x_factor_at_event', 'N/A')}

## SECTION 6 — SPEED PROFILE
Dominant wrist:
  Peak speed (normalised): {speed_analysis.get('r_wrist', {}).get('peak_speed', 'N/A')}
  Speed at event:          {speed_analysis.get('r_wrist', {}).get('speed_at_event', 'N/A')}

Non-dominant wrist:
  Peak speed (normalised): {speed_analysis.get('l_wrist', {}).get('peak_speed', 'N/A')}
  Speed at event:          {speed_analysis.get('l_wrist', {}).get('speed_at_event', 'N/A')}

Arm coordination:
  Sync offset: {speed_analysis.get('arm_coordination', {}).get('sync_offset_frames', 'N/A')} frames
  Coordination: {speed_analysis.get('arm_coordination', {}).get('coordination_assessment', 'N/A')}

## SECTION 7 — BILATERAL ANALYSIS
"""
    
    for joint, data in bilateral.items():
        brief += f"""
{joint}:
  Right mean: {data.get('right_mean', 'N/A')} | Left mean: {data.get('left_mean', 'N/A')}
  Asymmetry: {data.get('absolute_difference', 'N/A')} ({data.get('percent_difference', 'N/A')}%)
  Concern: {data.get('concern_level', 'N/A')}
"""
    
    brief += f"""

## SECTION 8 — RANGE OF MOTION

"""
    
    for metric, data in rom.items():
        brief += f"""
{metric}:
  Min: {data.get('minimum', 'N/A')} | Max: {data.get('maximum', 'N/A')}
  ROM: {data.get('range_of_motion', 'N/A')} | Quality: {data.get('quality_assessment', 'N/A')}
"""
    
    brief += f"""

## SECTION 9 — MOVEMENT SMOOTHNESS

"""
    
    for joint, data in smoothness.items():
        brief += f"""
{joint}:
  Smoothness: {data.get('smoothness_level', 'N/A')} (score: {data.get('smoothness_score', 'N/A')})
  Peak acceleration: {data.get('peak_acceleration', 'N/A')}
"""
    
    brief += f"""

## SECTION 10 — REPORT INSTRUCTIONS
Generate a coaching report with these sections:
1. Overview (2-3 sentences specific to {action.lower()})
2. Experts Coach's review of the athletes performance. The good and the weaknesses. Be encouraging and positive.
3. Coaching Recommendations (focused on {action.lower()} technique)
4. Practice Plan table
5. Overall Scores table
6. Risk assessments
7. Key Measurements table
8. {action}-Specific Phase Analysis
9. Data Notes
10. No player name has been provided in this report.

Tone: Encouraging, professional, direct, action-specific. Length: 700-900 words.

=== END OF BRIEF ===
"""
    
    return brief


if __name__ == "__main__":
    # Example usage
    sample_telo = {
        "sport": "TENNIS",
        "action": "RALLY",
        "metadata": {"dominant_side": "right", "fps": 30, "total_frames": 150},
        "metrics": {},
        "event_snapshot": {},
        "phase_snapshots": {},
        "speed_analysis": {},
        "rotation_analysis": {},
        "bilateral_analysis": {},
        "rom_analysis": {},
        "smoothness_analysis": {},
    }
    
    brief = generate_brief(sample_telo, "TENNIS", "RALLY")
    print(brief)
