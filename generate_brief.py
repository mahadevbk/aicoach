import json
import sys

def _dominant_side_ids(dominant_side):
    if dominant_side == "left":
        return {
            "dom_prefix": "l_", "nondom_prefix": "r_",
            "dom_side": "left", "nondom_side": "right"
        }
    return {
        "dom_prefix": "r_", "nondom_prefix": "l_",
        "dom_side": "right", "nondom_side": "left"
    }

def _benchmark(sport, action, metric, value):
    sport = sport.upper()
    racket_sports = ["TENNIS", "PICKLEBALL", "BADMINTON", "SQUASH"]
    
    if sport in racket_sports:
        if metric == "dom_elbow":
            if value > 160: return (">160°", "GOOD")
            if value >= 140: return ("140-160°", "WARN")
            return ("<140°", "ISSUE")
        if metric == "dom_knee":
            if value > 150: return (">150°", "GOOD")
            if value >= 130: return ("130-150°", "WARN")
            return ("<130°", "ISSUE")
        if metric == "trunk_forward_lean":
            if 25 <= value <= 40: return ("25-40°", "GOOD")
            if 40 < value <= 55 or 15 <= value < 25: return ("15-25 / 40-55°", "WARN")
            return ("Extreme", "ISSUE")
        if metric == "trunk_lateral_lean":
            if abs(value) < 5: return ("abs <5°", "GOOD")
            if abs(value) <= 15: return ("5-15°", "WARN")
            return (">15°", "ISSUE")
        if metric == "dom_wrist_above_shoulder":
            return ("Above", "GOOD") if value else ("Below", "ISSUE")

    elif sport == "GOLF":
        if metric == "dom_elbow":
            if value > 150: return (">150°", "GOOD")
            if value >= 130: return ("130-150°", "WARN")
            return ("<130°", "ISSUE")
        if metric == "trunk_forward_lean_address":
            return ("25-35°", "GOOD") if 25 <= value <= 35 else ("25-35°", "WARN")
        if metric == "trunk_forward_lean_event":
            return ("20-45°", "GOOD") if 20 <= value <= 45 else ("20-45°", "WARN")
        if metric == "trunk_lateral_lean":
            return ("abs <8°", "GOOD") if abs(value) < 8 else ("abs <8°", "WARN")
        if metric == "x_factor_at_top":
            if value > 0.15: return (">0.15", "GOOD")
            if value >= 0.08: return ("0.08-0.15", "WARN")
            return ("<0.08", "ISSUE")

    elif sport == "GYM":
        is_squat_lunge = any(x in action.lower() for x in ["squat", "lunge"])
        if is_squat_lunge:
            if metric == "dom_knee":
                if value < 90: return ("<90° (Deep)", "GOOD")
                if value <= 120: return ("90-120°", "WARN")
                return (">120°", "ISSUE")
            if metric == "trunk_forward_lean":
                if value < 35: return ("<35°", "GOOD")
                if value <= 45: return ("35-45°", "WARN")
                return (">45° (Safety)", "ISSUE")
            if metric == "trunk_lateral_lean":
                return ("abs <5°", "GOOD") if abs(value) < 5 else ("abs <5°", "WARN")
        
        if "deadlift" in action.lower():
            if metric == "trunk_forward_lean":
                return ("<45°", "GOOD") if value < 45 else ("<45°", "ISSUE")
            if metric == "dom_knee_start":
                return ("80-100°", "GOOD") if 80 <= value <= 100 else ("80-100°", "WARN")
        
        if any(x in action.lower() for x in ["press", "bench", "push"]):
            if metric == "dom_elbow":
                if value < 90: return ("<90° (Full)", "GOOD")
                if value <= 110: return ("90-110°", "WARN")
                return (">110°", "ISSUE")

    elif sport == "YOGA":
        if metric == "trunk_lateral_lean":
            return ("abs <5°", "GOOD") if abs(value) < 5 else ("abs <5°", "WARN")
        if metric == "bilateral_knee_diff":
            if value < 10: return ("<10°", "GOOD")
            if value <= 20: return ("10-20°", "WARN")
            return (">20°", "ISSUE")

    elif sport == "CRICKET":
        if "bowl" in action.lower():
            if metric == "dom_knee":
                return (">150° (Braced)", "GOOD") if value > 150 else (">150°", "WARN")
            if metric == "trunk_forward_lean":
                return ("40-60°", "GOOD") if 40 <= value <= 60 else ("40-60°", "WARN")

    return ("—", "OK")

def _phase_blocks(phase_snapshots, dominant_side):
    side_map = _dominant_side_ids(dominant_side)
    dp, ndp = side_map["dom_prefix"], side_map["nondom_prefix"]
    
    lines = []
    # Phase offsets dictionary mapping
    offsets = {
        "trophy": -40, "swing_start": -15, "follow_through": 20,
        "address": -80, "top": -30, "downswing": -12, "follow": 25,
        "start": -45, "midpoint": -22, "finish": 30,
        "approach": -30, "exit": 30
    }
    
    for name, snap in phase_snapshots.items():
        off = offsets.get(name, "??")
        lines.append(f"### Phase: {name}  (frame offset {off})")
        lines.append(f"    Dominant elbow:       {snap.get(f'{dp}elbow_angle', 0):.1f}°")
        lines.append(f"    Non-dominant elbow:   {snap.get(f'{ndp}elbow_angle', 0):.1f}°")
        lines.append(f"    Dominant knee:        {snap.get(f'{dp}knee_angle', 0):.1f}°")
        lines.append(f"    Non-dominant knee:    {snap.get(f'{ndp}knee_angle', 0):.1f}°")
        lines.append(f"    Trunk forward lean:   {snap.get('trunk_forward_lean', 0):.1f}°")
        lines.append(f"    Trunk lateral lean:   {snap.get('trunk_lateral_lean', 0):.1f}°")
        lines.append(f"    Shoulder z-diff:      {snap.get('shoulder_z_diff', 0):.3f}")
        lines.append(f"    Hip z-diff:           {snap.get('hip_z_diff', 0):.3f}")
        xf = snap.get('hip_shoulder_separation', 0)
        lines.append(f"    X-factor:             {xf:.3f}  [{'GOOD' if xf > 0.10 else 'LOW'}]")
        lines.append(f"    Dom wrist above sh:   {snap.get(f'{dp}wrist_above_{dp}shoulder', False)}")
        lines.append(f"    Feet grounded:        {snap.get('feet_grounded', False)}")
        sw = snap.get('stance_width_ratio')
        lines.append(f"    Stance width ratio:   {sw if sw is not None else 'unreliable'}")
        lines.append("")
    return "\n".join(lines)

def _sport_context(sport, action, dominant_side):
    sport = sport.upper()
    if sport in ["TENNIS", "PICKLEBALL", "BADMINTON", "SQUASH"]:
        return (f"Sport context: {sport} {action}\n"
                f"Event moment represents: ball/shuttle contact.\n"
                f"Dominant side is the racket arm ({dominant_side}).\n"
                f"Phase vocabulary: trophy=loading -40fr, swing_start=-15fr, follow_through=+20fr\n"
                f"Benchmarks: dom elbow >160°, trunk lean 25-40°, dom wrist above shoulder,\n"
                f"hip leads shoulder, x-factor at trophy >0.10")
    
    if sport == "GOLF":
        ctx = (f"Sport context: GOLF {action}\n"
               f"Event moment represents: ball impact.\n"
               f"Phase vocabulary: address=-80fr, top=-30fr, downswing=-12fr, follow=+25fr\n"
               f"Benchmarks: forward lean at address 25-35°, x-factor at top >0.15,\n"
               f"hip leads shoulder, head stability critical (nose variance <0.0005)")
        if any(x in action.lower() for x in ["putt", "chip"]):
            ctx += "\nMinimal body rotation expected — arms-led stroke"
        return ctx

    if sport == "GYM":
        ctx = (f"Sport context: GYM {action}\n"
               f"Event moment represents: peak exertion.\n"
               f"Phase vocabulary: start=-45fr, midpoint=-22fr, finish=+30fr")
        return ctx

    if sport == "YOGA":
        ctx = (f"Sport context: YOGA {action}\n"
               f"Event moment represents: peak hold — deepest point of pose.\n"
               f"Phase vocabulary: approach=-30fr, exit=+30fr\n"
               f"Benchmarks: lateral lean near 0° (alignment), bilateral symmetry key metric")
        if any(x in action.lower() for x in ["warrior-3", "tree", "balance"]):
            ctx += "\nBalance stability is primary metric"
        return ctx

    if sport == "CRICKET":
        if "bowl" in action.lower():
            return (f"Sport context: CRICKET {action}\n"
                    f"Event moment represents: ball release.\n"
                    f"Phase vocabulary: run_up=-60fr, bound=-30fr, front_foot=-12fr, follow=+25fr\n"
                    f"Benchmarks: front knee >150° at release, trunk lean 40-60°, hip leads shoulder")
        return (f"Sport context: CRICKET {action}\n"
                f"Event moment represents: bat-ball contact.\n"
                f"Phase vocabulary: stance=-50fr, backlift=-25fr, follow=+20fr\n"
                f"Benchmarks: dom elbow >140° at contact, head stability critical")

    return f"Sport context: {sport} {action}\nEvent moment: peak of movement."

def _report_instructions(sport, action):
    sport = sport.upper()
    if sport in ["TENNIS", "PICKLEBALL", "BADMINTON", "SQUASH"]:
        return ("Generate a coaching report with these sections:\n"
                "1. Overview (3-4 sentences)\n"
                "2. Overall Scores table: category | score/100 | verdict\n"
                "   Categories: Ball Contact, Loading Position, Arm Extension,\n"
                "   Leg Drive, Racket Speed, Kinetic Chain, Follow-Through\n"
                "3. Key Measurements table: metric | value | benchmark | status\n"
                "4. Phase Analysis: Trophy, Contact, Speed, Leg Drive, Kinetic Chain\n"
                "5. Coaching Recommendations (max 5, by impact): title | priority | detail | drill\n"
                "6. Practice Plan table: drill | focus | reps\n"
                "7. Data Notes\n"
                "Tone: professional, direct. Length: 700-900 words.")
    
    if sport == "GOLF":
        return ("1. Overview  2. Scores (Setup, Backswing, X-Factor, Impact, Speed,\n"
                "Weight Transfer, Follow-Through)  3. Measurements  4. Phase Analysis\n"
                "(Address, Top, Downswing, Impact, Speed, Follow)  5. Recommendations (max 5)\n"
                "6. Drills  7. Data Notes.  Tone: technical but accessible. 700-900 words.")

    if sport == "GYM":
        return ("1. Overview  2. Scores (Range of Motion, Bilateral Symmetry, Spinal Safety,\n"
                "Depth, Tempo, Stability)  3. Measurements  4. Phase Analysis (Start, Mid, Peak,\n"
                "Return)  5. Safety Flags if needed  6. Recommendations (max 4, safety first)\n"
                "7. Progression Plan.  Tone: safety-conscious, encouraging. 500-750 words.")

    if sport == "YOGA":
        return ("1. Overview  2. Scores (Depth, Alignment, Symmetry, Balance, Stability,\n"
                "Progression)  3. Measurements  4. Pose Analysis (Approach, Peak, Exit,\n"
                "Symmetry)  5. Alignment Notes  6. Coaching Cues (max 4, mindful language)\n"
                "7. Progressions.  Tone: calm, mindful. 400-600 words.")

    if sport == "CRICKET":
        return ("1. Overview  2. Scores  3. Measurements  4. Phase Analysis\n"
                "5. Recommendations (max 5)  6. Drills  7. Data Notes.\n"
                "Tone: technical, coach-to-player. 600-800 words.")

    return ("Generate a structured coaching report covering: overview, key\n"
            "measurements, phase analysis, recommendations, and practice drills.")

def generate_brief(oj: dict) -> str:
    sport = oj.get("sport", "UNKNOWN")
    action = oj.get("action", "Action")
    camera = oj.get("camera", "lead")
    meta = oj.get("metadata", {})
    fps = meta.get("fps", 30.0)
    total_frames = meta.get("total_frames", 0)
    duration = total_frames / fps if fps > 0 else 0
    offset = meta.get("offset", 0)
    event_frame = total_frames + offset
    dominant_side = meta.get("dominant_side", "right")
    
    warnings = meta.get("validation_warnings", [])
    quality = "clean"
    if any("interpolated" in w.lower() for w in warnings): quality = "interpolated"
    if any("clamped" in w.lower() or "glitch" in w.lower() for w in warnings): quality = "glitches_clamped"
    
    side_map = _dominant_side_ids(dominant_side)
    dp, ndp = side_map["dom_prefix"], side_map["nondom_prefix"]
    
    ev = oj.get("event_snapshot", {})
    
    def get_b(metric, val): return _benchmark(sport, action, metric, val)
    
    dom_elbow_val = ev.get(f"{dp}elbow_angle", 0)
    dom_elbow_b, dom_elbow_s = get_b("dom_elbow", dom_elbow_val)
    
    dom_knee_val = ev.get(f"{dp}knee_angle", 0)
    dom_knee_b, dom_knee_s = get_b("dom_knee", dom_knee_val)
    
    fwd_lean = ev.get("trunk_forward_lean", 0)
    fwd_b, fwd_s = get_b("trunk_forward_lean", fwd_lean)
    
    lat_lean = ev.get("trunk_lateral_lean", 0)
    lat_b, lat_s = get_b("trunk_lateral_lean", lat_lean)
    
    wrist_above = ev.get(f"{dp}wrist_above_{dp}shoulder", False)
    wa_b, wa_s = get_b("dom_wrist_above_shoulder", wrist_above)

    brief = []
    brief.append("=== VECTOR VICTOR AI - BIO MECHANICAL ANALYSIS ===")
    brief.append("")
    brief.append("## SECTION 1 — IDENTITY")
    brief.append(f"Sport:          {sport}")
    brief.append(f"Action:         {action}")
    brief.append(f"Dominant side:  {dominant_side}")
    brief.append(f"Camera:         {camera}")
    brief.append(f"Capture:        {total_frames} frames @ {fps} fps  ({duration:.1f} seconds)")
    brief.append(f"Event frame:    {event_frame}")
    brief.append(f"Data quality:   {quality}")
    if warnings:
        brief.append(f"Warnings:       {' | '.join(warnings)}")
    
    brief.append("")
    brief.append("## SECTION 2 — SPORT CONTEXT")
    brief.append(_sport_context(sport, action, dominant_side))
    
    brief.append("")
    brief.append(f"## SECTION 3 — EVENT MOMENT  (frame {event_frame})")
    brief.append("")
    brief.append("Joint angles at event:")
    brief.append(f"  Dominant elbow:       {dom_elbow_val:.1f}°    benchmark: {dom_elbow_b}  [{dom_elbow_s}]")
    brief.append(f"  Non-dominant elbow:   {ev.get(f'{ndp}elbow_angle', 0):.1f}°")
    brief.append(f"  Dominant knee:        {dom_knee_val:.1f}°     benchmark: {dom_knee_b}   [{dom_knee_s}]")
    brief.append(f"  Non-dominant knee:    {ev.get(f'{ndp}knee_angle', 0):.1f}°")
    brief.append(f"  Dominant hip:         {ev.get(f'{dp}hip_angle', 0):.1f}°")
    brief.append(f"  Non-dominant hip:     {ev.get(f'{ndp}hip_angle', 0):.1f}°")
    brief.append(f"  Dom shoulder abduct:  {ev.get(f'{dp}shoulder_abduction_angle', 0):.1f}°")
    
    brief.append("")
    brief.append("Trunk at event:")
    brief.append(f"  Forward lean:         {fwd_lean:.1f}°   benchmark: {fwd_b}  [{fwd_s}]")
    brief.append(f"  Lateral lean:         {lat_lean:.1f}°   benchmark: near 0°  [{lat_s}]")
    brief.append(f"  Shoulder tilt:        {ev.get('shoulder_tilt_deg', 0):.1f}°")
    
    brief.append("")
    brief.append("Rotation at event:")
    s_z = ev.get("shoulder_z_diff", 0)
    h_z = ev.get("hip_z_diff", 0)
    xf_ev = ev.get("hip_shoulder_separation", 0)
    brief.append(f"  Shoulder z-diff:      {s_z:.3f}")
    brief.append(f"  Hip z-diff:           {h_z:.3f}")
    brief.append(f"  X-factor:             {xf_ev:.3f}  [{'GOOD' if xf_ev > 0.10 else 'LOW'}]")
    
    brief.append("")
    brief.append("Position flags at event:")
    brief.append(f"  Dom wrist above dom shoulder:    {wrist_above}  [{wa_s}]")
    brief.append(f"  Feet grounded:                   {ev.get('feet_grounded', False)}")
    brief.append(f"  Stance width ratio:              {ev.get('stance_width_ratio') or 0:.3f}")
    
    brief.append("")
    brief.append("## SECTION 4 — PHASE SNAPSHOTS")
    brief.append(_phase_blocks(oj.get("phase_snapshots", {}), dominant_side))
    
    brief.append("")
    brief.append("## SECTION 5 — KINETIC CHAIN")
    rot = oj.get("rotation_analysis", {})
    brief.append("Rotation series (hip and shoulder z-diff, every 5 frames):")
    brief.append("  offset | hip_z  | shoulder_z")
    brief.append("  -------|--------|----------")
    for row in rot.get("rotation_series", []):
        off_val = row['offset']
        suffix = " <- EVENT" if off_val == 0 else ""
        brief.append(f"  {off_val:<6} | {row['hip_z']:<6.3f} | {row['shoulder_z']:<10.3f}{suffix}")
    
    brief.append("")
    brief.append("Sequencing:")
    brief.append(f"  Hip leads shoulder:     {rot.get('hip_leads_shoulder', False)}  [{'GOOD' if rot.get('hip_leads_shoulder') else 'ISSUE'}]")
    brief.append(f"  Hip rotation peak:      frame offset {rot.get('hip_peak_offset', 0)}")
    brief.append(f"  Shoulder rotation peak: frame offset {rot.get('shoulder_peak_offset', 0)}")
    brief.append(f"  Separation gap:         {rot.get('shoulder_peak_offset', 0) - rot.get('hip_peak_offset', 0)} frames")
    xf_tr = rot.get("x_factor_at_trophy", rot.get("x_factor_at_top", 0))
    brief.append(f"  X-factor at trophy/top: {xf_tr:.3f}  [{'GOOD' if xf_tr > 0.10 else 'LOW'}]")
    
    brief.append("")
    brief.append("## SECTION 6 — SPEED PROFILE")
    speed = oj.get("speed_analysis", {}).get(f"{dp}wrist", {})
    brief.append(f"Dominant wrist ({dominant_side}):")
    peak_off = speed.get("peak_frame_offset", 0)
    brief.append(f"  Peak speed (normalised): {speed.get('peak_speed', 0):.3f}  at frame offset {peak_off}")
    brief.append(f"  Speed at event:          {speed.get('speed_at_event', 0):.3f}")
    brief.append(f"  Frames decelerating before event: {speed.get('frames_decelerating_before_event', 0)}")
    
    brief.append("")
    brief.append("Speed series around event:")
    brief.append("  offset |  speed")
    brief.append("  -------|-------")
    for row in speed.get("speed_series_around_event", []):
        off_val = row['offset']
        suffix = ""
        if off_val == peak_off: suffix += " <- PEAK"
        if off_val == 0: suffix += " <- EVENT"
        brief.append(f"  {off_val:<6} | {row['speed']:<7.3f}{suffix}")
        
    brief.append("")
    brief.append("## SECTION 7 — BALANCE & STABILITY")
    bal = oj.get("balance_stability", {})
    nx = bal.get("nose_x_variance", 0)
    brief.append(f"Head tracking:")
    brief.append(f"  X variance:   {nx:.4f}   [{'stable' if nx < 0.0005 else 'drifting'}]")
    brief.append(f"  Y variance:   {bal.get('nose_y_variance', 0):.4f}")
    brief.append(f"  Drift:        {bal.get('head_drift_before_event', 'stable')}")
    brief.append("")
    brief.append("Feet:")
    brief.append(f"  Grounded at event:  {bal.get('feet_grounded_at_event', False)}")
    brief.append(f"  Heel rise detected: {bal.get('heel_rise_detected', False)}")
    
    brief.append("")
    brief.append("## SECTION 8 — REPORT INSTRUCTIONS")
    brief.append(_report_instructions(sport, action))
    
    brief.append("")
    brief.append("=== END OF BRIEF ===")
    
    return "\n".join(brief)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_brief.py <optimised_json_path>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        data = json.load(f)
    brief_text = generate_brief(data)
    print(brief_text)
    out_path = sys.argv[1].replace('.json', '_brief.txt')
    with open(out_path, 'w') as f:
        f.write(brief_text)
    print(f"\nBrief written to {out_path}", file=sys.stderr)
