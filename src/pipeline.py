import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np
import os  # Added missing import

def calculate_3d_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm_product == 0: return 0
    cosine_angle = np.dot(ba, bc) / norm_product
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def run_pipeline(input_path, output_video):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(input_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_height = orig_height + 300 
    
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, new_height))

    # We only cache metrics and landmark results to save RAM
    analysis_cache = []
    max_extension = 0
    peak_velocity_ms = 0
    impact_frame_idx = 0
    
    print("--- Phase 1: Analyzing ---")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        metrics = {"angle": 0, "vel_ms": 0, "torque_m": 0, "raw_wrist": None}
        
        if results.pose_world_landmarks:
            lm = results.pose_world_landmarks.landmark
            angle = calculate_3d_angle([lm[11].x, lm[11].y, lm[11].z], 
                                     [lm[13].x, lm[13].y, lm[13].z], 
                                     [lm[15].x, lm[15].y, lm[15].z])
            
            vel_ms = 0
            if frame_idx > 0 and analysis_cache[frame_idx-1]["raw_wrist"] is not None:
                prev_wrist = analysis_cache[frame_idx-1]["raw_wrist"]
                curr_wrist = np.array([lm[15].x, lm[15].y, lm[15].z])
                vel_ms = np.linalg.norm(curr_wrist - prev_wrist) * fps 
            
            torque_m = abs((lm[11].z - lm[12].z) - (lm[23].z - lm[24].z))
            
            metrics = {
                "angle": angle, 
                "vel_ms": vel_ms, 
                "torque_m": torque_m,
                "raw_wrist": np.array([lm[15].x, lm[15].y, lm[15].z]),
                "landmarks": results.pose_landmarks # Keep screen landmarks for drawing
            }
            
            if vel_ms > peak_velocity_ms:
                peak_velocity_ms = vel_ms
                impact_frame_idx = frame_idx
            if angle > max_extension:
                max_extension = angle

        analysis_cache.append(metrics)
        frame_idx += 1
    
    # Reset video to start for the second pass
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"--- Phase 2: Rendering (Impact Frame: {impact_frame_idx}) ---")
    for i in range(len(analysis_cache)):
        ret, frame = cap.read()
        if not ret: break
        
        metrics = analysis_cache[i]
        canvas = np.zeros((new_height, width, 3), dtype=np.uint8)
        canvas[0:orig_height, 0:width] = frame
        
        if metrics["landmarks"]:
            mp_drawing.draw_landmarks(canvas[0:orig_height, 0:width], metrics["landmarks"], mp_pose.POSE_CONNECTIONS)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_base = orig_height + 60
            col2 = width // 2
            
            cv2.putText(canvas, f"WRIST SPEED: {metrics['vel_ms']:.2f} m/s", (30, y_base + 10), font, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, f"TORQUE DEPTH: {metrics['torque_m']:.3f} m", (30, y_base + 60), font, 0.8, (255, 255, 255), 2)
            
            if i >= impact_frame_idx:
                cv2.putText(canvas, f"PEAK VELOCITY: {peak_velocity_ms:.2f} m/s", (col2, y_base + 10), font, 0.8, (0, 255, 0), 2)
                eff_score = (metrics['angle'] / 180 * 50) + (peak_velocity_ms / 30 * 50)
                cv2.putText(canvas, f"EFFICIENCY: {min(100, eff_score):.1f}%", (col2, y_base + 60), font, 0.8, (0, 255, 255), 2)

        out.write(canvas)

    cap.release()
    out.release()
    print("Metric Pipeline Complete.")

if __name__ == "__main__":
    run_pipeline("data/test_swing.mp4", "output/pro_analysis.mp4")
