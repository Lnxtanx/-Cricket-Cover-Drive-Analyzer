import cv2
import numpy as np
import time
import json
import os
import logging
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

# Try ultralytics first, fallback to basic tracking if not available
try:
    from ultralytics import YOLO
    POSE_MODEL_AVAILABLE = True
except ImportError:
    POSE_MODEL_AVAILABLE = False
    print("Warning: Ultralytics not available, using basic pose tracking")

from config import *

# ===============================
# Setup Logging
# ===============================
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["file"]),
        logging.StreamHandler() if LOGGING_CONFIG["console_output"] else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===============================
# Data Classes (same as before)
# ===============================
@dataclass
class PhaseInfo:
    name: str
    start_frame: int
    end_frame: int
    duration: int
    key_metrics: Dict

@dataclass
class ContactMoment:
    frame_number: int
    confidence: float
    wrist_velocity: float
    elbow_acceleration: float

@dataclass
class AnalysisResult:
    phases: List[PhaseInfo]
    contact_moment: Optional[ContactMoment]
    smoothness_metrics: Dict
    skill_grade: str
    overall_score: float
    reference_deviations: Dict
    performance_stats: Dict

# ===============================
# Enhanced Pose Analysis Class with Ultralytics
# ===============================
class EnhancedCoverDriveAnalyzer:
    def __init__(self):
        if POSE_MODEL_AVAILABLE:
            try:
                self.pose_model = YOLO('yolo11n-pose.pt')  # Lightweight pose model
                self.use_yolo = True
                logger.info("Using YOLO pose estimation")
            except Exception as e:
                logger.warning(f"YOLO model failed to load: {e}, using basic tracking")
                self.use_yolo = False
        else:
            self.use_yolo = False
            logger.info("Using basic pose tracking (no YOLO)")
        
        # Tracking variables
        self.frame_buffer = deque(maxlen=VIDEO_CONFIG["buffer_size"])
        self.velocity_history = deque(maxlen=PHASE_DETECTION["smoothing_window"])
        self.angle_history = deque(maxlen=50)
        self.performance_stats = {
            "fps_history": [], 
            "processing_times": [],
            "current_fps": 0,
            "target_fps": PERFORMANCE_CONFIG["fps_target"]
        }
        
        # Bat tracking variables
        self.bat_positions = deque(maxlen=BAT_DETECTION["swing_smoothing_window"])
        self.swing_path = []
        self.bat_detected_frames = 0
        
        # Performance optimization
        self.frame_skip_count = 0
        self.optimization_level = 0  # 0=full quality, 1=medium, 2=fast
    
    def extract_pose_keypoints(self, frame):
        """Extract pose keypoints using YOLO or basic tracking."""
        if self.use_yolo:
            try:
                results = self.pose_model(frame, verbose=False)
                if results and len(results) > 0 and results[0].keypoints is not None:
                    # Get the first person's keypoints
                    keypoints = results[0].keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)
                    confidences = results[0].keypoints.conf[0].cpu().numpy()  # Shape: (17,)
                    
                    # Convert to landmark-like format
                    landmarks = []
                    for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
                        # Create landmark-like object
                        landmark = type('Landmark', (), {
                            'x': kp[0] / frame.shape[1],  # Normalize to 0-1
                            'y': kp[1] / frame.shape[0],  # Normalize to 0-1
                            'visibility': conf
                        })()
                        landmarks.append(landmark)
                    
                    return landmarks
            except Exception as e:
                logger.warning(f"YOLO pose estimation failed: {e}")
        
        # Fallback: create dummy landmarks for basic functionality
        return self._create_dummy_landmarks(frame)
    
    def _create_dummy_landmarks(self, frame):
        """Create dummy landmarks for basic functionality when pose detection fails."""
        # Create 17 dummy landmarks (YOLO pose format)
        landmarks = []
        h, w = frame.shape[:2]
        
        # Rough body position estimates (center of frame)
        center_x, center_y = w // 2, h // 2
        
        dummy_positions = [
            (center_x, center_y - 100),  # nose
            (center_x - 20, center_y - 80), (center_x + 20, center_y - 80),  # eyes
            (center_x - 30, center_y - 60), (center_x + 30, center_y - 60),  # ears
            (center_x - 80, center_y), (center_x + 80, center_y),  # shoulders
            (center_x - 100, center_y + 50), (center_x + 100, center_y + 50),  # elbows
            (center_x - 120, center_y + 100), (center_x + 120, center_y + 100),  # wrists
            (center_x - 40, center_y + 150), (center_x + 40, center_y + 150),  # hips
            (center_x - 50, center_y + 250), (center_x + 50, center_y + 250),  # knees
            (center_x - 60, center_y + 350), (center_x + 60, center_y + 350),  # ankles
        ]
        
        for pos in dummy_positions:
            landmark = type('Landmark', (), {
                'x': pos[0] / w,
                'y': pos[1] / h,
                'visibility': 0.3  # Low confidence for dummy data
            })()
            landmarks.append(landmark)
        
        return landmarks
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    def calculate_velocity(self, point1, point2, dt=1):
        """Calculate velocity between two points."""
        if point1 is None or point2 is None:
            return 0
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.sqrt(dx*dx + dy*dy) / dt
    
    def check_performance_and_optimize(self, current_fps):
        """Monitor performance and auto-optimize if needed."""
        self.performance_stats["current_fps"] = current_fps
        
        if PERFORMANCE_CONFIG["auto_optimize"] and len(self.performance_stats["fps_history"]) > 10:
            avg_fps = np.mean(self.performance_stats["fps_history"][-10:])
            target_fps = PERFORMANCE_CONFIG["fps_target"]
            
            # Auto-optimize if FPS is below target
            if avg_fps < target_fps * 0.8:  # 80% of target
                if self.optimization_level < 2:
                    self.optimization_level += 1
                    logger.warning(f"Performance optimization level increased to {self.optimization_level} (FPS: {avg_fps:.1f})")
            elif avg_fps > target_fps * 1.2:  # 120% of target
                if self.optimization_level > 0:
                    self.optimization_level -= 1
                    logger.info(f"Performance optimization level decreased to {self.optimization_level} (FPS: {avg_fps:.1f})")
        
        return self.optimization_level
    
    def should_skip_frame(self, frame_count):
        """Determine if frame should be skipped for performance."""
        if self.optimization_level == 0:
            return False
        elif self.optimization_level == 1:
            return frame_count % 2 == 0  # Skip every other frame
        else:  # optimization_level == 2
            return frame_count % 3 != 0  # Process every 3rd frame only
    
    def detect_bat_basic(self, frame):
        """Basic bat detection using color and shape analysis."""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks for different bat colors
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for color_name, color_range in BAT_DETECTION["color_ranges"].items():
            lower = np.array(color_range["lower"])
            upper = np.array(color_range["upper"])
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours for bat-like shapes
        bat_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if BAT_DETECTION["min_contour_area"] < area < BAT_DETECTION["max_contour_area"]:
                # Get bounding rectangle
                rect = cv2.minAreaRect(contour)
                (center_x, center_y), (width, height), angle = rect
                
                # Check aspect ratio (bat should be elongated)
                if width > 0 and height > 0:
                    aspect_ratio = min(width, height) / max(width, height)
                    length = max(width, height)
                    
                    if (BAT_DETECTION["aspect_ratio_range"][0] < aspect_ratio < BAT_DETECTION["aspect_ratio_range"][1] and
                        length > BAT_DETECTION["length_threshold"]):
                        
                        bat_candidates.append({
                            "contour": contour,
                            "center": (int(center_x), int(center_y)),
                            "angle": angle,
                            "length": length,
                            "width": min(width, height),
                            "rect": rect
                        })
        
        # Select best bat candidate (largest valid contour)
        if bat_candidates:
            best_bat = max(bat_candidates, key=lambda x: x["length"])
            self.bat_detected_frames += 1
            
            # Track bat position for swing path
            if BAT_DETECTION["track_swing_path"]:
                self.bat_positions.append({
                    "center": best_bat["center"],
                    "angle": best_bat["angle"],
                    "timestamp": len(self.swing_path)
                })
            
            return [best_bat]
        
        return []
    
    def extract_enhanced_metrics(self, landmarks, frame_width, frame_height, prev_landmarks=None):
        """Extract comprehensive cricket-specific metrics from pose landmarks."""
        # YOLO pose keypoint indices (different from MediaPipe)
        YOLO_KEYPOINTS = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
        # Convert normalized landmarks to pixel coordinates
        coords = lambda idx: (
            int(landmarks[idx].x * frame_width), 
            int(landmarks[idx].y * frame_height)
        ) if idx < len(landmarks) and landmarks[idx].visibility > 0.3 else None
        
        # Key body points for cricket analysis
        left_shoulder = coords(YOLO_KEYPOINTS['left_shoulder'])
        right_shoulder = coords(YOLO_KEYPOINTS['right_shoulder'])
        left_elbow = coords(YOLO_KEYPOINTS['left_elbow'])
        right_elbow = coords(YOLO_KEYPOINTS['right_elbow'])
        left_wrist = coords(YOLO_KEYPOINTS['left_wrist'])
        right_wrist = coords(YOLO_KEYPOINTS['right_wrist'])
        left_hip = coords(YOLO_KEYPOINTS['left_hip'])
        right_hip = coords(YOLO_KEYPOINTS['right_hip'])
        left_knee = coords(YOLO_KEYPOINTS['left_knee'])
        right_knee = coords(YOLO_KEYPOINTS['right_knee'])
        left_ankle = coords(YOLO_KEYPOINTS['left_ankle'])
        right_ankle = coords(YOLO_KEYPOINTS['right_ankle'])
        nose = coords(YOLO_KEYPOINTS['nose'])
        
        # Calculate cricket metrics (same logic as before)
        head_steady = True
        head_knee_alignment = 0
        if nose and left_knee:
            head_knee_alignment = abs(nose[0] - left_knee[0])
        
        shoulder_tilt = 0
        shoulder_hip_alignment = True
        if left_shoulder and right_shoulder and left_hip and right_hip:
            shoulder_slope = (right_shoulder[1] - left_shoulder[1]) / (right_shoulder[0] - left_shoulder[0] + 1e-6)
            shoulder_tilt = math.degrees(math.atan(shoulder_slope))
            
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
            alignment_diff = abs(shoulder_center[0] - hip_center[0])
            shoulder_hip_alignment = alignment_diff < 20
        
        front_elbow_angle = 0
        front_elbow_elevated = False
        if left_shoulder and left_elbow and left_wrist:
            front_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            front_elbow_elevated = left_elbow[1] < left_shoulder[1]
        
        back_elbow_angle = 0
        if right_shoulder and right_elbow and right_wrist:
            back_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        wrist_velocity = 0
        if left_wrist and prev_landmarks:
            prev_wrist = coords(YOLO_KEYPOINTS['left_wrist']) if prev_landmarks else None
            if prev_wrist:
                wrist_velocity = self.calculate_velocity(prev_wrist, left_wrist)
        
        hip_rotation = 0
        if left_hip and right_hip:
            hip_slope = (right_hip[1] - left_hip[1]) / (right_hip[0] - left_hip[0] + 1e-6)
            hip_rotation = math.degrees(math.atan(hip_slope))
        
        front_knee_bend = 0
        front_knee_alignment = True
        if left_hip and left_knee and left_ankle:
            front_knee_bend = self.calculate_angle(left_hip, left_knee, left_ankle)
            knee_ankle_alignment = abs(left_knee[0] - left_ankle[0])
            front_knee_alignment = knee_ankle_alignment < 15
        
        front_foot_direction = 0
        foot_spread = 0
        if left_ankle and right_ankle:
            foot_spread = abs(left_ankle[0] - right_ankle[0])
            if left_knee and left_ankle:
                foot_slope = (left_knee[1] - left_ankle[1]) / (left_knee[0] - left_ankle[0] + 1e-6)
                front_foot_direction = math.degrees(math.atan(foot_slope))
        
        spine_lean = 0
        if left_hip and left_shoulder:
            spine_slope = (left_shoulder[1] - left_hip[1]) / (left_shoulder[0] - left_hip[0] + 1e-6)
            spine_lean = abs(90 - math.degrees(math.atan(spine_slope)))
        
        balance_score = 0.5
        if nose and left_ankle:
            head_foot_alignment = abs(nose[0] - left_ankle[0])
            balance_score = max(0, 1 - (head_foot_alignment / 100))
        
        metrics = {
            "head_steady": head_steady,
            "head_knee_alignment": head_knee_alignment,
            "shoulder_tilt": shoulder_tilt,
            "shoulder_hip_alignment": shoulder_hip_alignment,
            "front_elbow_angle": front_elbow_angle,
            "back_elbow_angle": back_elbow_angle,
            "front_elbow_elevated": front_elbow_elevated,
            "wrist_velocity": wrist_velocity,
            "wrist_position_quality": "good",
            "hip_rotation": hip_rotation,
            "hip_line_vs_crease": abs(hip_rotation),
            "front_knee_bend": front_knee_bend,
            "front_knee_alignment": front_knee_alignment,
            "front_foot_direction": front_foot_direction,
            "back_foot_stability": True,
            "foot_spread": foot_spread,
            "spine_lean": spine_lean,
            "balance_score": balance_score,
            "elbow_angle": front_elbow_angle,
            "spine_angle": 90 - spine_lean,
            "knee_angle": front_knee_bend,
            "head_knee_dist": head_knee_alignment,
            "elbow_velocity": 0,
            "body_lean": spine_lean,
            "landmarks": {
                "nose": nose,
                "left_shoulder": left_shoulder, "right_shoulder": right_shoulder,
                "left_elbow": left_elbow, "right_elbow": right_elbow,
                "left_wrist": left_wrist, "right_wrist": right_wrist,
                "left_hip": left_hip, "right_hip": right_hip,
                "left_knee": left_knee, "right_knee": right_knee,
                "left_ankle": left_ankle, "right_ankle": right_ankle
            }
        }
        
        return metrics
    
    # ... (rest of the methods remain the same as before)
    
# Rest of the file remains the same - just using different pose detection backend