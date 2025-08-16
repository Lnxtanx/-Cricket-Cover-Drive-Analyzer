import cv2
import mediapipe as mp
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
# Data Classes
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
# Enhanced Pose Analysis Class
# ===============================
class EnhancedCoverDriveAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Lightweight model for speed
            enable_segmentation=False,
            min_detection_confidence=VIDEO_CONFIG["confidence_threshold"],
            min_tracking_confidence=VIDEO_CONFIG["confidence_threshold"]
        )
        
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
        """Basic bat detection using color and shape analysis with tracking."""
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
    
    def analyze_swing_path(self):
        """Analyze bat swing path for straightness and impact angle."""
        if len(self.bat_positions) < 3:
            return {
                "swing_straightness": 0, 
                "impact_angle": 0, 
                "swing_quality": "insufficient_data",
                "bat_detections": self.bat_detected_frames,
                "swing_path_length": len(self.bat_positions)
            }
        
        # Extract positions
        positions = [pos["center"] for pos in self.bat_positions]
        angles = [pos["angle"] for pos in self.bat_positions]
        
        # Calculate swing straightness (deviation from ideal arc)
        if len(positions) >= 3:
            # Fit a curve to the swing path
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # Calculate deviations from smooth curve
            deviations = []
            for i in range(1, len(positions) - 1):
                # Simple straightness check: angle between consecutive segments
                p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
                
                # Vectors
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                # Angle between vectors
                if (v1[0]**2 + v1[1]**2) > 0 and (v2[0]**2 + v2[1]**2) > 0:
                    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                    cos_angle = dot_product / (mag1 * mag2)
                    angle_dev = abs(math.acos(max(-1, min(1, cos_angle))))
                    deviations.append(angle_dev)
            
            avg_deviation = np.mean(deviations) if deviations else 0
            swing_straightness = max(0, 1 - (avg_deviation / math.pi))  # Normalize to 0-1
        else:
            swing_straightness = 0
        
        # Impact angle (bat angle at last position)
        impact_angle = angles[-1] if angles else 0
        
        # Overall swing quality assessment
        if swing_straightness > 0.8:
            swing_quality = "excellent"
        elif swing_straightness > 0.6:
            swing_quality = "good"
        elif swing_straightness > 0.4:
            swing_quality = "fair"
        else:
            swing_quality = "poor"
        
        return {
            "swing_straightness": swing_straightness,
            "impact_angle": impact_angle,
            "swing_quality": swing_quality,
            "bat_detections": self.bat_detected_frames,
            "swing_path_length": len(self.bat_positions)
        }
    
    def extract_enhanced_metrics(self, landmarks, frame_width, frame_height, prev_landmarks=None):
        """Extract comprehensive cricket-specific metrics from pose landmarks."""
        # Convert normalized landmarks to pixel coordinates
        coords = lambda idx: (
            int(landmarks[idx].x * frame_width), 
            int(landmarks[idx].y * frame_height)
        ) if landmarks[idx].visibility > 0.5 else None
        
        # Key body points for cricket analysis
        left_shoulder = coords(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        right_shoulder = coords(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        left_elbow = coords(self.mp_pose.PoseLandmark.LEFT_ELBOW.value)
        right_elbow = coords(self.mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        left_wrist = coords(self.mp_pose.PoseLandmark.LEFT_WRIST.value)
        right_wrist = coords(self.mp_pose.PoseLandmark.RIGHT_WRIST.value)
        left_hip = coords(self.mp_pose.PoseLandmark.LEFT_HIP.value)
        right_hip = coords(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        left_knee = coords(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
        right_knee = coords(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
        left_ankle = coords(self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
        right_ankle = coords(self.mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        nose = coords(self.mp_pose.PoseLandmark.NOSE.value)
        
        # === CORE CRICKET METRICS ===
        
        # 1. Head Position Analysis
        head_steady = True  # Will be calculated based on frame-to-frame movement
        head_knee_alignment = 0
        if nose and left_knee:
            head_knee_alignment = abs(nose[0] - left_knee[0])  # Horizontal distance
        
        # 2. Shoulder Analysis 
        shoulder_tilt = 0
        shoulder_hip_alignment = True
        if left_shoulder and right_shoulder and left_hip and right_hip:
            # Shoulder tilt (degrees from horizontal)
            shoulder_slope = (right_shoulder[1] - left_shoulder[1]) / (right_shoulder[0] - left_shoulder[0] + 1e-6)
            shoulder_tilt = math.degrees(math.atan(shoulder_slope))
            
            # Shoulder-hip alignment
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
            alignment_diff = abs(shoulder_center[0] - hip_center[0])
            shoulder_hip_alignment = alignment_diff < 20  # pixels threshold
        
        # 3. Elbow Analysis (Front elbow - assuming left is front for right-handed batsman)
        front_elbow_angle = 0
        front_elbow_elevated = False
        if left_shoulder and left_elbow and left_wrist:
            front_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            # Check if elbow is elevated (elbow Y position higher than shoulder)
            front_elbow_elevated = left_elbow[1] < left_shoulder[1]
        
        # Back elbow angle for comparison
        back_elbow_angle = 0
        if right_shoulder and right_elbow and right_wrist:
            back_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # 4. Wrist Analysis
        wrist_velocity = 0
        wrist_position_quality = "good"
        if left_wrist and prev_landmarks:
            prev_wrist = coords(self.mp_pose.PoseLandmark.LEFT_WRIST.value) if prev_landmarks else None
            if prev_wrist:
                wrist_velocity = self.calculate_velocity(prev_wrist, left_wrist)
        
        # 5. Hip Analysis (rotation and balance)
        hip_rotation = 0
        hip_line_vs_crease = 0
        if left_hip and right_hip:
            # Hip rotation (degrees from horizontal)
            hip_slope = (right_hip[1] - left_hip[1]) / (right_hip[0] - left_hip[0] + 1e-6)
            hip_rotation = math.degrees(math.atan(hip_slope))
            
            # Hip line vs crease (assuming crease is horizontal)
            hip_line_vs_crease = abs(hip_rotation)
        
        # 6. Knee Analysis
        front_knee_bend = 0
        front_knee_alignment = True
        if left_hip and left_knee and left_ankle:
            front_knee_bend = self.calculate_angle(left_hip, left_knee, left_ankle)
            # Check knee alignment (knee should be over ankle)
            knee_ankle_alignment = abs(left_knee[0] - left_ankle[0])
            front_knee_alignment = knee_ankle_alignment < 15  # pixels threshold
        
        # 7. Feet Analysis
        front_foot_direction = 0  # Toe angle vs crease
        back_foot_stability = True
        foot_spread = 0
        
        if left_ankle and right_ankle:
            foot_spread = abs(left_ankle[0] - right_ankle[0])
            
            # Front foot direction (approximate from ankle-knee line)
            if left_knee and left_ankle:
                foot_slope = (left_knee[1] - left_ankle[1]) / (left_knee[0] - left_ankle[0] + 1e-6)
                front_foot_direction = math.degrees(math.atan(foot_slope))
        
        # === DERIVED METRICS ===
        
        # Spine Lean (hip-shoulder line vs vertical)
        spine_lean = 0
        if left_hip and left_shoulder:
            spine_slope = (left_shoulder[1] - left_hip[1]) / (left_shoulder[0] - left_hip[0] + 1e-6)
            spine_lean = abs(90 - math.degrees(math.atan(spine_slope)))
        
        # Balance assessment (weight distribution)
        balance_score = 0.5  # Default neutral
        if nose and left_ankle:
            # Check if head is over front foot
            head_foot_alignment = abs(nose[0] - left_ankle[0])
            balance_score = max(0, 1 - (head_foot_alignment / 100))  # Normalize
        
        # Cricket-specific metrics
        metrics = {
            # === CORE DETECTIONS ===
            # Head
            "head_steady": head_steady,
            "head_knee_alignment": head_knee_alignment,
            
            # Shoulders  
            "shoulder_tilt": shoulder_tilt,
            "shoulder_hip_alignment": shoulder_hip_alignment,
            
            # Elbows
            "front_elbow_angle": front_elbow_angle,
            "back_elbow_angle": back_elbow_angle,
            "front_elbow_elevated": front_elbow_elevated,
            
            # Wrists
            "wrist_velocity": wrist_velocity,
            "wrist_position_quality": wrist_position_quality,
            
            # Hips
            "hip_rotation": hip_rotation,
            "hip_line_vs_crease": hip_line_vs_crease,
            
            # Knees
            "front_knee_bend": front_knee_bend,
            "front_knee_alignment": front_knee_alignment,
            
            # Feet
            "front_foot_direction": front_foot_direction,
            "back_foot_stability": back_foot_stability,
            "foot_spread": foot_spread,
            
            # === DERIVED METRICS ===
            "spine_lean": spine_lean,
            "balance_score": balance_score,
            
            # === LEGACY COMPATIBILITY ===
            "elbow_angle": front_elbow_angle,  # For backward compatibility
            "spine_angle": 90 - spine_lean,   # Convert to angle from vertical
            "knee_angle": front_knee_bend,
            "head_knee_dist": head_knee_alignment,
            "elbow_velocity": 0,  # Will be calculated if needed
            "body_lean": spine_lean,
            
            # === LANDMARKS FOR DRAWING ===
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
    
    def detect_phases(self, metrics_history: List[Dict]) -> List[PhaseInfo]:
        """Detect cricket shot phases using velocity and angle analysis."""
        if len(metrics_history) < 20:
            return []
        
        phases = []
        current_phase = "stance"
        phase_start = 0
        
        for i, metrics in enumerate(metrics_history):
            wrist_vel = metrics.get("wrist_velocity", 0)
            elbow_angle = metrics.get("elbow_angle", 0)
            
            # Phase transition logic
            if current_phase == "stance" and wrist_vel > 20:
                # Transition to stride
                if i - phase_start > PHASE_DETECTION["min_phase_duration"]:
                    phases.append(PhaseInfo(
                        name="stance",
                        start_frame=phase_start,
                        end_frame=i,
                        duration=i - phase_start,
                        key_metrics=self._calculate_phase_metrics(metrics_history[phase_start:i])
                    ))
                current_phase = "stride"
                phase_start = i
                
            elif current_phase == "stride" and wrist_vel > 50:
                # Transition to downswing
                if i - phase_start > PHASE_DETECTION["min_phase_duration"]:
                    phases.append(PhaseInfo(
                        name="stride",
                        start_frame=phase_start,
                        end_frame=i,
                        duration=i - phase_start,
                        key_metrics=self._calculate_phase_metrics(metrics_history[phase_start:i])
                    ))
                current_phase = "downswing"
                phase_start = i
                
            elif current_phase == "downswing" and wrist_vel > 80:
                # Transition to impact
                if i - phase_start > PHASE_DETECTION["min_phase_duration"]:
                    phases.append(PhaseInfo(
                        name="downswing",
                        start_frame=phase_start,
                        end_frame=i,
                        duration=i - phase_start,
                        key_metrics=self._calculate_phase_metrics(metrics_history[phase_start:i])
                    ))
                current_phase = "impact"
                phase_start = i
                
            elif current_phase == "impact" and wrist_vel < 50:
                # Transition to follow-through
                if i - phase_start > 5:  # Impact is brief
                    phases.append(PhaseInfo(
                        name="impact",
                        start_frame=phase_start,
                        end_frame=i,
                        duration=i - phase_start,
                        key_metrics=self._calculate_phase_metrics(metrics_history[phase_start:i])
                    ))
                current_phase = "follow_through"
                phase_start = i
        
        # Add final phase
        if len(metrics_history) - phase_start > PHASE_DETECTION["min_phase_duration"]:
            phases.append(PhaseInfo(
                name=current_phase,
                start_frame=phase_start,
                end_frame=len(metrics_history),
                duration=len(metrics_history) - phase_start,
                key_metrics=self._calculate_phase_metrics(metrics_history[phase_start:])
            ))
        
        logger.info(f"Detected {len(phases)} phases: {[p.name for p in phases]}")
        return phases
    
    def _calculate_phase_metrics(self, phase_metrics: List[Dict]) -> Dict:
        """Calculate average metrics for a phase."""
        if not phase_metrics:
            return {}
        
        avg_metrics = {}
        for key in ["elbow_angle", "spine_angle", "wrist_velocity", "head_knee_dist"]:
            values = [m.get(key, 0) for m in phase_metrics if m.get(key, 0) > 0]
            avg_metrics[f"avg_{key}"] = np.mean(values) if values else 0
            avg_metrics[f"std_{key}"] = np.std(values) if values else 0
        
        return avg_metrics
    
    def detect_contact_moment(self, metrics_history: List[Dict]) -> Optional[ContactMoment]:
        """Detect the moment of bat-ball contact using motion analysis."""
        if len(metrics_history) < 10:
            return None
        
        # Look for peak wrist velocity and elbow acceleration
        wrist_velocities = [m.get("wrist_velocity", 0) for m in metrics_history]
        
        # Find velocity peaks
        peaks = []
        for i in range(2, len(wrist_velocities) - 2):
            if (wrist_velocities[i] > wrist_velocities[i-1] and 
                wrist_velocities[i] > wrist_velocities[i+1] and
                wrist_velocities[i] > CONTACT_DETECTION["wrist_velocity_threshold"]):
                peaks.append((i, wrist_velocities[i]))
        
        if not peaks:
            return None
        
        # Select the highest peak as likely contact moment
        contact_frame, peak_velocity = max(peaks, key=lambda x: x[1])
        
        # Calculate elbow acceleration around contact
        elbow_acc = 0
        if contact_frame > 1 and contact_frame < len(metrics_history) - 1:
            v1 = metrics_history[contact_frame - 1].get("elbow_velocity", 0)
            v2 = metrics_history[contact_frame + 1].get("elbow_velocity", 0)
            elbow_acc = abs(v2 - v1)
        
        confidence = min(peak_velocity / 100.0, 1.0)  # Normalize confidence
        
        logger.info(f"Contact detected at frame {contact_frame} with confidence {confidence:.2f}")
        
        return ContactMoment(
            frame_number=contact_frame,
            confidence=confidence,
            wrist_velocity=peak_velocity,
            elbow_acceleration=elbow_acc
        )
    
    def calculate_smoothness_metrics(self, metrics_history: List[Dict]) -> Dict:
        """Calculate temporal smoothness and consistency metrics."""
        if len(metrics_history) < 10:
            return {"smoothness_score": 0, "consistency_score": 0}
        
        # Extract angle sequences
        elbow_angles = [m.get("elbow_angle", 0) for m in metrics_history]
        spine_angles = [m.get("spine_angle", 0) for m in metrics_history]
        
        # Calculate frame-to-frame deltas
        elbow_deltas = [abs(elbow_angles[i] - elbow_angles[i-1]) for i in range(1, len(elbow_angles))]
        spine_deltas = [abs(spine_angles[i] - spine_angles[i-1]) for i in range(1, len(spine_angles))]
        
        # Smoothness metrics
        elbow_smoothness = 1.0 - (np.mean(elbow_deltas) / SMOOTHNESS_CONFIG["max_angle_delta"])
        spine_smoothness = 1.0 - (np.mean(spine_deltas) / SMOOTHNESS_CONFIG["max_angle_delta"])
        
        # Consistency metrics (variance)
        elbow_consistency = 1.0 - (np.var(elbow_angles) / SMOOTHNESS_CONFIG["variance_threshold"])
        spine_consistency = 1.0 - (np.var(spine_angles) / SMOOTHNESS_CONFIG["variance_threshold"])
        
        smoothness_score = max(0, min(1, (elbow_smoothness + spine_smoothness) / 2))
        consistency_score = max(0, min(1, (elbow_consistency + spine_consistency) / 2))
        
        return {
            "smoothness_score": smoothness_score,
            "consistency_score": consistency_score,
            "elbow_smoothness": elbow_smoothness,
            "spine_smoothness": spine_smoothness,
            "avg_elbow_delta": np.mean(elbow_deltas),
            "avg_spine_delta": np.mean(spine_deltas),
            "elbow_variance": np.var(elbow_angles),
            "spine_variance": np.var(spine_angles)
        }
    
    def compare_with_reference(self, phases: List[PhaseInfo]) -> Dict:
        """Compare analyzed shot with ideal reference metrics."""
        deviations = {}
        
        for phase in phases:
            phase_name = phase.name
            if phase_name not in IDEAL_METRICS:
                continue
            
            phase_deviations = {}
            ideal_phase = IDEAL_METRICS[phase_name]
            
            for metric, values in ideal_phase.items():
                if f"avg_{metric}" in phase.key_metrics:
                    actual_value = phase.key_metrics[f"avg_{metric}"]
                    optimal_value = values["optimal"]
                    min_value = values["min"]
                    max_value = values["max"]
                    
                    # Calculate deviation from optimal
                    deviation = abs(actual_value - optimal_value)
                    
                    # Calculate score (0-1) based on how close to acceptable range
                    if min_value <= actual_value <= max_value:
                        score = 1.0 - (deviation / (max_value - min_value))
                    else:
                        score = 0.0
                    
                    phase_deviations[metric] = {
                        "actual": actual_value,
                        "optimal": optimal_value,
                        "deviation": deviation,
                        "score": score
                    }
            
            deviations[phase_name] = phase_deviations
        
        return deviations
    
    def predict_skill_grade(self, overall_score: float) -> str:
        """Map overall score to skill grade."""
        for grade, range_info in SKILL_GRADES.items():
            if range_info["min_score"] <= overall_score < range_info["max_score"]:
                return grade.title()
        return "Beginner"
    
    def create_smoothness_chart(self, metrics_history: List[Dict], output_dir: str):
        """Create and save temporal smoothness chart."""
        if not SMOOTHNESS_CONFIG["export_charts"] or len(metrics_history) < 10:
            return
        
        # Extract data for plotting
        frames = list(range(len(metrics_history)))
        elbow_angles = [m.get("elbow_angle", 0) for m in metrics_history]
        spine_angles = [m.get("spine_angle", 0) for m in metrics_history]
        wrist_velocities = [m.get("wrist_velocity", 0) for m in metrics_history]
        
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Cricket Shot Analysis - Temporal Metrics', fontsize=16)
        
        # Elbow angle over time
        ax1.plot(frames, elbow_angles, 'b-', linewidth=2)
        ax1.set_title('Elbow Angle Over Time')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Angle (degrees)')
        ax1.grid(True, alpha=0.3)
        
        # Spine angle over time
        ax2.plot(frames, spine_angles, 'r-', linewidth=2)
        ax2.set_title('Spine Angle Over Time')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Angle (degrees)')
        ax2.grid(True, alpha=0.3)
        
        # Wrist velocity over time
        ax3.plot(frames, wrist_velocities, 'g-', linewidth=2)
        ax3.set_title('Wrist Velocity Over Time')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Velocity (px/frame)')
        ax3.grid(True, alpha=0.3)
        
        # Combined normalized view
        norm_elbow = np.array(elbow_angles) / max(elbow_angles) if max(elbow_angles) > 0 else np.zeros_like(elbow_angles)
        norm_spine = np.array(spine_angles) / max(spine_angles) if max(spine_angles) > 0 else np.zeros_like(spine_angles)
        norm_velocity = np.array(wrist_velocities) / max(wrist_velocities) if max(wrist_velocities) > 0 else np.zeros_like(wrist_velocities)
        
        ax4.plot(frames, norm_elbow, 'b-', label='Elbow Angle', alpha=0.7)
        ax4.plot(frames, norm_spine, 'r-', label='Spine Angle', alpha=0.7)
        ax4.plot(frames, norm_velocity, 'g-', label='Wrist Velocity', alpha=0.7)
        ax4.set_title('Normalized Metrics Comparison')
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Normalized Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(output_dir, "smoothness_analysis.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Smoothness chart saved to {chart_path}")
    
    def calculate_overall_score(self, phases: List[PhaseInfo], smoothness_metrics: Dict, 
                              reference_deviations: Dict, contact_moment: Optional[ContactMoment]) -> float:
        """Calculate weighted overall score."""
        # Phase consistency score
        phase_score = len(phases) / 5.0  # Expecting 5 phases
        phase_score = min(1.0, phase_score)
        
        # Technique accuracy from reference comparison
        technique_scores = []
        for phase_name, phase_devs in reference_deviations.items():
            for metric_name, metric_data in phase_devs.items():
                technique_scores.append(metric_data["score"])
        technique_score = np.mean(technique_scores) if technique_scores else 0.5
        
        # Smoothness score
        smoothness_score = smoothness_metrics.get("smoothness_score", 0.5)
        
        # Timing score (based on contact detection)
        timing_score = contact_moment.confidence if contact_moment else 0.3
        
        # Reference deviation score
        reference_score = technique_score
        
        # Calculate weighted final score
        final_score = (
            SCORING_WEIGHTS["phase_consistency"] * phase_score +
            SCORING_WEIGHTS["technique_accuracy"] * technique_score +
            SCORING_WEIGHTS["smoothness"] * smoothness_score +
            SCORING_WEIGHTS["timing"] * timing_score +
            SCORING_WEIGHTS["reference_deviation"] * reference_score
        )
        
        return min(10.0, final_score * 10)  # Scale to 0-10

# ===============================
# Main Analysis Function
# ===============================
def analyze_video(video_path: str, output_dir: str = "output") -> Dict:
    """
    Main function to analyze cricket cover drive video.
    
    Args:
        video_path: Path to input video
        output_dir: Directory for outputs
        
    Returns:
        Dictionary with complete analysis results
    """
    logger.info(f"Starting analysis of {video_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = EnhancedCoverDriveAnalyzer()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate output dimensions maintaining aspect ratio
    aspect_ratio = original_width / original_height
    target_width = VIDEO_CONFIG["resize_width"]
    target_height = int(target_width / aspect_ratio)
    
    # Ensure height doesn't exceed max height
    max_height = VIDEO_CONFIG["resize_height"]
    if target_height > max_height:
        target_height = max_height
        target_width = int(target_height * aspect_ratio)
    
    output_width = target_width
    output_height = target_height
    
    logger.info(f"Video dimensions: {original_width}x{original_height} -> {output_width}x{output_height}")
    
    out_path = os.path.join(output_dir, "annotated_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, min(fps, VIDEO_CONFIG["target_fps"]), (output_width, output_height))
    
    # Analysis variables
    metrics_history = []
    prev_landmarks = None
    frame_count = 0
    start_time = time.time()
    
    logger.info(f"Processing {total_frames} frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start_time = time.time()
        
        # Performance optimization: skip frames if needed
        if analyzer.should_skip_frame(frame_count):
            frame_count += 1
            continue
        
        # Resize frame for performance
        frame_resized = cv2.resize(frame, (output_width, output_height))
        
        # Process pose (lighter processing if optimizing)
        if analyzer.optimization_level < 2:
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = analyzer.pose.process(rgb_frame)
        else:
            # Skip pose processing for maximum speed
            results = None
        
        bat_info = []
        if results and results.pose_landmarks:
            # Extract metrics
            metrics = analyzer.extract_enhanced_metrics(
                results.pose_landmarks.landmark, 
                output_width, 
                output_height, 
                prev_landmarks
            )
            metrics_history.append(metrics)
            
            # Basic bat detection
            bat_info = analyzer.detect_bat_basic(frame_resized)
            
            # Annotate frame
            frame_annotated = annotate_frame_enhanced(
                frame_resized, results, metrics, bat_info, frame_count
            )
            
            prev_landmarks = results.pose_landmarks.landmark
        else:
            # No pose detected or skipped - still try bat detection
            bat_info = analyzer.detect_bat_basic(frame_resized)
            frame_annotated = frame_resized.copy()
            
            # Add "processing optimized" message
            if analyzer.optimization_level > 0:
                cv2.putText(frame_annotated, f"Optimized Mode L{analyzer.optimization_level}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(frame_annotated, "No pose detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write frame
        out.write(frame_annotated)
        frame_count += 1
        
        # Performance tracking and optimization
        frame_time = time.time() - frame_start_time
        analyzer.performance_stats["processing_times"].append(frame_time)
        
        # Calculate and log FPS periodically
        if frame_count % PERFORMANCE_CONFIG["fps_log_interval"] == 0:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            analyzer.performance_stats["fps_history"].append(current_fps)
            
            # Check performance and optimize if needed
            analyzer.check_performance_and_optimize(current_fps)
            
            if PERFORMANCE_CONFIG["log_fps"]:
                target_fps = PERFORMANCE_CONFIG["fps_target"]
                status = "✅" if current_fps >= target_fps else "⚠️" if current_fps >= target_fps * 0.8 else "❌"
                logger.info(f"{status} Frame {frame_count}/{total_frames}, FPS: {current_fps:.2f} (Target: {target_fps})")
                
                # Log bat detection stats
                if bat_info:
                    logger.info(f"🏏 Bat detected in frame {frame_count}")
    
    # Cleanup
    cap.release()
    out.release()
    analyzer.pose.close()
    
    # Calculate final performance stats
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    avg_processing_time = np.mean(analyzer.performance_stats["processing_times"])
    
    # Analyze swing path
    swing_analysis = analyzer.analyze_swing_path()
    
    # Performance summary
    performance_status = "✅ EXCELLENT" if avg_fps >= PERFORMANCE_CONFIG["fps_target"] else "⚠️ ACCEPTABLE" if avg_fps >= PERFORMANCE_CONFIG["fps_target"] * 0.8 else "❌ NEEDS OPTIMIZATION"
    logger.info(f"🎯 PERFORMANCE SUMMARY: {performance_status}")
    logger.info(f"📊 Processing: {frame_count} frames in {total_time:.2f}s (avg FPS: {avg_fps:.2f})")
    logger.info(f"🏏 Bat Detection: {swing_analysis['bat_detections']} frames, Swing Quality: {swing_analysis['swing_quality']}")
    
    logger.info(f"Processing complete: {frame_count} frames in {total_time:.2f}s (avg FPS: {avg_fps:.2f})")
    
    # Analyze results
    phases = analyzer.detect_phases(metrics_history)
    contact_moment = analyzer.detect_contact_moment(metrics_history)
    smoothness_metrics = analyzer.calculate_smoothness_metrics(metrics_history)
    reference_deviations = analyzer.compare_with_reference(phases)
    overall_score = analyzer.calculate_overall_score(phases, smoothness_metrics, reference_deviations, contact_moment)
    skill_grade = analyzer.predict_skill_grade(overall_score)
    
    # Create charts
    analyzer.create_smoothness_chart(metrics_history, output_dir)
    
    # Compile results
    analysis_result = AnalysisResult(
        phases=phases,
        contact_moment=contact_moment,
        smoothness_metrics=smoothness_metrics,
        skill_grade=skill_grade,
        overall_score=overall_score,
        reference_deviations=reference_deviations,
        performance_stats={
            "avg_fps": avg_fps,
            "total_processing_time": total_time,
            "avg_frame_processing_time": avg_processing_time,
            "total_frames": frame_count,
            "fps_target_met": avg_fps >= PERFORMANCE_CONFIG["fps_target"],
            "optimization_level": analyzer.optimization_level,
            "swing_analysis": swing_analysis
        }
    )
    
    # Save detailed results
    save_detailed_results(analysis_result, output_dir)
    
    # Generate legacy format for compatibility
    legacy_result = generate_legacy_format(analysis_result)
    
    # Also save the legacy evaluation.json for compatibility
    evaluation_path = os.path.join(output_dir, "evaluation.json")
    with open(evaluation_path, 'w') as f:
        json.dump(legacy_result, f, indent=4)
    
    logger.info(f"Legacy evaluation saved to {evaluation_path}")
    
    return legacy_result

def annotate_frame_enhanced(frame, results, metrics, bat_info, frame_number):
    """Enhanced frame annotation with cricket metrics, bat detection, and performance info."""
    mp_drawing = mp.solutions.drawing_utils
    
    # Draw pose landmarks
    if results and results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, 
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )
    
    # Draw bat detection
    if bat_info:
        for bat in bat_info:
            # Draw bat contour
            cv2.drawContours(frame, [bat["contour"]], -1, (0, 255, 255), 2)
            
            # Draw bat center and orientation
            center = bat["center"]
            cv2.circle(frame, center, 5, (0, 255, 255), -1)
            
            # Draw bat orientation line
            angle_rad = math.radians(bat["angle"])
            length = bat["length"] / 2
            end_x = int(center[0] + length * math.cos(angle_rad))
            end_y = int(center[1] + length * math.sin(angle_rad))
            cv2.line(frame, center, (end_x, end_y), (255, 0, 255), 3)
            
            # Bat info text
            cv2.putText(frame, f"Bat: {bat['angle']:.1f}°", (center[0] + 10, center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Get frame dimensions for responsive positioning
    frame_height, frame_width = frame.shape[:2]
    
    # === LEFT SIDE METRICS (Clear spacing) ===
    left_x = 15
    y_start = 30
    y_spacing = 22
    font_scale = 0.55
    thickness = 2
    
    # Background rectangles for better text visibility
    overlay = frame.copy()
    
    # Frame info with performance indicator
    y_pos = y_start
    cv2.rectangle(overlay, (left_x-5, y_pos-18), (left_x+250, y_pos+5), (0, 0, 0), -1)
    perf_indicator = "🟢" if len(bat_info) > 0 else "🔴"
    cv2.putText(frame, f"Frame: {frame_number} {perf_indicator}", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    y_pos += y_spacing
    
    # Only show cricket metrics if we have pose data
    if results and results.pose_landmarks and metrics:
        # Key cricket angles with color coding
        front_elbow = metrics.get('front_elbow_angle', 0)
        elbow_color = (0, 255, 0) if 120 <= front_elbow <= 160 else (0, 165, 255) if front_elbow > 100 else (0, 0, 255)
        cv2.rectangle(overlay, (left_x-5, y_pos-18), (left_x+250, y_pos+5), (0, 0, 0), -1)
        cv2.putText(frame, f"Front Elbow: {front_elbow:.1f}°", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, elbow_color, thickness)
        y_pos += y_spacing
        
        spine_lean = metrics.get('spine_lean', 0)
        spine_color = (0, 255, 0) if spine_lean <= 10 else (0, 165, 255) if spine_lean <= 20 else (0, 0, 255)
        cv2.rectangle(overlay, (left_x-5, y_pos-18), (left_x+250, y_pos+5), (0, 0, 0), -1)
        cv2.putText(frame, f"Spine Lean: {spine_lean:.1f}°", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, spine_color, thickness)
        y_pos += y_spacing
        
        knee_bend = metrics.get('front_knee_bend', 0)
        knee_color = (0, 255, 0) if 140 <= knee_bend <= 170 else (0, 165, 255)
        cv2.rectangle(overlay, (left_x-5, y_pos-18), (left_x+250, y_pos+5), (0, 0, 0), -1)
        cv2.putText(frame, f"Knee Bend: {knee_bend:.1f}°", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, knee_color, thickness)
        y_pos += y_spacing
        
        wrist_vel = metrics.get('wrist_velocity', 0)
        vel_color = (0, 255, 0) if wrist_vel > 20 else (255, 255, 0)
        cv2.rectangle(overlay, (left_x-5, y_pos-18), (left_x+280, y_pos+5), (0, 0, 0), -1)
        cv2.putText(frame, f"Wrist Velocity: {wrist_vel:.1f}", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, vel_color, thickness)
        y_pos += y_spacing
        
        head_knee_align = metrics.get('head_knee_alignment', 0)
        head_color = (0, 255, 0) if head_knee_align <= 30 else (0, 165, 255) if head_knee_align <= 50 else (0, 0, 255)
        cv2.rectangle(overlay, (left_x-5, y_pos-18), (left_x+280, y_pos+5), (0, 0, 0), -1)
        cv2.putText(frame, f"Head-Knee Dist: {head_knee_align:.1f}", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, head_color, thickness)
    
    # Blend overlay for semi-transparent background
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # === RIGHT SIDE FEEDBACK (Clear positioning) ===
    if results and results.pose_landmarks and metrics:
        feedback_x = frame_width - 220
        feedback_y = 35
        feedback_spacing = 30
        feedback_font = 0.65
        feedback_thickness = 2
        
        front_elbow = metrics.get('front_elbow_angle', 0)
        spine_lean = metrics.get('spine_lean', 0)
        head_knee_align = metrics.get('head_knee_alignment', 0)
        
        # Cricket feedback
        if front_elbow < 100:
            cv2.putText(frame, "❌ Low elbow", (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, feedback_font, (0, 0, 255), feedback_thickness)
        elif front_elbow > 160:
            cv2.putText(frame, "⚠️ High elbow", (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, feedback_font, (0, 165, 255), feedback_thickness)
        else:
            cv2.putText(frame, "✅ Good elbow", (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, feedback_font, (0, 255, 0), feedback_thickness)
        feedback_y += feedback_spacing
        
        if head_knee_align > 50:
            cv2.putText(frame, "❌ Head alignment", (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, feedback_font, (0, 0, 255), feedback_thickness)
        elif head_knee_align > 30:
            cv2.putText(frame, "⚠️ Head position", (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, feedback_font, (0, 165, 255), feedback_thickness)
        else:
            cv2.putText(frame, "✅ Head positioned", (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, feedback_font, (0, 255, 0), feedback_thickness)
        feedback_y += feedback_spacing
        
        if spine_lean > 20:
            cv2.putText(frame, "❌ Too much lean", (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, feedback_font, (0, 0, 255), feedback_thickness)
        elif spine_lean > 10:
            cv2.putText(frame, "⚠️ Slight lean", (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, feedback_font, (0, 165, 255), feedback_thickness)
        else:
            cv2.putText(frame, "✅ Good balance", (feedback_x, feedback_y), cv2.FONT_HERSHEY_SIMPLEX, feedback_font, (0, 255, 0), feedback_thickness)
    
    # === BAT DETECTION STATUS (Top center) ===
    if bat_info:
        bat_status_x = frame_width // 2 - 80
        cv2.rectangle(frame, (bat_status_x-10, 5), (bat_status_x+160, 30), (0, 100, 0), -1)
        cv2.putText(frame, "🏏 BAT DETECTED", (bat_status_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # === BOTTOM PERFORMANCE INFO ===
    bottom_y = frame_height - 25
    cv2.rectangle(frame, (10, bottom_y-20), (frame_width-10, frame_height-5), (0, 0, 0), -1)
    status_text = f"🎯 Real-time Analysis | Bat Tracking: {'ON' if bat_info else 'OFF'} | Frame: {frame_number}"
    cv2.putText(frame, status_text, (15, bottom_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    
    return frame

def save_detailed_results(analysis_result: AnalysisResult, output_dir: str):
    """Save comprehensive analysis results."""
    # Convert to serializable format
    results_dict = {
        "phases": [
            {
                "name": phase.name,
                "start_frame": phase.start_frame,
                "end_frame": phase.end_frame,
                "duration": phase.duration,
                "key_metrics": phase.key_metrics
            } for phase in analysis_result.phases
        ],
        "contact_moment": {
            "frame_number": analysis_result.contact_moment.frame_number,
            "confidence": analysis_result.contact_moment.confidence,
            "wrist_velocity": analysis_result.contact_moment.wrist_velocity,
            "elbow_acceleration": analysis_result.contact_moment.elbow_acceleration
        } if analysis_result.contact_moment else None,
        "smoothness_metrics": analysis_result.smoothness_metrics,
        "skill_grade": analysis_result.skill_grade,
        "overall_score": analysis_result.overall_score,
        "reference_deviations": analysis_result.reference_deviations,
        "performance_stats": analysis_result.performance_stats
    }
    
    # Save detailed JSON
    detailed_path = os.path.join(output_dir, "detailed_analysis.json")
    with open(detailed_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    logger.info(f"Detailed results saved to {detailed_path}")

def generate_legacy_format(analysis_result: AnalysisResult) -> Dict:
    """Generate cricket-specific evaluation with comprehensive scoring."""
    
    # Extract cricket-specific scores from analysis data
    scores = {}
    feedback = {}
    
    # Calculate scores based on cricket fundamentals
    if analysis_result.reference_deviations:
        # Get average scores from phase analysis
        avg_scores = {}
        for phase_name, phase_devs in analysis_result.reference_deviations.items():
            phase_scores = [metric_data["score"] for metric_data in phase_devs.values()]
            if phase_scores:
                avg_scores[phase_name] = np.mean(phase_scores)
    else:
        avg_scores = {}
    
    # === CRICKET-SPECIFIC SCORING (1-10 scale) ===
    
    # 1. Footwork - stride length, placement, direction
    footwork_score = int(avg_scores.get("stride", 0.6) * 10) if avg_scores else 6
    footwork_feedback = "Good stride length and placement" if footwork_score > 6 else "Work on front foot placement and stride timing"
    
    # 2. Head Position - steady, aligned over front knee
    head_score = int(avg_scores.get("stance", 0.6) * 10) if avg_scores else 6
    head_feedback = "Head steady and well positioned" if head_score > 6 else "Keep head steady and over front knee"
    
    # 3. Swing Control - elbow elevation, wrist action, consistency
    swing_score = int(avg_scores.get("downswing", 0.6) * 10) if avg_scores else 6
    swing_feedback = "Controlled swing with good elbow elevation" if swing_score > 6 else "Lift front elbow higher, control swing path"
    
    # 4. Balance - spine lean, weight transfer, stability
    balance_score = int(analysis_result.smoothness_metrics.get("consistency_score", 0.6) * 10)
    balance_feedback = "Excellent balance throughout shot" if balance_score > 7 else "Improve balance and weight transfer"
    
    # 5. Follow-through - completion, finishing position
    followthrough_score = int(avg_scores.get("follow_through", 0.6) * 10) if avg_scores else 6
    followthrough_feedback = "Smooth follow-through to completion" if followthrough_score > 6 else "Complete the follow-through, high finish"
    
    # Compile scores and feedback
    scores = {
        "Footwork": footwork_score,
        "Head Position": head_score, 
        "Swing Control": swing_score,
        "Balance": balance_score,
        "Follow-through": followthrough_score
    }
    
    feedback = {
        "Footwork": footwork_feedback,
        "Head Position": head_feedback,
        "Swing Control": swing_feedback, 
        "Balance": balance_feedback,
        "Follow-through": followthrough_feedback
    }
    
    # === DETAILED CRICKET ANALYSIS ===
    cricket_analysis = {
        "technique_breakdown": {
            "front_elbow_analysis": "Good elevation" if avg_scores.get("impact", 0.6) > 0.7 else "Needs more elevation",
            "spine_lean_analysis": "Balanced lean" if analysis_result.smoothness_metrics.get("smoothness_score", 0.5) > 0.6 else "Excessive lean detected",
            "head_knee_alignment": "Well aligned" if head_score > 6 else "Improve head position over front knee",
            "foot_direction": "Good direction" if footwork_score > 6 else "Check front foot direction vs crease",
            "weight_transfer": "Smooth transfer" if balance_score > 6 else "Work on weight shift timing"
        },
        "shot_phases_detected": len(analysis_result.phases),
        "contact_quality": "High confidence" if analysis_result.contact_moment and analysis_result.contact_moment.confidence > 0.7 else "Contact timing needs work",
        "consistency_rating": analysis_result.smoothness_metrics.get("smoothness_score", 0.5)
    }
    
    return {
        "scores": scores,
        "feedback": feedback,
        "skill_grade": analysis_result.skill_grade,
        "overall_score": analysis_result.overall_score,
        "performance_stats": analysis_result.performance_stats,
        "cricket_analysis": cricket_analysis,
        "phases": [
            {
                "name": phase.name,
                "start_frame": phase.start_frame,
                "end_frame": phase.end_frame,
                "duration": phase.duration,
                "key_metrics": phase.key_metrics
            } for phase in analysis_result.phases
        ] if analysis_result.phases else [],
        "contact_moment": {
            "frame_number": analysis_result.contact_moment.frame_number,
            "confidence": analysis_result.contact_moment.confidence,
            "wrist_velocity": analysis_result.contact_moment.wrist_velocity,
            "elbow_acceleration": analysis_result.contact_moment.elbow_acceleration
        } if analysis_result.contact_moment else None,
        "smoothness_metrics": analysis_result.smoothness_metrics,
        "reference_deviations": analysis_result.reference_deviations
    }

if __name__ == "__main__":
    # Test with existing video
    if os.path.exists("input_video.mp4"):
        result = analyze_video("input_video.mp4")
        print(f"Analysis complete! Overall score: {result['overall_score']:.1f}/10")
        print(f"Skill grade: {result['skill_grade']}")
    else:
        print("No input video found. Please provide input_video.mp4 for testing.")