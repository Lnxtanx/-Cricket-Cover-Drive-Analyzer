# Cricket Cover Drive Analysis - Configuration
# All thresholds, paths, and parameters for the analysis system

import os

# ===============================
# File Paths & Directories
# ===============================
PATHS = {
    "output_dir": "output",
    "config_dir": "config",
    "reports_dir": "reports",
    "temp_dir": "temp"
}

# ===============================
# Video Processing Settings
# ===============================
VIDEO_CONFIG = {
    "target_fps": 15.0,  # Increased target FPS for real-time performance
    "resize_width": 480,  # Reduced resolution for better performance
    "resize_height": 360,
    "buffer_size": 3,     # Reduced buffer for faster processing
    "confidence_threshold": 0.5,
    "skip_frames": 1,     # Process every nth frame for speed
    "performance_target_fps": 10.0  # Minimum FPS target
}

# ===============================
# Phase Detection Thresholds
# ===============================
PHASE_DETECTION = {
    "velocity_threshold": 50,    # px/frame for motion detection
    "angle_delta_threshold": 15, # degrees for phase transitions
    "smoothing_window": 5,       # frames for velocity smoothing
    "min_phase_duration": 10     # minimum frames per phase
}

# ===============================
# Contact Detection Settings
# ===============================
CONTACT_DETECTION = {
    "wrist_velocity_threshold": 100,  # px/frame for impact detection
    "elbow_acceleration_threshold": 80,
    "impact_search_window": 30,       # frames around suspected impact
    "velocity_spike_factor": 2.0      # multiplier for spike detection
}

# ===============================
# Ideal Cover Drive Reference
# ===============================
IDEAL_METRICS = {
    "stance": {
        "elbow_angle": {"min": 130, "max": 160, "optimal": 145},
        "spine_angle": {"min": 80, "max": 100, "optimal": 90},
        "foot_spread": {"min": 40, "max": 60, "optimal": 50}
    },
    "stride": {
        "elbow_angle": {"min": 120, "max": 150, "optimal": 135},
        "spine_angle": {"min": 75, "max": 95, "optimal": 85},
        "head_knee_dist": {"min": 20, "max": 40, "optimal": 30}
    },
    "downswing": {
        "elbow_angle": {"min": 90, "max": 120, "optimal": 105},
        "spine_angle": {"min": 70, "max": 90, "optimal": 80},
        "swing_plane_angle": {"min": 30, "max": 50, "optimal": 40}
    },
    "impact": {
        "elbow_angle": {"min": 140, "max": 170, "optimal": 155},
        "spine_angle": {"min": 75, "max": 95, "optimal": 85},
        "bat_angle": {"min": 20, "max": 40, "optimal": 30}
    },
    "follow_through": {
        "elbow_angle": {"min": 160, "max": 180, "optimal": 170},
        "spine_angle": {"min": 80, "max": 100, "optimal": 90},
        "swing_completion": {"min": 0.8, "max": 1.0, "optimal": 0.95}
    }
}

# ===============================
# Smoothness Metrics
# ===============================
SMOOTHNESS_CONFIG = {
    "max_angle_delta": 20,      # degrees per frame
    "variance_threshold": 100,   # variance threshold for smoothness
    "chart_width": 800,
    "chart_height": 400,
    "export_charts": True
}

# ===============================
# Skill Grading Thresholds
# ===============================
SKILL_GRADES = {
    "beginner": {"min_score": 0, "max_score": 4.5},
    "intermediate": {"min_score": 4.5, "max_score": 7.0},
    "advanced": {"min_score": 7.0, "max_score": 10.0}
}

# ===============================
# Scoring Weights
# ===============================
SCORING_WEIGHTS = {
    "phase_consistency": 0.2,
    "technique_accuracy": 0.3,
    "smoothness": 0.2,
    "timing": 0.15,
    "reference_deviation": 0.15
}

# ===============================
# Basic Bat Detection Settings
# ===============================
BAT_DETECTION = {
    "color_ranges": {
        "brown_wood": {"lower": [8, 50, 50], "upper": [25, 255, 200]},
        "light_wood": {"lower": [15, 30, 100], "upper": [35, 150, 255]},
        "white_grip": {"lower": [0, 0, 180], "upper": [180, 30, 255]}
    },
    "min_contour_area": 300,
    "max_contour_area": 3000,
    "aspect_ratio_range": [0.05, 0.4],  # For elongated bat shapes
    "length_threshold": 80,  # Minimum bat length in pixels
    "track_swing_path": True,
    "swing_smoothing_window": 5
}

# ===============================
# Performance Monitoring
# ===============================
PERFORMANCE_CONFIG = {
    "log_fps": True,
    "log_processing_time": True,
    "performance_log_file": "performance.log",
    "benchmark_mode": False,
    "fps_target": 10.0,  # Real-time target
    "fps_log_interval": 30,  # Log every 30 frames
    "auto_optimize": True  # Automatically reduce quality if FPS drops
}

# ===============================
# Report Generation
# ===============================
REPORT_CONFIG = {
    "generate_html": True,
    "generate_pdf": True,  # PDF generation enabled
    "include_charts": True,
    "include_frame_samples": True,
    "template_file": "report_template.html"
}

# ===============================
# Logging Configuration
# ===============================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "file": "cricket_analysis.log",
    "console_output": True
}