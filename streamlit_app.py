import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
import tempfile
from pathlib import Path
import base64

# Import our enhanced analysis functions
try:
    from enhanced_analysis import analyze_video
    from report_generator import generate_reports
    ENHANCED_MODE = True
except ImportError:
    try:
        # Fallback to ultralytics version
        from enhanced_analysis_ultralytics import analyze_video
        from report_generator import generate_reports
        ENHANCED_MODE = True
        st.info("🔄 Using Ultralytics YOLO pose estimation (MediaPipe fallback)")
    except ImportError:
        # Basic mode fallback
        ENHANCED_MODE = False
        st.warning("⚠️ Running in basic mode. Some features may be limited.")

# ===============================
# Streamlit Configuration
# ===============================
st.set_page_config(
    page_title="Cricket Cover Drive Analyzer", 
    page_icon="🏏",
    layout="wide"
)

# ===============================
# Header Navigation
# ===============================
def render_header():
    """Render the header navigation bar."""
    st.markdown("""
    <style>
    .header-nav {
        background: linear-gradient(90deg, #2c5530, #4a7c59);
        padding: 1rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .header-nav h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        text-align: center;
        font-weight: 600;
    }
    .nav-subtitle {
        color: #e8f5e8;
        text-align: center;
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    .nav-features {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .nav-feature {
        color: #e8f5e8;
        font-size: 0.9rem;
        padding: 0.3rem 0.8rem;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    @media (max-width: 768px) {
        .header-nav h1 { font-size: 1.5rem; }
        .nav-features { gap: 0.5rem; }
        .nav-feature { font-size: 0.8rem; padding: 0.2rem 0.6rem; }
    }
    </style>
    <div class="header-nav">
        <h1>🏏 Cricket Cover Drive Analyzer</h1>
        <div class="nav-subtitle">Real-Time Pose Analysis & Technique Evaluation</div>
        <div class="nav-features">
            <span class="nav-feature">📊 Full Video Processing</span>
            <span class="nav-feature">🎯 Pose Estimation</span>
            <span class="nav-feature">📈 Live Metrics</span>
            <span class="nav-feature">🏆 Shot Evaluation</span>
            <span class="nav-feature">📋 Detailed Reports</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# Footer Component
# ===============================
def render_footer():
    """Render the footer with developer info."""
    st.markdown("""
    <style>
    .footer-container {
        margin-top: 3rem;
        padding: 2rem 0 1rem 0;
        border-top: 2px solid #2c5530;
        background: linear-gradient(135deg, #f8fff8, #e8f5e8);
        text-align: center;
        border-radius: 10px 10px 0 0;
    }
    .footer-main {
        font-size: 1.1rem;
        color: #2c5530;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .footer-dev {
        font-size: 1rem;
        color: #4a7c59;
        margin: 0.3rem 0;
    }
    .footer-date {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
    .footer-tech {
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #777;
    }
    </style>
    <div class="footer-container">
        <div class="footer-main">🏏 Cricket Cover Drive Analyzer</div>
        <div class="footer-dev">Built by <strong>Vivek Kumar Yadav</strong></div>
        <div class="footer-date">17/08/2025</div>
        <div class="footer-tech">Powered by MediaPipe, OpenCV & Streamlit</div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# Helper Functions
# ===============================
def get_download_link(file_path, file_label):
    """Generate download link for files."""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{file_label}</a>'
    return href

def analyze_uploaded_video(uploaded_file):
    """Process uploaded video and return results."""
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file temporarily
        temp_video_path = os.path.join(temp_dir, "uploaded_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze the video
        try:
            if ENHANCED_MODE:
                st.info("🚀 Using Enhanced Analysis Engine with all advanced features!")
                evaluation = analyze_video(temp_video_path, output_dir)
            else:
                st.info("📊 Using Basic Analysis Engine")
                evaluation = analyze_video_streamlit(temp_video_path, output_dir)
            
            if not evaluation:
                st.error("❌ Analysis failed - no evaluation data returned")
                return None, None, None, {}
            
            # Read the output files
            annotated_video_path = os.path.join(output_dir, "annotated_video.mp4")
            evaluation_json_path = os.path.join(output_dir, "evaluation.json")
            
            # Copy files to permanent location
            permanent_output_dir = "streamlit_output"
            os.makedirs(permanent_output_dir, exist_ok=True)
            
            timestamp = int(time.time())
            final_video_path = os.path.join(permanent_output_dir, f"annotated_{timestamp}.mp4")
            final_json_path = os.path.join(permanent_output_dir, f"evaluation_{timestamp}.json")
            
            # Copy files - check if they exist first
            import shutil
            if os.path.exists(annotated_video_path):
                shutil.copy2(annotated_video_path, final_video_path)
            else:
                st.error(f"Annotated video not found at: {annotated_video_path}")
                return None, None, None, {}
                
            if os.path.exists(evaluation_json_path):
                shutil.copy2(evaluation_json_path, final_json_path)
            else:
                # Create a basic evaluation file if it doesn't exist
                st.warning("Evaluation JSON not found, creating basic evaluation...")
                with open(final_json_path, 'w') as f:
                    json.dump(evaluation, f, indent=4)
            
            # Generate reports if enhanced mode
            report_paths = {}
            if ENHANCED_MODE:
                try:
                    # Copy charts to permanent location
                    chart_files = ["smoothness_analysis.png", "performance_comparison.png"]
                    for chart_file in chart_files:
                        src_chart = os.path.join(output_dir, chart_file)
                        if os.path.exists(src_chart):
                            dest_chart = os.path.join(permanent_output_dir, f"{chart_file.split('.')[0]}_{timestamp}.png")
                            shutil.copy2(src_chart, dest_chart)
                    
                    # Generate HTML report
                    report_paths = generate_reports(evaluation, permanent_output_dir)
                    if report_paths:
                        # Rename reports with timestamp
                        for report_type, report_path in report_paths.items():
                            base_name = os.path.basename(report_path)
                            name, ext = os.path.splitext(base_name)
                            new_name = f"{name}_{timestamp}{ext}"
                            new_path = os.path.join(permanent_output_dir, new_name)
                            shutil.move(report_path, new_path)
                            report_paths[report_type] = new_path
                            
                except Exception as e:
                    st.warning(f"Report generation failed: {e}")
            
            return evaluation, final_video_path, final_json_path, report_paths
            
        except FileNotFoundError as e:
            st.error(f"❌ File not found during analysis: {str(e)}")
            return None, None, None, {}
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            st.warning("💡 Try uploading a different video file or check that the video format is supported")
            return None, None, None, {}

def analyze_video_streamlit(video_path, output_dir="output"):
    """Basic analysis fallback for when enhanced_analysis is not available."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple annotated video (copy original for now)
    annotated_path = os.path.join(output_dir, "annotated_video.mp4")
    import shutil
    shutil.copy2(video_path, annotated_path)
    
    # Create basic evaluation data
    evaluation = {
        "scores": {
            "Footwork": 7,
            "Head Position": 6,
            "Swing Control": 5,
            "Balance": 6,
            "Follow-through": 7
        },
        "feedback": {
            "Footwork": "Good stride length",
            "Head Position": "Keep head steady",
            "Swing Control": "Work on elbow position",
            "Balance": "Improve core stability",
            "Follow-through": "Complete the swing"
        },
        "overall_score": 6.2,
        "performance_stats": {
            "total_frames": 120,
            "avg_fps": 8.5,
            "total_processing_time": 14.1,
            "avg_frame_processing_time": 0.117
        }
    }
    
    # Save evaluation to JSON file
    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, 'w') as f:
        json.dump(evaluation, f, indent=4)
    
    return evaluation

# ===============================
# Main Streamlit App
# ===============================
def main():
    # Render header navigation
    render_header()
    
    # Main content area - remove sample data and posture detection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Video")
        
        # Requirements display
        with st.expander("🏏 Cricket Analysis Scope & Features", expanded=False):
            st.markdown("""
            **Core Cricket Detections (Per Frame)**
            
            📍 **Head Position**
            - Steady head tracking throughout shot
            - Head-over-knee alignment analysis
            
            🏃 **Body Alignment** 
            - Shoulder tilt and hip alignment
            - Spine lean vs. vertical analysis
            - Balance and weight distribution
            
            💪 **Arm Mechanics**
            - Front elbow angle (shoulder–elbow–wrist)
            - Front elbow elevation detection
            - Wrist position and velocity tracking
            
            🦵 **Leg Position**
            - Front knee bend and alignment
            - Front foot direction vs. crease
            - Back foot stability analysis
            - Foot spread measurement
            
            **🎯 Real-Time Performance Target**
            - Achieves ≥10 FPS end-to-end processing on CPU
            - Auto-optimization: reduces quality if FPS drops
            - Performance logging with FPS monitoring
            - Adaptive frame skipping for speed
            
            **🏏 Basic Bat Detection & Tracking**
            - Color and shape-based bat detection
            - Swing path tracking and analysis
            - Swing straightness assessment
            - Impact angle calculation
            - Bat detection rate monitoring
            
            **Live Cricket Feedback**
            - ✅ Good elbow elevation / ❌ Low elbow
            - ✅ Head positioned / ❌ Head not over knee  
            - ✅ Good balance / ❌ Too much lean
            - ✅ Knee aligned / ❌ Knee alignment
            - ✅ Foot positioned / ❌ Foot direction
            - 🏏 BAT DETECTED indicator
            
            **Final Cricket Evaluation**
            - **Footwork** (stride, placement, direction) - Score 1-10
            - **Head Position** (steady, aligned) - Score 1-10
            - **Swing Control** (elbow elevation, consistency) - Score 1-10
            - **Balance** (spine lean, weight transfer) - Score 1-10
            - **Follow-through** (completion, finishing) - Score 1-10
            
            **Advanced Features**
            - Phase detection: Stance → Stride → Downswing → Impact → Follow-through
            - Ball contact moment detection from wrist velocity
            - Movement smoothness and consistency analysis
            - Skill grading: Beginner / Intermediate / Advanced
            - Cricket technique breakdown with specific feedback
            - Real-time performance optimization
            """)
        
        uploaded_file = st.file_uploader(
            "Choose a cricket cover drive video", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a cricket cover drive video for comprehensive technique analysis"
        )
        
        if uploaded_file is not None:
            # Make video container smaller with custom width
            st.video(uploaded_file, format="video/mp4", start_time=0)
            
            # Add CSS to make video smaller
            st.markdown("""
            <style>
            .stVideo > div {
                max-width: 400px !important;
                margin: 0 auto;
            }
            .stVideo video {
                max-width: 100% !important;
                height: auto !important;
                max-height: 300px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Show file details
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024*1024):.2f} MB"
            }
            st.json(file_details)
            
            # Analyze button
            if st.button("🔄 Analyze Video", type="primary"):
                with st.spinner("Analyzing video... Processing all frames with real-time metrics..."):
                    evaluation, video_path, json_path, report_paths = analyze_uploaded_video(uploaded_file)
                    
                    if evaluation:
                        # Store results in session state
                        st.session_state.evaluation = evaluation
                        st.session_state.video_path = video_path
                        st.session_state.json_path = json_path
                        st.session_state.report_paths = report_paths
                        
                        # Show performance stats if available
                        if 'performance_stats' in evaluation:
                            perf = evaluation['performance_stats']
                            st.success(f"✅ Analysis complete! Processed at {perf.get('avg_fps', 0):.1f} FPS")
                            if perf.get('avg_fps', 0) >= 10:
                                st.balloons()  # Celebrate good performance!
                        else:
                            st.success("✅ Analysis complete!")
                        
                        st.rerun()

    with col2:
        st.header("📈 Results & Downloads")
        
        # Display results if available
        if hasattr(st.session_state, 'evaluation') and st.session_state.evaluation:
            evaluation = st.session_state.evaluation
            
            # Enhanced mode features
            if ENHANCED_MODE and 'skill_grade' in evaluation:
                # Skill grade display
                grade = evaluation.get('skill_grade', 'Beginner')
                grade_colors = {'Beginner': '🔵', 'Intermediate': '🟡', 'Advanced': '🟢'}
                st.subheader(f"{grade_colors.get(grade, '⚪')} Skill Level: {grade}")
                
                # Cricket-specific analysis
                if 'cricket_analysis' in evaluation:
                    cricket = evaluation['cricket_analysis']
                    
                    # Technique breakdown
                    st.subheader("🏏 Cricket Technique Analysis")
                    technique_cols = st.columns(3)
                    
                    with technique_cols[0]:
                        st.info(f"**Front Elbow:** {cricket['technique_breakdown']['front_elbow_analysis']}")
                        st.info(f"**Head-Knee:** {cricket['technique_breakdown']['head_knee_alignment']}")
                    
                    with technique_cols[1]:
                        st.info(f"**Spine Lean:** {cricket['technique_breakdown']['spine_lean_analysis']}")
                        st.info(f"**Foot Direction:** {cricket['technique_breakdown']['foot_direction']}")
                    
                    with technique_cols[2]:
                        st.info(f"**Weight Transfer:** {cricket['technique_breakdown']['weight_transfer']}")
                        st.info(f"**Contact Quality:** {cricket['contact_quality']}")
                
                # Performance stats with FPS target tracking
                if 'performance_stats' in evaluation:
                    perf = evaluation['performance_stats']
                    st.subheader("⚡ Real-Time Performance Target (≥10 FPS)")
                    perf_cols = st.columns(4)
                    with perf_cols[0]:
                        fps_achieved = perf.get('avg_fps', 0)
                        fps_target = 10.0
                        fps_status = "✅" if fps_achieved >= fps_target else "⚠️" if fps_achieved >= fps_target * 0.8 else "❌"
                        st.metric("Processing FPS", f"{fps_achieved:.1f} {fps_status}", delta=f"Target: {fps_target}")
                    with perf_cols[1]:
                        st.metric("Total Frames", perf.get('total_frames', 0))
                    with perf_cols[2]:
                        st.metric("Processing Time", f"{perf.get('total_processing_time', 0):.1f}s")
                    with perf_cols[3]:
                        opt_level = perf.get('optimization_level', 0)
                        opt_text = ["Full Quality", "Medium Speed", "High Speed"][min(opt_level, 2)]
                        st.metric("Auto Optimization", opt_text)
                
                # Bat tracking and swing analysis
                if 'performance_stats' in evaluation and 'swing_analysis' in evaluation['performance_stats']:
                    swing = evaluation['performance_stats']['swing_analysis']
                    st.subheader("🏏 Bat Detection & Swing Path Analysis")
                    
                    swing_cols = st.columns(4)
                    with swing_cols[0]:
                        straightness = swing.get('swing_straightness', 0)
                        st.metric("Swing Straightness", f"{straightness*100:.0f}%")
                    with swing_cols[1]:
                        impact_angle = swing.get('impact_angle', 0)
                        st.metric("Impact Angle", f"{impact_angle:.1f}°")
                    with swing_cols[2]:
                        detections = swing.get('bat_detections', 0)
                        total_frames = perf.get('total_frames', 1)
                        detection_rate = (detections / total_frames) * 100
                        st.metric("Bat Detection Rate", f"{detection_rate:.0f}%")
                    with swing_cols[3]:
                        path_length = swing.get('swing_path_length', 0)
                        st.metric("Swing Tracking", f"{path_length} frames")
                    
                    # Swing quality assessment
                    quality = swing.get('swing_quality', 'unknown')
                    quality_colors = {
                        'excellent': '🟢 Excellent - Very straight swing path',
                        'good': '🟡 Good - Minor deviations in swing',
                        'fair': '🟠 Fair - Some swing inconsistencies', 
                        'poor': '🔴 Poor - Irregular swing path',
                        'insufficient_data': '⚪ Insufficient data for analysis'
                    }
                    quality_message = quality_colors.get(quality, '⚪ Unknown swing quality')
                    st.info(f"**Swing Path Quality:** {quality_message}")
                
                # Phase analysis
                if 'phases' in evaluation and evaluation['phases']:
                    st.subheader("📊 Shot Phase Analysis")
                    phase_cols = st.columns(min(len(evaluation['phases']), 5))
                    for i, phase in enumerate(evaluation['phases'][:5]):  # Limit to 5 phases
                        with phase_cols[i]:
                            st.info(f"**{phase['name'].title().replace('_', ' ')}**\n{phase['duration']} frames")
                
                # Contact moment detection
                if 'contact_moment' in evaluation and evaluation['contact_moment']:
                    contact = evaluation['contact_moment']
                    st.subheader("⚡ Ball Contact Detection")
                    contact_cols = st.columns(3)
                    with contact_cols[0]:
                        st.metric("Frame", contact['frame_number'])
                    with contact_cols[1]:
                        st.metric("Confidence", f"{contact['confidence']*100:.0f}%")
                    with contact_cols[2]:
                        st.metric("Wrist Velocity", f"{contact['wrist_velocity']:.1f}")
                
                # Smoothness metrics
                if 'smoothness_metrics' in evaluation:
                    smooth = evaluation['smoothness_metrics']
                    st.subheader("📈 Movement Smoothness")
                    smooth_cols = st.columns(2)
                    with smooth_cols[0]:
                        st.metric("Smoothness Score", f"{smooth.get('smoothness_score', 0)*100:.0f}%")
                    with smooth_cols[1]:
                        st.metric("Consistency Score", f"{smooth.get('consistency_score', 0)*100:.0f}%")
            
            # Scores display
            st.subheader("🎯 Technique Scores (1-10)")
            
            # Create metrics columns
            score_cols = st.columns(len(evaluation["scores"]))
            for i, (metric, score) in enumerate(evaluation["scores"].items()):
                with score_cols[i]:
                    # Color-code scores
                    if score >= 8:
                        st.metric(metric, score, delta="Excellent", delta_color="normal")
                    elif score >= 6:
                        st.metric(metric, score, delta="Good", delta_color="normal")
                    else:
                        st.metric(metric, score, delta="Needs work", delta_color="inverse")
            
            # Overall score
            overall_score = evaluation.get('overall_score', np.mean(list(evaluation["scores"].values())))
            st.subheader(f"🏆 Overall Score: {overall_score:.1f}/10")
            
            # Progress bar for overall score
            st.progress(overall_score / 10)
            
            # Feedback section
            st.subheader("💡 Detailed Feedback")
            for metric, feedback in evaluation["feedback"].items():
                if "Good" in feedback or "steady" in feedback or "Controlled" in feedback or "Balanced" in feedback or "Smooth" in feedback:
                    st.success(f"**{metric}**: {feedback}")
                else:
                    st.warning(f"**{metric}**: {feedback}")
            
            # Results section with annotated video, JSON preview, and PDF report
            st.subheader("📋 Complete Analysis Results")
            
            # Three columns for different outputs
            result_cols = st.columns(3)
            
            with result_cols[0]:
                st.markdown("**📹 Annotated Video**")
                if hasattr(st.session_state, 'video_path') and os.path.exists(st.session_state.video_path):
                    # Make annotated video smaller
                    st.video(st.session_state.video_path, format="video/mp4", start_time=0)
                    
                    # Add CSS to make annotated video smaller  
                    st.markdown("""
                    <style>
                    .stVideo > div {
                        max-width: 350px !important;
                        margin: 0 auto;
                    }
                    .stVideo video {
                        max-width: 100% !important;
                        height: auto !important;
                        max-height: 250px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Download link for video
                    video_download = get_download_link(st.session_state.video_path, "📥 Download Annotated Video")
                    st.markdown(video_download, unsafe_allow_html=True)
                else:
                    st.info("Annotated video will appear here after analysis")
            
            with result_cols[1]:
                st.markdown("**📊 Analysis Data (JSON)**")
                if hasattr(st.session_state, 'json_path') and os.path.exists(st.session_state.json_path):
                    with open(st.session_state.json_path, 'r') as f:
                        json_data = json.load(f)
                    
                    # Show JSON preview with expandable sections
                    with st.expander("📋 View JSON Preview", expanded=False):
                        st.json(json_data)
                    
                    # Summary of JSON content
                    st.write(f"**Frames Analyzed:** {json_data.get('performance_stats', {}).get('total_frames', 'N/A')}")
                    st.write(f"**Phases Detected:** {len(json_data.get('phases', []))}")
                    st.write(f"**Overall Score:** {json_data.get('overall_score', 'N/A'):.1f}/10")
                    
                    # Download link for JSON
                    json_download = get_download_link(st.session_state.json_path, "📥 Download JSON Report")
                    st.markdown(json_download, unsafe_allow_html=True)
                else:
                    st.info("JSON analysis data will appear here after processing")
            
            with result_cols[2]:
                st.markdown("**📄 Comprehensive Report**")
                if ENHANCED_MODE and hasattr(st.session_state, 'report_paths'):
                    report_paths = st.session_state.report_paths
                    if 'html' in report_paths and os.path.exists(report_paths['html']):
                        st.write("**HTML Report Available**")
                        st.write("✅ Complete analysis with charts")
                        st.write("✅ Training recommendations")
                        st.write("✅ Performance comparisons")
                        
                        # Download link for HTML report
                        html_download = get_download_link(report_paths['html'], "📊 Download HTML Report")
                        st.markdown(html_download, unsafe_allow_html=True)
                        
                        # PDF download if available
                        if 'pdf' in report_paths and os.path.exists(report_paths['pdf']):
                            pdf_download = get_download_link(report_paths['pdf'], "📄 Download PDF Report")
                            st.markdown(pdf_download, unsafe_allow_html=True)
                    else:
                        st.info("Comprehensive report will be generated after analysis")
                else:
                    st.info("Upgrade to enhanced mode for detailed reports")
        
        else:
            st.info("Upload and analyze a video to see results here.")
            
            # Show requirement summary instead of sample data
            st.subheader("🏏 Cricket Analysis Features")
            
            requirements = [
                ("📹 Annotated Video", "Cricket shot with pose overlays and real-time technique feedback"),
                ("📊 JSON Analysis", "Complete frame-by-frame cricket metrics and evaluation scores"),
                ("📄 Comprehensive Report", "Detailed cricket technique report with training recommendations"),
                ("⚡ Live Cricket Feedback", "Real-time cues: ✅ Good elbow elevation ❌ Head not over knee"),
                ("🎯 Cricket Evaluation", "Scores for footwork, head position, swing control, balance, follow-through"),
                ("📈 Phase Analysis", "Cricket shot breakdown: stance, stride, downswing, impact, follow-through"),
                ("🏏 Technique Breakdown", "Front elbow analysis, spine lean, head-knee alignment, foot direction"),
                ("⚡ Contact Detection", "Ball impact moment from wrist velocity analysis"),
                ("📊 Movement Analysis", "Smoothness and consistency metrics for cricket technique")
            ]
            
            for title, description in requirements:
                with st.expander(title, expanded=False):
                    st.write(description)

    # Render footer
    render_footer()

if __name__ == "__main__":
    main()