"""
Report Generation Module for Cricket Cover Drive Analysis
Generates HTML and PDF reports with comprehensive analysis results
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from jinja2 import Template
import logging

logger = logging.getLogger(__name__)

# HTML Report Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cricket Cover Drive Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #2c5530;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2c5530;
            margin: 0;
            font-size: 2.5em;
        }
        .header .subtitle {
            color: #666;
            font-size: 1.2em;
            margin-top: 10px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .summary-card {
            background: linear-gradient(135deg, #2c5530, #4a7c59);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            font-size: 1.1em;
        }
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .scores-section {
            margin: 40px 0;
        }
        .scores-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .score-card {
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            transition: transform 0.2s;
        }
        .score-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .score-excellent { border-color: #4CAF50; background-color: #f8fff8; }
        .score-good { border-color: #FF9800; background-color: #fff8f0; }
        .score-poor { border-color: #f44336; background-color: #fff5f5; }
        
        .score-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .score-excellent .score-value { color: #4CAF50; }
        .score-good .score-value { color: #FF9800; }
        .score-poor .score-value { color: #f44336; }
        
        .phases-section {
            margin: 40px 0;
        }
        .phase-timeline {
            display: flex;
            justify-content: space-between;
            margin: 20px 0;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }
        .phase-item {
            text-align: center;
            flex: 1;
            padding: 10px;
            margin: 0 5px;
            background-color: white;
            border-radius: 5px;
            border: 2px solid #ddd;
        }
        .phase-item.detected {
            border-color: #4CAF50;
            background-color: #f8fff8;
        }
        .feedback-section {
            margin: 40px 0;
        }
        .feedback-item {
            margin: 15px 0;
            padding: 15px;
            border-left: 4px solid #2c5530;
            background-color: #f9f9f9;
        }
        .feedback-positive { border-left-color: #4CAF50; background-color: #f8fff8; }
        .feedback-warning { border-left-color: #FF9800; background-color: #fff8f0; }
        .feedback-negative { border-left-color: #f44336; background-color: #fff5f5; }
        
        .charts-section {
            margin: 40px 0;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .performance-section {
            margin: 40px 0;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }
        .performance-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        .performance-item {
            text-align: center;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
        }
        .recommendations {
            margin: 40px 0;
            padding: 20px;
            background: linear-gradient(135deg, #e8f5e8, #f0f8f0);
            border-radius: 8px;
            border: 1px solid #4CAF50;
        }
        .recommendation-item {
            margin: 10px 0;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            border-left: 3px solid #4CAF50;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
            .container { padding: 15px; }
            .summary-grid, .scores-grid { grid-template-columns: 1fr; }
            .phase-timeline { flex-direction: column; }
            .header h1 { font-size: 2em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🏏 Cricket Cover Drive Analysis</h1>
            <div class="subtitle">Comprehensive Technique Report</div>
            <div class="subtitle">Generated on {{ report_date }}</div>
        </div>

        <!-- Summary Cards -->
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Overall Score</h3>
                <div class="value">{{ "%.1f"|format(overall_score) }}/10</div>
                <div>{{ skill_grade }}</div>
            </div>
            <div class="summary-card">
                <h3>Phases Detected</h3>
                <div class="value">{{ phases|length }}</div>
                <div>Shot segments</div>
            </div>
            <div class="summary-card">
                <h3>Processing Speed</h3>
                <div class="value">{{ "%.1f"|format(performance_stats.avg_fps) }}</div>
                <div>FPS</div>
            </div>
            <div class="summary-card">
                <h3>Smoothness Score</h3>
                <div class="value">{{ "%.0f"|format(smoothness_metrics.smoothness_score * 100) }}%</div>
                <div>Technique flow</div>
            </div>
        </div>

        <!-- Technique Scores -->
        <div class="scores-section">
            <h2>🎯 Technique Scores</h2>
            <div class="scores-grid">
                {% for metric, score in scores.items() %}
                <div class="score-card {% if score >= 8 %}score-excellent{% elif score >= 6 %}score-good{% else %}score-poor{% endif %}">
                    <h3>{{ metric }}</h3>
                    <div class="score-value">{{ score }}/10</div>
                    <div>{{ feedback[metric] }}</div>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- Phase Analysis -->
        <div class="phases-section">
            <h2>📊 Shot Phase Analysis</h2>
            <div class="phase-timeline">
                {% for phase in phases %}
                <div class="phase-item detected">
                    <h4>{{ phase.name.title().replace('_', ' ') }}</h4>
                    <div>{{ phase.duration }} frames</div>
                    <div>{{ phase.start_frame }}-{{ phase.end_frame }}</div>
                </div>
                {% endfor %}
            </div>
            
            {% if contact_moment %}
            <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <h4>⚡ Contact Detection</h4>
                <p><strong>Frame:</strong> {{ contact_moment.frame_number }}</p>
                <p><strong>Confidence:</strong> {{ "%.0f"|format(contact_moment.confidence * 100) }}%</p>
                <p><strong>Wrist Velocity:</strong> {{ "%.1f"|format(contact_moment.wrist_velocity) }} px/frame</p>
            </div>
            {% endif %}
        </div>

        <!-- Detailed Feedback -->
        <div class="feedback-section">
            <h2>💡 Detailed Analysis</h2>
            {% for metric, comment in feedback.items() %}
            <div class="feedback-item {% if 'Good' in comment or 'Excellent' in comment %}feedback-positive{% elif 'Improve' in comment or 'Work' in comment %}feedback-warning{% else %}feedback-negative{% endif %}">
                <strong>{{ metric }}:</strong> {{ comment }}
            </div>
            {% endfor %}
        </div>

        <!-- Charts Section -->
        {% if charts_available %}
        <div class="charts-section">
            <h2>📈 Temporal Analysis</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{{ smoothness_chart }}" alt="Smoothness Analysis Chart">
                <p><em>Frame-by-frame analysis showing elbow angle, spine angle, and wrist velocity over time</em></p>
            </div>
        </div>
        {% endif %}

        <!-- Performance Stats -->
        <div class="performance-section">
            <h2>⚡ Performance Statistics</h2>
            <div class="performance-grid">
                <div class="performance-item">
                    <h4>Total Frames</h4>
                    <div style="font-size: 1.5em; font-weight: bold;">{{ performance_stats.total_frames }}</div>
                </div>
                <div class="performance-item">
                    <h4>Processing Time</h4>
                    <div style="font-size: 1.5em; font-weight: bold;">{{ "%.1f"|format(performance_stats.total_processing_time) }}s</div>
                </div>
                <div class="performance-item">
                    <h4>Average FPS</h4>
                    <div style="font-size: 1.5em; font-weight: bold;">{{ "%.1f"|format(performance_stats.avg_fps) }}</div>
                </div>
                <div class="performance-item">
                    <h4>Frame Processing</h4>
                    <div style="font-size: 1.5em; font-weight: bold;">{{ "%.0f"|format(performance_stats.avg_frame_processing_time * 1000) }}ms</div>
                </div>
            </div>
        </div>

        <!-- Recommendations -->
        <div class="recommendations">
            <h2>🎯 Training Recommendations</h2>
            {% for recommendation in training_recommendations %}
            <div class="recommendation-item">
                {{ recommendation }}
            </div>
            {% endfor %}
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>Report generated by Cricket Cover Drive Analyzer</p>
            <p>Powered by MediaPipe & Computer Vision</p>
        </div>
    </div>
</body>
</html>
"""

def generate_training_recommendations(scores: Dict, feedback: Dict, skill_grade: str) -> List[str]:
    """Generate personalized training recommendations based on analysis."""
    recommendations = []
    
    # Grade-based recommendations
    if skill_grade.lower() == "beginner":
        recommendations.append("🎯 Focus on basic stance and grip fundamentals")
        recommendations.append("📚 Practice shadow batting to develop muscle memory")
        recommendations.append("🎥 Watch professional players' cover drive techniques")
    elif skill_grade.lower() == "intermediate":
        recommendations.append("⚡ Work on timing and rhythm consistency")
        recommendations.append("🎯 Practice against varied bowling speeds")
        recommendations.append("💪 Focus on follow-through completion")
    else:  # Advanced
        recommendations.append("🔥 Fine-tune timing for different ball lengths")
        recommendations.append("🎯 Work on shot placement accuracy")
        recommendations.append("⚡ Practice under pressure situations")
    
    # Score-based specific recommendations
    low_scores = [metric for metric, score in scores.items() if score < 6]
    
    for metric in low_scores:
        if metric == "Footwork":
            recommendations.append("🦶 Practice front foot movement to the pitch of the ball")
            recommendations.append("📏 Work on stride length consistency")
        elif metric == "Head Position":
            recommendations.append("👁️ Focus on keeping head still and eyes level")
            recommendations.append("🎯 Practice head position drills with coach")
        elif metric == "Swing Control":
            recommendations.append("💪 Work on elbow position and bat swing path")
            recommendations.append("🎯 Practice controlled swing exercises")
        elif metric == "Balance":
            recommendations.append("⚖️ Improve core strength and stability")
            recommendations.append("🧘 Practice balance drills without bat")
        elif metric == "Follow-through":
            recommendations.append("🔄 Complete the bat swing through the shot")
            recommendations.append("💪 Work on shoulder and wrist flexibility")
    
    return recommendations

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string for embedding in HTML."""
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return encoded
    except Exception as e:
        logger.warning(f"Failed to encode image {image_path}: {e}")
        return ""

def generate_html_report(analysis_data: Dict, output_dir: str) -> str:
    """Generate comprehensive HTML report."""
    try:
        # Prepare template data
        template_data = {
            "report_date": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            "overall_score": analysis_data.get("overall_score", 0),
            "skill_grade": analysis_data.get("skill_grade", "Beginner"),
            "scores": analysis_data.get("scores", {}),
            "feedback": analysis_data.get("feedback", {}),
            "phases": analysis_data.get("phases", []),
            "contact_moment": analysis_data.get("contact_moment"),
            "smoothness_metrics": analysis_data.get("smoothness_metrics", {}),
            "performance_stats": analysis_data.get("performance_stats", {}),
            "training_recommendations": generate_training_recommendations(
                analysis_data.get("scores", {}),
                analysis_data.get("feedback", {}),
                analysis_data.get("skill_grade", "Beginner")
            )
        }
        
        # Check for charts
        chart_path = os.path.join(output_dir, "smoothness_analysis.png")
        if os.path.exists(chart_path):
            template_data["charts_available"] = True
            template_data["smoothness_chart"] = encode_image_to_base64(chart_path)
        else:
            template_data["charts_available"] = False
        
        # Render template
        template = Template(HTML_TEMPLATE)
        html_content = template.render(**template_data)
        
        # Save HTML report
        report_path = os.path.join(output_dir, "analysis_report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        return ""

def generate_pdf_report(html_path: str, output_dir: str) -> str:
    """Generate PDF report from HTML (requires wkhtmltopdf)."""
    try:
        import pdfkit
        
        pdf_path = os.path.join(output_dir, "analysis_report.pdf")
        
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        
        pdfkit.from_file(html_path, pdf_path, options=options)
        logger.info(f"PDF report generated: {pdf_path}")
        return pdf_path
        
    except ImportError:
        logger.warning("pdfkit not installed. PDF generation skipped.")
        return ""
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        return ""

def create_comparison_chart(analysis_data: Dict, output_dir: str):
    """Create a comparison chart showing actual vs ideal metrics."""
    try:
        reference_deviations = analysis_data.get("reference_deviations", {})
        if not reference_deviations:
            return
        
        # Prepare data for plotting
        phases = []
        actual_scores = []
        ideal_scores = []
        
        for phase_name, phase_data in reference_deviations.items():
            for metric_name, metric_data in phase_data.items():
                phases.append(f"{phase_name}\n{metric_name}")
                actual_scores.append(metric_data["score"] * 10)  # Scale to 0-10
                ideal_scores.append(10)  # Ideal is always 10
        
        if not phases:
            return
        
        # Create comparison chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(phases))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, actual_scores, width, label='Actual Performance', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, ideal_scores, width, label='Ideal Performance', color='lightgreen', alpha=0.8)
        
        ax.set_xlabel('Phase - Metric')
        ax.set_ylabel('Score (0-10)')
        ax.set_title('Actual vs Ideal Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(output_dir, "performance_comparison.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison chart saved to {chart_path}")
        
    except Exception as e:
        logger.error(f"Failed to create comparison chart: {e}")

def generate_reports(analysis_data: Dict, output_dir: str) -> Dict[str, str]:
    """Generate all available report formats."""
    reports = {}
    
    try:
        # Generate comparison chart
        create_comparison_chart(analysis_data, output_dir)
        
        # Generate HTML report
        html_path = generate_html_report(analysis_data, output_dir)
        if html_path:
            reports["html"] = html_path
        
        # Generate PDF report if requested
        from config import REPORT_CONFIG
        if REPORT_CONFIG.get("generate_pdf", False) and html_path:
            pdf_path = generate_pdf_report(html_path, output_dir)
            if pdf_path:
                reports["pdf"] = pdf_path
        
        logger.info(f"Generated {len(reports)} report(s)")
        return reports
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {}

if __name__ == "__main__":
    # Test report generation with sample data
    sample_data = {
        "overall_score": 7.2,
        "skill_grade": "Intermediate",
        "scores": {
            "Footwork": 8,
            "Head Position": 7,
            "Swing Control": 6,
            "Balance": 5,
            "Follow-through": 7
        },
        "feedback": {
            "Footwork": "Good stride length and timing",
            "Head Position": "Head steady throughout shot",
            "Swing Control": "Work on elbow position",
            "Balance": "Improve core stability",
            "Follow-through": "Complete the swing follow-through"
        },
        "phases": [
            {"name": "stance", "start_frame": 0, "end_frame": 20, "duration": 20},
            {"name": "stride", "start_frame": 20, "end_frame": 35, "duration": 15},
            {"name": "downswing", "start_frame": 35, "end_frame": 50, "duration": 15},
            {"name": "impact", "start_frame": 50, "end_frame": 55, "duration": 5},
            {"name": "follow_through", "start_frame": 55, "end_frame": 80, "duration": 25}
        ],
        "contact_moment": {
            "frame_number": 52,
            "confidence": 0.85,
            "wrist_velocity": 95.2,
            "elbow_acceleration": 45.1
        },
        "smoothness_metrics": {
            "smoothness_score": 0.75,
            "consistency_score": 0.68
        },
        "performance_stats": {
            "total_frames": 120,
            "total_processing_time": 12.5,
            "avg_fps": 9.6,
            "avg_frame_processing_time": 0.104
        }
    }
    
    reports = generate_reports(sample_data, "output")
    print(f"Test reports generated: {reports}")