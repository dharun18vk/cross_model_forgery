import os
import warnings
import io
import json
import subprocess
from datetime import datetime
import contextlib
from pathlib import Path
from typing import Final, List, Dict, Any, Optional, Tuple
import hashlib
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from collections import defaultdict
import retinaface
retinaface.__version__ = "0.0.1"
from retinaface import RetinaFace

plt.set_loglevel('warning')

from collections import deque
import sys
from pathlib import Path
try:
    from preprocessing.model_lipreading import LipReadingModel
    from preprocessing.dataset_lip import build_vocab
    from preprocessing.extract_mouth import extract_mouth_frame
    import editdistance
except ImportError:
    st.warning("Lip sync modules not available. Lip sync analysis will be limited.")

# =============================================================================
# WEIGHTED ENSEMBLE MODEL CONFIGURATION
# =============================================================================
WEIGHTED_ENSEMBLE_MODELS = {
    # Primary weighted ensemble models
    "Celeb + Image Model (Level 2.5 - Advanced)": {
        "path": "DEEPFAKE_MODELS/2.5-df.pth",
        "weight": 0.50
    },
    "Xception Progressive Fine-tuned (Level 2)": {
        "path": "DEEPFAKE_MODELS/progressive_fine_tuned_model/2nd_tuned_xception_model.pth",
        "weight": 0.30
    },
    "New Deepfake Model (Experimental – Epoch 10)": {
        "path": "DEEPFAKE_MODELS/new_deepfake_model/checkpoint_epoch_10.pth",
        "weight": 0.20
    }
}

# Safety layer model (must also say REAL for final REAL verdict)
SAFETY_MODEL = {
    "Xception Model (Deployment Ready)": {
        "path": "DEEPFAKE_MODELS/xception_deepfake_model/best_face_model.pth",
        "weight": 0.0  # Not part of weighted score, only for safety check
    }
}

# Models to exclude (biased)
EXCLUDED_MODELS = [
    "Celeb-DF Fine-tuned (Best Model)",
    "Base FF++ Stage-3 Model", 
    "Xception Fine-tuned (Level 1)",
    "FF++ Stage-3 Training Checkpoint (Latest)"
]

# =============================================================================
# LIP SYNC CONFIGURATION
# =============================================================================
LIPSYNC_CONFIG = {
    "MODEL_PATH": "preprocessing/models/lip_model_best.pth",
    "IMG_SIZE": 64,
    "SEQ_LEN": 25,
    "CER_THRESHOLD": 0.35,
    "CONF_THRESHOLD": 0.45,
    "FREEZE_LIMIT": 10
}

# =============================================================================
# PLOTLY THEME CONFIGURATION
# =============================================================================

# Plotly theme colors
CATEGORY_0: Final = "#000001"
CATEGORY_1: Final = "#000002"
CATEGORY_2: Final = "#000003"
CATEGORY_3: Final = "#000004"
CATEGORY_4: Final = "#000005"
CATEGORY_5: Final = "#000006"
CATEGORY_6: Final = "#000007"
CATEGORY_7: Final = "#000008"
CATEGORY_8: Final = "#000009"
CATEGORY_9: Final = "#000010"

SEQUENTIAL_0: Final = "#000011"
SEQUENTIAL_1: Final = "#000012"
SEQUENTIAL_2: Final = "#000013"
SEQUENTIAL_3: Final = "#000014"
SEQUENTIAL_4: Final = "#000015"
SEQUENTIAL_5: Final = "#000016"
SEQUENTIAL_6: Final = "#000017"
SEQUENTIAL_7: Final = "#000018"
SEQUENTIAL_8: Final = "#000019"
SEQUENTIAL_9: Final = "#000020"

DIVERGING_0: Final = "#000021"
DIVERGING_1: Final = "#000022"
DIVERGING_2: Final = "#000023"
DIVERGING_3: Final = "#000024"
DIVERGING_4: Final = "#000025"
DIVERGING_5: Final = "#000026"
DIVERGING_6: Final = "#000027"
DIVERGING_7: Final = "#000028"
DIVERGING_8: Final = "#000029"
DIVERGING_9: Final = "#000030"
DIVERGING_10: Final = "#000031"

INCREASING: Final = "#000032"
DECREASING: Final = "#000033"
TOTAL: Final = "#000034"

GRAY_70: Final = "#000036"
GRAY_90: Final = "#000037"
BG_COLOR: Final = "#000038"
FADED_TEXT_05: Final = "#000039"
BG_MIX: Final = "#000040"


def configure_streamlit_plotly_theme() -> None:
    """Configure the Streamlit chart theme for Plotly."""
    with contextlib.suppress(ImportError):
        import plotly.graph_objects as go
        import plotly.io as pio

        streamlit_colorscale = [
            [0.0, SEQUENTIAL_0],
            [0.1111111111111111, SEQUENTIAL_1],
            [0.2222222222222222, SEQUENTIAL_2],
            [0.3333333333333333, SEQUENTIAL_3],
            [0.4444444444444444, SEQUENTIAL_4],
            [0.5555555555555556, SEQUENTIAL_5],
            [0.6666666666666666, SEQUENTIAL_6],
            [0.7777777777777778, SEQUENTIAL_7],
            [0.8888888888888888, SEQUENTIAL_8],
            [1.0, SEQUENTIAL_9],
        ]

        pio.templates["streamlit"] = go.layout.Template(
            data=go.layout.template.Data(
                candlestick=[
                    go.layout.template.data.Candlestick(
                        decreasing=go.candlestick.Decreasing(
                            line=go.candlestick.decreasing.Line(color=DECREASING)
                        ),
                        increasing=go.candlestick.Increasing(
                            line=go.candlestick.increasing.Line(color=INCREASING)
                        ),
                    )
                ],
                contour=[
                    go.layout.template.data.Contour(colorscale=streamlit_colorscale)
                ],
                contourcarpet=[
                    go.layout.template.data.Contourcarpet(
                        colorscale=streamlit_colorscale
                    )
                ],
                heatmap=[
                    go.layout.template.data.Heatmap(colorscale=streamlit_colorscale)
                ],
                histogram2d=[
                    go.layout.template.data.Histogram2d(colorscale=streamlit_colorscale)
                ],
                icicle=[
                    go.layout.template.data.Icicle(
                        textfont=go.icicle.Textfont(color="white")
                    )
                ],
                sankey=[
                    go.layout.template.data.Sankey(
                        textfont=go.sankey.Textfont(color=GRAY_70)
                    )
                ],
                scatter=[
                    go.layout.template.data.Scatter(
                        marker=go.scatter.Marker(line=go.scatter.marker.Line(width=0))
                    )
                ],
                table=[
                    go.layout.template.data.Table(
                        cells=go.table.Cells(
                            fill=go.table.cells.Fill(color=BG_COLOR),
                            font=go.table.cells.Font(color=GRAY_90),
                            line=go.table.cells.Line(color=FADED_TEXT_05),
                        ),
                        header=go.table.Header(
                            font=go.table.header.Font(color=GRAY_70),
                            line=go.table.header.Line(color=FADED_TEXT_05),
                            fill=go.table.header.Fill(color=BG_MIX),
                        ),
                    )
                ],
                waterfall=[
                    go.layout.template.data.Waterfall(
                        increasing=go.waterfall.Increasing(
                            marker=go.waterfall.increasing.Marker(color=INCREASING)
                        ),
                        decreasing=go.waterfall.Decreasing(
                            marker=go.waterfall.decreasing.Marker(color=DECREASING)
                        ),
                        totals=go.waterfall.Totals(
                            marker=go.waterfall.totals.Marker(color=TOTAL)
                        ),
                        connector=go.waterfall.Connector(
                            line=go.waterfall.connector.Line(color=GRAY_70, width=2)
                        ),
                    )
                ],
            ),
            layout=go.Layout(
                colorway=[
                    CATEGORY_0,
                    CATEGORY_1,
                    CATEGORY_2,
                    CATEGORY_3,
                    CATEGORY_4,
                    CATEGORY_5,
                    CATEGORY_6,
                    CATEGORY_7,
                    CATEGORY_8,
                    CATEGORY_9,
                ],
                colorscale=go.layout.Colorscale(
                    sequential=streamlit_colorscale,
                    sequentialminus=streamlit_colorscale,
                    diverging=[
                        [0.0, DIVERGING_0],
                        [0.1, DIVERGING_1],
                        [0.2, DIVERGING_2],
                        [0.3, DIVERGING_3],
                        [0.4, DIVERGING_4],
                        [0.5, DIVERGING_5],
                        [0.6, DIVERGING_6],
                        [0.7, DIVERGING_7],
                        [0.8, DIVERGING_8],
                        [0.9, DIVERGING_9],
                        [1.0, DIVERGING_10],
                    ],
                ),
                coloraxis=go.layout.Coloraxis(colorscale=streamlit_colorscale),
            ),
        )

        pio.templates.default = "streamlit"


# =============================================================================
# CUSTOM JSON ENCODER
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and other non-serializable objects."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.datetime64, np.timedelta64)):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, (bytes, bytearray)):
            return obj.decode('utf-8', errors='ignore')
        return super(NumpyEncoder, self).default(obj)


# =============================================================================
# FORENSIC METADATA ANALYSIS (COMPLETE ANALYSIS)
# =============================================================================

def sha256_hash(file_path):
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"Error: {str(e)}"

def safe_eval(rate):
    """Safely evaluate fraction strings."""
    try:
        if '/' in str(rate):
            num, den = str(rate).split('/')
            return float(num) / float(den) if float(den) != 0 else None
        return float(rate)
    except:
        return None

class CompleteForensicMetadataAnalyzer:
    """Complete forensic metadata analysis with all details shown at bottom."""
    
    def __init__(self):
        self.meta = None
        self.video_stream = None
        self.audio_stream = None
        self.format_tags = None
        self.indicators = []
        self.verdict = "UNKNOWN"
        self.risk_score = 0
    
    def analyze_video(self, video_path):
        """Perform complete forensic metadata analysis."""
        try:
            # Run ffprobe to get metadata
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", 
                   "-show_format", "-show_streams", video_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                st.error(f"❌ FFprobe error: {result.stderr}")
                return None
            
            self.meta = json.loads(result.stdout)
            
            # Get video and audio streams
            for stream in self.meta.get("streams", []):
                if stream.get("codec_type") == "video":
                    self.video_stream = stream
                elif stream.get("codec_type") == "audio":
                    self.audio_stream = stream
            
            self.format_tags = self.meta.get("format", {}).get("tags", {})
            
            # Perform complete analysis
            analysis_results = self.perform_complete_analysis(video_path)
            
            return analysis_results
            
        except Exception as e:
            st.error(f"❌ Forensic analysis error: {str(e)}")
            return None
    
    def perform_complete_analysis(self, video_path):
        """Perform all analysis steps and return complete results."""
        analysis_results = {}
        
        # 1. File Integrity Analysis
        analysis_results["file_integrity"] = self.analyze_file_integrity(video_path)
        
        # 2. Creation & Author Metadata
        analysis_results["creation_metadata"] = self.analyze_creation_metadata()
        
        # 3. Encoding Information
        analysis_results["encoding_info"] = self.analyze_encoding_info()
        
        # 4. Frame & Duration Details
        analysis_results["frame_details"] = self.analyze_frame_details()
        
        # 5. Audio Information
        analysis_results["audio_info"] = self.analyze_audio_info()
        
        # 6. Change/Edit Detection
        analysis_results["change_detection"] = self.detect_changes_edits()
        
        # 7. Forensic Summary
        analysis_results["forensic_summary"] = self.generate_forensic_summary()
        
        return analysis_results
    
    def analyze_file_integrity(self, video_path):
        """Analyze file integrity and hash."""
        file_stats = os.stat(video_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        
        return {
            "file_path": video_path,
            "file_size_mb": round(file_size_mb, 2),
            "sha256_hash": sha256_hash(video_path),
            "file_extension": os.path.splitext(video_path)[1]
        }
    
    def analyze_creation_metadata(self):
        """Analyze creation and author metadata."""
        creation_time = self.format_tags.get("creation_time", "Not Available")
        artist = self.format_tags.get("artist", "Not Available")
        comment = self.format_tags.get("comment", "Not Available")
        encoder = self.format_tags.get("encoder", "Not Available")
        encoded_by = self.format_tags.get("encoded_by", "Not Available")
        
        # Check for missing metadata
        if creation_time == "Not Available":
            self.indicators.append("Missing creation timestamp")
            self.risk_score += 1
        
        return {
            "creation_time": creation_time,
            "author": artist,
            "comment": comment,
            "encoder": encoder,
            "encoded_by": encoded_by,
            "has_creation_time": creation_time != "Not Available",
            "has_author": artist != "Not Available",
            "has_comment": comment != "Not Available"
        }
    
    def analyze_encoding_info(self):
        """Analyze encoding information."""
        encoder = self.format_tags.get("encoder", "Not Available")
        encoded_by = self.format_tags.get("encoded_by", "Not Available")
        format_name = self.meta.get("format", {}).get("format_name", "Unknown")
        bitrate = self.meta.get("format", {}).get("bit_rate")
        duration = self.meta.get("format", {}).get("duration")
        
        # Check for FFmpeg re-encoding
        if "Lavf" in str(encoder):
            self.indicators.append("Re-encoded using FFmpeg")
            self.risk_score += 1
        
        return {
            "encoder": encoder,
            "encoded_by": encoded_by,
            "format_name": format_name,
            "bitrate": bitrate,
            "duration": duration,
            "is_ffmpeg_encoded": "Lavf" in str(encoder)
        }
    
    def analyze_frame_details(self):
        """Analyze frame and resolution details."""
        if not self.video_stream:
            return {}
        
        codec = self.video_stream.get("codec_name", "Unknown")
        width = self.video_stream.get("width", "?")
        height = self.video_stream.get("height", "?")
        avg_fps = safe_eval(self.video_stream.get("avg_frame_rate"))
        real_fps = safe_eval(self.video_stream.get("r_frame_rate"))
        total_frames = self.video_stream.get("nb_frames", "Unknown")
        pixel_format = self.video_stream.get("pix_fmt", "Unknown")
        
        # Check for FPS mismatch
        if avg_fps and real_fps and abs(avg_fps - real_fps) > 0.5:
            self.indicators.append(f"FPS mismatch (avg: {avg_fps:.2f}, real: {real_fps:.2f}) - possible frame edits")
            self.risk_score += 1
        
        # Check for missing frame count
        if not total_frames or total_frames == "Unknown":
            self.indicators.append("Frame count unavailable (possible VFR/edit)")
            self.risk_score += 1
        
        return {
            "codec": codec,
            "resolution": f"{width} x {height}",
            "width": width,
            "height": height,
            "avg_fps": avg_fps,
            "real_fps": real_fps,
            "total_frames": total_frames,
            "pixel_format": pixel_format,
            "has_fps_mismatch": avg_fps and real_fps and abs(avg_fps - real_fps) > 0.5
        }
    
    def analyze_audio_info(self):
        """Analyze audio information."""
        if not self.audio_stream:
            return {}
        
        audio_codec = self.audio_stream.get("codec_name", "Unknown")
        sample_rate = self.audio_stream.get("sample_rate", "Unknown")
        channels = self.audio_stream.get("channels", "Unknown")
        
        return {
            "audio_codec": audio_codec,
            "sample_rate": sample_rate,
            "channels": channels
        }
    
    def detect_changes_edits(self):
        """Detect potential changes and edits."""
        indicators = self.indicators.copy()
        
        # Additional checks
        if not self.video_stream:
            indicators.append("No video stream found")
            self.risk_score += 1
        
        # Check for suspicious encoder patterns
        encoder = self.format_tags.get("encoder", "").lower()
        suspicious_encoders = ["handbrake", "avidemux", "virtualdub", "adobe"]
        for sus_enc in suspicious_encoders:
            if sus_enc in encoder:
                indicators.append(f"Suspicious encoder detected: {encoder}")
                self.risk_score += 0.5
                break
        
        return {
            "indicators": indicators,
            "count": len(indicators),
            "risk_score": self.risk_score
        }
    
    def generate_forensic_summary(self):
        """Generate forensic summary and verdict."""
        # Determine verdict based on risk score
        if self.risk_score >= 3:
            verdict = "HIGH probability of modification"
            verdict_color = "#FF6B6B"
        elif self.risk_score == 2:
            verdict = "MODERATE probability of modification"
            verdict_color = "#FFD166"
        elif self.risk_score == 1:
            verdict = "LOW probability of modification"
            verdict_color = "#4ECDC4"
        else:
            verdict = "NO STRONG EVIDENCE of modification"
            verdict_color = "#06D6A0"
        
        self.verdict = verdict
        
        return {
            "verdict": verdict,
            "risk_score": self.risk_score,
            "indicators_count": len(self.indicators),
            "verdict_color": verdict_color,
            "confidence": "HIGH" if self.risk_score >= 3 else "MEDIUM" if self.risk_score >= 1 else "LOW"
        }
    
    def display_detailed_forensic_report(self, forensic_report, video_path):
        """Display the complete forensic metadata analysis in detailed format."""
        
        # 1. FILE INTEGRITY - Display exactly as in original code
        st.markdown("### [1] FILE INTEGRITY")
        
        file_integrity = forensic_report.get("file_integrity", {})
        file_hash = file_integrity.get('sha256_hash', '')
        
        # Create a formatted display like the original console output
        st.markdown(f"**File Path** : `{video_path}`")
        st.markdown(f"**SHA-256 Hash** : `{file_hash}`")
        
        st.markdown("---")
        
        # 2. CREATION & AUTHOR METADATA
        st.markdown("### [2] CREATION & AUTHOR METADATA")
        
        creation_meta = forensic_report.get("creation_metadata", {})
        creation_time = creation_meta.get('creation_time', 'Not Available')
        author = creation_meta.get('author', 'Not Available')
        comment = creation_meta.get('comment', 'Not Available')
        
        # Display in original format
        st.markdown(f"**Creation Time** : `{creation_time}`")
        st.markdown(f"**Author** : `{author}`")
        st.markdown(f"**Comment** : `{comment}`")
        
        st.markdown("---")
        
        # 3. ENCODING INFORMATION
        st.markdown("### [3] ENCODING INFORMATION")
        
        encoding_info = forensic_report.get("encoding_info", {})
        encoder = encoding_info.get('encoder', 'Not Available')
        encoded_by = encoding_info.get('encoded_by', 'Not Available')
        format_name = encoding_info.get('format_name', 'Unknown')
        bitrate = encoding_info.get('bitrate')
        duration = encoding_info.get('duration')
        
        # Display in original format
        st.markdown(f"**Encoder** : `{encoder}`")
        st.markdown(f"**Encoded By** : `{encoded_by}`")
        st.markdown(f"**Format** : `{format_name}`")
        
        if bitrate:
            bitrate_kbps = int(bitrate) / 1000
            st.markdown(f"**Bitrate** : `{bitrate_kbps:,.0f} kbps`")
        else:
            st.markdown(f"**Bitrate** : `Not Available`")
        
        st.markdown("---")
        
        # 4. FRAME & DURATION DETAILS
        st.markdown("### [4] FRAME & DURATION DETAILS")
        
        frame_details = forensic_report.get("frame_details", {})
        codec = frame_details.get('codec', 'Unknown')
        resolution = frame_details.get('resolution', 'Unknown')
        avg_fps = frame_details.get('avg_fps')
        real_fps = frame_details.get('real_fps')
        total_frames = frame_details.get('total_frames', 'Unknown')
        
        # Display in original format
        st.markdown(f"**Codec** : `{codec}`")
        st.markdown(f"**Resolution** : `{resolution}`")
        
        if avg_fps:
            st.markdown(f"**Average FPS** : `{avg_fps:.2f}`")
        else:
            st.markdown(f"**Average FPS** : `Not Available`")
            
        if real_fps:
            st.markdown(f"**Real FPS** : `{real_fps:.2f}`")
        else:
            st.markdown(f"**Real FPS** : `Not Available`")
        
        st.markdown(f"**Total Frames** : `{total_frames}`")
        
        if duration:
            st.markdown(f"**Duration (s)** : `{float(duration):.2f}`")
        else:
            st.markdown(f"**Duration (s)** : `Not Available`")
        
        st.markdown("---")
        
        # 5. CHANGE / EDIT DETECTION
        st.markdown("### [5] CHANGE / EDIT DETECTION")
        
        change_detection = forensic_report.get("change_detection", {})
        indicators = change_detection.get('indicators', [])
        
        if indicators:
            st.markdown("**Indicators Found:**")
            for i, indicator in enumerate(indicators, 1):
                # Check what type of indicator it is and display appropriately
                if "Missing creation timestamp" in indicator:
                    st.error(f"⚠ {indicator}")
                elif "FPS mismatch" in indicator:
                    st.error(f"⚠ {indicator}")
                elif "Re-encoded using FFmpeg" in indicator:
                    st.error(f"⚠ {indicator}")
                elif "Frame count unavailable" in indicator:
                    st.error(f"⚠ {indicator}")
                else:
                    st.warning(f"⚠ {indicator}")
        else:
            st.success("✓ No obvious metadata anomalies")
        
        st.markdown("---")
        
        # 6. FORENSIC SUMMARY
        st.markdown("### [6] FORENSIC SUMMARY")
        
        forensic_summary = forensic_report.get("forensic_summary", {})
        verdict = forensic_summary.get('verdict', 'UNKNOWN')
        risk_score = forensic_summary.get('risk_score', 0)
        
        # Calculate score based on indicators (matching original logic)
        score = len(indicators)
        
        # Determine verdict exactly as in original code
        if score >= 3:
            final_verdict = "HIGH probability of modification"
            verdict_color = "#FF6B6B"
            icon = "🔴"
        elif score == 2:
            final_verdict = "MODERATE probability of modification"
            verdict_color = "#FFD166"
            icon = "🟡"
        elif score == 1:
            final_verdict = "LOW probability of modification"
            verdict_color = "#4ECDC4"
            icon = "🔵"
        else:
            final_verdict = "NO STRONG EVIDENCE of modification"
            verdict_color = "#06D6A0"
            icon = "✅"
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔍 Indicators Found", score)
        with col2:
            st.metric("📊 Risk Score", risk_score)
        with col3:
            st.metric("🎯 Confidence", forensic_summary.get('confidence', 'UNKNOWN'))
        
        # Display final verdict
        st.markdown(f"""
        <div style="background: {verdict_color}; padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
            <h2 style="color: white; margin: 0;">{icon} Verdict: {final_verdict}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("=" * 70)
        st.success("✅ **Analysis completed successfully**")
        
        # Also display the original console-style output in expandable section
        with st.expander("📋 View Raw Console Output Format"):
            self.display_console_style_output(forensic_report, video_path, indicators, final_verdict)
    
    def display_console_style_output(self, forensic_report, video_path, indicators, verdict):
        """Display metadata in the exact console format from the original code."""
        file_integrity = forensic_report.get("file_integrity", {})
        creation_meta = forensic_report.get("creation_metadata", {})
        encoding_info = forensic_report.get("encoding_info", {})
        frame_details = forensic_report.get("frame_details", {})
        
        output = f"""
{'=' * 70}
VIDEO METADATA & FORENSIC ANALYSIS
{'=' * 70}

[1] FILE INTEGRITY
File Path     : {video_path}
SHA-256 Hash  : {file_integrity.get('sha256_hash', 'Not Available')}

[2] CREATION & AUTHOR METADATA
Creation Time : {creation_meta.get('creation_time', 'Not Available')}
Author        : {creation_meta.get('author', 'Not Available')}
Comment       : {creation_meta.get('comment', 'Not Available')}

[3] ENCODING INFORMATION
Encoder       : {encoding_info.get('encoder', 'Not Available')}
Encoded By    : {encoding_info.get('encoded_by', 'Not Available')}
Format        : {encoding_info.get('format_name', 'Unknown')}
Bitrate       : {encoding_info.get('bitrate', 'Not Available')}

[4] FRAME & DURATION DETAILS
Codec         : {frame_details.get('codec', 'Unknown')}
Resolution    : {frame_details.get('resolution', 'Unknown')}
Average FPS   : {frame_details.get('avg_fps', 'Not Available')}
Real FPS      : {frame_details.get('real_fps', 'Not Available')}
Total Frames  : {frame_details.get('total_frames', 'Unknown')}
Duration (s)  : {encoding_info.get('duration', 'Not Available')}

[5] CHANGE / EDIT DETECTION
"""
        
        if indicators:
            for indicator in indicators:
                output += f"⚠ {indicator}\n"
        else:
            output += "✓ No obvious metadata anomalies\n"
        
        output += f"""
[6] FORENSIC SUMMARY
Verdict: {verdict}

Analysis completed.
{'=' * 70}
"""
        
        st.code(output, language="text")

# =============================================================================
# METADATA ANALYSIS UTILITIES
# =============================================================================

def run_ffprobe_json(path: str):
    """Run ffprobe and return parsed JSON"""
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path]
        out = subprocess.check_output(cmd)
        return json.loads(out)
    except Exception as e:
        st.error(f"❌ FFprobe error for {path}: {e}")
        return None

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def normalize_rate(rate_str):
    """Convert ffprobe rate string like '30000/1001' to float"""
    if not rate_str or rate_str == "0/0":
        return None
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/")
            return float(num) / float(den)
        return float(rate_str)
    except Exception:
        return None

def extract_features(path: str):
    """Extract metadata features from video using ffprobe"""
    j = run_ffprobe_json(path)
    if j is None:
        return None
        
    fmt = j.get("format", {}) or {}
    streams = j.get("streams", []) or []
    v = next((s for s in streams if s.get("codec_type") == "video"), {}) or {}
    a = next((s for s in streams if s.get("codec_type") == "audio"), {}) or {}

    features = {
        "path": path,
        "format_name": fmt.get("format_name", ""),
        "format_bit_rate": safe_float(fmt.get("bit_rate")),
        "format_duration": safe_float(fmt.get("duration")),
        "v_codec": v.get("codec_name", ""),
        "v_width": int(v.get("width") or 0),
        "v_height": int(v.get("height") or 0),
        "v_bit_rate": safe_float(v.get("bit_rate")),
        "v_avg_frame_rate": normalize_rate(v.get("avg_frame_rate")),
        "a_codec": a.get("codec_name", ""),
        "a_sample_rate": safe_float(a.get("sample_rate")),
        "a_bit_rate": safe_float(a.get("bit_rate")),
        "encoder": v.get("tags", {}).get("encoder", "") if v.get("tags") else "",
    }
    features["has_audio"] = bool(features["a_codec"])
    return features

def most_common(series):
    vals = series.dropna().astype(str).tolist()
    if not vals:
        return None
    from collections import Counter
    return Counter(vals).most_common(1)[0][0]

def build_baseline_from_features(df):
    baseline = {
        "n_videos": len(df),
        "median_format_bit_rate": float(np.nanmedian(df["format_bit_rate"].dropna())) if "format_bit_rate" in df else None,
        "median_v_bit_rate": float(np.nanmedian(df["v_bit_rate"].dropna())) if "v_bit_rate" in df else None,
        "median_width": int(np.nanmedian(df["v_width"].dropna())) if "v_width" in df else None,
        "median_height": int(np.nanmedian(df["v_height"].dropna())) if "v_height" in df else None,
        "median_frame_rate": float(np.nanmedian(df["v_avg_frame_rate"].dropna())) if "v_avg_frame_rate" in df else None,
        "most_common_v_codec": most_common(df["v_codec"]),
        "most_common_a_codec": most_common(df["a_codec"]),
        "most_common_format_name": most_common(df["format_name"]),
    }
    return baseline

# Thresholds for metadata comparison
BITRATE_RATIO_THRESHOLD = 0.6
FORMAT_BITRATE_RATIO_THRESHOLD = 0.6
FRAME_RATE_DIFF_THRESHOLD = 1.5
DURATION_RATIO_THRESHOLD = 0.9

def compare_against_baseline(feat, baseline):
    issues = []

    # 1) Codec change
    if baseline.get("most_common_v_codec") and feat.get("v_codec") != baseline["most_common_v_codec"]:
        issues.append({"type": "v_codec_changed", "message": f"Codec changed from {baseline['most_common_v_codec']} to {feat['v_codec']}"})

    # 2) Resolution
    if baseline.get("median_width") and (feat["v_width"], feat["v_height"]) != (baseline["median_width"], baseline["median_height"]):
        issues.append({"type": "resolution_changed", "message": f"Resolution differs (baseline {baseline['median_width']}x{baseline['median_height']}, found {feat['v_width']}x{feat['v_height']})"})

    # 3) Bitrate difference
    if baseline.get("median_v_bit_rate") and feat.get("v_bit_rate"):
        ratio = feat["v_bit_rate"] / baseline["median_v_bit_rate"]
        if ratio < BITRATE_RATIO_THRESHOLD or ratio > 1.8:
            issues.append({"type": "v_bitrate_changed", "message": f"Bitrate differs significantly (ratio {ratio:.2f})"})

    # 4) Audio missing
    if not feat.get("has_audio"):
        issues.append({"type": "audio_missing", "message": "Audio stream missing"})

    # 5) Frame rate
    if baseline.get("median_frame_rate") and feat.get("v_avg_frame_rate"):
        diff = abs(baseline["median_frame_rate"] - feat["v_avg_frame_rate"])
        if diff > FRAME_RATE_DIFF_THRESHOLD:
            issues.append({"type": "frame_rate_changed", "message": f"Frame rate differs (baseline {baseline['median_frame_rate']}, found {feat['v_avg_frame_rate']})"})

    summary = "✅ No major metadata differences found." if not issues else f"⚠️ {len(issues)} differences detected."
    return {"summary": summary, "issues": issues}

# =============================================================================
# FRAME EXTRACTOR
# =============================================================================

class FrameExtractor:
    """Extract frames from video files."""
    def __init__(self, output_dir="extracted_frames"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_frames(self, video_path, frames_per_second=1):
        """Extract frames from video at specified rate."""
        with st.status("🎬 **Extracting frames from video...**", expanded=True) as status:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"❌ Error opening video file: {video_path}")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 FPS", f"{fps:.2f}")
            with col2:
                st.metric("📊 Total Frames", f"{total_frames}")
            with col3:
                st.metric("⏱️ Duration", f"{duration:.2f}s")
            
            frame_interval = max(1, int(fps / frames_per_second))
            extracted_frames = []
            frame_count = 0
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_frames_dir = os.path.join(self.output_dir, video_name)
            os.makedirs(video_frames_dir, exist_ok=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = os.path.join(video_frames_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    extracted_frames.append({
                        'path': frame_path,
                        'frame_number': int(frame_count),
                        'timestamp': float(frame_count / fps)
                    })
                
                frame_count += 1
                if frame_count % 50 == 0:
                    progress = min(float(frame_count / total_frames), 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            progress_bar.progress(1.0)
            status.update(label=f"✅ **Frame Extraction Complete** - Extracted {len(extracted_frames)} frames", state="complete")
        return extracted_frames

# =============================================================================
# FACE DETECTOR
# =============================================================================

class RetinaFaceDetector:
    """Detect faces in frames using RetinaFace."""
    def __init__(self):
        self.face_output_dir = "extracted_faces"
        os.makedirs(self.face_output_dir, exist_ok=True)
    
    def detect_and_extract_faces(self, frame_paths, min_face_size=40, confidence_threshold=0.9):
        """Detect and extract faces from frames."""
        with st.status("👤 **Detecting and extracting faces using RetinaFace...**", expanded=True) as status:
            all_faces = []
            face_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, frame_info in enumerate(frame_paths):
                frame_path = frame_info['path']
                
                try:
                    faces = RetinaFace.detect_faces(frame_path)
                    
                    if faces and isinstance(faces, dict):
                        for face_id, face_info in faces.items():
                            facial_area = face_info['facial_area']
                            score = face_info['score']
                            
                            if score >= confidence_threshold:
                                x1, y1, x2, y2 = facial_area
                                face_width = x2 - x1
                                face_height = y2 - y1
                                
                                if face_width >= min_face_size and face_height >= min_face_size:
                                    image = Image.open(frame_path).convert('RGB')
                                    padding = 20
                                    x1_pad = max(0, x1 - padding)
                                    y1_pad = max(0, y1 - padding)
                                    x2_pad = min(image.width, x2 + padding)
                                    y2_pad = min(image.height, y2 + padding)
                                    
                                    face_image = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))
                                    face_filename = f"face_{face_count:06d}.jpg"
                                    face_path = os.path.join(self.face_output_dir, face_filename)
                                    face_image.save(face_path)
                                    
                                    face_info = {
                                        'face_path': face_path,
                                        'frame_path': frame_path,
                                        'frame_number': int(frame_info['frame_number']),
                                        'timestamp': float(frame_info['timestamp']),
                                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                        'confidence': float(score),
                                        'face_id': int(face_count)
                                    }
                                    all_faces.append(face_info)
                                    face_count += 1
                    
                except Exception as e:
                    continue
                
                progress = float((i + 1) / len(frame_paths))
                progress_bar.progress(progress)
                status_text.text(f"🔍 Processed {i+1}/{len(frame_paths)} frames - Found {face_count} faces")
            
            status.update(label=f"✅ **Face Detection Complete** - Detected {len(all_faces)} faces", state="complete")
            
            # Show face detection summary
            if all_faces:
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"👥 **Total Faces Found:** {len(all_faces)}")
                with col2:
                    avg_confidence = np.mean([face['confidence'] for face in all_faces])
                    st.info(f"🎯 **Average Confidence:** {avg_confidence:.3f}")
        return all_faces

# =============================================================================
# DEEPFAKE DETECTOR WITH WEIGHTED ENSEMBLE
# =============================================================================

class DeepFakeDetector:
    """Deepfake detection using weighted ensemble of models."""
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ensemble_models = {}
        self.safety_model = None
        self.transform = self.get_transform()
        self.load_ensemble_models()
    
    def load_model(self, model_path, model_name=""):
        """Load a single deepfake detection model."""
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            return None
        
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load with strict=False to handle architecture differences
            model.load_state_dict(state_dict, strict=False)
            
        except Exception as e:
            st.error(f"❌ Error loading model {model_name}: {e}")
            return None
        
        model.to(self.device)
        model.eval()
        return model
    
    def load_ensemble_models(self):
        """Load all weighted ensemble models and safety model."""
        # Load weighted ensemble models
        for model_name, model_info in WEIGHTED_ENSEMBLE_MODELS.items():
            with st.spinner(f"Loading {model_name}..."):
                model = self.load_model(model_info["path"], model_name)
                if model:
                    self.ensemble_models[model_name] = {
                        "model": model,
                        "weight": model_info["weight"]
                    }
        
        # Load safety model
        safety_name = list(SAFETY_MODEL.keys())[0]
        safety_info = SAFETY_MODEL[safety_name]
        with st.spinner(f"Loading safety model {safety_name}..."):
            self.safety_model = self.load_model(safety_info["path"], safety_name)
    
    def get_transform(self):
        """Get image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_single_face(self, face_path):
        """Predict if a single face is real or fake using weighted ensemble."""
        if not self.ensemble_models:
            return None
            
        try:
            image = Image.open(face_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions from all ensemble models
            ensemble_predictions = []
            model_details = {}
            
            for model_name, model_info in self.ensemble_models.items():
                model = model_info["model"]
                weight = model_info["weight"]
                
                with torch.no_grad():
                    output = model(image_tensor)
                    probability = output.item()
                    prediction = "FAKE" if probability > 0.5 else "REAL"
                    confidence = probability if prediction == "FAKE" else 1 - probability
                    
                    model_details[model_name] = {
                        'prediction': prediction,
                        'confidence': float(confidence),
                        'fake_probability': float(probability),
                        'weight': float(weight)
                    }
                    
                    # Calculate weighted score contribution
                    weighted_score = probability * weight
                    ensemble_predictions.append(weighted_score)
            
            # Calculate weighted ensemble score
            weighted_score = sum(ensemble_predictions)
            
            # Get safety model prediction if available
            safety_prediction = None
            if self.safety_model:
                with torch.no_grad():
                    safety_output = self.safety_model(image_tensor)
                    safety_probability = safety_output.item()
                    safety_prediction = "FAKE" if safety_probability > 0.5 else "REAL"
            
            # Apply decision rule: score ≥ 0.5 → FAKE
            ensemble_prediction = "FAKE" if weighted_score >= 0.5 else "REAL"
            
            # Apply safety layer: If ensemble says REAL, require safety model to also say REAL
            if ensemble_prediction == "REAL" and safety_prediction == "FAKE":
                ensemble_prediction = "FAKE"  # Override to FAKE if safety model disagrees
            
            return {
                'prediction': ensemble_prediction,
                'weighted_score': float(weighted_score),
                'model_details': model_details,
                'safety_prediction': safety_prediction,
                'safety_override': (ensemble_prediction == "REAL" and safety_prediction == "FAKE"),
                'is_ensemble': True
            }
        
        except Exception as e:
            return None
    
    def predict_multiple_faces(self, face_data_list):
        """Predict multiple faces using weighted ensemble."""
        if not self.ensemble_models:
            return []
            
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, face_data in enumerate(face_data_list):
            result = self.predict_single_face(face_data['face_path'])
            if result:
                # Add face metadata to result
                result.update({
                    'frame_number': face_data['frame_number'],
                    'timestamp': face_data['timestamp'],
                    'face_id': face_data['face_id'],
                    'bbox': face_data['bbox'],
                    'detection_confidence': face_data['confidence']
                })
                results.append(result)
            
            progress = float((i + 1) / len(face_data_list))
            progress_bar.progress(progress)
            if (i + 1) % 10 == 0 or (i + 1) == len(face_data_list):
                status_text.text(f"🧪 Analyzed {i+1}/{len(face_data_list)} faces")
        
        return results

# =============================================================================
# METADATA ANALYZER
# =============================================================================

class MetadataAnalyzer:
    def __init__(self):
        self.video_exts = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi")
        self.baselines_dir = "baselines"
        self.reports_dir = "metadata_reports"
        os.makedirs(self.baselines_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def create_metadata_report(self, video_path, baseline_name=None):
        """Create comprehensive metadata analysis report"""
        features = extract_features(video_path)
        if features is None:
            return None
        
        # If baseline is provided, compare against it
        baseline_comparison = None
        if baseline_name:
            baseline_path = os.path.join(self.baselines_dir, f"{baseline_name}.json")
            if os.path.exists(baseline_path):
                with open(baseline_path, "r", encoding="utf-8") as f:
                    baseline = json.load(f)
                baseline_comparison = compare_against_baseline(features, baseline)
        
        # Analyze for general anomalies
        anomalies = self.analyze_metadata_anomalies(features)
        
        report = {
            "video": os.path.basename(video_path),
            "video_path": video_path,
            "analyzed_at": datetime.utcnow().isoformat() + "Z",
            "metadata_features": features,
            "anomalies": anomalies,
            "baseline_comparison": baseline_comparison,
            "anomaly_count": len(anomalies),
            "summary": f"Found {len(anomalies)} metadata anomalies" if anomalies else "No significant metadata anomalies detected"
        }
        
        if baseline_comparison:
            report["baseline_issues"] = len(baseline_comparison.get("issues", []))
            report["summary"] += f" | {baseline_comparison['summary']}"
        
        # Save report
        stem = os.path.splitext(os.path.basename(video_path))[0]
        json_path = os.path.join(self.reports_dir, f"{stem}_metadata_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def analyze_metadata_anomalies(self, features):
        """Analyze metadata for potential tampering indicators"""
        anomalies = []
        
        # Check for common re-encoding indicators
        if features.get('encoder'):
            encoder_lower = features['encoder'].lower()
            if 'lavf' in encoder_lower or 'lavc' in encoder_lower:
                anomalies.append({
                    'type': 'reencoding_indicator',
                    'message': f'Video encoded with FFmpeg (encoder: {features["encoder"]}) - possible re-encoding',
                    'severity': 'medium'
                })
        
        # Check for unusual codec combinations
        common_codecs = ['h264', 'h265', 'hevc', 'vp9', 'avc']
        if features.get('v_codec') and features['v_codec'].lower() not in common_codecs:
            anomalies.append({
                'type': 'unusual_codec',
                'message': f'Unusual video codec: {features["v_codec"]}',
                'severity': 'low'
            })
        
        # Check for missing common metadata
        if not features.get('v_bit_rate') or features['v_bit_rate'] == 0:
            anomalies.append({
                'type': 'missing_bitrate',
                'message': 'Video bitrate information missing',
                'severity': 'low'
            })
        
        # Check for extremely low bitrates (potential compression artifacts)
        if features.get('v_bit_rate') and features['v_bit_rate'] < 100000:  # Less than 100 kbps
            bitrate_str = f"{features['v_bit_rate']:,.0f}"
            anomalies.append({
                'type': 'low_bitrate',
                'message': f'Very low video bitrate: {bitrate_str} bps - potential heavy compression',
                'severity': 'medium'
            })
            
        return anomalies
    
    def get_available_baselines(self):
        """Get list of available baselines"""
        if not os.path.exists(self.baselines_dir):
            return []
        return [f.replace('.json', '') for f in os.listdir(self.baselines_dir) if f.endswith('.json')]
    
    def build_baseline_from_folder(self, folder_path, baseline_name):
        """Build baseline from folder of videos"""
        rows = []
        for fname in os.listdir(folder_path):
            path = os.path.join(folder_path, fname)
            if os.path.isfile(path) and any(path.lower().endswith(ext) for ext in self.video_exts):
                try:
                    feat = extract_features(path)
                    if feat:
                        rows.append(feat)
                except Exception as e:
                    print(f"Skip: {fname} - {e}")

        if not rows:
            raise Exception(f"No valid videos found in: {folder_path}")

        df = pd.DataFrame(rows)
        baseline = build_baseline_from_features(df)
        baseline["name"] = baseline_name
        baseline["built_at"] = datetime.utcnow().isoformat() + "Z"
        
        os.makedirs(self.baselines_dir, exist_ok=True)
        baseline_path = os.path.join(self.baselines_dir, f"{baseline_name}.json")
        
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)
        
        return baseline

# =============================================================================
# DEEPFAKE ANALYSIS PIPELINE (Weighted Ensemble)
# =============================================================================

class DeepfakeAnalysisPipeline:
    """Pipeline for deepfake analysis using weighted ensemble."""
    def __init__(self):
        self.frame_extractor = FrameExtractor()
        self.face_detector = RetinaFaceDetector()
        self.detector = DeepFakeDetector()
        self.results_dir = "deepfake_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def analyze_video(self, video_path, frames_per_second=1):
        """Analyze video with weighted ensemble deepfake detection."""
        # Main header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
            <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">🤖 Deepfake Detection (Weighted Ensemble)</h1>
            <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Weighted Ensemble: Celeb 2.5 (50%) + Xception L2 (30%) + New DF Epoch10 (20%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display ensemble configuration
        st.markdown("### 🎯 Ensemble Configuration")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Celeb 2.5", "50%", "Primary Model")
        with col2:
            st.metric("Xception L2", "30%", "Support Model")
        with col3:
            st.metric("New DF E10", "20%", "Experimental")
        with col4:
            st.metric("Safety Layer", "Active", "Xception Deployment")
        
        st.info("**Decision Rule:** Weighted score ≥ 0.5 → FAKE | **Safety:** REAL requires Xception Deployment confirmation")
        
        # Step 1: Frame Extraction
        with st.container():
            st.markdown("### 🎬 STEP 1: Frame Extraction")
            frames = self.frame_extractor.extract_frames(video_path, frames_per_second)
            if not frames:
                st.error("❌ No frames extracted. Exiting.")
                return None
        
        # Step 2: Face Detection
        with st.container():
            st.markdown("### 👤 STEP 2: Face Detection")
            faces = self.face_detector.detect_and_extract_faces(frames)
            if not faces:
                st.error("❌ No faces detected. Exiting.")
                return None
        
        # Step 3: Deepfake Analysis
        with st.container():
            st.markdown("### 🧪 STEP 3: Weighted Ensemble Analysis")
            with st.spinner("Analyzing faces with weighted ensemble..."):
                predictions = self.detector.predict_multiple_faces(faces)
            
            if not predictions:
                st.error("❌ No predictions generated.")
                return None
            
            # Generate report
            report = self.generate_report(predictions, video_path)
        
        # Step 4: Display Results
        with st.container():
            st.markdown("### 📊 STEP 4: Analysis Results")
            self.display_results(report, video_path)
        
        return report
    
    def generate_report(self, predictions, video_path):
        """Generate analysis report."""
        total_faces = len(predictions)
        
        if total_faces == 0:
            st.error("❌ No faces were successfully analyzed.")
            return None
        
        # Calculate statistics
        fake_faces = sum(1 for pred in predictions if pred['prediction'] == 'FAKE')
        real_faces = total_faces - fake_faces
        
        weighted_scores = [pred['weighted_score'] for pred in predictions]
        avg_weighted_score = float(np.mean(weighted_scores)) if weighted_scores else 0.0
        
        # Model contributions
        model_contributions = {}
        for pred in predictions:
            for model_name, details in pred.get('model_details', {}).items():
                if model_name not in model_contributions:
                    model_contributions[model_name] = {
                        'fake_count': 0,
                        'real_count': 0,
                        'total': 0,
                        'weight': details['weight']
                    }
                
                model_contributions[model_name]['total'] += 1
                if details['prediction'] == 'FAKE':
                    model_contributions[model_name]['fake_count'] += 1
                else:
                    model_contributions[model_name]['real_count'] += 1
        
        # Safety layer statistics
        safety_overrides = sum(1 for pred in predictions if pred.get('safety_override', False))
        
        report = {
            'video_path': video_path,
            'analysis_timestamp': str(np.datetime64('now')),
            'analysis_mode': 'Weighted Ensemble Deepfake Detection',
            'ensemble_config': {
                'models': {k: v['weight'] for k, v in WEIGHTED_ENSEMBLE_MODELS.items()},
                'safety_model': list(SAFETY_MODEL.keys())[0],
                'decision_threshold': 0.5,
                'safety_rule': 'REAL requires safety model confirmation'
            },
            'results_summary': {
                'total_faces_analyzed': int(total_faces),
                'fake_faces_detected': int(fake_faces),
                'real_faces_detected': int(real_faces),
                'fake_percentage': float((fake_faces / total_faces * 100) if total_faces > 0 else 0),
                'avg_weighted_score': float(avg_weighted_score),
                'safety_overrides': int(safety_overrides),
                'final_verdict': "LIKELY FAKE" if fake_faces > real_faces else "LIKELY REAL"
            },
            'model_contributions': model_contributions,
            'detailed_predictions': predictions
        }
        
        # Save report
        report_path = os.path.join(
            self.results_dir, 
            f"deepfake_{os.path.basename(video_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            st.success(f"✅ Deepfake report saved: {report_path}")
        except Exception as e:
            st.error(f"❌ Error saving report: {e}")
        
        return report
    
    def display_results(self, report, video_path):
        """Display analysis results."""
        if not report:
            return
        
        summary = report.get('results_summary', {})
        
        # Display key metrics
        st.markdown("### 🎯 Detection Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Faces", summary.get('total_faces_analyzed', 0))
        with col2:
            fake_percentage = summary.get('fake_percentage', 0)
            st.metric("Fake %", f"{fake_percentage:.1f}%")
        with col3:
            avg_score = summary.get('avg_weighted_score', 0)
            st.metric("Avg Weighted Score", f"{avg_score:.3f}")
        with col4:
            safety_overrides = summary.get('safety_overrides', 0)
            st.metric("Safety Overrides", safety_overrides)
        
        # Display final verdict
        verdict = summary.get('final_verdict', 'UNKNOWN')
        fake_percentage = summary.get('fake_percentage', 0)
        
        if verdict == "LIKELY FAKE":
            verdict_color = "#FF6B6B"
            icon = "❌"
            message = f"FAKE DETECTED: {fake_percentage:.1f}% of faces show deepfake artifacts"
        else:
            verdict_color = "#06D6A0"
            icon = "✅"
            message = f"REAL: Only {fake_percentage:.1f}% of faces show potential artifacts"
        
        st.markdown(f"""
        <div style="background: {verdict_color}; padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
            <h2 style="color: white; margin: 0;">{icon} Final Verdict: {verdict}</h2>
            <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.1rem;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model contributions visualization
        st.markdown("### 🤖 Model Contributions")
        
        model_data = []
        for model_name, stats in report.get('model_contributions', {}).items():
            if stats['total'] > 0:
                fake_pct = (stats['fake_count'] / stats['total'] * 100) if stats['total'] > 0 else 0
                model_data.append({
                    'Model': model_name,
                    'Weight': stats['weight'],
                    'Fake %': fake_pct,
                    'Total Faces': stats['total']
                })
        
        if model_data:
            df_models = pd.DataFrame(model_data)
            fig = px.bar(
                df_models,
                x='Model',
                y='Fake %',
                color='Weight',
                title='Model Contributions to Fake Detection',
                labels={'Fake %': 'Fake Faces (%)'},
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Weighted score distribution
        st.markdown("### 📊 Weighted Score Distribution")
        weighted_scores = [pred['weighted_score'] for pred in report.get('detailed_predictions', [])]
        
        if weighted_scores:
            fig_hist = px.histogram(
                x=weighted_scores,
                nbins=20,
                title='Distribution of Weighted Scores',
                labels={'x': 'Weighted Score', 'y': 'Count'},
                color_discrete_sequence=['#667eea']
            )
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", 
                             annotation_text="Decision Threshold", 
                             annotation_position="top")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Add statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Score", f"{np.mean(weighted_scores):.3f}")
            with col2:
                st.metric("Std Dev", f"{np.std(weighted_scores):.3f}")
            with col3:
                above_threshold = sum(1 for s in weighted_scores if s >= 0.5)
                st.metric("Above Threshold", f"{above_threshold}/{len(weighted_scores)}")

# =============================================================================
# LIP SYNC ANALYSIS
# =============================================================================

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

@st.cache_resource
def load_model():
    # Force GPU only - raise error if CUDA not available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA (GPU) is not available. This application requires GPU for lip sync analysis.")
    
    device = torch.device("cuda")
    st.info(f"🔥 Using device: {device}")
    
    char_to_idx, idx_to_char = build_vocab()
    model = LipReadingModel(len(char_to_idx) + 1).to(device)
    model.load_state_dict(torch.load(LIPSYNC_CONFIG["MODEL_PATH"], map_location=device))
    model.eval()
    
    return model, device, idx_to_char

def greedy_decode(probs, idx_to_char):
    blank = 0
    prev = None
    out = []

    for p in probs.argmax(dim=-1):
        p = p.item()
        if p != blank and p != prev:
            out.append(idx_to_char.get(p, ""))
        prev = p

    return "".join(out)

def cer(a, b):
    if len(b) == 0:
        return 0.0
    import editdistance
    return editdistance.eval(a, b) / len(b)

def language_quality(text):
    words = text.split()
    if len(words) == 0:
        return 0.0
    avg_len = sum(len(w) for w in words) / len(words)
    return avg_len

class LipSyncAnalyzer:
    """Lip synchronization analysis for deepfake detection."""
    
    def __init__(self):
        self.config = LIPSYNC_CONFIG
        self.model = None
        self.device = None
        self.idx_to_char = None
        self.results_dir = "lip_sync_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_model_wrapper(self):
        """Load lip reading model with GPU only."""
        try:
            self.model, self.device, self.idx_to_char = load_model()
            return True
        except RuntimeError as e:
            st.error(f"❌ {str(e)}")
            return False
        except Exception as e:
            st.error(f"❌ Error loading lip sync model: {e}")
            return False
    
    def analyze_video(self, video_path, cer_threshold=None, conf_threshold=None, freeze_limit=None):
        """Perform lip sync analysis on video."""
        
        # Main header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
            <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">👄 Lip Sync Analysis</h1>
            <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Audio-Visual Synchronization Verification</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use provided thresholds or defaults
        cer_threshold = cer_threshold or self.config["CER_THRESHOLD"]
        conf_threshold = conf_threshold or self.config["CONF_THRESHOLD"]
        freeze_limit = freeze_limit or self.config["FREEZE_LIMIT"]
        seq_len = self.config["SEQ_LEN"]
        
        # Step 1: Load model
        with st.container():
            st.markdown("### 🤖 STEP 1: Loading Lip Reading Model")
            with st.spinner("Loading model (GPU only)..."):
                if not self.load_model_wrapper():
                    st.error("❌ Failed to load lip sync model")
                    return None
            
            st.success(f"✅ Model loaded successfully on {self.device}")
        
        # Step 2: Initialize video analysis
        with st.container():
            st.markdown("### 🎬 STEP 2: Video Analysis")
            
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"❌ Failed to open video: {video_path}")
                return None
            
            buffer = deque(maxlen=seq_len)
            prev_text = ""
            freeze_count = 0
            
            # Get video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Display video info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Frames", total_frames)
            with col2:
                st.metric("FPS", f"{fps:.2f}")
            with col3:
                st.metric("Duration", f"{duration:.2f}s")
            
            # Create containers for dynamic updates
            progress_placeholder = st.empty()
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()
            results_container = st.container()
            
            # Results tracking
            results = {
                'frame_count': 0,
                'fake_count': 0,
                'real_count': 0,
                'total_decisions': 0,
                'all_predictions': [],
                'all_texts': [],
                'all_cer': [],
                'all_confidence': []
            }
            
            # Start analysis
            with results_container:
                st.markdown("### 📊 Real-time Analysis")
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Analysis loop
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_idx += 1
                    results['frame_count'] = frame_idx
                    
                    # Update progress
                    progress = min(frame_idx / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 Processing frame {frame_idx}/{total_frames}")
                    
                    # Extract mouth region
                    mouth = extract_mouth_frame(frame, self.config["IMG_SIZE"])
                    if mouth is None:
                        continue
                    
                    mouth = mouth.astype("float32") / 255.0
                    buffer.append(mouth)
                    
                    # Only process when buffer is full
                    if len(buffer) == seq_len:
                        seq_np = np.stack(buffer)
                        seq = torch.from_numpy(seq_np) \
                                   .unsqueeze(0) \
                                   .unsqueeze(2) \
                                   .float() \
                                   .to(self.device)
                        
                        with torch.no_grad():
                            logits = self.model(seq)
                            probs = logits.softmax(dim=-1)[0]
                            curr_text = greedy_decode(probs, self.idx_to_char)
                        
                        # Calculate metrics
                        drift = cer(curr_text, prev_text)
                        confidence = probs.max(dim=1)[0].mean().item()
                        lang_score = language_quality(curr_text)
                        
                        # Freeze detection
                        if curr_text == prev_text and len(curr_text) > 5:
                            freeze_count += 1
                        else:
                            freeze_count = 0
                        
                        # Make prediction
                        prediction = None
                        if frame_idx < seq_len * 3:
                            status = "WARMING UP"
                            prediction = "WARMUP"
                        else:
                            if (drift > cer_threshold or 
                                confidence < conf_threshold or 
                                lang_score < 2.5 or 
                                freeze_count > freeze_limit):
                                status = "❌ FAKE FRAME"
                                prediction = "FAKE"
                                results['fake_count'] += 1
                            else:
                                status = "✅ REAL FRAME"
                                prediction = "REAL"
                                results['real_count'] += 1
                            
                            results['total_decisions'] += 1
                        
                        # Store results
                        if prediction:
                            results['all_predictions'].append(prediction)
                            results['all_texts'].append(curr_text)
                            results['all_cer'].append(drift)
                            results['all_confidence'].append(confidence)
                        
                        prev_text = curr_text
                        
                        # Update display
                        self._update_display(
                            frame, frame_idx, curr_text, drift, confidence, 
                            lang_score, status, frame_placeholder, metrics_placeholder
                        )
            
            # Clean up
            cap.release()
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            
            # Display final results
            self._display_final_results(results)
        
        # Save results
        report_path = self.save_results(results, video_path)
        
        return results
    
    def _update_display(self, frame, frame_idx, curr_text, drift, confidence, lang_score, status, frame_placeholder, metrics_placeholder):
        """Update the display with current frame and metrics."""
        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add overlay text
        overlay = frame_rgb.copy()
        y_offset = 40
        cv2.putText(overlay, f"Frame: {frame_idx}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        cv2.putText(overlay, f"Status: {status}", 
                   (20, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0) if "REAL" in status else (255, 0, 0) if "FAKE" in status else (255, 165, 0), 2)
        
        if curr_text:
            cv2.putText(overlay, f"Text: {curr_text[:20]}...", 
                       (20, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
        
        cv2.putText(overlay, f"CER Drift: {drift:.3f}", 
                   (20, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
        
        cv2.putText(overlay, f"Confidence: {confidence:.3f}", 
                   (20, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        # Display frame
        frame_placeholder.image(overlay, channels="RGB", caption=f"Frame {frame_idx}")
        
        # Display metrics
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Status", status)
            with col2:
                st.metric("CER Drift", f"{drift:.3f}")
            with col3:
                st.metric("Confidence", f"{confidence:.3f}")
            with col4:
                st.metric("Language Score", f"{lang_score:.2f}")
    
    def _display_final_results(self, results):
        """Display final analysis results."""
        st.success("✅ Analysis Complete!")
        
        # Display results summary
        st.markdown("## 📊 Results Summary")
        
        if results['total_decisions'] > 0:
            # Calculate metrics
            total_decisions = results['total_decisions']
            real_count = results['real_count']
            fake_count = results['fake_count']
            fake_ratio = fake_count / total_decisions if total_decisions > 0 else 0
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Frames", results['frame_count'])
            with col2:
                st.metric("Total Decisions", total_decisions)
            with col3:
                st.metric("REAL Count", real_count)
            with col4:
                st.metric("FAKE Count", fake_count)
            
            # Create prediction chart
            if results['all_predictions']:
                st.markdown("### 📈 Prediction Timeline")
                predictions_df = pd.DataFrame({
                    'Sequence': range(len(results['all_predictions'])),
                    'Prediction': [1 if p == "REAL" else 0 if p == "FAKE" else 0.5 for p in results['all_predictions']]
                })
                
                fig = px.line(
                    predictions_df, 
                    x='Sequence', 
                    y='Prediction',
                    title='Real/Fake Predictions Over Time',
                    labels={'Prediction': 'Prediction (1=Real, 0=Fake, 0.5=Warmup)'}
                )
                fig.update_traces(line=dict(color='blue', width=2))
                fig.update_yaxes(range=[-0.1, 1.1])
                st.plotly_chart(fig, use_container_width=True)
            
            # Final prediction based on 50% threshold
            st.markdown("### 🎯 Final Assessment")
            
            if total_decisions == 0:
                st.warning("⚠️ **INCONCLUSIVE** - Not enough decisions made")
                final_verdict = "INCONCLUSIVE"
            elif fake_ratio > 0.50:
                st.error(f"❌ **FAKE VIDEO** - {fake_ratio*100:.1f}% of frames show lip sync issues (over 50%)")
                final_verdict = "FAKE"
            elif fake_ratio >= 0.30:
                st.warning(f"⚠️ **SUSPICIOUS** - {fake_ratio*100:.1f}% of frames show potential lip sync issues")
                final_verdict = "SUSPICIOUS"
            else:
                st.success(f"✅ **REAL VIDEO** - Only {fake_ratio*100:.1f}% of frames show lip sync issues")
                final_verdict = "REAL"
            
            # Display verdict
            st.markdown(f"""
            <div style="background: {'#FF6B6B' if final_verdict == 'FAKE' else '#FFD166' if final_verdict == 'SUSPICIOUS' else '#06D6A0' if final_verdict == 'REAL' else '#6C757D'}; 
                        padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
                <h2 style="color: white; margin: 0;">Final Verdict: {final_verdict}</h2>
                <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                    Fake Frames: {fake_count}/{total_decisions} ({fake_ratio*100:.1f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Return the verdict for use in combined analysis
            results['final_verdict'] = final_verdict
            results['fake_ratio'] = fake_ratio
            
        else:
            st.warning("⚠️ No decisions were made during analysis.")
            results['final_verdict'] = "INCONCLUSIVE"
            results['fake_ratio'] = 0
    
    def save_results(self, results, video_path):
        """Save lip sync analysis results."""
        fake_ratio = (results['fake_count'] / results['total_decisions'] * 100) if results['total_decisions'] > 0 else 0
        
        # Determine final verdict based on 50% threshold
        if results['total_decisions'] == 0:
            final_verdict = "INCONCLUSIVE"
        elif fake_ratio > 50:
            final_verdict = "FAKE"
        elif fake_ratio >= 30:
            final_verdict = "SUSPICIOUS"
        else:
            final_verdict = "REAL"
        
        report = {
            'video_path': video_path,
            'analysis_timestamp': str(np.datetime64('now')),
            'analysis_mode': 'Lip Sync Analysis',
            'config': {
                'cer_threshold': self.config["CER_THRESHOLD"],
                'conf_threshold': self.config["CONF_THRESHOLD"],
                'freeze_limit': self.config["FREEZE_LIMIT"]
            },
            'results_summary': {
                'total_frames': results['frame_count'],
                'total_decisions': results['total_decisions'],
                'real_count': results['real_count'],
                'fake_count': results['fake_count'],
                'fake_percentage': fake_ratio,
                'final_verdict': final_verdict,
                'verdict_criteria': 'FAKE if > 50% fake frames',
                'avg_cer': float(np.mean(results['all_cer'])) if results['all_cer'] else 0,
                'avg_confidence': float(np.mean(results['all_confidence'])) if results['all_confidence'] else 0
            },
            'detailed_results': results
        }
        
        # Save report
        report_path = os.path.join(
            self.results_dir, 
            f"lip_sync_{os.path.basename(video_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            st.success(f"✅ Lip sync report saved: {report_path}")
            
            # Create download button
            with open(report_path, 'r') as f:
                report_data = f.read()
            
            st.download_button(
                label="📥 Download Lip Sync Report",
                data=report_data,
                file_name=os.path.basename(report_path),
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Error saving report: {e}")
        
        return report_path

# =============================================================================
# METADATA ANALYSIS PIPELINE
# =============================================================================

def run_metadata_analysis(video_path, baseline_name=None):
    """Run metadata analysis and display results."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">🔬 Metadata Analysis</h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Complete Forensic Metadata Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Performing Forensic Metadata Analysis...")
    
    # Initialize analyzers
    metadata_analyzer = MetadataAnalyzer()
    forensic_analyzer = CompleteForensicMetadataAnalyzer()
    
    # Run basic metadata analysis
    with st.status("📋 **Analyzing basic metadata...**", expanded=True) as status:
        metadata_report = metadata_analyzer.create_metadata_report(video_path, baseline_name)
        if metadata_report:
            status.update(label="✅ **Basic Metadata Analysis Complete**", state="complete")
        else:
            st.warning("⚠️ Basic metadata analysis incomplete")
    
    # Run forensic metadata analysis
    with st.status("🔬 **Performing forensic metadata analysis...**", expanded=True) as status:
        forensic_report = forensic_analyzer.analyze_video(video_path)
        if forensic_report:
            status.update(label="✅ **Forensic Metadata Analysis Complete**", state="complete")
        else:
            st.warning("⚠️ Forensic metadata analysis incomplete")
    
    # Display forensic results
    if forensic_report:
        st.markdown("### 📊 Forensic Metadata Results")
        forensic_analyzer.display_detailed_forensic_report(forensic_report, video_path)
    
    # Create comprehensive report
    report = {
        "video_path": video_path,
        "analysis_timestamp": str(np.datetime64('now')),
        "analysis_mode": "Metadata Analysis",
        "metadata_report": metadata_report,
        "forensic_report": forensic_report
    }
    
    # Save report
    report_path = f"metadata_{os.path.basename(video_path)}.json"
    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        st.success(f"✅ Metadata report saved: {report_path}")
    except Exception as e:
        st.error(f"❌ Error saving report: {e}")
    
    return report

# =============================================================================
# COMPLETE ANALYSIS PIPELINE
# =============================================================================

class CompleteVideoAnalyzer:
    """Complete video analysis with all three methods."""
    
    def __init__(self):
        self.deepfake_pipeline = DeepfakeAnalysisPipeline()
        self.lip_sync_analyzer = LipSyncAnalyzer()
        self.forensic_analyzer = CompleteForensicMetadataAnalyzer()
        self.metadata_analyzer = MetadataAnalyzer()
        self.results_dir = "complete_analysis_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def analyze_video(self, video_path, frames_per_second=1, baseline_name=None):
        """Complete video analysis with all three methods."""
        # Main header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
            <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">🔬 Complete Video Analysis</h1>
            <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Deepfake Detection + Lip Sync + Metadata Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Store all results
        all_results = {}
        
        # 1. Deepfake Analysis
        st.markdown("## 🤖 PART 1: Deepfake Detection")
        deepfake_report = self.deepfake_pipeline.analyze_video(video_path, frames_per_second)
        all_results["deepfake"] = deepfake_report
        
        st.markdown("---")
        
        # 2. Lip Sync Analysis
        st.markdown("## 👄 PART 2: Lip Sync Analysis")
        lip_sync_report = self.lip_sync_analyzer.analyze_video(video_path)
        all_results["lip_sync"] = lip_sync_report
        
        st.markdown("---")
        
        # 3. Metadata Analysis
        st.markdown("## 🔍 PART 3: Metadata Analysis")
        metadata_report = run_metadata_analysis(video_path, baseline_name)
        all_results["metadata"] = metadata_report
        
        # 4. Combined Assessment
        self.display_combined_assessment(all_results, video_path)
        
        return all_results
    
    def display_combined_assessment(self, all_results, video_path):
        """Display combined assessment from all three methods."""
        st.markdown("## 🚨 COMBINED RISK ASSESSMENT")
        st.markdown("---")
        
        # Extract verdicts from each analysis
        deepfake_verdict = all_results.get("deepfake", {}).get("results_summary", {}).get("final_verdict", "UNKNOWN")
        deepfake_fake_pct = all_results.get("deepfake", {}).get("results_summary", {}).get("fake_percentage", 0)
        
        lip_sync_verdict = all_results.get("lip_sync", {}).get("results_summary", {}).get("final_verdict", "UNKNOWN")
        lip_sync_fake_pct = all_results.get("lip_sync", {}).get("results_summary", {}).get("fake_percentage", 0)
        
        # Get forensic risk from metadata
        forensic_risk = 0
        if all_results.get("metadata", {}).get("forensic_report", {}).get("forensic_summary"):
            forensic_risk = all_results["metadata"]["forensic_report"]["forensic_summary"].get("risk_score", 0)
            forensic_verdict = all_results["metadata"]["forensic_report"]["forensic_summary"].get("verdict", "UNKNOWN")
        
        # Display individual verdicts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if deepfake_verdict == "LIKELY FAKE":
                st.error(f"**Deepfake:** {deepfake_verdict}")
                st.metric("Fake %", f"{deepfake_fake_pct:.1f}%")
            else:
                st.success(f"**Deepfake:** {deepfake_verdict}")
                st.metric("Fake %", f"{deepfake_fake_pct:.1f}%")
        
        with col2:
            if lip_sync_verdict == "FAKE":
                st.error(f"**Lip Sync:** {lip_sync_verdict}")
                st.metric("Fake %", f"{lip_sync_fake_pct:.1f}%")
            elif lip_sync_verdict == "SUSPICIOUS":
                st.warning(f"**Lip Sync:** {lip_sync_verdict}")
                st.metric("Fake %", f"{lip_sync_fake_pct:.1f}%")
            else:
                st.success(f"**Lip Sync:** {lip_sync_verdict}")
                st.metric("Fake %", f"{lip_sync_fake_pct:.1f}%")
        
        with col3:
            if forensic_risk >= 3:
                st.error(f"**Metadata:** HIGH risk")
                st.metric("Risk Score", forensic_risk)
            elif forensic_risk >= 1:
                st.warning(f"**Metadata:** MEDIUM risk")
                st.metric("Risk Score", forensic_risk)
            else:
                st.success(f"**Metadata:** LOW risk")
                st.metric("Risk Score", forensic_risk)
        
        # Calculate overall risk
        overall_risk = self.calculate_overall_risk(
            deepfake_fake_pct, 
            lip_sync_fake_pct, 
            forensic_risk
        )
        
        # Display final overall verdict
        st.markdown("### 🎯 FINAL OVERALL VERDICT")
        
        if overall_risk >= 0.7:
            final_verdict = "HIGH RISK - LIKELY DEEPFAKE"
            verdict_color = "#FF6B6B"
            icon = "❌"
        elif overall_risk >= 0.4:
            final_verdict = "MODERATE RISK - SUSPICIOUS"
            verdict_color = "#FFD166"
            icon = "⚠️"
        else:
            final_verdict = "LOW RISK - LIKELY AUTHENTIC"
            verdict_color = "#06D6A0"
            icon = "✅"
        
        st.markdown(f"""
        <div style="background: {verdict_color}; padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
            <h2 style="color: white; margin: 0;">{icon} {final_verdict}</h2>
            <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                Overall Risk Score: {overall_risk:.2f}/1.00
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Save complete report
        report = {
            "video_path": video_path,
            "analysis_timestamp": str(np.datetime64('now')),
            "analysis_mode": "Complete Analysis",
            "deepfake_analysis": all_results.get("deepfake", {}),
            "lip_sync_analysis": all_results.get("lip_sync", {}),
            "metadata_analysis": all_results.get("metadata", {}),
            "combined_assessment": {
                "overall_risk_score": overall_risk,
                "final_verdict": final_verdict,
                "individual_verdicts": {
                    "deepfake": deepfake_verdict,
                    "lip_sync": lip_sync_verdict,
                    "metadata_risk": forensic_risk
                }
            }
        }
        
        report_path = os.path.join(
            self.results_dir, 
            f"complete_{os.path.basename(video_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            st.success(f"✅ Complete report saved: {report_path}")
        except Exception as e:
            st.error(f"❌ Error saving report: {e}")
    
    def calculate_overall_risk(self, deepfake_fake_pct, lip_sync_fake_pct, forensic_risk):
        """Calculate overall risk score from all three analyses."""
        # Normalize forensic risk (0-3 scale to 0-1 scale)
        forensic_normalized = min(forensic_risk / 3.0, 1.0)
        
        # Normalize deepfake fake percentage (0-100 to 0-1)
        deepfake_normalized = deepfake_fake_pct / 100.0
        
        # Normalize lip sync fake percentage (0-100 to 0-1)
        lip_sync_normalized = lip_sync_fake_pct / 100.0
        
        # Weighted combination (adjust weights as needed)
        overall_risk = (
            deepfake_normalized * 0.40 +      # Deepfake detection is most important
            lip_sync_normalized * 0.35 +      # Lip sync is also important
            forensic_normalized * 0.25        # Metadata provides supporting evidence
        )
        
        return min(overall_risk, 1.0)  # Cap at 1.0

# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def main():
    """Main Streamlit application."""
    # Configure page
    st.set_page_config(
        page_title="Video Authenticity Analyzer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Configure Plotly theme
    configure_streamlit_plotly_theme()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .mode-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .ensemble-info {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Video Authenticity Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">Weighted Ensemble Deepfake Detection + Lip Sync + Metadata Analysis</p>', unsafe_allow_html=True)
    
    # Navigation
    analysis_mode = st.sidebar.selectbox(
        "🎯 Select Analysis Mode",
        [
            "Deepfake Detection", 
            "Lip Sync Analysis", 
            "Metadata Analysis",
            "Complete Analysis"
        ]
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div class="ensemble-info">
            <h4>🤖 Weighted Ensemble</h4>
            <p style="margin: 0; font-size: 0.9rem;">
                <strong>Celeb 2.5:</strong> 50%<br>
                <strong>Xception L2:</strong> 30%<br>
                <strong>New DF E10:</strong> 20%<br>
                <strong>Safety:</strong> Xception Deployment
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Analysis Parameters")
        
        if analysis_mode == "Deepfake Detection":
            frames_per_second = st.slider(
                "Frames per second:",
                min_value=1,
                max_value=10,
                value=2,
                help="Higher values provide more detailed analysis"
            )
            
            confidence_threshold = st.slider(
                "Face detection confidence:",
                min_value=0.1,
                max_value=1.0,
                value=0.9,
                step=0.05,
                help="Minimum confidence score for face detection"
            )
        
        elif analysis_mode == "Lip Sync Analysis":
            cer_threshold = st.slider(
                "CER Threshold",
                min_value=0.1,
                max_value=0.8,
                value=0.35,
                step=0.05,
                help="Character Error Rate threshold for detecting anomalies"
            )
            
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.45,
                step=0.05,
                help="Minimum confidence required for REAL classification"
            )
            
            freeze_limit = st.slider(
                "Freeze Limit",
                min_value=5,
                max_value=50,
                value=10,
                step=1,
                help="Number of frames with same text before flagging as FAKE"
            )
        
        elif analysis_mode in ["Metadata Analysis", "Complete Analysis"]:
            # Baseline selection for metadata modes
            metadata_analyzer = MetadataAnalyzer()
            available_baselines = metadata_analyzer.get_available_baselines()
            baseline_name = None
            if available_baselines:
                baseline_name = st.selectbox(
                    "Select Baseline (Optional)",
                    [""] + available_baselines,
                    help="Compare against a pre-built metadata baseline"
                )
            
            if analysis_mode == "Complete Analysis":
                frames_per_second = st.slider(
                    "Frames per second:",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Higher values provide more detailed analysis"
                )
        
        # System info
        st.markdown("### 📊 System Information")
        st.info(f"**Device:** {'🚀 GPU' if torch.cuda.is_available() else '💻 CPU'}")
        
        # Check ffprobe availability
        try:
            subprocess.check_output(["ffprobe", "-version"])
            st.success("✅ FFprobe available")
        except:
            st.warning("⚠️ FFprobe not found - metadata analysis limited")
    
    # Main content area
    st.markdown("---")
    st.markdown("### 📁 Upload Video for Analysis")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your video file here:",
        type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm', 'm4v'],
        help="Supported formats: MP4, AVI, MOV, MKV, FLV, WMV, WEBM, M4V"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
        
        # Display video preview
        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(uploaded_file)
        with col2:
            mode_info = {
                "Deepfake Detection": "Weighted Ensemble Analysis",
                "Lip Sync Analysis": "Audio-Visual Synchronization",
                "Metadata Analysis": "Forensic Metadata Analysis",
                "Complete Analysis": "All Three Methods"
            }
            st.markdown(f"""
            <div class="mode-card">
                <h4 style="margin: 0 0 1rem 0;">📹 Video Details</h4>
                <p style="margin: 0.5rem 0;"><strong>File:</strong> {uploaded_file.name[:30]}{'...' if len(uploaded_file.name) > 30 else ''}</p>
                <p style="margin: 0.5rem 0;"><strong>Size:</strong> {uploaded_file.size / (1024*1024):.2f} MB</p>
                <p style="margin: 0.5rem 0;"><strong>Mode:</strong> {analysis_mode}</p>
                <p style="margin: 0.5rem 0;"><strong>Analysis:</strong> {mode_info.get(analysis_mode, 'Custom')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis button
        if st.button(f"🚀 Start {analysis_mode}", type="primary", use_container_width=True):
            try:
                if analysis_mode == "Deepfake Detection":
                    pipeline = DeepfakeAnalysisPipeline()
                    report = pipeline.analyze_video(video_path, frames_per_second)
                
                elif analysis_mode == "Lip Sync Analysis":
                    pipeline = LipSyncAnalyzer()
                    report = pipeline.analyze_video(video_path, cer_threshold, conf_threshold, freeze_limit)
                
                elif analysis_mode == "Metadata Analysis":
                    report = run_metadata_analysis(video_path, baseline_name)
                
                else:  # Complete Analysis
                    pipeline = CompleteVideoAnalyzer()
                    report = pipeline.analyze_video(video_path, frames_per_second, baseline_name)
                
                # Download report option
                if report:
                    st.markdown("---")
                    st.markdown("### 💾 Download Results")
                    
                    try:
                        json_report = json.dumps(report, indent=2, cls=NumpyEncoder)
                        st.download_button(
                            label="📥 Download Complete Report (JSON)",
                            data=json_report,
                            file_name=f"{analysis_mode.lower().replace(' ', '_')}_{uploaded_file.name}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"❌ Error creating download file: {e}")
            
            except Exception as e:
                st.error(f"❌ Analysis Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(video_path)
                except:
                    pass
    
    else:
        # Welcome message
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 3rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
            <h2 style="color: #2c3e50; margin: 0 0 1rem 0;">🔬 Ready for Authenticity Analysis</h2>
            <p style="color: #2c3e50; margin: 0; font-size: 1.1rem;">
                Upload a video file to start comprehensive authenticity analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("### ✨ Analysis Modes Available")
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="mode-card">
                <h4>🤖 Deepfake Detection</h4>
                <p>Weighted ensemble with safety layer</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="mode-card">
                <h4>👄 Lip Sync Analysis</h4>
                <p>Audio-visual synchronization verification</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="mode-card">
                <h4>🔍 Metadata Analysis</h4>
                <p>Complete forensic metadata analysis</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="mode-card">
                <h4>🔬 Complete Analysis</h4>
                <p>All three methods combined</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Ensemble details
        st.markdown("### 🎯 Weighted Ensemble Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info("**Primary Model**\n\nCeleb + Image (Level 2.5)\n\n**Weight:** 50%")
        with col2:
            st.info("**Support Model**\n\nXception Progressive (L2)\n\n**Weight:** 30%")
        with col3:
            st.info("**Experimental**\n\nNew Deepfake (Epoch 10)\n\n**Weight:** 20%")
        with col4:
            st.info("**Safety Layer**\n\nXception Deployment\n\n**Confirms REAL verdicts**")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p><strong>Video Authenticity Analyzer</strong> | Weighted Ensemble Deepfake Detection + Lip Sync + Metadata Analysis</p>
        <p style="font-size: 0.9rem;">Decision Rule: Weighted score ≥ 0.5 → FAKE | Safety: REAL requires Xception Deployment confirmation</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()