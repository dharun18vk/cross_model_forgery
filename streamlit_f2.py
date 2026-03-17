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
# MODEL PATHS CONFIGURATION
# =============================================================================
MODEL_PATHS = {
    # ==========================================================
    # BASE / BENCHMARK MODELS
    # ==========================================================
    "Base FF++ Stage-3 Model":
        "DEEPFAKE_MODELS/best_stage3_ffpp_frames.pth",

    "Base Xception Face Model":
        "DEEPFAKE_MODELS/xception_face_model.pth",

    # ==========================================================
    # CELEB-DF & IMAGE MIXED TRAINING (PROGRESSIVE LEVELS)
    # ==========================================================
    "Celeb + Image Model (Level 0 - Initial)":
        "DEEPFAKE_MODELS/0-df.pth",

    "Celeb + Image Model (Level 0.5)":
        "DEEPFAKE_MODELS/0.5-df.pth",

    "Celeb + Image Model (Level 1)":
        "DEEPFAKE_MODELS/1-df.pth",

    "Celeb + Image Model (Level 1.5 - Multi Fine-tuned)":
        "DEEPFAKE_MODELS/1.5-df.pth",

    "Celeb + Image Model (Level 2)":
        "DEEPFAKE_MODELS/2-df.pth",

    "=== Celeb + Image Model (Level 2.5 - Advanced)":
        "DEEPFAKE_MODELS/2.5-df.pth",

    # ==========================================================
    # CELEB-DF SPECIALIZED MODELS
    # ==========================================================
    "Celeb-DF Fine-tuned (Best Model)":
        "DEEPFAKE_MODELS/best_celebf_finetuned.pth",

    "Celeb-DF Training Checkpoint (Latest)":
        "DEEPFAKE_MODELS/latest_celebf_checkpoint.pth",

    # ==========================================================
    # XCEPTION – FINE-TUNED & PROGRESSIVE TRAINING
    # ==========================================================
    "Xception Fine-tuned (Level 1)":
        "DEEPFAKE_MODELS/fine_tuned_xception_model/best_fine_tuned_model.pth",

    "=== Xception Progressive Fine-tuned (Level 2)":
        "DEEPFAKE_MODELS/progressive_fine_tuned_model/2nd_tuned_xception_model.pth",

    "Xception Progressive Fine-tuned (Final)":
        "DEEPFAKE_MODELS/progressive_fine_tuned_model/final_progressive_model.pth",

    # ==========================================================
    # NEW / EXPERIMENTAL MODELS
    # ==========================================================
    "=== New Deepfake Model (Experimental – Epoch 10)":
        "DEEPFAKE_MODELS/new_deepfake_model/checkpoint_epoch_10.pth",

    "New Deepfake Model (Working – Unverified)":
        "DEEPFAKE_MODELS/new_deepfake_model/unknown_working_model.pth",

    # ==========================================================
    # PRODUCTION / DEPLOYMENT READY
    # ==========================================================
    "Best EfficientNet-B4 Model (Production)":
        "DEEPFAKE_MODELS/best_B4_model.pth",

    "Xception Model (Deployment Ready)":
        "DEEPFAKE_MODELS/xception_deepfake_model/best_face_model.pth",

    # ==========================================================
    # TRAINING RESUME CHECKPOINTS
    # ==========================================================
    "FF++ Stage-3 Training Checkpoint (Latest)":
        "DEEPFAKE_MODELS/latest_stage3_checkpoint.pth",
}
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
# METADATA ANALYSIS UTILITIES (Existing)
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
        with st.spinner("🎬 **Extracting frames from video...**"):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"❌ Error opening video file: {video_path}")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            frame_interval = max(1, int(fps / frames_per_second))
            extracted_frames = []
            frame_count = 0
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_frames_dir = os.path.join(self.output_dir, video_name)
            os.makedirs(video_frames_dir, exist_ok=True)
            
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
            
            cap.release()
            
        return extracted_frames, {
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'extracted_count': len(extracted_frames)
        }


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
        all_faces = []
        face_count = 0
        
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
        
        return all_faces


# =============================================================================
# DEEPFAKE DETECTOR
# =============================================================================

class DeepFakeDetector:
    """Deepfake detection using trained models."""
    def __init__(self, model_path, model_name=""):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.model_name = model_name
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
    
    def load_model(self, model_path):
        """Load the deepfake detection model."""
        if not os.path.exists(model_path):
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
            return None
        
        model.to(self.device)
        model.eval()
        return model
    
    def get_transform(self):
        """Get image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_single_face(self, face_path):
        """Predict if a single face is real or fake."""
        if self.model is None:
            return None
            
        try:
            image = Image.open(face_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probability = output.item()
                prediction = "FAKE" if probability > 0.5 else "REAL"
                confidence = probability if prediction == "FAKE" else 1 - probability
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'fake_probability': float(probability),
                'real_probability': float(1 - probability),
                'model_name': self.model_name
            }
        
        except Exception as e:
            return None
    
    def predict_multiple_faces(self, face_data_list):
        """Predict multiple faces."""
        if self.model is None:
            return []
            
        results = []
        
        for face_data in face_data_list:
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
        
        return results


# =============================================================================
# METADATA ANALYZER (Existing)
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
# DEEPFAKE DETECTION PIPELINE
# =============================================================================

class DeepfakeDetectionPipeline:
    """Pipeline for deepfake detection only."""
    def __init__(self):
        self.frame_extractor = FrameExtractor()
        self.face_detector = RetinaFaceDetector()
        self.results_dir = "deepfake_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def analyze_video(self, video_path, selected_models, frames_per_second=1):
        """Analyze video for deepfake detection only."""
        all_results = {}
        
        # Step 1: Frame Extraction
        with st.spinner("🎬 Extracting frames from video..."):
            frames, frame_stats = self.frame_extractor.extract_frames(video_path, frames_per_second)
            if not frames:
                st.error("❌ No frames extracted. Exiting.")
                return None
        
        # Step 2: Face Detection
        with st.spinner("👤 Detecting faces..."):
            faces = self.face_detector.detect_and_extract_faces(frames)
            if not faces:
                st.error("❌ No faces detected. Exiting.")
                return None
        
        # Step 3: Initialize Models
        detectors = {}
        for model_name, model_path in selected_models.items():
            detector = DeepFakeDetector(model_path, model_name)
            if detector.model is not None:
                detectors[model_name] = detector
        
        if not detectors:
            st.error("❌ No models loaded successfully.")
            return None
        
        # Step 4: Deepfake Analysis
        with st.spinner("🤖 Analyzing faces for deepfake detection..."):
            all_predictions = {}
            
            for model_name, detector in detectors.items():
                predictions = detector.predict_multiple_faces(faces)
                all_predictions[model_name] = predictions
            
            # Generate deepfake report
            deepfake_report = self.generate_deepfake_report(all_predictions, video_path, frame_stats, len(faces))
            all_results["deepfake_analysis"] = deepfake_report
        
        return all_results
    
    def generate_deepfake_report(self, all_predictions, video_path, frame_stats, total_faces):
        """Generate deepfake analysis report."""
        model_reports = {}
        
        for model_name, predictions in all_predictions.items():
            if not predictions:
                continue
                
            total_predictions = len(predictions)
            fake_faces = sum(1 for pred in predictions if pred['prediction'] == 'FAKE')
            real_faces = total_predictions - fake_faces
            
            fake_confidence_avg = float(np.mean([pred['confidence'] for pred in predictions if pred['prediction'] == 'FAKE'])) if fake_faces > 0 else 0.0
            real_confidence_avg = float(np.mean([pred['confidence'] for pred in predictions if pred['prediction'] == 'REAL'])) if real_faces > 0 else 0.0
            
            model_reports[model_name] = {
                'total_faces_analyzed': int(total_predictions),
                'fake_faces_detected': int(fake_faces),
                'real_faces_detected': int(real_faces),
                'fake_percentage': float((fake_faces / total_predictions * 100) if total_predictions > 0 else 0),
                'average_fake_confidence': float(fake_confidence_avg),
                'average_real_confidence': float(real_confidence_avg),
                'overall_verdict': "LIKELY FAKE" if fake_faces > real_faces else "LIKELY REAL",
                'confidence_score': float(max(fake_confidence_avg, real_confidence_avg))
            }
        
        # Calculate ensemble results
        if model_reports:
            fake_percentages = [report['fake_percentage'] for report in model_reports.values()]
            ensemble_fake_percentage = np.mean(fake_percentages) if fake_percentages else 0
            
            model_reports['ENSEMBLE'] = {
                'fake_percentage': float(ensemble_fake_percentage),
                'overall_verdict': "LIKELY FAKE" if ensemble_fake_percentage > 50 else "LIKELY REAL",
                'confidence_score': float(np.mean([r['confidence_score'] for r in model_reports.values()]))
            }
        
        report = {
            'video_path': video_path,
            'analysis_timestamp': str(np.datetime64('now')),
            'models_used': list(all_predictions.keys()),
            'video_stats': frame_stats,
            'total_faces_detected': total_faces,
            'model_reports': model_reports,
            'ensemble_report': model_reports.get('ENSEMBLE', {})
        }
        
        # Save report
        report_path = os.path.join(self.results_dir, f"deepfake_{os.path.basename(video_path)}.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            pass
        
        return report
    
    def display_deepfake_report(self, deepfake_report):
        """Display deepfake detection results in a user-friendly format."""
        st.markdown("## 🤖 Deepfake Detection Report")
        st.markdown("---")
        
        if deepfake_report:
            ensemble_report = deepfake_report.get('ensemble_report', {})
            video_stats = deepfake_report.get('video_stats', {})
            models_used = deepfake_report.get('models_used', [])
            
            # Video Info
            st.markdown("### 📹 Video Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{video_stats.get('duration', 0):.1f}s")
            with col2:
                st.metric("FPS", f"{video_stats.get('fps', 0):.1f}")
            with col3:
                st.metric("Total Frames", video_stats.get('total_frames', 0))
            with col4:
                st.metric("Extracted Frames", video_stats.get('extracted_count', 0))
            
            # Analysis Summary
            st.markdown("### 📊 Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Models Used", len(models_used))
            with col2:
                st.metric("Faces Detected", deepfake_report.get('total_faces_detected', 0))
            with col3:
                fake_percentage = ensemble_report.get('fake_percentage', 0)
                st.metric("Fake %", f"{fake_percentage:.1f}%")
            with col4:
                verdict = ensemble_report.get('overall_verdict', 'UNKNOWN')
                verdict_color = "#FF6B6B" if verdict == "LIKELY FAKE" else "#4ECDC4"
                st.markdown(f"""
                <div style="background: {verdict_color}; padding: 0.5rem; border-radius: 10px; text-align: center;">
                    <h4 style="color: white; margin: 0;">Verdict: {verdict}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed Results
            st.markdown("### 🔍 Detailed Results")
            
            if len(models_used) > 1:
                # Model comparison chart
                model_names = []
                fake_percentages = []
                
                for model_name, model_report in deepfake_report.get('model_reports', {}).items():
                    if model_name != 'ENSEMBLE':
                        model_names.append(model_name)
                        fake_percentages.append(model_report.get('fake_percentage', 0))
                
                if model_names:
                    comparison_data = pd.DataFrame({
                        'Model': model_names,
                        'Fake Percentage': fake_percentages
                    })
                    
                    fig_comparison = px.bar(
                        comparison_data,
                        x='Model',
                        y='Fake Percentage',
                        title='Deepfake Detection by Model',
                        color='Fake Percentage',
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig_comparison.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Expert Details (in expander)
            with st.expander("🔬 Expert Technical Details"):
                st.markdown("#### Model Performance")
                model_details = []
                for model_name, model_report in deepfake_report.get('model_reports', {}).items():
                    if model_name != 'ENSEMBLE':
                        model_details.append({
                            'Model': model_name,
                            'Faces Analyzed': model_report.get('total_faces_analyzed', 0),
                            'Fake Detected': model_report.get('fake_faces_detected', 0),
                            'Fake %': f"{model_report.get('fake_percentage', 0):.1f}%",
                            'Avg Fake Confidence': f"{model_report.get('average_fake_confidence', 0):.3f}",
                            'Verdict': model_report.get('overall_verdict', 'N/A')
                        })
                
                if model_details:
                    st.table(pd.DataFrame(model_details))
                
                st.markdown("#### Ensemble Results")
                st.json(ensemble_report)
            
            # Final Verdict
            st.markdown("### 🎯 Final Assessment")
            
            fake_percentage = ensemble_report.get('fake_percentage', 0)
            if fake_percentage > 50:
                st.error(f"**❌ FAKE VIDEO DETECTED** - {fake_percentage:.1f}% of analyzed faces show signs of manipulation")
                st.markdown("""
                **Recommendation:** This video is likely artificially generated or manipulated. 
                Do not trust its authenticity without additional verification.
                """)
            elif fake_percentage > 25:
                st.warning(f"**⚠️ SUSPICIOUS VIDEO** - {fake_percentage:.1f}% of analyzed faces show potential manipulation")
                st.markdown("""
                **Recommendation:** Exercise caution. Some artificial manipulation may be present. 
                Consider additional verification methods.
                """)
            else:
                st.success(f"**✅ AUTHENTIC VIDEO** - Only {fake_percentage:.1f}% of faces show potential manipulation")
                st.markdown("""
                **Recommendation:** This video appears to be authentic. 
                No significant signs of artificial manipulation detected.
                """)
            
            # Confidence Score
            confidence_score = ensemble_report.get('confidence_score', 0)
            st.info(f"**Confidence Score:** {confidence_score:.2f}/1.0")


# =============================================================================
# LIP SYNC ANALYSIS PIPELINE
# =============================================================================

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# =====================================================
# 🔧 CONFIG
# =====================================================
MODEL_PATH = "preprocessing/models/lip_model_best.pth"
IMG_SIZE = 64
SEQ_LEN = 25
CER_THRESHOLD = 0.35
CONF_THRESHOLD = 0.45
FREEZE_LIMIT = 10

# =====================================================
# 🧠 LOAD MODEL (CACHED)
# =====================================================
@st.cache_resource
def load_model():
    # Force GPU only - raise error if CUDA not available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA (GPU) is not available. This application requires GPU for lip sync analysis.")
    
    device = torch.device("cuda")
    
    char_to_idx, idx_to_char = build_vocab()
    model = LipReadingModel(len(char_to_idx) + 1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    return model, device, idx_to_char

# =====================================================
# 🔤 GREEDY CTC DECODER
# =====================================================
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

# =====================================================
# 📏 CER
# =====================================================
def cer(a, b):
    if len(b) == 0:
        return 0.0
    import editdistance
    return editdistance.eval(a, b) / len(b)

# =====================================================
# 🧠 LANGUAGE QUALITY
# =====================================================
def language_quality(text):
    words = text.split()
    if len(words) == 0:
        return 0.0
    avg_len = sum(len(w) for w in words) / len(words)
    return avg_len

# =====================================================
# 🎥 LIP SYNC ANALYSIS CLASS
# =====================================================

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
        # Use provided thresholds or defaults
        cer_threshold = cer_threshold or self.config["CER_THRESHOLD"]
        conf_threshold = conf_threshold or self.config["CONF_THRESHOLD"]
        freeze_limit = freeze_limit or self.config["FREEZE_LIMIT"]
        
        # Load model
        with st.spinner("👄 Loading lip sync analysis model..."):
            if not self.load_model_wrapper():
                st.error("❌ Failed to load lip sync model")
                return None
        
        # Initialize video analysis
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"❌ Failed to open video: {video_path}")
            return None
        
        buffer = deque(maxlen=SEQ_LEN)
        prev_text = ""
        freeze_count = 0
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Results tracking
        results = {
            'frame_count': 0,
            'fake_count': 0,
            'real_count': 0,
            'total_decisions': 0,
            'all_predictions': [],
            'all_texts': [],
            'all_cer': [],
            'all_confidence': [],
            'video_stats': {
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration
            }
        }
        
        # Analysis loop
        with st.spinner("👄 Analyzing lip sync..."):
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                results['frame_count'] = frame_idx
                
                # Extract mouth region
                mouth = extract_mouth_frame(frame, IMG_SIZE)
                if mouth is None:
                    continue
                
                mouth = mouth.astype("float32") / 255.0
                buffer.append(mouth)
                
                # Only process when buffer is full
                if len(buffer) == SEQ_LEN:
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
                    if frame_idx < SEQ_LEN * 3:
                        prediction = "WARMUP"
                    else:
                        if (drift > cer_threshold or 
                            confidence < conf_threshold or 
                            lang_score < 2.5 or 
                            freeze_count > freeze_limit):
                            prediction = "FAKE"
                            results['fake_count'] += 1
                        else:
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
        
        # Clean up
        cap.release()
        
        # Determine final verdict
        if results['total_decisions'] == 0:
            results['final_verdict'] = "INCONCLUSIVE"
            results['fake_ratio'] = 0
        else:
            fake_ratio = results['fake_count'] / results['total_decisions']
            if fake_ratio > 0.50:
                results['final_verdict'] = "FAKE"
            elif fake_ratio >= 0.30:
                results['final_verdict'] = "SUSPICIOUS"
            else:
                results['final_verdict'] = "REAL"
            results['fake_ratio'] = fake_ratio
        
        # Save results
        self.save_results(results, video_path)
        
        return results
    
    def save_results(self, results, video_path):
        """Save lip sync analysis results."""
        fake_ratio = results.get('fake_ratio', 0) * 100
        
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
                'final_verdict': results.get('final_verdict', 'INCONCLUSIVE'),
                'verdict_criteria': 'FAKE if > 50% fake frames',
                'avg_cer': float(np.mean(results['all_cer'])) if results['all_cer'] else 0,
                'avg_confidence': float(np.mean(results['all_confidence'])) if results['all_confidence'] else 0
            },
            'video_stats': results.get('video_stats', {})
        }
        
        # Save report
        report_path = os.path.join(
            self.results_dir, 
            f"lip_sync_{os.path.basename(video_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            pass
        
        return report_path
    
    def display_lip_sync_report(self, results):
        """Display lip sync analysis results in a user-friendly format."""
        st.markdown("## 👄 Lip Sync Analysis Report")
        st.markdown("---")
        
        video_stats = results.get('video_stats', {})
        
        # Video Info
        st.markdown("### 📹 Video Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{video_stats.get('duration', 0):.1f}s")
        with col2:
            st.metric("FPS", f"{video_stats.get('fps', 0):.1f}")
        with col3:
            st.metric("Total Frames", video_stats.get('total_frames', 0))
        
        # Analysis Summary
        st.markdown("### 📊 Analysis Summary")
        
        if results['total_decisions'] > 0:
            total_decisions = results['total_decisions']
            real_count = results['real_count']
            fake_count = results['fake_count']
            fake_ratio = results.get('fake_ratio', 0) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Frames Analyzed", results['frame_count'])
            with col2:
                st.metric("Decisions Made", total_decisions)
            with col3:
                st.metric("Sync Issues", fake_count)
            with col4:
                final_verdict = results.get('final_verdict', 'INCONCLUSIVE')
                verdict_color = "#FF6B6B" if final_verdict == "FAKE" else "#FFD166" if final_verdict == "SUSPICIOUS" else "#06D6A0" if final_verdict == "REAL" else "#6C757D"
                st.markdown(f"""
                <div style="background: {verdict_color}; padding: 0.5rem; border-radius: 10px; text-align: center;">
                    <h4 style="color: white; margin: 0;">Verdict: {final_verdict}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics
            st.markdown("### 📈 Analysis Metrics")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Sync Issue %", f"{fake_ratio:.1f}%")
            with metrics_col2:
                avg_cer = np.mean(results['all_cer']) if results['all_cer'] else 0
                st.metric("Avg Sync Drift", f"{avg_cer:.3f}")
            with metrics_col3:
                avg_conf = np.mean(results['all_confidence']) if results['all_confidence'] else 0
                st.metric("Avg Confidence", f"{avg_conf:.3f}")
            
            # Prediction timeline (if enough data)
            if len(results['all_predictions']) > 10:
                st.markdown("### 📊 Sync Quality Over Time")
                predictions_df = pd.DataFrame({
                    'Sequence': range(len(results['all_predictions'])),
                    'Sync Quality': [1 if p == "REAL" else 0 if p == "FAKE" else 0.5 for p in results['all_predictions']]
                })
                
                fig = px.line(
                    predictions_df, 
                    x='Sequence', 
                    y='Sync Quality',
                    title='Lip Sync Quality Over Time',
                    labels={'Sync Quality': 'Sync Quality (1=Good, 0=Poor)'}
                )
                fig.update_traces(line=dict(color='blue', width=2))
                fig.update_yaxes(range=[-0.1, 1.1])
                st.plotly_chart(fig, use_container_width=True)
            
            # Expert Details (in expander)
            with st.expander("🔬 Expert Technical Details"):
                st.markdown("#### Detailed Metrics")
                details = {
                    'Metric': ['Character Error Rate (CER)', 'Confidence Score', 'Language Quality', 'Freeze Detection', 'Total Decisions'],
                    'Value': [
                        f"{np.mean(results['all_cer']):.3f}" if results['all_cer'] else "N/A",
                        f"{np.mean(results['all_confidence']):.3f}" if results['all_confidence'] else "N/A",
                        f"{language_quality(' '.join(results['all_texts'])):.2f}" if results['all_texts'] else "N/A",
                        f"Detected at {self.config['FREEZE_LIMIT']} frames",
                        f"{total_decisions}"
                    ]
                }
                st.table(pd.DataFrame(details))
                
                st.markdown("#### Sample Detected Speech")
                unique_texts = list(dict.fromkeys([t for t in results['all_texts'] if t.strip()]))
                if unique_texts:
                    for i, text in enumerate(unique_texts[:3]):
                        st.text(f"{i+1}. \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            
            # Final Assessment
            st.markdown("### 🎯 Final Assessment")
            
            if final_verdict == "FAKE":
                st.error(f"**❌ POOR LIP SYNC DETECTED** - {fake_ratio:.1f}% of speech frames show sync issues")
                st.markdown("""
                **Recommendation:** This video likely has artificial or poorly synced audio. 
                The lip movements do not match the spoken words accurately.
                """)
            elif final_verdict == "SUSPICIOUS":
                st.warning(f"**⚠️ MODERATE SYNC ISSUES** - {fake_ratio:.1f}% of speech frames show sync issues")
                st.markdown("""
                **Recommendation:** Some sync issues detected. This could indicate minor manipulation 
                or natural speech variations. Further verification recommended.
                """)
            elif final_verdict == "REAL":
                st.success(f"**✅ GOOD LIP SYNC** - Only {fake_ratio:.1f}% of speech frames show minor sync issues")
                st.markdown("""
                **Recommendation:** Lip movements appear to match spoken words accurately. 
                This is typical of authentic video footage.
                """)
            else:
                st.info("**ℹ️ INCONCLUSIVE ANALYSIS** - Insufficient speech data for reliable analysis")
                st.markdown("""
                **Recommendation:** Video may not contain sufficient speech for lip sync analysis. 
                Consider using other verification methods.
                """)
            
            # Confidence Indicator
            sync_confidence = 1.0 - (fake_ratio / 100)
            st.info(f"**Sync Confidence:** {sync_confidence:.2f}/1.0")
            
        else:
            st.warning("**⚠️ NO SPEECH DETECTED**")
            st.markdown("""
            **Analysis Result:** The video does not contain sufficient speech for lip sync analysis.
            
            **Possible reasons:**
            - Video has no audio track
            - Audio contains no speech (music/silence only)
            - Speech is too short for analysis
            - Technical issues with audio extraction
            """)

# =============================================================================
# METADATA ANALYSIS PIPELINE
# =============================================================================

class MetadataAnalysisPipeline:
    """Pipeline for metadata analysis only."""
    def __init__(self):
        self.metadata_analyzer = MetadataAnalyzer()
        self.forensic_analyzer = CompleteForensicMetadataAnalyzer()
        self.results_dir = "metadata_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def analyze_video(self, video_path, baseline_name=None):
        """Perform metadata analysis only."""
        all_results = {}
        
        # Step 1: Basic Metadata Analysis
        with st.spinner("🔍 Analyzing video metadata..."):
            metadata_report = self.metadata_analyzer.create_metadata_report(video_path, baseline_name)
            if metadata_report:
                all_results["metadata_analysis"] = metadata_report
        
        # Step 2: Forensic Metadata Analysis
        with st.spinner("🔬 Performing forensic analysis..."):
            forensic_report = self.forensic_analyzer.analyze_video(video_path)
            if forensic_report:
                all_results["forensic_analysis"] = forensic_report
        
        return all_results
    
    def display_metadata_report(self, all_results, video_path):
        """Display metadata analysis results."""
        st.markdown("## 🔍 Metadata Analysis Report")
        st.markdown("---")
        
        forensic_report = all_results.get("forensic_analysis")
        if forensic_report:
            self.forensic_analyzer.display_detailed_forensic_report(forensic_report, video_path)


# =============================================================================
# COMPLETE ANALYSIS PIPELINE (ALL 3)
# =============================================================================

class CompleteAnalysisPipeline:
    """Complete analysis pipeline with all 3 components."""
    def __init__(self):
        self.deepfake_pipeline = DeepfakeDetectionPipeline()
        self.lip_sync_pipeline = LipSyncAnalyzer()
        self.metadata_pipeline = MetadataAnalysisPipeline()
        self.results_dir = "complete_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def analyze_video(self, video_path, selected_models, frames_per_second=1, 
                      baseline_name=None, cer_threshold=0.35, conf_threshold=0.45, freeze_limit=10):
        """Perform complete analysis with all 3 components."""
        all_results = {}
        
        # Create tabs for each analysis
        tab1, tab2, tab3 = st.tabs(["🤖 Deepfake Detection", "👄 Lip Sync Analysis", "🔍 Metadata Analysis"])
        
        with tab1:
            with st.spinner("🤖 Running deepfake detection..."):
                deepfake_results = self.deepfake_pipeline.analyze_video(video_path, selected_models, frames_per_second)
                if deepfake_results:
                    all_results["deepfake"] = deepfake_results
                    self.deepfake_pipeline.display_deepfake_report(deepfake_results["deepfake_analysis"])
        
        with tab2:
            with st.spinner("👄 Analyzing lip sync..."):
                lip_sync_results = self.lip_sync_pipeline.analyze_video(video_path, cer_threshold, conf_threshold, freeze_limit)
                if lip_sync_results:
                    all_results["lip_sync"] = lip_sync_results
                    self.lip_sync_pipeline.display_lip_sync_report(lip_sync_results)
        
        with tab3:
            with st.spinner("🔍 Analyzing metadata..."):
                metadata_results = self.metadata_pipeline.analyze_video(video_path, baseline_name)
                if metadata_results:
                    all_results["metadata"] = metadata_results
                    self.metadata_pipeline.display_metadata_report(metadata_results, video_path)
        
        # Display combined results
        if all_results:
            self.display_combined_report(all_results, video_path)
        
        return all_results
    
    def display_combined_report(self, all_results, video_path):
        """Display combined results from all 3 analyses."""
        st.markdown("## 🚨 Comprehensive Analysis Report")
        st.markdown("---")
        
        # Collect verdicts from each analysis
        verdicts = {}
        
        # Deepfake verdict
        if "deepfake" in all_results:
            deepfake_report = all_results["deepfake"].get("deepfake_analysis", {})
            ensemble_report = deepfake_report.get("ensemble_report", {})
            verdicts["deepfake"] = {
                "verdict": ensemble_report.get("overall_verdict", "UNKNOWN"),
                "confidence": ensemble_report.get("fake_percentage", 0),
                "color": "#FF6B6B" if ensemble_report.get("overall_verdict") == "LIKELY FAKE" else "#4ECDC4"
            }
        
        # Lip sync verdict
        if "lip_sync" in all_results:
            lip_sync_results = all_results["lip_sync"]
            final_verdict = lip_sync_results.get("final_verdict", "INCONCLUSIVE")
            if final_verdict == "FAKE":
                color = "#FF6B6B"
            elif final_verdict == "SUSPICIOUS":
                color = "#FFD166"
            elif final_verdict == "REAL":
                color = "#06D6A0"
            else:
                color = "#6C757D"
            
            verdicts["lip_sync"] = {
                "verdict": final_verdict,
                "confidence": lip_sync_results.get("fake_ratio", 0) * 100,
                "color": color
            }
        
        # Metadata verdict
        if "metadata" in all_results:
            forensic_report = all_results["metadata"].get("forensic_analysis", {})
            if forensic_report and "forensic_summary" in forensic_report:
                summary = forensic_report["forensic_summary"]
                verdicts["metadata"] = {
                    "verdict": summary.get("verdict", "UNKNOWN"),
                    "confidence": summary.get("risk_score", 0) * 20,  # Scale 0-5 to 0-100
                    "color": summary.get("verdict_color", "#6C757D")
                }
        
        # Display verdicts
        if verdicts:
            st.markdown("### 📊 Analysis Results Summary")
            cols = st.columns(len(verdicts))
            
            for i, (analysis_name, verdict_info) in enumerate(verdicts.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div style="background: {verdict_info['color']}; padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                        <h4 style="color: white; margin: 0;">{analysis_name.upper()}</h4>
                        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;">{verdict_info['verdict']}</p>
                        <p style="color: white; margin: 0; font-size: 0.9rem;">Confidence: {verdict_info['confidence']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Calculate overall risk
            overall_risk = self.calculate_overall_risk(verdicts)
            
            st.markdown("### 🎯 OVERALL VERDICT")
            st.markdown(f"""
            <div style="background: {overall_risk['color']}; padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
                <h2 style="color: white; margin: 0;">{overall_risk['icon']} {overall_risk['verdict']}</h2>
                <p style="color: white; margin: 1rem 0 0 0; font-size: 1.1rem;">{overall_risk['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed recommendations
            st.markdown("### 📋 Recommendations")
            
            if overall_risk['verdict'] == "HIGHLY SUSPICIOUS":
                st.error("""
                **⚠️ CRITICAL WARNING - HIGH RISK DETECTED**
                
                **Immediate Actions:**
                1. **DO NOT** trust this video for any official or verification purposes
                2. Report the video to appropriate authorities if it contains misleading information
                3. Seek additional verification from multiple independent sources
                4. Consider the source and context of the video carefully
                
                **Technical Findings:**
                - Multiple analysis methods detected significant manipulation
                - High probability of artificial content creation
                - Metadata suggests potential tampering
                """)
            elif overall_risk['verdict'] == "SUSPICIOUS":
                st.warning("""
                **⚠️ CAUTION REQUIRED - MODERATE RISK DETECTED**
                
                **Recommended Actions:**
                1. Verify information from this video with other reliable sources
                2. Consider the credibility of the video source
                3. Look for additional context or corroborating evidence
                4. Be aware that some artificial manipulation may be present
                
                **Technical Findings:**
                - Some analysis methods detected potential issues
                - Minor inconsistencies found across different verification methods
                """)
            elif overall_risk['verdict'] == "LOW RISK":
                st.info("""
                **ℹ️ MINIMAL CONCERNS - LOW RISK DETECTED**
                
                **General Guidance:**
                1. Video appears mostly authentic but has minor inconsistencies
                2. Usable for general reference but verify critical information
                3. Consider typical video compression artifacts as possible cause
                
                **Technical Findings:**
                - Most analysis methods indicate authenticity
                - Minor technical anomalies detected
                """)
            else:  # AUTHENTIC
                st.success("""
                **✅ HIGH CONFIDENCE - AUTHENTIC VIDEO**
                
                **Verification Complete:**
                1. Video appears authentic across all verification methods
                2. No significant signs of artificial manipulation detected
                3. Metadata consistent with typical video recordings
                
                **Technical Findings:**
                - All analysis methods confirm authenticity
                - Consistent results across deepfake, lip sync, and metadata analysis
                """)
            
            # Save complete report
            report_path = os.path.join(self.results_dir, f"complete_{os.path.basename(video_path)}.json")
            try:
                with open(report_path, 'w') as f:
                    json.dump(all_results, f, indent=2, cls=NumpyEncoder)
                
                # Download button
                with open(report_path, 'r') as f:
                    report_data = f.read()
                
                st.download_button(
                    label="📥 Download Complete Report",
                    data=report_data,
                    file_name=os.path.basename(report_path),
                    mime="application/json",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ Error saving report: {e}")
    
    def calculate_overall_risk(self, verdicts):
        """Calculate overall risk based on individual verdicts."""
        # Calculate risk scores
        risk_scores = []
        
        for analysis_name, verdict_info in verdicts.items():
            verdict = verdict_info["verdict"].upper()
            confidence = verdict_info["confidence"] / 100  # Convert to 0-1 scale
            
            if "FAKE" in verdict or "HIGH" in verdict:
                risk_score = 0.8 + (confidence * 0.2)  # 0.8-1.0
            elif "SUSPICIOUS" in verdict or "MODERATE" in verdict:
                risk_score = 0.5 + (confidence * 0.3)  # 0.5-0.8
            elif "LOW" in verdict:
                risk_score = 0.2 + (confidence * 0.3)  # 0.2-0.5
            elif "REAL" in verdict or "NO STRONG EVIDENCE" in verdict:
                risk_score = 0.0 + (confidence * 0.2)  # 0.0-0.2
            else:
                risk_score = 0.5  # Default for inconclusive
            
            risk_scores.append(risk_score)
        
        # Calculate average risk
        if risk_scores:
            avg_risk = np.mean(risk_scores)
        else:
            avg_risk = 0.5
        
        # Determine overall verdict
        if avg_risk >= 0.7:
            return {
                "verdict": "HIGHLY SUSPICIOUS",
                "description": "Multiple analyses indicate potential manipulation",
                "color": "#FF6B6B",
                "icon": "🔴"
            }
        elif avg_risk >= 0.5:
            return {
                "verdict": "SUSPICIOUS",
                "description": "Some analyses show potential issues",
                "color": "#FFD166",
                "icon": "🟡"
            }
        elif avg_risk >= 0.3:
            return {
                "verdict": "LOW RISK",
                "description": "Minor inconsistencies detected",
                "color": "#4ECDC4",
                "icon": "🔵"
            }
        else:
            return {
                "verdict": "AUTHENTIC",
                "description": "No significant manipulation detected",
                "color": "#06D6A0",
                "icon": "✅"
            }


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
    .analysis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .mode-selector {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .stSpinner > div {
        border-color: #667eea !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Video Authenticity Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">Complete Video Verification System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="analysis-card">
            <h3 style="color: white; margin: 0;">🎯 Select Analysis Mode</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis mode selection
        analysis_mode = st.radio(
            "",
            ["🤖 Deepfake Detection", "👄 Lip Sync Analysis", "🔍 Metadata Analysis", "🔬 Complete Analysis (All 3)"],
            index=0,
            label_visibility="collapsed"
        )
        
        # Get available models
        available_models = {}
        for model_name, model_path in MODEL_PATHS.items():
            if os.path.exists(model_path):
                available_models[model_name] = model_path
        
        # Always use all available models for modes that require them
        selected_models = available_models
        
        # Deepfake analysis parameters
        if analysis_mode in ["🤖 Deepfake Detection", "🔬 Complete Analysis (All 3)"]:
            st.markdown("### ⚙️ Deepfake Parameters")
            frames_per_second = st.slider(
                "Frames per second:",
                min_value=1,
                max_value=10,
                value=2,
                help="Higher values provide more detailed analysis"
            )
        
        # Lip sync parameters
        if analysis_mode in ["👄 Lip Sync Analysis", "🔬 Complete Analysis (All 3)"]:
            st.markdown("### 👄 Lip Sync Parameters")
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
        
        # Metadata parameters
        if analysis_mode in ["🔍 Metadata Analysis", "🔬 Complete Analysis (All 3)"]:
            st.markdown("### 🔍 Metadata Parameters")
            baseline_name = None
            metadata_analyzer = MetadataAnalyzer()
            available_baselines = metadata_analyzer.get_available_baselines()
            if available_baselines:
                baseline_name = st.selectbox(
                    "Select Baseline (Optional)",
                    [""] + available_baselines,
                    help="Compare against a pre-built metadata baseline"
                )
                if baseline_name == "":
                    baseline_name = None
        
        # System info
        st.markdown("### 📊 System Information")
        st.info(f"**Device:** {'🚀 GPU' if torch.cuda.is_available() else '💻 CPU'}")
        if analysis_mode in ["🤖 Deepfake Detection", "🔬 Complete Analysis (All 3)"]:
            st.info(f"**Models Available:** {len(available_models)}")
        
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
        
        # Display video preview and info
        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(uploaded_file)
        with col2:
            mode_descriptions = {
                "🤖 Deepfake Detection": "AI-powered face manipulation detection",
                "👄 Lip Sync Analysis": "Audio-visual synchronization verification",
                "🔍 Metadata Analysis": "Comprehensive forensic metadata analysis",
                "🔬 Complete Analysis (All 3)": "Full video authenticity verification"
            }
            st.markdown(f"""
            <div class="analysis-card">
                <h4 style="color: white; margin: 0 0 1rem 0;">📹 Video Details</h4>
                <p style="color: white; margin: 0.5rem 0;"><strong>File:</strong> {uploaded_file.name[:30]}{'...' if len(uploaded_file.name) > 30 else ''}</p>
                <p style="color: white; margin: 0.5rem 0;"><strong>Size:</strong> {uploaded_file.size / (1024*1024):.2f} MB</p>
                <p style="color: white; margin: 0.5rem 0;"><strong>Mode:</strong> {analysis_mode}</p>
                <p style="color: white; margin: 0.5rem 0;"><strong>Analysis:</strong> {mode_descriptions.get(analysis_mode, 'Custom analysis')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis button
        if st.button(f"🚀 Start {analysis_mode}", type="primary", use_container_width=True):
            try:
                # Initialize appropriate pipeline based on selected mode
                if analysis_mode == "🤖 Deepfake Detection":
                    if not selected_models:
                        st.error("❌ No models found. Please check your model paths.")
                        return
                    
                    pipeline = DeepfakeDetectionPipeline()
                    with st.spinner(f"Running deepfake detection analysis..."):
                        results = pipeline.analyze_video(video_path, selected_models, frames_per_second)
                        if results:
                            pipeline.display_deepfake_report(results["deepfake_analysis"])
                
                elif analysis_mode == "👄 Lip Sync Analysis":
                    pipeline = LipSyncAnalyzer()
                    with st.spinner("Running lip sync analysis..."):
                        results = pipeline.analyze_video(video_path, cer_threshold, conf_threshold, freeze_limit)
                        if results:
                            pipeline.display_lip_sync_report(results)
                
                elif analysis_mode == "🔍 Metadata Analysis":
                    pipeline = MetadataAnalysisPipeline()
                    with st.spinner("Running metadata analysis..."):
                        results = pipeline.analyze_video(video_path, baseline_name)
                        if results:
                            pipeline.display_metadata_report(results, video_path)
                
                elif analysis_mode == "🔬 Complete Analysis (All 3)":
                    pipeline = CompleteAnalysisPipeline()
                    with st.spinner("Running complete analysis (this may take a while)..."):
                        results = pipeline.analyze_video(
                            video_path, 
                            selected_models, 
                            frames_per_second,
                            baseline_name,
                            cer_threshold,
                            conf_threshold,
                            freeze_limit
                        )
                
                # Show success message
                if results:
                    st.balloons()
                    st.success(f"✅ {analysis_mode} completed successfully!")
            
            except Exception as e:
                st.error(f"❌ Analysis Error: {str(e)}")
                import traceback
                with st.expander("View Error Details"):
                    st.code(traceback.format_exc())
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(video_path)
                except:
                    pass
    
    else:
        # Welcome message with mode descriptions
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 3rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
            <h2 style="color: #2c3e50; margin: 0 0 1rem 0;">🔬 Video Authenticity Verification</h2>
            <p style="color: #2c3e50; margin: 0; font-size: 1.1rem;">
                Upload a video file and select an analysis mode to verify its authenticity
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis mode descriptions
        st.markdown("### ✨ Available Analysis Modes")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="mode-selector">
                <h4>🤖 Deepfake Detection</h4>
                <p>Detect AI-generated face manipulations using multiple models</p>
                <ul style="text-align: left; padding-left: 1.2rem;">
                    <li>Face detection & extraction</li>
                    <li>Multiple model analysis</li>
                    <li>Ensemble verdict</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="mode-selector">
                <h4>👄 Lip Sync Analysis</h4>
                <p>Verify audio-visual synchronization using lip reading</p>
                <ul style="text-align: left; padding-left: 1.2rem;">
                    <li>Real-time lip tracking</li>
                    <li>Speech-text alignment</li>
                    <li>Sync anomaly detection</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="mode-selector">
                <h4>🔍 Metadata Analysis</h4>
                <p>Analyze video metadata for tampering evidence</p>
                <ul style="text-align: left; padding-left: 1.2rem;">
                    <li>File integrity checks</li>
                    <li>Encoding analysis</li>
                    <li>Edit detection</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="mode-selector">
                <h4>🔬 Complete Analysis</h4>
                <p>Combine all 3 methods for comprehensive verification</p>
                <ul style="text-align: left; padding-left: 1.2rem;">
                    <li>All detection methods</li>
                    <li>Cross-verification</li>
                    <li>Overall risk assessment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p><strong>Video Authenticity Analyzer</strong> | Complete Video Verification System</p>
        <p style="font-size: 0.9rem;">Deepfake Detection • Lip Sync Analysis • Metadata Forensics</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()