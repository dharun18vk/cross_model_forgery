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

    "Celeb + Image Model (Level 2.5 - Advanced)":
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

    "Xception Progressive Fine-tuned (Level 2)":
        "DEEPFAKE_MODELS/progressive_fine_tuned_model/2nd_tuned_xception_model.pth",

    "Xception Progressive Fine-tuned (Final)":
        "DEEPFAKE_MODELS/progressive_fine_tuned_model/final_progressive_model.pth",

    # ==========================================================
    # NEW / EXPERIMENTAL MODELS
    # ==========================================================
    "New Deepfake Model (Experimental – Epoch 10)":
        "DEEPFAKE_MODELS/new_deepfake_model/checkpoint_epoch_10.pth",

    "New Deepfake Model (Working – Unverified)":
        "DEEPFAKE_MODELS/new_deepfake_model/unknown_working_model.pth",

    # ==========================================================
    # PRODUCTION / DEPLOYMENT READY
    # ==========================================================
    "Best EfficientNet-B4 Model (Production)":
        "DEEPFAKE_MODELS/best_B4_model.pth",

    "Xception Face Model (Deployment Ready)":
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
            st.error(f"❌ Error loading model {self.model_name}: {e}")
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
# MODIFIED PIPELINES WITH METADATA INTEGRATION
# =============================================================================

class SingleModelAnalysisPipeline:
    """Pipeline for single model analysis with face detection AND metadata."""
    def __init__(self):
        self.frame_extractor = FrameExtractor()
        self.face_detector = RetinaFaceDetector()
        self.metadata_analyzer = MetadataAnalyzer()
        self.forensic_analyzer = CompleteForensicMetadataAnalyzer()
        self.results_dir = "single_model_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def analyze_video(self, video_path, model_path, model_name, frames_per_second=1, baseline_name=None):
        """Analyze video with single model, face detection, AND metadata."""
        # Main header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
            <h2 style="color: white; text-align: center; margin: 0; font-size: 2rem;">🔍 Single Model Analysis</h2>
            <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1rem;">Deepfake Detection + Metadata Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Store all results
        all_results = {
            "deepfake_analysis": None,
            "metadata_analysis": None,
            "forensic_analysis": None
        }
        
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
        
        # Step 3: Initialize Model
        with st.container():
            st.markdown("### 🤖 STEP 3: Model Initialization")
            detector = DeepFakeDetector(model_path, model_name)
            if detector.model is None:
                st.error("❌ Model failed to load. Exiting.")
                return None
        
        # Step 4: Deepfake Analysis
        with st.container():
            st.markdown("### 🧪 STEP 4: Deepfake Analysis")
            with st.spinner("Analyzing faces..."):
                predictions = detector.predict_multiple_faces(faces)
            
            if not predictions:
                st.error("❌ No predictions generated.")
                return None
            
            # Generate deepfake report
            deepfake_report = self.generate_report(predictions, video_path, model_name)
            all_results["deepfake_analysis"] = deepfake_report
        
        # Step 5: Metadata Analysis
        with st.container():
            st.markdown("### 📋 STEP 5: Metadata Analysis")
            metadata_report = self.metadata_analyzer.create_metadata_report(video_path, baseline_name)
            if metadata_report:
                all_results["metadata_analysis"] = metadata_report
            
            # Run forensic analysis
            forensic_report = self.forensic_analyzer.analyze_video(video_path)
            if forensic_report:
                all_results["forensic_analysis"] = forensic_report
        
        # Step 6: Display Results
        with st.container():
            st.markdown("### 📊 STEP 6: Analysis Results")
            self.display_combined_results(all_results, video_path, deepfake_report)
        
        return all_results
    
    def generate_report(self, predictions, video_path, model_name):
        """Generate analysis report."""
        total_faces = len(predictions)
        
        if total_faces == 0:
            st.error("❌ No faces were successfully analyzed.")
            return None
        
        fake_faces = sum(1 for pred in predictions if pred['prediction'] == 'FAKE')
        real_faces = total_faces - fake_faces
        
        fake_confidence_avg = float(np.mean([pred['confidence'] for pred in predictions if pred['prediction'] == 'FAKE'])) if fake_faces > 0 else 0.0
        real_confidence_avg = float(np.mean([pred['confidence'] for pred in predictions if pred['prediction'] == 'REAL'])) if real_faces > 0 else 0.0
        
        report = {
            'video_path': video_path,
            'model_name': model_name,
            'analysis_timestamp': str(np.datetime64('now')),
            'total_faces_analyzed': int(total_faces),
            'fake_faces_detected': int(fake_faces),
            'real_faces_detected': int(real_faces),
            'fake_percentage': float((fake_faces / total_faces * 100) if total_faces > 0 else 0),
            'average_fake_confidence': float(fake_confidence_avg),
            'average_real_confidence': float(real_confidence_avg),
            'overall_verdict': "LIKELY FAKE" if fake_faces > real_faces else "LIKELY REAL",
            'confidence_score': float(max(fake_confidence_avg, real_confidence_avg)),
            'detailed_analysis': predictions
        }
        
        # Save report
        report_path = os.path.join(self.results_dir, f"analysis_{os.path.basename(video_path)}_{model_name}.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            st.success(f"✅ Report saved: {report_path}")
        except Exception as e:
            st.error(f"❌ Error saving report: {e}")
        
        return report
    
    def display_combined_results(self, all_results, video_path, deepfake_report):
        """Display combined deepfake and metadata results."""
        # Deepfake results
        st.markdown("## 🎯 DEEPFAKE ANALYSIS RESULTS")
        st.markdown("---")
        
        if deepfake_report:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model Used", deepfake_report['model_name'])
            with col2:
                st.metric("Faces Analyzed", deepfake_report['total_faces_analyzed'])
            with col3:
                st.metric("Fake %", f"{deepfake_report['fake_percentage']:.1f}%")
            with col4:
                verdict = deepfake_report['overall_verdict']
                verdict_color = "#FF6B6B" if verdict == "LIKELY FAKE" else "#4ECDC4"
                st.markdown(f"""
                <div style="background: {verdict_color}; padding: 0.5rem; border-radius: 10px; text-align: center;">
                    <h4 style="color: white; margin: 0;">Verdict: {verdict}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualization
            df = pd.DataFrame(deepfake_report['detailed_analysis'])
            
            # Results Distribution Pie Chart
            fig_pie = px.pie(
                names=['Real', 'Fake'],
                values=[deepfake_report['real_faces_detected'], deepfake_report['fake_faces_detected']],
                title='Detection Results',
                color=['Real', 'Fake'],
                color_discrete_map={'Real': '#4ECDC4', 'Fake': '#FF6B6B'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Metadata results
        forensic_report = all_results.get("forensic_analysis")
        if forensic_report:
            st.markdown("---")
            st.markdown("## 🔍 METADATA ANALYSIS RESULTS")
            st.markdown("=" * 70)
            self.forensic_analyzer.display_detailed_forensic_report(forensic_report, video_path)
    
    def display_detailed_forensic_report(self, forensic_report, video_path):
        """Display forensic metadata analysis."""
        self.forensic_analyzer.display_detailed_forensic_report(forensic_report, video_path)

class MultiModelAnalysisPipeline:
    """Pipeline for multi-model ensemble analysis with face detection AND metadata."""
    def __init__(self):
        self.frame_extractor = FrameExtractor()
        self.face_detector = RetinaFaceDetector()
        self.metadata_analyzer = MetadataAnalyzer()
        self.forensic_analyzer = CompleteForensicMetadataAnalyzer()
        self.results_dir = "multi_model_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def analyze_video(self, video_path, selected_models, frames_per_second=1, baseline_name=None):
        """Analyze video with multiple models, face detection, AND metadata."""
        # Main header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
            <h2 style="color: white; text-align: center; margin: 0; font-size: 2rem;">🤖 Multi-Model Ensemble Analysis</h2>
            <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1rem;">Ensemble Deepfake Detection + Metadata Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Store all results
        all_results = {
            "deepfake_analysis": None,
            "metadata_analysis": None,
            "forensic_analysis": None
        }
        
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
        
        # Step 3: Initialize Models
        with st.container():
            st.markdown("### 🤖 STEP 3: Model Initialization")
            detectors = {}
            for model_name, model_path in selected_models.items():
                detector = DeepFakeDetector(model_path, model_name)
                if detector.model is not None:
                    detectors[model_name] = detector
            
            if not detectors:
                st.error("❌ No models loaded successfully.")
                return None
        
        # Step 4: Multi-model analysis
        with st.container():
            st.markdown("### 🔍 STEP 4: Multi-Model Analysis")
            all_predictions = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (model_name, detector) in enumerate(detectors.items()):
                status_text.text(f"Analyzing with {model_name}...")
                with st.spinner(f"Processing {model_name}..."):
                    predictions = detector.predict_multiple_faces(faces)
                    all_predictions[model_name] = predictions
                
                progress = float((i + 1) / len(detectors))
                progress_bar.progress(progress)
            
            # Generate ensemble report
            ensemble_report = self.generate_ensemble_report(all_predictions, video_path, faces)
            all_results["deepfake_analysis"] = ensemble_report
        
        # Step 5: Metadata Analysis
        with st.container():
            st.markdown("### 📋 STEP 5: Metadata Analysis")
            metadata_report = self.metadata_analyzer.create_metadata_report(video_path, baseline_name)
            if metadata_report:
                all_results["metadata_analysis"] = metadata_report
            
            # Run forensic analysis
            forensic_report = self.forensic_analyzer.analyze_video(video_path)
            if forensic_report:
                all_results["forensic_analysis"] = forensic_report
        
        # Step 6: Display Results
        with st.container():
            st.markdown("### 📊 STEP 6: Analysis Results")
            self.display_combined_results(all_results, video_path, ensemble_report)
        
        return all_results
    
    def generate_ensemble_report(self, all_predictions, video_path, faces):
        """Generate ensemble analysis report."""
        model_reports = {}
        
        for model_name, predictions in all_predictions.items():
            if not predictions:
                continue
                
            total_faces = len(predictions)
            fake_faces = sum(1 for pred in predictions if pred['prediction'] == 'FAKE')
            real_faces = total_faces - fake_faces
            
            fake_confidence_avg = float(np.mean([pred['confidence'] for pred in predictions if pred['prediction'] == 'FAKE'])) if fake_faces > 0 else 0.0
            real_confidence_avg = float(np.mean([pred['confidence'] for pred in predictions if pred['prediction'] == 'REAL'])) if real_faces > 0 else 0.0
            
            model_reports[model_name] = {
                'total_faces_analyzed': int(total_faces),
                'fake_faces_detected': int(fake_faces),
                'real_faces_detected': int(real_faces),
                'fake_percentage': float((fake_faces / total_faces * 100) if total_faces > 0 else 0),
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
            'total_faces_detected': len(faces),
            'model_reports': model_reports,
            'ensemble_report': model_reports.get('ENSEMBLE', {}),
            'all_predictions': all_predictions
        }
        
        # Save report
        report_path = os.path.join(self.results_dir, f"ensemble_analysis_{os.path.basename(video_path)}.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            st.success(f"✅ Ensemble report saved: {report_path}")
        except Exception as e:
            st.error(f"❌ Error saving report: {e}")
        
        return report
    
    def display_combined_results(self, all_results, video_path, ensemble_report):
        """Display combined deepfake and metadata results."""
        # Deepfake results
        st.markdown("## 🎯 ENSEMBLE DEEPFAKE ANALYSIS")
        st.markdown("---")
        
        if ensemble_report:
            ensemble = ensemble_report.get('ensemble_report', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Models Used", len(ensemble_report.get('models_used', [])))
            with col2:
                st.metric("Total Faces", ensemble_report.get('total_faces_detected', 0))
            with col3:
                fake_percentage = ensemble.get('fake_percentage', 0)
                st.metric("Ensemble Fake %", f"{fake_percentage:.1f}%")
            with col4:
                verdict = ensemble.get('overall_verdict', 'UNKNOWN')
                verdict_color = "#FF6B6B" if verdict == "LIKELY FAKE" else "#4ECDC4"
                st.markdown(f"""
                <div style="background: {verdict_color}; padding: 0.5rem; border-radius: 10px; text-align: center;">
                    <h4 style="color: white; margin: 0;">Verdict: {verdict}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # Model comparison chart
            model_names = []
            fake_percentages = []
            
            for model_name, model_report in ensemble_report.get('model_reports', {}).items():
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
                    title='Model Performance Comparison',
                    color='Fake Percentage',
                    color_continuous_scale='RdYlGn_r'
                )
                fig_comparison.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Metadata results
        forensic_report = all_results.get("forensic_analysis")
        if forensic_report:
            st.markdown("---")
            st.markdown("## 🔍 METADATA ANALYSIS RESULTS")
            st.markdown("=" * 70)
            self.forensic_analyzer.display_detailed_forensic_report(forensic_report, video_path)
    
    def display_detailed_forensic_report(self, forensic_report, video_path):
        """Display forensic metadata analysis."""
        self.forensic_analyzer.display_detailed_forensic_report(forensic_report, video_path)

# =============================================================================
# LIP SYNC ANALYSIS CLASS - REPLACED WITH streamlit_sync.py IMPLEMENTATION
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
    st.info(f"🔥 Using device: {device}")
    
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
        
        # Step 1: Load model
        with st.container():
            st.markdown("### 🤖 STEP 1: Loading Lip Reading Model")
            with st.spinner("Loading model (GPU only)..."):
                if not self.load_model_wrapper():
                    st.error("❌ Failed to load lip sync model")
                    return None
            
            st.success(f"✅ Model loaded successfully on {self.device}")
        
        # Step 2: Analyze video
        with st.container():
            st.markdown("### 🎬 STEP 2: Video Analysis")
            
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            buffer = deque(maxlen=SEQ_LEN)
            prev_text = ""
            freeze_count = 0
            
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
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Display video info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Frames", total_frames)
            with col2:
                st.metric("FPS", f"{fps:.2f}")
            with col3:
                st.metric("Duration", f"{total_frames/fps:.2f}s")
            
            # Analysis loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results['frame_count'] += 1
                current_frame = results['frame_count']
                
                # Update progress
                if total_frames > 0:
                    progress = min(current_frame / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 Processing frame {current_frame}/{total_frames}")
                
                # Extract mouth region
                mouth = extract_mouth_frame(frame, IMG_SIZE)
                if mouth is None:
                    continue
                
                mouth = mouth.astype("float32") / 255.0
                buffer.append(mouth)
                
                # Process when buffer is full
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
                    if current_frame < SEQ_LEN * 3:
                        status = "WARMING UP"
                        status_color = "warning"
                        prediction = "WARMUP"
                    else:
                        if (drift > cer_threshold or 
                            confidence < conf_threshold or 
                            lang_score < 2.5 or 
                            freeze_count > freeze_limit):
                            status = "❌ FAKE"
                            status_color = "error"
                            prediction = "FAKE"
                            results['fake_count'] += 1
                        else:
                            status = "✅ REAL"
                            status_color = "success"
                            prediction = "REAL"
                            results['real_count'] += 1
                        
                        results['total_decisions'] += 1
                    
                    # Store results
                    results['all_predictions'].append(prediction)
                    results['all_texts'].append(curr_text)
                    results['all_cer'].append(drift)
                    results['all_confidence'].append(confidence)
                    prev_text = curr_text
                    
                    # Display video frame with overlay
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Add overlay text
                    y_offset = 40
                    cv2.putText(frame_rgb, f"Status: {status}", 
                               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0) if prediction == "REAL" else (255, 0, 0), 2)
                    
                    cv2.putText(frame_rgb, f"Text: {curr_text}", 
                               (20, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 2)
                    
                    cv2.putText(frame_rgb, f"CER Drift: {drift:.2f}", 
                               (20, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 2)
                    
                    cv2.putText(frame_rgb, f"Confidence: {confidence:.2f}", 
                               (20, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 2)
                    
                    # Display video
                    video_placeholder = st.empty()
                    video_placeholder.image(frame_rgb, channels="RGB", 
                                          caption=f"Frame: {current_frame}")
                    
                    # Update metrics
                    with metrics_placeholder.container():
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Current Status", status)
                        with m2:
                            st.metric("CER Drift", f"{drift:.3f}")
                        with m3:
                            st.metric("Confidence", f"{confidence:.3f}")
                        with m4:
                            st.metric("Language Score", f"{lang_score:.2f}")
                    
                    # Update prediction chart
                    if len(results['all_predictions']) > 1:
                        chart_placeholder = st.empty()
                        with chart_placeholder.container():
                            import pandas as pd
                            import plotly.graph_objects as go
                            
                            df = pd.DataFrame({
                                'frame': range(len(results['all_predictions'])),
                                'prediction': [1 if p == "REAL" else 0 for p in results['all_predictions']]
                            })
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df['frame'],
                                y=df['prediction'],
                                mode='lines+markers',
                                name='Prediction',
                                line=dict(color='blue', width=2),
                                marker=dict(size=6)
                            ))
                            
                            fig.update_layout(
                                title="Real/Fake Predictions Over Time",
                                xaxis_title="Sequence Number",
                                yaxis_title="Prediction (1=Real, 0=Fake)",
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Display recent text predictions
                    text_placeholder = st.empty()
                    with text_placeholder.container():
                        st.subheader("Recent Text Predictions")
                        recent_texts = results['all_texts'][-5:]
                        for i, text in enumerate(recent_texts[::-1]):
                            if text:
                                st.text(f"Seq {len(results['all_texts']) - i}: {text}")
                
                # Add small delay for better visualization
                import time
                time.sleep(0.01)
            
            cap.release()
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            
            # Display final results
            st.success("✅ Analysis Complete!")
            
            # Display results summary
            st.header("📊 Results Summary")
            
            if results['total_decisions'] > 0:
                # Calculate metrics
                total_decisions = results['total_decisions']
                real_count = results['real_count']
                fake_count = results['fake_count']
                fake_ratio = fake_count / total_decisions if total_decisions > 0 else 0
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Frames Processed", results['frame_count'])
                with col2:
                    st.metric("Total Decisions", total_decisions)
                with col3:
                    st.metric("REAL Count", real_count)
                with col4:
                    st.metric("FAKE Count", fake_count)
                
                # Final prediction
                st.subheader("🎯 Final Prediction")
                if fake_ratio >= 0.40:
                    st.error("❌ **FAKE** - High probability of lip sync manipulation")
                else:
                    st.success("✅ **REAL** - Likely genuine lip movements")
                
                # Detailed breakdown
                st.subheader("📈 Detailed Analysis")
                
                # Create pie chart
                import plotly.graph_objects as go
                
                fig = go.Figure(data=[go.Pie(
                    labels=['REAL', 'FAKE', 'WARMUP'],
                    values=[real_count, fake_count, total_decisions - real_count - fake_count],
                    hole=.3
                )])
                
                fig.update_layout(
                    title="Prediction Distribution",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show sample predictions
                st.subheader("📝 Sample Predictions")
                if results['all_texts']:
                    unique_texts = list(dict.fromkeys([t for t in results['all_texts'] if t]))
                    for i, text in enumerate(unique_texts[:5]):
                        st.text(f"{i+1}. {text}")
        
        # Save results
        self.save_results(results, video_path)
        
        return results
    
    def save_results(self, results, video_path):
        """Save lip sync analysis results."""
        report = {
            'video_path': video_path,
            'analysis_timestamp': str(np.datetime64('now')),
            'analysis_mode': 'Lip Sync Analysis',
            'results': results
        }
        
        # Save report
        report_path = os.path.join(
            self.results_dir, 
            f"lip_sync_{os.path.basename(video_path)}.json"
        )
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            st.success(f"✅ Lip sync report saved: {report_path}")
        except Exception as e:
            st.error(f"❌ Error saving report: {e}")
        
        return report_path


# =============================================================================
# COMPLETE ANALYSIS PIPELINE WITH DETAILED METADATA AT BOTTOM
# =============================================================================

class CompleteVideoAnalyzer:
    """Complete video analysis with detailed metadata shown at bottom."""
    
    def __init__(self):
        self.frame_extractor = FrameExtractor()
        self.face_detector = RetinaFaceDetector()
        self.metadata_analyzer = MetadataAnalyzer()
        self.forensic_analyzer = CompleteForensicMetadataAnalyzer()
        self.results_dir = "complete_analysis_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def analyze_video(self, video_path, frames_per_second=1, baseline_name=None):
        """Complete video analysis with deepfake detection and detailed metadata."""
        # Main header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
            <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">🔬 Complete Video Analysis</h1>
            <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Deepfake Detection + Detailed Metadata Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Store all results for final display
        all_results = {
            "deepfake_analysis": None,
            "metadata_analysis": None,
            "forensic_analysis": None
        }
        
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
        
        # Step 3: Initialize Deepfake Models
        with st.container():
            st.markdown("### 🤖 STEP 3: Loading Deepfake Models")
            available_models = {}
            for model_name, model_path in MODEL_PATHS.items():
                if os.path.exists(model_path):
                    available_models[model_name] = model_path
            
            if not available_models:
                st.error("❌ No model files found!")
                return None
            
            model_status = st.status("🔄 **Loading models...**", expanded=True)
            detectors = {}
            
            for model_name, model_path in available_models.items():
                with model_status:
                    st.write(f"📦 Loading {model_name}...")
                    detector = DeepFakeDetector(model_path, model_name)
                    if detector.model is not None:
                        detectors[model_name] = detector
                        st.success(f"✅ {model_name} loaded successfully")
                    else:
                        st.error(f"❌ Failed to load {model_name}")
            
            model_status.update(label=f"✅ **Models Loaded** - {len(detectors)} models ready", state="complete")
        
        # Step 4: Deepfake Analysis
        with st.container():
            st.markdown("### 🧪 STEP 4: Deepfake Analysis")
            all_predictions = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (model_name, detector) in enumerate(detectors.items()):
                status_text.text(f"🧪 Analyzing with {model_name}...")
                predictions = detector.predict_multiple_faces(faces)
                all_predictions[model_name] = predictions
                
                progress = float((i + 1) / len(detectors))
                progress_bar.progress(progress)
            
            status_text.text("✅ All models completed analysis")
            
            # Generate deepfake report
            deepfake_report = self.generate_deepfake_report(all_predictions, video_path)
            all_results["deepfake_analysis"] = deepfake_report
        
        # Step 5: Basic Metadata Analysis
        with st.container():
            st.markdown("### 📋 STEP 5: Basic Metadata Analysis")
            metadata_report = self.metadata_analyzer.create_metadata_report(video_path, baseline_name)
            if metadata_report:
                all_results["metadata_analysis"] = metadata_report
                st.success("✅ Metadata analysis complete")
            else:
                st.warning("⚠️ Metadata analysis incomplete")
        
        # Step 6: Complete Forensic Metadata Analysis
        with st.container():
            st.markdown("### 🔍 STEP 6: Complete Forensic Metadata Analysis")
            forensic_report = self.forensic_analyzer.analyze_video(video_path)
            if forensic_report:
                all_results["forensic_analysis"] = forensic_report
                st.success("✅ Forensic analysis complete")
            else:
                st.warning("⚠️ Forensic analysis incomplete")
        
        # Step 7: Display Complete Results
        with st.container():
            st.markdown("### 📊 STEP 7: Complete Analysis Results")
            self.display_complete_results(all_results, video_path)
        
        return all_results
    
    def generate_deepfake_report(self, all_predictions, video_path):
        """Generate deepfake analysis report."""
        model_reports = {}
        
        for model_name, predictions in all_predictions.items():
            if not predictions:
                continue
                
            total_faces = len(predictions)
            fake_faces = sum(1 for pred in predictions if pred['prediction'] == 'FAKE')
            real_faces = total_faces - fake_faces
            
            fake_confidence_avg = float(np.mean([pred['confidence'] for pred in predictions if pred['prediction'] == 'FAKE'])) if fake_faces > 0 else 0.0
            real_confidence_avg = float(np.mean([pred['confidence'] for pred in predictions if pred['prediction'] == 'REAL'])) if real_faces > 0 else 0.0
            
            model_reports[model_name] = {
                'total_faces_analyzed': int(total_faces),
                'fake_faces_detected': int(fake_faces),
                'real_faces_detected': int(real_faces),
                'fake_percentage': float((fake_faces / total_faces * 100) if total_faces > 0 else 0),
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
            'model_reports': model_reports,
            'ensemble_report': model_reports.get('ENSEMBLE', {})
        }
        
        return report
    
    def display_complete_results(self, all_results, video_path):
        """Display all analysis results with metadata at bottom."""
        
        # ============================================
        # SECTION 1: DEEPFAKE ANALYSIS RESULTS (Top)
        # ============================================
        st.markdown("## 🎯 DEEPFAKE ANALYSIS RESULTS")
        st.markdown("---")
        
        deepfake_report = all_results.get("deepfake_analysis")
        if deepfake_report:
            ensemble_report = deepfake_report.get('ensemble_report', {})
            
            # Create metrics dashboard
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🤖 Models Used", len(deepfake_report.get('models_used', [])))
            with col2:
                st.metric("👥 Total Faces", sum([r.get('total_faces_analyzed', 0) for r in deepfake_report.get('model_reports', {}).values()]))
            with col3:
                fake_percentage = ensemble_report.get('fake_percentage', 0)
                st.metric("🎭 Fake %", f"{fake_percentage:.1f}%")
            with col4:
                verdict = ensemble_report.get('overall_verdict', 'UNKNOWN')
                verdict_color = "#FF6B6B" if verdict == "LIKELY FAKE" else "#4ECDC4"
                st.markdown(f"""
                <div style="background: {verdict_color}; padding: 0.5rem; border-radius: 10px; text-align: center;">
                    <h4 style="color: white; margin: 0;">Verdict: {verdict}</h4>
                </div>
                """, unsafe_allow_html=True)
            
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
        
        # ============================================
        # SECTION 2: COMPLETE METADATA ANALYSIS (Bottom)
        # ============================================
        st.markdown("## 🔬 COMPLETE METADATA & FORENSIC ANALYSIS")
        st.markdown("=" * 70)
        
        forensic_report = all_results.get("forensic_analysis")
        if forensic_report:
            self.forensic_analyzer.display_detailed_forensic_report(forensic_report, video_path)
        
        # ============================================
        # SECTION 3: COMBINED RISK ASSESSMENT
        # ============================================
        st.markdown("## 🚨 COMBINED RISK ASSESSMENT")
        st.markdown("---")
        
        # Calculate combined risk
        forensic_risk = 0
        forensic_verdict = "UNKNOWN"
        if forensic_report and forensic_report.get("forensic_summary"):
            forensic_risk = forensic_report["forensic_summary"].get("risk_score", 0)
            forensic_verdict = forensic_report["forensic_summary"].get("verdict", "UNKNOWN")
        
        deepfake_risk_score = ensemble_report.get('fake_percentage', 0) / 100 if ensemble_report else 0
        deepfake_risk = "HIGH" if deepfake_risk_score > 0.5 else "MEDIUM" if deepfake_risk_score > 0.25 else "LOW"
        
        # Combined risk calculation
        combined_risk_score = (forensic_risk * 0.6) + (deepfake_risk_score * 0.4)
        if combined_risk_score > 0.7:
            overall_risk = "HIGH"
            risk_color = "#FF6B6B"
        elif combined_risk_score > 0.4:
            overall_risk = "MEDIUM"
            risk_color = "#FFD166"
        else:
            overall_risk = "LOW"
            risk_color = "#06D6A0"
        
        # Display combined risk
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Forensic Risk:** {forensic_verdict}")
        
        with col2:
            st.info(f"**Deepfake Risk:** {deepfake_risk}")
        
        with col3:
            st.markdown(f"""
            <div style="background: {risk_color}; padding: 1rem; border-radius: 10px; text-align: center;">
                <h3 style="color: white; margin: 0;">Overall Risk: {overall_risk}</h3>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS FOR METADATA ANALYSIS
# =============================================================================

def run_metadata_analysis(video_path, baseline_name=None):
    """Run metadata analysis and display results."""
    st.markdown("---")
    st.markdown("### 🔍 Performing Metadata Analysis...")
    
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
    
    return {
        "metadata_report": metadata_report,
        "forensic_report": forensic_report
    }

def run_metadata_only_analysis(video_path, baseline_name=None):
    """Run metadata analysis only (no deepfake detection)."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">🔬 Metadata Analysis Only</h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Comprehensive Forensic Metadata Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Run metadata analysis
    metadata_results = run_metadata_analysis(video_path, baseline_name)
    
    # Create comprehensive report
    report = {
        "video_path": video_path,
        "analysis_timestamp": str(np.datetime64('now')),
        "analysis_mode": "Metadata Analysis Only",
        "metadata_results": metadata_results
    }
    
    # Save report
    report_path = f"metadata_only_{os.path.basename(video_path)}.json"
    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        st.success(f"✅ Metadata report saved: {report_path}")
    except Exception as e:
        st.error(f"❌ Error saving report: {e}")
    
    return report


# =============================================================================
# MODIFIED STREAMLIT APPLICATION WITH METADATA IN ALL MODES
# =============================================================================

def main():
    """Main Streamlit application."""
    # Configure page
    st.set_page_config(
        page_title="Complete Video Authenticity Analyzer",
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
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .forensic-box {
        background-color: #f8f9fa;
        border-left: 4px solid #6c757d;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    .forensic-alert {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    .forensic-danger {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    .forensic-success {
        background-color: #d1e7dd;
        border-left: 4px solid #198754;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Complete Video Authenticity Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">Deepfake Detection + Complete Forensic Metadata Analysis</p>', unsafe_allow_html=True)
    
    # Navigation
    # Navigation
    analysis_mode = st.sidebar.selectbox(
        "🎯 Select Analysis Mode",
        [
            "Single Model Analysis", 
            "Multi-Model Ensemble Analysis", 
            "Complete Analysis with Detailed Metadata",
            "Lip Sync Analysis",  # Add this line
            "Metadata Analysis Only"
        ]
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
            <h3 style="color: white; text-align: center; margin: 0;">🔧 Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Get available models
        available_models = {}
        for model_name, model_path in MODEL_PATHS.items():
            if os.path.exists(model_path):
                available_models[model_name] = model_path
        
        if not available_models and analysis_mode != "Metadata Analysis Only" and analysis_mode != "Lip Sync Analysis":
            st.error("❌ No model files found!")
            st.stop()
        
        # Model selection based on mode
        if analysis_mode == "Single Model Analysis":
            st.markdown("### 🤖 Model Selection")
            selected_model_name = st.selectbox(
                "Choose a model:",
                options=list(available_models.keys()),
                help="Select a single model for analysis",
                key="single_model_select"
            )
            selected_models = {selected_model_name: available_models[selected_model_name]}
        
        elif analysis_mode == "Multi-Model Ensemble Analysis":
            st.markdown("### 🤖 Model Selection")
            selected_model_names = st.multiselect(
                "Choose models for ensemble:",
                options=list(available_models.keys()),
                default=list(available_models.keys())[:min(3, len(available_models))],
                help="Select multiple models for ensemble analysis",
                key="multi_model_select"
            )
            selected_models = {name: available_models[name] for name in selected_model_names}
        
        elif analysis_mode == "Complete Analysis with Detailed Metadata":
            st.markdown("### 🤖 Model Selection")
            st.info("All available models will be used for complete analysis")
            selected_models = available_models
        
        elif analysis_mode == "Lip Sync Analysis":
            st.markdown("### 👄 Lip Sync Analysis")
            st.info("Audio-visual synchronization verification")
            selected_models = {}  # No deepfake models needed for lip sync
        else:  # Metadata Analysis Only
            st.markdown("### 🔍 Metadata Analysis Only")
            st.info("No deepfake models will be loaded - only metadata analysis will be performed")
            selected_models = {}
        
        # Analysis parameters - add lip sync parameters here
        st.markdown("### ⚙️ Analysis Parameters")
        
        if analysis_mode == "Lip Sync Analysis":
            # Lip sync specific parameters
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
            
            # Set default values for other parameters
            frames_per_second = 1
            confidence_threshold = 0.9
        
        elif analysis_mode != "Metadata Analysis Only":
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
            
            # Default values for lip sync parameters
            cer_threshold = None
            conf_threshold = None
            freeze_limit = None
        else:
            # Default values for metadata-only mode
            frames_per_second = 1
            confidence_threshold = 0.9
            cer_threshold = None
            conf_threshold = None
            freeze_limit = None
        
        # Baseline selection for any mode with metadata
        baseline_name = None
        if analysis_mode in ["Complete Analysis with Detailed Metadata", "Metadata Analysis Only"]:
            metadata_analyzer = MetadataAnalyzer()
            available_baselines = metadata_analyzer.get_available_baselines()
            if available_baselines:
                baseline_name = st.selectbox(
                    "Select Baseline (Optional)",
                    [""] + available_baselines,
                    help="Compare against a pre-built metadata baseline"
                )
        
        # System info
        st.markdown("### 📊 System Information")
        st.info(f"**Device:** {'🚀 GPU' if torch.cuda.is_available() else '💻 CPU'}")
        if analysis_mode not in ["Metadata Analysis Only", "Lip Sync Analysis"]:
            st.info(f"**Available Models:** {len(available_models)}")
            if analysis_mode not in ["Complete Analysis with Detailed Metadata", "Lip Sync Analysis"]:
                st.info(f"**Selected Models:** {len(selected_models)}")
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
                "Single Model Analysis": f"Single Model",
                "Multi-Model Ensemble Analysis": f"{len(selected_models)} Models",
                "Complete Analysis with Detailed Metadata": "All Models + Metadata",
                "Lip Sync Analysis": "Lip Sync Detection",
                "Metadata Analysis Only": "Metadata Only"
            }
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1.5rem; border-radius: 10px;">
                <h4 style="color: white; margin: 0 0 1rem 0;">📹 Video Details</h4>
                <p style="color: white; margin: 0.5rem 0;"><strong>File:</strong> {uploaded_file.name[:30]}{'...' if len(uploaded_file.name) > 30 else ''}</p>
                <p style="color: white; margin: 0.5rem 0;"><strong>Size:</strong> {uploaded_file.size / (1024*1024):.2f} MB</p>
                <p style="color: white; margin: 0.5rem 0;"><strong>Mode:</strong> {analysis_mode}</p>
                <p style="color: white; margin: 0.5rem 0;"><strong>Analysis:</strong> {mode_info.get(analysis_mode, 'Custom')}</p>
                <p style="color: white; margin: 0.5rem 0;"><strong>Forensic:</strong> {'✅ Enabled' if analysis_mode in ['Complete Analysis with Detailed Metadata', 'Metadata Analysis Only'] else '✅ Light'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Analysis button
        if st.button(f"🚀 Start {analysis_mode}", type="primary", use_container_width=True):
            try:
                if analysis_mode == "Single Model Analysis":
                    pipeline = SingleModelAnalysisPipeline()
                    model_name = list(selected_models.keys())[0]
                    model_path = list(selected_models.values())[0]
                    
                    with st.spinner(f"Running single model analysis with {model_name}..."):
                        report = pipeline.analyze_video(
                            video_path, 
                            model_path, 
                            model_name, 
                            frames_per_second,
                            baseline_name
                        )
                
                elif analysis_mode == "Multi-Model Ensemble Analysis":
                    pipeline = MultiModelAnalysisPipeline()
                    with st.spinner(f"Running multi-model analysis with {len(selected_models)} models..."):
                        report = pipeline.analyze_video(
                            video_path, 
                            selected_models, 
                            frames_per_second,
                            baseline_name
                        )
                
                elif analysis_mode == "Complete Analysis with Detailed Metadata":
                    pipeline = CompleteVideoAnalyzer()
                    with st.spinner(f"Running complete analysis with detailed metadata..."):
                        report = pipeline.analyze_video(
                            video_path, 
                            frames_per_second, 
                            baseline_name
                        )
                
                elif analysis_mode == "Lip Sync Analysis":
                    pipeline = LipSyncAnalyzer()
                    with st.spinner(f"Running lip sync analysis..."):
                        report = pipeline.analyze_video(
                            video_path, 
                            cer_threshold, 
                            conf_threshold, 
                            freeze_limit
                        )
                
                else:  # Metadata Analysis Only
                    with st.spinner(f"Running metadata analysis only..."):
                        report = run_metadata_only_analysis(video_path, baseline_name)
                
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
            <h2 style="color: #2c3e50; margin: 0 0 1rem 0;">🔬 Ready for Complete Analysis</h2>
            <p style="color: #2c3e50; margin: 0; font-size: 1.1rem;">
                Upload a video file to start comprehensive authenticity analysis with complete forensic metadata
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("### ✨ Analysis Modes Available")
        
        # Update the feature highlights in the welcome message section:
        col1, col2, col3, col4, col5 = st.columns(5)  # Change from 4 to 5 columns

        with col1:
            st.markdown("""
            <div class="forensic-box">
                <h4>🔍 Single Model</h4>
                <p>Deepfake detection with one model + metadata analysis</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="forensic-box">
                <h4>🤖 Multi-Model</h4>
                <p>Ensemble deepfake detection + metadata analysis</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="forensic-box">
                <h4>📊 Complete Analysis</h4>
                <p>All models + detailed forensic metadata</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div class="forensic-box">
                <h4>👄 Lip Sync</h4>
                <p>Audio-visual synchronization verification</p>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown("""
            <div class="forensic-box">
                <h4>🔬 Metadata Only</h4>
                <p>Only forensic metadata analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        analysis_sections = [
            ("[1] File Integrity", "SHA-256 hash, file size, verification"),
            ("[2] Creation Metadata", "Timestamps, author info, comments"),
            ("[3] Encoding Information", "Encoder details, bitrate, format"),
            ("[4] Frame Details", "Resolution, FPS, codec information"),
            ("[5] Edit Detection", "Tampering indicators, FPS mismatches"),
            ("[6] Forensic Summary", "Risk assessment and final verdict")
        ]
        
        for i, (title, desc) in enumerate(analysis_sections):
            if i % 2 == 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**{title}**\n\n{desc}")
            else:
                with col2:
                    st.info(f"**{title}**\n\n{desc}")
    
    # Update the footer:
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p><strong>Complete Video Authenticity Analyzer</strong> | Deepfake Detection + Forensic Metadata + Lip Sync Analysis</p>
        <p style="font-size: 0.9rem;">Now with audio-visual synchronization verification</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()