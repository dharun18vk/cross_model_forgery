<<<<<<< HEAD
import streamlit as st
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import deque
import sys
import tempfile
import time

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from preprocessing.model_lipreading import LipReadingModel
from preprocessing.dataset_lip import build_vocab
from preprocessing.extract_mouth import extract_mouth_frame

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# 🎥 MAIN APPLICATION
# =====================================================
def main():
    st.set_page_config(
        page_title="Lip Sync Verification",
        page_icon="👄",
        layout="wide"
    )
    
    st.title("👄 Lip Sync Verification System")
    st.markdown("---")
    
    # Load model (cached)
    with st.spinner("Loading model..."):
        model, device, idx_to_char = load_model()
    st.success("✅ Model loaded successfully!")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Threshold sliders
    cer_threshold = st.sidebar.slider(
        "CER Threshold",
        min_value=0.1,
        max_value=0.8,
        value=CER_THRESHOLD,
        step=0.05,
        help="Higher values make the system more sensitive to text changes"
    )
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=CONF_THRESHOLD,
        step=0.05,
        help="Minimum confidence required for REAL classification"
    )
    
    freeze_limit = st.sidebar.slider(
        "Freeze Limit",
        min_value=5,
        max_value=50,
        value=FREEZE_LIMIT,
        step=1,
        help="Number of frames with same text before flagging as FAKE"
    )
    
    # Video upload
    st.sidebar.header("📹 Video Input")
    video_file = st.sidebar.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )
    
    # Use sample video if no file uploaded
    use_sample = st.sidebar.checkbox("Use sample video", value=False)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Video Analysis")
        
        if video_file is not None:
            # Save uploaded file to temporary location
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            video_path = tfile.name
            st.info(f"📹 Analyzing uploaded video: {video_file.name}")
        elif use_sample:
            video_path = "video/real/000.mp4"
            st.info("📹 Using sample video: video/real/000.mp4")
        else:
            video_path = None
            st.warning("Please upload a video file or select 'Use sample video'")
            return
        
        # Initialize session state for tracking
        if 'analysis_started' not in st.session_state:
            st.session_state.analysis_started = False
        if 'results' not in st.session_state:
            st.session_state.results = {
                'frame_count': 0,
                'fake_count': 0,
                'real_count': 0,
                'total_decisions': 0,
                'all_predictions': [],
                'all_texts': []
            }
        
        # Analysis button
        if st.button("▶️ Start Analysis", type="primary") or st.session_state.analysis_started:
            st.session_state.analysis_started = True
            
            # Create containers for live updates
            video_placeholder = st.empty()
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            text_placeholder = st.empty()
            
            # Initialize analysis
            cap = cv2.VideoCapture(video_path)
            buffer = deque(maxlen=SEQ_LEN)
            prev_text = ""
            freeze_count = 0
            
            # Reset results
            st.session_state.results = {
                'frame_count': 0,
                'fake_count': 0,
                'real_count': 0,
                'total_decisions': 0,
                'all_predictions': [],
                'all_texts': []
            }
            
            # Progress bar
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                st.session_state.results['frame_count'] += 1
                current_frame = st.session_state.results['frame_count']
                
                # Update progress
                if total_frames > 0:
                    progress_bar.progress(min(current_frame / total_frames, 1.0))
                
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
                               .to(device)
                    
                    with torch.no_grad():
                        logits = model(seq)
                        probs = logits.softmax(dim=-1)[0]
                        curr_text = greedy_decode(probs, idx_to_char)
                    
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
                            st.session_state.results['fake_count'] += 1
                        else:
                            status = "✅ REAL"
                            status_color = "success"
                            prediction = "REAL"
                            st.session_state.results['real_count'] += 1
                        
                        st.session_state.results['total_decisions'] += 1
                    
                    # Store results
                    st.session_state.results['all_predictions'].append(prediction)
                    st.session_state.results['all_texts'].append(curr_text)
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
                    if len(st.session_state.results['all_predictions']) > 1:
                        with chart_placeholder.container():
                            import pandas as pd
                            import plotly.graph_objects as go
                            
                            df = pd.DataFrame({
                                'frame': range(len(st.session_state.results['all_predictions'])),
                                'prediction': [1 if p == "REAL" else 0 for p in st.session_state.results['all_predictions']]
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
                    with text_placeholder.container():
                        st.subheader("Recent Text Predictions")
                        recent_texts = st.session_state.results['all_texts'][-5:]
                        for i, text in enumerate(recent_texts[::-1]):
                            if text:
                                st.text(f"Seq {len(st.session_state.results['all_texts']) - i}: {text}")
                
                # Add small delay for better visualization
                time.sleep(0.01)
            
            cap.release()
            progress_bar.progress(1.0)
            st.balloons()
            
            # Display final results
            st.success("✅ Analysis Complete!")
            
    with col2:
        st.header("📊 Results Summary")
        
        if st.session_state.results['total_decisions'] > 0:
            # Calculate metrics
            total_decisions = st.session_state.results['total_decisions']
            real_count = st.session_state.results['real_count']
            fake_count = st.session_state.results['fake_count']
            fake_ratio = fake_count / total_decisions if total_decisions > 0 else 0
            
            # Display metrics
            st.metric("Total Frames Processed", st.session_state.results['frame_count'])
            st.metric("Total Decisions", total_decisions)
            st.metric("REAL Count", real_count)
            st.metric("FAKE Count", fake_count)
            st.metric("Fake Ratio", f"{fake_ratio:.2%}")
            
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
            if st.session_state.results['all_texts']:
                unique_texts = list(dict.fromkeys([t for t in st.session_state.results['all_texts'] if t]))
                for i, text in enumerate(unique_texts[:5]):
                    st.text(f"{i+1}. {text}")
        else:
            st.info("Run analysis to see results here")
        
        # Download results button
        if st.session_state.results['total_decisions'] > 0:
            import json
            results_json = json.dumps(st.session_state.results, indent=2)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=results_json,
                file_name="lip_sync_results.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.caption("Lip Sync Verification System | Thresholds can be adjusted in the sidebar")

if __name__ == "__main__":
=======
import streamlit as st
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import deque
import sys
import tempfile
import time

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from preprocessing.model_lipreading import LipReadingModel
from preprocessing.dataset_lip import build_vocab
from preprocessing.extract_mouth import extract_mouth_frame

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# 🎥 MAIN APPLICATION
# =====================================================
def main():
    st.set_page_config(
        page_title="Lip Sync Verification",
        page_icon="👄",
        layout="wide"
    )
    
    st.title("👄 Lip Sync Verification System")
    st.markdown("---")
    
    # Load model (cached)
    with st.spinner("Loading model..."):
        model, device, idx_to_char = load_model()
    st.success("✅ Model loaded successfully!")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Threshold sliders
    cer_threshold = st.sidebar.slider(
        "CER Threshold",
        min_value=0.1,
        max_value=0.8,
        value=CER_THRESHOLD,
        step=0.05,
        help="Higher values make the system more sensitive to text changes"
    )
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=CONF_THRESHOLD,
        step=0.05,
        help="Minimum confidence required for REAL classification"
    )
    
    freeze_limit = st.sidebar.slider(
        "Freeze Limit",
        min_value=5,
        max_value=50,
        value=FREEZE_LIMIT,
        step=1,
        help="Number of frames with same text before flagging as FAKE"
    )
    
    # Video upload
    st.sidebar.header("📹 Video Input")
    video_file = st.sidebar.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )
    
    # Use sample video if no file uploaded
    use_sample = st.sidebar.checkbox("Use sample video", value=False)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Video Analysis")
        
        if video_file is not None:
            # Save uploaded file to temporary location
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            video_path = tfile.name
            st.info(f"📹 Analyzing uploaded video: {video_file.name}")
        elif use_sample:
            video_path = "video/real/000.mp4"
            st.info("📹 Using sample video: video/real/000.mp4")
        else:
            video_path = None
            st.warning("Please upload a video file or select 'Use sample video'")
            return
        
        # Initialize session state for tracking
        if 'analysis_started' not in st.session_state:
            st.session_state.analysis_started = False
        if 'results' not in st.session_state:
            st.session_state.results = {
                'frame_count': 0,
                'fake_count': 0,
                'real_count': 0,
                'total_decisions': 0,
                'all_predictions': [],
                'all_texts': []
            }
        
        # Analysis button
        if st.button("▶️ Start Analysis", type="primary") or st.session_state.analysis_started:
            st.session_state.analysis_started = True
            
            # Create containers for live updates
            video_placeholder = st.empty()
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            text_placeholder = st.empty()
            
            # Initialize analysis
            cap = cv2.VideoCapture(video_path)
            buffer = deque(maxlen=SEQ_LEN)
            prev_text = ""
            freeze_count = 0
            
            # Reset results
            st.session_state.results = {
                'frame_count': 0,
                'fake_count': 0,
                'real_count': 0,
                'total_decisions': 0,
                'all_predictions': [],
                'all_texts': []
            }
            
            # Progress bar
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                st.session_state.results['frame_count'] += 1
                current_frame = st.session_state.results['frame_count']
                
                # Update progress
                if total_frames > 0:
                    progress_bar.progress(min(current_frame / total_frames, 1.0))
                
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
                               .to(device)
                    
                    with torch.no_grad():
                        logits = model(seq)
                        probs = logits.softmax(dim=-1)[0]
                        curr_text = greedy_decode(probs, idx_to_char)
                    
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
                            st.session_state.results['fake_count'] += 1
                        else:
                            status = "✅ REAL"
                            status_color = "success"
                            prediction = "REAL"
                            st.session_state.results['real_count'] += 1
                        
                        st.session_state.results['total_decisions'] += 1
                    
                    # Store results
                    st.session_state.results['all_predictions'].append(prediction)
                    st.session_state.results['all_texts'].append(curr_text)
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
                    if len(st.session_state.results['all_predictions']) > 1:
                        with chart_placeholder.container():
                            import pandas as pd
                            import plotly.graph_objects as go
                            
                            df = pd.DataFrame({
                                'frame': range(len(st.session_state.results['all_predictions'])),
                                'prediction': [1 if p == "REAL" else 0 for p in st.session_state.results['all_predictions']]
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
                    with text_placeholder.container():
                        st.subheader("Recent Text Predictions")
                        recent_texts = st.session_state.results['all_texts'][-5:]
                        for i, text in enumerate(recent_texts[::-1]):
                            if text:
                                st.text(f"Seq {len(st.session_state.results['all_texts']) - i}: {text}")
                
                # Add small delay for better visualization
                time.sleep(0.01)
            
            cap.release()
            progress_bar.progress(1.0)
            st.balloons()
            
            # Display final results
            st.success("✅ Analysis Complete!")
            
    with col2:
        st.header("📊 Results Summary")
        
        if st.session_state.results['total_decisions'] > 0:
            # Calculate metrics
            total_decisions = st.session_state.results['total_decisions']
            real_count = st.session_state.results['real_count']
            fake_count = st.session_state.results['fake_count']
            fake_ratio = fake_count / total_decisions if total_decisions > 0 else 0
            
            # Display metrics
            st.metric("Total Frames Processed", st.session_state.results['frame_count'])
            st.metric("Total Decisions", total_decisions)
            st.metric("REAL Count", real_count)
            st.metric("FAKE Count", fake_count)
            st.metric("Fake Ratio", f"{fake_ratio:.2%}")
            
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
            if st.session_state.results['all_texts']:
                unique_texts = list(dict.fromkeys([t for t in st.session_state.results['all_texts'] if t]))
                for i, text in enumerate(unique_texts[:5]):
                    st.text(f"{i+1}. {text}")
        else:
            st.info("Run analysis to see results here")
        
        # Download results button
        if st.session_state.results['total_decisions'] > 0:
            import json
            results_json = json.dumps(st.session_state.results, indent=2)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=results_json,
                file_name="lip_sync_results.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.caption("Lip Sync Verification System | Thresholds can be adjusted in the sidebar")

if __name__ == "__main__":
>>>>>>> d6265bf9ccf1d66490ca26d8658512566d956537
    main()