import sys
from pathlib import Path
from collections import deque
import cv2
import torch
import numpy as np

# =====================================================
# 🔧 ADD PROJECT ROOT TO PATH
# =====================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from preprocessing.model_lipreading import LipReadingModel
from preprocessing.dataset_lip import build_vocab
from preprocessing.extract_mouth import extract_mouth_frame

# =====================================================
# 🔧 CONFIG
# =====================================================
VIDEO_PATH = "video/real/000.mp4"
MODEL_PATH = "preprocessing/models/lip_model_best.pth"

IMG_SIZE = 64
SEQ_LEN = 25                 # sliding window (~1 sec)
CER_THRESHOLD = 0.35
CONF_THRESHOLD = 0.45
FREEZE_LIMIT = 10

# =====================================================
# 🔥 DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

# =====================================================
# 🧠 LOAD MODEL
# =====================================================
char_to_idx, idx_to_char = build_vocab()

model = LipReadingModel(len(char_to_idx) + 1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded")

# =====================================================
# 🔤 GREEDY CTC DECODER
# =====================================================
def greedy_decode(probs):
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
# 🎥 VIDEO LOOP
# =====================================================
cap = cv2.VideoCapture(VIDEO_PATH)

print("📹 Video path:", VIDEO_PATH)
print("📹 Video opened:", cap.isOpened())

buffer = deque(maxlen=SEQ_LEN)
prev_text = ""
freeze_count = 0
frame_count = 0

# ================== COUNTERS ==================
fake_count = 0
real_count = 0
total_decisions = 0
# ==============================================

print("🎬 Starting video verification...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Debug raw video
    cv2.imshow("RAW VIDEO", frame)
    cv2.waitKey(1)

    mouth = extract_mouth_frame(frame, IMG_SIZE)
    if mouth is None:
        cv2.imshow("Lip Sync Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    mouth = mouth.astype("float32") / 255.0
    buffer.append(mouth)

    status = "COLLECTING..."
    color = (0, 255, 255)

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
            curr_text = greedy_decode(probs)

        drift = cer(curr_text, prev_text)
        confidence = probs.max(dim=1)[0].mean().item()
        lang_score = language_quality(curr_text)

        # Freeze detection
        if curr_text == prev_text and len(curr_text) > 5:
            freeze_count += 1
        else:
            freeze_count = 0

        # Warm-up phase
        if frame_count < SEQ_LEN * 3:
            status = "WARMING UP..."
            color = (255, 255, 0)
        else:
            if (
                drift > CER_THRESHOLD
                or confidence < CONF_THRESHOLD
                or lang_score < 2.5
                or freeze_count > FREEZE_LIMIT
            ):
                status = "❌ FAKE"
                color = (0, 0, 255)
                fake_count += 1
            else:
                status = "✅ REAL"
                color = (0, 255, 0)
                real_count += 1

            total_decisions += 1

        prev_text = curr_text

        # ================= OVERLAY =================
        cv2.putText(frame, f"TEXT: {curr_text}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)

        cv2.putText(frame, f"CER Drift: {drift:.2f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)

        cv2.putText(frame, f"Confidence: {confidence:.2f}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)

        cv2.putText(frame, f"Lang Score: {lang_score:.2f}",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)

    cv2.putText(frame, status,
                (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 3)

    cv2.imshow("Lip Sync Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n========== VERIFICATION SUMMARY ==========")
print(f"Total decisions made : {total_decisions}")
print(f"REAL count           : {real_count}")
print(f"FAKE count           : {fake_count}")

if total_decisions > 0:
    fake_ratio = fake_count / total_decisions
    print(f"Fake ratio           : {fake_ratio:.2f}")

    # ===== FINAL VIDEO-LEVEL PREDICTION =====
    if fake_ratio >= 0.40:
        final_prediction = "❌ FAKE"
    else:
        final_prediction = "✅ REAL"

    print(f"\nFINAL PREDICTION     : {final_prediction}")
else:
    print("No valid decisions were made.")

