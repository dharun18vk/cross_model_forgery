import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =====================================================
# 📁 PATHS
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMES_ROOT = PROJECT_ROOT / "data" / "processed" / "frames"
MOUTH_ROOT  = PROJECT_ROOT / "data" / "processed" / "mouth"

# =====================================================
# 🧠 MEDIAPIPE SETUP
# =====================================================
mp_face_mesh = mp.solutions.face_mesh

# Mouth landmarks (MediaPipe standard)
MOUTH_LANDMARKS = list(range(61, 88))

# =====================================================
# ✂️ MOUTH CROPPING (COMMON)
# =====================================================
def crop_mouth(frame, landmarks, img_size=64, scale=2.0):
    """
    Crop mouth region from full frame using landmarks
    """
    h, w, _ = frame.shape

    xs = np.array([int(landmarks[i].x * w) for i in MOUTH_LANDMARKS])
    ys = np.array([int(landmarks[i].y * h) for i in MOUTH_LANDMARKS])

    cx, cy = xs.mean().astype(int), ys.mean().astype(int)
    size = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * scale)

    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(cx + size // 2, w)
    y2 = min(cy + size // 2, h)

    mouth = frame[y1:y2, x1:x2]
    if mouth.size == 0:
        return None

    mouth = cv2.resize(mouth, (img_size, img_size))
    mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
    return mouth

# =====================================================
# 🎞 OFFLINE PROCESSING (FRAMES → MOUTH IMAGES)
# =====================================================
def process_clip(frames_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as face_mesh:

        for frame_path in sorted(frames_dir.glob("*.jpg")):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                continue

            landmarks = result.multi_face_landmarks[0].landmark
            mouth = crop_mouth(frame, landmarks)

            if mouth is None:
                continue

            cv2.imwrite(str(out_dir / frame_path.name), mouth)

# =====================================================
# 🔴 LIVE / REAL-TIME USE (SINGLE FRAME)
# =====================================================
_live_face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

def extract_mouth_frame(frame, img_size=64):
    """
    Input : BGR frame
    Output: grayscale mouth (img_size x img_size) or None
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _live_face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    return crop_mouth(frame, landmarks, img_size=img_size)

# =====================================================
# ▶️ MAIN (OFFLINE PIPELINE)
# =====================================================
def main():
    clips = sorted(FRAMES_ROOT.iterdir())

    for clip in tqdm(clips, desc="Extracting mouth regions"):
        if not clip.is_dir():
            continue

        out_dir = MOUTH_ROOT / clip.name
        process_clip(clip, out_dir)

if __name__ == "__main__":
    main()
