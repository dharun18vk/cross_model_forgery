import cv2
from pathlib import Path
from tqdm import tqdm

# ROOTS (CHANGE ONLY IF YOUR PATH DIFFERS)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LRS2_ROOT = PROJECT_ROOT / "data" / "main"
OUT_ROOT = PROJECT_ROOT / "data" / "processed" / "frames"

TARGET_FPS = 25  # fixed fps for temporal consistency

def extract_frames(video_path: Path, out_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        return

    frame_interval = max(int(round(fps / TARGET_FPS)), 1)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_id, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            out_path = out_dir / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_id += 1

    cap.release()

def main():
    speaker_dirs = sorted(LRS2_ROOT.iterdir())

    for speaker_dir in tqdm(speaker_dirs, desc="Processing speakers"):
        if not speaker_dir.is_dir():
            continue

        for file in speaker_dir.iterdir():
            if file.suffix.lower() == ".mp4":
                clip_id = file.stem
                out_dir = OUT_ROOT / f"{speaker_dir.name}_{clip_id}"
                extract_frames(file, out_dir)

if __name__ == "__main__":
    main()
