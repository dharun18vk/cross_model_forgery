import os
from pathlib import Path
import cv2
from retinaface import RetinaFace
from tqdm import tqdm

# ----------------------------
# CONFIGURATION - Set paths here
# ----------------------------

# List of dataset folders (each should have train/val → real/fake)
DATASET_DIRS = [
    "Dataset_deepfake",
    "CelebDF_Frames",
    "FFPP_Frames"
]

# Output folder for combined faces
OUTPUT_DIR = "D:/DEEPFAKE_TRAIN/combined_faces"

# RetinaFace confidence threshold
DETECTOR_THRESHOLD = 0.9
# ----------------------------

def get_existing_faces(output_dir):
    """
    Scan output directory and return a set of all already processed source images.
    Returns a dictionary with structure: {split: {label: set(source_image_stems)}}
    """
    existing_files = {}
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return existing_files
    
    for split in ['train', 'val']:
        split_path = output_path / split
        if not split_path.exists():
            continue
            
        existing_files[split] = {}
        for label in ['real', 'fake']:
            label_path = split_path / label
            if not label_path.exists():
                existing_files[split][label] = set()
                continue
                
            # Extract source image names from face filenames
            # Format: {source_name}_{dataset_name}_{face_index}.jpg
            source_images = set()
            for face_file in label_path.glob("*.jpg"):
                # Remove the _{dataset_name}_{face_index} part to get original source
                parts = face_file.stem.split('_')
                # Keep all parts except the last two (dataset_name and face_index)
                if len(parts) >= 3:
                    source_name = '_'.join(parts[:-2])
                    source_images.add(source_name)
                else:
                    # Fallback: if format doesn't match, use the whole name
                    source_images.add(face_file.stem)
            
            existing_files[split][label] = source_images
    
    return existing_files

def extract_faces_from_datasets(dataset_dirs, output_dir, detector_threshold=0.9):
    """
    Extract faces from multiple datasets using RetinaFace and save them into a single structured folder.
    Automatically resumes from previous progress.
    """
    output_dir = Path(output_dir)
    
    # Get already processed files for resume functionality
    existing_files = get_existing_faces(output_dir)
    print("🔍 Scanning for previously processed images...")
    
    total_skipped = 0
    total_processed = 0

    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            print(f"⚠️  Dataset directory not found: {dataset_dir}, skipping...")
            continue
            
        print(f"\n📁 Processing dataset: {dataset_dir.name}")

        for split in ['train', 'val']:
            for label in ['real', 'fake']:
                input_path = dataset_dir / split / label
                output_path = output_dir / split / label
                output_path.mkdir(parents=True, exist_ok=True)

                if not input_path.exists():
                    print(f"⚠️  Input path not found: {input_path}, skipping...")
                    continue

                # Get list of image files
                image_files = list(input_path.glob("*.*"))
                if not image_files:
                    print(f"ℹ️  No images found in: {input_path}")
                    continue
                
                # Filter out already processed images
                processed_set = existing_files.get(split, {}).get(label, set())
                images_to_process = []
                
                for img_file in image_files:
                    source_stem = img_file.stem
                    if source_stem not in processed_set:
                        images_to_process.append(img_file)
                    else:
                        total_skipped += 1
                
                print(f"🎯 {split}/{label}: {len(images_to_process)}/{len(image_files)} images to process "
                      f"({len(image_files) - len(images_to_process)} already done)")
                
                if not images_to_process:
                    continue

                # Process remaining images
                for img_file in tqdm(images_to_process, desc=f"{dataset_dir.name} {split}/{label}"):
                    try:
                        img = cv2.imread(str(img_file))
                        if img is None:
                            print(f"⚠️  Could not read image: {img_file}")
                            continue

                        detections = RetinaFace.detect_faces(str(img_file))
                        if isinstance(detections, dict):
                            faces_extracted = 0
                            for i, (key, face_info) in enumerate(detections.items()):
                                confidence = face_info["score"]
                                if confidence >= detector_threshold:
                                    x1, y1, x2, y2 = face_info["facial_area"]
                                    face_crop = img[y1:y2, x1:x2]

                                    # Unique filename to avoid overwriting
                                    base_name = img_file.stem
                                    unique_name = f"{base_name}_{dataset_dir.name}_{faces_extracted}.jpg"
                                    cv2.imwrite(str(output_path / unique_name), face_crop)
                                    faces_extracted += 1
                                    total_processed += 1
                            
                            if faces_extracted == 0:
                                print(f"⚠️  No faces detected in {img_file} (confidence < {detector_threshold})")

                    except Exception as e:
                        print(f"❌ Error processing {img_file}: {e}")

    # Summary
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"✅ Successfully processed: {total_processed} new faces")
    print(f"⏭️  Skipped (already processed): {total_skipped} images")
    print(f"💾 All faces saved in: '{output_dir}'")

if __name__ == "__main__":
    extract_faces_from_datasets(DATASET_DIRS, OUTPUT_DIR, DETECTOR_THRESHOLD)