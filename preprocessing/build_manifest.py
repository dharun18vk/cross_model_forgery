import os
import csv

MAIN_DIR = "C:\PROJECTS\FinalYearProject\data\main"
MOUTH_DIR = "C:\PROJECTS\FinalYearProject\data\processed\mouth"
OUT_FILE = "data/manifest.csv"
print("🔍 MAIN_DIR:", MAIN_DIR)
print("🔍 MOUTH_DIR:", MOUTH_DIR)

rows = []

if not os.path.exists(MAIN_DIR):
    print("❌ MAIN_DIR does not exist")
    exit()

if not os.path.exists(MOUTH_DIR):
    print("❌ MOUTH_DIR does not exist")
    exit()

speaker_folders = os.listdir(MAIN_DIR)
print(f"📁 Found {len(speaker_folders)} speaker folders")

for speaker_id in speaker_folders:
    speaker_path = os.path.join(MAIN_DIR, speaker_id)
    if not os.path.isdir(speaker_path):
        continue

    files = os.listdir(speaker_path)
    print(f"➡️ Speaker {speaker_id}: {len(files)} files")

    for file in files:
        if not file.endswith(".mp4"):
            continue

        clip_id = file.replace(".mp4", "")
        txt_file = clip_id + ".txt"

        txt_path = os.path.join(speaker_path, txt_file)
        sample_id = f"{speaker_id}_{clip_id}"
        mouth_path = os.path.join(MOUTH_DIR, sample_id)

        print("   🔹 Checking sample:", sample_id)

        if not os.path.exists(txt_path):
            print("      ❌ transcript missing")
            continue

        if not os.path.exists(mouth_path):
            print("      ❌ mouth folder missing:", mouth_path)
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

        if transcript == "":
            print("      ❌ empty transcript")
            continue

        rows.append([sample_id, mouth_path, transcript])
        print("      ✅ added")

print("\n📊 TOTAL VALID SAMPLES:", len(rows))

if len(rows) == 0:
    print("❌ No valid samples found. Manifest NOT created.")
    exit()

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_id", "mouth_dir", "transcript"])
    writer.writerows(rows)

print(f"\n✅ Manifest created at: {OUT_FILE}")