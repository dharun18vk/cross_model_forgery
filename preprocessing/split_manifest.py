import csv
import random

MANIFEST = "C:\PROJECTS\FinalYearProject\data\manifest.csv"
OUT_DIR = "C:\PROJECTS\FinalYearProject\data\splits"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

with open(MANIFEST, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

random.shuffle(rows)

n = len(rows)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_rows = rows[:n_train]
val_rows = rows[n_train:n_train + n_val]
test_rows = rows[n_train + n_val:]

def write_split(name, data):
    path = f"{OUT_DIR}/{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

import os
os.makedirs(OUT_DIR, exist_ok=True)

write_split("train", train_rows)
write_split("val", val_rows)
write_split("test", test_rows)

print("✅ Train/Val/Test split created")
print(f"Train: {len(train_rows)} | Val: {len(val_rows)} | Test: {len(test_rows)}")
