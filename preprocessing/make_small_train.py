import csv
import random

SRC = r"C:\PROJECTS\FinalYearProject\data\splits\train.csv"
DST = r"C:\PROJECTS\FinalYearProject\data\splits\train_small.csv"

N = 1000   # 👈 change to 500 / 2000 as needed

with open(SRC, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

random.shuffle(rows)
rows = rows[:N]

with open(DST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Created train_small.csv with {N} samples")
