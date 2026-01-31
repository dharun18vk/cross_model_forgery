import os
import cv2
import torch
import csv
import re
import numpy as np
from torch.utils.data import Dataset

# ---------------- VOCAB ----------------
CHARS = "abcdefghijklmnopqrstuvwxyz '"

def build_vocab():
    char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}
    idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}
    return char_to_idx, idx_to_char

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z ']", "", text)
    text = " ".join(text.split())
    return text

# ---------------- DATASET ----------------
class LipDataset(Dataset):
    def __init__(self, manifest_csv, img_size=64, max_frames=40):
        self.img_size = img_size
        self.max_frames = max_frames
        self.char_to_idx, _ = build_vocab()

        with open(manifest_csv, "r", encoding="utf-8") as f:
            self.samples = list(csv.DictReader(f))

    def load_mouth(self, mouth_dir):
        if not os.path.exists(mouth_dir):
            return None

        frames = []
        files = sorted(os.listdir(mouth_dir))[:self.max_frames]

        for f in files:
            img_path = os.path.join(mouth_dir, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype("float32") / 255.0
            frames.append(img)

        if len(frames) == 0:
            return None

        return torch.from_numpy(np.stack(frames))

    def text_to_labels(self, text):
        return torch.tensor(
            [self.char_to_idx[c] for c in text if c in self.char_to_idx],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        frames = self.load_mouth(row["mouth_dir"])
        if frames is None:
            return self.__getitem__((idx + 1) % len(self.samples))

        text = clean_text(row["transcript"])
        labels = self.text_to_labels(text)

        # 🔐 CTC safety
        if len(labels) == 0 or frames.shape[0] < len(labels):
            return self.__getitem__((idx + 1) % len(self.samples))

        return frames, labels, text
