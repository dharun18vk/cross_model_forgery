import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_lip import LipDataset, build_vocab
from model_lipreading import LipReadingModel

# =====================================================
# 🔥 DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

torch.backends.cudnn.benchmark = True

# =====================================================
# 📁 PATHS
# =====================================================
TRAIN_CSV = "C:/PROJECTS/FinalYearProject/data/splits/train.csv"
VAL_CSV   = "C:/PROJECTS/FinalYearProject/data/splits/val.csv"

MODEL_OUT = "models/lip_model_best.pth"
LOG_FILE  = "training_log.json"

# =====================================================
# ⚙️ CONFIG
# =====================================================
EPOCHS = 25
BATCH_SIZE = 8
IMG_SIZE = 64
LR = 1e-4
PATIENCE = 5        # early stopping

# =====================================================
# 📚 DATA
# =====================================================
train_ds = LipDataset(TRAIN_CSV, IMG_SIZE)
val_ds   = LipDataset(VAL_CSV, IMG_SIZE)

char_to_idx, _ = build_vocab()

def collate_fn(batch):
    frames, labels, _ = zip(*batch)
    T_max = max(f.shape[0] for f in frames)

    padded = torch.zeros(len(frames), T_max, 1, IMG_SIZE, IMG_SIZE)
    input_lengths = []

    for i, f in enumerate(frames):
        padded[i, :f.shape[0], 0] = f
        input_lengths.append(f.shape[0])

    return (
        padded.to(device),
        torch.tensor(input_lengths, device=device),
        torch.cat(labels).to(device),
        torch.tensor([len(l) for l in labels], device=device),
    )

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE,
    shuffle=True, collate_fn=collate_fn,
    num_workers=0
)

val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=collate_fn,
    num_workers=0
)

# =====================================================
# 🧠 MODEL (WITH DROPOUT)
# =====================================================
model = LipReadingModel(len(char_to_idx) + 1).to(device)

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val = float("inf")
epochs_no_improve = 0
history = []

print("🚀 Training started")

# =====================================================
# 🚀 TRAIN LOOP
# =====================================================
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for frames, in_len, targets, tgt_len in train_loader:
        out = model(frames)
        out = out.log_softmax(2).permute(1, 0, 2)

        loss = criterion(out, targets, in_len, tgt_len)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for frames, in_len, targets, tgt_len in val_loader:
            out = model(frames)
            out = out.log_softmax(2).permute(1, 0, 2)
            loss = criterion(out, targets, in_len, tgt_len)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train: {train_loss:.4f} | "
        f"Val: {val_loss:.4f}"
    )

    # ---------------- EARLY STOP ----------------
    if val_loss < best_val:
        best_val = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_OUT)
        print("💾 Best model saved")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("⏹ Early stopping triggered")
            break

# =====================================================
# 💾 SAVE LOG
# =====================================================
with open(LOG_FILE, "w") as f:
    json.dump(history, f, indent=2)

print("✅ Training complete")
