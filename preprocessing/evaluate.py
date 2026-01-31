import torch
from dataset_lip import LipDataset, build_vocab
from model_lipreading import LipReadingModel
from decode import greedy_decode
from metrics import wer

# ================= CONFIG =================
TEST_MANIFEST = "C:/PROJECTS/FinalYearProject/data/splits/test.csv"
MODEL_PATH = "lip_model_best.pth"
IMG_SIZE = 64
# =========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Using device:", device)

# -------- DATASET --------
dataset = LipDataset(TEST_MANIFEST, IMG_SIZE)
char_to_idx, idx_to_char = build_vocab()

# -------- MODEL --------
model = LipReadingModel(vocab_size=len(char_to_idx) + 1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

wers = []

print("🔍 Evaluating on test set...")

for i in range(len(dataset)):
    frames, _, gt_text = dataset[i]
    frames = frames.unsqueeze(0).unsqueeze(2).to(device)

    with torch.no_grad():
        outputs = model(frames)
        probs = outputs.softmax(2)[0]  # (T, C)

    pred_text = greedy_decode(probs, idx_to_char)
    score = wer(pred_text, gt_text)
    wers.append(score)

avg_wer = sum(wers) / len(wers)

print("\n========== TEST RESULTS ==========")
print(f"Total test samples : {len(wers)}")
print(f"Average WER        : {avg_wer:.4f}")
print("==================================")
