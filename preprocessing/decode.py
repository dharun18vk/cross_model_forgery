import torch

def greedy_decode(probs, idx_to_char):
    """
    probs: Tensor of shape (T, C) after softmax
    """
    blank = 0
    pred = torch.argmax(probs, dim=-1).cpu().numpy()

    result = []
    prev = blank

    for p in pred:
        if p != prev and p != blank:
            result.append(idx_to_char.get(int(p), ""))
        prev = p

    return "".join(result)
