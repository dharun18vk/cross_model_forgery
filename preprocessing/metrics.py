def wer(pred, ref):
    """
    Word Error Rate
    """
    pred_words = pred.split()
    ref_words = ref.split()

    if len(ref_words) == 0:
        return 1.0

    dp = [[0] * (len(ref_words) + 1) for _ in range(len(pred_words) + 1)]

    for i in range(len(pred_words) + 1):
        dp[i][0] = i
    for j in range(len(ref_words) + 1):
        dp[0][j] = j

    for i in range(1, len(pred_words) + 1):
        for j in range(1, len(ref_words) + 1):
            cost = 0 if pred_words[i - 1] == ref_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[-1][-1] / len(ref_words)
