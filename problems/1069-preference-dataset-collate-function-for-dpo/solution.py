import numpy as np

def collate_preference_batch(batch, pad_token_id=0):
    """
    Collate a list of preference items into a padded batch for DPO training.
    """
    chosen_sequences = []
    rejected_sequences = []

    # Build full sequences first.
    for item in batch:
        prompt = list(item["prompt"])
        chosen = list(item["chosen"])
        rejected = list(item["rejected"])

        chosen_sequences.append(prompt + chosen)
        rejected_sequences.append(prompt + rejected)

    # Maximum length across both chosen and rejected sequences.
    max_len = 0
    for seq in chosen_sequences + rejected_sequences:
        max_len = max(max_len, len(seq))

    chosen_ids = []
    rejected_ids = []
    chosen_masks = []
    rejected_masks = []

    for item, chosen_seq, rejected_seq in zip(
        batch, chosen_sequences, rejected_sequences
    ):
        prompt_len = len(item["prompt"])
        chosen_len = len(item["chosen"])
        rejected_len = len(item["rejected"])

        # Right-padding.
        chosen_padded = chosen_seq + [pad_token_id] * (
            max_len - len(chosen_seq)
        )
        rejected_padded = rejected_seq + [pad_token_id] * (
            max_len - len(rejected_seq)
        )

        # Response-only boolean masks.
        chosen_mask = [False] * max_len
        rejected_mask = [False] * max_len

        for t in range(prompt_len, prompt_len + chosen_len):
            chosen_mask[t] = True

        for t in range(prompt_len, prompt_len + rejected_len):
            rejected_mask[t] = True

        chosen_ids.append(chosen_padded)
        rejected_ids.append(rejected_padded)
        chosen_masks.append(chosen_mask)
        rejected_masks.append(rejected_mask)

    return {
        "chosen_input_ids": np.asarray(chosen_ids).tolist(),
        "rejected_input_ids": np.asarray(rejected_ids).tolist(),
        "chosen_mask": np.asarray(chosen_masks).tolist(),
        "rejected_mask": np.asarray(rejected_masks).tolist(),
    }