import numpy as np

def detect_attention_sinks(attn_weights: np.ndarray, threshold: float) -> dict:
    """
    Detect attention sink tokens from multi-head attention weight matrices.
    """
    # Average over heads and query positions, leaving one value per key position.
    avg_attention_received = np.mean(attn_weights, axis=(0, 1))

    # Positions are naturally ordered, so this is already sorted.
    sink_positions = np.where(avg_attention_received >= threshold)[0].tolist()

    avg_attention_received = np.round(avg_attention_received, 4).tolist()
    sink_scores = [
        avg_attention_received[pos]
        for pos in sink_positions
    ]

    return {
        "sink_positions": sink_positions,
        "avg_attention_received": avg_attention_received,
        "sink_scores": sink_scores,
    }