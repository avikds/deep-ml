def rejection_sampling_best_of_k(candidates, scores):
    """
    Select the highest-scoring candidate per prompt.
    """
    selected = []

    for prompt_candidates, prompt_scores in zip(candidates, scores):
        best_idx = max(
            range(len(prompt_scores)),
            key=lambda i: prompt_scores[i]
        )
        selected.append(prompt_candidates[best_idx])

    return selected