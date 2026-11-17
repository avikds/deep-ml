import torch

def expert_choice_recursion(h, router_w, block, n_recursions, capacity):
    """
    Expert-choice recursive routing.

    Returns:
        h: updated token states, shape (T, d)
        depths: number of recursion steps per token, shape (T,), int64
    """
    h = h.clone()
    T = h.shape[0]
    device = h.device

    active = torch.ones(T, dtype=torch.bool, device=device)
    depths = torch.zeros(T, dtype=torch.int64, device=device)

    for r in range(n_recursions):
        # Capacity can be shared or specified independently per step.
        cap = capacity[r] if isinstance(capacity, (list, tuple)) else capacity

        active_count = int(active.sum().item())
        k = min(int(cap), active_count)

        if k <= 0:
            break

        # Calculate routing scores from the current token states.
        scores = torch.sigmoid(h @ router_w[r])

        # Inactive tokens cannot be selected.
        scores = scores.masked_fill(~active, float("-inf"))

        # Select the highest-scoring active tokens.
        selected = torch.topk(scores, k=k).indices

        selected_h = h.index_select(0, selected)
        selected_scores = scores.index_select(0, selected).unsqueeze(-1)

        # Apply the block and update selected token states.
        updated = selected_h + selected_scores * block(selected_h)
        h.index_copy_(0, selected, updated)

        # Increment recursion depths.
        depths[selected] += 1

        # Only selected tokens remain active for the next recursion.
        active = torch.zeros(T, dtype=torch.bool, device=device)
        active[selected] = True

    return h, depths