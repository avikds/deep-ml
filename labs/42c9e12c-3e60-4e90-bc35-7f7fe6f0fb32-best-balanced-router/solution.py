import numpy as np


class Router:
    def __init__(self, n_experts, k):
        self.E = n_experts
        self.k = k
        self.saved = None

    def route(self, logits, training):
        logits = np.asarray(logits, dtype=np.float32)
        M, n, E = logits.shape

        # Average routing preferences within each micro-batch.
        scores = logits.astype(np.float64).mean(axis=1)

        # Initial independent top-k assignments.
        preferred = np.argsort(
            -scores, axis=1, kind="stable"
        )[:, :self.k]

        selected = np.zeros((M, E), dtype=bool)
        rows = np.arange(M)

        selected[rows[:, None], preferred] = True
        counts = selected.sum(axis=0).astype(np.int64)

        # Establish balanced expert capacities.
        total = M * self.k
        base, remainder = divmod(total, E)
        capacity = np.full(E, base, dtype=np.int64)

        if remainder:
            extra = np.argsort(
                -scores.sum(axis=0), kind="stable"
            )[:remainder]
            capacity[extra] += 1

        # Repair overloaded and underloaded experts.
        # Each move replaces one expert without duplicating
        # an expert within a micro-batch.
        while True:
            over = np.flatnonzero(counts > capacity)
            under = np.flatnonzero(counts < capacity)

            if len(over) == 0:
                break

            best_cost = np.inf
            best_row = -1
            best_old = -1
            best_new = -1

            for old in over:
                candidate_rows = np.flatnonzero(selected[:, old])

                for new in under:
                    available = candidate_rows[
                        ~selected[candidate_rows, new]
                    ]

                    if len(available) == 0:
                        continue

                    # Minimize the loss in routing preference.
                    costs = scores[available, old] - scores[available, new]
                    pos = np.argmin(costs)
                    cost = costs[pos]

                    if cost < best_cost:
                        best_cost = cost
                        best_row = available[pos]
                        best_old = old
                        best_new = new

            if best_row < 0:
                break

            selected[best_row, best_old] = False
            selected[best_row, best_new] = True

            counts[best_old] -= 1
            counts[best_new] += 1

        # Recover exactly k expert IDs per micro-batch.
        batch_ids = np.empty((M, self.k), dtype=np.int64)

        for i in range(M):
            chosen = np.flatnonzero(selected[i])
            chosen = chosen[
                np.argsort(-scores[i, chosen], kind="stable")
            ]
            batch_ids[i] = chosen

        # Every token in a micro-batch uses the same expert pair.
        expert_ids = np.broadcast_to(
            batch_ids[:, None, :],
            (M, n, self.k)
        ).copy()

        # Calculate per-token normalized gates.
        selected_logits = np.take_along_axis(
            logits, expert_ids, axis=2
        ).astype(np.float64)

        selected_logits -= selected_logits.max(
            axis=2, keepdims=True
        )

        exp_values = np.exp(selected_logits)
        gates = exp_values / exp_values.sum(
            axis=2, keepdims=True
        )

        gates = gates.astype(np.float32)

        if training:
            self.saved = (
                expert_ids.reshape(-1, self.k).copy(),
                gates.reshape(-1, self.k).astype(np.float64),
                logits.shape
            )

        return expert_ids, gates

    def backward(self, grad_gates):
        if self.saved is None:
            raise RuntimeError("No preceding training route")

        ids, gates, shape = self.saved

        gg = np.asarray(
            grad_gates, dtype=np.float64
        ).reshape(-1, self.k)

        # Backpropagate through the selected-expert softmax.
        weighted = np.sum(
            gg * gates, axis=1, keepdims=True
        )

        selected_grad = gates * (gg - weighted)

        grad_logits = np.zeros(
            (len(ids), self.E), dtype=np.float32
        )

        np.put_along_axis(
            grad_logits,
            ids,
            selected_grad.astype(np.float32),
            axis=1
        )

        return grad_logits.reshape(shape)