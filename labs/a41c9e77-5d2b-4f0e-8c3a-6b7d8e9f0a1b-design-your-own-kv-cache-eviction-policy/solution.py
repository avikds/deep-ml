import numpy as np


class EvictionPolicy:
    def __init__(self, capacity_blocks):
        self.capacity = int(capacity_blocks)

        # Statistics are kept even for blocks that are not resident.
        # This acts like a small "ghost cache" and lets us recognize
        # blocks that have been seen before, avoiding one-shot pollution.
        self.seen = {}          # block -> total lookup count
        self.last_seen = {}     # block -> most recent request
        self.depth = {}         # block -> prompt depth
        self.gap = {}           # block -> EMA of reuse distance

        # Resident metadata.
        self.resident = {}      # block -> {count, last, depth, score_parts}

    def _priority(self, block, t=None):
        """Estimated future value of keeping a block resident."""
        freq = self.seen.get(block, 0)
        d = self.depth.get(block, 0)

        # Frequency is the strongest signal.  Shallow prefix blocks get
        # an additional advantage because they are more likely to recur
        # across conversations/prompts.
        score = 3.0 * np.log1p(freq)
        score += 2.5 / (1.0 + d)

        # Prefer blocks with short historical reuse intervals.
        g = self.gap.get(block, None)
        if g is not None:
            score += 2.0 / (1.0 + g)

            # Small imminence bonus for blocks whose next use is roughly due.
            if t is not None:
                age = max(0.0, float(t - self.last_seen.get(block, t)))
                score += 1.5 / (1.0 + abs(age - g))

        # Recency is a tie-breaker rather than the main policy.
        if t is not None:
            age = max(0.0, float(t - self.last_seen.get(block, t)))
            score += 0.75 / (1.0 + age)

        return float(score)

    def admit(self, block, depth, prompt_blocks):
        # Record the lookup even if we reject it.  This lets a repeated
        # block become eligible later.
        previous = self.seen.get(block, 0)
        self.seen[block] = previous + 1

        self.depth[block] = int(depth)

        # Already resident should normally never reach admit(), but accepting
        # it is harmless.
        if block in self.resident:
            return True

        # No room is a strong reason not to pollute the cache with a block
        # that has never been seen before.  Shallow prefix blocks are admitted
        # immediately because they have unusually high sharing potential.
        if previous == 0 and depth > 2 and len(self.resident) >= self.capacity:
            return False

        # For unseen deep blocks, admit when there is spare capacity.
        # Once a block has been observed before, it is a proven reuse candidate.
        return self.capacity > 0

    def touch(self, block, depth, prompt_blocks, t, hit):
        t = int(t)
        depth = int(depth)

        self.depth[block] = depth

        old_last = self.last_seen.get(block)
        if old_last is not None:
            gap = t - old_last
            old_gap = self.gap.get(block)
            if old_gap is None:
                self.gap[block] = float(gap)
            else:
                self.gap[block] = 0.7 * old_gap + 0.3 * gap

        self.last_seen[block] = t

        # On a hit, count another lookup.  On a miss, admit() already
        # counted this occurrence.
        if hit:
            self.seen[block] = self.seen.get(block, 0) + 1

        self.resident[block] = {
            "last": t,
            "depth": depth,
        }

    def victim(self):
        # The grader calls victim() only when the cache is full, so there
        # should be at least one resident block.  Select the least useful
        # resident block; ties are resolved deterministically.
        if not self.resident:
            raise RuntimeError("victim() called with an empty cache")

        best_block = None
        best_score = float("inf")

        for block in self.resident:
            score = self._priority(block)

            if score < best_score:
                best_score = score
                best_block = block
            elif score == best_score:
                # Deterministic fallback for hashable opaque keys.
                if repr(block) < repr(best_block):
                    best_block = block

        del self.resident[best_block]
        return best_block