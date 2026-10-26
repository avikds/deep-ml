import collections


class Router:
    def __init__(self, n_replicas, block_size, g1_blocks, g2_blocks):
        self.n = int(n_replicas)
        self.block_size = int(block_size)
        self.g1_capacity = int(g1_blocks)
        self.g2_capacity = int(g2_blocks)

        self.g1 = [collections.OrderedDict() for _ in range(self.n)]
        self.g2 = [collections.OrderedDict() for _ in range(self.n)]

        self.requests = {}
        self.session_replica = {}
        self.miss_ratio = [0.65] * self.n
        self.completed = set()

    def _snapshot(self, request):
        # Request is an object, not a subscriptable dictionary.
        return {
            "id": request.id,
            "arrival": float(request.arrival),
            "session": request.session,
            "prompt_tokens": int(request.prompt_tokens),
            "block_hashes": tuple(request.block_hashes),
        }

    def _cache_cost(self, req, replica):
        g1 = self.g1[replica]
        g2 = self.g2[replica]

        g1_hits = 0
        g2_hits = 0

        # Cache lookup stops at the first missing block.
        for h in req["block_hashes"]:
            if h in g1:
                g1_hits += 1
            elif h in g2:
                g2_hits += 1
            else:
                break

        cached_tokens = min(
            req["prompt_tokens"],
            (g1_hits + g2_hits) * self.block_size
        )
        uncached_tokens = max(0, req["prompt_tokens"] - cached_tokens)
        g2_tokens = g2_hits * self.block_size

        service = (
            0.015
            + uncached_tokens / 12000.0
            + g2_tokens * 6.5e-6
        )

        return service, cached_tokens

    def _queue_delay(self, info, replica):
        queued = max(0, int(info["queued_requests"]))
        queued_tokens = max(0, int(info["queued_prompt_tokens"]))

        # Estimate queued work using the observed uncached fraction.
        miss = self.miss_ratio[replica]
        return queued * 0.015 + queued_tokens * miss / 12000.0

    def _remember_blocks(self, replica, hashes):
        g1 = self.g1[replica]
        g2 = self.g2[replica]

        for h in hashes:
            # Promote the block to the most-recently-used position in G1.
            g1.pop(h, None)
            g2.pop(h, None)

            if self.g1_capacity <= 0:
                if self.g2_capacity > 0:
                    g2[h] = None
                    while len(g2) > self.g2_capacity:
                        g2.popitem(last=False)
                continue

            g1[h] = None

            if len(g1) > self.g1_capacity:
                evicted, _ = g1.popitem(last=False)

                if self.g2_capacity > 0:
                    g2.pop(evicted, None)
                    g2[evicted] = None

                    while len(g2) > self.g2_capacity:
                        g2.popitem(last=False)

    def route(self, request, replicas):
        req = self._snapshot(request)
        self.requests[req["id"]] = req

        now = req["arrival"]
        session = req["session"]
        previous = self.session_replica.get(session)

        scores = []

        for i, info in enumerate(replicas):
            queue_delay = self._queue_delay(info, i)
            service, _ = self._cache_cost(req, i)
            scores.append(queue_delay + service)

        best = min(range(self.n), key=lambda i: scores[i])

        # Prefer session locality when predicted delays are nearly equal.
        if previous is not None and scores[previous] <= scores[best] + 0.03:
            best = previous

        self.session_replica[session] = best
        return int(best)

    def on_first_token(self, request_id, replica, cached_tokens, ttft_seconds):
        if request_id in self.completed:
            return

        req = self.requests.get(request_id)
        if req is None:
            return

        replica = int(replica)
        if not 0 <= replica < self.n:
            return

        prompt_len = max(1, req["prompt_tokens"])
        observed_miss = max(
            0.0,
            min(1.0, 1.0 - float(cached_tokens) / prompt_len)
        )

        self.miss_ratio[replica] = (
            0.8 * self.miss_ratio[replica] + 0.2 * observed_miss
        )

        self._remember_blocks(replica, req["block_hashes"])
        self.completed.add(request_id)