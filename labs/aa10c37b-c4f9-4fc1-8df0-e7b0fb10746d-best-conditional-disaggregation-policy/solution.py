import math

class Policy:
    def __init__(self, cluster):
        self.cluster = cluster
        self.turn = 0
        self.target = 2
        self.last_change = -1000000.0
        self.high_pressure = 0
        self.low_pressure = 0
        self.session_worker = {}

    def _prefill_time(self, new_tokens, cached_tokens):
        n = max(0.0, float(new_tokens))
        c = max(0.0, float(cached_tokens))
        return (n / 40000.0) * (
            1.0 + (c + n / 2.0) / 131072.0
        )

    def _length(self, value):
        if value is None:
            return 0
        if isinstance(value, (int, float)):
            return max(0, int(value))
        try:
            return len(value)
        except Exception:
            return 0

    def _active_workers(self, state):
        workers = state.get("workers", [])
        decode = []
        prefill = []

        for w in workers:
            if w.get("status") != "active":
                continue

            if w.get("role") == "decode":
                decode.append(w)
            elif w.get("role") == "prefill":
                prefill.append(w)

        return decode, prefill

    def rebalance(self, state):
        now = float(state.get("time", 0.0))
        qtokens = max(
            0.0,
            float(state.get("prefill_queue_new_tokens", 0.0))
        )
        qcount = self._length(state.get("prefill_queue", []))

        # Estimate recent prompt sizes if completion records expose them.
        recent = state.get("recent", [])
        if isinstance(recent, dict):
            recent = recent.get("requests", [])

        prompt_sizes = []
        if isinstance(recent, (list, tuple)):
            for item in recent:
                if isinstance(item, dict):
                    value = item.get("prompt_tokens")
                    if value is not None:
                        prompt_sizes.append(float(value))

        avg_prompt = (
            sum(prompt_sizes) / len(prompt_sizes)
            if prompt_sizes else 20000.0
        )

        # Determine desired pool size from queue pressure and prompt size.
        if qtokens > 1000000 or avg_prompt > 85000:
            desired = 5
        elif qtokens > 450000 or avg_prompt > 50000:
            desired = 4
        elif qtokens > 120000 or avg_prompt > 25000:
            desired = 3
        elif qtokens < 20000 and avg_prompt < 12000 and qcount < 3:
            desired = 1
        else:
            desired = 2

        # Require persistent pressure before changing the cluster split.
        if desired > self.target:
            self.high_pressure += 1
            self.low_pressure = 0
        elif desired < self.target:
            self.low_pressure += 1
            self.high_pressure = 0
        else:
            self.high_pressure = 0
            self.low_pressure = 0

        can_switch = now - self.last_change >= 60.0

        if can_switch:
            if desired > self.target and (
                self.high_pressure >= 2 or qtokens > 700000
            ):
                self.target = desired
                self.last_change = now
                self.high_pressure = 0

            elif desired < self.target and self.low_pressure >= 4:
                self.target = desired
                self.last_change = now
                self.low_pressure = 0

        return int(max(0, min(7, self.target)))

    def _decode_cost(self, worker, request):
        running = max(0, int(worker.get("running", 0)))
        reserved = max(0, float(worker.get("kv_reserved", 0)))
        capacity = max(
            1.0, float(worker.get("kv_capacity", 900000))
        )

        prompt = max(0, int(request["prompt_tokens"]))
        output = max(1, int(request["max_new_tokens"]))
        reserve = prompt + output

        projected_kv = reserved + reserve
        projected_running = running + 1

        # Exact iteration model from the problem statement.
        iteration = (
            0.020
            + 0.0003 * projected_running
            + 0.004 * projected_kv / 100000.0
        )

        measured = float(worker.get("iteration_ms", 0.0)) / 1000.0
        iteration = max(iteration, measured)

        # Estimate waiting caused by the worker's KV and sequence limits.
        overflow = max(0.0, projected_kv - capacity)
        queue_wait = 8.0 * overflow / capacity

        queue_wait += 0.025 * (
            self._length(worker.get("local_queue", []))
            + self._length(worker.get("awaiting_transfer", []))
        )

        # Decode SLO is 25 tokens per second, or 40 ms per iteration.
        slow_penalty = max(0.0, iteration - 0.040) * output

        load_penalty = (
            0.10 * running / 48.0
            + 0.10 * reserved / capacity
        )

        return (
            queue_wait + slow_penalty + load_penalty,
            iteration,
            running,
            reserved,
            capacity
        )

    def place(self, request, state):
        self.turn += 1

        decode_workers, prefill_workers = self._active_workers(state)

        if not decode_workers:
            # The environment should normally guarantee an active
            # decode worker; retain a safe fallback.
            for w in state.get("workers", []):
                if w.get("role") == "decode":
                    return "local", int(w["id"])

        prompt = max(0, int(request["prompt_tokens"]))
        output = max(1, int(request["max_new_tokens"]))
        slo = max(
            0.01,
            float(request.get(
                "ttft_slo_s", 0.4 + prompt / 30000.0
            ))
        )

        session = request.get("session")
        pool_count = len(prefill_workers)

        # Prefill-pool queue estimate.
        pool_queue_tokens = max(
            0.0,
            float(state.get("prefill_queue_new_tokens", 0.0))
        )
        pool_queue_count = self._length(
            state.get("prefill_queue", [])
        )

        if pool_count > 0:
            queue_per_worker = pool_queue_tokens / pool_count

            remote_wait = (
                queue_per_worker / 40000.0
                * (1.0 + queue_per_worker / 262144.0)
            )
            remote_wait += 0.025 * pool_queue_count / pool_count

            pool_cache = max(
                0, int(state.get("prefill_cached_tokens", 0))
            )
            pool_cache = min(prompt, pool_cache)

            remote_service = self._prefill_time(
                prompt - pool_cache, pool_cache
            )

            transfer = 0.005 + 1.4e-6 * prompt

            remote_ttft = (
                remote_wait + remote_service + transfer
            )
        else:
            remote_ttft = float("inf")

        best_remote_worker = None
        best_remote_cost = float("inf")

        best_local_worker = None
        best_local_cost = float("inf")

        # Evaluate each active decode worker.
        for worker in decode_workers:
            decode_penalty, iteration, running, reserved, capacity = (
                self._decode_cost(worker, request)
            )

            cached = max(
                0, int(worker.get("cached_tokens", 0))
            )
            cached = min(prompt, cached)

            local_service = self._prefill_time(
                prompt - cached, cached
            )

            local_queue_tokens = max(
                0.0,
                float(worker.get("local_queue_new_tokens", 0.0))
            )
            local_queue_count = self._length(
                worker.get("local_queue", [])
            )

            local_wait = (
                local_queue_tokens / 40000.0
                * (1.0 + local_queue_tokens / 262144.0)
            )
            local_wait += 0.025 * local_queue_count

            local_ttft = local_wait + local_service

            # Local prefill chunks slow every sequence already decoding
            # on this worker. Penalize that externality.
            local_interference = (
                local_service
                * min(running, 12)
                * 0.45
            )

            # Prefer workers with capacity and low decode latency.
            local_cost = (
                local_ttft
                + local_interference
                + decode_penalty
            )

            remote_cost = (
                remote_ttft
                + decode_penalty
            )

            # A small session-affinity advantage is appropriate only
            # when there is an actual cached prefix.
            if session is not None:
                previous = self.session_worker.get(session)
                if previous == worker["id"] and cached > 0:
                    local_cost *= 0.97

            if local_cost < best_local_cost:
                best_local_cost = local_cost
                best_local_worker = worker

            if remote_cost < best_remote_cost:
                best_remote_cost = remote_cost
                best_remote_worker = worker

        if best_remote_worker is None:
            chosen = best_local_worker
            mode = "local"
        elif best_local_worker is None:
            chosen = best_remote_worker
            mode = "remote"
        else:
            # Compare local and remote costs with a slight preference
            # for remote prefill when its delay is comparable, since
            # it protects already-running decode sequences.
            if best_local_cost < 0.88 * best_remote_cost:
                chosen = best_local_worker
                mode = "local"
            else:
                chosen = best_remote_worker
                mode = "remote"

        if chosen is None:
            # Fallback to the first active decode worker.
            chosen = decode_workers[0]
            mode = "remote" if pool_count else "local"

        if session is not None:
            self.session_worker[session] = chosen["id"]

        return mode, int(chosen["id"])