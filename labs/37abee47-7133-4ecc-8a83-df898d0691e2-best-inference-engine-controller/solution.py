import numpy as np


class Controller:
    def __init__(self, engine):
        self.engine = engine
        self.max_running = int(engine.get("max_running", 256))
        self.max_spec_k = int(engine.get("max_spec_k", 8))
        self.max_prefill_tokens = int(engine.get("max_prefill_tokens", 32768))
        self.acceptance = 0.65

    def step(self, state):
        e = self.engine
        now = float(state["t"])
        waiting = state["waiting"]
        running = state["running"]
        free = max(0.0, float(state["kv_free_bytes"]))

        n_wait = len(waiting)
        n_run = len(running)
        slots = max(0, self.max_running - n_run)

        dec = running[running["status"] == 3] if n_run else running
        nd = len(dec)

        # Estimate acceptance from observed speculative decoding outcomes.
        if nd:
            drafted = dec["drafted"].astype(np.float64)
            accepted = dec["accepted"].astype(np.float64)
            total = float(drafted.sum())

            if total > 0:
                observed = float(accepted.sum()) / total
                self.acceptance = (
                    0.85 * self.acceptance
                    + 0.15 * min(0.95, max(0.05, observed))
                )

        # Estimate the KV cost of decoding requests.
        fp16_bpt = float(e["kv_bytes_per_token_fp16"])
        if nd:
            lengths = (
                dec["prompt_len"].astype(np.float64)
                + dec["generated"].astype(np.float64)
            )
            bpt = fp16_bpt * np.where(dec["kv_fp8"].astype(bool), 0.5, 1.0)
            total_kv = float(np.dot(lengths, bpt))
            kv_per_token = float(bpt.sum())
            cached_tokens = float(lengths.sum())

            rem_output = np.maximum(
                0.0, dec["max_new"].astype(np.float64)
                - dec["generated"].astype(np.float64)
            )

            ps = np.full(nd, self.acceptance, dtype=np.float64)
            valid = drafted > 0
            ps[valid] = (
                accepted[valid] + 8.0 * self.acceptance
            ) / (drafted[valid] + 8.0)
            ps = np.clip(ps, 0.05, 0.95)

            itl_slo = np.maximum(
                0.001, dec["itl_slo"].astype(np.float64)
            )
        else:
            lengths = np.empty(0)
            bpt = np.empty(0)
            total_kv = 0.0
            kv_per_token = 0.0
            cached_tokens = 0.0
            rem_output = np.empty(0)
            ps = np.empty(0)
            itl_slo = np.empty(0)

        bw = max(float(e["hbm_bandwidth"]), 1.0)
        peak = max(float(e["fp8_flops"]), 1.0)
        fpt = float(e["flops_per_token"])
        attn = float(e["attn_flops_per_pair"])
        overhead = float(e["step_overhead_ms"])
        weight_bytes = float(e["weight_bytes"])

        # Predict the emission count for each possible spec_k.
        def expected_emissions(p, k):
            result = np.ones_like(p)
            power = p.copy()
            for _ in range(k):
                result += power
                power *= p
            return result

        # Reserve space for the next verification and a small safety margin
        # before allowing speculative decoding at a particular length.
        best_k = 0
        best_score = -1.0

        max_k = self.max_spec_k if nd else 0
        for k in range(max_k + 1):
            if nd:
                needed = float(
                    np.sum(np.minimum(rem_output, k + 1) * bpt)
                )
                if needed > free:
                    continue

                emissions = expected_emissions(ps, k)
                target_flops = (
                    nd * (k + 1) * fpt
                    + attn * (k + 1) * cached_tokens
                )
                target_ms = overhead + 1000.0 * max(
                    (weight_bytes + total_kv) / bw,
                    target_flops / peak,
                )

                if k:
                    draft_ms = k * (
                        float(e["draft_launch_ms"])
                        + 1000.0 * max(
                            float(e["draft_weight_bytes"]) / bw,
                            nd * float(e["draft_flops_per_token"]) / peak,
                        )
                    )
                    target_ms += draft_ms

                # Use an approximate lower-tail emission count to limit ITL.
                per_token_ms = target_ms / 1000.0 / emissions
                threshold = np.quantile(itl_slo, 0.10)
                violation = float(np.mean(per_token_ms > itl_slo))

                score = float(emissions.sum()) / max(target_ms, 1e-6)
                if violation > 0.10:
                    score *= 0.2
                if float(np.quantile(per_token_ms, 0.90)) > threshold:
                    score *= 0.75
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_k = k

        spec_k = best_k

        # Keep enough unused KV capacity for active decoders to grow.
        running_guard = 0.0
        if nd:
            growth = np.minimum(rem_output, spec_k + 5)
            running_guard = float(np.sum(growth * bpt))

        # Build a compact list of waiting requests, ranked by estimated
        # urgency (deadline slack), then domain and prompt length.
        candidates = []
        if n_wait and slots:
            ids = waiting["id"]
            arrivals = waiting["arrival"].astype(np.float64)
            prompts = waiting["prompt_len"].astype(np.int64)
            generated = waiting["generated"].astype(np.int64)
            max_new = waiting["max_new"].astype(np.int64)
            slos = waiting["ttft_slo"].astype(np.float64)
            domains = waiting["domain"].astype(np.int32)
            strict = waiting["strict"].astype(bool)

            # A simple service-time forecast for prioritization.
            estimated = (
                0.015
                + prompts * (fpt / peak)
                + np.maximum(1, np.ceil(prompts / 8192.0))
                * weight_bytes / bw
            )
            slack = slos - (now - arrivals) - estimated

            order = np.lexsort((arrivals, prompts, domains, slack))
            candidates = [
                (
                    int(i),
                    int(ids[i]),
                    int(prompts[i]),
                    int(generated[i]),
                    int(max_new[i]),
                    bool(strict[i]),
                    float(arrivals[i]),
                    float(slos[i]),
                    int(domains[i]),
                    float(slack[i]),
                )
                for i in order
            ]

        # Admission is deliberately conservative: reserve several output
        # tokens beyond prompt plus existing output, reducing later preemptions.
        available = max(0.0, free - running_guard)
        admissions = []
        admitted_ids = []
        used = 0.0

        for row in candidates:
            if len(admissions) >= slots:
                break

            _, rid, prompt, generated, max_new, strict, arrival, slo, domain, slack = row

            use_fp8 = not strict
            token_bytes = fp16_bpt * (0.5 if use_fp8 else 1.0)

            # Keep a small per-request growth reserve; cap it to avoid
            # rejecting useful requests with long output limits.
            future = min(8, max(0, max_new - generated))
            need = (prompt + generated + future) * token_bytes

            if used + need > available:
                continue

            admissions.append((rid, "fp8") if use_fp8 else rid)
            admitted_ids.append(rid)
            used += need

        # Determine the prefill order using urgent deadlines and short prompts.
        prefills = []

        if n_run:
            for r in running:
                if int(r["status"]) == 2:
                    arrival = float(r["arrival"])
                    prompt_len = int(r["prompt_len"])
                    prefilled = int(r["prefilled"])
                    slo = float(r["ttft_slo"])
                    domain = int(r["domain"])
                    slack = slo - (now - arrival) - max(
                        0.0, prompt_len - prefilled
                    ) * fpt / peak

                    prefills.append((
                        slack, domain, prompt_len - prefilled, arrival,
                        int(r["id"])
                    ))

        for row in candidates:
            _, rid, prompt, generated, max_new, strict, arrival, slo, domain, slack = row
            if rid in admitted_ids:
                prefills.append((
                    slack, domain, prompt, arrival, rid
                ))

        prefills.sort()
        prefill_first = [row[4] for row in prefills]

        # With no decoders, larger chunks improve TTFT and weight utilization.
        # With decoders, size the chunk to leave time for their next emission.
        if nd == 0:
            prefill_budget = min(self.max_prefill_tokens, 16384)
        else:
            base_flops = (
                nd * (spec_k + 1) * fpt
                + attn * (spec_k + 1) * cached_tokens
            )
            base_ms = overhead + 1000.0 * max(
                (weight_bytes + total_kv) / bw,
                base_flops / peak,
            )

            if spec_k:
                base_ms += spec_k * (
                    float(e["draft_launch_ms"])
                    + 1000.0 * max(
                        float(e["draft_weight_bytes"]) / bw,
                        nd * float(e["draft_flops_per_token"]) / peak,
                    )
                )

            emissions = expected_emissions(ps, spec_k)
            budgets = itl_slo * emissions * 1000.0 * 0.85
            target_ms = float(np.quantile(budgets, 0.10))

            # Approximate the added work of a chunk to stay within the
            # lower-tail ITL budget while still making prefills progress.
            def predicted_chunk_ms(c):
                if prefills:
                    first_id = prefills[0][4]
                    offset = 0
                    for r in running:
                        if int(r["id"]) == first_id:
                            offset = int(r["prefilled"])
                            break
                else:
                    offset = 0

                chunk_flops = (
                    c * fpt + attn * c * (offset + c / 2.0)
                )
                return overhead + 1000.0 * max(
                    (weight_bytes + total_kv) / bw,
                    (base_flops + chunk_flops) / peak,
                )

            lo, hi = 0, min(8192, self.max_prefill_tokens)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if predicted_chunk_ms(mid) <= target_ms:
                    lo = mid
                else:
                    hi = mid - 1

            prefill_budget = lo
            if prefill_budget == 0 and prefills:
                prefill_budget = min(128, self.max_prefill_tokens)

        return {
            "admit": admissions,
            "prefill_tokens": int(prefill_budget),
            "prefill_first": prefill_first,
            "spec_k": int(spec_k),
        }