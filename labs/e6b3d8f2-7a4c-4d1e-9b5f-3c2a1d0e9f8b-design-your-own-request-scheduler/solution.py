import numpy as np

class Scheduler:
    def __init__(
        self,
        max_batch,
        total_blocks,
        block_size,
        prefill_tps,
        decode_ms_base,
        decode_ms_per_seq,
    ):
        self.max_batch = int(max_batch)
        self.total_blocks = int(total_blocks)
        self.block_size = int(block_size)
        self.prefill_tps = float(prefill_tps)
        self.decode_ms_base = float(decode_ms_base)
        self.decode_ms_per_seq = float(decode_ms_per_seq)

        # Keep decode + prefill steps comfortably below the 150 ms ITL limit.
        self.target_step_s = 0.13

    def plan(self, t, waiting, running, free_blocks):
        """Latency-aware admission, short/urgent-first prefill, limited preemption."""
        t = float(t)
        free_blocks = int(free_blocks)

        decode_count = sum(r["phase"] == "decode" for r in running)
        room = max(0, self.max_batch - len(running))

        # Choose the largest prefill budget that keeps the combined step
        # comfortably below the ITL target.
        decode_s = (
            self.decode_ms_base
            + self.decode_ms_per_seq * decode_count
        ) / 1000.0

        budget = int(max(
            256,
            (self.target_step_s - decode_s) * self.prefill_tps
        ))
        budget = max(256, min(budget, 1800))

        if not waiting or room == 0:
            return [], [], budget

        # Preserve some blocks for running decoders to grow.
        reserve_blocks = min(free_blocks, decode_count)
        available_blocks = max(0, free_blocks - reserve_blocks)

        candidates = list(waiting)

        # Prioritize requests by deadline slack:
        # age + estimated prefill time determines urgency. Short prompts
        # naturally come first, while old requests eventually become urgent.
        def priority(r):
            age = max(0.0, t - float(r["arrival"]))
            est_prefill = float(r["prompt_len"]) / self.prefill_tps
            slack = 1.0 - age - est_prefill
            return (
                slack,
                int(r["prompt_len"]),
                float(r["arrival"]),
                int(r["id"]),
            )

        candidates.sort(key=priority)

        admit_ids = []
        used_blocks = 0
        slots = room

        for r in candidates:
            if slots <= 0:
                break

            need = int(r["blocks_needed"])

            if used_blocks + need <= available_blocks:
                admit_ids.append(r["id"])
                used_blocks += need
                slots -= 1

        # If the most urgent waiting request cannot fit, allow a small amount
        # of voluntary preemption. Prefer a prefilling request that has done
        # little work, or a decoder that is far from completion and holds many
        # blocks. This is deliberately conservative because preemption causes
        # the request to restart its entire prefill.
        preempt_ids = []

        if candidates:
            most_urgent = candidates[0]
            age = max(0.0, t - float(most_urgent["arrival"]))
            need = int(most_urgent["blocks_needed"])

            deficit = max(0, need - available_blocks)

            if (
                age >= 0.60
                and deficit > 0
                and room == 0
                and running
            ):
                possible = []

                for r in running:
                    phase = r["phase"]
                    blocks = int(r.get("blocks_held", 0))
                    generated = int(r.get("generated", 0))
                    max_new = max(1, int(r.get("max_new", 1)))

                    # Prefer prefill jobs or jobs with little generation
                    # progress, while freeing many blocks.
                    preempt_cost = (
                        0 if phase == "prefill" else 1,
                        generated / max_new,
                        -blocks,
                        float(r.get("arrival", 0.0)),
                        int(r["id"]),
                    )
                    possible.append((preempt_cost, r))

                possible.sort(key=lambda x: x[0])
                preempt_ids = [possible[0][1]["id"]]

        # Return admissions in the prefill order selected above.
        return admit_ids, preempt_ids, budget