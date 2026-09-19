import numpy as np

def disaggregated_serving_sim(
    requests,
    num_prefill,
    num_decode,
    prefill_rate,
    decode_rate,
    kv_transfer_rate
):
    """
    Simulate disaggregated prefill-decode LLM serving.
    """

    if not requests:
        return {
            "avg_ttft": 0.0,
            "avg_total_latency": 0.0,
            "throughput": 0.0,
            "prefill_utilization": 0.0,
            "decode_utilization": 0.0
        }

    if (
        num_prefill <= 0
        or num_decode <= 0
        or prefill_rate <= 0
        or decode_rate <= 0
        or kv_transfer_rate < 0
    ):
        raise ValueError("Invalid simulator parameters")

    # Process requests in arrival-time order.
    reqs = sorted(
        requests,
        key=lambda r: r["arrival_time"]
    )

    prefill_available = np.zeros(num_prefill, dtype=float)
    decode_available = np.zeros(num_decode, dtype=float)

    total_prefill_busy = 0.0
    total_decode_busy = 0.0

    ttfts = []
    latencies = []
    completion_times = []

    for r in reqs:
        arrival = float(r["arrival_time"])
        prompt_tokens = int(r["prompt_tokens"])
        output_tokens = int(r["output_tokens"])

        # ------------------------------------------------------------
        # Prefill assignment: earliest available GPU, lowest index on tie
        # ------------------------------------------------------------
        p_idx = int(np.argmin(prefill_available))

        prefill_start = max(
            arrival,
            prefill_available[p_idx]
        )

        prefill_duration = prompt_tokens / prefill_rate
        prefill_finish = prefill_start + prefill_duration

        prefill_available[p_idx] = prefill_finish
        total_prefill_busy += prefill_duration

        # ------------------------------------------------------------
        # KV transfer
        # ------------------------------------------------------------
        kv_duration = prompt_tokens * kv_transfer_rate
        transfer_finish = prefill_finish + kv_duration

        # ------------------------------------------------------------
        # Decode assignment: earliest available GPU, lowest index on tie
        # ------------------------------------------------------------
        d_idx = int(np.argmin(decode_available))

        decode_start = max(
            transfer_finish,
            decode_available[d_idx]
        )

        decode_duration = output_tokens / decode_rate
        decode_finish = decode_start + decode_duration

        decode_available[d_idx] = decode_finish
        total_decode_busy += decode_duration

        # Metrics
        ttfts.append(decode_start - arrival)
        latencies.append(decode_finish - arrival)
        completion_times.append(decode_finish)

    makespan = max(completion_times) - min(
        float(r["arrival_time"]) for r in reqs
    )

    total_output_tokens = sum(
        int(r["output_tokens"]) for r in reqs
    )

    avg_ttft = float(np.mean(ttfts))
    avg_total_latency = float(np.mean(latencies))

    if makespan > 0:
        throughput = total_output_tokens / makespan
        prefill_utilization = (
            total_prefill_busy
            / (num_prefill * makespan)
        )
        decode_utilization = (
            total_decode_busy
            / (num_decode * makespan)
        )
    else:
        throughput = 0.0
        prefill_utilization = 0.0
        decode_utilization = 0.0

    return {
        "avg_ttft": round(avg_ttft, 4),
        "avg_total_latency": round(avg_total_latency, 4),
        "throughput": round(throughput, 4),
        "prefill_utilization": round(prefill_utilization, 4),
        "decode_utilization": round(decode_utilization, 4)
    }