def track_gpu_utilization(timestamps_ms: list, gpu_util_pct: list, phase_labels: list) -> dict:
    """
    Analyze GPU utilization during RL training and compute per-phase statistics.
    """
    durations = [
        timestamps_ms[i + 1] - timestamps_ms[i]
        for i in range(len(gpu_util_pct))
    ]

    total_time = sum(durations)

    if total_time == 0:
        avg_util = 0.0
    else:
        avg_util = sum(
            duration * util
            for duration, util in zip(durations, gpu_util_pct)
        ) / total_time

    phase_data = {}

    for i, (duration, util, phase) in enumerate(
        zip(durations, gpu_util_pct, phase_labels)
    ):
        if phase not in phase_data:
            phase_data[phase] = {
                "time": 0.0,
                "weighted_util": 0.0,
                "wasted_gpu_time": 0.0,
                "first_index": i,
            }

        data = phase_data[phase]
        data["time"] += duration
        data["weighted_util"] += duration * util
        data["wasted_gpu_time"] += duration * (100.0 - util)

    phase_stats = {}

    for phase in sorted(phase_data):
        data = phase_data[phase]
        phase_time = data["time"]

        avg_phase_util = (
            data["weighted_util"] / phase_time
            if phase_time > 0 else 0.0
        )

        time_fraction = (
            phase_time / total_time * 100.0
            if total_time > 0 else 0.0
        )

        phase_stats[phase] = {
            "total_time_ms": round(phase_time, 2),
            "avg_util_pct": round(avg_phase_util, 2),
            "time_fraction_pct": round(time_fraction, 2),
        }

    # Highest wasted GPU-time wins; ties go to the phase seen earliest.
    bottleneck_phase = None
    if phase_data:
        bottleneck_phase = max(
            phase_data,
            key=lambda phase: (
                phase_data[phase]["wasted_gpu_time"],
                -phase_data[phase]["first_index"],
            ),
        )

    return {
        "total_time_ms": round(total_time, 2),
        "avg_gpu_util_pct": round(avg_util, 2),
        "phase_stats": phase_stats,
        "bottleneck_phase": bottleneck_phase,
        "gpu_idle_fraction_pct": round(100.0 - avg_util, 2),
    }