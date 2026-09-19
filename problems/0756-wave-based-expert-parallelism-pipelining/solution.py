def wave_pipeline_makespan(transfer_times, compute_times, send_times):
    """
    Compute the makespan of the three-stage wave pipeline.
    """
    n = len(transfer_times)

    if n == 0:
        return 0

    transfer_end = 0
    compute_end = 0
    send_end = 0

    for i in range(n):
        # Transfer stage: serialized across waves.
        transfer_start = transfer_end
        transfer_end = transfer_start + transfer_times[i]

        # Compute waits for both its transfer and the previous compute.
        compute_start = max(transfer_end, compute_end)
        compute_end = compute_start + compute_times[i]

        # Send waits for both its compute and the previous send.
        send_start = max(compute_end, send_end)
        send_end = send_start + send_times[i]

    return send_end