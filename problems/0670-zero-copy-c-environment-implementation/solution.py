import numpy as np

def zero_copy_env_simulation(
    num_envs: int,
    obs_dim: int,
    step_data: list,
    commands: list
) -> dict:
    """
    Simulate a vectorized C environment with zero-copy buffer management.
    """

    # Pre-allocated internal buffers. Never reallocate them.
    obs_buffer = np.zeros((num_envs, obs_dim), dtype=np.float64)
    reward_buffer = np.zeros(num_envs, dtype=np.float64)
    done_buffer = np.zeros(num_envs, dtype=np.int8)

    snapshots = {}
    reads = []
    alias_checks = []

    total_copies = 0
    total_views = 0
    total_bytes_saved = 0

    step_index = 0

    for command in commands:
        op = command[0]

        if op == "step":
            if step_index >= len(step_data):
                raise IndexError("No more step_data entries available")

            obs_data, rewards_data, dones_data = step_data[step_index]

            # In-place writes into the pre-allocated buffers.
            obs_buffer[...] = np.asarray(obs_data, dtype=np.float64)
            reward_buffer[...] = np.asarray(rewards_data, dtype=np.float64)
            done_buffer[...] = np.asarray(dones_data, dtype=np.int8)

            step_index += 1

        elif op == "store_view":
            name = command[1]

            # Slice creates a view while preserving the same buffer memory.
            snapshots[name] = obs_buffer.view()

            total_views += 1
            total_bytes_saved += num_envs * obs_dim * 8

        elif op == "store_copy":
            name = command[1]

            # Explicit independent copy.
            snapshots[name] = obs_buffer.copy()

            total_copies += 1

        elif op == "read":
            name = command[1]

            # Convert current snapshot contents to an independent nested list.
            reads.append(
                np.round(snapshots[name], 4).tolist()
            )

        elif op == "read_buffer":
            reads.append(
                np.round(obs_buffer, 4).tolist()
            )

        elif op == "check_alias":
            name = command[1]

            alias_checks.append(
                bool(np.shares_memory(snapshots[name], obs_buffer))
            )

        elif op == "auto_reset":
            # Reset only environments whose done flag is currently set.
            done_mask = done_buffer.astype(bool)

            obs_buffer[done_mask] = 0.0
            reward_buffer[done_mask] = 0.0

        else:
            raise ValueError(f"Unknown command: {op}")

    return {
        "reads": reads,
        "alias_checks": alias_checks,
        "total_copies": int(total_copies),
        "total_views": int(total_views),
        "total_bytes_saved": int(total_bytes_saved),
        "buffer_reallocs": 0
    }