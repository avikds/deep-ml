import numpy as np

def async_env_pool(
    num_envs: int,
    step_durations: list,
    rewards_per_step: list,
    total_ticks: int,
    batch_size: int
) -> tuple:
    """
    Simulate an asynchronous environment pool.
    """
    if total_ticks == 0:
        return (0, 0.0, 0.0, 0.0)

    # remaining[i] == 0 means environment i is ready.
    remaining = np.zeros(num_envs, dtype=int)

    total_steps = 0
    total_reward = 0.0
    active_counts = []

    for _ in range(total_ticks):
        # 1. Advance all busy environments.
        busy = remaining > 0
        remaining[busy] -= 1

        # Environments that reach zero have completed their step.
        completed = np.where(busy & (remaining == 0))[0]

        for env in completed:
            total_steps += 1
            total_reward += float(rewards_per_step[env])

        # 2. Dispatch ready environments in ascending index order.
        ready = np.where(remaining == 0)[0]
        dispatch = ready[:batch_size]

        for env in dispatch:
            remaining[env] = int(step_durations[env])

        # 3. Record busy environments after dispatch.
        active_counts.append(int(np.sum(remaining > 0)))

    avg_envs_active = float(np.mean(active_counts))
    throughput = total_steps / total_ticks

    return (
        int(total_steps),
        round(total_reward, 4),
        round(avg_envs_active, 4),
        round(throughput, 4)
    )