import numpy as np
from multiprocessing import Pool

def simulate_episode(args: tuple) -> tuple:
    """
    Worker function for simulating a single episode in a tabular MDP.
    """
    (
        worker_id,
        seed,
        max_steps,
        trans_probs,
        rewards,
        terminal_states,
        policy,
        gamma,
        start_state
    ) = args

    rng = np.random.RandomState(seed)

    terminal_states = set(terminal_states)

    state = int(start_state)
    total_return = 0.0
    discount = 1.0
    episode_length = 0

    for _ in range(max_steps):
        # Stop before taking an action from a terminal state.
        if state in terminal_states:
            break

        # Select action according to the policy.
        action = int(
            rng.choice(
                len(policy[state]),
                p=policy[state]
            )
        )

        # Collect immediate reward.
        reward = float(rewards[state, action])

        # Transition to the next state according to the model.
        next_state = int(
            rng.choice(
                len(trans_probs[state, action]),
                p=trans_probs[state, action]
            )
        )

        total_return += discount * reward
        discount *= gamma
        episode_length += 1

        state = next_state

    return worker_id, float(total_return), int(episode_length)


def parallel_env_simulate(
    num_workers: int,
    max_steps: int,
    trans_probs: np.ndarray,
    rewards: np.ndarray,
    terminal_states: list,
    policy: np.ndarray,
    gamma: float,
    start_state: int,
    seeds: list
) -> dict:
    """
    Run parallel environment simulations using multiprocessing.
    """
    args = [
        (
            worker_id,
            seeds[worker_id],
            max_steps,
            trans_probs,
            rewards,
            terminal_states,
            policy,
            gamma,
            start_state
        )
        for worker_id in range(num_workers)
    ]

    with Pool(processes=num_workers) as pool:
        results = pool.map(simulate_episode, args)

    # Ensure deterministic worker ordering.
    results.sort(key=lambda x: x[0])

    returns = [result[1] for result in results]
    lengths = [result[2] for result in results]

    return {
        "mean_return": round(float(np.mean(returns)), 4),
        "std_return": round(float(np.std(returns)), 4),
        "returns": [round(float(x), 4) for x in returns],
        "lengths": [int(x) for x in lengths],
        "total_steps": int(np.sum(lengths))
    }