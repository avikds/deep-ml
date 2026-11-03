import numpy as np

class Agent:
    def __init__(self, n_problems, k, n_steps):
        self.n = n_problems
        self.k = k
        self.Q = np.zeros((n_problems, k), dtype=np.float64)
        self.alpha = 0.1
        self.epsilon = 0.1

    def act(self):
        # Randomly break ties between equally valued arms.
        noise = np.random.random((self.n, self.k)) * 1e-10
        greedy = np.argmax(self.Q + noise, axis=1)

        explore = np.random.random(self.n) < self.epsilon
        random_actions = np.random.randint(0, self.k, size=self.n)

        return np.where(explore, random_actions, greedy).astype(np.int64)

    def update(self, actions, rewards):
        rows = np.arange(self.n)
        errors = rewards - self.Q[rows, actions]
        self.Q[rows, actions] += self.alpha * errors