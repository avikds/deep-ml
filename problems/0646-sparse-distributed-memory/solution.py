import numpy as np

class SparseDistributedMemory:
    def __init__(self, address_length, word_length, num_hard_locations, activation_radius, seed=42):
        self.address_length = address_length
        self.word_length = word_length
        self.num_hard_locations = num_hard_locations
        self.activation_radius = activation_radius

        rng = np.random.RandomState(seed)

        # Random binary hard-location addresses.
        self.hard_locations = rng.randint(
            0, 2, size=(num_hard_locations, address_length)
        )

        # Integer counters for each hard location and word bit.
        self.counters = np.zeros(
            (num_hard_locations, word_length),
            dtype=int
        )

    def _activated_locations(self, address):
        address = np.asarray(address, dtype=int)

        # Hamming distance = number of differing bits.
        distances = np.sum(self.hard_locations != address, axis=1)

        return distances <= self.activation_radius

    def write(self, address, word):
        word = np.asarray(word, dtype=int)

        active = self._activated_locations(address)
        n_active = int(np.sum(active))

        if n_active > 0:
            # 1 -> +1, 0 -> -1
            update = 2 * word - 1
            self.counters[active] += update

        return n_active

    def read(self, address):
        active = self._activated_locations(address)

        if not np.any(active):
            return [0] * self.word_length

        summed = np.sum(self.counters[active], axis=0)

        # Non-negative -> 1, negative -> 0.
        return (summed >= 0).astype(int).tolist()