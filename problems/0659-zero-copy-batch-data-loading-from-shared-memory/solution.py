import numpy as np

class ZeroCopyBatchLoader:
    def __init__(self, data: np.ndarray, batch_size: int):
        """Store data in a flat contiguous buffer simulating shared memory."""
        data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError("data must be a 2D array")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.n_samples, self.n_features = data.shape
        self.batch_size = int(batch_size)

        # Own a contiguous 1D buffer.
        self._buffer = np.array(data, copy=True, order="C").reshape(-1)

    def num_batches(self) -> int:
        """Return total number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def get_batch(self, batch_idx: int) -> np.ndarray:
        """Return batch as a zero-copy view into the buffer."""
        if not isinstance(batch_idx, (int, np.integer)):
            raise TypeError("batch_idx must be an integer")

        if batch_idx < 0 or batch_idx >= self.num_batches():
            raise IndexError("batch index out of range")

        start_row = batch_idx * self.batch_size
        end_row = min(start_row + self.batch_size, self.n_samples)

        start = start_row * self.n_features
        end = end_row * self.n_features

        return self._buffer[start:end].reshape(
            end_row - start_row, self.n_features
        )

    def is_zero_copy(self, batch_idx: int) -> bool:
        """Check whether the batch shares memory with the buffer."""
        batch = self.get_batch(batch_idx)
        return bool(np.shares_memory(batch, self._buffer))

    def get_batch_means(self) -> list:
        """Return per-batch mean values rounded to 4 decimals."""
        means = []
        for i in range(self.num_batches()):
            batch = self.get_batch(i)
            means.append(round(float(np.mean(batch)), 4))
        return means

    def write_to_buffer(self, row: int, col: int, value: float) -> None:
        """Write a value directly into the flat buffer at (row, col)."""
        if row < 0 or row >= self.n_samples:
            raise IndexError("row index out of range")
        if col < 0 or col >= self.n_features:
            raise IndexError("column index out of range")

        self._buffer[row * self.n_features + col] = value