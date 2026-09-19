from tinygrad import Tensor

class MyTransform:
    def __call__(self, x: Tensor) -> Tensor:
        """
        x: tinygrad Tensor of shape (1, 28, 28), float, in [0,1].
        Return: transformed Tensor, same shape and dtype.
        """
        # Mild contrast reduction + brightness shift.
        # Output remains in [0.05, 0.95] for inputs in [0, 1].
        return x * 0.9 + 0.05