import numpy as np

def reshape_matrix(a: list[list[int | float]], new_shape: tuple[int, int]) -> list[list[int | float]]:
    try:
        arr = np.array(a)
        reshaped_matrix = arr.reshape(new_shape)
        return reshaped_matrix.tolist()
    except ValueError:
        return []