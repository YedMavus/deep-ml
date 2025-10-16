import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == 'column':
        means = np.mean(matrix, axis=0)
    elif mode == 'row':
        means = np.mean(matrix, axis=1)
    else:
        means = 0
	return means