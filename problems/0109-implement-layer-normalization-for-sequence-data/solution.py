import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	# Your code here
	eps = 1e-10
	mean = np.mean(X, axis=-1, keepdims=True)
	var = np.var(X, axis=-1, keepdims= True)
	return gamma*(X-mean)/np.sqrt(var+eps) + beta