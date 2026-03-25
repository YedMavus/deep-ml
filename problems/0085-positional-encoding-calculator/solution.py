import numpy as np

def pos_encoding(position: int, d_model: int):
	# Your code here
	pos = np.arange(position)[:, np.newaxis]
	i = np.arange(d_model)[np.newaxis, :]

	angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
	angle_rads = pos * angle_rates

	pos_encoding = np.zeros((position, d_model))
	pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
	pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])  

	pos_encoding = np.float16(pos_encoding)
	return pos_encoding