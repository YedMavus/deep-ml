import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	if padding>0:
		input_matrix = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode = 'constant')
	
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape
	out_height = (input_height - kernel_height) // stride + 1
	out_width = (input_width - kernel_width) // stride + 1
	output_matrix = np.zeros((out_height, out_width))
	for i in range(out_height):
			for j in range(out_width):
				h_start = i * stride
				w_start = j * stride

				region = input_matrix[h_start:h_start+kernel_height, w_start:w_start+kernel_width]
				output_matrix[i, j] = np.sum(region * kernel)

	# Your code here
    
	return output_matrix
