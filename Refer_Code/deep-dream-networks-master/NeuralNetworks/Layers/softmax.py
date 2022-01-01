import numpy
from NeuralNetworks.Layers.activations import lambda_from_function

class Softmax:
		
	def __init__(self, nodes):

		self.nodes = nodes

	def init(self, previous_layer):

		self.previous_layer = previous_layer

		# number of inputs is the product of the shape array from the previous layer
		num_inputs = numpy.prod(previous_layer.output_shape)

		# We divide by input_len to reduce the variance of our initial values
		self.weights = numpy.random.randn(num_inputs, self.nodes) / num_inputs
		self.biases = numpy.zeros(self.nodes)

	def forward(self, input):

		self.last_input_shape = input.shape

		input = input.flatten()
		self.last_input = input

		input_len, nodes = self.weights.shape

		linear_output = numpy.dot(input, self.weights) + self.biases
		self.saved_linear_output = linear_output

		exp = numpy.exp(linear_output)

		self.layer_output = exp / numpy.sum(exp, axis=0)
		return self.layer_output

	def backward(self, learning_rate, error_gradient_in, previous_layer_output):

		# We know only 1 element of d_L_d_out will be nonzero
		for i, gradient_in in enumerate(error_gradient_in):
			if gradient_in == 0:
				continue

			# e ^ ( I * W )
			exponent_linear_output = numpy.exp(self.saved_linear_output) 

			# Sum of all e^logits
			sum_exponent_linear_outputs = numpy.sum(exponent_linear_output)

			# Gradients of error_gradient_in[i] against linear output
			dErrorGradientIn_dLinearOutput = -exponent_linear_output[i] * exponent_linear_output / (sum_exponent_linear_outputs ** 2)
			dErrorGradientIn_dLinearOutput[i] = exponent_linear_output[i] * (sum_exponent_linear_outputs - exponent_linear_output[i]) / (sum_exponent_linear_outputs ** 2)

			# Gradients of linear output against weights/biases/input
			dLinearOutputs_dWeights = self.last_input
			dLinearOutputs_dBias = 1
			dLinearOutputs_dInput = self.weights

			# Gradients of loss against linear output
			dLoss_dLinearOutput = gradient_in * dErrorGradientIn_dLinearOutput

			# Gradients of loss against weights/biases/input
			dLoss_dWeights = dLinearOutputs_dWeights[numpy.newaxis].T @ dLoss_dLinearOutput[numpy.newaxis]
			dLoss_dBias = dLoss_dLinearOutput * dLinearOutputs_dBias
			dLoss_dInput = dLinearOutputs_dInput @ dLoss_dLinearOutput

			# Update weights / biases
			self.weights -= learning_rate * dLoss_dWeights
			self.biases -= learning_rate * dLoss_dBias

			error_gradient_out = dLoss_dInput.reshape(self.last_input_shape)

			return error_gradient_out
