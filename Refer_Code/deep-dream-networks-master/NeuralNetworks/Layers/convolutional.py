import numpy
from NeuralNetworks.Layers.activations import lambda_from_function

class Convolutional:
    
	def __init__(self, num_filters = 1, input_shape = (1, 1)):

		# set number of nodes
		self.num_filters = num_filters

		# store input shape
		self.input_shape = input_shape

	def init(self, previous_layer):

		self.previous_layer = previous_layer
			
		self.filters = numpy.random.randn(self.num_filters, 3, 3) / 9

		height, width = self.input_shape
		self.output_shape = (height - 2, width - 2, self.num_filters)

	def iterate_regions(self, image):

		height, width = image.shape

		for y in range(height - 2):
			for x in range(width - 2):
				image_region = image[y:(y + 3), x:(x + 3)]
				yield image_region, y, x

	def forward(self, input):

		self.last_input = input

		height, width = input.shape
		output = numpy.zeros((height - 2, width - 2, self.num_filters))

		for image_region, y, x in self.iterate_regions(input):
			output[y, x] = numpy.sum(image_region * self.filters, axis=(1, 2))

		self.layer_output = output

		assert(self.output_shape == output.shape)

		return self.layer_output

	def backward(self, learning_rate, error_gradient_in, previous_layer_output):

		filter_gradient_update = numpy.zeros(self.filters.shape)

		for image_region, i, j in self.iterate_regions(self.last_input):
			for f in range(self.num_filters):
				filter_gradient_update[f] += error_gradient_in[i, j, f] * image_region

		self.filters -= learning_rate * filter_gradient_update

