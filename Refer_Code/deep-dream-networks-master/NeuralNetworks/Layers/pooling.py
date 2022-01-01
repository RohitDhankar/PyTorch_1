
## ## Code Source >>  https://github.com/hiandersson/deep-dream-networks/blob/master/NeuralNetworks/Layers/pooling.py
## ## Code Source -- CREDITS >> hiandersson


import numpy
from NeuralNetworks.Layers.activations import lambda_from_function

class MaxPooling:

	def init(self, previous_layer):

		self.previous_layer = previous_layer

		height, width, num_filters = previous_layer.output_shape

		self.output_shape = (height // 2, width // 2, num_filters)

	def iterate_regions(self, image):

		height, width, _ = image.shape
		new_height = height // 2
		new_width = width // 2

		for y in range(new_height):
			for x in range(new_width):
				image_region = image[(y * 2):(y * 2 + 2), (x * 2):(x * 2 + 2)]
				yield image_region, y, x

	def forward(self, input):

		self.last_input = input
		
		height, width, num_filters = input.shape
		output = numpy.zeros((height // 2, width // 2, num_filters))

		for image_region, y, x in self.iterate_regions(input):
			output[y, x] = numpy.amax(image_region, axis=(0, 1))

		assert(self.output_shape == output.shape)

		self.layer_output = output
		return self.layer_output

	def backward(self, learning_rate, error_gradient_in, previous_layer_output):

		error_gradient_out = numpy.zeros(self.last_input.shape)

		for image_region, y, x in self.iterate_regions(self.last_input):
			height, width, image_filter = image_region.shape
			amax = numpy.amax(image_region, axis=(0, 1))

			# If this pixel was the max value, copy the gradient to it.
			for y2 in range(height):
				for x2 in range(width):
					for filter2 in range(image_filter):
						if image_region[y2, x2, filter2] == amax[filter2]:
							error_gradient_out[y * 2 + y2, x * 2 + x2, filter2] = error_gradient_in[y, x, filter2]

		return error_gradient_out

