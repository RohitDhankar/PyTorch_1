import numpy
from NeuralNetworks.Layers.activations import lambda_from_function

class Dense:
    
    def __init__(self, num_nodes = 1, input_dim = None, activation = 'sigmoid'):

        # set number of nodes
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.activation = activation

        # activation and derivate functions
        self.activation_function, self.activation_gradient = lambda_from_function(activation)

    def init(self, previous_layer):

        self.previous_layer = previous_layer

        if previous_layer == None:
            input_dim = self.input_dim
        else:
            input_dim = previous_layer.num_nodes
            
        self.weights = numpy.random.normal(0.0, pow(input_dim, -0.5), (self.num_nodes, input_dim))

        self.output_shape = (self.num_nodes, 1)

    def forward(self, input):

        # calculate signals into hidden layer
        hidden_input = numpy.dot(self.weights, input)

        # calculate s emerging from hidden layer
        output = self.activation_function(hidden_input)

        assert(self.output_shape == output.shape)

        self.layer_output = output
        return self.layer_output

    def backward(self, learning_rate, error_gradient_in, previous_layer_output):

        # delta w = old d_W - alpha * (d_E / d_W) = learningrate * error_next * sigmoid(output_this) * (1 - sigmoid(output_this)) * output_previous
        self.weights += learning_rate * numpy.dot(
            (error_gradient_in * self.activation_gradient(self.layer_output)), 
            numpy.transpose(previous_layer_output))

        # propagate the gradient error to previous layer
        error_gradient_out = numpy.dot(self.weights.T, error_gradient_in)

        return error_gradient_out
