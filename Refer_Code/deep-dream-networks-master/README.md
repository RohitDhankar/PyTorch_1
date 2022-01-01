# Deep Dream Networks 

### What is this?
- A deep learning library that implements the basic building blocks of neural networks
- Built for being easy to learn from
- Support for dense, convolutional, maxpooling and softmax layers
- Support for building a sequential model of layers of different types and sizes
- Support for mini batches
- Support for activations like sigmoid, tanh and relu
- Not performant for production since its written ground up in Python

### How to use

Built to follow same easy to use style as Keras.io:

````
# import model
from NeuralNetworks.Layers import Dense
from NeuralNetworks.Model import Sequential
from NeuralNetworks.Layers import Softmax
from NeuralNetworks.Datasets import mnist_flat

# load data
(x_train, y_train), (x_test, y_test) = mnist_flat.load_data()

# create model
model = Sequential()
model.add(Dense(200, input_dim=784))
model.add(Dense(10, activation='sigmoid'))
model.compile(lr=0.1, loss='mse')

# fit model
model.fit(x_train, y_train, batch_size = 1, epochs = 5, verbose = 1, test_accuracy=(x_test, y_test))

"""
Epoch: 0: 100%|████████████████████████████████████| 60000.0/60000 [00:23<00:00, 2548.14it/s, loss=0.0121, test_accuracy=94.86%]
Epoch: 1: 100%|████████████████████████████████████| 60000.0/60000 [00:24<00:00, 2441.56it/s, loss=0.0061, test_accuracy=96.36%]
Epoch: 2: 100%|████████████████████████████████████| 60000.0/60000 [00:23<00:00, 2502.89it/s, loss=0.0049, test_accuracy=96.46%]
Epoch: 3: 100%|████████████████████████████████████| 60000.0/60000 [00:24<00:00, 2479.33it/s, loss=0.0042, test_accuracy=96.81%]
Epoch: 4: 100%|████████████████████████████████████| 60000.0/60000 [00:24<00:00, 2470.29it/s, loss=0.0038, test_accuracy=97.00%]
"""
````
### To learn from

Code is written so you can more easy understand the inner workings of neural networks:

`````
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

