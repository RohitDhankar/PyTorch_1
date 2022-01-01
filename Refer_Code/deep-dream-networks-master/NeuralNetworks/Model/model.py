import numpy
import random
from tqdm import tqdm
from NeuralNetworks.Model.losses import loss_from_function

class Sequential:
    
    def __init__(self):

        self.layers = list()
        self.lr = 0.01
        
        pass

    def add(self, layer):

        self.layers.append(layer)

        pass

    def compile(self, lr, loss='mse'):

        # store learning rate
        self.lr = lr

        # store loss function
        self.loss = loss

        # init weight matrices and store previous layers
        previous_layer = None
        for layer in self.layers:
            layer.init(previous_layer)
            previous_layer = layer

    def back_propagate_gradients(self, error_gradients, network_input):

        # sum and divide the error gradient
        error_gradient_batch = (1 / len(error_gradients)) * numpy.sum(error_gradients, axis = 0)

        # backward and update weights
        for layer in reversed(self.layers):

            # previous layer, until input
            if layer.previous_layer == None:
                previous_layer_output = network_input
            else:
                previous_layer_output = layer.previous_layer.layer_output

            error_gradient_batch = layer.backward(self.lr, error_gradient_batch, previous_layer_output)

    def fit(self, train_x, train_y, batch_size = 1, epochs = 1, verbose = 1, test_accuracy = None):

        # Cache for updating accuracy fast
        self.test_accuracy = test_accuracy

        # go through all epochs
        for epoch in range(epochs):

            loss_array = []
            error_gradients = []

            if verbose >= 1:
                pbar = tqdm(total=len(train_x), desc = "Epoch: {}".format(epoch), ncols = 128)
                pbar_counter = 0
                update_every = len(train_x) / 100

            # go through all batches
            for (batch_x, batch_y) in zip(train_x, train_y):

                # convert into 2d arrays
                network_input = numpy.array(batch_x, ndmin=2).T
                network_target = numpy.array(batch_y, ndmin=2).T

                # feed forward all layers
                network_layer_forward = network_input
                for layer in self.layers:
                    network_layer_forward = layer.forward(network_layer_forward)
                network_predicted = network_layer_forward

                # loss from output layer
                error_gradient, loss_metric = loss_from_function(self.loss, network_target, network_predicted)
                loss_array.append(loss_metric)
 
                # In mini-batch gradient descent, the cost function (and therefore gradient) is averaged over a small number of samples, from around 10-500.
                error_gradients.append(error_gradient)
                if len(error_gradients) == batch_size:
                    self.back_propagate_gradients(error_gradients, network_input)
                    error_gradients = []

                if verbose >= 1:
                    if pbar_counter % update_every == 0:
                        loss = "{:.4f}".format(numpy.mean(loss_array))
                        pbar.set_postfix(loss=loss, test_accuracy="--.--%")
                        pbar.update(update_every)
                    pbar_counter += 1

            # propagate last batch if anything remains
            if len(error_gradients) > 0:
                self.back_propagate_gradients(error_gradients, network_input)
                error_gradients = []

            # test accuracy in the end
            if test_accuracy != None and verbose >= 1:
                accuracy = self.model_test_accuracy()
                loss = "{:.4f}".format(numpy.mean(loss_array))
                pbar.set_postfix(loss=loss, test_accuracy=accuracy)

            if verbose >= 1:
                pbar.close()

    def predict(self, test_x):

        # convert into 2d arrays
        network_input = numpy.array(test_x, ndmin=2).T

        # feed forward all layers
        feed_forward = network_input
        for layer in self.layers:
            feed_forward = layer.forward(feed_forward)
        network_predicted = feed_forward

        return network_predicted

    def model_test_accuracy(self):

        (x_test, y_test) = self.test_accuracy

        # evaluate
        test_score = []
        for (batch_x, batch_y) in zip(x_test, y_test):

            outputs = self.predict(batch_x)

            predicted_label = numpy.argmax(outputs)
            correct_label = numpy.argmax(batch_y)
            
            if (predicted_label == correct_label):
                test_sNeuralNetworks.append(1)
            else:
                test_sNeuralNetworks.append(0)

        test_score_array = numpy.asarray(test_score)
        accuracy = "{0:.2f}%".format(100.0*(test_score_array.sum() / test_score_array.size))

        return accuracy

