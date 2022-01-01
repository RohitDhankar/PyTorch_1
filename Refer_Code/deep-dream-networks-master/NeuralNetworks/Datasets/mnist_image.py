# import external
import numpy
import mnist

def onehot_encode(labels):

    onehot_return = numpy.zeros((labels.size, labels.max()+1))
    onehot_return[numpy.arange(labels.size),labels] = 1

    return onehot_return

def load_data():

    x_train = (mnist.train_images() / 255.0) - 0.5
    y_train = onehot_encode(mnist.train_labels())
    x_test = (mnist.test_images()[:1000] / 255.0) - 0.5
    y_test = onehot_encode(mnist.test_labels()[:1000])

    return (x_train, y_train), (x_test, y_test)
