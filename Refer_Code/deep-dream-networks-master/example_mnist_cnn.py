
# import model
from NeuralNetworks.Layers import Convolutional
from NeuralNetworks.Layers import MaxPooling
from NeuralNetworks.Layers import Softmax
from NeuralNetworks.Model import Sequential
from NeuralNetworks.Datasets import mnist_image

# load data
(x_train, y_train), (x_test, y_test) = mnist_image.load_data()
input_shape = (28, 28)

# create model
model = Sequential()
model.add(Convolutional(8, input_shape=input_shape))
model.add(MaxPooling())
model.add(Softmax(10))
model.compile(lr=0.005, loss='categorical_crossentropy')

# fit model
model.fit(x_train, y_train, batch_size = 1, epochs = 2, verbose = 1, test_accuracy=(x_test, y_test))

"""
Epoch: 0: 100%|██████████████████████████████████████| 60000.0/60000 [37:42<00:00, 26.52it/s, loss=0.2525, test_accuracy=95.00%]
Epoch: 1: 100%|██████████████████████████████████████| 60000.0/60000 [39:33<00:00, 25.28it/s, loss=0.1331, test_accuracy=96.70%]
"""