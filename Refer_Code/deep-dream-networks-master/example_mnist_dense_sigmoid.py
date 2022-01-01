
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