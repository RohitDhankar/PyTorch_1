import mnist
import numpy as np
from PIL import Image
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

train_images = mnist.train_images()[:100]
train_labels = mnist.train_labels()[:100]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(13 * 13 * 8, 10)

def forward(image, label):
  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)

  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

def train(image, label, lr=.005):
  out, loss, acc = forward(image, label)

  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]
  gradient = softmax.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  gradient = conv.backprop(gradient, lr)

  return out, loss, acc

def start_training():
  loss = 0
  num_correct = 0
  for i, (image, label) in enumerate(zip(train_images, train_labels)):
    if i % 100 == 99:
      print(
        f'[Step {i + 1}] Past 100 steps: Average loss {loss / 100}. Accuracy: {num_correct}%'
      )
      loss = 0
      num_correct = 0

    _, l, acc = train(image, label)
    loss += l
    num_correct += acc

def test(image, label):
  np_image = np.array(image)
  result, loss, acc = train(np_image, label)
  return result.tolist()
