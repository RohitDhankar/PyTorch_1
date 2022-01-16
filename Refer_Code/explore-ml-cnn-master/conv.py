# conda activate pytorch_venv

## ## Code Source >> https://github.com/teralion/explore-ml-cnn/blob/master/conv.py
## ## Code Source -- CREDITS >> BORIS 
#  https://github.com/teralion/explore-ml-cnn

import numpy as np

class Conv3x3:
  def __init__(self, num_filters):
    self.num_filters = num_filters
    self.filters = np.random.randn(num_filters, 3, 3) / 9

  def iterate_regions(self, image):
    w, h = image.shape

    for i in range(h - 2):
      for j in range(w - 2):
        image_region = image[i:(i + 3), j:(j + 3)]
        yield image_region, i, j

  def forward(self, input):
    self.last_input = input
    w, h = input.shape
    output = np.zeros((w - 2, h - 2, self.num_filters))

    for image_region, i, j in self.iterate_regions(input):
      output[i, j] = np.sum(image_region * self.filters, axis=(1, 2))

    return output

  def backprop(self, d_L_d_out, learn_rate):
    d_L_d_filters = np.zeros(self.filters.shape)

    for image_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * image_region

    self.filters -= learn_rate * d_L_d_filters

    return None
