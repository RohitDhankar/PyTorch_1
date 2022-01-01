import numpy as np

class MaxPool2:
  def iterate_regions(self, image):
    h, w, _ = image.shape

    new_h = h // 2
    new_w = w // 2

    for i in range(new_h):
      for j in range(new_w):
        image_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
        yield image_region, i, j

  def forward(self, input):
    self.last_input = input
    h, w, num_filters = input.shape
    output = np.zeros((h // 2, w // 2, num_filters))

    for image_region, i, j in self.iterate_regions(input):
      output[i, j] = np.amax(image_region, axis=(0, 1))

    return output

  def backprop(self, d_L_d_out):
    d_L_d_input = np.zeros(self.last_input.shape)
    for image_region, i, j in self.iterate_regions(self.last_input):
      h, w, f = image_region.shape
      amax = np.amax(image_region, axis=(0, 1))

      for i2 in range(h):
        for j2 in range(w):
          for f2 in range(f):
            if image_region[i2, j2, f2] == amax[f2]:
              d_L_d_input[i * 2 + j2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

      return d_L_d_input
