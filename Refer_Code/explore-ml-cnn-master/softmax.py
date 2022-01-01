import numpy as np

class Softmax:
  def __init__(self, input_len, nodes):
    self.weights = np.random.randn(input_len, nodes) / input_len
    self.biases = np.zeros(nodes)

  def forward(self, input):
    self.last_input_shape = input.shape

    input = input.flatten()
    self.last_input = input

    input_len, nodes = self.weights.shape
    totals = np.dot(input, self.weights) + self.biases
    self.last_totals = totals
    exp = np.exp(totals)

    return exp / np.sum(exp, axis=0)

  def backprop(self, d_L_d_out, learn_rate):
    for i, gradient in enumerate(d_L_d_out):
      if gradient == 0:
        continue

      t_exp = np.exp(self.last_totals)
      S = np.sum(t_exp)

      d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
      # since only one i passes and it is the label
      d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

      d_t_d_w = self.last_input
      d_t_d_b = 1
      d_t_d_inputs = self.weights

      d_L_d_t = gradient * d_out_d_t

      d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
      d_L_d_b = d_L_d_t * d_t_d_b
      d_L_d_inputs = d_t_d_inputs @ d_L_d_t

      self.weights -= learn_rate * d_L_d_w
      self.biases -= learn_rate * d_L_d_b

      return d_L_d_inputs.reshape(self.last_input_shape)
