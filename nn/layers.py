import numpy as np

class Layer():
  def __init__(self):
    self.input = None
    self.output = None

  def forward_propagate(self, input):
    raise NotImplementedError

  def backward_propagate(self, output_error, learning_rate):
    raise NotImplementedError


class DenseLayer(Layer):
  def __init__(self, input_size, output_size):
    self.weights = np.random.rand(input_size, output_size) - 0.5
    self.bias = np.random.rand(1, output_size) - 0.5

  def forward_propagate(self, input_data):
    self.input = input_data
    self.output = np.dot(self.input, self.weights) + self.bias
    return self.output

  def backward_propagate(self, output_error, learning_rate):
    input_error = np.dot(output_error, self.weights.T)
    weights_error = np.dot(self.input.T, output_error)
    self.weights -= learning_rate * weights_error
    # Note dE/dB = dE/dY
    self.bias -= learning_rate * output_error
    return input_error


class ActivationLayer(Layer):
  def __init__(self, activation, activation_prime):
    self.activation = activation
    self.activation_prime = activation_prime

  def forward_propagate(self, input_data):
      self.input = input_data
      self.output = self.activation(self.input)
      return self.output

  def backward_propagate(self, output_error, learning_rate):
      return self.activation_prime(self.input) * output_error