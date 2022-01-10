import numpy as np
from . import activations

class Layer():
  def __init__(self):
    self.has_weights = False
    self.input = None
    self.output = None

  def forward_propagate(self, input):
    raise NotImplementedError

  def backward_propagate(self, output_error, learning_rate):
    raise NotImplementedError

  def __repr__(self):
    return str(self)


class DenseLayer(Layer):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.has_weights = True
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

  def __repr__(self):
    return f'[DENSE] :: {self.weights.shape[0]} neurons => {self.weights.shape[1]} neurons'


class ActivationLayer(Layer):
  def __init__(self, activation_fn_name):
    super().__init__()
    self.activation_name = activation_fn_name
    self.activation, self.activation_prime = activations.activation_map[activation_fn_name]

  def forward_propagate(self, input_data):
    self.input = input_data
    self.output = self.activation(self.input)
    return self.output

  def backward_propagate(self, output_error, learning_rate):
    return self.activation_prime(self.input) * output_error

  def __repr__(self):
    return f'[ACTIVATION] :: {self.activation_name}'

class Dropout(Layer):
  def __init__(self, rate):
    super().__init__()
    self.dropout_rate = rate

  def forward_propagate(self, input_data):
    self.input = input_data
    self.output = np.vectorize(lambda x: 0 if np.random.random() < self.dropout_rate else x)(input_data)
    return self.output

  def backward_propagate(self, output_error, learning_rate):
      return output_error

  def __repr__(self):
    return f'[DROPOUT] :: rate {self.dropout_rate}'