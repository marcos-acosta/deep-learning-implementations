import numpy as np

class NeuralNetwork():
  def __init__(self):
    self.layers = []
    self.loss = None
    self.loss_prime = None

  def add(self, layer):
    self.layers.append(layer)

  def use_loss(self, loss, loss_prime):
    self.loss = loss
    self.loss_prime = loss_prime

  def predict(self, input_data):
    n_samples = len(input_data)
    result = []
    for i in range(n_samples):
      output = input_data[i]
      for layer in self.layers:
        output = layer.forward_propagate(output)
      result.append(output)
    return np.array(result)

  def fit(self, x_train, y_train, n_epochs, learning_rate):
    n_samples = len(x_train)
    for i in range(n_epochs):
      err = 0
      for j in range(n_samples):
        output = x_train[j]
        for layer in self.layers:
          output = layer.forward_propagate(output)
        err += self.loss(y_train[j], output)
        error = self.loss_prime(y_train[j], output)
        for layer in reversed(self.layers):
          error = layer.backward_propagate(error, learning_rate)
      err /= n_samples
      print(f'Epoch {i+1}/{n_epochs} :: loss {err}')