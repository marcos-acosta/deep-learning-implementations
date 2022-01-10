import numpy as np
from . import losses
from . import metrics

class NeuralNetwork():
  def __init__(self):
    self.layers = []
    self.loss = None
    self.loss_prime = None

  def add(self, layer):
    self.layers.append(layer)

  def use_loss(self, loss_name):
    self.loss, self.loss_prime = losses.loss_map[loss_name]

  def predict(self, input_data):
    n_samples = len(input_data)
    result = []
    for i in range(n_samples):
      output = input_data[i]
      for layer in self.layers:
        output = layer.forward_propagate(output)
      result.append(output)
    return np.array(result)

  def evaluate(self, X_test, y_test, classification=False):
    pred = self.predict(X_test)
    if classification:
      return metrics.accuracy(y_test, pred)
    else:
      return losses.mse(y_test, pred)

  def fit(self, X_train, y_train, n_epochs, learning_rate):
    n_samples = len(X_train)
    for i in range(n_epochs):
      err = 0
      for j in range(n_samples):
        output = X_train[j]
        for layer in self.layers:
          output = layer.forward_propagate(output)
        err += self.loss(y_train[j], output)
        error = self.loss_prime(y_train[j], output)
        for layer in reversed(self.layers):
          error = layer.backward_propagate(error, learning_rate)
      err /= n_samples
      print(f'Epoch {i+1}/{n_epochs} :: loss {err}')

  def __repr__(self):
    ret = '\nMODEL SUMMARY\n~~~~~~~~~~~~~\n'
    n_parameters = 0
    for layer in self.layers:
      ret += str(layer) + '\n'
      n_parameters += layer.weights.size + layer.bias.size if layer.has_weights else 0
    ret += f'\n{n_parameters} trainable parameters\n'
    return ret