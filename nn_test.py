from nn.models import NeuralNetwork
from nn.layers import DenseLayer, ActivationLayer
from nn.activations import tanh, tanh_prime
from nn.losses import mse, mse_prime
from argparse import ArgumentParser
import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils

parser = ArgumentParser()
parser.add_argument('test', help='Name of the test to run in {xor, mnist}', type=str, default='xor')
args = parser.parse_args()

def xor_test():
  X_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
  y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

  net = NeuralNetwork()
  net.add(DenseLayer(2, 3))
  net.add(ActivationLayer(tanh, tanh_prime))
  net.add(DenseLayer(3, 1))
  net.add(ActivationLayer(tanh, tanh_prime))

  net.use_loss(mse, mse_prime)
  net.fit(X_train, y_train, n_epochs=1000, learning_rate=0.1)

  out = net.predict(X_train)
  print(out)

def mnist_test():
  (X_train, y_train), (x_test, y_test) = mnist.load_data()

  X_train = X_train.reshape(X_train.shape[0], 1, 28*28)
  X_train = X_train.astype('float32')
  X_train /= 255
  y_train = np_utils.to_categorical(y_train)

  x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
  x_test = x_test.astype('float32')
  x_test /= 255
  y_test = np_utils.to_categorical(y_test)

  # Network
  net = NeuralNetwork()
  net.add(DenseLayer(28*28, 100))
  net.add(ActivationLayer(tanh, tanh_prime))
  net.add(DenseLayer(100, 50))
  net.add(ActivationLayer(tanh, tanh_prime))
  net.add(DenseLayer(50, 10))
  net.add(ActivationLayer(tanh, tanh_prime))

  net.use_loss(mse, mse_prime)
  net.fit(X_train[0:1000], y_train[0:1000], n_epochs=35, learning_rate=0.1)

  out = net.predict(x_test[0:3])
  print("\n")
  print("predicted values : ")
  print(out)
  print("true values : ")
  print(y_test[0:3])

if __name__ == '__main__':
  if args.test == 'xor':
    xor_test()
  elif args.test == 'mnist':
    mnist_test()
  else:
    raise ValueError(f"Unknown test name '{args.test}'")