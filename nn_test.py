from nn.models import NeuralNetwork
from nn.layers import DenseLayer, ActivationLayer, Dropout
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
  net.add(ActivationLayer("tanh"))
  net.add(DenseLayer(3, 1))
  net.add(ActivationLayer("sigmoid"))

  print(net)

  net.use_loss("mse")
  net.fit(X_train, y_train, n_epochs=100, learning_rate=0.1)

  out = net.predict(X_train)
  print(out)

def mnist_test():
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

  X_train = X_train.reshape(X_train.shape[0], 1, 28*28)
  X_train = X_train.astype('float32')
  X_train /= 255
  y_train = np_utils.to_categorical(y_train)

  X_test = X_test.reshape(X_test.shape[0], 1, 28*28)
  X_test = X_test.astype('float32')
  X_test /= 255
  y_test = np_utils.to_categorical(y_test)

  # Network
  net = NeuralNetwork()
  net.add(DenseLayer(28*28, 100))
  net.add(ActivationLayer("tanh"))
  net.add(DenseLayer(100, 50))
  net.add(Dropout(0.2))
  net.add(ActivationLayer("tanh"))
  net.add(DenseLayer(50, 10))
  net.add(ActivationLayer("softmax"))

  print(net)

  net.use_loss("mse")
  net.fit(X_train[0:1000], y_train[0:1000], n_epochs=50, learning_rate=0.1)
  sample_indexes = np.random.randint(0, len(X_test), 10)

  print(f'[TEST ACCURACY] :: {net.evaluate(X_test, y_test, classification=True)}')

  print('\n[SAMPLE PREDICTIONS]')
  out = net.predict(X_test[sample_indexes])
  print("Predicted values")
  print(np.squeeze(np.argmax(out, axis=-1)))
  print("True values")
  print(np.argmax(y_test[sample_indexes], axis=-1))

if __name__ == '__main__':
  if args.test == 'xor':
    xor_test()
  elif args.test == 'mnist':
    mnist_test()
  else:
    raise ValueError(f"Unknown test name '{args.test}'")