import numpy as np

def tanh(x):
  return np.tanh(x)
  
def tanh_prime(x):
  return 1 - np.tanh(x) ** 2

def sigmoid(x):
  return 1 / (1 + np.exp(-x + np.max(x)))
  
def sigmoid_prime(x):
  return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / np.sum(e_x)

def softmax_prime(x):
  return softmax(x) * (1 - softmax(x))

activation_map = {
  "tanh": (tanh, tanh_prime),
  "sigmoid": (sigmoid, sigmoid_prime),
  "softmax": (softmax, softmax_prime)
}