import numpy as np

def mse(y_true, y_pred):
  return np.mean(np.power(y_pred - y_true, 2))
  
def mse_prime(y_true, y_pred):
  return 2 * (y_pred - y_true) / y_true.size

loss_map = {
  'mse': (mse, mse_prime),
}