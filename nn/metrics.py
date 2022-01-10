import numpy as np

def accuracy(y_true, y_pred):
  y_true, y_pred = np.squeeze(y_true), np.squeeze(y_pred)
  classes_pred = np.argmax(y_pred, axis=-1)
  classes_true = np.argmax(y_true, axis=-1)
  correct_indexes = np.where(classes_pred - classes_true == 0)[0]
  return len(correct_indexes) / len(y_true)

