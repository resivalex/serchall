from sklearn.metrics import mean_absolute_percentage_error
import numpy as np


def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)


def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
