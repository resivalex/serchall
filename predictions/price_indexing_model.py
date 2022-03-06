from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator
import numpy as np


class PriceIndexingModel(BaseEstimator):

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, x, y):
        x = np.log(x)
        self.model.fit(x, y)

    def predict(self, x):
        x = np.log(x)
        return self.model.predict(x)
