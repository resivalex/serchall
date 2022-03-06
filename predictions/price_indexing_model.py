from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator


class PriceIndexingModel(BaseEstimator):

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)
