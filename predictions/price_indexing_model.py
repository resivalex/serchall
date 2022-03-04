from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator


class PriceIndexingModel(BaseEstimator):

    def __init__(self):
        self.model = RandomForestRegressor()

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)
