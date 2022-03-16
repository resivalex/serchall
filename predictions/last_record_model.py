from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class Model(BaseEstimator, RegressorMixin):

    def __init__(self):
        self.name_predictions = None

    def fit(self, x, y):
        data = x.copy()
        data['price'] = y
        self.name_predictions = {}
        for name, df in data.groupby('name'):
            df = df.sort_values('order_date', ascending=False)
            self.name_predictions[name] = df.iloc[0]['price']

    def predict(self, x):
        y = []
        for i, row in x.iterrows():
            name = row['name']
            y.append(self.name_predictions.get(name, None))

        return np.array(y)
