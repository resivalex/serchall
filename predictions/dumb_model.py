from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


def _geometric_mean(x):
    return np.exp(np.log(x).mean())


class Model(BaseEstimator, RegressorMixin):

    def __init__(self, mean_kind='geometric'):
        self.name_predictions = None
        self.mean_kind = mean_kind

    def fit(self, x, y):
        data = x.copy()
        data['price'] = y
        self.name_predictions = {}
        for name, df in data.groupby('name'):
            if self.mean_kind == 'geometric':
                pred = _geometric_mean(df['price'])
            elif self.mean_kind == 'arithmetic':
                pred = df['price'].mean()
            else:
                raise Exception('Unknown mean_kind')
            self.name_predictions[name] = pred

    def predict(self, x):
        y = []
        for i, row in x.iterrows():
            name = row['name']
            y.append(self.name_predictions.get(name, None))

        return np.array(y)
