from sklearn.base import BaseEstimator, RegressorMixin
from .construction import build_price_index
import numpy as np


def _geometric_mean(x):
    return np.exp(np.log(x).mean())


class Model(BaseEstimator, RegressorMixin):

    def __init__(self, use_cleaned_name=False):
        self.train_data = None
        self.day_price_changes = None
        self.price_index = None
        self.extended_line = None
        self.iso_price_index = None
        self.use_cleaned_name = use_cleaned_name

    def fit(self, x, y):
        data = x.copy()
        if self.use_cleaned_name:
            data['name'] = data['cleaned_name']
        data['price'] = y
        data = data.copy()[~data['order_date'].isna()]
        self.train_data = data
        data_for_index = \
            data[['name', 'price', 'order_date']] \
                .rename({'order_date': 'date'}, axis=1)
        result = build_price_index(data_for_index)
        self.day_price_changes = result['day_price_changes']
        self.price_index = result['price_index']
        self.extended_line = result['extended_line']
        self.iso_price_index = dict([
            (date.date().isoformat(), price)
            for date, price
            in zip(self.extended_line['date'], self.extended_line['coef'])
        ])

    def get_date_price_coef(self, date):
        key = date.isoformat()
        return self.iso_price_index.get(key, 1.0) # Hack. Prices in the past without data should be lower

    def predict(self, x):
        x = x.copy()
        if self.use_cleaned_name:
            x['name'] = x['cleaned_name']
        y = []
        for i, row in x.iterrows():
            price_coefs = []
            name_df = self.train_data[self.train_data['name'] == row['name']]
            if len(name_df) == 0:
                y.append(None)
                continue
            for date, price in zip(name_df['order_date'], name_df['price']):
                price_coef = self.get_date_price_coef(date)
                price_coefs.append(price / price_coef)
            price_coef = _geometric_mean(price_coefs)
            y.append(self.get_date_price_coef(row['order_date']) * price_coef)

        return np.array(y)
