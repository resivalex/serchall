from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
from predictions.price_indexing.construction import build_price_index as construct_price_index


BASE_DATE = datetime.date(2015, 1, 1)


def _geometric_mean(x):
    return np.exp(np.log(x).mean())


class Model(BaseEstimator, RegressorMixin):

    def __init__(self):
        self.name_average_prices = None
        self.model = LinearRegression()
        self.common_price_index_dict = None

    def fit(self, x, y):
        data = x.copy()
        data['price'] = y
        common_price_index = construct_price_index(data.rename({'order_date': 'date'}, axis=1)[['name', 'date', 'price']])['price_index']
        self.common_price_index_dict = {
            date: coef
            for date, coef
            in zip(common_price_index['date'], common_price_index['coef'])
        }
        data['price_index_coef'] = [self.__get_common_index_coef(date) for date in data['order_date']]
        self.name_average_prices = {}
        for name, df in data.groupby('name'):
            df = df.sort_values('order_date', ascending=False)
            self.name_average_prices[name] = _geometric_mean(df['price'])
        data['rel_price'] = data.apply((lambda row: row['price'] / self.name_average_prices[row['name']]), axis=1)
        x = [[(row['order_date'] - BASE_DATE).days * 0, row['price_index_coef']] for _, row in data.iterrows()]
        y = np.log(data['rel_price'])
        self.model.fit(x, y)

    def predict(self, x):
        y = []
        for i, row in x.iterrows():
            name = row['name']
            avg_price = self.name_average_prices.get(name, None)
            if avg_price is None:
                y.append(None)
                continue
            x = [[(row['order_date'] - BASE_DATE).days * 0, self.__get_common_index_coef(row['order_date'])]]
            y_pred = self.model.predict(x)[0]
            y_pred = np.exp(y_pred)
            price = y_pred * avg_price
            y.append(price)

        return np.array(y)

    def __get_common_index_coef(self, date):
        return self.common_price_index_dict.get(date, 1.0)
