import pandas as pd
import numpy as np
import datetime
import warnings


warnings.filterwarnings('ignore')


def _geometric_mean(x):
    return np.exp(np.log(x).mean())


MIN_DEFAULT_DATE = datetime.date(2015, 1, 1)
MAX_DEFAULT_DATE = datetime.date.today() + datetime.timedelta(366 * 3) # 3 years from now


def build_price_index(df,
                      min_date=MIN_DEFAULT_DATE,
                      max_date=MAX_DEFAULT_DATE,
                      outliers_percentile=0):
    """
    Строит индекс изменения цен

    Parameters
    ----------
    df : pandas.DataFrame
        Датафрейм с колонками "name", "date", "price"
    min_date : datetime.date
        Первая дата результирующего индекса
    max_date: datetime.date
        Последняя дата результирующего индекса
    outliers_percentile: int
        Сколько процентов отрезков отбрасывать с каждой стороны по угловому коэффициенту

    Returns
    -------
    pandas.DataFrame
        Датафрейм с колонками "date", "coef"

    Examples
    --------
    >>> build_price_index(pd.DataFrame({
    >>>    'name': ['A', 'A'],
    >>>    'date': [datetime.date(2016, 1, 1), datetime.date(2021, 1, 1)],
    >>>    'price': [200, 300]
    >>> }))
    """
    price_changes = get_normalized_price_changes(df)
    price_changes_without_outliers = remove_outliers(price_changes, outliers_percentile)
    daily_price_changes = calculate_price_changes_by_dates(price_changes_without_outliers)
    price_index = convert_day_changes_to_index(daily_price_changes, min_date, max_date)
    return dict(
        price_index=price_index,
        daily_price_changes=daily_price_changes
    )


def get_normalized_price_changes(df):
    price_changes = []
    for name, name_df in df.groupby('name'):
        name_price_df = name_df.groupby('date', as_index=False)['price'].mean()
        name_price_df = name_price_df.sort_values('date')
        if len(name_price_df) == 1:
            continue
        for (date_1, price_1), (date_2, price_2) in zip(name_price_df.values, name_price_df.values[1:]):
            price_changes.append((date_1, date_2, price_2 / price_1))
    return price_changes


def remove_outliers(price_changes, outliers_percentile):
    day_slopes = [coef ** (1.0 / (date_2 - date_1).days) for date_1, date_2, coef in price_changes]
    left_percentile = np.percentile(day_slopes, outliers_percentile)
    right_percentile = np.percentile(day_slopes, 100 - outliers_percentile)

    return [
        item
        for item, day_slope
        in zip(price_changes, day_slopes)
        if day_slope > left_percentile and day_slope < right_percentile
    ]


def calculate_price_changes_by_dates(price_changes):
    day_coefs = []
    for date_1, date_2, coef in price_changes:
        days = (date_2 - date_1).days
        day_coef = coef ** (1.0 / days)
        for i in range(days):
            cur_date = date_1 + datetime.timedelta(days=i)
            day_coefs.append((cur_date, day_coef))
    df = pd.DataFrame(day_coefs, columns=['date', 'coef'])
    aggregated_coefs = []
    for date, date_df in df.groupby('date'):
        aggregated_coefs.append((date, _geometric_mean(date_df['coef'])))
    df = pd.DataFrame(aggregated_coefs, columns=['date', 'coef'])
    df = df.sort_values('date')

    return df


def convert_day_changes_to_index(day_coefs, min_date, max_date):
    average_day_coef = _geometric_mean(day_coefs['coef'])
    day_coef_dict = {row['date']: row['coef'] for _, row in day_coefs.iterrows()}
    cur_date = min_date
    cur_coef = 1.0
    result = []
    while cur_date <= max_date:
        result.append((cur_date, cur_coef))
        cur_coef *= day_coef_dict.get(cur_date, average_day_coef)
        cur_date = cur_date + datetime.timedelta(days=1)
    return pd.DataFrame(result, columns=['date', 'coef'])
