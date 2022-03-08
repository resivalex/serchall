import pandas as pd
import numpy as np
import datetime
from prophet import Prophet


def construct(df):
    price_changes = get_normalized_price_changes(df)
    price_changes_without_outliers = remove_outliers(price_changes)
    day_price_changes = calculate_price_changes_by_dates(price_changes_without_outliers)
    price_index = convert_day_changes_to_index(day_price_changes)
    extended_line = extend_line(price_index, datetime.datetime.today().date())
    return dict(
        price_index=price_index,
        extended_line=extended_line,
        day_price_changes=day_price_changes
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


def remove_outliers(price_changes):
    day_slopes = [coef ** (1.0 / (date_2 - date_1).days) for date_1, date_2, coef in price_changes]
    left_percentile = np.percentile(day_slopes, 10)
    right_percentile = np.percentile(day_slopes, 90)

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
    df = df.groupby('date', as_index=False)['coef'].mean()
    df = df.sort_values('date')

    return df


def convert_day_changes_to_index(day_coefs):
    cur_date = day_coefs.iloc[0]['date']
    cur_coef = 1.0
    result = [(cur_date, cur_coef)]
    for i in day_coefs.index:
        cur_coef *= day_coefs.at[i, 'coef']
        result.append((day_coefs.at[i, 'date'] + datetime.timedelta(days=1), cur_coef))
    return pd.DataFrame(result, columns=['date', 'coef'])


def extend_line(df, end_date):
    df = df.copy()
    original_columns = list(df.columns)
    df.columns = ['ds', 'y']
    days_to_add = (end_date - df['ds'].max()).days
    model = Prophet()
    model.fit(df)
    future_days = 366 * 3 # 3 years
    future_df = model.make_future_dataframe(periods=days_to_add + future_days)
    pred_df = model.predict(future_df)
    result = pred_df[['ds', 'yhat']]
    result.columns = original_columns

    return result
