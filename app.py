import datetime
import streamlit as st
import pandas as pd
from preprocessing.secondary_preprocess import preprocess
from predictions.price_indexing.construction import construct
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np


st.set_page_config(page_title='Мониторинг цен', layout='wide')


def plot_price_changes(price_changes):
    fig, ax = plt.subplots()
    ax.set_title('Ежедневное изменение цен')
    sns.lineplot(data=price_changes, x='date', y='coef', ax=ax, color='black')
    st.pyplot(fig)


def plot_price_index(price_index, extended_index):
    fig, ax = plt.subplots()
    ax.set_title('Индекс цен')
    sns.lineplot(data=extended_index, x='date', y='coef', ax=ax, color='lightgreen')
    sns.lineplot(data=price_index, x='date', y='coef', ax=ax, color='black')
    st.pyplot(fig)


def geometric_mean(x):
    return np.exp(np.log(x).mean())


def predict_today_price_block(data, price_index):
    names = data.sort_values('name', ascending=True)['name'].unique()
    name = st.selectbox('Наименование', names)
    name_df = data[data['name'] == name]
    iso_price_index = dict([
                               (date.date().isoformat(), price)
                               for date, price
                               in zip(price_index['date'], price_index['coef'])
    ])

    def get_date_coef(date):
        return iso_price_index[date.isoformat()]

    price_coefs = []
    for date, price in zip(name_df['calculated_order_date'], name_df['price']):
        price_coefs.append(price / get_date_coef(date))
    price_coef = geometric_mean(price_coefs)
    name_price_index = price_index.copy()
    name_price_index['price'] = name_price_index['coef'] * price_coef

    today = datetime.datetime.today().date()
    today_price = price_coef * get_date_coef(today)
    st.text(f'Прогнозная цена на {today.isoformat()}: {today_price:,.02f}₽')
    st.dataframe(name_df)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.set_title('Прогнозные цены')
        sns.lineplot(data=name_price_index, x='date', y='price', ax=ax, color='black')
        sns.scatterplot(data=pd.DataFrame([{'date': today, 'price': today_price}]),
                        x='date', y='price', ax=ax, color='black', s=50)
        sns.scatterplot(data=name_df, x='calculated_order_date', y='price', ax=ax, color='black')
        st.pyplot(fig)


def model_page(data):
    data = preprocess(data)

    data_for_index = data[['name', 'price', 'calculated_order_date']].rename({'calculated_order_date': 'date'}, axis=1)
    def reset_cache():
        os.remove('cache/common_index.joblib')
        result = construct(data_for_index)
        joblib.dump(result, 'cache/common_index.joblib')
        return result

    if os.path.exists('cache/common_index.joblib'):
        result = joblib.load('cache/common_index.joblib')
        if result['extended_line']['date'].max() != datetime.datetime.today().date():
            result = reset_cache()
    else:
        result = construct(data_for_index)
        joblib.dump(result, 'cache/common_index.joblib')
    with st.expander('Технические детали'):
        col1, col2 = st.columns(2)
        with col1:
            plot_price_changes(result['day_price_changes'])
        with col2:
            plot_price_index(result['price_index'], result['extended_line'])
        st.button('Очистить кэш и пересчитать индекс', on_click=reset_cache)
    with st.expander('Прогнозы', expanded=True):
        predict_today_price_block(data, result['extended_line'])


def main():
    data = pd.read_excel('ini_data/datamon.xlsx')
    model_page(data)


if __name__ == "__main__":
    main()
