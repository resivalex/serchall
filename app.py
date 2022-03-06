import pendulum
import streamlit as st
import pandas as pd
from preprocessing.preprocess_initial import preprocess
from predictions.price_indexing_model import PriceIndexingModel
import random
import matplotlib.pyplot as plt


BASE_DATE = pendulum.date(2000, 1, 1)


st.set_page_config(page_title='#2', layout='wide')


def filter_by_name(data, original_columns):
    needle = st.text_input('Фильтр')
    data = data[[(needle.lower() in name.lower()) for name in data['name']]]

    data.columns = original_columns
    st.text(f'Строк: {len(data)}')
    st.dataframe(data, height=600)


def preprocessing(data):
    data = preprocess(data)
    st.dataframe(data)


def prepare_train_data(data):
    data = data[~data['order_date'].isna()]
    train_columns = ['shift', 'distance', 'coef']
    train_data = []
    for name, df in data.groupby('name'):
        if len(df) < 5:
            continue
        df = df.sort_values('order_date')
        for i in range(len(df) * 6): # Just random
            i1 = random.randint(0, len(df) - 1)
            i2 = random.randint(0, len(df) - 1)
            if i1 >= i2:
                continue
            distance = float((df.iloc[i2]['order_date'] - df.iloc[i1]['order_date']).days)
            if distance == 0:
                continue
            shift = float((df.iloc[i1]['order_date'] - BASE_DATE).days)
            coef = float(df.iloc[i2]['price'] / df.iloc[i1]['price'])
            if coef < 0.1 or coef > 10:
                continue
            train_data.append((shift, distance, coef))

    return pd.DataFrame(train_data, columns=train_columns)


def add_today_prediction(data, model):
    today = pendulum.today().date()
    today_price_key = f'{today.isoformat()}_price'
    data['coef'] = 1.0
    data[today_price_key] = 0.0
    for i in data.index:
        if data.at[i, 'order_date'] is None:
            data.at[i, today_price_key] = data.at[i, 'price']
        else:
            coef = model.predict(pd.DataFrame([{
                'shift': float((data.at[i, 'order_date'] - BASE_DATE).days),
                'distance': float((today - data.at[i, 'order_date']).days)
            }]))[0]
            data.at[i, 'coef'] = coef
            data.at[i, today_price_key] = data.at[i, 'price'] * coef
    return data


def model_page(data):
    data = preprocess(data)
    st.text('Количество сделок по запчасти')
    st.dataframe(data.groupby('name', as_index=False).size()[['name', 'size']].sort_values('size', ascending=False))
    train_data = prepare_train_data(data)
    st.text('Тренировочные данные')
    st.dataframe(train_data)
    st.text('Предсказания на сегодня...')
    model = PriceIndexingModel()
    model.fit(train_data[['shift', 'distance']], train_data['coef'])
    data = add_today_prediction(data, model)

    fig, ax = plt.subplots(figsize=(6, 4))
    today = pendulum.today().date()
    for i in data.index:
        if data.at[i, 'order_date'] is not None:
            ax.plot([data.at[i, 'order_date'], today], [1.0, data.at[i, 'coef']], alpha=0.1, linewidth=0.5, color='black')
    st.pyplot(fig)
    st.dataframe(data[['name', 'order_date', 'price', 'coef', data.columns[-1]]].sort_values('name'))


def main():
    data = pd.read_excel('ini_data/datamon.xlsx', dtype=str, na_filter=False)
    original_columns = data.columns
    data.columns = [
        'name',
        'delivery_date',
        'order_date',
        'delivery_period',
        'planned_delivery_period',
        'region',
        'amount',
        'price',
        'payment_conditions',
        'out_of_plan',
        'supplier'
    ]
    page_name = st.radio('Страница', [
        'Препроцессинг',
        'Фильтр по названию',
        'Модель'
    ])
    if page_name == 'Препроцессинг':
        preprocessing(data)
    elif page_name == 'Фильтр по названию':
        filter_by_name(data, original_columns)
    elif page_name == 'Модель':
        model_page(data)


if __name__ == "__main__":
    main()
