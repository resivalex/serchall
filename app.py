import datetime
import streamlit as st
import pandas as pd
from preprocessing.secondary_preprocess import preprocess
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from predictions.price_indexing.model import Model as PriceIndexingModel


st.set_page_config(page_title='Мониторинг цен', layout='wide')


def plot_price_changes(price_changes):
    fig, ax = plt.subplots()
    ax.set_title('Ежедневное изменение цен')
    sns.lineplot(data=price_changes, x='date', y='coef', ax=ax, color='black')
    st.pyplot(fig)


def plot_price_index(price_index):
    fig, ax = plt.subplots()
    ax.set_title('Индекс цен')
    sns.lineplot(data=price_index, x='date', y='coef', ax=ax, color='black')
    st.pyplot(fig)


def predict_today_price_block(data, model: PriceIndexingModel):
    names = data.sort_values('name', ascending=True)['name'].unique()
    name = st.selectbox('Наименование', names)
    name_df = data[data['name'] == name]

    today = datetime.datetime.today().date()
    x = pd.DataFrame([{'order_date': today, 'name': name}])

    today_price = model.predict(x)[0]
    st.text(f'Прогнозная цена на {today.isoformat()}: {today_price:,.02f}₽')
    st.dataframe(name_df)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.set_title('Прогнозные цены')
        name_price_index = model.price_index.copy()
        name_price_index['price'] =  name_price_index['coef'] * today_price / model.get_date_price_coef(today)
        sns.lineplot(data=name_price_index, x='date', y='price', ax=ax, color='black')
        sns.scatterplot(data=pd.DataFrame([{'date': today, 'price': today_price}]),
                        x='date', y='price', ax=ax, color='black', s=50)
        sns.scatterplot(data=name_df, x='calculated_order_date', y='price', ax=ax, color='black')
        st.pyplot(fig)


def model_page(data):
    data = preprocess(data)
    data = data[~data['order_date'].isna()]

    def reset_cache():
        os.remove('cache/price_indexing_model.joblib')

    if os.path.exists('cache/price_indexing_model.joblib'):
        try:
            model = joblib.load('cache/price_indexing_model.joblib')
        except Exception as exc:
            st.error(exc)
            reset_cache()
            st.text('Reload the page')
            return
    else:
        model = PriceIndexingModel()
        model.fit(data.drop('price', axis=1), data['price'])
        joblib.dump(model, 'cache/price_indexing_model.joblib')
    with st.expander('Технические детали'):
        col1, col2 = st.columns(2)
        with col1:
            plot_price_changes(model.daily_price_changes)
        with col2:
            plot_price_index(model.price_index)
        st.button('Очистить кэш и пересчитать индекс', on_click=reset_cache)
    with st.expander('Прогнозы', expanded=True):
        predict_today_price_block(data, model)


def main():
    data = pd.read_excel('ini_data/datamon.xlsx')
    model_page(data)


if __name__ == "__main__":
    main()
