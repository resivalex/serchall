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
    st.metric('Прогнозная цена', f'{today_price:,.02f} ₽'.replace(',', ' '))
    name_df['out_of_plan'] = name_df['out_of_plan'].apply(lambda x: 'Да' if x else 'Нет')
    name_df = name_df.rename({
        'name': 'Наименование',
        'delivery_date': 'Дата поставки',
        'order_date': 'Дата заказа',
        'delivery_period': 'Срок поставки',
        'planned_delivery_period': 'Плановый срок поставки',
        'region': 'Регион',
        'amount': 'Объем заказа',
        'price': 'Цена, руб',
        'payment_conditions': 'Условия платежа',
        'out_of_plan': 'Внеплановая закупка',
        'supplier': 'Поставщик',
        'group': 'Группа'
    }, axis=1)
    name_df = name_df.drop(['calculated_order_date', 'has_order_date'], axis=1)

    st.dataframe(name_df)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.set_title('Индекс цен')
        name_price_index = model.price_index.copy()
        name_price_index['price'] =  name_price_index['coef'] * today_price / model.get_date_price_coef(today)
        sns.lineplot(data=name_price_index, x='date', y='price', ax=ax, color='black')
        sns.scatterplot(data=pd.DataFrame([{'date': today, 'price': today_price}]),
                        x='date', y='price', ax=ax, color='black', s=50)
        sns.scatterplot(data=name_df, x='Дата поставки', y='Цена, руб', ax=ax, color='black')
        plt.ylabel('Цена, руб')
        plt.xlabel('Дата')
        st.pyplot(fig)
    with col2:
        st.markdown('#### Статистические метрики')
        st.table(pd.DataFrame(index=[
            'Количество измерений наименования',
            'Минимальная цена',
            'Максимальная цена',
            'Группа',
            'Количество измерений в группе'
        ],
            data={'': [
                str(len(name_df)),
                f"{name_df['Цена, руб'].min():,.02f} ₽".replace(',', ' '),
                f"{name_df['Цена, руб'].max():,.02f} ₽".replace(',', ' '),
                name_df.iloc[0]['Группа'],
                str(len(data[data['group'] == name_df.iloc[0]['Группа']]))
            ]}
        ))


def model_page(data):

    def reset_cache():
        os.remove('cache/preprocessed_data.joblib')
        os.remove('cache/price_indexing_model.joblib')

    if os.path.exists('cache/preprocessed_data.joblib'):
        data = joblib.load('cache/preprocessed_data.joblib')
    else:
        data = preprocess(data)
        data = data[~data['order_date'].isna()]
        data['name'] = data['cleaned_name'].apply(lambda x: x.capitalize())
        data = data.drop(['cleaned_name'], axis=1)
        outliers = {'редуктор 3572-05-11-000', 'подшипник седловой 3546-03-04-000-03', 'шпилька 3550-05-00-012-03'}
        data = data[[name not in outliers for name in data['name']]]
        data = data.sort_values(['name', 'order_date'])
        data.index = list(range(len(data)))
        joblib.dump(data, 'cache/preprocessed_data.joblib')

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
    with st.expander('Анализ по наименованию', expanded=True):
        predict_today_price_block(data, model)


def main():
    data = pd.read_excel('ini_data/datamon.xlsx')
    model_page(data)


if __name__ == "__main__":
    main()
