import streamlit as st
import pandas as pd


st.set_page_config(page_title='#2', layout='wide')


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

    needle = st.text_input('Фильтр')
    data = data[[(needle.lower() in name.lower()) for name in data['name']]]

    data.columns = original_columns
    st.text(f'Строк: {len(data)}')
    st.dataframe(data, height=600)


if __name__ == "__main__":
    main()
