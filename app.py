import streamlit as st
import pandas as pd


st.set_page_config(page_title='#2', layout='wide')


def main():
    st.text('Welcome!')

    data = pd.read_csv('data.csv')
    st.dataframe(data)

if __name__ == "__main__":
    main()
