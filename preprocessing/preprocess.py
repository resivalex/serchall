import pandas as pd
import re


def preprocess(df):
    df['Наименование2'] = df.Наименование.apply(
        lambda s: s.lower().strip().replace("  ", "").replace('.', '-')
    )
    sub_dict = {
        r'(зуб)\s(.*)\s*(ковша)\s*(.+)': r'\1 \3 \2 \4',
        '   ': ' ',
        '  ': ' ',
        ' ,': ',',
        r',(\S)': r', \1',
        ',* *с наплавкой': ' наплавка'
    }
    df['Наименование3'] = df.Наименование2

    for i, (k, v) in enumerate(sub_dict.items()):
        df.Наименование3 = df.Наименование3.apply(lambda s: re.sub(k, v, s))

    df.drop(columns="Наименование2", inplace=True)

    # Can't use rows with empty delivery date.
    df = df[df['Дата поставки'].notnull()]

    return df


if __name__ == '__main__':
    data_in = pd.read_excel('../ini_data/datamon.xlsx')
    data_out = preprocess(data_in)
    print(data_out.sample(10))

    import os
    os.system('pip freeze > requirements.txt')
