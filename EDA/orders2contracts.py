import pandas as pd


def orders2contracts(df0: pd.DataFrame, name_col: str = 'name2') -> pd.DataFrame:
    names = df0[name_col].unique()
    df1 = df0.copy()
    drop_idxs = []

    for name in names:
        prices = []
        flt = df0[df0[name_col] == name]

        for i, row in flt.iterrows():
            if row.price in prices:
                drop_idxs.append(i)
            else:
                prices.append(row.price)

    df1.drop(index=drop_idxs, inplace=True)

    return df1


if __name__ == '__main__':
    df = pd.read_excel('../ini_data/ini_data_processed.xlsx')
    df.columns = ['name', 'delivery_date', 'order_date', 'delivery_period',
                  'scheduled_delivery_period', 'region', 'quantity', 'price', 'pmt_term',
                  'non_scheduled', 'supplier', 'order_date_est', 'order_date_factual',
                  'name2', 'group']
    df.sort_values('order_date_est', ascending=True, inplace=True)
    df.drop(
        columns=['delivery_date', 'delivery_period', 'pmt_term', 'name', 'order_date'],
        inplace=True)
    df.non_scheduled = df.non_scheduled.fillna(0)
    df = df[df.order_date_est.notnull()]
    df.order_date_est = df.order_date_est.apply(lambda d: int(round(d.timestamp()) / 100))
    df.order_date_est = df.order_date_est.apply(lambda d: d - df.order_date_est.min())

    min_count = 5
    filtered_names = df.groupby('name2').count()['quantity'].sort_values(ascending=False)
    filtered_names = filtered_names[filtered_names > min_count]

    df_flt = df[df.name2.isin(filtered_names.index)]

    out = orders2contracts(df_flt)
    print(out)
