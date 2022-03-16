from .metrics import mape, rmspe


def check_model(model, train_data, test_data, title='Some model'):
    print('Model:', title)
    model.fit(train_data.drop('price', axis=1), train_data['price'])
    y_pred = model.predict(test_data.drop('price', axis=1))
    y_pred_not_na_indexing = [item is not None for item in y_pred]
    print('With predictions:', sum(y_pred_not_na_indexing), '/', len(y_pred))
    y_pred_not_na = y_pred[y_pred_not_na_indexing]
    y_true_not_na = test_data['price'][y_pred_not_na_indexing]
    rmspe_value = rmspe(y_true_not_na, y_pred_not_na)
    print(f'RMSPE: {rmspe_value * 100:.02f}%')


def usage_example():
    from predictions.last_record_model import Model as LastRecordModel
    import datetime
    import pandas as pd
    from preprocessing.secondary_preprocess import preprocess

    data = pd.read_excel('../ini_data/datamon.xlsx')
    data = preprocess(data)
    data = data[~data['order_date'].isna()]
    first_test_date = datetime.date(2021, 1, 1)
    train_data = data[data['order_date'] < first_test_date]
    test_data = data[data['order_date'] >= first_test_date]
    print('First test date:', first_test_date.isoformat())
    print('Train size:', len(train_data))
    print('Test size:', len(test_data))

    check_model(LastRecordModel(), train_data, test_data, title='Just last available price')
