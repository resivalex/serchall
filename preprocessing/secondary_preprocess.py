from .preprocess import preprocess as primary_preprocess
import numpy as np


def preprocess(data):
    data = primary_preprocess(data)
    data = data.copy()
    renamings = {
        'Наименование': 'name',
        'Дата поставки': 'delivery_date',
        'Дата заказа': 'order_date',
        'Срок поставки': 'delivery_period',
        'Плановый срок поставки': 'planned_delivery_period',
        'Регион': 'region',
        'Объем заказа': 'amount',
        'Цена, руб': 'price',
        'Условия платежа': 'payment_conditions',
        'НРП - нерегламентная потребность (внеплановая закупка)': 'out_of_plan',
        'Поставщик': 'supplier',
        'Расчетная дата заказа': 'calculated_order_date',
        'Дата заказа фактическая': 'has_order_date',
        'Наименование3': 'cleaned_name',
        'Группа': 'group'
    }
    if list(data.columns) != list(renamings.keys()):
        raise Exception(
            'Unexpected set of columns.'
            f' {set(data.columns) - set(renamings.keys())}, {set(renamings.keys()) - set(data.columns)}'
        )
    data = data.rename(renamings, axis=1)
    data['out_of_plan'] = data['out_of_plan'] == 1.0
    data['delivery_period'] = data['delivery_period'].apply(lambda x: str(int(x)) if not np.isnan(x) else '')
    data['planned_delivery_period'] = data['planned_delivery_period'].apply(lambda x: str(int(x)) if not np.isnan(x) else '')
    # data['has_order_date'] = data['has_order_date'].fillna(False)
    # dates = sorted(data['calculated_order_date'].dropna())
    # median_date = dates[len(dates) // 2]
    # data['calculated_order_date'] = data['calculated_order_date'].fillna(median_date)
    # data['calculated_order_date'] = [t.date() for t in data['calculated_order_date']]
    data['order_date'] = [t.date() for t in data['order_date']]

    return data
