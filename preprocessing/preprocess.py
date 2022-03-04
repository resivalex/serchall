import pendulum
import re


def _parse_date(s):
    try:
        if s == '':
            return
        if re.match('^\d{2}\.\d{2}\.\d{4}$', s):
            return pendulum.from_format(s, 'DD.MM.YYYY').date()

        return pendulum.parse(s).date()
    except:
        print(s)
        raise


def preprocess(data):
    for str_field in ['name', 'region', 'payment_conditions']:
        data[str_field] = [s.lower() for s in data[str_field]]
    data['delivery_date'] = [_parse_date(s) for s in data['delivery_date']]
    data['order_date'] = [_parse_date(s) for s in data['order_date']]
    for int_field in ['delivery_period', 'planned_delivery_period', 'amount', 'supplier']:
        data[int_field] = [(int(s) if s != '' else None) for s in data[int_field]]
    data['price'] = data['price'].astype(float)
    data['out_of_plan'] = (data['out_of_plan'] != '').astype(int)
    return data
