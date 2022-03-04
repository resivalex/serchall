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
    data['name'] = [s.lower() for s in data['name']]
    data['delivery_date'] = [_parse_date(s) for s in data['delivery_date']]
    data['order_date'] = [_parse_date(s) for s in data['order_date']]
    for int_field in ['delivery_period', 'planned_delivery_period', 'amount', 'supplier']:
        data[int_field] = [(int(s) if s != '' else None) for s in data[int_field]]
    data['price'] = data['price'].astype(float)
    data['payment_conditions'] = [s.lower() for s in data['payment_conditions']]
    data['out_of_plan'] = (data['out_of_plan'] != '').astype(int)
    return data
