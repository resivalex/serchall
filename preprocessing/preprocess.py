import pandas as pd
from datetime import datetime
import re


def preprocess(df):
    # parse delivery dates
    def date_parse(s):
        if isinstance(s, datetime):
            return s
        elif isinstance(s, str):
            return datetime.strptime(s, '%d.%m.%Y')
        return None

    df['Дата поставки'] = df['Дата поставки'].apply(date_parse)

    # clean names
    df['Наименование2'] = df.Наименование.apply(
        lambda s: s.lower().strip().replace("  ", "").replace('.', '-')
    )
    # clean names
    sub_dict = {
        r"(зуб)\s(.*)\s*(ковша)\s*(.+)": r"\1 \3 \2 \4",
        ",* *с наплавкой": " наплавка",
        "венец зубчатый": "венец зубч",
        r"блоки голов-": "блоки голов",
        r"блоки головные": "блоки голов",
        r"блоки отклон-": "блоки отклон",
        r"блок-шестерня(\d)": r"блок-шестерня \1",
        r"блок(\d)": "блок \\1",  # TODO: "блок3"
        r"блок3": "блок 3",
        "блок управления": "блок упр",
        r"блок,": "блок",
        "блоки ": "блок ",
        "блок головной": "блок голов",
        "венец3": "венец 3",
        r"венец(\d)": "венец \\1",
        r"втулка(\d)": "втулка \\1",
        # TODO: "днище"
        r"засов(\d)": "засов \\1",
        r"звено(\d)": "звено \\1",
        r"зуб(\d)": "зуб \\1",
        r"ковш(\d)": "ковш \\1",
        r"колесо(\d)": "колесо \\1",
        r"^([а-яa-z]+)(\d)": "\\1 \\2",
        # TODO: колесо
        "- ": " ",
        "   ": " ",
        "  ": " ",
        " ,": ",",
        r",(\S)": r", \1",
    }

    df['Наименование3'] = df.Наименование2

    for k, v in sub_dict.items():
        df.Наименование3 = df.Наименование3.apply(lambda s: re.sub(k, v, s))

    # group items
    group_dict = {
        "амортизатор \d": "амортизатор",
        "барабан \d": "барабан",
        "блок \d": "блок",
        "блок голов": "блок голов",
        "блок трмз": "блок трмз",
        "блок упр": "блок упр",
        "блок-шес": "блок-шестерня",
        "болт \d": "болт",
        "бронь конуса": "бронь конуса",
        "вал \d": "вал",
        "вал веду": "вал ведущий",
        "вал промеж": "вал промежуточный",
        "вал трмз": "вал трмз",
        "вал-шест": "вал-шестерня",
        "вант ": None,
        "венец зубч": "венец зубч",
        "венец \d": "венец",
        "вентилятор": None,
        "винт": None,
        "вкладыш": None,
        "водило": None,
        "втулка \d": "втулка",
        "втулка бронз": None,
        "втулка колеса": None,
        "втулка напорной оси": None,
        "втулка сзсм": None,
        "втулка трубы": None,
        "г/цил": None,
        "гайка": None,
        "джойстик": None,
        "днище": None,
        "засов": None,
        "звездочка": None,
        "звено": None,
        "зуб": None,
        "изолятор": None,
        "клин": None,
        "ковш": None,
        "колесо": None
    }

    df.drop(columns="Наименование2", inplace=True)

    df['Группа'] = None
    for k, v in group_dict.items():
        df.Группа = df.apply(
            lambda r:
            (v if v else k)
            if (
                    re.search(k, r.Наименование3)
                    and re.search(k, r.Наименование3).start() == 0
            )
            else r.Группа,
            axis=1
        )

    # Can't use rows with empty delivery date.
    df = df[df['Дата поставки'].notnull()]

    return df


if __name__ == '__main__':
    data_in = pd.read_excel('../ini_data/datamon.xlsx')
    data_out = preprocess(data_in)
    print(data_out.sample(10))
