import pandas as pd
import holidays
import numpy as np

def cyclical_encoding(col):

    if not isinstance(col, pd.Series):
        raise TypeError("Input must be a pandas Series")

    max_val = col.max()
    scaled = (col * 2 * np.pi) / max_val

    return np.sin(scaled), np.cos(scaled)

def get_features(data: pd.DataFrame, idcol: str = "id", encode_cyclical: bool = True):

    fr_holidays = holidays.France()
    df = data.copy()

    df[idcol] = pd.to_datetime(df[idcol])
    df["year"] = df[idcol].dt.year
    df["month"] = df[idcol].dt.month
    df["hour"] = df[idcol].dt.hour
    df["weekday"] = df[idcol].dt.dayofweek
    df["is_weekend"] = df[idcol].dt.dayofweek >= 5
    df["is_holiday"] = df[idcol].dt.date.apply(lambda x: x in fr_holidays)

    if encode_cyclical:
        cyclical_features = ["month", "weekday", "hour"]
        for col in cyclical_features:
            df[f"{col}_sin"], df[f"{col}_cos"] = cyclical_encoding(df[col])
        df.drop(columns=cyclical_features, inplace=True)

    return df.sort_values(by=idcol, ascending=True)