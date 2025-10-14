import pandas as pd
import holidays

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    fr_holidays = holidays.France()
    df = df.copy()
    
    df["datetime"] = pd.to_datetime(df["id"], format="%Y-%m-%d %H")
    df["date"] = df["datetime"].dt.date

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["day_of_week"] = df["datetime"].dt.dayofweek  # Monday=0
    df["hour"] = df["datetime"].dt.hour
    df["is_weekend"] = df["datetime"].dt.dayofweek >= 5
    df["IsHoliday"] = df["datetime"].dt.date.apply(lambda x: x in fr_holidays)

    return df

def add_covid_stringency_index(df: pd.DataFrame, path: str) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(pd.to_datetime(df["id"], format="%Y-%m-%d %H").dt.date)
    min_date, max_date = df["date"].min(), df["date"].max()

    covid_index = pd.read_csv(path)
    covid_index["date"] = pd.to_datetime(covid_index["Date"], format="%Y%m%d")

    covid_index = (
        covid_index
        .query("CountryName == 'France'")
        .query("date >= @min_date and date <= @max_date")
        [["date", "StringencyIndex_Average"]]
    )

    df = df.merge(covid_index, on="date", how="left").fillna(0).drop(columns='date')
    return df

