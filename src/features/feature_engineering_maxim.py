import pandas as pd
import numpy as np
from src.features.feature_engineering_maxim import *
from datetime import datetime
import pytz

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from lightgbm import LGBMRegressor

import holidays


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


######

def fill_weather_gaps(df_weather: pd.DataFrame,
                      start=None,
                      end=None) -> pd.DataFrame:
    """
    Fill missing rows and interpolate raw Meteostat hourly data (no timezone ops).

    Expected columns (after dropping unused ones):
      ['time','temp','dwpt','rhum','prcp','wdir','wspd','pres']

    Output:
      Same columns, hourly-continuous, 'time' stays naive local datetime column.
    """

    df = df_weather.copy()

    # 1) Parse and sort; keep only valid datetimes
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    # 2) Handle duplicate timestamps (e.g., DST fall-back): keep first occurrence
    df = df[~df["time"].duplicated(keep="first")]

    # 3) Build full hourly index (naive local)
    if start is None:
        start = df["time"].min().floor("H")
    else:
        start = pd.to_datetime(start).floor("H")
    if end is None:
        end = df["time"].max().ceil("H")
    else:
        end = pd.to_datetime(end).ceil("H")

    full_idx = pd.date_range(start, end, freq="H")
    df = df.set_index("time").reindex(full_idx)

    # 4) Interpolate continuous meteorological variables
    cont_cols = ["temp", "dwpt", "rhum", "wdir", "wspd", "pres"]
    for col in cont_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="time", limit_direction="both")

    # 5) Precipitation: missing -> 0 (safe)
    if "prcp" in df.columns:
        df["prcp"] = df["prcp"].fillna(0)

    # 6) Final small clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill(limit=3).bfill(limit=3)

    # 7) Return with 'time' as column
    df = df.reset_index().rename(columns={"index": "time"})

    # Reorder to original columns if possible
    cols_order = [c for c in ["time","temp","dwpt","rhum","prcp","wdir","wspd","pres"] if c in df.columns]
    df = df[cols_order + [c for c in df.columns if c not in cols_order]]

    return df



def enrich_weather_features(df_weather: pd.DataFrame,
                            tz_local: str = "Europe/Paris") -> pd.DataFrame:
    """
    Enrich hourly weather data for air quality modeling/imputation.

    Parameters
    ----------
    df_weather : pd.DataFrame
        Cleaned Meteostat dataframe with columns:
        ['time','temp','dwpt','rhum','prcp','wdir','wspd','pres']
        (no tsun, coco, wpgt, or snow)
    tz_local : str
        Local timezone for solar calculations (default: Europe/Paris)

    Returns
    -------
    pd.DataFrame
        DataFrame with additional physically meaningful features,
        keeping 'time' as a naive local datetime column.
    """

    df = df_weather.copy()
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # --- Wind features
    wdir = df["wdir"].fillna(0)
    wdir_rad = np.deg2rad(wdir % 360)
    df["wind_u"] = -df["wspd"] * np.sin(wdir_rad)
    df["wind_v"] = -df["wspd"] * np.cos(wdir_rad)

    # Wind sector (optional categorical)
    wdir_normalized = wdir % 360
    sector_idx = ((wdir_normalized + 22.5) // 45) % 8
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    df["wind_sector"] = pd.Categorical([labels[int(i)] for i in sector_idx], categories=labels, ordered=False)

    # --- Humidity physics
    es = 0.6108 * np.exp(17.27 * df["temp"] / (df["temp"] + 237.3))
    ea = 0.6108 * np.exp(17.27 * df["dwpt"] / (df["dwpt"] + 237.3))
    df["vpd"] = (es - ea).clip(lower=0)
    df["abs_humidity"] = 216.7 * (ea / (df["temp"] + 273.15))

    # --- Rain features
    df["is_wet_hour"] = (df["prcp"].fillna(0) > 0).astype(int)
    df["rain_3h_sum"] = df["prcp"].rolling(3, min_periods=1).sum()

    # --- Pressure anomaly (7-day rolling)
    df["pres_anom_7d"] = df["pres"] - df["pres"].rolling(24 * 7, min_periods=12).mean()

    # --- Solar geometry (approximation for Paris)
    lat = np.deg2rad(48.8566)
    # temporary tz-aware index for day-of-year and hour
    df_local = df.copy()
    df_local["time_tz"] = pd.to_datetime(df_local["time"]).dt.tz_localize(tz_local, ambiguous="NaT", nonexistent="shift_forward")
    doy = df_local["time_tz"].dt.dayofyear.values
    hour = df_local["time_tz"].dt.hour.values + df_local["time_tz"].dt.minute.values / 60.0

    decl = 23.44 * np.pi / 180 * np.sin(2 * np.pi * (284 + doy) / 365)
    h_angle = np.pi / 12 * (hour - 12)
    solar_elev = np.arcsin(np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(h_angle))
    df["solar_elev_sin"] = np.sin(np.clip(solar_elev, 0, np.pi/2))
    df["is_daylight"] = (solar_elev > 0).astype(int)

    # --- Cleanup
    df = df.replace([np.inf, -np.inf], np.nan).ffill(limit=3)

    # --- Final selection of relevant features
    keep_cols = [
        "time", "temp", "rhum", "pres",
        "wind_u", "wind_v",
        "prcp", "is_wet_hour", "rain_3h_sum",
        "vpd", "abs_humidity",
        "solar_elev_sin", "is_daylight"
    ]

    return df[keep_cols]


def merge_weather(main: pd.DataFrame, weather: pd.DataFrame, main_col: str = "id", weather_col: str = "time") -> pd.DataFrame:

    assert main[main_col].duplicated().sum() == 0, "Non-unique dates in main"
    assert weather[weather_col].duplicated().sum() == 0, "Non-unique dates in weather"
    assert len(set(main[main_col]) - set(weather[weather_col])) == 0, "Weather does not cover all hours"
    

    output = main.merge(weather, left_on=main_col, right_on=weather_col, how="inner")

    return output



# from meteostat import Hourly, Stations
# import pandas as pd
# from datetime import datetime

# start = datetime(2020,1,1)
# end = datetime(2025,9,30)

# # Find Paris station
# station = Stations().nearby(48.8566, 2.3522).fetch(1)
# data = Hourly(station, start, end).fetch()
# data.to_csv("2paris_meteostat_2020_2025.csv")


#####################



def make_lagged_features(df: pd.DataFrame,
                         pollutants: list = ["valeur_NO2","valeur_CO","valeur_O3","valeur_PM10","valeur_PM25"],
                         weather_vars=None,
                         lags=(1, 3, 6, 12, 24),
                         rolls=(3, 6, 12, 24)) -> pd.DataFrame:

    df = df.copy().sort_values("id" if "id" in df.columns else "time")

    for col in pollutants:
        if col not in df.columns:
            continue

        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

        for win in rolls:
            roll = df[col].rolling(window=win, min_periods=1)
            df[f"{col}_roll{win}_mean"] = roll.mean()
            df[f"{col}_roll{win}_std"] = roll.std()

    df = df.loc[:, ~df.columns.duplicated()]

    return df


def impute_pollutants(df: pd.DataFrame, pollutants, weather, cal, base):

    df = df.copy()

    for p in pollutants:
        df[p + "_missing"] = df[p].isna().astype(int)

    lag_roll_cols = []
    for p in pollutants:
        lag_roll_cols += [c for c in df.columns if c.startswith(p + "_lag")]
        lag_roll_cols += [c for c in df.columns if c.startswith(p + "_roll")]

    cols = pollutants + weather + cal + lag_roll_cols + [p+"_missing" for p in pollutants]
    cols = [c for c in cols if c in df.columns]

    X = df[cols].astype(float)

    print(f"Input cols: {X.columns}")

    imp = IterativeImputer(
        estimator=base,
        max_iter=15,
        sample_posterior=False,  
        initial_strategy="median",
        skip_complete=True,
        random_state=1,
        verbose=1
    )

    X_imp = imp.fit_transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=X.columns, index=X.index)

    df_imp = df.copy()
    for p in pollutants:
        mask = df_imp[p].isna()
        df_imp.loc[mask, p] = X_imp_df.loc[mask, p]
        df_imp[p] = df_imp[p].clip(lower=0)

    print(df_imp[pollutants].isna().sum())

    return df_imp, imp