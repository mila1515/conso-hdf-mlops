"""Construction des variables explicatives (features).

Stratégie de prévision « directe » à horizon 24 h : toutes les variables de retard (lags)
ont un décalage >= 24 h. On peut donc prévoir les 24 prochaines heures en une seule passe,
sans boucle récursive et SANS fuite de données (data leakage).
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

TARGET = "consommation"
LOCAL_TZ = "Europe/Paris"

CALENDAR_FEATURES = [
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
LAG_FEATURES = ["lag_24h", "lag_48h", "lag_168h", "lag_336h", "roll_mean_24h", "roll_std_24h", "roll_mean_168h"]
WEATHER_FEATURES = ["temperature", "heating_degree", "temp_mean_24h"]


def feature_columns(use_weather: bool) -> list[str]:
    return CALENDAR_FEATURES + LAG_FEATURES + (WEATHER_FEATURES if use_weather else [])


def _calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert(LOCAL_TZ)
    fr_holidays = holidays.France(years=range(local.year.min(), local.year.max() + 1))
    df = pd.DataFrame(index=index)
    df["hour"] = local.hour
    df["dayofweek"] = local.dayofweek
    df["month"] = local.month
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)
    df["is_holiday"] = np.array([d in fr_holidays for d in local.date], dtype=int)
    df["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    return df


def build_features(
    conso: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    future_hours: int = 0,
) -> pd.DataFrame:
    """Renvoie un DataFrame horaire avec features (+ cible quand elle est connue).

    `future_hours` > 0 ajoute des lignes futures (cible inconnue) pour la prévision.
    """
    y = conso[TARGET].asfreq("1h")
    if future_hours:
        future_idx = pd.date_range(y.index.max() + pd.Timedelta(hours=1), periods=future_hours, freq="1h")
        y = pd.concat([y, pd.Series(np.nan, index=future_idx)])
        y.index.name = "date_heure"

    df = _calendar(y.index)
    df[TARGET] = y
    # Lags >= 24h uniquement (connus au moment de prévoir J+1)
    for h in (24, 48, 168, 336):
        df[f"lag_{h}h"] = y.shift(h)
    shifted = y.shift(24)
    df["roll_mean_24h"] = shifted.rolling(24, min_periods=18).mean()
    df["roll_std_24h"] = shifted.rolling(24, min_periods=18).std()
    df["roll_mean_168h"] = shifted.rolling(168, min_periods=120).mean()

    if weather is not None:
        temp = weather["temperature"].reindex(df.index)
        temp = temp.interpolate(limit=3, limit_area="inside")
        df["temperature"] = temp
        # Degrés-jours de chauffage : la conso augmente quand il fait froid (< 15 °C)
        df["heating_degree"] = (15 - temp).clip(lower=0)
        df["temp_mean_24h"] = temp.rolling(24, min_periods=18).mean()
    return df


def training_frame(conso: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    df = build_features(conso, weather)
    cols = feature_columns(weather is not None) + [TARGET]
    return df[cols].dropna()
