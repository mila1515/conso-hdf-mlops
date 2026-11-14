import numpy as np

from conso_hdf.features import TARGET, build_features, feature_columns, training_frame
from conso_hdf.metrics import regression_metrics
from tests.conftest import make_synthetic_conso


def test_lags_have_no_leakage():
    conso = make_synthetic_conso(days=30)
    df = build_features(conso)
    t = df.index[400]
    assert df.loc[t, "lag_24h"] == conso.loc[t - np.timedelta64(24, "h"), TARGET]
    assert df.loc[t, "lag_168h"] == conso.loc[t - np.timedelta64(168, "h"), TARGET]


def test_future_rows_have_features_but_no_target():
    conso = make_synthetic_conso(days=30)
    df = build_features(conso, future_hours=24)
    future = df.tail(24)
    assert future[TARGET].isna().all()
    assert future["lag_24h"].notna().all()  # lags >= 24h => connus pour tout l'horizon
    assert future.index.min() == conso.index.max() + np.timedelta64(1, "h")


def test_training_frame_columns():
    df = training_frame(make_synthetic_conso(days=30), None)
    assert list(df.columns) == feature_columns(False) + [TARGET]
    assert not df.isna().any().any()


def test_metrics():
    m = regression_metrics([100, 200], [110, 180])
    assert m["mae"] == 15
    assert round(m["mape"], 2) == 10.0
