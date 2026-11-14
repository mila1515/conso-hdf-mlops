import numpy as np
import pandas as pd

from conso_hdf.monitoring import purge_old_reports, report_path, seasonal_reference, wasserstein_norm


def test_wasserstein_norm_detects_shift_only():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(10, 2, 2000))
    same = pd.Series(rng.normal(10, 2, 500))
    shifted = pd.Series(rng.normal(14, 2, 500))  # +2 écarts-types
    assert wasserstein_norm(ref, same) < 0.2
    assert wasserstein_norm(ref, shifted) > 1.5


def test_seasonal_reference_uses_previous_years_only():
    idx = pd.date_range("2022-01-01", "2026-11-14 23:00", freq="1h", tz="UTC")
    feats = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)
    current = feats[feats.index > "2026-11-07"]
    ref = seasonal_reference(feats, current)
    assert set(ref.index.year) == {2022, 2023, 2024, 2025}
    # Uniquement la même période de l'année (± 3 semaines autour de mi-novembre)
    assert set(ref.index.month) <= {10, 11, 12}


def test_report_retention_and_relative_path(cfg):
    for i in range(cfg.reports_keep + 5):
        (cfg.reports_dir / f"drift_report_20260101_{i:06d}.html").write_text("x")
    purge_old_reports(cfg)
    remaining = sorted(cfg.reports_dir.glob("drift_report_*.html"))
    assert len(remaining) == cfg.reports_keep
    assert remaining[-1].name.endswith(f"{cfg.reports_keep + 4:06d}.html")
    # Ancien format (chemin absolu d'un autre environnement, ex. Docker) -> résolu dans REPORTS_DIR
    assert report_path(cfg, "/app/reports/drift_report_x.html") == cfg.reports_dir / "drift_report_x.html"
