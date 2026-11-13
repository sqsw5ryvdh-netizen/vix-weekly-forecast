"""Integrita' dei dataset (holdout bloccato) e correttezza di metriche e baseline probabilistica."""
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from src import baselines as B
from src import config as C
from src import evaluate as E


@pytest.fixture(scope="module")
def data():
    read = lambda p: pd.read_csv(p, parse_dates=["date"])
    return read(C.CLEAN_FILE), read(C.DEV_FILE), read(C.HOLDOUT_FILE)


def test_holdout_is_last_104_weeks_and_disjoint(data):
    clean, dev, hold = data
    assert len(hold) == C.HOLDOUT_WEEKS
    assert len(dev) + len(hold) == len(clean)
    assert dev["date"].max() < hold["date"].min()
    assert set(dev["date"]).isdisjoint(hold["date"])
    pd.testing.assert_frame_equal(pd.concat([dev, hold], ignore_index=True), clean)


def test_clean_dataset_is_regular(data):
    clean, _, _ = data
    assert clean["date"].is_monotonic_increasing and clean["date"].is_unique
    assert (pd.to_datetime(clean["week_start"]).diff().dropna().dt.days == 7).all()
    assert (clean["close"] > 0).all() and clean["close"].notna().all()
    np.testing.assert_allclose(clean["log_close"], np.log(clean["close"]), atol=1e-6)


def test_raw_file_untouched_and_partial_week_excluded(data):
    clean, _, _ = data
    raw = pd.read_csv(C.RAW_FILE)
    partial = raw.loc[raw["note"].notna(), "last_trading_day"]
    assert not set(pd.to_datetime(partial)) & set(clean["date"])
    assert len(clean) == int(raw["note"].isna().sum())


def test_analysis_code_never_reads_holdout():
    """EDA, baseline e valutazione possono leggere solo il file dev."""
    for mod in ("eda.py", "baselines.py", "evaluate.py", "report.py"):
        src = (pathlib.Path(C.ROOT) / "src" / mod).read_text()
        assert not re.search(r"HOLDOUT_FILE|CLEAN_FILE|RAW_FILE|vix_weekly_holdout|vix_weekly_clean", src), mod


def test_generated_forecasts_stop_before_holdout(data):
    _, dev, hold = data
    path = C.REPORTS_DIR / "baseline_previsioni.csv"
    if not path.exists():
        pytest.skip("eseguire prima python -m src.run_phase0")
    fc = pd.read_csv(path, parse_dates=["data_origine", "data_obiettivo"])
    assert fc["data_obiettivo"].max() <= dev["date"].max() < hold["date"].min()
    assert (fc["data_origine"] < fc["data_obiettivo"]).all()


def test_diebold_mariano_sign_and_degenerate_case():
    rng = np.random.default_rng(C.SEED)
    e_good, e_bad = rng.standard_normal(500), 2 * rng.standard_normal(500)
    stat, p = E.diebold_mariano(e_bad, e_good, h=1)      # benchmark peggiore -> statistica positiva
    assert stat > 3 and p < 0.01
    stat2, _ = E.diebold_mariano(e_good, e_bad, h=4)
    assert stat2 < -2
    assert np.isnan(E.diebold_mariano(e_good, e_good, h=1)[0])


def test_point_metrics_known_values():
    g = pd.DataFrame({"y": [1.0, 2.0, 3.0, 2.0], "x0": [0.5, 2.5, 2.5, 2.0],
                      "f": [1.0, 2.0, 2.0, 2.5], "f_bench": [0.5, 2.5, 2.5, 2.0]})
    m = E.point_metrics(g, h=1)
    assert m["rmse_log"] == pytest.approx(0.5590169943749475)
    assert m["mae_log"] == pytest.approx(0.375)
    assert m["acc_direzionale"] == pytest.approx(2 / 3)   # la 4a riga (y = x0) e' esclusa; la 3a ha direzione sbagliata
    assert m["r2_oos_vs_rw"] == pytest.approx(1 - 1.25 / 0.75)


def test_markov_probability_on_known_chain():
    # alternanza perfetta 10 <-> 40: da "basso" si va sempre in "alto" e viceversa
    lv = np.array([10.0, 40.0] * 20)                     # termina a 40 (stato alto)
    assert B.prob_markov(lv, 1) == pytest.approx(0.0)
    assert B.prob_markov(lv, 2) == pytest.approx(1.0)
    assert B.prob_markov(lv, 4) == pytest.approx(1.0)
    assert B.prob_climatology(lv, 1) == pytest.approx(0.5)
    assert B.prob_persistence(lv, 1) == 1.0


def test_band_index_boundaries():
    assert list(B.band_index([14.99, 15.0, 19.99, 20.0, 25.0, 29.99, 30.0, 80.0])) == [0, 1, 1, 2, 3, 3, 4, 4]
