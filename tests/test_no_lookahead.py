"""Verifica che nessuna previsione usi dati successivi alla propria origine."""
import numpy as np
import pandas as pd
import pytest

from src import baselines as B
from src import config as C
from src import evaluate as E

ALL = [(n, f, "log") for n, f in B.MODELS.items()] + [(n, f, "level") for n, f in B.PROB_MODELS.items()]


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(C.SEED)
    x = np.empty(420)
    x[0] = 3.0
    for t in range(1, len(x)):                       # AR(1) sintetico attorno a log(20)
        x[t] = 3.0 + 0.92 * (x[t - 1] - 3.0) + 0.14 * rng.standard_normal()
    return x


@pytest.mark.parametrize("name,model,kind", ALL)
@pytest.mark.parametrize("h", C.HORIZONS)
def test_future_values_do_not_change_past_forecasts(series, name, model, kind, h):
    """Stravolgere i dati da k in poi non deve cambiare alcuna previsione con origine < k."""
    x = series if kind == "log" else np.exp(series)
    k = 350
    x2 = x.copy()
    x2[k:] = x2[k:] * 3.0 + 7.0
    o1, f1 = B.walk_forward(x, model, h, min_train=260)
    o2, f2 = B.walk_forward(x2, model, h, min_train=260)
    assert np.array_equal(o1, o2)
    before = o1 < k
    assert before.sum() > 50 and (~before).sum() > 10
    np.testing.assert_array_equal(f1[before], f2[before])
    if name not in ("climatologia",):                # controllo che il test "morda": dopo k le previsioni cambiano
        assert not np.allclose(f1[~before], f2[~before])


@pytest.mark.parametrize("h", C.HORIZONS)
def test_engine_passes_only_history(series, h):
    """Il motore passa al modello esattamente x[:t+1], in copia."""
    seen = []

    def spy(hist, hh):
        seen.append(len(hist))
        hist[:] = np.nan                             # un modello "maligno" non deve poter sporcare la serie
        return 0.0

    x = series.copy()
    origins, _ = B.walk_forward(x, spy, h, min_train=260)
    assert seen == [t + 1 for t in origins]
    assert origins[0] == 259 and origins[-1] == len(x) - h - 1
    assert not np.isnan(x).any()


def test_har_features_are_causal(series):
    full = B.har_features(series)
    part = B.har_features(series[:300])
    np.testing.assert_allclose(full[: len(part)], part)
    i = 200                                          # controllo esplicito delle medie mobili
    np.testing.assert_allclose(full[i - 11], [1, series[i], series[i - 3:i + 1].mean(), series[i - 11:i + 1].mean()])


def test_forecast_table_alignment(series):
    """Nella tabella finale: obiettivo = origine + h righe, y = valore realizzato, RW = valore all'origine."""
    dates = pd.date_range("2000-01-07", periods=len(series), freq="W-FRI")
    df = pd.DataFrame({"date": dates, "log_close": series, "close": np.exp(series)})
    fc = E.build_forecasts(df)
    pos = pd.Series(np.arange(len(df)), index=dates)
    for h in C.HORIZONS:
        g = fc[(fc["h"] == h) & (fc["tipo"] == "puntuale")]
        o, t = pos[g["data_origine"]].to_numpy(), pos[g["data_obiettivo"]].to_numpy()
        assert (t - o == h).all() and o.min() == C.MIN_TRAIN_WEEKS - 1
        np.testing.assert_allclose(g["y"], series[t])
        np.testing.assert_allclose(g["x0"], series[o])
        rw = g[g["modello"] == "random_walk"]
        np.testing.assert_allclose(rw["f"], rw["x0"])
