"""Fase 1: anti look-ahead per tutti i modelli nuovi, blocco dell'holdout, correttezza di modelli e test statistici."""
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from src import baselines as B
from src import config as C
from src.fase1 import comune as K
from src.fase1 import modelli as M
from src.fase1 import pattern as P
from src.fase1 import valuta as V

ALL_MODELS = list((M.MODELS | M.SENSITIVITY).items())
FASE1_DIR = pathlib.Path(C.ROOT) / "src" / "fase1"


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(C.SEED + 1)
    x = np.empty(420)
    x[0] = 3.0
    for t in range(1, len(x)):
        x[t] = 3.0 + 0.93 * (x[t - 1] - 3.0) + 0.14 * rng.standard_normal()
    return x


# ------------------------------------------------------------------ anti look-ahead

@pytest.mark.parametrize("name,model", ALL_MODELS)
@pytest.mark.parametrize("h", K.HORIZONS)
def test_future_values_do_not_change_past_forecasts(series, name, model, h):
    """Stravolgere i dati da k in poi non deve cambiare alcuna previsione con origine < k."""
    k = 350
    x2 = series.copy()
    x2[k:] = 3.0 + 2.5 * (x2[k:] - 3.0) + 0.4
    o1, f1 = B.walk_forward(series, model, h, min_train=260)
    o2, f2 = B.walk_forward(x2, model, h, min_train=260)
    assert np.array_equal(o1, o2)
    before = o1 < k
    assert before.sum() > 50 and (~before).sum() > 10
    np.testing.assert_array_equal(f1[before], f2[before])
    assert not np.allclose(f1[~before], f2[~before])          # il test "morde": dopo k le previsioni cambiano


@pytest.mark.parametrize("name,model", ALL_MODELS)
def test_models_do_not_modify_history(series, name, model):
    hist = series[:300].copy()
    model(hist, 4)
    np.testing.assert_array_equal(hist, series[:300])


def test_table_model_matches_bruteforce(series):
    """M5 ricostruito con un ciclo esplicito: solo casi con esito noto (i + h <= t), soglia di 20 casi."""
    for t in (270, 333, 419):
        for h in K.HORIZONS:
            hist = series[: t + 1]
            lv = np.exp(hist)
            cell, band = [], []
            b0, d0 = K.band7(lv[t]), K.direction(hist[t] - hist[t - 4])
            for i in range(4, t + 1):
                if i + h > t:
                    continue
                if K.band7(lv[i]) == b0:
                    band.append(hist[i + h] - hist[i])
                    if K.direction(hist[i] - hist[i - 4]) == d0:
                        cell.append(hist[i + h] - hist[i])
            use = cell if len(cell) >= M.MIN_CELL else band
            exp = hist[t] + (np.median(use) if use else 0.0)
            assert M.forecast_table(hist, h) == pytest.approx(exp)


def test_analog_model_matches_bruteforce(series):
    """M6 ricostruito con un ciclo esplicito: stati standardizzati sulla storia, candidati con i + h <= t."""
    for t in (270, 419):
        for h in K.HORIZONS:
            hist = series[: t + 1]
            st = {}
            for i in range(12, t + 1):
                d = np.diff(hist[i - 12: i + 1])
                st[i] = np.array([hist[i], hist[i] - hist[i - 4], hist[i] - hist[i - 12], d.std(ddof=1)])
            S = np.array(list(st.values()))
            z = {i: (v - S.mean(axis=0)) / S.std(axis=0) for i, v in st.items()}
            cand = [i for i in st if i + h <= t]
            cand.sort(key=lambda i: (np.sqrt(((z[i] - z[t]) ** 2).sum()), i))
            exp = hist[t] + np.median([hist[i + h] - hist[i] for i in cand[: M.K_ANALOG]])
            assert M.forecast_analog(hist, h) == pytest.approx(exp)


def test_har_ext_and_rolling_definitions(series):
    hist = series[:400]
    assert M.forecast_har_roll(hist[:300], 4) == pytest.approx(B.forecast_har(hist[:300], 4))   # storia < 520: uguale a M1
    long = np.concatenate([series, series[::-1], series])                                    # storia > 520
    assert M.forecast_har_roll(long, 4) == pytest.approx(B.forecast_har(long[-520:], 4))
    assert M.forecast_har_roll(long, 4) != pytest.approx(B.forecast_har(long, 4))
    assert M.forecast_combo(hist, 4) == pytest.approx(
        np.mean([M.forecast_arma(hist, 4), M.forecast_har_ext(hist, 4), M.forecast_analog(hist, 4)]))


def test_model_list_is_closed():
    assert list(M.MODELS) == ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]
    assert (M.ROLL_WINDOW, M.MIN_CELL, M.K_ANALOG) == (520, 20, 30)
    assert K.HORIZONS == (1, 4, 12) and K.BANDS7 == (12.0, 15.0, 20.0, 25.0, 30.0, 40.0) and K.DIR_THR == 0.10


# ------------------------------------------------------------------ blocco dell'holdout

def test_phase1_modules_never_reference_holdout_or_clean():
    """Fallisce se un modulo di Fase 1 nomina il file di holdout, il file clean o il file grezzo.

    Solo `comune.py` puo' nominarli, e solo per costruire la guardia che ne vieta l'apertura.
    """
    banned = r"HOLDOUT_FILE|CLEAN_FILE|RAW_FILE|vix_weekly_holdout|vix_weekly_clean|VIX_settimanale"
    files = sorted(FASE1_DIR.glob("*.py")) + [pathlib.Path(C.ROOT) / "src" / "run_phase1.py"]
    assert len(files) >= 5
    for f in files:
        src = f.read_text()
        if f.name == "comune.py":
            assert re.findall(banned, src) == ["HOLDOUT_FILE", "CLEAN_FILE", "RAW_FILE"], "comune.py: usi non previsti"
            assert src.count("read_csv(") == 1 and "read_csv(C.DEV_FILE" in src
        else:
            assert not re.search(banned, src), f.name
            assert "DEV_FILE" not in src, f"{f.name}: i dati si leggono solo con comune.load_dev()"
            assert not re.search(r"read_csv\((?!R / |K\.OUT / )", src), f"{f.name}: read_csv ammesso solo sui file di risultati"


def test_guard_blocks_holdout_clean_and_raw_at_runtime():
    with K.guardia_dati():
        for path in (C.HOLDOUT_FILE, C.CLEAN_FILE, C.RAW_FILE):
            with pytest.raises(PermissionError):
                open(path)
            with pytest.raises(PermissionError):
                pd.read_csv(path)
        assert len(K.load_dev()) > 1000                 # il file dev resta leggibile


def test_run_phase1_runs_under_guard():
    src = (pathlib.Path(C.ROOT) / "src" / "run_phase1.py").read_text()
    assert "with K.guardia_dati():" in src


def test_dev_loader_drops_open_and_high_low_only_in_a2():
    assert "open" not in K.load_dev().columns
    for f in FASE1_DIR.glob("*.py"):
        src = f.read_text()
        assert not re.search(r"\[\s*[\"']open[\"']\s*\]", src.replace('drop(columns=["open"', "")), f.name
        if f.name != "pattern.py":
            assert not re.search(r"[\"'](high|low)[\"']", src), f"{f.name}: high/low ammessi solo in A2"
    pat = (FASE1_DIR / "pattern.py").read_text()
    assert len(re.findall(r'df\["high"\]|df\["low"\]', pat)) == 2      # una sola riga, dentro week_measures (A2)


def test_generated_phase1_forecasts(series):
    path = K.OUT / "b_previsioni.csv"
    if not path.exists():
        pytest.skip("eseguire prima python -m src.run_phase1")
    fc = pd.read_csv(path, parse_dates=["data_origine", "data_obiettivo"])
    dev_end = K.load_dev()["date"].max()
    assert fc["data_obiettivo"].max() <= dev_end and (fc["data_origine"] < fc["data_obiettivo"]).all()
    np.testing.assert_allclose(fc["f_M0"], fc["x0"], atol=1e-6)
    p0 = pd.read_csv(C.REPORTS_DIR / "baseline_previsioni.csv", parse_dates=["data_origine"])
    p0 = p0[(p0["modello"] == "har")][["h", "data_origine", "f"]]
    m = fc.merge(p0, on=["h", "data_origine"])
    assert len(m) == ((fc["h"] == 1) | (fc["h"] == 4)).sum()
    np.testing.assert_allclose(m["f_M1"], m["f"], atol=2e-6)       # M1 e' esattamente l'HAR della Fase 0


# ------------------------------------------------------------------ statistica e definizioni

def test_holm_known_values():
    np.testing.assert_allclose(V.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(V.holm([0.5, 0.9]), [1.0, 1.0])


def test_clark_west_detects_useful_regressor():
    rng = np.random.default_rng(C.SEED)
    z = rng.standard_normal(600)
    y = 0.4 * z + rng.standard_normal(600)
    f_small, f_big = np.zeros(600), 0.4 * z
    stat, p = V.clark_west(y - f_small, y - f_big, f_small, f_big, h=1)
    assert stat > 3 and p < 0.01
    noise = 0.4 * rng.standard_normal(600)             # regressore inutile: nessun rifiuto sistematico
    stat2, p2 = V.clark_west(y - f_small, y - noise, f_small, noise, h=4)
    assert p2 > 0.01
    assert np.isnan(V.clark_west(y, y, f_small, f_small, h=1)[0])


def _frame(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"date": pd.date_range("2000-01-07", periods=len(close), freq="W-FRI"), "close": close,
                         "log_close": np.log(close), "high": close * 1.05, "low": close * 0.95})


def test_episode_definition_merges_short_gaps():
    c = np.full(120, 18.0)
    c[40:43] = [31, 45, 33]          # episodio 1
    c[47] = 32                       # distanza 4 settimane -> unito
    c[80] = 35                       # distanza > 4 -> nuovo episodio
    ep = P.episodes(_frame(c), "soglia30")
    assert list(zip(ep["s"], ep["e"], ep["p"])) == [(40, 47, 41), (80, 80, 80)]
    assert ep["base"].tolist() == [18.0, 18.0] and ep["classe"].tolist() == ["40-50", "30-40"]
    assert ep.loc[0, "sett_recupero_100"] == 2 and ep.loc[0, "sett_sotto_25"] == 2   # 43 -> 33, poi 18
    rel = P.episodes(_frame(c), "relativa")       # 1,5 x 18 = 27
    assert rel["s"].tolist()[0] == 40


def test_calm_period_definition_requires_eight_weeks():
    c = np.full(60, 20.0)
    c[5:12] = 13.0                   # 7 settimane: non basta
    c[20:28] = 13.0                  # 8 settimane: periodo
    per = P.calm_periods(_frame(c), "sotto15")
    assert list(zip(per["s"], per["e"], per["durata"])) == [(20, 27, 8)]
    assert per.loc[0, "var_1"] == pytest.approx(np.log(20 / 13)) and per.loc[0, "uscita_con_rialzo_oltre_20pct"] == 1.0


def test_bootstrap_is_reproducible_and_covers_estimate():
    v = np.random.default_rng(1).standard_normal(80)
    a = K.boot_iid(len(v), lambda i: np.median(v[i]), "prova")
    b = K.boot_iid(len(v), lambda i: np.median(v[i]), "prova")
    assert a == b and a[0] < np.median(v) < a[1]
