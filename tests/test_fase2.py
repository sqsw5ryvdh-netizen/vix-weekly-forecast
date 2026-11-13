"""Fase 2: anti look-ahead per i modelli di densita' e per l'allineamento dei dati esterni, blocco dell'holdout, metriche."""
import pathlib
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from src import baselines as B
from src import config as C
from src import fetch_external as FE
from src.fase1 import comune as K
from src.fase1 import modelli as M
from src.fase2 import densita as D
from src.fase2 import scheda as SC
from src.fase2 import valuta as V

FASE2_DIR = pathlib.Path(C.ROOT) / "src" / "fase2"
N, CUT = 440, 405


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(C.SEED + 2)
    x = np.empty(N)
    x[0] = 3.0
    for t in range(1, N):
        x[t] = 3.0 + 0.93 * (x[t - 1] - 3.0) + 0.14 * rng.standard_normal()
    return x


def pipeline(x, name, h):
    """Intera catena: centri walk-forward, densita', quantili e probabilita' degli eventi."""
    c1 = D.centers(x, M.forecast_har, h)
    c7 = D.centers(x, M.forecast_combo, h) if name == "D1b" else None
    o, ds = D.density_walk(x, name, h, c1, c7)
    Q = np.array([d.quantiles() for d in ds])
    P = np.array([[d.prob_above(D.event_threshold(e, x[t])) for e in D.events_for(h)] for d, t in zip(ds, o)])
    return o, Q, P


# ------------------------------------------------------------------ anti look-ahead: densita'

@pytest.mark.parametrize("name", list(D.DENSITY))
@pytest.mark.parametrize("h", K.HORIZONS)
def test_density_future_values_do_not_change_past_forecasts(series, name, h):
    if name == "D5" and h not in D.D5_HORIZONS:
        assert D.d5(D.contesto(series, 400, h, D.centers(series, M.forecast_har, h), None)) is None
        return
    x2 = series.copy()
    x2[CUT:] = 3.0 + 2.5 * (x2[CUT:] - 3.0) + 0.4
    o1, q1, p1 = pipeline(series, name, h)
    o2, q2, p2 = pipeline(x2, name, h)
    assert np.array_equal(o1, o2) and o1[0] == D.first_origin(h)
    before = o1 < CUT
    assert before.sum() >= 25 and (~before).sum() >= 10
    np.testing.assert_array_equal(q1[before], q2[before])
    np.testing.assert_array_equal(p1[before], p2[before])
    assert not np.allclose(q1[~before], q2[~before])           # il test "morde"


@pytest.mark.parametrize("h", K.HORIZONS)
def test_event_models_are_causal(series, h):
    x2 = series.copy()
    x2[CUT:] += 1.0
    for e in D.events_for(h):
        for t in (380, 404):
            assert D.e1(series[: t + 1], h, e) == D.e1(x2[: t + 1], h, e)
            assert D.reference_prob(series[: t + 1], h, e) == D.reference_prob(x2[: t + 1], h, e)
        assert 0.0 <= D.e1(series[:420], h, e) <= 1.0


def test_context_exposes_only_known_errors(series):
    """Il contesto contiene solo storia e centri fino a t; gli errori usabili hanno esito x[i+h] con i + h <= t."""
    h, t = 4, 400
    c1 = D.centers(series, M.forecast_har, h)
    ctx = D.contesto(series, t, h, c1, None)
    assert len(ctx.hist) == t + 1 and len(ctx.c1) == t - D.O0 + 1
    i, e = ctx.past(ctx.c1)
    assert i[0] == D.O0 and i[-1] == t - h and (i + h <= t).all()
    manual = np.array([series[k + h] - B.forecast_har(series[: k + 1], h) for k in range(D.O0, t - h + 1)])
    np.testing.assert_allclose(e, manual)
    np.testing.assert_allclose(D.d1(ctx).quantiles(), B.forecast_har(series[: t + 1], h) + np.quantile(manual, D.TAUS))
    ctx.hist[:] = 0                                             # il contesto e' una copia: la serie non cambia
    assert series[0] == 3.0


def test_first_origin_requires_min_errors():
    for h in K.HORIZONS:
        t = D.first_origin(h)
        assert (t - h) - D.O0 + 1 == D.MIN_ERR == 104


def test_conditional_rules_match_bruteforce(series):
    """D2 e D3: regola dei 20 casi ricostruita con un ciclo esplicito."""
    for h in (1, 4):
        c1 = D.centers(series, M.forecast_har, h)
        for t in (370, 420, 439 - h):
            ctx = D.contesto(series, t, h, c1, None)
            b0, d0 = K.band7(np.exp(series[t])), K.direction(series[t] - series[t - 4])
            cell, band, allv = [], [], []
            for i in range(D.O0, t - h + 1):
                err = series[i + h] - c1[i - D.O0]
                allv.append(err)
                if K.band7(np.exp(series[i])) == b0:
                    band.append(err)
                    if K.direction(series[i] - series[i - 4]) == d0:
                        cell.append(err)
            exp2 = band if len(band) >= D.MIN_CELL else allv
            exp3 = cell if len(cell) >= D.MIN_CELL else exp2
            np.testing.assert_allclose(D.d2(ctx).quantiles(), c1[t - D.O0] + np.quantile(exp2, D.TAUS))
            np.testing.assert_allclose(D.d3(ctx).quantiles(), c1[t - D.O0] + np.quantile(exp3, D.TAUS))


def test_model_list_and_parameters_are_closed():
    assert list(D.DENSITY) == ["D1", "D2", "D3", "D4", "D5", "D6", "D1b"] and D.SELECTABLE == ("D1", "D2", "D3", "D4", "D5", "D6")
    assert (D.MIN_ERR, D.MIN_CELL, D.K_ANALOG, D.D5_HORIZONS) == (104, 20, 30, (1,))
    assert D.Q_REPORT == (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95) and set(D.Q_REPORT) <= set(D.TAUS) and len(D.TAUS) == 19
    assert D.events_for(1) == ["VIX>20", "VIX>25", "VIX>30", "rialzo>20%"] and D.events_for(4) == ["VIX>20", "VIX>25", "VIX>30"]
    assert D.event_threshold("rialzo>20%", 3.0) == pytest.approx(3.0 + np.log(1.2)) and D.event_threshold("VIX>25", 3.0) == pytest.approx(np.log(25))


# ------------------------------------------------------------------ componenti

def test_quantile_regression_matches_statsmodels():
    import statsmodels.api as sm
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(400), rng.standard_normal((400, 3))])
    y = X @ np.array([1.0, 0.5, -0.3, 0.2]) + rng.standard_t(4, 400) * (1 + 0.3 * np.abs(X[:, 1]))
    taus = [0.05, 0.25, 0.5, 0.9]
    b = D.quantreg(X, y, taus)
    loss = lambda beta, tau: V.pinball(y, (X @ beta)[:, None], [tau]).mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, tau in enumerate(taus):
            ref = sm.QuantReg(y, X).fit(q=tau).params
            assert loss(b[k], tau) <= loss(ref, tau) * (1 + 1e-4)


def test_crossed_quantiles_are_sorted_and_flagged():
    q = np.linspace(2.5, 3.5, 19)
    assert not D.Griglia(q).crossed
    bad = q.copy()
    bad[[3, 4]] = bad[[4, 3]]
    g = D.Griglia(bad)
    assert g.crossed and (np.diff(g.quantiles()) >= 0).all()
    assert g.prob_above(q[9]) == pytest.approx(0.5) and g.prob_above(-10) == 1.0 and g.prob_above(10) == 0.0
    assert 0.0 < g.prob_above(q[-1] + 0.01) < 0.05


def test_pinball_brier_and_dm():
    y = np.array([1.0, 2.0])
    pb = V.pinball(y, np.array([[0.0], [3.0]]), [0.9])
    np.testing.assert_allclose(pb[:, 0], [0.9, 0.1])
    p, o = np.array([0.05, 0.05, 0.95, 0.95]), np.array([0.0, 0.0, 1.0, 0.0])
    r = V.brier_parts(p, o)
    assert r["brier"] == pytest.approx(((p - o) ** 2).mean())
    assert r["affidabilita"] - r["risoluzione"] + r["incertezza"] == pytest.approx(r["brier"], abs=1e-12)   # previsioni costanti per classe
    rng = np.random.default_rng(1)
    a, b = rng.random(400), rng.random(400) * 0.5
    assert V.dm_loss(a, b, 1)[0] > 3 and V.dm_loss(b, a, 4)[0] < -3


def test_scheda_uses_only_data_up_to_origin():
    df = K.load_dev()
    scelta = {1: "D4", 4: "D1", 12: "D3"}
    t = 700
    full = SC.scheda(df, t, scelta)
    cut = SC.scheda(df.iloc[: t + 1].copy(), t, scelta)
    assert full == cut
    assert set(full["orizzonti"]) == {"1", "4", "12"}
    for h, s in full["orizzonti"].items():
        q = list(s["quantili"].values())
        assert q == sorted(q) and s["mediana"] == s["quantili"]["q50"]
        assert s["intervalli"]["90"][0] <= s["intervalli"]["50"][0] <= s["mediana"] <= s["intervalli"]["50"][1] <= s["intervalli"]["90"][1]
        assert all(0.0 <= v <= 1.0 for v in s["probabilita"].values())
        assert ("rialzo>20%" in s["probabilita"]) == (h == "1")
    assert "realizzato" not in " ".join(full.keys())


# ------------------------------------------------------------------ dati esterni: allineamento

def test_weekly_alignment_uses_only_available_values():
    days = pd.bdate_range("2020-01-01", "2020-12-31")
    rng = np.random.default_rng(5)
    s = pd.Series(rng.standard_normal(len(days)), index=days)
    fridays = pd.date_range("2020-02-07", "2020-12-25", freq="W-FRI")
    for lag in (0, 1, 5):
        a = FE.allinea_settimanale(s, fridays, lag)
        assert ((a["data_osservazione"] + pd.Timedelta(days=lag)) <= a.index).all()        # mai un valore non ancora disponibile
        cut = pd.Timestamp("2020-07-03")
        s2 = s.copy()
        s2[s2.index + pd.Timedelta(days=lag) > cut] += 100.0                               # stravolge cio' che diventa noto dopo il taglio
        a2 = FE.allinea_settimanale(s2, fridays, lag)
        before = a.index <= cut
        np.testing.assert_array_equal(a["valore"][before], a2["valore"][before])
        assert not np.allclose(a["valore"][~before], a2["valore"][~before])
    a0, a5 = FE.allinea_settimanale(s, fridays, 0), FE.allinea_settimanale(s, fridays, 5)
    assert (a0["data_osservazione"] == a0.index).all()                                     # venerdì stesso
    assert ((a5.index - a5["data_osservazione"]).dt.days >= 5).all()                       # col ritardo si arretra


def test_weekly_alignment_handles_gaps_and_stale_values():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-03", "2020-01-10"]))
    a = FE.allinea_settimanale(s, pd.to_datetime(["2019-12-27", "2020-01-03", "2020-01-10", "2020-01-17", "2020-02-14"]))
    assert np.isnan(a["valore"].iloc[0]) and a["valore"].iloc[1:4].tolist() == [1.0, 2.0, 2.0]
    assert np.isnan(a["valore"].iloc[4])                                                  # piu' vecchio di 14 giorni -> mancante


def test_continuous_futures_and_split():
    d = pd.bdate_range("2021-01-04", "2021-02-26")
    mk = lambda v: pd.Series(v, index=d, dtype=float)
    cont = FE.continui_futures([(pd.Timestamp("2021-01-20"), mk(20.0)), (pd.Timestamp("2021-02-17"), mk(22.0)), (pd.Timestamp("2021-03-17"), mk(24.0))])
    assert cont.loc["2021-01-19", ["vx1", "vx2"]].tolist() == [20.0, 22.0]
    assert cont.loc["2021-01-20", ["vx1", "vx2"]].tolist() == [22.0, 24.0]                # il contratto in scadenza esce il giorno della scadenza
    assert (cont["vx1_scadenza"] > cont.index).all()
    dev, hold = FE.dividi(cont, pd.Timestamp("2021-01-29"))
    assert dev.index.max() <= pd.Timestamp("2021-01-29") < hold.index.min() and len(dev) + len(hold) == len(cont)


def test_weekly_bars_from_daily():
    d = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-08", "2021-01-11"])
    daily = pd.DataFrame({"open": [10, 11, 12, 13], "high": [15, 12, 18, 14], "low": [9, 8, 11, 12], "close": [11, 12, 13, 14]}, index=d)
    w = FE.settimanale_da_giornaliero(daily)
    assert w.loc["2021-01-04", ["open", "high", "low", "close", "trading_days"]].tolist() == [10, 18, 8, 13, 3]
    assert w.loc["2021-01-04", "last_trading_day"] == pd.Timestamp("2021-01-08")


def test_generated_external_files_respect_boundary():
    wk = FE.DEV / "esterni_settimanali_dev.csv"
    if not wk.exists():
        pytest.skip("eseguire prima python -m src.fetch_external")
    confine = K.load_dev()["date"].max()
    w = pd.read_csv(wk, parse_dates=["date"])
    assert w["date"].max() == confine and len(w) == len(K.load_dev())
    for name, lag in FE.LAGS.items():
        if name not in w.columns:
            continue
        obs = pd.to_datetime(w[f"{name}_data_oss"])
        ok = obs.notna()
        assert ((obs[ok] + pd.Timedelta(days=lag)) <= w.loc[ok, "date"]).all(), name
    dd = pd.read_csv(FE.DEV / "esterni_giornalieri_dev.csv", parse_dates=["data_osservazione"])
    assert dd["data_osservazione"].max() <= confine


# ------------------------------------------------------------------ blocco dell'holdout

def test_guard_blocks_external_raw_and_holdout():
    with K.guardia_dati():
        for path in (FE.RAW / "cboe" / "VIX_History.csv", FE.HOLD / "esterni_settimanali_holdout.csv", FE.RAW / "qualunque.csv", C.HOLDOUT_FILE):
            with pytest.raises(PermissionError):
                open(path)
        assert len(K.load_dev()) > 1000


def test_phase2_modules_never_reference_forbidden_data():
    """Fallisce se un modulo di Fase 2 nomina holdout, clean, grezzo o dati esterni, o legge dati fuori da load_dev().

    `report.py` puo' citare nel testo i dati esterni e le colonne (legge solo i file di risultati in reports/fase2).
    """
    files_banned = r"HOLDOUT_FILE|CLEAN_FILE|RAW_FILE|vix_weekly_holdout|vix_weekly_clean|VIX_settimanale|DEV_FILE"
    files = sorted(FASE2_DIR.glob("*.py")) + [pathlib.Path(C.ROOT) / "src" / "run_phase2.py"]
    assert len(files) >= 5
    for f in files:
        src = f.read_text()
        assert not re.search(files_banned, src), f.name
        assert not re.search(r"read_csv\((?!R / |OUT / )", src), f"{f.name}: i dati si leggono solo con comune.load_dev()"
        assert not re.search(r"import.*fetch_external|from \.\. import fetch|EXT\b|FE\.", src), f"{f.name}: nessun accesso ai dati esterni"
        if f.name != "report.py":
            code = "\n".join(l for l in src.splitlines() if "python -m src.fetch_external" not in l)
            assert not re.search(r"fetch_external|external[\"'/]", code), f.name
            assert not re.search(r"[\"'](open|high|low)[\"']", code), f"{f.name}: colonne non ammesse"
    assert "with K.guardia_dati():" in (pathlib.Path(C.ROOT) / "src" / "run_phase2.py").read_text()


def test_fetch_external_never_prints_values():
    """Il programma di scarico stampa solo nomi, conteggi e date; le statistiche sono calcolate sul solo periodo dev."""
    src = (pathlib.Path(C.ROOT) / "src" / "fetch_external.py").read_text()
    prints = re.findall(r"(?<![\w.])(?:print|log)\((.*)\)", src)
    assert len(prints) >= 4
    for p in prints:
        assert not re.search(r"\.(mean|std|median|describe|to_string|head|tail|iloc|to_numpy)\(", p), p
    assert "corr(" not in src and "import matplotlib" not in src


def test_all_models_share_the_same_origins():
    path = V.OUT / "a_quantili.csv"
    if not path.exists():
        pytest.skip("eseguire prima python -m src.run_phase2")
    q = pd.read_csv(path, parse_dates=["data_origine", "data_obiettivo"])
    dev_end = K.load_dev()["date"].max()
    assert q["data_obiettivo"].max() <= dev_end
    for h, g in q.groupby("h"):
        for m in D.DENSITY:
            col = g[f"{m}_q50"]
            assert col.notna().all() if (m != "D5" or h in D.D5_HORIZONS) else col.isna().all(), (h, m)
        qs = g[[f"D4_q{int(100 * t):02d}" for t in D.Q_REPORT]].to_numpy()
        assert (np.diff(qs, axis=1) >= 0).all()                 # quantili riordinati
