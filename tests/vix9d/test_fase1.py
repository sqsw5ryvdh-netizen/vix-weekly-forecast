"""Fase 9D-1 (VIX9D): candele settimanali, guardia, variabili e modelli senza look-ahead, regole di composizione,
versioni congelate, programma di previsione, piano d'esame.

I test che richiedono i file prodotti da `python -m src.vix9d.run_fase1` vengono saltati se mancano."""
import json
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from src import baselines as B
from src import config as C
from src.fase1 import comune as K
from src.vix9d import config as G
from src.vix9d import congela, dati, forecast as F, lettura as LE, modelli as M, prepara, valuta as VA, versioni as VS

DIR = pathlib.Path(C.ROOT) / "src" / "vix9d"
pronto = pytest.mark.skipif(not (G.OUT / "info.json").exists() or not G.FILE_DEV.exists(), reason="eseguire prima python -m src.vix9d.run_fase1")


@pytest.fixture(scope="module")
def df():
    return dati.load_dev()


@pytest.fixture(scope="module")
def scelte():
    return json.loads((G.OUT / "info.json").read_text())["scelte"]


def _stravolgi(df, t):
    d = df.copy()
    dopo = d.index > t
    for c in ("open", "high", "low", "close", "vix"):
        d.loc[dopo, c] = d.loc[dopo, c] * 3.0 + 7.0
    d.loc[dopo, "candela"] = ~d.loc[dopo, "candela"]
    d["log_close"] = np.log(d["close"])
    return d


def _senza_vix_origine(ris):
    """La previsione senza il campo informativo `vix_origine` (G1 lo riporta ma non lo usa)."""
    prev, cand = ris
    return {k: v for k, v in prev.items() if k != "vix_origine"}, cand


def _giornaliero(inizio="2013-10-07", n=60, piatte=10, seed=3):
    """Sedute sintetiche: le prime `piatte` hanno un solo valore, come il file Cboe prima di novembre 2013."""
    rng = np.random.default_rng(seed)
    giorni = pd.bdate_range(inizio, periods=n)
    c = 15 * np.exp(np.cumsum(0.05 * rng.standard_normal(n)))
    o, h, l = c * 1.03, c * 1.08, c * 0.95
    o[:piatte], h[:piatte], l[:piatte] = c[:piatte], c[:piatte], c[:piatte]
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=giorni)


# ------------------------------------------------------------------ dati

def test_weekly_candles_follow_the_vix_conventions():
    d = _giornaliero()
    w = prepara.settimanali(d)
    assert list(w.columns) == prepara.COLONNE
    for r in w.itertuples():
        s = d[(d.index >= r.week_start) & (d.index < r.week_start + pd.Timedelta(days=7))]
        assert r.week_start.day_of_week == 0 and r.last_trading_day == s.index[-1]
        assert r.open == s["open"].iloc[0] and r.close == s["close"].iloc[-1] and r.high == s["high"].max() and r.low == s["low"].min()
        assert r.trading_days == len(s)
    assert w["sedute_solo_chiusura"].tolist()[:3] == [5, 5, 0]          # 10 sedute piatte = prime due settimane
    assert (w["note"] == "").all()                                       # 60 sedute = 12 settimane complete


def test_incomplete_last_week_is_flagged_and_excluded():
    w = prepara.settimanali(_giornaliero(n=58))                          # l'ultima settimana ha 3 sedute
    assert w["note"].iloc[-1].startswith("Settimana in corso") and (w["note"].iloc[:-1] == "").all()
    dev, hold = prepara.dividi(w)
    assert len(dev) + len(hold) == len(w) - 1


def test_split_uses_the_vix_boundary_and_caps_the_exam_period():
    giorni = pd.bdate_range("2024-09-02", periods=5 * 130)
    d = pd.DataFrame({"open": 15.0, "high": 16.0, "low": 14.0, "close": 15.5}, index=giorni)
    dev, hold = prepara.dividi(prepara.settimanali(d))
    assert G.CONFINE == "2024-11-08" and G.SETTIMANE_HOLDOUT == C.HOLDOUT_WEEKS == 104
    assert dev["date"].max() == pd.Timestamp(G.CONFINE) and hold["date"].min() == pd.Timestamp("2024-11-15")
    assert len(hold) == 104                                              # le settimane ancora successive non entrano nell'holdout


@pronto
def test_development_file_stops_at_the_boundary(df):
    assert df["date"].max() <= pd.Timestamp(G.CONFINE) and df["date"].is_monotonic_increasing
    assert df["vix"].notna().all() and df["vix_stessa_seduta"].all()
    assert not df["candela"].iloc[0] and df["candela"].iloc[-1]
    prima = df.loc[df["candela"], "date"].iloc[0]
    assert df.loc[df["date"] >= prima, "candela"].all()                  # dopo la prima candela vera non ci sono piu' settimane piatte


@pronto
def test_guard_blocks_every_file_with_weeks_after_the_boundary():
    with dati.guardia():
        for p in (G.FILE_COMPLETO, G.FILE_VIX_COMPLETO, G.FILE_HOLDOUT, C.HOLDOUT_FILE):
            with pytest.raises(PermissionError):
                open(p).close()
        assert len(dati.load_dev()) > 0
    open(G.FILE_COMPLETO).close()                                        # fuori dalla guardia il programma puo' leggere


def test_only_prepara_and_forecast_touch_the_full_files_and_no_exam_command_exists():
    vietati = re.compile(r"FILE_COMPLETO|FILE_VIX_COMPLETO|FILE_HOLDOUT|HOLDOUT_FILE|read_csv\([^)]*external")
    for p in sorted(DIR.glob("*.py")):
        if p.name in ("prepara.py", "forecast.py", "config.py", "dati.py"):
            continue
        assert not vietati.search(p.read_text()), p.name
    assert "read_csv(G.FILE_COMPLETO" not in (DIR / "dati.py").read_text() and "FILE_HOLDOUT" not in (DIR / "dati.py").read_text().split("def guardia")[1]
    assert not list(DIR.glob("*esame*")) and not list((pathlib.Path(C.ROOT) / "src").glob("*esame*"))
    run = (DIR / "run_fase1.py").read_text()
    assert "with dati.guardia():" in run and run.index("prepara.main") < run.index("with dati.guardia():")


@pronto
def test_cboe_weekly_vix_matches_the_project_vix_on_the_development_part():
    v = pd.read_csv(G.FILE_VIX_COMPLETO, parse_dates=["week_start", "last_trading_day"])
    v = v[v["last_trading_day"] <= pd.Timestamp(G.CONFINE)]              # solo la parte di sviluppo
    p = pd.read_csv(C.DEV_FILE, parse_dates=["date", "week_start"])
    m = v.merge(p, on="week_start", suffixes=("_cboe", "_progetto"))
    assert len(m) > 700 and (m["last_trading_day"] == m["date"]).all()
    for k in ("open", "high", "low", "close"):
        assert np.allclose(m[f"{k}_cboe"], m[f"{k}_progetto"], atol=0.011)


# ------------------------------------------------------------------ variabili e modelli

def test_candidate_lists_are_the_closed_lists():
    assert M.CHIUSURA == {"R1": (), "R2": ("ln_rapporto_vix",), "R3": ("escursione", "escursione_4", "posizione"), "R4": ("escursione", "escursione_4", "posizione", "ln_rapporto_vix")}
    assert M.ESTREMI == {"S0": (), "S1": M.CANDELA, "S2": M.CANDELA + M.RAPPORTO} and M.DIREZIONE == {"BASE": (), "RAPP": M.RAPPORTO}
    assert M.SENZA_VIX == {"chiusura": ("R1", "R3"), "estremi": ("S0", "S1"), "direzione": ("BASE",)}
    assert M.CON_VIX == {"chiusura": ("R2", "R4"), "estremi": ("S2",), "direzione": ("RAPP",)}
    for gruppo in ("chiusura", "estremi", "direzione"):
        tutti = {"chiusura": M.CHIUSURA, "estremi": M.ESTREMI, "direzione": M.DIREZIONE}[gruppo]
        assert all("ln_rapporto_vix" not in tutti[k] for k in M.SENZA_VIX[gruppo]) and all("ln_rapporto_vix" in tutti[k] for k in M.CON_VIX[gruppo])
    assert G.ORIZZONTI == (1, 2) and G.MIN_SETTIMANE == 156 and G.SOGLIE == (20.0, 25.0, 30.0) and G.NEUTRA == (0.40, 0.60) and G.FINESTRA_CORPO == 260
    assert G.TAUS == (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


@pronto
def test_variables_match_definitions_and_ignore_the_future(df):
    V = M.variabili(df)
    t = 400
    r = df.iloc[t]
    assert V["escursione"].iloc[t] == pytest.approx(np.log(r["high"] / r["low"]))
    assert V["escursione_4"].iloc[t] == pytest.approx(np.log(df["high"].iloc[t - 3:t + 1] / df["low"].iloc[t - 3:t + 1]).mean())
    assert V["posizione"].iloc[t] == pytest.approx((r["close"] - r["low"]) / (r["high"] - r["low"]))
    assert V["ln_rapporto_vix"].iloc[t] == pytest.approx(np.log(r["close"] / r["vix"]))
    senza = np.flatnonzero(~df["candela"].to_numpy())
    assert V.loc[senza, ["escursione", "escursione_4", "posizione"]].isna().all().all() and V["ln_rapporto_vix"].notna().all()
    primo = int(np.flatnonzero(df["candela"].to_numpy())[0])
    assert V["escursione_4"].iloc[primo:primo + 3].isna().all() and np.isfinite(V["escursione_4"].iloc[primo + 3])
    pd.testing.assert_frame_equal(V.iloc[: t + 1], M.variabili(_stravolgi(df, t)).iloc[: t + 1])


@pronto
def test_models_need_the_minimum_history_and_use_only_past_pairs(df):
    x = df["log_close"].to_numpy()
    Z0 = np.empty((len(x), 0))
    assert np.isnan(M.chiusura(x[:155], Z0[:155], 1)) and np.isfinite(M.chiusura(x[:156], Z0[:156], 1))      # prima stima su 156 settimane
    n, h = 300, 2
    i = M.coppie(n, Z0[:n], h)
    assert i.min() == 11 and i.max() == n - 1 - h                                                              # l'esito piu' recente usato e' la settimana di origine
    F_ = B.har_features(x[:n])
    beta = np.linalg.lstsq(F_[i - 11], x[i + h], rcond=None)[0]
    assert M.chiusura(x[:n], Z0[:n], h) == pytest.approx(float(F_[-1] @ beta))
    q = M.quantili(x[:n], Z0[:n], h)
    assert len(q) == 7 and (np.diff(q) >= 0).all()
    assert 0 < M.direzione(x[:n], Z0[:n], h) < 1
    assert M.frequenza_rialzo(x[:n], h) == pytest.approx(float((x[h:n] > x[:n - h]).mean()))


@pronto
def test_extremes_use_only_weeks_with_a_real_candle(df):
    x, cand = df["log_close"].to_numpy(), df["candela"].to_numpy(bool)
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    Z0 = np.empty((len(x), 0))
    primo = int(np.flatnonzero(cand)[0])
    n_ok = primo + G.MIN_COPPIE                                           # coppie con esito a candela vera: i = primo - 1 ... n - 2, cioe' n - primo
    assert np.isnan(M.estremo(x[:n_ok - 1], hi[:n_ok - 1], Z0[:n_ok - 1], 1, cand[:n_ok - 1])) and np.isfinite(M.estremo(x[:n_ok], hi[:n_ok], Z0[:n_ok], 1, cand[:n_ok]))
    n = 500
    finti = hi.copy()
    finti[:primo] = 1e6                                                   # gli estremi delle settimane senza candela non devono contare
    for h in G.ORIZZONTI:
        assert M.estremo(x[:n], hi[:n], Z0[:n], h, cand[:n]) == M.estremo(x[:n], finti[:n], Z0[:n], h, cand[:n])
        assert M.regola_fissa(x[:n], hi[:n], h, cand[:n]) == M.regola_fissa(x[:n], finti[:n], h, cand[:n])
        assert M.estremo(x[:n], hi[:n], Z0[:n], h, cand[:n]) > M.estremo(x[:n], lo[:n], Z0[:n], h, cand[:n])
    i = np.arange(0, n - 1)
    i = i[cand[i + 1]]
    assert M.regola_fissa(x[:n], hi[:n], 1, cand[:n]) == pytest.approx(float(np.exp(x[n - 1]) * np.median(hi[i + 1] / np.exp(x[i]))))


def test_candle_is_commanded_by_the_extremes():
    comp = {"rapporto_apertura": 1.10, "p_su": 0.27, "corpo": 0.20}
    k = M.candela_prevista(20.0, comp, 23.0, 17.0)                       # apertura 22, chiusura 22 x exp(-0,2) = 18,01: tutto dentro
    assert k["apertura"] == pytest.approx(22.0) and k["chiusura"] == pytest.approx(22.0 * np.exp(-0.20)) and not k["apertura_tagliata"] and not k["chiusura_tagliata"]
    k = M.candela_prevista(20.0, comp, 21.0, 19.0)                       # apertura e chiusura fuori: tagliate, gli estremi non si spostano
    assert (k["apertura"], k["chiusura"], k["massimo"], k["minimo"]) == (21.0, 19.0, 21.0, 19.0) and k["apertura_tagliata"] and k["chiusura_tagliata"]
    k = M.candela_prevista(20.0, comp | {"p_su": 0.6}, 30.0, 17.0)       # corpo verso l'alto se la frequenza storica e' >= 0,5
    assert k["chiusura"] == pytest.approx(22.0 * np.exp(0.20))
    k = M.candela_prevista(20.0, comp, 17.0, 23.0)                       # estremi incrociati: scambiati e segnalati
    assert (k["massimo"], k["minimo"]) == (23.0, 17.0) and k["estremi_scambiati"]
    assert M.etichetta(0.60)[0] == "rialzo" and M.etichetta(0.40)[0] == "ribasso" and M.etichetta(0.59) == ("neutra", "neutra") and M.etichetta(0.21)[1] == "ribasso 79%"
    assert M.quintile(5.0, np.arange(1.0, 101.0)) == 1 and M.quintile(50.0, np.arange(1.0, 101.0)) == 3 and M.quintile(99.0, np.arange(1.0, 101.0)) == 5


@pronto
def test_candle_components_follow_definitions(df):
    n = 600
    o, c, cand = (df[k].to_numpy()[:n] for k in ("open", "close", "candela"))
    comp = M.componenti_candela(o, c, cand)
    j = np.arange(1, n)
    j = j[cand[j] & cand[j - 1]]
    k = np.flatnonzero(cand)
    assert comp["rapporto_apertura"] == pytest.approx(float(np.median(o[j] / c[j - 1])))
    assert comp["p_su"] == pytest.approx(float((c[k] > o[k]).mean())) and comp["corpo"] == pytest.approx(float(np.abs(np.log(c[k] / o[k]))[-260:].mean()))
    assert np.isnan(M.componenti_candela(o[:200], c[:200], cand[:200])["corpo"])


# ------------------------------------------------------------------ versioni e misure

@pronto
@pytest.mark.parametrize("versione", VS.VERSIONI)
def test_versions_ignore_everything_after_the_origin(df, scelte, versione):
    for t in (450, 700):
        a = VS.prevedi(df, t, versione, scelte)
        assert a == VS.prevedi(_stravolgi(df, t), t, versione, scelte) == VS.prevedi(df.iloc[: t + 1], t, versione, scelte)
        prev, cand = a
        assert prev["data_origine"] == str(df["date"].iloc[t].date()) and [s["h"] for s in prev["settimane"]] == [1, 2]
        for s, k in zip(prev["settimane"], cand):
            q = list(s["quantili"].values())
            assert q == sorted(q) and s["chiusura"] == s["quantili"]["q50"] and s["fasce"]["80"] == [s["quantili"]["q10"], s["quantili"]["q90"]]
            assert k["minimo"] <= k["apertura"] <= k["massimo"] and k["minimo"] <= k["chiusura"] <= k["massimo"]
            assert all(0 <= p <= 1 for p in s["probabilita"].values()) and s["probabilita"]["VIX>20"] >= s["probabilita"]["VIX>25"] >= s["probabilita"]["VIX>30"]
            assert s["n_errori"] >= G.MIN_ERRORI and k["colore"] == k["direzione_chiusura"] == M.etichetta(k["p_chiusura_su"])[0]
        assert cand[1]["apertura_tagliata"] or cand[1]["apertura"] == pytest.approx(prev["settimane"][0]["chiusura"] * cand[1]["rapporto_apertura"])


@pronto
def test_versions_are_not_available_without_history_or_without_vix(df, scelte):
    with pytest.raises(VS.NonDisponibile):
        VS.prevedi(df, 200, "g1", scelte)                                # estremi: meno di 144 esiti con candela vera
    d = df.copy()
    d.loc[d.index[700], "vix"] = np.nan
    assert VS.usa_vix(scelte, "gs") and not VS.usa_vix(scelte, "g1")
    with pytest.raises(VS.NonDisponibile):
        VS.prevedi(d, 700, "gs", scelte)
    assert _senza_vix_origine(VS.prevedi(d, 700, "g1", scelte)) == _senza_vix_origine(VS.prevedi(df, 700, "g1", scelte))      # G1 non guarda mai il VIX
    assert VS.prevedi(d, 700, "g1", scelte)[0]["vix_origine"] is None


@pronto
@pytest.mark.parametrize("versione", VS.VERSIONI)
def test_development_files_match_the_program(df, scelte, versione):
    f = pd.read_csv(G.OUT / f"previsioni_dev_{versione}.csv", parse_dates=["data_origine"])
    pos = {pd.Timestamp(x): i for i, x in enumerate(df["date"])}
    o2 = f.loc[f["h"] == 2, "data_origine"]                               # origini con entrambe le candele
    for data in (o2.iloc[0], o2.iloc[len(o2) // 5], o2.iloc[-1]):
        prev, cand = VS.prevedi(df, pos[pd.Timestamp(data)], versione, scelte)
        for s, k in zip(prev["settimane"], cand):
            r = f[(f["data_origine"] == data) & (f["h"] == s["h"])].iloc[0]
            assert r[f"{versione}_q50"] == pytest.approx(s["chiusura"], abs=1e-4) and r[f"{versione}_q10"] == pytest.approx(s["quantili"]["q10"], abs=1e-4)
            assert r[f"{versione}_centro"] == pytest.approx(s["centro"], abs=1e-4) and r[f"{versione}_p_su"] == pytest.approx(k["p_chiusura_su"], abs=1e-5)
            assert r[f"{versione}_massimo"] == pytest.approx(k["massimo"], abs=1e-4) and r[f"{versione}_minimo"] == pytest.approx(k["minimo"], abs=1e-4)
            assert r[f"{versione}_apertura"] == pytest.approx(k["apertura"], abs=1e-4) and r[f"{versione}_chiusura"] == pytest.approx(k["chiusura"], abs=1e-4)
            assert r[f"{versione}_p_25"] == pytest.approx(s["probabilita"]["VIX>25"], abs=1e-5)


@pronto
def test_every_comparison_is_on_the_same_origins_and_only_in_development():
    b, cf, cd, de, ev = (pd.read_csv(G.OUT / f"{k}.csv") for k in ("b_chiusura", "c_fasce", "c_direzione", "d_estremi", "e_versioni"))
    info = json.loads((G.OUT / "info.json").read_text())
    for h in G.ORIZZONTI:
        n = info["origini"][str(h)]["comuni"]
        assert set(b.loc[(b["campione"] == "origini comuni") & (b["h"] == h), "n"]) == {n} == set(cf.loc[cf["h"] == h, "n"]) == set(cd.loc[cd["h"] == h, "n"])
        assert set(de.loc[(de["h"] == h) & (de["gruppo"] == "tutte"), "n"]) == {n} == set(ev.loc[(ev["h"] == h) & (ev["gruppo"] == "tutte"), "n"])
        assert de.loc[(de["h"] == h) & de["gruppo"].str.startswith("quintile") & (de["insieme"] == "S0") & (de["estremo"] == "massimo"), "n"].sum() == n
        assert info["origini"][str(h)]["ultima_data_obiettivo"] <= G.CONFINE
    for v in VS.VERSIONI:
        f = pd.read_csv(G.OUT / f"previsioni_dev_{v}.csv")
        assert f["data_obiettivo"].max() <= G.CONFINE and (f["data_origine"] < f["data_obiettivo"]).all()


# ------------------------------------------------------------------ regole

def test_selection_rule_needs_lower_error_and_holm_below_five_percent():
    r = lambda chiave, cand, stat, p, err: {"famiglia": "prova", "chiave": chiave, "riferimento": "R1", "candidato": cand, "test": "t", "stat": stat, "p": p, "errore": err, "errore_riferimento": 1.0}
    s = VA.scegli([r("1", "A", 3.0, 0.001, 0.90), r("1", "B", 3.0, 0.002, 0.80), r("2", "A", 2.0, 0.060, 0.95), r("2", "B", -3.0, 0.001, 1.10)])
    assert s["scelta"] == {"1": "B", "2": "R1"}                          # +1: passano entrambi, vince l'errore piu' basso; +2: A non e' significativo e B e' peggiore
    t = s["tabella"].set_index(["chiave", "candidato"])
    assert t.loc[("2", "A"), "p_holm"] == pytest.approx(0.06) and not bool(t.loc[("2", "A"), "batte"]) and t.loc[("1", "B"), "p_holm"] == pytest.approx(0.004)
    assert bool(t.loc[("1", "A"), "batte"]) and not bool(t.loc[("2", "B"), "batte"])
    assert VA.scegli([r("1", "A", 3.0, 0.04, 0.9), r("2", "A", 3.0, 0.04, 0.9)])["scelta"] == {"1": "R1", "2": "R1"}      # 0,04 x 2 = 0,08 dopo Holm
    e = VA.etichette(np.array([0.7, 0.62, 0.5, 0.3, 0.1]), np.array([1.0, 0.0, 1.0, 0.0, 1.0]), 0.60)
    assert (e["n_non_neutre_60"], e["n_rialzo_60"], e["n_ribasso_60"]) == (4, 2, 2) and e["accuratezza_60"] == 0.5 and e["quota_non_neutre_60"] == 0.8


@pronto
def test_versions_are_composed_by_the_rules(scelte):
    rg = pd.read_csv(G.OUT / "regole.csv")
    assert set(rg["famiglia"]) == {"chiusura senza VIX", "chiusura con VIX", "estremi senza VIX", "estremi con VIX", "direzione con VIX"}
    assert (rg["batte"] == ((rg["p_holm"] < 0.05) & (rg["stat"] > 0) & (rg["errore"] < rg["errore_riferimento"]))).all()
    vince = lambda fam, chiave: rg[(rg["famiglia"] == fam) & (rg["chiave"].astype(str) == chiave) & rg["batte"]].sort_values("errore")["candidato"].tolist()
    for h in ("1", "2"):
        g1 = scelte["chiusura"]["g1"][h]
        assert g1 in M.SENZA_VIX["chiusura"] and g1 == (vince("chiusura senza VIX", h) or ["R1"])[0]
        assert scelte["chiusura"]["gs"][h] == (vince("chiusura con VIX", h) or [g1])[0]
        assert scelte["direzione"]["g1"][h] == "BASE" and scelte["direzione"]["gs"][h] == (vince("direzione con VIX", h) or ["BASE"])[0]
        for e in VA.EST:
            s1 = scelte["estremi"]["g1"][e][h]
            assert s1 in M.SENZA_VIX["estremi"] and s1 == (vince("estremi senza VIX", f"{e}|{h}") or ["S0"])[0]
            assert scelte["estremi"]["gs"][e][h] == (vince("estremi con VIX", f"{e}|{h}") or [s1])[0]
    for fam, k in (("chiusura senza VIX", 2), ("chiusura con VIX", 4), ("estremi senza VIX", 4), ("estremi con VIX", 4), ("direzione con VIX", 2)):
        assert (rg["famiglia"] == fam).sum() == k


@pronto
def test_frozen_configurations_match_the_code(scelte):
    attese = congela.costruisci()
    for v in VS.VERSIONI:
        path = G.VERSIONI / f"{v}.json"
        if not path.exists():
            pytest.skip("eseguire python -m src.vix9d.congela")
        cfg = json.loads(path.read_text())
        assert cfg["versione"] == v and re.fullmatch(r"[0-9a-f]{40}", cfg["commit"]) and cfg["sorgenti_identici_al_commit"] is True
        assert {k: x for k, x in cfg.items() if k not in congela.VARIABILI_ESECUZIONE} == json.loads(json.dumps(attese[v]))
        assert cfg["sha256_sorgenti"] == congela.impronte() and set(cfg["sha256_sorgenti"]) == set(VS.SORGENTI_CONGELATI)
        assert cfg["scelte"] == scelte and cfg["piano_d_esame"] == "reports/vix9d/piano_esame_holdout.md"
    assert "VIX settimanale (chiusura)" in attese["gs"]["dati"] and "VIX settimanale (chiusura)" not in attese["g1"]["dati"]
    assert {"src/vix9d/modelli.py", "src/vix9d/versioni.py", "src/vix9d/config.py"} <= set(VS.SORGENTI_CONGELATI)


# ------------------------------------------------------------------ programma

@pronto
def test_situation_reading_uses_only_data_up_to_the_origin(df):
    t = 650
    L = LE.lettura(df, t)
    assert L == LE.lettura(_stravolgi(df, t), t) and L["data_origine"] == str(df["date"].iloc[t].date()) and L["settimane_di_storia"] == t + 1
    c = df["close"].to_numpy()
    assert L["livello"]["chiusura"] == c[t] and L["livello"]["media_12_settimane"] == pytest.approx(c[t - 11:t + 1].mean())
    assert L["dinamica"]["variazione_4_settimane"]["punti"] == pytest.approx(c[t] - c[t - 4]) and L["vix"]["rapporto"] == pytest.approx(c[t] / df["vix"].iloc[t])
    assert L["precedenti"]["n"] >= 0 and L["precedenti"]["meno_di_20_casi"] == (L["precedenti"]["n"] < 20)


def _file_grezzi(df, cartella, togli_ultimo_vix=False, riga_incompleta=False):
    """File nel formato dei file completi, costruiti dal solo campione di sviluppo."""
    w = df[["week_start", "date", "open", "high", "low", "close", "trading_days", "sedute_solo_chiusura"]].rename(columns={"date": "last_trading_day"}).assign(note="")
    if riga_incompleta:
        u = w.iloc[-1].copy()
        u["week_start"], u["last_trading_day"], u["trading_days"], u["note"] = u["week_start"] + pd.Timedelta(days=7), u["week_start"] + pd.Timedelta(days=8), 2, "Settimana in corso"
        u["close"] = 99.0
        w = pd.concat([w, u.to_frame().T], ignore_index=True)
    v = df[["week_start", "date", "vix"]].rename(columns={"date": "last_trading_day", "vix": "close"}).assign(open=lambda d: d["close"], high=lambda d: d["close"], low=lambda d: d["close"], trading_days=5, sedute_solo_chiusura=0, note="")
    if togli_ultimo_vix:
        v = v.iloc[:-1]
    fmt = dict(index=False, date_format="%Y-%m-%d")
    w.to_csv(cartella / "v9.csv", **fmt)
    v[prepara.COLONNE].to_csv(cartella / "vix.csv", **fmt)
    return cartella / "v9.csv", cartella / "vix.csv"


@pronto
def test_forecast_two_versions_registry_and_incomplete_week(df, scelte, tmp_path, capsys):
    if not (G.VERSIONI / "g1.json").exists() or not (G.OUT / "g3.json").exists():
        pytest.skip("eseguire python -m src.vix9d.congela")
    f9, fv = _file_grezzi(df, tmp_path, riga_incompleta=True)
    r = F.main([], out_dir=tmp_path, oggi="2030-01-01", file9=f9, file_vix=fv)
    t = len(df) - 1
    assert r["versione"] == "g1" and set(r["versioni"]) == {"g1", "gs"} and r["registrate"] == ["g1", "gs"] and r["info"]["ultima_data_completa"] == str(df["date"].iloc[-1].date())
    assert r["versioni"]["g1"] == VS.prevedi(df, t, "g1", scelte) and r["versioni"]["gs"] == VS.prevedi(df, t, "gs", scelte)       # la riga incompleta (chiusura 99) non entra
    tab = r["tabella"]
    assert tab["settimana"].tolist() == ["+1", "+2"] and (tab["minimo"] <= tab["chiusura"]).all() and (tab["chiusura"] <= tab["massimo"]).all()
    assert {"fascia_80_inf", "copertura_storica_80", "p_sopra_20", "p_sopra_30", "etichetta_chiusura", "chiusura_centrale", "massimo_entro_5_storico"} <= set(tab.columns)
    md = r["md"].read_text()
    for pezzo in ("Lettura della situazione", "Distanza dal VIX", "Precedenti in situazioni simili", "Confronto tra le versioni", "Verifica visiva sul passato", "non è un consiglio di investimento"):
        assert pezzo in md
    assert (tmp_path / f"g1_{r['previsione']['data_origine']}.png").stat().st_size > 20000
    reg = pd.read_csv(tmp_path / "registro_previsioni.csv")
    assert len(reg) == 4 and set(reg["versione"]) == {"g1", "gs"} and set(reg["data_esecuzione"]) == {"2030-01-01"}
    r2 = F.main(["--versione", "gs"], out_dir=tmp_path, oggi="2030-01-02", file9=f9, file_vix=fv)
    assert r2["versione"] == "gs" and r2["registrate"] == [] and len(pd.read_csv(tmp_path / "registro_previsioni.csv")) == 4       # nessun duplicato
    capsys.readouterr()
    v = F.main(["--verifica"], out_dir=tmp_path, file9=f9, file_vix=fv)
    assert v["dettaglio"].empty and "non c'è nulla da verificare" in capsys.readouterr().out
    F.main(["--origine", str(df["date"].iloc[-3].date())], out_dir=tmp_path, file9=f9, file_vix=fv)
    assert "non aggiornato" in capsys.readouterr().out and len(pd.read_csv(tmp_path / "registro_previsioni.csv")) == 4
    v = F.main(["--verifica"], out_dir=tmp_path, file9=f9, file_vix=fv)                                                              # nulla di completato dopo l'origine registrata
    assert v["dettaglio"].empty


@pronto
def test_forecast_falls_back_to_g1_when_the_vix_is_missing(df, scelte, tmp_path, capsys):
    if not (G.VERSIONI / "gs.json").exists() or not (G.OUT / "g3.json").exists():
        pytest.skip("eseguire python -m src.vix9d.congela")
    f9, fv = _file_grezzi(df, tmp_path, togli_ultimo_vix=True)
    r = F.main(["--versione", "gs"], out_dir=tmp_path, oggi="2030-01-01", file9=f9, file_vix=fv)
    out = capsys.readouterr().out
    assert r["versione"] == "g1" and "Si ripiega su g1" in out and set(r["versioni"]) == {"g1"} and "gs" in r["note"] and r["registrate"] == ["g1"]
    assert _senza_vix_origine(r["versioni"]["g1"]) == _senza_vix_origine(VS.prevedi(df, len(df) - 1, "g1", scelte))
    assert "non è disponibile" in r["md"].read_text() and "non è calcolabile" in r["md"].read_text()
    r = F.main([], out_dir=tmp_path / "solo", oggi="2030-01-01", file9=f9)                                                           # senza file del VIX: solo G1
    assert set(r["versioni"]) == {"g1"} and r["lettura"]["vix"] is None


def test_update_rebuilds_only_the_full_weekly_files(tmp_path):
    d = _giornaliero(inizio="2026-01-05", n=33, piatte=0)
    testo = "DATE,OPEN,HIGH,LOW,CLOSE\n" + "\n".join(f"{g.strftime('%m/%d/%Y')},{r.open:.4f},{r.high:.4f},{r.low:.4f},{r.close:.4f}" for g, r in zip(d.index, d.itertuples()))
    chiesti = []
    out = F.aggiorna(scarica=lambda url: (chiesti.append(url), testo.encode())[1], cartella=tmp_path / "scaricati", destinazione=tmp_path / "dati")
    assert chiesti == list(G.INDIRIZZI.values()) and set(out) == {"vix9d", "vix"} and all("cboe.com" in u for u in chiesti)
    assert sorted(p.name for p in (tmp_path / "dati").iterdir()) == sorted([G.FILE_COMPLETO.name, G.FILE_VIX_COMPLETO.name])       # ne' sviluppo ne' holdout
    w = pd.read_csv(tmp_path / "dati" / G.FILE_COMPLETO.name)
    assert len(w) == 7 and w["note"].fillna("").iloc[-1].startswith("Settimana in corso") and w["trading_days"].iloc[-1] == 3
    letto, _ = F.leggi_settimanale(tmp_path / "dati" / G.FILE_COMPLETO.name)
    assert len(letto) == 6 and letto["date"].iloc[-1] == d.index[29]


def test_forecast_measures_nothing_after_the_boundary_except_registry_verification():
    src = (DIR / "forecast.py").read_text()
    corpo = src.split("def esegui_verifica")[0] + src.split("def main")[1]
    assert "RG.verifica" not in corpo.replace("esegui_verifica(df, reg_path, out)", "") and "_reale" not in src and "pagella_riga" not in src
    assert 'pd.read_csv(G.OUT / "e_versioni.csv")' in src                # le misure accanto alla previsione sono quelle di sviluppo, solo lette
    for nome in ("valuta", "analisi", "verifica", "report", "piano"):
        assert f"import {nome}" not in src and f", {nome}" not in src.split("OUT_DIR")[0]


# ------------------------------------------------------------------ report e piano

@pronto
def test_report_and_exam_plan_are_written_and_the_exam_is_not_executed():
    rep, piano = G.OUT / "fase1_report.md", G.OUT / "piano_esame_holdout.md"
    if not rep.exists():
        pytest.skip("eseguire python -m src.vix9d.run_fase1")
    t = rep.read_text()
    for pezzo in ("## Sintesi in 10 righe", "## 1. Audit dei dati", "## 2. Analisi compatta", "## 3. Chiusura (Parte B)", "## 4. Fasce, probabilità e direzione (Parte C)", "## 5. Massimo e minimo (Parte D)",
                  "## 6. Candele e versioni (Parte E)", "## 7. Quanto aggiunge il VIX", "## 8. VIX e VIX9D a confronto", "## 10. Problemi e ambiguità", "## 11. Proposte per il passo successivo"):
        assert pezzo in t, pezzo
    assert len(re.findall(r"^\d+\. \*\*", t.split("## Sintesi in 10 righe")[1].split("## 1.")[0], flags=re.M)) == 10
    assert "nan" not in t.lower().replace("finanz", "") and "p = <" not in t
    p = piano.read_text()
    assert "non eseguito" in p and "Non esiste alcun comando che esegua l'esame" in p and "104 settimane" in p and "G1" in p and "GS" in p
    assert "Diebold-Mariano" in p and "Holm" in p and "non distinguibili" in p and "resta G1" in p
    assert not list(G.OUT.glob("*esame_risultati*")) and not list(G.OUT.glob("esame*"))
    man = json.loads((G.OUT / "run_manifest.json").read_text())
    assert man["guardia_holdout_attiva"] is True and man["file_dati_letti_dalle_misure"] == [G.FILE_DEV.name, K.NOME_FILE_DATI]
