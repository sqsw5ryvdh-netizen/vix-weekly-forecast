"""Fase 4c: massimo e minimo diretti della prima candela (insiemi S0-S5), previsione incrociata con l'S&P 500 (B1, B3;
tetto teorico B2 solo in valutazione), corpo K5, candela comandata dagli estremi, versioni F1 e FS, configurazioni
congelate, programma, piano d'esame.

Anti look-ahead di tutte le variabili e di tutti i modelli nuovi; regole di scelta; ripiego degli estremi di FS;
il tetto teorico non entra nel programma; guardia sull'holdout.
I test che richiedono file prodotti da `run_phase4c` o i dati esterni di sviluppo vengono saltati se mancano."""
import json
import pathlib
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from src import baselines as B
from src import config as C
from src import forecast as F
from src.fase1 import comune as K
from src.fase2b import dati as dati2b
from src.fase2c import componenti as CP
from src.fase3 import dati as dati3
from src.fase4 import esterni as ES4
from src.fase4b import prima as PR, versioni as V4B
from src.fase4c import candela as KC, congela, esterni as ESC, estremi as ES, incrociata as IN, tetto as TT, valuta as VA, variabili as V, versioni as VS

DIR = pathlib.Path(C.ROOT) / "src" / "fase4c"
OUT = C.REPORTS_DIR / "fase4c"
SCELTE = {"variante": "A1", "prima_candela": "K2", "minimo": "diretto", "candidato": "P5", "insieme_massimo_f1": "S0", "insieme_minimo_f1": "S1",
          "insieme_massimo_fs": "S0", "insieme_minimo_fs": "S0", "corpo": "K2"}


@pytest.fixture(scope="module")
def df():
    return dati2b.load_dev_candele()


def _stravolgi(df, t):
    d = df.copy()
    for c in ("open", "high", "low", "close"):
        d.loc[d.index > t, c] = d.loc[d.index > t, c] * 3.0 + 7.0
    d["log_close"] = np.log(d["close"])
    return d


def _sintetico(n_sett=420, seed=7):
    """Calendario sintetico: 5 sedute a settimana, data settimanale = venerdi'. Restituisce (VIX settimanale, giornalieri)."""
    rng = np.random.default_rng(seed)
    giorni = pd.bdate_range("2010-01-04", periods=5 * n_sett)
    cammino = lambda s0, vol: s0 * np.exp(np.cumsum(vol * rng.standard_normal(len(giorni))))
    vix = 18 * np.exp(0.25 * np.sin(np.arange(len(giorni)) / 37) + 0.05 * rng.standard_normal(len(giorni)))
    g = pd.DataFrame({"vix": vix, "vix3m": vix * (1.08 + 0.03 * rng.standard_normal(len(giorni))), "vvix": cammino(90, 0.02), "spx": cammino(1100, 0.01)}, index=giorni)
    sett = vix.reshape(n_sett, 5)
    df = pd.DataFrame({"date": giorni[4::5], "open": sett[:, 0] * 1.02, "high": sett.max(axis=1) * 1.03, "low": sett.min(axis=1) * 0.97, "close": sett[:, 4]})
    return df, g


@pytest.fixture(scope="module")
def sint():
    df, g = _sintetico()
    return df, g, V.costruisci(df, g)


def _z(df, inizio=500):
    x = np.log(df["close"].to_numpy())
    z = 0.12 - 0.3 * (x - pd.Series(x).rolling(12, min_periods=1).mean().to_numpy()) + 0.02 * np.random.default_rng(4).standard_normal(len(x))
    z[:inizio] = np.nan
    return z


# ------------------------------------------------------------------ variabili

def test_variable_sets_are_the_closed_list():
    assert tuple(V.INSIEMI) == ("S0", "S1", "S2", "S3", "S4", "S5") and V.INSIEMI["S0"] == () and V.INSIEMI["S1"] == V.S1
    assert V.INSIEMI["S2"] == V.S1 + V.S2 and V.INSIEMI["S3"] == V.S1 + V.S3 and V.INSIEMI["S4"] == V.S1 + V.S4 and V.INSIEMI["S5"] == V.S1 + V.S2 + V.S3 + V.S4
    assert [len(V.INSIEMI[s]) for s in V.INSIEMI] == [0, 3, 5, 7, 4, 10] and V.SERIE["S1"] == () and set(V.SERIE["S5"]) == {"vvix", "spx", "vix", "vix3m"}
    assert ES.FINESTRA == 520 and ES.MINIMO_COPPIE == 156 and VA.STRAPPO == 0.20


def test_variables_match_their_definitions(sint):
    df, g, VV = sint
    t = 300
    d, prec = df["date"].iloc[t], df["date"].iloc[t - 1]
    esc = np.log(df["high"] / df["low"])
    assert VV["s1_escursione"][t] == pytest.approx(esc[t]) and VV["s1_escursione_4"][t] == pytest.approx(esc[t - 3:t + 1].mean())
    assert VV["s1_posizione"][t] == pytest.approx((df["close"][t] - df["low"][t]) / (df["high"][t] - df["low"][t])) and 0 <= VV["s1_posizione"].min() and VV["s1_posizione"].max() <= 1
    vv = g["vvix"][:d]
    assert VV["s2_ln_vvix"][t] == pytest.approx(np.log(vv.iloc[-1]))
    assert VV["s2_ln_vvix_meno_media52"][t] == pytest.approx(np.log(vv.iloc[-1]) - np.log(vv[vv.index > d - pd.Timedelta(days=364)]).mean())
    s = g["spx"][:d]
    r = np.log(s).diff()
    assert VV["s3_volreal5"][t] == pytest.approx(100 * np.sqrt(252 * (r.iloc[-5:] ** 2).mean())) and VV["s3_volreal20"][t] == pytest.approx(100 * np.sqrt(252 * (r.iloc[-20:] ** 2).mean()))
    sett = s[s.index > prec]
    assert len(sett) == 5 and VV["s3_escursione_spx"][t] == pytest.approx(np.log(sett.max() / sett.min())) and VV["s3_rend_spx"][t] == pytest.approx(np.log(s.iloc[-1] / s[:prec].iloc[-1]))
    assert VV["s4_vix3m_su_vix"][t] == pytest.approx(g["vix3m"][d] / g["vix"][d] - 1)
    assert VV["b_rend4"][t] == pytest.approx(np.log(s.iloc[-1] / s[:df["date"].iloc[t - 4]].iloc[-1]))
    assert VV["b_escursione_spx_4"][t] == pytest.approx(VV["s3_escursione_spx"][t - 3:t + 1].mean()) and VV["b_escursione_spx_12"][t] == pytest.approx(VV["s3_escursione_spx"][t - 11:t + 1].mean())
    assert np.isnan(VV["s2_ln_vvix_meno_media52"][:39]).all() and np.isfinite(VV.iloc[60:].to_numpy()).all()       # servono almeno 200 valori giornalieri


def test_variables_ignore_everything_after_the_week(sint):
    df, g, VV = sint
    X = VV.to_numpy()
    for t in (40, 200, 419):
        tronco = V.costruisci(df.iloc[: t + 1], g[g.index <= df["date"].iloc[t]]).to_numpy()
        np.testing.assert_allclose(tronco, X[: t + 1], rtol=0, atol=1e-12, equal_nan=True)
    t = 250
    g2, d2 = g.copy(), df.copy()
    g2.loc[g2.index > df["date"].iloc[t]] = g2.loc[g2.index > df["date"].iloc[t]] * 3.0 + 7.0
    for c in ("high", "low", "close"):
        d2.loc[d2.index > t, c] = d2.loc[d2.index > t, c] * 3.0 + 7.0
    Y = V.costruisci(d2, g2).to_numpy()
    np.testing.assert_allclose(Y[: t + 1], X[: t + 1], rtol=0, atol=1e-12, equal_nan=True)
    assert not np.allclose(Y[t + 1:t + 40], X[t + 1:t + 40], equal_nan=True)                     # il test "morde"
    g3 = g.copy()
    g3.loc[df["date"].iloc[t]] = g3.loc[df["date"].iloc[t]] * 1.5                                 # anche l'ultima seduta della settimana conta
    assert not np.allclose(V.costruisci(df, g3).to_numpy()[t], X[t], equal_nan=True)


def test_missing_series_give_missing_variables_never_carried_forward(sint):
    df, g, VV = sint
    t = 200
    g2 = g.copy()
    g2.loc[(g2.index > df["date"].iloc[t - 1]) & (g2.index <= df["date"].iloc[t]), "vvix"] = np.nan
    W = V.costruisci(df, g2)
    assert np.isnan(W["s2_ln_vvix"][t]) and np.isfinite(W["s2_ln_vvix"][t + 1]) and W["s3_volreal5"].equals(VV["s3_volreal5"])
    solo_vix = V.costruisci(df, None)                                                            # senza dati giornalieri: solo le variabili della candela del VIX
    assert solo_vix[list(V.S1)].equals(VV[list(V.S1)]) and solo_vix[list(V.S2 + V.S3 + V.S4)].isna().all().all()
    senza_spx = V.costruisci(df, g[["vix", "vix3m", "vvix"]])
    assert senza_spx[list(V.S3)].isna().all().all() and senza_spx["s4_vix3m_su_vix"].equals(VV["s4_vix3m_su_vix"])


def test_real_variables_use_only_data_up_to_each_week(df):
    if not dati3.esterni_disponibili():
        pytest.skip("dati esterni di sviluppo assenti: eseguire python -m src.fetch_external")
    with K.guardia_dati():
        g = dati3.load_giornaliero()
    VV = V.costruisci(df, g)
    for t in (300, 900, 1296):
        tronco = V.costruisci(df.iloc[: t + 1], g[g.index <= df["date"].iloc[t]])
        np.testing.assert_allclose(tronco.to_numpy(), VV.iloc[: t + 1].to_numpy(), rtol=0, atol=1e-12, equal_nan=True)
    assert np.isfinite(VV[list(V.S1)].iloc[3:].to_numpy()).all() and np.isfinite(VV[list(V.S3)].iloc[1:].to_numpy()).all()


# ------------------------------------------------------------------ modelli degli estremi

def test_direct_extremes_are_median_regressions_on_a_window(df):
    import statsmodels.api as sm
    x, low, high = np.log(df["close"].to_numpy()), df["low"].to_numpy(dtype=float), df["high"].to_numpy(dtype=float)
    VV = V.costruisci(df, None)
    Z0, Z1 = np.empty((len(x), 0)), VV[list(V.S1)].to_numpy()
    t = 1000
    assert ES.mediana_diretta(x[: t + 1], low[: t + 1], Z0[: t + 1]) == PR.minimo_m1q(x[: t + 1], low[: t + 1])          # S0 sul minimo = M1q della Fase 4b
    i = ES.coppie(t + 1, Z1[: t + 1])
    assert len(i) == 520 and i.max() == t - 1
    Fh = B.har_features(x[: t + 1])
    Zs = (Z1 - Z1[i].mean(axis=0)) / Z1[i].std(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.QuantReg(np.log(high[i + 1]) - x[i], np.column_stack([Fh[i - 11], Zs[i]])).fit(q=0.5).params
    atteso = float(np.exp(x[t] + np.concatenate([Fh[-1], Zs[t]]) @ ref))
    assert ES.mediana_diretta(x[: t + 1], high[: t + 1], Z1[: t + 1]) == pytest.approx(atteso, rel=3e-3)
    assert ES.regola_fissa(df["close"].to_numpy()[: t + 1], high[: t + 1]) == pytest.approx(df["close"][t] * np.median(high[1:t + 1] / df["close"].to_numpy()[:t]))
    for L in (low, high):                                                                         # nessun dato successivo all'origine
        a = ES.mediana_diretta(x[: t + 1], L[: t + 1], Z1[: t + 1])
        L2 = L.copy()
        L2[t - 40:t + 1] *= 0.9
        assert a != ES.mediana_diretta(x[: t + 1], L2[: t + 1], Z1[: t + 1]) and np.isfinite(a)
    Zc = Z1.copy()                                                                                # meno di 156 coppie valide, oppure variabili mancanti all'origine: nessuna previsione
    Zc[: t - 150] = np.nan
    assert np.isnan(ES.mediana_diretta(x[: t + 1], low[: t + 1], Zc[: t + 1]))
    Zc[: t - 160] = Z1[: t - 160]
    Zc[t - 160:t - 150] = Z1[t - 160:t - 150]
    assert np.isfinite(ES.mediana_diretta(x[: t + 1], low[: t + 1], Zc[: t + 1]))
    Zb = Z1.copy()
    Zb[t] = np.nan
    assert np.isnan(ES.mediana_diretta(x[: t + 1], low[: t + 1], Zb[: t + 1]))


def test_cross_forecast_real_version_and_theoretical_ceiling(sint):
    df, g, VV = sint
    x, high = np.log(df["close"].to_numpy()), df["high"].to_numpy(dtype=float)
    t = 380
    xt, VVt = x[: t + 1], VV.iloc[: t + 1]
    r, e = IN.prevedi_spx(xt, VVt)
    rend = VV["s3_rend_spx"].to_numpy()
    Xr = np.column_stack([VV[IN.PRED_REND].to_numpy(), x])
    j = np.arange(t)[np.isfinite(Xr[:t]).all(axis=1) & np.isfinite(rend[1:t + 1])][-520:]
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(j)), Xr[j]]), rend[j + 1], rcond=None)[0]
    assert r == pytest.approx(float(np.concatenate([[1.0], Xr[t]]) @ beta)) and np.isfinite(e)
    M = IN.matrice_legame(VVt)
    assert M.shape == (t + 1, 5) and np.isnan(M[t, 3:]).all() and M[t - 1, 3] == pytest.approx(rend[t]) and M[t - 1, 4] == pytest.approx(VV["s3_escursione_spx"][t])
    b3 = IN.b3(xt, high[: t + 1], VVt)
    assert b3 == pytest.approx(ES.mediana_diretta(xt, high[: t + 1], M, np.concatenate([VV[list(V.S1)].to_numpy()[t], [r, e]])))
    futuro = (rend[t + 1], VV["s3_escursione_spx"][t + 1])
    b2 = TT.b2_tetto_teorico(xt, high[: t + 1], VVt, futuro)
    assert b2 == pytest.approx(ES.mediana_diretta(xt, high[: t + 1], M, np.concatenate([VV[list(V.S1)].to_numpy()[t], futuro]))) and b2 != b3
    # B3 non dipende da nulla di successivo all'origine; B2 dipende SOLO dal dato futuro passato in modo esplicito
    df2, g2 = df.copy(), g.copy()
    g2.loc[g2.index > df["date"].iloc[t]] *= 1.7
    for c in ("high", "low", "close"):
        df2.loc[df2.index > t, c] *= 1.7
    VV2 = V.costruisci(df2, g2)
    assert IN.b3(np.log(df2["close"].to_numpy())[: t + 1], df2["high"].to_numpy()[: t + 1], VV2.iloc[: t + 1]) == b3
    assert TT.b2_tetto_teorico(xt, high[: t + 1], VVt, (futuro[0] - 0.05, futuro[1] * 2)) != b2


# ------------------------------------------------------------------ candela e versioni

def test_body_is_cut_inside_predicted_extremes():
    assert KC.CORPI == ("K2", "K5")
    assert KC.chiusura_k5(20.0, "ribasso", 0.08, 0.10, 19.0) == pytest.approx(20 * np.exp(-0.08)) and KC.chiusura_k5(20.0, "rialzo", 0.08, 0.10, 19.0) == pytest.approx(20 * np.exp(0.10))
    assert KC.chiusura_k5(20.0, "neutra", 0.08, 0.10, 19.0) == 19.0                                # neutra: come K2
    k = KC.componi(20.6, 18.2, 22.0, 19.0)                                                        # la chiusura esce sotto il minimo: si accorcia il corpo
    assert (k["apertura"], k["chiusura"], k["massimo"], k["minimo"]) == (20.6, 19.0, 22.0, 19.0) and k["chiusura_tagliata"] and not k["apertura_tagliata"]
    k = KC.componi(23.0, 21.0, 22.0, 19.0)                                                        # l'apertura esce sopra il massimo
    assert (k["apertura"], k["massimo"]) == (22.0, 22.0) and k["apertura_tagliata"] and not k["chiusura_tagliata"]
    k = KC.componi(20.6, 19.5, 22.0, 19.0)
    assert not k["apertura_tagliata"] and not k["chiusura_tagliata"] and not k["estremi_scambiati"]
    k = KC.componi(20.0, 19.5, 18.0, 21.0)                                                        # minimo sopra il massimo: scambiati e contati
    assert (k["minimo"], k["massimo"]) == (18.0, 21.0) and k["estremi_scambiati"]


def test_versions_first_candle_ignores_future_and_follows_definition(df):
    z, t = _z(df), 1100
    VV = V.costruisci(df, None)
    f1 = VS.prevedi_f1(df, t, SCELTE)
    fs = VS.prevedi_fs(df, t, z, VV, SCELTE)
    d2 = _stravolgi(df, t)
    assert f1 == VS.prevedi_f1(d2, t, SCELTE) == VS.prevedi_f1(df.iloc[: t + 1].copy(), t, SCELTE) == VS.prevedi_f1(df, t, SCELTE, VV)
    assert fs == VS.prevedi_fs(d2, t, z, V.costruisci(d2, None), SCELTE) == VS.prevedi_fs(df.iloc[: t + 1].copy(), t, z[: t + 1], VV.iloc[: t + 1], SCELTE)
    assert VS.prevedi_f1(_stravolgi(df, t - 30), t, SCELTE)[1][0] != f1[1][0]                                    # il test "morde"
    b1, bs = V4B.prevedi_f1(df, t, VS.scelte_4b(SCELTE)), V4B.prevedi_fs(df, t, z, VS.scelte_4b(SCELTE))
    x, hi, lo = np.log(df["close"].to_numpy()[: t + 1]), df["high"].to_numpy(dtype=float)[: t + 1], df["low"].to_numpy(dtype=float)[: t + 1]
    Z0, Z1 = np.empty((t + 1, 0)), VV[list(V.S1)].to_numpy()[: t + 1]
    for (prev, cand), base, ins_min in ((f1, b1, Z1), (fs, bs, Z0)):
        c = cand[0]
        H, L = ES.mediana_diretta(x, hi, Z0), ES.mediana_diretta(x, lo, ins_min)
        assert c["massimo"] == pytest.approx(H) and c["minimo"] == pytest.approx(L) and c["apertura"] == pytest.approx(np.clip(base[1][0]["apertura"], L, H))
        assert c["chiusura"] == pytest.approx(np.clip(base[1][0]["chiusura"], L, H)) and c["minimo"] <= min(c["apertura"], c["chiusura"]) <= max(c["apertura"], c["chiusura"]) <= c["massimo"]
        assert c["colore"] == c["direzione_chiusura"] and c["costruzione"] == "K2+estremi/K2m-A1" == cand[3]["costruzione"] and not c["estremi_da_ripiego"]
        assert prev["settimane"] == base[0]["settimane"] and all(a[k] == b[k] for a, b in zip(cand[1:], base[1][1:]) for k in KC.CAMPI)     # +2…+4 invariate
    assert (f1[1][0]["massimo_da"], f1[1][0]["minimo_da"]) == ("S0", "S1") and (fs[1][0]["massimo_da"], fs[1][0]["minimo_da"]) == ("S0", "S0")
    k5 = VS.prevedi_f1(df, t, SCELTE | {"corpo": "K5"})[1][0]                                                    # corpo K5: la chiusura segue l'etichetta se non e' neutra
    d, u = PR.mosse(x)
    attesa = KC.chiusura_k5(float(np.exp(x[-1])), k5["direzione_chiusura"], d, u, b1[1][0]["chiusura"])
    assert k5["chiusura"] == pytest.approx(np.clip(attesa, k5["minimo"], k5["massimo"])) and k5["costruzione"].startswith("K5+estremi")


def test_challenger_extremes_fall_back_to_f1_sets_when_a_series_is_missing(df):
    z, t = _z(df), 1100
    sc = SCELTE | {"insieme_minimo_fs": "S2", "insieme_massimo_fs": "S3"}                            # sfidante ipotetica con VVIX e S&P 500
    VV = V.costruisci(df, None)                                                                    # nessun dato giornaliero: VVIX e S&P mancanti
    prev, cand = VS.prevedi_fs(df, t, z, VV, sc)
    f1 = VS.prevedi_f1(df, t, sc)
    c = cand[0]
    assert c["estremi_da_ripiego"] and (c["massimo_da"], c["minimo_da"]) == ("S0", "S1") and c["massimo"] == f1[1][0]["massimo"] and c["minimo"] == f1[1][0]["minimo"]
    VV2 = VV.copy()                                                                                # con le serie disponibili: usa i propri insiemi
    rng = np.random.default_rng(1)
    for col in V.S2 + V.S3:
        VV2[col] = rng.standard_normal(len(VV2))
    c2 = VS.prevedi_fs(df, t, z, VV2, sc)[1][0]
    assert not c2["estremi_da_ripiego"] and (c2["massimo_da"], c2["minimo_da"]) == ("S3", "S2") and c2["minimo"] != c["minimo"]
    VV3 = VV2.copy()
    VV3.loc[t, "s2_ln_vvix"] = np.nan                                                              # serie piu' vecchia dell'origine: variabile mancante nella settimana di origine
    assert VS.prevedi_fs(df, t, z, VV3, sc)[1][0]["estremi_da_ripiego"]
    assert VS.serie_richieste("S2") == ("vvix",) and VS.serie_richieste("B3") == ("spx",) and VS.serie_richieste("S1") == ()
    assert ESC.serie_di_fs(sc) == ("vix", "vix3m", "vvix", "spx") and ESC.serie_di_fs(SCELTE) == ("vix", "vix3m") and VS.INSIEMI_F1 == ("S0", "S1")


# ------------------------------------------------------------------ misure e regole

def test_metrics_known_values():
    p, r = np.array([20.9, 19.1, 25.0, 20.0]), np.array([20.0, 20.0, 20.0, 20.0])                # errori: +4,5%, -4,5%, +25%, 0
    m = VA.metriche(p, r)
    assert m["entro_5"] == 0.75 and m["entro_10"] == 0.75 and m["errore_pct_mediano"] == pytest.approx(0.045) and m["errore_pct_medio"] == pytest.approx(0.085)
    assert m["errore_punti_medio"] == pytest.approx(1.7) and m["oltre_5_per_eccesso"] == 0.25 and m["oltre_5_per_difetto"] == 0.0
    assert VA.metriche(np.array([18.0]), np.array([20.0]))["oltre_5_per_difetto"] == 1.0
    assert m["errore_log_medio"] == pytest.approx(np.abs(np.log(p / r)).mean())


def _regola(righe):
    return pd.DataFrame([{"bersaglio": "minimo", "modello": m, "n_variabili": k, "errore_log": e, "ammesso": a} for m, k, e, a in righe])


def test_set_rule_and_body_rule():
    assert VA.scegli_insieme(_regola([("S1", 3, 0.060, False), ("S2", 5, 0.058, False)]), "minimo") == "S0"                       # nessun ammesso: resta S0
    assert VA.scegli_insieme(_regola([("S1", 3, 0.060, True), ("S2", 5, 0.058, True), ("S5", 10, 0.050, False)]), "minimo") == "S2"  # errore piu' basso tra gli ammessi
    assert VA.scegli_insieme(_regola([("S4", 4, 0.060, True), ("S1", 3, 0.060, True)]), "minimo") == "S1"                         # parita': meno variabili
    rng = np.random.default_rng(3)
    n = 400
    f = pd.DataFrame({"idx_origine": np.arange(n), "minimo_reale": 15 + rng.standard_normal(n), "massimo_reale": 20 + rng.standard_normal(n), "chiusura_reale": 17 + rng.standard_normal(n)})
    rum = rng.standard_normal(n)
    for b, base in (("minimo", f["minimo_reale"]), ("massimo", f["massimo_reale"]), ("chiusura", f["chiusura_reale"])):
        f[f"S0_{b}"], f[f"S1_{b}"], f[f"S2_{b}"] = base + 1.0 * rum, base + 0.3 * rum, base + 2.0 * rum
    t = VA.confronti(f, ("S1", "S2"), "prova")
    r = t.set_index(["bersaglio", "modello"])
    assert r.loc[("minimo", "S1"), "ammesso"] and not r.loc[("minimo", "S2"), "ammesso"] and r.loc[("minimo", "S1"), "dm_stat"] > 0 > r.loc[("minimo", "S2"), "dm_stat"]
    assert VA.scegli_insieme(t, "minimo") == "S1" and (t.groupby("bersaglio")["p_holm"].min() >= t.groupby("bersaglio")["p"].min()).all()
    g = pd.DataFrame({"chiusura_reale": 20 + rng.standard_normal(n), "massimo_reale": 23.0, "minimo_reale": 18.0})
    for k, sd in (("K2c", 1.0), ("K5c", 0.3)):
        g[f"{k}_chiusura"], g[f"{k}_massimo"], g[f"{k}_minimo"] = g["chiusura_reale"] + sd * rng.standard_normal(n), 22.5, 18.4
    assert VA.scegli_corpo(g)["scelta"] == "K5"                                                   # chiusura significativamente migliore, sovrapposizione uguale
    g["K5c_chiusura"] = g["K2c_chiusura"] + 1e-3 * rng.standard_normal(n)
    assert VA.scegli_corpo(g)["scelta"] == "K2"                                                   # nessuna differenza significativa: resta K2
    g["K5c_chiusura"], g["K5c_massimo"] = g["chiusura_reale"] + 0.3 * rng.standard_normal(n), 20.5 + 0.2 * rng.standard_normal(n)
    s = VA.scegli_corpo(g)
    assert s["sovrapposizione_significativamente_peggiore"] and s["scelta"] == "K2"                 # chiusura migliore ma sovrapposizione peggiore


def test_development_files_match_program(df):
    path = OUT / "previsioni_dev.csv"
    if not path.exists() or not dati3.esterni_disponibili():
        pytest.skip("eseguire prima python -m src.run_phase4c (richiede i dati esterni di sviluppo)")
    info = json.loads((OUT / "info.json").read_text())
    sc = info["scelte"]
    p = pd.read_csv(path, parse_dates=["data_origine"])
    with K.guardia_dati():
        g = dati3.load_giornaliero()
    VV = V.costruisci(df, g)
    x, hi, lo = np.log(df["close"].to_numpy()), df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
    for t in (700, 1250):
        r = p[p["data_origine"] == df["date"].iloc[t]].iloc[0]
        c = VS.prevedi_f1(df, t, sc)[1][0]
        assert all(c[k] == pytest.approx(r[f"{sc['corpo']}c_{k}"], abs=3e-4) for k in KC.CAMPI)
        for s in ("S0", "S1", "S5"):
            if np.isfinite(r[f"{s}_massimo"]):
                Z = VV[list(V.INSIEMI[s])].to_numpy()[: t + 1]
                assert ES.mediana_diretta(x[: t + 1], hi[: t + 1], Z) == pytest.approx(r[f"{s}_massimo"], abs=3e-4) and ES.mediana_diretta(x[: t + 1], lo[: t + 1], Z) == pytest.approx(r[f"{s}_minimo"], abs=3e-4)
        assert IN.b3(x[: t + 1], lo[: t + 1], VV.iloc[: t + 1]) == pytest.approx(r["B3_minimo"], abs=3e-4)
        assert bool(r["strappo"]) == bool(r["chiusura_reale"] > 1.2 * r["chiusura_origine"])
    assert sc["insieme_massimo_f1"] in VS.INSIEMI_F1 and sc["insieme_minimo_f1"] in VS.INSIEMI_F1 and sc["insieme_massimo_fs"] in VS.INSIEMI_FS and sc["corpo"] in KC.CORPI
    assert p["data_obiettivo"].max() <= str(df["date"].max().date())
    st = pd.read_csv(OUT / "statistiche_programma.csv")
    assert set(st["versione"]) == {"f1", "fs"} and st.loc[st["h"] == 1, ["massimo_entro_5", "minimo_entro_5"]].notna().all().all()
    tb = pd.read_csv(OUT / "b_incrociata.csv")
    assert set(tb["bersaglio"]) == {"massimo", "minimo", "chiusura"} and (tb["entro_5_b2_tetto"] > tb["entro_5_s0"]).all()       # il tetto teorico sa piu' di S0


# ------------------------------------------------------------------ configurazioni congelate

def test_frozen_configurations_match_the_code():
    files = sorted(VS.CARTELLA.glob("f*.json"))
    if not files or not (OUT / "info.json").exists():
        pytest.skip("eseguire prima python -m src.run_phase4c e python -m src.fase4c.congela")
    attese = congela.costruisci()
    assert [f.stem for f in files] == ["f1", "fs"] == sorted(attese), "all'esame vanno solo F1 e FS"
    for f in files:
        cfg = json.loads(f.read_text())
        assert re.fullmatch(r"[0-9a-f]{40}", cfg["commit"]) and cfg["congelata_il"] and isinstance(cfg["sorgenti_identici_al_commit"], bool)
        resto = {k: v for k, v in cfg.items() if k not in congela.VARIABILI_ESECUZIONE}
        assert resto == json.loads(json.dumps(attese[f.stem])), f"{f.name} non coincide con la configurazione ricostruita dal codice"
        assert cfg["sha256_sorgenti"] == congela.impronte(), "un sorgente congelato e' cambiato dopo il congelamento: rieseguire congela PRIMA dell'esame"
        assert {"src/fase4c/estremi.py", "src/fase4c/variabili.py", "src/fase4c/candela.py", "src/fase4c/versioni.py", "src/fase4b/versioni.py"} <= set(cfg["sha256_sorgenti"])
        assert "src/fase4c/tetto.py" not in cfg["sha256_sorgenti"]                                 # il tetto teorico non fa parte di nessuna versione
        pc = cfg["candele"]["prima_candela"]
        v = cfg["versione"]
        assert pc["massimo"]["insieme"] == cfg["scelte"][f"insieme_massimo_{v}"] and pc["minimo"]["insieme"] == cfg["scelte"][f"insieme_minimo_{v}"] and pc["corpo"] == cfg["scelte"]["corpo"]
        assert pc["modello_degli_estremi"]["finestra_settimane"] == 520 and pc["modello_degli_estremi"]["minimo_coppie"] == 156
    f1 = json.loads((VS.CARTELLA / "f1.json").read_text())
    assert f1["candele"]["prima_candela"]["massimo"]["insieme"] in VS.INSIEMI_F1 and f1["candele"]["prima_candela"]["minimo"]["serie_giornaliere"] == []   # F1 resta solo storico del VIX


# ------------------------------------------------------------------ programma

def _file_grezzo(df, path):
    rows = [{"week_start": (r.date - pd.Timedelta(days=r.date.weekday())).date(), "last_trading_day": r.date.date(), "open": r.open, "high": r.high,
             "low": r.low, "close": r.close, "trading_days": 5, "note": ""} for r in df.itertuples()]
    pd.DataFrame(rows).to_csv(path, index=False)


def _file_cboe(df, cartella, fino_a=None, vix3m_da=600, altre=()):
    """File giornalieri nel formato Cboe: una seduta per settimana (la data settimanale)."""
    cartella.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    d = df if fino_a is None else df[df["date"] <= fino_a]
    date = d["date"].dt.strftime("%m/%d/%Y")
    for nome, val, da in (("vix", d["close"].to_numpy(), 0), ("vix3m", d["close"].to_numpy() * (1.10 + 0.04 * rng.standard_normal(len(d))), vix3m_da)):
        pd.DataFrame({"DATE": date, "OPEN": val, "HIGH": val, "LOW": val, "CLOSE": np.round(val, 2)}).iloc[da:].to_csv(cartella / ES4.FILE[nome], index=False)
    for nome in altre:                                                                             # VVIX e S&P 500: una colonna, sedute giornaliere, come nei file Cboe
        giorni = pd.bdate_range(d["date"].iloc[0] - pd.Timedelta(days=4), d["date"].iloc[-1])
        val = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(len(giorni))))
        pd.DataFrame({"DATE": giorni.strftime("%m/%d/%Y"), nome.upper(): np.round(val, 2)}).to_csv(cartella / ESC.FILE[nome], index=False)


def test_forecast_two_versions_fallback_and_registry(df, tmp_path, capsys):
    if not (VS.CARTELLA / "fs.json").exists():
        pytest.skip("eseguire prima python -m src.fase4c.congela")
    cfg = F.carica_versioni()
    sc = cfg["f1"]["scelte"]
    assert sorted(cfg) == ["f1", "fs"] and sc == cfg["fs"]["scelte"]
    if "f2" not in VS.CANDIDATI[sc["candidato"]]:
        pytest.skip("la sfidante congelata non usa VIX3M")
    base, out = tmp_path / "base.csv", tmp_path / "out"
    _file_grezzo(df, base)
    r = F.main(["--file", str(base), "--versione", "fs"], out_dir=out, oggi="2024-11-09")                 # nessun file giornaliero: avviso e ripiego su f1
    testo = capsys.readouterr().out
    assert r["versione"] == "f1" and list(r["versioni"]) == ["f1"] and "ATTENZIONE" in testo and "Si ripiega su f1" in testo
    assert r["candele"] == VS.prevedi_f1(df, len(df) - 1, sc)[1] and r["candele"][0]["massimo_da"] == sc["insieme_massimo_f1"]
    est = tmp_path / "esterni"
    _file_cboe(df, est)
    r2 = F.main(["--file", str(base), "--versione", "fs"], out_dir=out, esterni_dir=est, oggi="2024-11-10")
    assert r2["versione"] == "fs" and list(r2["versioni"]) == ["f1", "fs"] and r2["md"].name.endswith("_fs.md") and r2["versioni"]["f1"] == r["versioni"]["f1"]
    c = r2["candele"][0]
    assert (c["massimo_da"], c["minimo_da"]) == (sc["insieme_massimo_fs"], sc["insieme_minimo_fs"]) and c["minimo"] <= c["chiusura"] <= c["massimo"] and c["colore"] in ("rialzo", "ribasso", "neutra")
    conf, md = r2["confronto"], r2["md"].read_text()
    assert set(conf["versione"]) == {"f1", "fs"} and len(conf) == 8 and "Confronto tra le versioni" in md and "comandano la candela" in md and "entro il 5%" in md and "una sola volta" in md
    reg = pd.read_csv(out / "registro_previsioni.csv")
    assert set(reg["versione"]) == {"f1", "fs"} and len(reg) == 8 and r2["registrate"] == ["fs"] and (reg["costruzione"] == VS.nome_costruzione(sc)).all()
    vecchio = tmp_path / "vecchio"                                                                         # VIX3M piu' vecchio dell'origine: avviso e ripiego sulla versione f1
    _file_cboe(df, vecchio, fino_a=df["date"].iloc[-2])
    capsys.readouterr()
    r3 = F.main(["--file", str(base), "--versione", "fs", "--origine", str(df["date"].iloc[-1].date())], out_dir=tmp_path / "o3", esterni_dir=vecchio)
    testo = capsys.readouterr().out
    assert r3["versione"] == "f1" and "VIX3M manca nella settimana di origine" in testo
    r4 = F.main(["--file", str(base), "--origine", str(df["date"].iloc[-30].date())], out_dir=tmp_path / "o4", esterni_dir=est)   # dati successivi all'origine: ignorati
    tronco = tmp_path / "tronco"
    _file_cboe(df, tronco, fino_a=df["date"].iloc[-30])
    r5 = F.main(["--file", str(base), "--origine", str(df["date"].iloc[-30].date())], out_dir=tmp_path / "o5", esterni_dir=tronco)
    assert r4["versioni"]["fs"] == r5["versioni"]["fs"] and r4["versione"] == "f1"


def test_program_falls_back_to_f1_extremes_when_a_needed_series_is_old(df, tmp_path, capsys, monkeypatch):
    if not (VS.CARTELLA / "fs.json").exists():
        pytest.skip("eseguire prima python -m src.fase4c.congela")
    cfg = F.carica_versioni()
    if "f2" not in VS.CANDIDATI[cfg["fs"]["scelte"]["candidato"]]:
        pytest.skip("la sfidante congelata non usa VIX3M")
    ipotetica = {k: dict(v) for k, v in cfg.items()}                                                       # sfidante ipotetica: minimo con il VVIX (S2)
    ipotetica["fs"]["scelte"] = cfg["fs"]["scelte"] | {"insieme_minimo_fs": "S2"}
    monkeypatch.setattr(F, "carica_versioni", lambda: ipotetica)
    base = tmp_path / "base.csv"
    _file_grezzo(df, base)
    origine = str(df["date"].iloc[-1].date())
    senza = tmp_path / "senza_vvix"
    _file_cboe(df, senza)
    r = F.main(["--file", str(base), "--versione", "fs", "--origine", origine], out_dir=tmp_path / "o1", esterni_dir=senza)
    testo = capsys.readouterr().out
    c = r["candele"][0]
    assert r["versione"] == "fs" and c["estremi_da_ripiego"] and c["minimo_da"] == cfg["f1"]["scelte"]["insieme_minimo_f1"] and "ripiega sull'insieme di F1" in testo and "vvix" in testo
    con = tmp_path / "con_vvix"
    _file_cboe(df, con, altre=("vvix",))
    r2 = F.main(["--file", str(base), "--versione", "fs", "--origine", origine], out_dir=tmp_path / "o2", esterni_dir=con)
    c2 = r2["candele"][0]
    assert not c2["estremi_da_ripiego"] and c2["minimo_da"] == "S2" and (tmp_path / "o2" / f"fs_variabili_estremi_{origine}.csv").exists()
    capsys.readouterr()
    _file_cboe(df[df["date"] <= df["date"].iloc[-2]], con, altre=("vvix",))                               # VVIX piu' vecchio dell'origine
    _file_cboe(df, con)                                                                                    # (VIX e VIX3M restano aggiornati)
    r3 = F.main(["--file", str(base), "--versione", "fs", "--origine", origine], out_dir=tmp_path / "o3", esterni_dir=con)
    assert r3["candele"][0]["estremi_da_ripiego"] and "ripiega sull'insieme di F1" in capsys.readouterr().out
    chiesti = []                                                                                           # --aggiorna scarica solo le serie che servono a FS
    info = ESC.aggiorna(tmp_path / "e", ESC.serie_di_fs(ipotetica["fs"]["scelte"]), scarica=lambda url: chiesti.append(url) or b"DATE,VVIX\n01/02/2020,90\n")
    assert sorted(info) == ["vix", "vix3m", "vvix"] and all(u.startswith("https://cdn.cboe.com/") for u in chiesti) and len(chiesti) == 3
    assert ESC.serie_di_fs(cfg["fs"]["scelte"]) == tuple(k for k in ESC.CODICI if k in set(cfg["fs"]["scelte"] and ESC.serie_di_fs(cfg["fs"]["scelte"])))


# ------------------------------------------------------------------ guardia, tetto teorico e piano d'esame

def test_theoretical_ceiling_never_reaches_the_program_and_measures_read_only_dev():
    banned = r"HOLDOUT_FILE|CLEAN_FILE|RAW_FILE|vix_weekly_holdout|vix_weekly_clean|VIX_settimanale|[\"']holdout[\"']|[\"']raw[\"']"
    src_dir = pathlib.Path(C.ROOT) / "src"
    for f in sorted(DIR.glob("*.py")) + [src_dir / "run_phase4c.py"]:
        src = f.read_text()
        if f.name == "piano.py":
            assert not re.search(r"read_csv|open\(|load_|dati\.", src), f.name
        else:
            assert not re.search(banned, src), f.name
        if f.name != "esterni.py":                                                  # le misure non usano i dati giornalieri del programma
            assert not re.search(r"[\"']external[\"']|read_csv\(C\.(DEV|PROCESSED|RAW|ROOT)|import esterni|esterni as|fetch_external import|\.aggiorna\(", src), f.name
        if f.name not in ("valuta.py", "tetto.py"):                                 # il tetto teorico e' usato solo dalla valutazione
            assert not re.search(r"import tetto|tetto as|from \.tetto|b2_tetto_teorico\(", src), f"{f.name}: il tetto teorico non deve entrare nelle versioni"
    for f in [src_dir / "forecast.py"] + sorted((src_dir / "fase4").glob("*.py")) + sorted((src_dir / "fase4b").glob("*.py")):
        assert not re.search(r"\btetto\b|b2_tetto_teorico", f.read_text()), f.name
    assert "src/fase4c/tetto.py" not in VS.SORGENTI_CONGELATI and "NON E' UNA PREVISIONE UTILIZZABILE" in (DIR / "tetto.py").read_text()
    assert "with K.guardia_dati():" in (src_dir / "run_phase4c.py").read_text()
    src = (src_dir / "forecast.py").read_text()
    assert not re.search(r"import.*valuta|evaluate|density_row|dm_loss|pinball|brier|HOLDOUT_FILE|vix_weekly_holdout", src)
    with K.guardia_dati():
        for path in (C.HOLDOUT_FILE, C.ROOT / "data" / "external" / "holdout" / "esterni_giornalieri_holdout.csv"):
            with pytest.raises(PermissionError):
                open(path)


def test_exam_plan_includes_five_percent_shares_and_is_not_executed():
    piano = C.REPORTS_DIR / "piano_esame_holdout.md"
    if not piano.exists() or not (OUT / "info.json").exists():
        pytest.skip("eseguire prima python -m src.run_phase4c")
    testo = piano.read_text()
    for parola in ("non è stato eseguito", "versioni/f1.json", "versioni/fs.json", "Criterio D", "D1. direzione quando non neutra", "D4. massimo entro il 5%", "D5. minimo entro il 5%", "cinque test",
                   "senza le settimane di strappo", "nota solo a posteriori", "Holm", "non distinguibili", "resta F1", "Una sola esecuzione", "comandano la candela"):
        assert parola in testo, parola
    assert not list((pathlib.Path(C.ROOT) / "src").glob("*esame*")), "il comando d'esame non deve esistere in questa fase"
    assert not list(C.REPORTS_DIR.glob("*esame*risultat*")) and not (C.REPORTS_DIR / "esame").exists()
