"""Fase 3: Parte 0 (nuove candele, etichetta, registro) e blocchi G1, X1-X4, X6, X7.

Anti look-ahead di tutte le variabili nuove e di tutti i modelli; definizioni delle variabili su dati sintetici;
equivalenza dei riferimenti con M1, E1 e D4; regole di scelta; guardia sull'holdout (VIX ed esterni).
I test che richiedono i dati esterni di sviluppo (non pubblicati) vengono saltati se mancano."""
import json
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from src import baselines as B
from src import config as C
from src import forecast as F
from src.fase1 import comune as K
from src.fase2 import densita as D
from src.fase2b import dati as dati2b
from src.fase2c import componenti as CP
from src.fase3 import candele as KA, dati, modelli as M, registro as RG3, valuta as VA, valuta_candele as VK, variabili as V

DIR = pathlib.Path(C.ROOT) / "src" / "fase3"
SCELTE = {"corpo": {1: "regressione", 2: "regressione", 3: "media", 4: "media"}, "ombra_sup": {h: "media" for h in (1, 2, 3, 4)},
          "ombra_inf": {1: "regressione", 2: "media", 3: "media", 4: "media"}}


@pytest.fixture(scope="module")
def df():
    return dati2b.load_dev_candele()


def _stravolgi(df, t):
    d = df.copy()
    for c in ("open", "high", "low", "close"):
        d.loc[d.index > t, c] = d.loc[d.index > t, c] * 3.0 + 7.0
    d["log_close"] = np.log(d["close"])
    return d


# ------------------------------------------------------------------ Parte 0: candele

@pytest.mark.parametrize("h", KA.ORIZZONTI)
def test_mean_components_ignore_data_after_origin(df, h):
    t = 800
    a = KA.componenti_medie(CP.serie(df), t, h)
    assert a == KA.componenti_medie(CP.serie(_stravolgi(df, t)), t, h) == KA.componenti_medie(CP.serie(df.iloc[: t + 1]), t, h)
    assert a != KA.componenti_medie(CP.serie(_stravolgi(df, t - 30)), t, h)                 # il test "morde"
    assert min(a[f"{c}_{k}"] for c in KA.COMPONENTI for k in ("reg", "media")) >= 0 and 0.0 < a["p_L0"] < 1.0
    G = CP.grandezze(CP.serie(df), t)                                                       # la media storica e' la media degli esiti noti
    i = np.arange(11, t - h + 1)
    i = i[~G["bad"][i] & ~G["bad"][i + h]]
    assert a["corpo_media"] == pytest.approx(G["corpo"][i + h].mean())


def test_new_candles_ignore_future_follow_definition_and_are_coherent(df):
    for t in (700, 1100):
        prev, cand = KA.prevedi_candele(df, t, SCELTE)
        assert (prev, cand) == KA.prevedi_candele(_stravolgi(df, t), t, SCELTE) == KA.prevedi_candele(df.iloc[: t + 1].copy(), t, SCELTE)
        S = CP.serie(df)
        for c in cand:
            h = c["h"]
            comp = KA.componenti_medie(S, t, h)
            base = prev["chiusura_origine"] if h == 1 else prev["settimane"][h - 2]["chiusura"]         # K2: ancorata alla mediana D4 precedente
            assert c["apertura"] == pytest.approx(base * comp["rapporto_apertura"])
            corpo = comp["corpo_reg" if SCELTE["corpo"][h] == "regressione" else "corpo_media"]
            s = 1 if comp["p_L0"] >= 0.5 else -1                                                       # direzione dalla frequenza storica (L0)
            assert c["chiusura"] == pytest.approx(c["apertura"] * np.exp(s * corpo))
            assert c["massimo"] == pytest.approx(max(c["apertura"], c["chiusura"]) * np.exp(comp["ombra_sup_media"]))
            assert c["minimo"] <= min(c["apertura"], c["chiusura"]) <= max(c["apertura"], c["chiusura"]) <= c["massimo"]
            assert c["costruzione"] == "K2m" and c["modello_direzione"] == "L0" and c["p_rialzo"] == comp["p_L0"]
            assert (c["direzione_chiusura"], c["etichetta"]) == KA.etichetta(c["p_chiusura_su"])
    a = KA.prevedi_candele(df, 900, SCELTE)[1]
    b = KA.prevedi_candele(df, 900)[1]                                                                 # senza scelte: tutte medie storiche
    assert a[0]["chiusura"] != b[0]["chiusura"] and a[3]["chiusura"] == b[3]["chiusura"]


def test_label_rule_and_probability():
    assert KA.etichetta(0.60) == ("rialzo", "rialzo 60%") and KA.etichetta(0.599)[0] == "neutra" and KA.etichetta(0.5) == ("neutra", "neutra")
    assert KA.etichetta(0.40) == ("ribasso", "ribasso 60%") and KA.etichetta(0.401)[0] == "neutra" and KA.etichetta(0.19) == ("ribasso", "ribasso 81%")


@pytest.mark.parametrize("h", (1, 4))
def test_closing_direction_probability_is_d1_and_ignores_future(df, h):
    x = np.log(df["close"].to_numpy())
    t = 900
    p = KA.prob_chiusura_su(x, t, h)
    x2 = x.copy()
    x2[t + 1:] += 1.0
    assert p == KA.prob_chiusura_su(x2, t, h) == KA.prob_chiusura_su(x[: t + 1], t, h)
    c1 = D.centers(x, B.forecast_har, h)
    assert p == pytest.approx(KA.prob_chiusura_su(x, t, h, c1))
    d1 = D.d1(D.contesto(x, t, h, c1, None))
    assert p == pytest.approx(float((d1.v > x[t]).mean())) and 0.0 <= p <= 1.0


def test_conditional_mean_is_ols():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 4))
    y = 0.1 + X @ np.array([0.02, -0.01, 0.0, 0.03]) + 0.05 * rng.standard_normal(300)
    x_new = np.array([0.5, -0.2, 0.3, 1.0])
    beta = np.linalg.lstsq(np.column_stack([np.ones(300), X]), y, rcond=None)[0]
    assert KA.media_condizionata(X, y, x_new) == pytest.approx(float(np.concatenate([[1.0], x_new]) @ beta))
    assert KA.media_condizionata(X, y, np.array([-50.0, 0, 0, 0])) == 0.0                              # mai negativa


def test_amplitude_choice_rule():
    """La regressione vale solo se ha errore quadratico piu' basso della media storica E Diebold-Mariano con p < 0,05."""
    rng = np.random.default_rng(5)
    n = 400
    y = np.abs(rng.standard_normal(n))
    f = pd.DataFrame({"idx_origine": np.arange(n), "corpo_reale": y, "ombra_sup_reale": y, "ombra_inf_reale": y,
                      "corpo_reg": y + 0.05 * rng.standard_normal(n), "corpo_media": np.full(n, y.mean()),                 # regressione molto migliore
                      "ombra_sup_reg": np.full(n, y.mean()) + 0.001 * rng.standard_normal(n), "ombra_sup_media": np.full(n, y.mean()),   # indistinguibile
                      "ombra_inf_reg": y.mean() + rng.standard_normal(n), "ombra_inf_media": np.full(n, y.mean())})        # regressione peggiore
    tab, scelte = VK.scegli_ampiezze({1: f})
    assert scelte == {"corpo": {1: "regressione"}, "ombra_sup": {1: "media"}, "ombra_inf": {1: "media"}}
    assert set(tab["scelta"]) == {"regressione", "media"} and (tab["n"] == n).all()


def test_candle_walk_forward_matches_program(df):
    path = VK.OUT / "p0_previsioni_dev.csv"
    if not path.exists():
        pytest.skip("eseguire prima python -m src.run_phase3")
    info = json.loads((VK.OUT / "p0_info.json").read_text())
    scelte = {c: {int(h): v for h, v in d.items()} for c, d in info["scelte"].items()}
    w = pd.read_csv(path, parse_dates=["data_origine"])
    t = 1000
    g = w[w["data_origine"] == df["date"].iloc[t]].set_index("h")
    _, cand = KA.prevedi_candele(df, t, scelte)
    for c in cand:
        for campo in ("apertura", "massimo", "minimo", "chiusura"):
            assert c[campo] == pytest.approx(g.loc[c["h"], f"K2m_{campo}"], abs=3e-4), (c["h"], campo)
        assert c["p_chiusura_su"] == pytest.approx(g.loc[c["h"], "p_chiusura_su"], abs=1e-5)
    assert w["data_obiettivo"].max() <= str(df["date"].max().date())
    assert info["modello_direzione"] == "L0"


def test_registry_keeps_label_and_live_check_scores_it(tmp_path):
    prev = {"data_origine": "2024-01-05", "chiusura_origine": 13.0,
            "settimane": [{"h": h, "data": f"2024-01-{5 + 7 * h:02d}", "quantili": {"q50": 13.5}, "fasce": {"50": [12, 15], "80": [11, 17], "90": [10, 19]},
                           "probabilita": {"VIX>20": 0.1, "VIX>25": 0.05, "VIX>30": 0.01}} for h in (1, 2)]}
    base = {"apertura": 13.4, "massimo": 14.5, "minimo": 12.5, "chiusura": 12.9, "direzione": "ribasso", "prob_direzione": 0.64, "costruzione": "K2m", "modello_direzione": "L0"}
    cand = [base | {"p_chiusura_su": 0.7, "direzione_chiusura": "rialzo"}, base | {"p_chiusura_su": 0.5, "direzione_chiusura": "neutra"}]
    nuove = RG3.righe(prev, cand, "2024-01-06")
    assert list(nuove["etichetta_chiusura"]) == ["rialzo", "neutra"] and list(nuove["p_chiusura_sopra_origine"]) == [0.7, 0.5]
    storia = pd.DataFrame({"date": pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"]), "open": [13.0, 13.4, 15.0], "high": [14.0, 16.0, 15.5],
                           "low": [12.0, 12.9, 12.0], "close": [13.0, 15.5, 12.5]})
    det, pag = RG3.verifica(nuove, storia)
    assert list(det["etichetta_non_neutra"]) == [True, False] and bool(det["etichetta_corretta"].iloc[0]) and pd.isna(det["etichetta_corretta"].iloc[1])
    assert list(pag["n_etichette_non_neutre"]) == [1, 0] and pag["etichette_corrette"].iloc[0] == 1.0
    vecchie = nuove.drop(columns=["p_chiusura_sopra_origine", "etichetta_chiusura"])              # registro scritto prima della Fase 3
    det2, pag2 = RG3.verifica(vecchie, storia)
    assert len(det2) == 2 and "etichetta_corretta" not in det2.columns


# ------------------------------------------------------------------ variabili: dati sintetici

def _sintetico(n_sett: int = 420, seed: int = 7):
    """Calendario sintetico: 5 sedute a settimana (lun-ven), data settimanale = venerdi'."""
    rng = np.random.default_rng(seed)
    giorni = pd.bdate_range("2010-01-04", periods=5 * n_sett)
    def cammino(s0, vol):
        return s0 * np.exp(np.cumsum(vol * rng.standard_normal(len(giorni))))
    vix = 18 * np.exp(0.25 * np.sin(np.arange(len(giorni)) / 37) + 0.05 * rng.standard_normal(len(giorni)))
    g = pd.DataFrame({"vix": vix, "vix3m": vix * (1.08 + 0.03 * rng.standard_normal(len(giorni))), "vvix": cammino(90, 0.02),
                      "spx": cammino(1100, 0.01), "vx1": vix * (1.04 + 0.02 * rng.standard_normal(len(giorni))),
                      "vx2": vix * (1.07 + 0.02 * rng.standard_normal(len(giorni)))}, index=giorni)
    date = giorni[4::5]
    return date, g["vix"].to_numpy()[4::5].copy(), g


def test_variables_match_their_definitions_on_synthetic_data():
    date, close, g = _sintetico()
    X = V.costruisci(date, close, g)
    assert list(X.columns) == list(V.TUTTE) and len(X) == len(date)
    t = 300
    d = date[t]
    v = g["vix"][:d]
    lv, s = np.log(v), g["spx"][:d]
    dv = lv.diff()
    assert X["g_var1"][t] == pytest.approx(lv.iloc[-1] - lv.iloc[-2]) and X["g_var3"][t] == pytest.approx(lv.iloc[-1] - lv.iloc[-4])
    assert X["g_sd5"][t] == pytest.approx(dv.iloc[-5:].std()) and X["g_sd20"][t] == pytest.approx(dv.iloc[-20:].std())
    assert X["g_max_var5"][t] == pytest.approx(dv.iloc[-5:].max())
    assert X["g_chiusura_su_media"][t] == pytest.approx(np.log(close[t] / v.iloc[-5:].mean()))
    u = v.iloc[-10:].to_numpy()
    assert X["g_sedute_dal_max10"][t] == 9 - int(np.flatnonzero(u == u.max())[-1])
    assert X["x1_vix3m_su_vix"][t] == pytest.approx(g["vix3m"][d] / g["vix"][d] - 1)
    assert X["x2_vx1_su_vix"][t] == pytest.approx(g["vx1"][d] / g["vix"][d] - 1) and X["x2_vx2_su_vx1"][t] == pytest.approx(g["vx2"][d] / g["vx1"][d] - 1)
    assert X["x3_log_vvix"][t] == pytest.approx(np.log(g["vvix"][d]))
    assert X["x4_rend1"][t] == pytest.approx(np.log(s.iloc[-1] / s.iloc[-6])) and X["x4_rend4"][t] == pytest.approx(np.log(s.iloc[-1] / s.iloc[-21]))
    r = np.log(s).diff().iloc[-20:]
    vol = 100 * np.sqrt(252 * (r ** 2).mean())
    assert X["x4_volreal4"][t] == pytest.approx(vol) and X["x4_vix_meno_volreal"][t] == pytest.approx(close[t] - vol)
    assert X["x4_dist_max52"][t] == pytest.approx(np.log(s.iloc[-1] / s[s.index > d - pd.Timedelta(days=364)].max())) and X["x4_dist_max52"].max() <= 0
    assert np.isnan(X["x4_rend4"][:4]).all() and np.isfinite(X.iloc[60:].to_numpy()).all()


def test_variables_ignore_everything_after_the_week():
    """Riga t invariata se si tolgono o si stravolgono i dati giornalieri successivi all'ultima seduta della settimana t."""
    date, close, g = _sintetico()
    X = V.costruisci(date, close, g).to_numpy()
    for t in (40, 200, 419):
        tronco = V.costruisci(date[: t + 1], close[: t + 1], g[g.index <= date[t]]).to_numpy()
        np.testing.assert_allclose(tronco, X[: t + 1], rtol=0, atol=1e-12, equal_nan=True)
    t = 250
    g2, c2 = g.copy(), close.copy()
    g2.loc[g2.index > date[t]] = g2.loc[g2.index > date[t]] * 3.0 + 7.0
    c2[t + 1:] = c2[t + 1:] * 3.0 + 7.0
    Y = V.costruisci(date, c2, g2).to_numpy()
    np.testing.assert_allclose(Y[: t + 1], X[: t + 1], rtol=0, atol=1e-12, equal_nan=True)
    assert not np.allclose(Y[t + 1:t + 60], X[t + 1:t + 60], equal_nan=True)                       # il test "morde": dopo t tutto cambia
    g3 = g.copy()                                                                                  # anche l'ULTIMA seduta della settimana t conta
    g3.loc[date[t]] = g3.loc[date[t]] * 1.5
    assert not np.allclose(V.costruisci(date, close, g3).to_numpy()[t], X[t], equal_nan=True)


def test_missing_values_are_never_carried_forward():
    date, close, g = _sintetico()
    t = 200
    g2 = g.copy()
    sett = (g2.index > date[t - 1]) & (g2.index <= date[t])
    g2.loc[sett, "vvix"] = np.nan                                  # nessun valore nella settimana t: variabile mancante
    g2.loc[g2.index == date[t], ["vix3m", "spx"]] = np.nan         # manca solo l'ultima seduta: vale la seduta precedente della stessa settimana
    g2.loc[g2.index < date[60], "vx1"] = np.nan                    # serie che inizia dopo
    X, X0 = V.costruisci(date, close, g2), V.costruisci(date, close, g)
    assert np.isnan(X["x3_log_vvix"][t]) and np.isfinite(X["x3_log_vvix"][t - 1]) and np.isfinite(X["x3_log_vvix"][t + 1])
    gio = g.index[g.index < date[t]][-1]
    assert X["x1_vix3m_su_vix"][t] == pytest.approx(g["vix3m"][gio] / g["vix"][gio] - 1)           # numeratore e denominatore dello stesso giorno
    assert X["x4_rend1"][t] == pytest.approx(np.log(g["spx"][gio] / g["spx"][date[t - 1]]))
    assert np.isnan(X["x2_vx1_su_vix"][:60]).all() and np.isnan(X["x2_vx2_su_vx1"][:60]).all() and np.isfinite(X["x2_vx1_su_vix"][60])
    assert X["g_var1"].equals(X0["g_var1"])
    inv = V.inventario(X, date).set_index("variabile")
    assert inv.loc["x3_log_vvix", "settimane_mancanti_dopo_l_inizio"] == 1 and inv.loc["x2_vx1_su_vix", "prima_settimana_valida"] == str(date[60].date())


def test_excluded_series_are_not_read():
    assert set(dati.SERIE) == {"vix", "vix3m", "vvix", "spx", "vx1", "vx2"}
    assert {s for v in V.SERIE_PER_BLOCCO.values() for s in v} == set(dati.SERIE)
    for f in sorted(DIR.glob("*.py")):
        assert not re.search(r"nfci|oas_|dgs\d", f.read_text()), f.name
    assert tuple(V.BLOCCHI) == ("G1", "X1", "X2", "X3", "X4") and M.MODELLI == ("G1", "X1", "X2", "X3", "X4", "X6", "X7") and len(V.TUTTE) == 16


# ------------------------------------------------------------------ modelli

@pytest.fixture(scope="module")
def sint():
    date, close, g = _sintetico()
    X = V.costruisci(date, close, g)
    X.iloc[:30, X.columns.get_loc("x1_vix3m_su_vix")] = np.nan                                     # un blocco che inizia dopo
    return np.log(close), X


FUNZIONI = [("t1", M.t1, {}), ("t1_m1_stessa_finestra", M.t1, {"usa": []}), ("t1_ridge", M.t1_ridge, {}), ("t2", M.t2, {}), ("t2_ridge", M.t2_ridge, {}),
            ("t3", lambda x, Z, h: M.t3(x, Z, h)[0], {})]


@pytest.mark.parametrize("nome,fn,kw", FUNZIONI)
@pytest.mark.parametrize("h", M.ORIZZONTI)
def test_models_ignore_data_after_origin(sint, nome, fn, kw, h):
    """Stravolgere x e le variabili da k in poi non cambia nessuna previsione con origine < k."""
    x, X = sint
    cols = list(V.TUTTE if "ridge" in nome else V.BLOCCHI["X1"] + V.BLOCCHI["G1"])
    Z = X[cols].to_numpy()
    k = 380
    x2, Z2 = x.copy(), Z.copy()
    x2[k:], Z2[k:] = x2[k:] * 1.3 + 0.5, Z2[k:] * 3.0 + 1.0
    o = M.origini(Z, h)
    o = o[o >= 370] if nome == "t3" else o                                                         # la regressione quantilica e' lenta: poche origini
    a = np.array([fn(x[: t + 1].copy(), Z[: t + 1].copy(), h, **kw) for t in o.tolist()])
    b = np.array([fn(x2[: t + 1].copy(), Z2[: t + 1].copy(), h, **kw) for t in o.tolist()])
    prima = o < k
    assert prima.sum() >= 5 and (~prima).sum() >= 5
    np.testing.assert_array_equal(a[prima], b[prima])
    assert not np.allclose(a[~prima], b[~prima])                                                   # il test "morde"
    np.testing.assert_array_equal(M.origini(Z2, h)[M.origini(Z2, h) < k], o[prima] if nome != "t3" else M.origini(Z, h)[M.origini(Z, h) < k])


def test_walk_passes_only_history_and_respects_admission_rule(sint):
    x, X = sint
    Z = X[list(V.BLOCCHI["X1"])].to_numpy()
    visti = []
    o, _ = M.walk(x, Z, 4, lambda xs, Zs, h: visti.append((len(xs), len(Zs))) or 0.0)
    assert visti == [(t + 1, t + 1) for t in o.tolist()] and o.max() == len(x) - 5
    assert o.min() == 30 + M.MIN_VALIDE - 1                         # il blocco inizia alla settimana 30: servono 249 settimane valide
    Z0 = np.empty((len(x), 0))
    assert M.origini(Z0, 1).min() == C.MIN_TRAIN_WEEKS - 1          # senza variabili aggiuntive: la regola di M1
    Zb = Z.copy()
    Zb[300] = np.nan                                                # origine con variabile mancante: esclusa, e non usata nelle stime
    assert 300 not in M.origini(Zb, 1) and 299 in M.origini(Zb, 1)
    assert M.t1(x[:351], Zb[:351], 1) != M.t1(x[:351], Z[:351], 1)


def test_references_coincide_with_m1_e1_style_logit_and_d4(sint):
    x, _ = sint
    Z0 = np.empty((len(x), 0))
    for h in M.ORIZZONTI:
        t = 400
        assert M.t1(x[: t + 1], Z0[: t + 1], h) == pytest.approx(B.forecast_har(x[: t + 1], h), abs=1e-12)
        F_ = B.har_features(x[: t + 1])[:, 1:]
        i = np.arange(11, t - h + 1)
        y = (x[i + h] > x[i]).astype(float)
        assert M.t2(x[: t + 1], Z0[: t + 1], h) == pytest.approx(CP.logistica(F_[: len(i)], y, F_[-1]), abs=1e-10)
        assert M.t2_frequenza(x[: t + 1], h) == pytest.approx(np.mean(x[h:t + 1] > x[:t + 1 - h]))
        q, _ = M.t3(x[: t + 1], Z0[: t + 1], h)
        np.testing.assert_allclose(q, D.d4(D.Contesto(x[: t + 1], h, None, None)).quantiles(), atol=1e-10)


def test_block_model_is_ols_on_m1_plus_block_and_ridge_shrinks_towards_m1(sint):
    x, X = sint
    Z = X[list(V.BLOCCHI["X4"])].to_numpy()
    t, h = 400, 4
    xs, Zs = x[: t + 1], Z[: t + 1]
    i = np.arange(11, t - h + 1)
    i = i[np.isfinite(Zs[i]).all(axis=1)]
    Fh = B.har_features(xs)
    A = np.column_stack([Fh[i - 11], Zs[i]])
    beta = np.linalg.lstsq(A, xs[i + h], rcond=None)[0]
    f, coef = M.t1(xs, Zs, h, coef=True)
    assert f == pytest.approx(float(np.concatenate([Fh[-1], Zs[-1]]) @ beta)) and len(coef) == 3 + Z.shape[1]
    assert M.t1(xs, Zs, h, usa=[]) == pytest.approx(float(Fh[-1] @ np.linalg.lstsq(Fh[i - 11], xs[i + h], rcond=None)[0]))
    Zstd = (Zs - Zs[i].mean(axis=0)) / Zs[i].std(axis=0)
    forte = M._ridge(Fh[i - 11], Zstd[i], xs[i + h], 1e9)
    assert np.abs(forte[4:]).max() < 1e-5 and np.allclose(forte[:4], np.linalg.lstsq(Fh[i - 11], xs[i + h], rcond=None)[0], atol=1e-4)
    debole = M._ridge(Fh[i - 11], Zstd[i], xs[i + h], 1e-10)
    assert float(np.concatenate([Fh[-1], Zstd[-1]]) @ debole) == pytest.approx(f, abs=1e-6)
    f6, g, _ = M.t1_ridge(xs, Zs, h, dettagli=True)
    p6, g2, _ = M.t2_ridge(xs, Zs, h, dettagli=True)
    assert g in M.GRIGLIA_RIDGE and g2 in M.GRIGLIA_RIDGE and np.isfinite(f6) and 0.0 < p6 < 1.0 and 0.0 < M.t2(xs, Zs, h) < 1.0
    tr, val = M._divisione(i, h)
    assert len(val) == M.VALIDAZIONE and (tr + h).max() <= val[0]                                # la scelta della penalita' non usa esiti successivi


# ------------------------------------------------------------------ metriche e regole di scelta

def test_t1_and_t2_metrics_known_values():
    y = np.log(np.array([20.0, 20.0, 20.0, 20.0] * 5))
    g = pd.DataFrame({"idx_origine": np.arange(20), "data_obiettivo": pd.date_range("2020-01-03", periods=20, freq="7D"), "y": y,
                      "f_A": np.log(np.array([21.0, 19.0, 25.0, 20.0] * 5)), "f_M1": np.log(np.array([23.0, 18.0, 26.0, 21.0] * 5)),
                      "f_M1fin_A": np.log(np.array([23.0, 18.0, 26.0, 21.0] * 5))})
    r = VA.t1_riga(g, "A", 1)
    assert r["n"] == 20 and r["entro_10"] == 0.75 and r["entro_20"] == 0.75 and r["entro_10_m1"] == 0.5 and r["entro_20_m1"] == 0.75
    assert r["mae_vix"] == pytest.approx(1.75) and r["mae_vix_m1"] == pytest.approx(3.0) and 0 < r["r2_vs_m1"] < 1
    assert r["r2_vs_m1"] == pytest.approx(r["r2_vs_m1_stessa_finestra"]) and r["cw_stat"] > 0
    su = np.array([1, 1, 0, 0, 1] * 4, dtype=float)
    g2 = pd.DataFrame({"idx_origine": np.arange(20), "data_obiettivo": g["data_obiettivo"], "su": su, "p_A": np.array([0.9, 0.62, 0.3, 0.45, 0.55] * 4),
                       "p_BASE": np.full(20, 0.5), "p_FREQ": np.full(20, 0.6)})
    r2 = VA.t2_riga(g2, "A", 1)
    assert r2["accuratezza"] == 1.0 and r2["accuratezza_freq"] == 0.6 and r2["brier_base"] == 0.25
    assert (r2["n_60"], r2["n_65"], r2["n_70"]) == (12, 8, 8) and r2["quota_60"] == 0.6 and r2["accuratezza_70"] == 1.0
    assert r2["skill_vs_base"] == pytest.approx(1 - r2["brier"] / 0.25) and r2["skill_vs_base"] > 0
    assert "skill_vs_base" not in VA.t2_riga(g2, "BASE", 1) and "skill_vs_freq" not in VA.t2_riga(g2, "FREQ", 1)


def _tab(vals):
    return pd.DataFrame([{"modello": m, "h": h, "stat": s, "p": p, "g": g} for m, r in vals.items() for h, (s, p, g) in zip((1, 4), r)])


def test_selection_rule():
    nessuno = {m: [(1.0, 0.5, 0.01), (0.5, 0.9, 0.02)] for m in M.MODELLI}
    assert VA.scegli(_tab(nessuno), "stat", "p", "g")["scelto"] is None                           # nessun significativo: resta il riferimento
    v = dict(nessuno) | {"X1": [(3.0, 0.01, 0.02), (1.0, 0.4, 0.01)], "X2": [(1.0, 0.3, 0.05), (2.9, 0.02, 0.03)], "X4": [(0.5, 0.6, 0.30), (0.2, 0.8, 0.30)]}
    s = VA.scegli(_tab(v), "stat", "p", "g")
    assert s["significativi"] == ["X1", "X2"] and s["scelto"] == "X2"                              # tra i significativi, guadagno medio piu' alto (X4 non lo e')
    assert s["significativo_per_orizzonte"] == {"X1": [1], "X2": [4]}
    peggio = dict(nessuno) | {"G1": [(-3.0, 0.01, -0.05), (0.1, 0.9, 0.0)]}                        # p piccolo ma nella direzione sbagliata
    assert VA.scegli(_tab(peggio), "stat", "p", "g")["scelto"] is None


# ------------------------------------------------------------------ dati reali di sviluppo (se disponibili) e guardia

def test_real_variables_use_only_data_up_to_each_week():
    if not dati.esterni_disponibili():
        pytest.skip("dati esterni di sviluppo assenti: eseguire python -m src.fetch_external")
    with K.guardia_dati():
        w, g = dati.load_settimanale(), dati.load_giornaliero()
    assert list(g.columns) == list(dati.SERIE) and g.index.max() <= w["date"].max() and not any(dati.ritardi().values())
    X = V.costruisci(w["date"], w["close"], g)
    for t in (30, 520, 1296):
        tronco = V.costruisci(w["date"].iloc[: t + 1], w["close"].iloc[: t + 1], g[g.index <= w["date"].iloc[t]])
        np.testing.assert_allclose(tronco.to_numpy(), X.iloc[: t + 1].to_numpy(), rtol=0, atol=1e-12, equal_nan=True)
    assert np.isfinite(X[list(V.BLOCCHI["G1"])].to_numpy()).all()                                 # il VIX giornaliero copre tutte le settimane
    path = VA.OUT / "previsioni_dev.csv"
    if not path.exists():
        pytest.skip("eseguire prima python -m src.run_phase3")
    p = pd.read_csv(path, parse_dates=["data_origine"])
    x = w["log_close"].to_numpy()
    for t, h in ((900, 1), (1200, 4)):
        r = p[(p["data_origine"] == w["date"].iloc[t]) & (p["h"] == h)].iloc[0]
        Z = X[list(V.BLOCCHI["X1"])].to_numpy()
        assert M.t1(x[: t + 1], Z[: t + 1], h) == pytest.approx(r["f_X1"], abs=2e-6) and M.t2(x[: t + 1], Z[: t + 1], h) == pytest.approx(r["p_X1"], abs=2e-6)
        assert B.forecast_har(x[: t + 1], h) == pytest.approx(r["f_M1"], abs=2e-6)
        medie = np.mean([M.t1(x[: t + 1], X[list(V.BLOCCHI[b])].to_numpy()[: t + 1], h) for b in M.BLOCCHI])
        assert medie == pytest.approx(r["f_X7"], abs=2e-6)
    assert p["data_obiettivo"].max() <= str(w["date"].max().date())


def test_phase3_modules_read_only_dev_and_run_under_guard():
    banned = r"HOLDOUT_FILE|CLEAN_FILE|RAW_FILE|vix_weekly_holdout|vix_weekly_clean|VIX_settimanale|holdout[\"'/]|[\"']raw[\"']"
    for f in sorted(DIR.glob("*.py")) + [pathlib.Path(C.ROOT) / "src" / "run_phase3.py"]:
        src = f.read_text()
        assert not re.search(banned, src), f.name
        if f.name != "dati.py":
            assert not re.search(r"[\"']external[\"']|read_csv\(C\.", src), f"{f.name}: i dati si leggono con fase3.dati o fase2b.dati"
    assert '"external" / "dev"' in (DIR / "dati.py").read_text()
    assert "with K.guardia_dati():" in (pathlib.Path(C.ROOT) / "src" / "run_phase3.py").read_text()
    with K.guardia_dati():
        for path in (C.HOLDOUT_FILE, C.ROOT / "data" / "external" / "holdout" / "esterni_giornalieri_holdout.csv", C.ROOT / "data" / "external" / "raw" / "cboe" / "VIX_History.csv"):
            with pytest.raises(PermissionError):
                open(path)
