"""Fase 4: varianti delle ampiezze delle candele, modello a regimi N1, gradient boosting N2/N3, versioni finaliste
F1/F2/F3, configurazioni congelate, programma con --versione e --aggiorna, registro con la versione.

Anti look-ahead di ogni modello nuovo; regole di scelta; ripiego di f2; guardia sull'holdout.
I test che richiedono file prodotti da `run_phase4` o i dati esterni di sviluppo vengono saltati se mancano."""
import json
import pathlib
import re

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import baselines as B
from src import config as C
from src import forecast as F
from src.fase1 import comune as K
from src.fase2 import densita as D
from src.fase2b import dati as dati2b, modello as MO
from src.fase2c import componenti as CP
from src.fase3 import candele as K3, dati as dati3, modelli as M, variabili as V
from src.fase4 import boosting as GB, candele as KA, congela, esterni as ES, regimi as RE, registro as RG, valuta_a as VA4, valuta_b as VB4, valuta_c as VC4, versioni as VS

DIR = pathlib.Path(C.ROOT) / "src" / "fase4"
OUT = C.REPORTS_DIR / "fase4"


@pytest.fixture(scope="module")
def df():
    return dati2b.load_dev_candele()


def _stravolgi(df, t):
    d = df.copy()
    for c in ("open", "high", "low", "close"):
        d.loc[d.index > t, c] = d.loc[d.index > t, c] * 3.0 + 7.0
    d["log_close"] = np.log(d["close"])
    return d


def _ar_sintetico(n=345, seed=3):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 3.0
    vol = np.where(rng.random(n) < 0.25, 0.20, 0.07)                 # settimane tranquille e agitate
    for t in range(1, n):
        x[t] = 3.0 + 0.93 * (x[t - 1] - 3.0) + vol[t] * rng.standard_normal()
    return x


# ------------------------------------------------------------------ Parte A: candele

@pytest.mark.parametrize("variante", list(KA.VARIANTI))
def test_amplitude_variants_use_their_windows_and_ignore_future(df, variante):
    S, t = CP.serie(df), 1000
    for h in KA.ORIZZONTI:
        a = KA.componenti(S, t, h, variante)
        assert a == KA.componenti(CP.serie(_stravolgi(df, t)), t, h, variante) == KA.componenti(CP.serie(df.iloc[: t + 1]), t, h, variante)
        assert a != KA.componenti(CP.serie(_stravolgi(df, t - 30)), t, h, variante)                    # il test "morde"
        G = CP.grandezze(S, t)
        i = np.arange(11, t - h + 1)
        i = i[~G["bad"][i] & ~G["bad"][i + h]]
        w = KA.VARIANTI[variante]
        im = i if w["media"] is None else i[-w["media"]:]
        ir = i if w["regressione"] is None else i[-w["regressione"]:]
        assert a["n_media"] == len(im) == (len(i) if w["media"] is None else w["media"]) and a["n_regressione"] == len(ir)
        assert a["ombra_sup"] == pytest.approx(G["ombra_sup"][im + h].mean()) and a["ombra_inf"] == pytest.approx(G["ombra_inf"][im + h].mean())
        if KA.ASSEGNAZIONE["corpo"][h] == "regressione":
            Xa = np.column_stack([G["har"][ir - 11][:, 1:], G["extra"][ir]])
            assert a["corpo"] == pytest.approx(K3.media_condizionata(Xa, G["corpo"][ir + h], np.concatenate([G["har"][-1][1:], G["extra"][t]])))
        else:
            assert a["corpo"] == pytest.approx(G["corpo"][im + h].mean())


def test_a0_is_the_phase3_candle_and_assignment_matches_phase3(df):
    S, t = CP.serie(df), 900
    prev = MO.prevedi(df, t)
    cen = {s["h"]: s["chiusura"] for s in prev["settimane"]}
    a0 = KA.candele(S, t, prev["chiusura_origine"], cen, "A0")
    _, k3 = K3.prevedi_candele(df, t, KA.ASSEGNAZIONE)
    for a, b in zip(a0, k3):
        assert all(a[c] == pytest.approx(b[c]) for c in KA.CAMPI) and a["costruzione"] == "K2m" == b["costruzione"]
    assert KA.costruzione("A1") == "K2m-A1" and tuple(KA.VARIANTI) == ("A0", "A1", "A2")
    assert KA.VARIANTI["A1"] == {"media": 260, "regressione": 520} and KA.VARIANTI["A2"] == {"media": 104, "regressione": 520}
    p0 = C.REPORTS_DIR / "fase3" / "p0_info.json"
    if p0.exists():
        sc = json.loads(p0.read_text())["scelte"]
        assert {c: {int(h): v for h, v in d.items()} for c, d in sc.items()} == KA.ASSEGNAZIONE


def _pag(rap):
    return pd.DataFrame([{"variante": v, "h": h, "rapporto_corpo": r} for v, r in rap.items() for h in (1, 4)])


def _f1(err):
    n = 400
    rng = np.random.default_rng(1)
    y = np.abs(rng.standard_normal(n)) * 0.1
    f = pd.DataFrame({f"{c}_reale": y for c in KA.COMPONENTI})
    for v, sd in err.items():
        e = sd * rng.standard_normal(n)
        for c in KA.COMPONENTI:
            f[f"{v}_{c}"] = y + e
    return f


def test_variant_rule():
    f = _f1({"A0": 0.02, "A1": 0.02, "A2": 0.02})
    s = VA4.scegli_variante(_pag({"A0": 0.85, "A1": 0.91, "A2": 0.95}), f)
    assert s["candidata"] == "A2" and s["scelta"] == "A2" and not s["significativamente_peggiore"]       # piu' vicina a 1 e non peggiore
    s = VA4.scegli_variante(_pag({"A0": 0.85, "A1": 0.91, "A2": 0.91}), f)
    assert s["candidata"] == "A1" and s["scelta"] == "A1"                                               # parita': la prima dell'elenco
    s = VA4.scegli_variante(_pag({"A0": 0.99, "A1": 0.91, "A2": 1.20}), f)
    assert s["scelta"] == "A0" and "dm_stat" not in s                                                   # A0 e' gia' la piu' vicina
    s = VA4.scegli_variante(_pag({"A0": 0.85, "A1": 0.91, "A2": 0.99}), _f1({"A0": 0.01, "A1": 0.01, "A2": 0.08}))
    assert s["candidata"] == "A2" and s["significativamente_peggiore"] and s["scelta"] == "A0"           # significativamente peggiore: resta A0 (non la seconda)


def test_candle_scorecard_file_matches_single_origin(df):
    path = OUT / "a_previsioni_dev.csv"
    if not path.exists():
        pytest.skip("eseguire prima python -m src.run_phase4")
    w = pd.read_csv(path, parse_dates=["data_origine"])
    S, t = CP.serie(df), 1010
    g = w[w["data_origine"] == df["date"].iloc[t]].set_index("h")
    cen = {int(h): float(g.loc[h, "mediana_d4"]) for h in g.index}
    for v in KA.VARIANTI:
        for c in KA.candele(S, t, float(S.c[t]), cen, v):
            for campo in KA.CAMPI:
                assert c[campo] == pytest.approx(g.loc[c["h"], f"{v}_{campo}"], abs=3e-4), (v, c["h"], campo)
    info = json.loads((OUT / "a_info.json").read_text())
    assert info["scelta"]["scelta"] in KA.VARIANTI and w["data_obiettivo"].max() <= str(df["date"].max().date())


# ------------------------------------------------------------------ Parte B: N1

PAR = {"P": np.array([[0.9, 0.1], [0.3, 0.7]]), "mu": np.array([2.7, 3.2]), "s2": np.array([0.005, 0.03]), "phi": 0.9}


@pytest.mark.parametrize("h", (1, 4))
def test_regime_forecast_matches_simulation(h):
    rng = np.random.default_rng(8)
    pi, x_t, n = np.array([0.35, 0.65]), 3.05, 400_000
    s = (rng.random(n) > pi[0]).astype(int)
    dev = x_t - PAR["mu"][s]
    for _ in range(h):
        s = (rng.random(n) > PAR["P"][s, 0]).astype(int)
        dev = PAR["phi"] * dev + np.sqrt(PAR["s2"][s]) * rng.standard_normal(n)
    sim = PAR["mu"][s] + dev
    w, m, sd = RE.distribuzione(x_t, pi, PAR, h)
    assert w.sum() == pytest.approx(1.0) and len(w) == 2 ** (h + 1)
    media, p_su = RE.previsione(np.array([3.0, x_t]), PAR, h, pi)
    assert media == pytest.approx(sim.mean(), abs=2e-3) and p_su == pytest.approx((sim > x_t).mean(), abs=4e-3)
    mix = VS.Mistura(w, m, sd)
    q = mix.quantiles([0.05, 0.5, 0.95])
    np.testing.assert_allclose(q, np.quantile(sim, [0.05, 0.5, 0.95]), atol=6e-3)
    np.testing.assert_allclose(mix.cdf(q), [0.05, 0.5, 0.95], atol=1e-9)
    assert mix.prob_above(x_t) == pytest.approx(p_su) and mix.mean == pytest.approx(media)


def test_regime_model_with_equal_regimes_is_an_ar1():
    par = {"P": np.array([[0.8, 0.2], [0.4, 0.6]]), "mu": np.array([3.0, 3.0]), "s2": np.array([0.01, 0.01]), "phi": 0.95}
    x = np.array([2.9, 3.3])
    for h in (1, 4):
        media, p = RE.previsione(x, par, h, np.array([0.5, 0.5]))
        sd = np.sqrt(0.01 * sum(0.95 ** (2 * k) for k in range(h)))
        assert media == pytest.approx(3.0 + 0.95 ** h * 0.3) and p == pytest.approx(stats.norm.sf((3.3 - media) / sd))
    a, phi = np.linalg.lstsq(np.column_stack([np.ones(len(_ar_sintetico()) - 1), _ar_sintetico()[:-1]]), _ar_sintetico()[1:], rcond=None)[0]
    f, p = RE.ar1(_ar_sintetico(), 1)
    assert f == pytest.approx(a + phi * _ar_sintetico()[-1]) and 0 < p < 1 and RE.ar1(_ar_sintetico(), 4)[0] == pytest.approx(B.forecast_ar1(_ar_sintetico(), 4))


def test_regime_walk_schedule_and_no_lookahead():
    x = _ar_sintetico()
    assert [RE.origine_di_stima(t) for t in (259, 284, 285, 310, 311)] == [259, 259, 285, 285, 311] and RE.PASSO_STIMA == 26
    w = RE.walk(x, (1, 4))
    assert sorted(w["stime"]) == [259, 285, 311, 337] and [r["idx_stima"] for r in w["ristime"]] == [259, 285, 311, 337]
    assert list(w["previsioni"][1]["origini"]) == list(range(259, len(x) - 1)) and list(w["previsioni"][4]["origini"]) == list(range(259, len(x) - 4))
    k = 320
    x2 = x.copy()
    x2[k:] = x2[k:] * 1.2 + 0.4
    w2 = RE.walk(x2, (1, 4))
    for h in (1, 4):
        prima = w["previsioni"][h]["origini"] < k
        for campo in ("media", "p_su"):
            np.testing.assert_array_equal(w["previsioni"][h][campo][prima], w2["previsioni"][h][campo][prima])
            assert not np.allclose(w["previsioni"][h][campo][~prima], w2["previsioni"][h][campo][~prima])     # il test "morde"
    valide = [ts for ts in sorted(w["stime"]) if w["stime"][ts] is not None]
    assert valide, "nessuna stima valida sul campione sintetico"
    for t in (valide[0] + 3, len(x) - 6):                                # il programma (una sola origine) usa gli stessi parametri del walk-forward
        par = RE.parametri_all_origine(x[: t + 1])
        atteso = [w["stime"][ts] for ts in valide if ts <= t][-1]
        np.testing.assert_allclose(par["params"], atteso["params"])
        j = list(w["previsioni"][4]["origini"]).index(t)
        mix = VS.Mistura(*RE.mistura_all_origine(x[: t + 1], 4, par))
        assert mix.mean == pytest.approx(w["previsioni"][4]["media"][j]) and mix.prob_above(x[t]) == pytest.approx(w["previsioni"][4]["p_su"][j])


def test_regime_walk_uses_last_valid_estimate_and_counts_failures(monkeypatch):
    x = _ar_sintetico()
    vera, chiamate = RE.stima, []

    def finta(xs):
        chiamate.append(len(xs) - 1)
        return None if len(xs) - 1 in (259, 311) else vera(xs)             # prima stima e terza stima non valide

    monkeypatch.setattr(RE, "stima", finta)
    w = RE.walk(x, (1,))
    assert chiamate == [259, 285, 311, 337] and [r["valida"] for r in w["ristime"]] == [False, True, False, True]
    p = pd.DataFrame(w["probabilita"]).set_index("idx_origine")
    assert p.loc[259:284, "p_regime_alto"].isna().all() and p.loc[285:, "p_regime_alto"].notna().all()         # senza stime valide: AR(1)
    assert set(p.loc[285:336, "idx_stima_usata"]) == {285} and set(p.loc[337:, "idx_stima_usata"]) == {337}    # 311 non valida: resta quella del 285
    o = list(w["previsioni"][1]["origini"])
    assert w["previsioni"][1]["media"][o.index(270)] == pytest.approx(RE.ar1(x[:271], 1)[0])


# ------------------------------------------------------------------ Parte B: N2, N3

def test_boosting_hyperparameters_are_the_fixed_ones():
    assert GB.PARAMETRI == {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "random_state": C.SEED} and GB.PASSO_STIMA == 13
    assert VB4.MODELLI == ("N1", "N2", "N3") and GB.NOMI_CANDELA == CP.NOMI_EXTRA


@pytest.mark.parametrize("obiettivo", ("T1", "T2"))
def test_boosting_walk_schedule_target_and_no_lookahead(obiettivo):
    x = _ar_sintetico(300, seed=5)
    rng = np.random.default_rng(2)
    Z = np.column_stack([np.concatenate([[0.0], np.diff(x)]), rng.standard_normal(len(x))])
    Z[:4] = np.nan
    h = 4
    o, v, rs = GB.walk(x, Z, h, obiettivo)
    assert list(o) == list(M.origini(Z, h)) and rs == list(range(int(o[0]), int(o[-1]) + 1, 13))                 # ristima ogni 13 settimane
    t = int(o[5])                                                                                                # tra due ristime il modello resta fermo
    mod = GB.stima(x[: o[0] + 1], Z[: o[0] + 1], h, obiettivo)
    assert v[5] == pytest.approx(GB.previsione(mod, x[: t + 1], Z[: t + 1], obiettivo))
    riga = GB._variabili(x[: t + 1], Z[: t + 1], [t])
    if obiettivo == "T1":
        assert v[5] - x[t] == pytest.approx(mod.predict(riga)[0])                                                # l'albero prevede la variazione
    else:
        assert 0.0 <= v.min() and v.max() <= 1.0 and v[5] == pytest.approx(mod.predict_proba(riga)[0, 1])
    k = int(o[0]) + 20
    x2, Z2 = x.copy(), Z.copy()
    x2[k:], Z2[k:] = x2[k:] * 1.3 + 0.5, Z2[k:] * 3.0 + 1.0
    _, v2, _ = GB.walk(x2, Z2, h, obiettivo)
    prima = o < k
    np.testing.assert_array_equal(v[prima], v2[prima])
    assert not np.allclose(v[~prima], v2[~prima])
    np.testing.assert_array_equal(v, GB.walk(x, Z, h, obiettivo)[1])                                             # seed fisso: stessi numeri


def test_boosting_variables_and_bad_week(df):
    VV = pd.DataFrame(np.zeros((len(df), len(V.TUTTE))), columns=list(V.TUTTE))
    ZZ = VB4.variabili_boosting(df, VV)
    assert ZZ["N2"][0].shape[1] == 6 + len(V.BLOCCHI["G1"]) == len(ZZ["N2"][1]) and ZZ["N3"][0].shape[1] == 6 + len(V.TUTTE)
    bad = np.flatnonzero(CP.serie(df).bad)
    assert len(bad) == 1 and np.isnan(ZZ["N2"][0][bad[0], :6]).all() and not M.valide(ZZ["N2"][0])[bad[0]]
    assert M.valide(ZZ["N2"][0])[bad[0] + 1]


# ------------------------------------------------------------------ Parte C: versioni

def _z(df, inizio=500):
    """X1 sintetica: funzione delle sole chiusure passate piu' rumore con seed fisso; mancante prima di `inizio`."""
    x = np.log(df["close"].to_numpy())
    z = 0.12 - 0.3 * (x - pd.Series(x).rolling(12, min_periods=1).mean().to_numpy()) + 0.02 * np.random.default_rng(4).standard_normal(len(x))
    z[:inizio] = np.nan
    return z


def test_versions_ignore_future_and_share_structure(df):
    z = _z(df)
    t = 1100
    f1 = VS.prevedi_f1(df, t, "A1")
    f2 = VS.prevedi_f2(df, t, z, "A1")
    f3 = VS.prevedi_f3(df, t, "A1")
    d2, z2 = _stravolgi(df, t), z.copy()
    z2[t + 1:] = z2[t + 1:] * 5 + 1
    assert f1 == VS.prevedi_f1(d2, t, "A1") == VS.prevedi_f1(df.iloc[: t + 1].copy(), t, "A1")
    assert f2 == VS.prevedi_f2(d2, t, z2, "A1") == VS.prevedi_f2(df.iloc[: t + 1].copy(), t, z[: t + 1], "A1")
    assert f3 == VS.prevedi_f3(d2, t, "A1") == VS.prevedi_f3(df.iloc[: t + 1].copy(), t, "A1")
    z3 = z.copy()
    z3[t - 40:t + 1] += 0.2                                                                                # il test "morde": X1 passata conta
    assert VS.prevedi_f2(df, t, z3, "A1")[0]["settimane"] != f2[0]["settimane"]
    for v, (prev, cand) in (("f1", f1), ("f2", f2), ("f3", f3)):
        assert prev["versione"] == v and [s["h"] for s in prev["settimane"]] == [1, 2, 3, 4] and len(cand) == 4
        for s, c in zip(prev["settimane"], cand):
            q = [s["quantili"][k] for k in ("q05", "q10", "q25", "q50", "q75", "q90", "q95")]
            assert q == sorted(q) and s["chiusura"] == s["quantili"]["q50"] == c["chiusura_centrale"] and s["fasce"]["80"] == [s["quantili"]["q10"], s["quantili"]["q90"]]
            assert 0 <= s["probabilita"]["VIX>30"] <= s["probabilita"]["VIX>25"] <= s["probabilita"]["VIX>20"] <= 1
            assert (c["direzione_chiusura"], c["etichetta"]) == K3.etichetta(c["p_chiusura_su"]) and c["versione"] == v and c["costruzione"] == "K2m-A1"
            assert c["minimo"] <= min(c["apertura"], c["chiusura"]) <= max(c["apertura"], c["chiusura"]) <= c["massimo"]
        comp = {h: KA.componenti(CP.serie(df), t, h, "A1") for h in KA.ORIZZONTI}                           # K2: ancorata alla chiusura centrale della versione
        for h in (2, 3, 4):
            assert cand[h - 1]["apertura"] == pytest.approx(prev["settimane"][h - 2]["chiusura"] * comp[h]["rapporto_apertura"])
    assert all(f1[1][0][c] == f2[1][0][c] == f3[1][0][c] for c in KA.CAMPI)                                 # la candela +1 e' la stessa in tutte le versioni
    assert f1[0]["settimane"] == MO.prevedi(df, t)["settimane"]                                             # F1 = D4 + D1, invariati
    assert f1[1][3]["p_chiusura_su"] == pytest.approx(K3.prob_chiusura_su(np.log(df["close"].to_numpy()), t, 4))


def test_f2_follows_its_definition_and_falls_back_when_x1_is_missing(df):
    z, t, h = _z(df), 1050, 4
    x, Z = np.log(df["close"].to_numpy()[: t + 1]), _z(df)[: t + 1].reshape(-1, 1)
    prev, cand = VS.prevedi_f2(df, t, z, "A2")
    s = prev["settimane"][h - 1]
    o, c = VS.centri_f2(x, Z, h)
    assert (o + h <= t).all() and o.min() == 500 + M.MIN_VALIDE - 1 and len(o) >= D.MIN_ERR
    centro = M.t1(x, Z, h)
    assert s["centro"] == pytest.approx(np.exp(centro)) and s["n_errori"] == len(o) and cand[0]["costruzione"] == "K2m-A2"
    err = x[o + h] - c
    assert cand[h - 1]["p_chiusura_su"] == pytest.approx(np.mean(centro + err > x[t])) and s["probabilita"]["VIX>20"] == pytest.approx(np.mean(centro + err > np.log(20)))
    q, _ = M.t3(x, Z, h)
    assert s["quantili"]["q50"] == pytest.approx(np.exp(q[list(np.round(D.TAUS, 2)).index(0.5)])) and prev["x1_origine"] == pytest.approx(z[t])
    z_buco = z.copy()
    z_buco[t] = np.nan
    assert VS.f2_ammessa(z, t) and not VS.f2_ammessa(z_buco, t) and not VS.f2_ammessa(z, 500 + M.MIN_VALIDE - 2)
    with pytest.raises(VS.NonDisponibile):
        VS.prevedi_f2(df, t, z_buco, "A1")
    with pytest.raises(VS.NonDisponibile):                                                               # origine ammessa ma meno di 104 errori passati
        VS.prevedi_f2(df, 500 + M.MIN_VALIDE + 20, z, "A1")


def test_x1_weekly_matches_phase3_variable():
    giorni = pd.bdate_range("2012-01-02", periods=5 * 300)
    rng = np.random.default_rng(6)
    vix = 18 * np.exp(0.2 * np.sin(np.arange(len(giorni)) / 40) + 0.04 * rng.standard_normal(len(giorni)))
    g = pd.DataFrame({"vix": vix, "vix3m": vix * (1.08 + 0.03 * rng.standard_normal(len(giorni)))}, index=giorni)
    g.loc[giorni[4 + 5 * 50], "vix3m"] = np.nan                       # manca l'ultima seduta: vale la precedente della stessa settimana
    g.loc[giorni[5 * 80:5 * 81], "vix3m"] = np.nan                     # settimana intera senza valori: mancante
    for s in ("vvix", "spx", "vx1", "vx2"):
        g[s] = 100.0 + np.arange(len(g)) * 0.01
    date = giorni[4::5]
    z = VS.x1_settimanale(date, g)
    np.testing.assert_allclose(z, V.costruisci(date, g["vix"].to_numpy()[4::5], g)[VS.X1].to_numpy(), equal_nan=True)
    assert np.isnan(z[80]) and z[50] == pytest.approx(g["vix3m"].iloc[3 + 5 * 50] / g["vix"].iloc[3 + 5 * 50] - 1) and np.isfinite(z).sum() == 299


def test_version_comparison_file_matches_single_origin(df):
    path = OUT / "c_previsioni_dev.csv"
    if not path.exists() or not dati3.esterni_disponibili():
        pytest.skip("eseguire prima python -m src.run_phase4 (richiede i dati esterni di sviluppo)")
    variante = json.loads((OUT / "a_info.json").read_text())["scelta"]["scelta"]
    with K.guardia_dati():
        w, g = dati3.load_settimanale(), dati3.load_giornaliero()
    z = VS.x1_settimanale(w["date"], g)
    p = pd.read_csv(path, parse_dates=["data_origine"])
    t = 1200
    r = p[p["data_origine"] == df["date"].iloc[t]].set_index("h")
    ris = {"F1": VS.prevedi_f1(df, t, variante), "F2": VS.prevedi_f2(df, t, z, variante)}
    if "F3_q50" in p.columns:
        ris["F3"] = VS.prevedi_f3(df, t, variante)
    for v, (prev, cand) in ris.items():
        for s, c in zip(prev["settimane"], cand):
            h = s["h"]
            assert s["chiusura"] == pytest.approx(r.loc[h, f"{v}_q50"], abs=3e-4) and s["fasce"]["80"][1] == pytest.approx(r.loc[h, f"{v}_q90"], abs=3e-4), (v, h)
            assert c["p_chiusura_su"] == pytest.approx(r.loc[h, f"{v}_p_su"], abs=2e-5) and c["chiusura"] == pytest.approx(r.loc[h, f"{v}_chiusura"], abs=3e-4), (v, h)
    assert p["data_obiettivo"].max() <= str(df["date"].max().date()) and p["data_origine"].min() >= pd.Timestamp("2014-06-01")
    conf = pd.read_csv(OUT / "c_confronto.csv")
    assert set(conf["gruppo"]) == {"tutte", "fascia_13_22"} and set(conf["h"]) == {1, 2, 3, 4} and {"F1", "F2"} <= set(conf["versione"])


def test_comparison_metrics_known_values():
    g = pd.DataFrame({"idx_origine": np.arange(4), "chiusura_origine": [20.0] * 4, "chiusura_reale": [22.0, 18.0, 20.5, 30.0], "y": np.log([22.0, 18.0, 20.5, 30.0]),
                      "apertura_reale": [20.5] * 4, "massimo_reale": [23.0, 21.0, 21.0, 31.0], "minimo_reale": [20.0, 17.5, 20.0, 20.0], "su": [1.0, 0.0, 1.0, 1.0]})
    for v, cen, p in (("F1", [21.0, 21.0, 21.0, 21.0], [0.7, 0.5, 0.3, 0.9]), ("F2", [22.0, 18.0, 20.5, 24.0], [0.8, 0.2, 0.55, 0.95])):
        g[f"{v}_q50"], g[f"{v}_centro"], g[f"{v}_p_su"] = cen, cen, p
        for k, d in (("q05", -6), ("q10", -4), ("q25", -1.5), ("q75", 1.5), ("q90", 4), ("q95", 6)):
            g[f"{v}_{k}"] = np.array(cen) + d
        g[f"{v}_apertura"], g[f"{v}_chiusura"], g[f"{v}_massimo"], g[f"{v}_minimo"] = 20.6, 19.5, 21.5, 19.0
    r1, r2 = VC4.riga(g, "F1"), VC4.riga(g, "F2")
    assert r1["mae_centrale"] == pytest.approx((1 + 3 + 0.5 + 9) / 4) and r2["mae_centrale"] == pytest.approx(6 / 4) and r2["entro_10"] == 0.75 and r2["entro_20"] == 1.0
    assert r1["copertura_50"] == 0.5 and r1["larghezza_80"] == 8.0 and r1["copertura_90"] == 0.75
    assert r1["quota_neutra"] == 0.25 and r1["n_non_neutre"] == 3 and r1["accuratezza_non_neutre"] == pytest.approx(2 / 3)
    assert r2["quota_neutra"] == 0.25 and r2["accuratezza_non_neutre"] == 1.0 and r2["brier_direzione"] < r1["brier_direzione"]
    assert VC4._versioni(g) == ("F1", "F2")


# ------------------------------------------------------------------ configurazioni congelate

def test_frozen_configurations_match_the_code():
    files = sorted(VS.CARTELLA.glob("f*.json"))
    if not files or not (OUT / "b_info.json").exists():
        pytest.skip("eseguire prima python -m src.run_phase4 e python -m src.fase4.congela")
    attese = congela.costruisci()
    assert [f.stem for f in files] == sorted(attese) and {"f1", "f2"} <= set(attese)
    for f in files:
        cfg = json.loads(f.read_text())
        assert re.fullmatch(r"[0-9a-f]{40}", cfg["commit"]) and cfg["congelata_il"] and isinstance(cfg["sorgenti_identici_al_commit"], bool)
        resto = {k: v for k, v in cfg.items() if k not in congela.VARIABILI_ESECUZIONE}
        assert resto == json.loads(json.dumps(attese[f.stem])), f"{f.name} non coincide con la configurazione ricostruita dal codice"
        assert cfg["sha256_sorgenti"] == congela.impronte(), "un sorgente congelato e' cambiato dopo il congelamento: rieseguire congela PRIMA dell'esame"
        assert cfg["candele"]["variante_ampiezze"] in KA.VARIANTI and cfg["orizzonti_settimane"] == [1, 2, 3, 4] and cfg["min_errori_passati"] == 104
    assert all((C.ROOT / s).exists() for s in VS.SORGENTI_CONGELATI)
    f2 = json.loads((VS.CARTELLA / "f2.json").read_text())
    assert f2["variabile_aggiuntiva"]["definizione"] == "VIX3M / VIX - 1" and f2["ammissione"]["min_settimane_valide"] == M.MIN_VALIDE
    passa = json.loads((OUT / "b_info.json").read_text())["passa"]
    assert ("f3" in attese) == ("N1" in passa)


# ------------------------------------------------------------------ programma

def _file_grezzo(df, path):
    rows = [{"week_start": (r.date - pd.Timedelta(days=r.date.weekday())).date(), "last_trading_day": r.date.date(), "open": r.open, "high": r.high,
             "low": r.low, "close": r.close, "trading_days": 5, "note": ""} for r in df.itertuples()]
    pd.DataFrame(rows).to_csv(path, index=False)


def _file_cboe(df, cartella, fino_a=None, vix3m_da=600):
    """File giornalieri nel formato Cboe: una seduta per settimana (la data settimanale)."""
    cartella.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    d = df if fino_a is None else df[df["date"] <= fino_a]
    for nome, val, da in (("vix", d["close"].to_numpy(), 0), ("vix3m", d["close"].to_numpy() * (1.10 + 0.04 * rng.standard_normal(len(d))), vix3m_da)):
        t = pd.DataFrame({"DATE": d["date"].dt.strftime("%m/%d/%Y"), "OPEN": val, "HIGH": val, "LOW": val, "CLOSE": np.round(val, 2)}).iloc[da:]
        t.to_csv(cartella / ES.FILE[nome], index=False)


def test_forecast_versions_fallback_and_registry(df, tmp_path, capsys):
    if not F.VS.CARTELLA.joinpath("f1.json").exists():
        pytest.skip("eseguire prima python -m src.fase4.congela")
    cfg = F.carica_versioni()
    base, out = tmp_path / "base.csv", tmp_path / "out"
    _file_grezzo(df, base)
    r = F.main(["--file", str(base), "--versione", "f2"], out_dir=out, oggi="2024-11-09")                 # nessun file giornaliero: avviso e ripiego su f1
    testo = capsys.readouterr().out
    assert r["versione"] == "f1" and "f2" not in r["versioni"] and "ATTENZIONE" in testo and "Si ripiega su f1" in testo
    est = tmp_path / "esterni"
    _file_cboe(df, est)
    r2 = F.main(["--file", str(base), "--versione", "f2"], out_dir=out, esterni_dir=est, oggi="2024-11-10")
    assert r2["versione"] == "f2" and {"f1", "f2"} <= set(r2["versioni"]) and r2["md"].name.endswith("_f2.md")
    assert r2["candele"] == r2["versioni"]["f2"][1] and r2["versioni"]["f1"] == r["versioni"]["f1"]
    usati = pd.read_csv(out / f"f2_valori_usati_{r2['previsione']['data_origine']}.csv")
    assert usati["data_settimanale"].max() == r2["previsione"]["data_origine"] and r2["note"]["f2"]["x1_origine"]["x1"] == pytest.approx(usati["x1"].iloc[-1], abs=1e-6)
    conf = r2["confronto"]
    assert set(conf["versione"]) == set(r2["versioni"]) and len(conf) == 4 * len(r2["versioni"])
    md = r2["md"].read_text()
    assert "Confronto tra le versioni" in md and VS.NOMI["f2"] in md and "piano_esame_holdout.md" in md
    reg = pd.read_csv(out / "registro_previsioni.csv")
    assert set(reg["versione"]) == set(r2["versioni"]) and len(reg) == 4 * len(r2["versioni"])               # f1 (e f3) del primo giro non duplicate, f2 aggiunta
    assert set(reg.loc[reg["versione"] == "f2", "data_esecuzione"]) == {"2024-11-10"} and set(reg.loc[reg["versione"] == "f1", "data_esecuzione"]) == {"2024-11-09"}
    assert r2["registrate"] == ["f2"] and (reg["costruzione"] == f"K2m-{cfg['f1']['candele']['variante_ampiezze']}").all()
    vecchio = tmp_path / "vecchio"                                                                         # VIX3M piu' vecchio dell'origine: avviso e ripiego
    _file_cboe(df, vecchio, fino_a=df["date"].iloc[-2])
    capsys.readouterr()
    r3 = F.main(["--file", str(base), "--versione", "f2", "--origine", str(df["date"].iloc[-1].date())], out_dir=tmp_path / "o3", esterni_dir=vecchio)
    testo = capsys.readouterr().out
    assert r3["versione"] == "f1" and "VIX3M manca nella settimana di origine" in testo and str(df["date"].iloc[-2].date()) in testo
    r4 = F.main(["--file", str(base), "--origine", str(df["date"].iloc[-30].date())], out_dir=tmp_path / "o4", esterni_dir=est)   # dati successivi all'origine: ignorati
    est_tronco = tmp_path / "tronco"
    _file_cboe(df, est_tronco, fino_a=df["date"].iloc[-30])
    r5 = F.main(["--file", str(base), "--origine", str(df["date"].iloc[-30].date())], out_dir=tmp_path / "o5", esterni_dir=est_tronco)
    assert r4["versioni"]["f2"] == r5["versioni"]["f2"] and r4["versione"] == "f1"
    if "f3" in cfg:
        r6 = F.main(["--file", str(base), "--versione", "f3", "--origine", str(df["date"].iloc[-1].date())], out_dir=tmp_path / "o6")
        assert r6["versione"] == "f3" and r6["candele"] == VS.prevedi_f3(df, len(df) - 1, cfg["f1"]["candele"]["variante_ampiezze"])[1]


def test_update_downloads_only_vix_and_vix3m_and_failure_is_not_fatal(df, tmp_path, monkeypatch, capsys):
    chiesti = []

    def finto(url):
        chiesti.append(url)
        return b"DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2020,12,12,12,12\n"

    info = ES.aggiorna(tmp_path / "e", scarica=finto)
    assert chiesti == [ES.INDIRIZZI["vix"], ES.INDIRIZZI["vix3m"]] and set(info) == {"vix", "vix3m"} and all(u.startswith("https://cdn.cboe.com/") for u in chiesti)
    assert sorted(p.name for p in (tmp_path / "e").iterdir()) == ["VIX3M_History.csv", "VIX_History.csv"]
    if not F.VS.CARTELLA.joinpath("f1.json").exists():
        pytest.skip("eseguire prima python -m src.fase4.congela")
    base = tmp_path / "base.csv"
    _file_grezzo(df, base)
    monkeypatch.setattr(ES, "aggiorna", lambda cartella: (_ for _ in ()).throw(RuntimeError("rete assente")))
    r = F.main(["--file", str(base), "--aggiorna", "--versione", "f2", "--origine", str(df["date"].iloc[-1].date())], out_dir=tmp_path / "o", esterni_dir=tmp_path / "vuota")
    testo = capsys.readouterr().out
    assert r["versione"] == "f1" and "scarico dei dati giornalieri non riuscito" in testo and "Si ripiega su f1" in testo


def test_registry_key_is_origin_plus_version_and_old_rows_are_kept(tmp_path):
    prev = {"data_origine": "2024-01-05", "chiusura_origine": 13.0, "versione": "f1",
            "settimane": [{"h": h, "data": f"2024-01-{5 + 7 * h:02d}", "quantili": {"q50": 13.5}, "fasce": {"50": [12, 15], "80": [11, 17], "90": [10, 19]},
                           "probabilita": {"VIX>20": 0.1, "VIX>25": 0.05, "VIX>30": 0.01}} for h in (1, 2)]}
    cand = [{"apertura": 13.4, "massimo": 14.5, "minimo": 12.5, "chiusura": 12.9, "direzione": "ribasso", "prob_direzione": 0.64, "costruzione": "K2m-A1",
             "modello_direzione": "L0", "p_chiusura_su": 0.7, "direzione_chiusura": "rialzo"}] * 2
    p = tmp_path / "r.csv"
    vecchie = RG.righe(prev, cand, "2024-01-05").drop(columns="versione").assign(costruzione="K2")           # registro scritto prima della Fase 4
    vecchie.to_csv(p, index=False, float_format="%.4f")
    a = RG.aggiungi(p, RG.righe(prev, cand, "2024-01-06"))
    b = RG.aggiungi(p, RG.righe(prev, cand, "2024-01-07"))
    c = RG.aggiungi(p, RG.righe(prev | {"versione": "f2"}, cand, "2024-01-07"))
    reg = pd.read_csv(p, dtype={"data_origine": str, "data_esecuzione": str})
    assert a[0] and not b[0] and c[0] and len(reg) == 6 and list(reg["versione"].fillna("")) == ["", "", "f1", "f1", "f2", "f2"]
    assert list(reg["costruzione"].iloc[:2]) == ["K2", "K2"] and "già registrata" in b[1]
    storia = pd.DataFrame({"date": pd.to_datetime(["2024-01-05", "2024-01-12"]), "open": [13.0, 13.4], "high": [14.0, 16.0], "low": [12.0, 12.9], "close": [13.0, 15.5]})
    det, pag = RG.verifica(reg, storia)
    assert list(det["versione"]) == [RG.SENZA_VERSIONE, "f1", "f2"] and list(pag["n"]) == [1, 1, 1] and list(pag["h"]) == [1, 1, 1]
    assert det["errore_chiusura"].tolist() == pytest.approx([2.6] * 3) and bool(det.loc[det["versione"] == "f1", "etichetta_corretta"].iloc[0])


# ------------------------------------------------------------------ guardia e piano d'esame

def test_phase4_measures_read_only_dev_and_never_touch_the_holdout():
    banned = r"HOLDOUT_FILE|CLEAN_FILE|RAW_FILE|vix_weekly_holdout|vix_weekly_clean|VIX_settimanale|[\"']holdout[\"']|[\"']raw[\"']"
    misura = ("valuta_a.py", "valuta_b.py", "valuta_c.py", "report.py", "piano.py", "candele.py", "regimi.py", "boosting.py", "versioni.py")
    for f in sorted(DIR.glob("*.py")) + [pathlib.Path(C.ROOT) / "src" / "run_phase4.py"]:
        src = f.read_text()
        if f.name == "piano.py":                                                # il piano nomina il file di holdout nel testo, ma non legge nessun dato
            assert not re.search(r"read_csv|open\(|load_|dati\.", src), f.name
        else:
            assert not re.search(banned, src), f.name
        if f.name != "esterni.py":
            assert not re.search(r"[\"']external[\"']|read_csv\(C\.", src), f"{f.name}: i dati si leggono con fase3.dati o fase2b.dati"
        if f.name in misura or f.name == "run_phase4.py":                       # i moduli di misura non usano i dati giornalieri del programma
            assert not re.search(r"import esterni|esterni as|fetch_external import|\.aggiorna\(", src), f.name
    assert '"external" / "forecast"' in (DIR / "esterni.py").read_text()
    assert "with K.guardia_dati():" in (pathlib.Path(C.ROOT) / "src" / "run_phase4.py").read_text()
    src = (pathlib.Path(C.ROOT) / "src" / "forecast.py").read_text()               # il programma non valuta e non legge l'holdout
    assert not re.search(r"import.*valuta|evaluate|density_row|dm_loss|pinball|brier|HOLDOUT_FILE|vix_weekly_holdout", src)
    with K.guardia_dati():
        for path in (C.HOLDOUT_FILE, C.ROOT / "data" / "external" / "holdout" / "esterni_giornalieri_holdout.csv"):
            with pytest.raises(PermissionError):
                open(path)


def test_exam_plan_is_written_and_nothing_executes_it():
    piano = C.REPORTS_DIR / "piano_esame_holdout.md"
    if not piano.exists():
        pytest.skip("eseguire prima python -m src.run_phase4")
    testo = piano.read_text()
    for parola in ("non è stato eseguito", "versioni/f1.json", "versioni/f2.json", "Holm", "non distinguibili", "resta F1", "fascia 13-22", "Una sola esecuzione"):
        assert parola in testo, parola
    assert not list((pathlib.Path(C.ROOT) / "src").glob("*esame*")), "il comando d'esame non deve esistere in questa fase"
    assert not list(C.REPORTS_DIR.glob("*esame*risultat*")) and not (C.REPORTS_DIR / "esame").exists()
