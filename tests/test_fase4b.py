"""Fase 4b: prima candela (mosse medie, K4, minimo M1q), candidati e regole di scelta, sfidante FS, configurazioni
congelate (f1, fs), programma con --versione f1|fs, piano d'esame aggiornato.

Anti look-ahead di ogni componente nuovo; regole delle Parti B, C, D; ripiego di fs; guardia sull'holdout.
I test che richiedono file prodotti da `run_phase4b` vengono saltati se mancano."""
import json
import pathlib
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import baselines as B
from src import config as C
from src import forecast as F
from src.fase1 import comune as K
from src.fase2b import dati as dati2b, grafici
from src.fase2c import componenti as CP
from src.fase3 import candele as K3
from src.fase4 import candele as KA, esterni as ES, versioni as V4
from src.fase4b import congela, prima as PR, valuta as VA, versioni as VS

DIR = pathlib.Path(C.ROOT) / "src" / "fase4b"
OUT = C.REPORTS_DIR / "fase4b"
K2M = {"variante": "A1", "prima_candela": "K2", "minimo": "M1q", "candidato": "P5"}
K4M = {"variante": "A1", "prima_candela": "K4", "minimo": "M0", "candidato": "P3"}


@pytest.fixture(scope="module")
def df():
    return dati2b.load_dev_candele()


def _stravolgi(df, t):
    d = df.copy()
    for c in ("open", "high", "low", "close"):
        d.loc[d.index > t, c] = d.loc[d.index > t, c] * 3.0 + 7.0
    d["log_close"] = np.log(d["close"])
    return d


def _z(df, inizio=500):
    x = np.log(df["close"].to_numpy())
    z = 0.12 - 0.3 * (x - pd.Series(x).rolling(12, min_periods=1).mean().to_numpy()) + 0.02 * np.random.default_rng(4).standard_normal(len(x))
    z[:inizio] = np.nan
    return z


# ------------------------------------------------------------------ componenti della prima candela

def test_average_moves_use_last_260_changes_and_ignore_future(df):
    x = np.log(df["close"].to_numpy())
    t = 900
    d, u = PR.mosse(x[: t + 1])
    dx = np.diff(x[: t + 1])[-260:]
    assert PR.FINESTRA_MOSSE == 260 and len(dx) == 260
    assert d == pytest.approx(-dx[dx < 0].mean()) and u == pytest.approx(dx[dx > 0].mean()) and d > 0 and u > 0
    x2 = x.copy()
    x2[t + 1:] += 1.0
    assert PR.mosse(x2[: t + 1]) == (d, u)
    x3 = x.copy()
    x3[: t - 261] += 5.0                                                  # cio' che precede la finestra non conta
    assert PR.mosse(x3[: t + 1]) == pytest.approx((d, u))
    x4 = x.copy()
    x4[t - 5] += 0.3
    assert PR.mosse(x4[: t + 1]) != (d, u)                                # il test "morde"
    assert PR.mosse(np.array([3.0, 3.0, 3.0])) == (0.0, 0.0)


def test_k4_follows_the_label():
    c0, rap, d, u, cen = 20.0, 1.03, 0.08, 0.10, 19.6
    for direzione, chiusura in (("ribasso", c0 * np.exp(-d)), ("rialzo", c0 * np.exp(u)), ("neutra", cen)):
        k = PR.candela_k4(c0, rap, direzione, d, u, cen, 0.06, 0.04)
        assert k["apertura"] == pytest.approx(c0 * rap) and k["chiusura"] == pytest.approx(chiusura)
        assert k["massimo"] == pytest.approx(max(k["apertura"], k["chiusura"]) * np.exp(0.06)) and k["minimo"] == pytest.approx(min(k["apertura"], k["chiusura"]) * np.exp(-0.04))
        assert k["minimo"] <= min(k["apertura"], k["chiusura"]) <= max(k["apertura"], k["chiusura"]) <= k["massimo"]
    assert PR.COSTRUZIONI == ("K2", "K4") and PR.MINIMI == ("M0", "M1q")


def test_minimum_model_is_a_median_regression_on_last_520_weeks_and_ignores_future(df):
    import statsmodels.api as sm
    x, low = np.log(df["close"].to_numpy()), df["low"].to_numpy(dtype=float)
    t = 1000
    m = PR.minimo_m1q(x[: t + 1], low[: t + 1])
    Fh = B.har_features(x[: t + 1])
    i = np.arange(11, t)[-520:]
    assert PR.FINESTRA_MINIMO == 520 and len(i) == 520 and (i + 1).max() == t
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.QuantReg(np.log(low[i + 1]) - x[i], Fh[i - 11]).fit(q=0.5).params
    assert m == pytest.approx(float(np.exp(x[t] + Fh[-1] @ ref)), rel=2e-3) and 0.7 * np.exp(x[t]) < m < np.exp(x[t]) * 1.02
    x2, l2 = x.copy(), low.copy()
    x2[t + 1:], l2[t + 1:] = x2[t + 1:] + 1, l2[t + 1:] * 3
    assert PR.minimo_m1q(x2[: t + 1], l2[: t + 1]) == m
    l3 = low.copy()
    l3[t - 40:t + 1] *= 0.8
    assert PR.minimo_m1q(x[: t + 1], l3[: t + 1]) != m                    # il test "morde"
    k = {"apertura": 20.6, "chiusura": 19.0, "massimo": 21.0, "minimo": 18.0}
    assert PR.con_minimo(k, 18.7) == k | {"minimo": 18.7, "minimo_limitato_dal_corpo": False}
    assert PR.con_minimo(k, 19.5) == k | {"minimo": 19.0, "minimo_limitato_dal_corpo": True}     # mai sopra il corpo


# ------------------------------------------------------------------ versioni

def test_f1_and_fs_ignore_future_and_follow_their_definitions(df):
    z, t = _z(df), 1100
    f1 = VS.prevedi_f1(df, t, K2M)
    fs = VS.prevedi_fs(df, t, z, K2M)
    d2, z2 = _stravolgi(df, t), z.copy()
    z2[t + 1:] = z2[t + 1:] * 5 + 1
    assert f1 == VS.prevedi_f1(d2, t, K2M) == VS.prevedi_f1(df.iloc[: t + 1].copy(), t, K2M)
    assert fs == VS.prevedi_fs(d2, t, z2, K2M) == VS.prevedi_fs(df.iloc[: t + 1].copy(), t, z[: t + 1], K2M)
    assert VS.prevedi_f1(_stravolgi(df, t - 30), t, K2M) != f1                                            # il test "morde"
    b1, b2, b3 = V4.prevedi_f1(df, t, "A1"), V4.prevedi_f2(df, t, z, "A1"), V4.prevedi_f3(df, t, "A1")
    assert VS.base_di("P5") == "f2" and VS.base_di("P3") == "f3" and VS.base_di("P4") == "f2" and VS.CANDIDATI["P5"] == ("f2", "f3")
    # FS: probabilita' +1 = media dei componenti; tutto il resto dalla costruzione di base (F2)
    c1 = fs[1][0]
    assert c1["p_chiusura_su"] == pytest.approx((b2[1][0]["p_chiusura_su"] + b3[1][0]["p_chiusura_su"]) / 2) and (c1["direzione_chiusura"], c1["etichetta"]) == K3.etichetta(c1["p_chiusura_su"])
    assert fs[0]["settimane"] == b2[0]["settimane"] and fs[0]["versione"] == "fs" and fs[0]["candidato"] == "P5" and fs[0]["costruzione_di_base"] == "f2"
    for a, b in zip(fs[1][1:], b2[1][1:]):                                                               # settimane +2…+4: candele ed etichette di F2
        assert all(a[k] == b[k] for k in KA.CAMPI + ("p_chiusura_su", "etichetta")) and a["versione"] == "fs" and "colore" not in a
    assert f1[0]["settimane"] == b1[0]["settimane"] and f1[1][0]["p_chiusura_su"] == b1[1][0]["p_chiusura_su"]
    # prima candela K2 + M1q: corpo K2 invariato, minimo dal modello dedicato (mai sopra il corpo), colore = etichetta
    x, low = np.log(df["close"].to_numpy()[: t + 1]), df["low"].to_numpy(dtype=float)[: t + 1]
    for (prev, cand), base in ((f1, b1), (fs, b2)):
        c = cand[0]
        assert all(c[k] == pytest.approx(base[1][0][k]) for k in ("apertura", "chiusura", "massimo"))
        assert c["minimo"] == pytest.approx(min(PR.minimo_m1q(x, low), c["apertura"], c["chiusura"])) and c["colore"] == c["direzione_chiusura"]
        assert c["costruzione"] == "K2+M1q/K2m-A1" == cand[3]["costruzione"] and c["minimo"] <= min(c["apertura"], c["chiusura"]) <= c["massimo"]
    assert all(f1[1][0][k] == fs[1][0][k] for k in KA.CAMPI)                                              # con K2 la prima candela non dipende dalla versione


def test_first_candle_k4_and_unavailable_challenger(df):
    z, t = _z(df), 1100
    prev, cand = VS.prevedi_fs(df, t, None, K4M)                                                          # candidato P3: non serve X1
    b3 = V4.prevedi_f3(df, t, "A1")
    c, x = cand[0], np.log(df["close"].to_numpy()[: t + 1])
    d, u = PR.mosse(x)
    comp = KA.componenti(CP.serie(df.iloc[: t + 1]), t, 1, "A1")
    atteso = PR.candela_k4(float(np.exp(x[-1])), comp["rapporto_apertura"], c["direzione_chiusura"], d, u, prev["settimane"][0]["chiusura"], comp["ombra_sup"], comp["ombra_inf"])
    assert all(c[k] == pytest.approx(atteso[k]) for k in KA.CAMPI) and c["costruzione"] == "K4/K2m-A1" and c["p_chiusura_su"] == pytest.approx(b3[1][0]["p_chiusura_su"])
    assert prev["costruzione_di_base"] == "f3" and prev["settimane"] == b3[0]["settimane"] and c["modello_direzione"] == "etichetta"
    for direzione, p in (("rialzo", 0.9), ("ribasso", 0.1), ("neutra", 0.5)):                              # la chiusura K4 segue l'etichetta
        k = VS.prima_candela(df, t, b3[1][0] | {"p_chiusura_su": p, "direzione_chiusura": direzione}, 21.0, K4M)
        assert k["chiusura"] == pytest.approx(PR.chiusura_k4(float(np.exp(x[-1])), direzione, d, u, 21.0)) and k["colore"] == direzione
    with pytest.raises(VS.NonDisponibile):
        VS.prevedi_fs(df, t, None, K2M)                                                                   # candidato con F2 ma senza dati giornalieri
    z_buco = z.copy()
    z_buco[t] = np.nan
    with pytest.raises(VS.NonDisponibile):
        VS.prevedi_fs(df, t, z_buco, K2M)


# ------------------------------------------------------------------ misure e regole

def test_label_metrics_and_candidates():
    p = np.array([0.7, 0.65, 0.5, 0.3, 0.2, 0.45])
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    m = VA.etichetta_metriche(p, y)
    assert m["n"] == 6 and m["quota_non_neutre"] == pytest.approx(4 / 6) and m["n_rialzo"] == 2 and m["accuratezza_rialzo"] == 0.5 and m["n_ribasso"] == 2 and m["accuratezza_ribasso"] == 0.5
    assert m["accuratezza_non_neutre"] == 0.5 and m["brier"] == pytest.approx(((p - y) ** 2).mean())
    c1 = pd.DataFrame({"F1_p_su": [0.2, 0.4, np.nan, 0.6], "F2_p_su": [0.3, 0.5, 0.5, np.nan], "F3_p_su": [0.4, 0.3, 0.2, 0.1], "chiusura_origine": [12.0, 18.0, 22.0, 33.0]})
    c = VA.candidati(c1)
    assert len(c) == 2 and list(c["P4"]) == pytest.approx([0.3, 0.4]) and list(c["P5"]) == pytest.approx([0.35, 0.4]) and list(c["P1"]) == [0.2, 0.4]
    assert list(c["fascia_origine"]) == ["<15", "15-20"] and tuple(VS.CANDIDATI) == ("P1", "P2", "P3", "P4", "P5") and VS.SFIDANTI == ("P2", "P3", "P4", "P5")


def _ta(brier, p_holm=0.01, dm=2.0):
    return pd.DataFrame([{"gruppo": "tutte", "candidato": k, "brier": v, "p_holm": p_holm, "dm_stat": dm} for k, v in brier.items()])


def test_challenger_rule():
    s = VA.scegli_candidato(_ta({"P1": 0.20, "P2": 0.24, "P3": 0.23, "P4": 0.235, "P5": 0.232}))
    assert s["candidato"] == "P3" and s["base"] == "f3" and s["componenti"] == ["f3"]                     # P1 non e' una sfidante, anche se ha il Brier piu' basso
    s = VA.scegli_candidato(_ta({"P1": 0.25, "P2": 0.23, "P3": 0.24, "P4": 0.23, "P5": 0.231}))
    assert s["candidato"] == "P2" and s["base"] == "f2"                                                   # parita': il primo dell'elenco
    s = VA.scegli_candidato(_ta({"P1": 0.25, "P2": 0.24, "P3": 0.24, "P4": 0.229, "P5": 0.231}, p_holm=0.2))
    assert s["candidato"] == "P4" and s["base"] == "f2" and not s["significativo_vs_p1_dopo_holm"]


def _fc(e2, e4, n=400, seed=1):
    rng = np.random.default_rng(seed)
    y = 20 + rng.standard_normal(n)
    return pd.DataFrame({"chiusura_reale": y, "K2_chiusura": y + e2 * rng.standard_normal(n), "K4_chiusura": y + e4 * rng.standard_normal(n)})


def test_first_candle_rule_and_minimum_rule():
    assert VA.scegli_costruzione(_fc(1.0, 1.0), {"sovrapposizione": 0.50}, {"sovrapposizione": 0.52})["scelta"] == "K4"      # sovrapposizione non piu' bassa, chiusura non peggiore
    assert VA.scegli_costruzione(_fc(1.0, 0.5), {"sovrapposizione": 0.52}, {"sovrapposizione": 0.50})["scelta"] == "K2"      # sovrapposizione piu' bassa: resta K2 anche se la chiusura migliora
    r = VA.scegli_costruzione(_fc(0.5, 2.0), {"sovrapposizione": 0.50}, {"sovrapposizione": 0.55})
    assert r["scelta"] == "K2" and r["chiusura_significativamente_peggiore"]                                                  # chiusura significativamente peggiore
    assert VA.scegli_costruzione(_fc(1.0, 1.0), {"sovrapposizione": 0.50}, {"sovrapposizione": 0.50})["scelta"] == "K4"      # «non piu' bassa» comprende l'uguaglianza
    rng = np.random.default_rng(2)
    reale = 15 + rng.standard_normal(500)
    rumore = rng.standard_normal(500)
    assert VA.scegli_minimo({"M0": reale + 1.0 * rumore - 0.8, "M1q": reale + 0.3 * rumore}, reale)["scelta"] == "M1q"
    assert VA.scegli_minimo({"M0": reale + 0.3 * rumore, "M1q": reale + 1.0 * rumore}, reale)["scelta"] == "M0"
    quasi = VA.scegli_minimo({"M0": reale + rumore, "M1q": reale + rumore * (1 - 1e-4)}, reale)
    assert quasi["mae"]["M1q"] < quasi["mae"]["M0"] and (quasi["scelta"] == "M1q") == (quasi["p"] < 0.05)                   # piu' basso ma serve anche il test
    f = pd.DataFrame({"K2_apertura": [20.6, 20.6], "K2_chiusura": [19.0, 19.0], "K2_minimo": [18.2, 18.2], "m1q": [18.6, 19.4], "minimo_reale": [18.5, 18.5]})
    tab, serie = VA.minimi(f, "K2")
    assert list(serie["M1q"]) == [18.6, 19.0] and list(serie["M1q senza vincolo"]) == [18.6, 19.4] and list(serie["M0"]) == [18.2, 18.2]
    r = tab.set_index("variante")
    assert r.loc["M0", "quota_troppo_bassi"] == 1.0 and r.loc["M1q", "quota_troppo_bassi"] == 0.0 and r.loc["M1q", "quota_limitati_dal_corpo"] == 0.5
    assert r.loc["M1q", "errore_assoluto_medio"] == pytest.approx(0.3) and r.loc["M1q", "entro_5"] == 1.0


def test_development_file_matches_program(df):
    path = OUT / "previsioni_dev.csv"
    if not path.exists():
        pytest.skip("eseguire prima python -m src.run_phase4b")
    info = json.loads((OUT / "info.json").read_text())
    sc = info["scelte"]
    p = pd.read_csv(path, parse_dates=["data_origine"])
    p = p[p["versione"] == "F1"]
    for t in (700, 1250):
        r = p[p["data_origine"] == df["date"].iloc[t]].iloc[0]
        prev, cand = VS.prevedi_f1(df, t, sc)
        assert all(cand[0][k] == pytest.approx(r[f"DOPO_{k}"], abs=3e-4) for k in KA.CAMPI) and cand[0]["direzione_chiusura"] == r["etichetta"]
        k4 = VS.prevedi_f1(df, t, sc | {"prima_candela": "K4", "minimo": "M0"})[1][0]
        assert all(k4[k] == pytest.approx(r[f"K4_{k}"], abs=3e-4) for k in KA.CAMPI)
        x = np.log(df["close"].to_numpy()[: t + 1])
        assert PR.mosse(x) == pytest.approx((r["mossa_ribasso"], r["mossa_rialzo"]), abs=2e-6) and PR.minimo_m1q(x, df["low"].to_numpy(dtype=float)[: t + 1]) == pytest.approx(r["m1q"], abs=3e-4)
    assert sc["prima_candela"] in PR.COSTRUZIONI and sc["minimo"] in PR.MINIMI and sc["candidato"] in VS.SFIDANTI and p["data_obiettivo"].max() <= str(df["date"].max().date())
    st = pd.read_csv(OUT / "statistiche_programma.csv")
    assert set(st["versione"]) == {"f1", "fs"} and set(st["h"]) == {1, 2, 3, 4} and st[["copertura_80", "mae_minimo", "accuratezza_non_neutre"]].notna().all().all()


# ------------------------------------------------------------------ configurazioni congelate

def test_frozen_configurations_match_the_code():
    files = sorted(VS.CARTELLA.glob("f*.json"))
    if not files or not (OUT / "info.json").exists():
        pytest.skip("eseguire prima python -m src.run_phase4b e python -m src.fase4b.congela")
    attese = congela.costruisci()
    assert [f.stem for f in files] == ["f1", "fs"] == sorted(attese), "all'esame vanno solo F1 e FS"
    for f in files:
        cfg = json.loads(f.read_text())
        assert re.fullmatch(r"[0-9a-f]{40}", cfg["commit"]) and cfg["congelata_il"] and isinstance(cfg["sorgenti_identici_al_commit"], bool)
        resto = {k: v for k, v in cfg.items() if k not in congela.VARIABILI_ESECUZIONE}
        assert resto == json.loads(json.dumps(attese[f.stem])), f"{f.name} non coincide con la configurazione ricostruita dal codice"
        assert cfg["sha256_sorgenti"] == congela.impronte(), "un sorgente congelato e' cambiato dopo il congelamento: rieseguire congela PRIMA dell'esame"
        assert {"src/fase4b/prima.py", "src/fase4b/versioni.py", "src/fase4/versioni.py", "src/fase4/regimi.py"} <= set(cfg["sha256_sorgenti"])
        pc = cfg["candele"]["prima_candela"]
        assert pc["costruzione"] == cfg["scelte"]["prima_candela"] and pc["minimo"] == cfg["scelte"]["minimo"] and cfg["orizzonti_settimane"] == [1, 2, 3, 4]
    fs = json.loads((VS.CARTELLA / "fs.json").read_text())
    info = json.loads((OUT / "info.json").read_text())
    assert fs["candidato"]["sigla"] == info["scelte"]["candidato"] and fs["costruzione_di_base"] == VS.base_di(info["scelte"]["candidato"])
    assert fs["candidato"]["componenti"] == list(VS.CANDIDATI[info["scelte"]["candidato"]]) and set(fs["componenti"]) == set(fs["candidato"]["componenti"])


# ------------------------------------------------------------------ programma

def _file_grezzo(df, path):
    rows = [{"week_start": (r.date - pd.Timedelta(days=r.date.weekday())).date(), "last_trading_day": r.date.date(), "open": r.open, "high": r.high,
             "low": r.low, "close": r.close, "trading_days": 5, "note": ""} for r in df.itertuples()]
    pd.DataFrame(rows).to_csv(path, index=False)


def _file_cboe(df, cartella, fino_a=None, vix3m_da=600):
    cartella.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    d = df if fino_a is None else df[df["date"] <= fino_a]
    for nome, val, da in (("vix", d["close"].to_numpy(), 0), ("vix3m", d["close"].to_numpy() * (1.10 + 0.04 * rng.standard_normal(len(d))), vix3m_da)):
        t = pd.DataFrame({"DATE": d["date"].dt.strftime("%m/%d/%Y"), "OPEN": val, "HIGH": val, "LOW": val, "CLOSE": np.round(val, 2)}).iloc[da:]
        t.to_csv(cartella / ES.FILE[nome], index=False)


def test_forecast_two_versions_fallback_and_registry(df, tmp_path, capsys):
    if not (VS.CARTELLA / "fs.json").exists():
        pytest.skip("eseguire prima python -m src.fase4b.congela")
    cfg = F.carica_versioni()
    assert sorted(cfg) == ["f1", "fs"] and cfg["f1"]["scelte"] == cfg["fs"]["scelte"]
    usa_x1 = "f2" in VS.CANDIDATI[cfg["fs"]["scelte"]["candidato"]]
    base, out = tmp_path / "base.csv", tmp_path / "out"
    _file_grezzo(df, base)
    r = F.main(["--file", str(base), "--versione", "fs"], out_dir=out, oggi="2024-11-09")                 # nessun file giornaliero
    testo = capsys.readouterr().out
    if not usa_x1:
        pytest.skip("la sfidante congelata non usa VIX3M: il ripiego non si applica")
    assert r["versione"] == "f1" and list(r["versioni"]) == ["f1"] and "ATTENZIONE" in testo and "Si ripiega su f1" in testo
    assert r["candele"] == VS.prevedi_f1(df, len(df) - 1, cfg["f1"]["scelte"])[1] and r["candele"][0]["colore"] in ("rialzo", "ribasso", "neutra")
    est = tmp_path / "esterni"
    _file_cboe(df, est)
    r2 = F.main(["--file", str(base), "--versione", "fs"], out_dir=out, esterni_dir=est, oggi="2024-11-10")
    assert r2["versione"] == "fs" and list(r2["versioni"]) == ["f1", "fs"] and r2["md"].name.endswith("_fs.md") and r2["versioni"]["f1"] == r["versioni"]["f1"]
    usati = pd.read_csv(out / f"fs_valori_usati_{r2['previsione']['data_origine']}.csv")
    assert usati["data_settimanale"].max() == r2["previsione"]["data_origine"] and r2["note"]["fs"]["x1_origine"]["x1"] == pytest.approx(usati["x1"].iloc[-1], abs=1e-6)
    conf, md = r2["confronto"], r2["md"].read_text()
    assert set(conf["versione"]) == {"f1", "fs"} and len(conf) == 8 and "Confronto tra le versioni" in md and VS.NOMI["fs"] in md and "piano_esame_holdout.md" in md
    assert "colorata secondo l'etichetta" in md and "una sola volta" in md
    reg = pd.read_csv(out / "registro_previsioni.csv")
    assert set(reg["versione"]) == {"f1", "fs"} and len(reg) == 8 and r2["registrate"] == ["fs"] and (reg["costruzione"] == VS.nome_costruzione(cfg["f1"]["scelte"])).all()
    assert set(reg.loc[reg["versione"] == "f1", "data_esecuzione"]) == {"2024-11-09"} and set(reg.loc[reg["versione"] == "fs", "data_esecuzione"]) == {"2024-11-10"}
    vecchio = tmp_path / "vecchio"                                                                         # VIX3M piu' vecchio dell'origine: avviso e ripiego
    _file_cboe(df, vecchio, fino_a=df["date"].iloc[-2])
    capsys.readouterr()
    r3 = F.main(["--file", str(base), "--versione", "fs", "--origine", str(df["date"].iloc[-1].date())], out_dir=tmp_path / "o3", esterni_dir=vecchio)
    testo = capsys.readouterr().out
    assert r3["versione"] == "f1" and "VIX3M manca nella settimana di origine" in testo and str(df["date"].iloc[-2].date()) in testo
    r4 = F.main(["--file", str(base), "--origine", str(df["date"].iloc[-30].date())], out_dir=tmp_path / "o4", esterni_dir=est)   # dati successivi all'origine: ignorati
    tronco = tmp_path / "tronco"
    _file_cboe(df, tronco, fino_a=df["date"].iloc[-30])
    r5 = F.main(["--file", str(base), "--origine", str(df["date"].iloc[-30].date())], out_dir=tmp_path / "o5", esterni_dir=tronco)
    assert r4["versioni"]["fs"] == r5["versioni"]["fs"] and r4["versione"] == "f1"
    with pytest.raises(SystemExit):
        F.main(["--file", str(base), "--versione", "f2"], out_dir=tmp_path / "o6")                         # f2 e f3 non sono piu' versioni del programma


def test_first_predicted_candle_is_coloured_by_label():
    fig, ax = plt.subplots()
    for x, colore in enumerate(("rialzo", "ribasso", "neutra")):
        grafici.candela(ax, x, 20.6, 22.0, 18.5, 19.2, 0.5, prevista=True, colore=colore)
    colori = [matplotlib.colors.to_hex(p.get_facecolor()[:3]) for p in ax.patches]
    assert colori == [grafici.COLORE_ETICHETTA[k].lower() for k in ("rialzo", "ribasso", "neutra")] and len(set(colori)) == 3
    grafici.candela(ax, 5, 20.6, 22.0, 18.5, 19.2, 0.5, prevista=True)                                     # senza colore: stile delle fasi precedenti (corpo vuoto)
    assert matplotlib.colors.to_hex(ax.patches[3].get_facecolor()[:3]) == "#ffffff"
    plt.close(fig)


# ------------------------------------------------------------------ guardia e piano d'esame

def test_phase4b_measures_read_only_dev_and_never_touch_the_holdout():
    banned = r"HOLDOUT_FILE|CLEAN_FILE|RAW_FILE|vix_weekly_holdout|vix_weekly_clean|VIX_settimanale|[\"']holdout[\"']|[\"']raw[\"']"
    for f in sorted(DIR.glob("*.py")) + [pathlib.Path(C.ROOT) / "src" / "run_phase4b.py"]:
        src = f.read_text()
        if f.name == "piano.py":                                                # il piano nomina il file di holdout nel testo, ma non legge nessun dato
            assert not re.search(r"read_csv|open\(|load_|dati\.", src), f.name
        else:
            assert not re.search(banned, src), f.name
        assert not re.search(r"[\"']external[\"']|read_csv\(C\.(DEV|PROCESSED|RAW|ROOT)|import esterni|esterni as|fetch_external|\.aggiorna\(", src), f"{f.name}: nessun dato esterno nelle misure della Fase 4b"
    assert "with K.guardia_dati():" in (pathlib.Path(C.ROOT) / "src" / "run_phase4b.py").read_text()
    src = (pathlib.Path(C.ROOT) / "src" / "forecast.py").read_text()
    assert not re.search(r"import.*valuta|evaluate|density_row|dm_loss|pinball|brier|HOLDOUT_FILE|vix_weekly_holdout", src)
    with K.guardia_dati():
        with pytest.raises(PermissionError):
            open(C.HOLDOUT_FILE)


def test_exam_plan_has_two_versions_and_first_candle_criterion():
    piano = C.REPORTS_DIR / "piano_esame_holdout.md"
    if not piano.exists() or not (OUT / "info.json").exists():
        pytest.skip("eseguire prima python -m src.run_phase4b")
    testo = piano.read_text()
    for parola in ("non è stato eseguito", "versioni/f1.json", "versioni/fs.json", "Criterio D", "D1. direzione quando non neutra", "D2. minimo", "D3. chiusura", "Holm", "non distinguibili",
                   "resta F1", "fascia 13-22", "Una sola esecuzione", "Due versioni"):
        assert parola in testo, parola
    assert "versioni/f2.json" not in testo and "versioni/f3.json" not in testo
    assert not list((pathlib.Path(C.ROOT) / "src").glob("*esame*")), "il comando d'esame non deve esistere in questa fase"
    assert not list(C.REPORTS_DIR.glob("*esame*risultat*")) and not (C.REPORTS_DIR / "esame").exists()
