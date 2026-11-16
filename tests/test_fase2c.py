"""Fase 2c: candele previste. Anti look-ahead delle componenti e delle costruzioni, coerenza delle candele,
regola di scelta, registro senza duplicati, verifica dal vivo, settimana incompleta senza nota."""
import pathlib
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import forecast as F
from src.fase1 import comune as K
from src.fase2b import dati
from src.fase2c import candele as KA, componenti as CP, registro as RG, settimana as SE, valuta as VC

DIR = pathlib.Path(C.ROOT) / "src" / "fase2c"
DIRS = {h: "L1" for h in CP.ORIZZONTI}


@pytest.fixture(scope="module")
def df():
    return dati.load_dev_candele()


def _stravolgi(df, t):
    d = df.copy()
    for c in ("open", "high", "low", "close"):
        d.loc[d.index > t, c] = d.loc[d.index > t, c] * 3.0 + 7.0
    d["log_close"] = np.log(d["close"])
    return d


# ------------------------------------------------------------------ anti look-ahead

@pytest.mark.parametrize("h", CP.ORIZZONTI)
def test_components_ignore_data_after_origin(df, h):
    t = 800
    a = CP.componenti(CP.serie(df), t, h)
    assert a == CP.componenti(CP.serie(_stravolgi(df, t)), t, h) == CP.componenti(CP.serie(df.iloc[: t + 1]), t, h)
    assert a != CP.componenti(CP.serie(_stravolgi(df, t - 30)), t, h)                 # il test "morde"
    assert all(0.0 <= a[f"p_{m}"] <= 1.0 for m in CP.MODELLI_DIREZIONE)
    assert min(a["corpo"], a["ombra_sup"], a["ombra_inf"]) >= 0 and 0.9 < a["rapporto_apertura"] < 1.2


@pytest.mark.parametrize("costruzione", KA.COSTRUZIONI)
def test_candles_ignore_data_after_origin_and_are_coherent(df, costruzione):
    for t in (700, 1100):
        prev, cand = KA.prevedi_candele(df, t, costruzione, DIRS)
        assert (prev, cand) == KA.prevedi_candele(_stravolgi(df, t), t, costruzione, DIRS) == KA.prevedi_candele(df.iloc[: t + 1].copy(), t, costruzione, DIRS)
        assert len(cand) == 4 and [c["h"] for c in cand] == [1, 2, 3, 4]
        for c in cand:
            assert c["minimo"] <= min(c["apertura"], c["chiusura"]) <= max(c["apertura"], c["chiusura"]) <= c["massimo"]
            assert 0.0 <= c["p_rialzo"] <= 1.0 and 0.0 <= c["prob_direzione"] <= 1.0
            assert c["direzione"] == ("rialzo" if c["chiusura"] >= c["apertura"] else "ribasso")
            if costruzione in ("K1", "K2"):
                assert c["prob_direzione"] >= 0.5                       # la candela mostra la direzione piu' probabile


def test_constructions_follow_their_definitions(df):
    t = 900
    S = CP.serie(df)
    comp = {h: CP.componenti(S, t, h) for h in CP.ORIZZONTI}
    pdir = {h: comp[h]["p_L1"] for h in CP.ORIZZONTI}
    prev, k1 = KA.prevedi_candele(df, t, "K1", DIRS)
    _, k2 = KA.prevedi_candele(df, t, "K2", DIRS)
    _, k0 = KA.prevedi_candele(df, t, "K0", DIRS)
    c0 = prev["chiusura_origine"]
    assert k1[0]["apertura"] == pytest.approx(c0 * comp[1]["rapporto_apertura"]) == pytest.approx(k2[0]["apertura"])
    for h in (2, 3, 4):
        assert k1[h - 1]["apertura"] == pytest.approx(k1[h - 2]["chiusura"] * comp[h]["rapporto_apertura"])                  # concatenata
        assert k2[h - 1]["apertura"] == pytest.approx(prev["settimane"][h - 2]["chiusura"] * comp[h]["rapporto_apertura"])    # ancorata alla mediana D4
    for h, c in zip(CP.ORIZZONTI, k2):
        s = 1 if pdir[h] >= 0.5 else -1
        assert c["chiusura"] == pytest.approx(c["apertura"] * np.exp(s * comp[h]["corpo"]))
        assert c["massimo"] == pytest.approx(max(c["apertura"], c["chiusura"]) * np.exp(comp[h]["ombra_sup"]))
        assert c["minimo"] == pytest.approx(min(c["apertura"], c["chiusura"]) * np.exp(-comp[h]["ombra_inf"]))
    assert [c["chiusura"] for c in k0] == [s["chiusura"] for s in prev["settimane"]]                                           # K0 = Fase 2b


def test_median_similar_case_rules(df):
    S = CP.serie(df)
    for t in (500, 1200):
        St = CP.Serie(S.o[: t + 1], S.h[: t + 1], S.l[: t + 1], S.c[: t + 1], S.date[: t + 1])
        i = KA.caso_mediano(St, t)
        assert i is not None and 12 <= i <= t - 4 and not S.bad[i + 1:i + 5].any()
        cand, j = KA.k3(St, t)
        assert j == i and cand[0]["chiusura"] == pytest.approx(S.c[i + 1] * S.c[t] / S.c[i]) and cand[3]["massimo"] == pytest.approx(S.h[i + 4] * S.c[t] / S.c[i])


def test_anomalous_open_is_excluded_from_estimates(df):
    """L'apertura incoerente (2,58 nella settimana dell'8 novembre 2004) non entra in nessuna stima."""
    S = CP.serie(df)
    bad = np.flatnonzero(S.bad)
    assert len(bad) == 1 and S.o[bad[0]] == pytest.approx(2.58) and str(S.date[bad[0]])[:10] == "2004-11-12"
    d2 = df.copy()
    d2.loc[bad[0], "open"] = 0.5                                   # un altro valore incoerente: nulla deve cambiare
    assert CP.componenti(CP.serie(d2), 900, 1) == CP.componenti(S, 900, 1)
    d3 = df.copy()
    d3.loc[bad[0], "open"] = 13.0                                  # valore coerente: la settimana rientra e le stime cambiano
    assert CP.componenti(CP.serie(d3), 900, 1) != CP.componenti(S, 900, 1)


def test_median_regression_matches_statsmodels():
    import statsmodels.api as sm
    rng = np.random.default_rng(4)
    X = rng.standard_normal((500, 4))
    y = np.abs(0.1 + 0.05 * X[:, 0] + 0.08 * rng.standard_t(4, 500))
    x_new = np.array([0.3, -0.2, 0.1, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.QuantReg(y, sm.add_constant(X)).fit(q=0.5).params
    assert CP.mediana_condizionata(X, y, x_new) == pytest.approx(max(0.0, float(np.concatenate([[1.0], x_new]) @ ref)), abs=2e-3)
    assert 0.0 < CP.logistica(X, (X[:, 0] > 0).astype(float), np.array([2.0, 0, 0, 0])) < 1.0
    assert CP.logistica(X, (X[:, 0] > 0).astype(float), np.array([2.0, 0, 0, 0])) > 0.9


# ------------------------------------------------------------------ pagella e regola di scelta

def test_scorecard_known_values():
    g = pd.DataFrame({"idx_origine": [1, 2], "chiusura_origine": [20.0, 20.0],
                      "K1_apertura": [20.0, 21.0], "K1_massimo": [22.0, 22.0], "K1_minimo": [18.0, 19.0], "K1_chiusura": [19.0, 20.0],
                      "apertura_reale": [21.0, 20.0], "massimo_reale": [23.0, 26.0], "minimo_reale": [19.0, 20.0], "chiusura_reale": [20.0, 25.0]})
    r = VC.pagella_riga(g, "K1")
    assert r["direzione_corpo"] == 0.5 and r["direzione_chiusura_vs_origine"] == 0.0          # 1° caso reale invariato rispetto all'origine: escluso
    assert r["mae_chiusura"] == pytest.approx(3.0) and r["mae_apertura"] == pytest.approx(1.0)
    assert r["rapporto_corpo"] == pytest.approx(2.0 / 6.0) and r["rapporto_escursione"] == pytest.approx(7.0 / 10.0)
    assert r["sovrapposizione"] == pytest.approx((3 / 5 + 2 / 7) / 2)
    assert r["chiusura_nel_corpo"] == 0.5 and r["chiusura_nell_escursione"] == 0.5


def _pag(vals):
    rows = []
    for k, (corpo, esc, iou) in vals.items():
        for h in (1, 2, 3, 4):
            rows.append({"costruzione": k, "h": h, "rapporto_corpo": corpo, "rapporto_escursione": esc, "sovrapposizione": iou})
    return pd.DataFrame(rows)


def test_selection_rule():
    a = VC.scegli_costruzione(_pag({"K0": (0.3, 0.9, 0.50), "K1": (0.9, 1.1, 0.30), "K2": (1.1, 0.85, 0.35), "K3": (1.0, 1.3, 0.60)}))
    assert a["costruzione"] == "K2" and a["ammesse"] == ["K1", "K2"]                 # tra le ammesse, sovrapposizione piu' alta
    b = VC.scegli_costruzione(_pag({"K0": (0.3, 0.9, 0.50), "K1": (0.7, 0.6, 0.30), "K2": (0.75, 0.6, 0.20), "K3": (0.6, 0.9, 0.60)}))
    assert b["costruzione"] == "K2" and b["ammesse"] == [] and b["spareggio"] is None  # nessuna ammessa: corpo piu' vicino a 1
    c = VC.scegli_costruzione(_pag({"K0": (0.3, 0.9, 0.50), "K1": (0.75, 0.6, 0.30), "K2": (0.75, 0.6, 0.35), "K3": (0.6, 0.9, 0.60)}))
    assert c["costruzione"] == "K2" and c["spareggio"]["in_parita"] == ["K1", "K2"]   # parita' esatta: decide la sovrapposizione


def test_walk_forward_matches_single_origin_candles(df):
    path = VC.OUT / "previsioni_dev.csv"
    if not path.exists():
        pytest.skip("eseguire prima python -m src.run_phase2c")
    import json
    w = pd.read_csv(path, parse_dates=["data_origine"])
    info = json.loads((VC.OUT / "info.json").read_text())
    md = {int(h): m for h, m in info["modello_direzione"].items()}
    t = 1000
    g = w[w["data_origine"] == df["date"].iloc[t]].set_index("h")
    for k in KA.COSTRUZIONI:
        _, cand = KA.prevedi_candele(df, t, k, md)
        for c in cand:
            for campo in ("apertura", "massimo", "minimo", "chiusura"):
                assert c[campo] == pytest.approx(g.loc[c["h"], f"{k}_{campo}"], abs=3e-4), (k, c["h"], campo)
    assert w["data_obiettivo"].max() <= str(df["date"].max().date())
    assert info["scelta"]["costruzione"] in KA.COSTRUZIONI and KA.COSTRUZIONI == ("K0", "K1", "K2", "K3")


# ------------------------------------------------------------------ registro e verifica

def _file_grezzo(df, path, extra: int, nota_parziale: bool | None = None):
    rows = [{"week_start": (r.date - pd.Timedelta(days=r.date.weekday())).date(), "last_trading_day": r.date.date(), "open": r.open, "high": r.high,
             "low": r.low, "close": r.close, "trading_days": 5, "note": ""} for r in df.itertuples()]
    rng = np.random.default_rng(11)
    last, c = df["date"].iloc[-1], float(df["close"].iloc[-1])
    for k in range(1, extra + 1):
        c2 = round(c * float(np.exp(0.08 * rng.standard_normal())), 2)
        day = last + pd.Timedelta(days=7 * k)
        rows.append({"week_start": (day - pd.Timedelta(days=4)).date(), "last_trading_day": day.date(), "open": round(c * 1.03, 2),
                     "high": round(max(c * 1.03, c2) * 1.04, 2), "low": round(min(c * 1.03, c2) * 0.96, 2), "close": c2, "trading_days": 5, "note": ""})
        c = c2
    if nota_parziale is not None:                                   # settimana in corso: 3 sedute, ultima il mercoledi'
        lun = last + pd.Timedelta(days=7 * (extra + 1) - 4)
        rows.append({"week_start": lun.date(), "last_trading_day": (lun + pd.Timedelta(days=2)).date(), "open": c, "high": c * 1.02, "low": c * 0.98,
                     "close": c * 1.01, "trading_days": 3, "note": "Settimana in corso" if nota_parziale else ""})
    pd.DataFrame(rows).to_csv(path, index=False)
    return last


def test_registry_never_duplicates_and_live_check_works(df, tmp_path, capsys):
    base, agg = tmp_path / "base.csv", tmp_path / "aggiornato.csv"
    last = _file_grezzo(df, base, extra=0)
    _file_grezzo(df, agg, extra=6)
    out = tmp_path / "out"
    r1 = F.main(["--file", str(base)], out_dir=out, oggi="2024-11-09")
    r2 = F.main(["--file", str(base)], out_dir=out, oggi="2024-11-10")
    reg = pd.read_csv(out / "registro_previsioni.csv")
    assert r1["registrata"] and not r2["registrata"] and len(reg) == 4 and set(reg["data_esecuzione"]) == {"2024-11-09"}
    assert "già registrata" in capsys.readouterr().out
    r3 = F.main(["--file", str(base), "--origine", "2024-06-28"], out_dir=out)        # con --origine non si registra
    assert not r3["registrata"] and len(pd.read_csv(out / "registro_previsioni.csv")) == 4
    v0 = F.main(["--file", str(base), "--verifica"], out_dir=out)                      # nessuna settimana completata
    assert v0["dettaglio"].empty
    v = F.main(["--file", str(agg), "--verifica"], out_dir=out)                        # file aggiornato: 4 settimane confrontabili
    det, pag = v["dettaglio"], v["pagella"]
    assert len(det) == 4 and list(pag["n"]) == [1, 1, 1, 1] and pag["meno_di_20_casi"].all() and det["registrata_prima_dell_esito"].all()
    real = pd.read_csv(agg).iloc[len(df): len(df) + 4]
    np.testing.assert_allclose(det["errore_chiusura"], np.abs(reg["chiusura"].to_numpy() - real["close"].to_numpy()), atol=1e-3)
    assert det["sovrapposizione"].between(0, 1).all()
    assert str(last.date()) == reg["data_origine"].iloc[0]
    F.main(["--file", str(agg)], out_dir=out, oggi="2024-12-21")                       # nuova origine: si aggiunge
    assert len(pd.read_csv(out / "registro_previsioni.csv")) == 8


def test_registry_functions(tmp_path):
    prev = {"data_origine": "2024-01-05", "chiusura_origine": 13.0,
            "settimane": [{"h": h, "data": f"2024-01-{5 + 7 * h:02d}", "quantili": {"q50": 13.5}, "fasce": {"50": [12, 15], "80": [11, 17], "90": [10, 19]},
                           "probabilita": {"VIX>20": 0.1, "VIX>25": 0.05, "VIX>30": 0.01}} for h in (1, 2)]}
    cand = [{"apertura": 13.4, "massimo": 14.5, "minimo": 12.5, "chiusura": 12.9, "direzione": "ribasso", "prob_direzione": 0.64, "costruzione": "K2", "modello_direzione": "L1"}] * 2
    nuove = RG.righe(prev, cand, "2024-01-06")
    p = tmp_path / "r.csv"
    assert RG.aggiungi(p, nuove)[0] and not RG.aggiungi(p, nuove)[0] and len(pd.read_csv(p)) == 2
    storia = pd.DataFrame({"date": pd.to_datetime(["2024-01-05", "2024-01-12"]), "open": [13.0, 13.4], "high": [14.0, 16.0], "low": [12.0, 12.9], "close": [13.0, 15.5]})
    det, pag = RG.verifica(pd.read_csv(p, dtype={"data_origine": str, "data_esecuzione": str}), storia)
    assert len(det) == 1 and det["errore_chiusura"].iloc[0] == pytest.approx(2.6) and not det["direzione_corpo_ok"].iloc[0]
    assert det["sovrapposizione"].iloc[0] == pytest.approx((14.5 - 12.9) / (16.0 - 12.5)) and not det["chiusura_nella_fascia_50"].iloc[0] and det["chiusura_nella_fascia_80"].iloc[0]


# ------------------------------------------------------------------ settimana incompleta senza nota

def test_incomplete_week_detected_without_note(df, tmp_path, capsys):
    con, senza = tmp_path / "con_nota.csv", tmp_path / "senza_nota.csv"
    _file_grezzo(df, con, extra=2, nota_parziale=True)
    _file_grezzo(df, senza, extra=2, nota_parziale=False)
    a, ia = F.leggi_storico(con)
    b, ib = F.leggi_storico(senza)
    assert len(a) == len(b) == len(df) + 2 and ia["settimane_incomplete_escluse"] == ib["settimane_incomplete_escluse"] == 1
    assert not ia["avvisi"] and "INCOMPLETA" in ib["avvisi"][0]
    F.main(["--file", str(senza), "--origine", str(df["date"].iloc[-1].date())], out_dir=tmp_path / "o")
    assert "ATTENZIONE" in capsys.readouterr().out                                    # in caso di dubbio avvisa a schermo


def test_incomplete_week_rule_and_holiday_calendar():
    assert SE.ultima_riga_incompleta("2026-11-09", "2026-11-12", 4, None)[0]           # giovedi', venerdi' non festivo
    assert SE.ultima_riga_incompleta("2026-11-09", "2026-11-12", 4, "Settimana in corso") == (True, None)
    assert not SE.ultima_riga_incompleta("2024-03-25", "2024-03-28", 4, None)[0]       # Venerdi' Santo
    assert not SE.ultima_riga_incompleta("2014-06-30", "2014-07-03", 4, float("nan"))[0]   # 4 luglio di venerdi' (prima del 2022: nessun valore nei festivi)
    assert SE.ultima_riga_incompleta("2026-11-23", "2026-11-26", 4, None)[0]           # dal 2022 il VIX ha un valore anche a Thanksgiving: manca il venerdi'
    assert not SE.ultima_riga_incompleta("2026-11-02", "2026-11-06", 5, None)[0]
    inc, msg = SE.ultima_riga_incompleta("2001-09-10", "2001-09-10", 1, None)          # chiusura straordinaria: come ultima riga sarebbe "incompleta"
    assert inc and "ATTENZIONE" in msg
    ok, nota = SE.ultima_riga_incompleta("2012-10-29", "2012-11-02", 3, None)          # sedute mancanti ma ultima seduta regolare: completa, caso dubbio
    assert not ok and "dubbio" in nota
    assert SE.festivo("2024-03-29") and SE.festivo("2022-06-20") and SE.festivo("2021-12-24") and not SE.festivo("2021-12-31") and SE.festivo("2023-11-23")


def test_holiday_calendar_explains_short_weeks_in_dev_file():
    """Sul file di sviluppo: ogni settimana che finisce prima del venerdi' e' spiegata dal calendario (tranne l'11 settembre 2001)."""
    d = pd.read_csv(C.DEV_FILE, parse_dates=["date", "week_start"])
    corte = d[d["date"] < d["week_start"] + pd.Timedelta(days=4)]
    sbagliate = [str(r.week_start.date()) for r in corte.itertuples() if SE.ultima_riga_incompleta(r.week_start, r.date, r.trading_days, None)[0]]
    assert sbagliate == ["2001-09-10"] and len(corte) > 30
    diverse = sum(len(SE.sedute_attese(r.week_start)) != r.trading_days for r in d.itertuples())
    assert diverse <= 5                                             # solo chiusure straordinarie (11 settembre, lutti nazionali, uragano Sandy)


# ------------------------------------------------------------------ guardia

def test_phase2c_modules_read_only_dev():
    banned = r"HOLDOUT_FILE|CLEAN_FILE|RAW_FILE|vix_weekly_holdout|vix_weekly_clean|VIX_settimanale|external"
    for f in sorted(DIR.glob("*.py")) + [pathlib.Path(C.ROOT) / "src" / "run_phase2c.py"]:
        src = f.read_text()
        assert not re.search(banned, src), f.name
        assert "read_csv(C." not in src, f"{f.name}: i dati si leggono con fase2b.dati"
    assert "with K.guardia_dati():" in (pathlib.Path(C.ROOT) / "src" / "run_phase2c.py").read_text()
    with K.guardia_dati():
        with pytest.raises(PermissionError):
            open(C.HOLDOUT_FILE)
