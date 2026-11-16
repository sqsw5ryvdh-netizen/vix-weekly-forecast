"""Fase 2b: programma di previsione. Anti look-ahead di previsione, candele e lettura; coerenza di candele, fasce e
probabilita'; funzionamento con un file aggiornato; blocco dell'holdout per le misure di sviluppo."""
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
from src.fase2b import dati, lettura as LE, modello as MO, valuta as VB

DIR = pathlib.Path(C.ROOT) / "src" / "fase2b"
ORIGINI = (400, 700, 1000, 1290)


@pytest.fixture(scope="module")
def df():
    return dati.load_dev_candele()


@pytest.fixture(scope="module")
def previsioni(df):
    return {t: MO.prevedi(df, t) for t in ORIGINI}


def _stravolgi(df, t):
    d = df.copy()
    for c in ("open", "high", "low", "close"):
        d.loc[d.index > t, c] = d.loc[d.index > t, c] * 3.0 + 7.0
    d["log_close"] = np.log(d["close"])
    return d


# ------------------------------------------------------------------ anti look-ahead

@pytest.mark.parametrize("t", (700, 1000))
def test_forecast_candles_and_reading_ignore_data_after_origin(df, previsioni, t):
    """Previsione, candele e lettura con origine t non cambiano modificando (o togliendo) i dati successivi a t."""
    alt = _stravolgi(df, t)
    assert not np.allclose(alt["close"].iloc[t + 1:], df["close"].iloc[t + 1:])
    assert MO.prevedi(alt, t) == previsioni[t] == MO.prevedi(df.iloc[: t + 1].copy(), t)
    L = LE.lettura(df, t)
    assert LE.lettura(alt, t) == L == LE.lettura(df.iloc[: t + 1].copy(), t)
    assert MO.prevedi(_stravolgi(df, t - 1), t) != previsioni[t]            # il test "morde": cambiando l'origine la previsione cambia


def test_high_low_regression_and_reference_are_causal(df):
    c, hi = df["close"].to_numpy(), df["high"].to_numpy()
    x, lh, t, h = np.log(c), np.log(hi), 500, 3
    F_ = B.har_features(x[: t + 1])
    idx = np.arange(11, t - h + 1)
    beta = np.linalg.lstsq(F_[: len(idx)], lh[idx + h], rcond=None)[0]
    assert MO.forecast_hl(x[: t + 1], lh[: t + 1], h) == pytest.approx(float(F_[-1] @ beta))
    ratios = [hi[i + h] / c[i] for i in range(0, t - h + 1)]             # solo esiti noti: i + h <= t
    assert MO.riferimento_hl(c[: t + 1], hi[: t + 1], h) == pytest.approx(c[t] * np.median(ratios))


def test_similar_cases_rules(df):
    x = np.log(df["close"].to_numpy())
    for t in (300, 900, 1296):
        casi = LE.casi_simili(x, t)
        assert 0 < len(casi) <= LE.MAX_CASI and all(k + LE.ESITO <= t for k in casi)
        s = sorted(casi)
        assert all(b - a >= LE.DISTANZA_CASI for a, b in zip(s, s[1:]))
    L = LE.lettura(df, 900)
    assert "non è il modello M6" in L["casi_simili"]["nota"] and len(L["casi_simili"]["casi"][0]["variazioni_pct"]) == 4


def test_reading_matches_manual_values(df):
    t = 900
    c = df["close"].to_numpy()[: t + 1]
    L = LE.lettura(df, t)
    assert L["livello"]["percentile_storia"] == pytest.approx(100 * (c <= c[-1]).mean())
    assert L["livello"]["percentile_5_anni"] == pytest.approx(100 * (c[-260:] <= c[-1]).mean())
    assert L["riferimenti"]["media_12_settimane"]["valore"] == pytest.approx(c[-12:].mean())
    assert L["riferimenti"]["mediana_storia"]["valore"] == pytest.approx(np.median(c))
    assert L["dinamica"]["variazione_4_settimane"]["punti"] == pytest.approx(c[-1] - c[-5])
    i = np.arange(4, t + 1)
    x = np.log(c)
    m = (K.band7(c[i]) == K.band7(c[t])) & (K.direction(x[i] - x[i - 4]) == K.direction(x[t] - x[t - 4])) & (i + 4 <= t)
    assert L["precedenti"]["dopo_4"]["n"] == int(m.sum())
    assert L["precedenti"]["dopo_4"]["meno_di_20_casi"] == bool(m.sum() < 20)


# ------------------------------------------------------------------ coerenza di candele, fasce, probabilita'

def test_candles_bands_and_probabilities_are_coherent(previsioni):
    for t, p in previsioni.items():
        assert [s["h"] for s in p["settimane"]] == [1, 2, 3, 4]
        prev_close = p["chiusura_origine"]
        for s in p["settimane"]:
            assert s["apertura"] == pytest.approx(prev_close)                       # apertura = chiusura (prevista) precedente
            assert s["minimo"] <= min(s["apertura"], s["chiusura"]) <= max(s["apertura"], s["chiusura"]) <= s["massimo"]
            f = s["fasce"]
            assert f["90"][0] <= f["80"][0] <= f["50"][0] <= s["chiusura"] <= f["50"][1] <= f["80"][1] <= f["90"][1]
            assert s["chiusura"] == s["quantili"]["q50"]
            pr = [s["probabilita"][f"VIX>{k}"] for k in (20, 25, 30)]
            assert all(0.0 <= v <= 1.0 for v in pr) and pr[0] >= pr[1] >= pr[2]
            assert s["n_errori_d1"] >= D.MIN_ERR
            prev_close = s["chiusura"]


def test_candle_constraints_are_enforced_and_counted():
    c = MO.candele(20.0, [22.0, 21.0], [21.0, 25.0], [20.5, 18.0])
    assert c[0]["massimo"] == 22.0 and c[0]["massimo_corretto"] and c[0]["minimo"] == 20.0 and c[0]["minimo_corretto"]
    assert c[1] | {} == c[1] and c[1]["apertura"] == 22.0 and not c[1]["massimo_corretto"] and not c[1]["minimo_corretto"]
    assert (c[1]["minimo"], c[1]["massimo"]) == (18.0, 25.0)


def test_origin_too_early_is_refused(df):
    with pytest.raises(ValueError):
        MO.prevedi(df, MO.origine_minima() - 1)
    assert MO.ORIZZONTI == (1, 2, 3, 4) and MO.SOGLIE == (20.0, 25.0, 30.0)


# ------------------------------------------------------------------ il comando, anche con un file aggiornato

def _file_grezzo(df, path, extra: int, parziale: bool = True):
    """File nel formato grezzo costruito dal campione di sviluppo + `extra` settimane sintetiche (+ una parziale con nota)."""
    d = df.copy()
    rng = np.random.default_rng(7)
    rows = [{"week_start": (r.date - pd.Timedelta(days=4)).date(), "last_trading_day": r.date.date(), "open": r.open, "high": r.high,
             "low": r.low, "close": r.close, "trading_days": 5, "note": ""} for r in d.itertuples()]
    last, c = d["date"].iloc[-1], float(d["close"].iloc[-1])
    for k in range(1, extra + 1 + int(parziale)):
        c2 = round(c * float(np.exp(0.08 * rng.standard_normal())), 2)
        day = last + pd.Timedelta(days=7 * k)
        rows.append({"week_start": (day - pd.Timedelta(days=4)).date(), "last_trading_day": day.date(), "open": c, "high": round(max(c, c2) * 1.04, 2),
                     "low": round(min(c, c2) * 0.96, 2), "close": c2, "trading_days": 5,
                     "note": "Settimana in corso" if (parziale and k == extra + 1) else ""})
        c = c2
    pd.DataFrame(rows).to_csv(path, index=False)
    return last


def test_command_works_with_updated_file(df, tmp_path):
    """Il comando funziona con un file che contiene settimane in piu' e ignora la settimana incompleta."""
    f0, f6 = tmp_path / "base.csv", tmp_path / "aggiornato.csv"
    last = _file_grezzo(df, f0, extra=0, parziale=False)
    _file_grezzo(df, f6, extra=6, parziale=True)
    r6 = F.main(["--file", str(f6)], out_dir=tmp_path / "o6")
    assert r6["info"]["settimane_incomplete_escluse"] == 1 and r6["info"]["settimane_complete"] == len(df) + 6
    assert r6["previsione"]["data_origine"] == str((last + pd.Timedelta(days=42)).date())      # ultima settimana COMPLETA
    assert r6["md"].exists() and "Come leggere il grafico" in r6["md"].read_text()
    d0 = r6["previsione"]["data_origine"]
    for nome in (f"tabella_{d0}.csv", f"previsione_{d0}.json", f"g1_{d0}.png", f"g2_{d0}.png"):
        assert (tmp_path / "o6" / nome).stat().st_size > 0
    tab = pd.read_csv(tmp_path / "o6" / f"tabella_{d0}.csv")
    assert len(tab) == 4 and {"apertura", "massimo", "minimo", "chiusura", "copertura_storica_80", "p_vix_sopra_30", "mae_storico_massimo"} <= set(tab.columns)
    # stessa origine, file diverso: le settimane in piu' non cambiano nulla
    a = F.main(["--file", str(f6), "--origine", str(last.date())], out_dir=tmp_path / "oa")
    b = F.main(["--file", str(f0)], out_dir=tmp_path / "ob")
    assert a["previsione"] == b["previsione"] == MO.prevedi(df, len(df) - 1) | {"versione": "f1"} and a["lettura"] == b["lettura"]
    # origine non coincidente con una data settimanale: si usa l'ultima chiusura disponibile
    c = F.main(["--file", str(f6), "--origine", str((last + pd.Timedelta(days=3)).date())], out_dir=tmp_path / "oc")
    assert c["previsione"]["data_origine"] == str(last.date())


def test_command_rejects_wrong_format(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"date": ["2020-01-03"], "close": [12.0]}).to_csv(p, index=False)
    with pytest.raises(SystemExit):
        F.main(["--file", str(p)], out_dir=tmp_path)


def test_forecast_command_does_not_evaluate():
    """src.forecast legge il file grezzo ma non importa i moduli di valutazione ne' calcola errori o coperture."""
    src = (pathlib.Path(C.ROOT) / "src" / "forecast.py").read_text()
    assert not re.search(r"import.*valuta|evaluate|density_row|dm_loss|pinball|brier", src)
    assert not re.search(r"HOLDOUT_FILE|CLEAN_FILE|vix_weekly_holdout", src)
    assert "statistiche_sviluppo" in src and ("chiusura_metriche.csv" in src or "statistiche_programma.csv" in src)   # dalla Fase 4b le misure lette sono raccolte in un solo file


# ------------------------------------------------------------------ misure di sviluppo sotto guardia

def test_phase2b_modules_read_only_dev_and_models_never_use_open():
    banned = r"HOLDOUT_FILE|CLEAN_FILE|RAW_FILE|vix_weekly_holdout|vix_weekly_clean|VIX_settimanale|external"
    for f in sorted(DIR.glob("*.py")) + [pathlib.Path(C.ROOT) / "src" / "run_phase2b.py"]:
        src = f.read_text()
        assert not re.search(banned, src), f.name
        if f.name not in ("dati.py", "grafici.py"):
            assert "read_csv(C." not in src and not re.search(r"[\"']open[\"']|\.open\b", src), f"{f.name}: la colonna open è ammessa solo per disegnare"
    assert "with K.guardia_dati():" in (pathlib.Path(C.ROOT) / "src" / "run_phase2b.py").read_text()
    assert "open" in dati.load_dev_candele().columns


def test_walk_forward_matches_single_origin_forecast(df):
    """Il walk-forward vettoriale e la funzione usata dal programma danno gli stessi numeri."""
    path = VB.OUT / "previsioni_dev.csv"
    if not path.exists():
        pytest.skip("eseguire prima python -m src.run_phase2b")
    w = pd.read_csv(path, parse_dates=["data_origine"])
    t = 1000
    p = MO.prevedi(df, t)
    g = w[w["data_origine"] == df["date"].iloc[t]].set_index("h")
    for s in p["settimane"]:
        r = g.loc[s["h"]]
        for a, b in ((s["chiusura"], r["mediana_d4"]), (s["massimo"], r["massimo_previsto"]), (s["minimo"], r["minimo_previsto"]),
                     (s["apertura"], r["apertura_prevista"]), (s["fasce"]["80"][1], r["q90"]), (s["probabilita"]["VIX>25"], r["p_25"]),
                     (s["massimo_riferimento"], r["massimo_riferimento"])):
            assert a == pytest.approx(b, abs=2e-4)
    assert w["data_obiettivo"].max() <= str(df["date"].max().date())


def test_phase2b_metrics_consistent_with_phase2():
    a, b = VB.OUT / "chiusura_metriche.csv", C.REPORTS_DIR / "fase2" / "a_densita_metriche.csv"
    if not a.exists():
        pytest.skip("eseguire prima python -m src.run_phase2b")
    m, p2 = pd.read_csv(a).set_index("h"), pd.read_csv(b)
    p2 = p2[p2["modello"] == "D4"].set_index("h")
    for h in (1, 4):
        assert m.loc[h, "n"] == p2.loc[h, "n"]
        assert m.loc[h, "crps"] == pytest.approx(p2.loc[h, "crps"], rel=2e-3)
        assert m.loc[h, "copertura_80"] == pytest.approx(p2.loc[h, "copertura_80"], abs=0.005)


def test_g3_origins_follow_fixed_rule(df):
    path = VB.OUT / "g3_pannelli.csv"
    if not path.exists():
        pytest.skip("eseguire prima python -m src.run_phase2b")
    g = pd.read_csv(path, parse_dates=["data_origine"])
    n = len(df)
    attese = [df["date"].iloc[n - 1 - 4 - 52 * k] for k in range(12)][::-1]
    assert list(g["data_origine"]) == attese and g["data_origine"].max() + pd.Timedelta(days=28) <= df["date"].max()
    info = json.loads((VB.OUT / "info.json").read_text())["g3"]
    assert info["dentro_50"] == int(g["dentro_50"].sum()) <= info["dentro_80"] == int(g["dentro_80"].sum()) <= 12
