"""VIX9D: verifica visiva sul passato. Dodici origini equidistanti nel periodo di valutazione, scelte con regola fissa:
tra le origini comuni a tutti i candidati (ordinate nel tempo, orizzonte +2), quelle di posizione round(k x (n - 1) / 11),
k = 0..11. Per ognuna: le 2 candele previste dalla versione G1 accanto alle 2 candele reali. Solo campione di sviluppo."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as G
from . import dati
from . import grafici
from . import versioni as VS

N_PANNELLI, STORIA = 12, 8


def origini(df: pd.DataFrame) -> list[int]:
    p = pd.read_csv(G.OUT / "previsioni_dev_g1.csv", parse_dates=["data_origine"])
    d = sorted(p.loc[p["h"] == 2, "data_origine"].unique())
    pos = {pd.Timestamp(x): i for i, x in enumerate(df["date"])}
    return [pos[pd.Timestamp(d[int(round(k * (len(d) - 1) / (N_PANNELLI - 1)))])] for k in range(N_PANNELLI)]


def run() -> None:
    df = dati.load_dev()
    sc = json.loads((G.OUT / "info.json").read_text())["scelte"]
    pan, rows = [], []
    for t in origini(df):
        prev, cand = VS.prevedi(df, t, "g1", sc)
        reali = df.iloc[t + 1:t + 3]
        pan.append({"previsione": grafici.vista(prev, cand), "storia": df.iloc[t - STORIA + 1:t + 1], "reali": reali})
        for c, r in zip(cand, reali.itertuples()):
            su = r.close > prev["chiusura_origine"]
            rows.append({"data_origine": prev["data_origine"], "h": c["h"], "massimo_previsto": c["massimo"], "massimo_reale": r.high, "minimo_previsto": c["minimo"], "minimo_reale": r.low,
                         "massimo_entro_5": bool(abs(c["massimo"] / r.high - 1) <= 0.05), "minimo_entro_5": bool(abs(c["minimo"] / r.low - 1) <= 0.05), "etichetta": c["direzione_chiusura"],
                         "etichetta_corretta": None if c["direzione_chiusura"] == "neutra" else bool(su == (c["direzione_chiusura"] == "rialzo"))})
    t = pd.DataFrame(rows)
    nn = t[t["etichetta"] != "neutra"]
    riga = (f"Su {len(t)} candele: massimo entro il 5% dal reale in {int(t['massimo_entro_5'].sum())}, minimo in {int(t['minimo_entro_5'].sum())}; "
            f"etichette non neutre {len(nn)}, corrette {int(nn['etichetta_corretta'].astype(bool).sum())}.")
    grafici.verifica(pan, G.FIG / "g3_verifica_passato.png", riga)
    t.to_csv(G.OUT / "g3_pannelli.csv", index=False, float_format="%.4f")
    (G.OUT / "g3.json").write_text(json.dumps({"riga": riga, "n_pannelli": N_PANNELLI, "regola": "12 origini equidistanti tra le origini comuni del periodo di valutazione (posizioni round(k x (n - 1) / 11))",
                                               "versione": "g1"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with dati.guardia():
        run()
