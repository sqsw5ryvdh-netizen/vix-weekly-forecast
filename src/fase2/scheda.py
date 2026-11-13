"""Fase 2, Parte B: scheda di previsione a una data origine.

Per ogni orizzonte usa il modello di densita' con il CRPS medio piu' basso nel walk-forward (regola fissata
prima; la scelta e' letta da reports/fase2/a_info.json). La scheda usa solo i dati fino all'origine e non
contiene alcun confronto con valori realizzati.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C
from ..fase1 import comune as K
from ..fase1 import modelli as M
from . import densita as D

OUT = C.REPORTS_DIR / "fase2"


def scheda(df: pd.DataFrame, t: int, scelta: dict[int, str]) -> dict:
    """Scheda all'origine di indice t. Vengono usate solo le righe 0..t di `df`."""
    x = df["log_close"].to_numpy()[: t + 1].copy()
    out = {"data_origine": str(pd.Timestamp(df["date"].iloc[t]).date()), "vix_origine": float(np.exp(x[-1])), "orizzonti": {}}
    for h in K.HORIZONS:
        name = scelta[h]
        c1 = D.centers(x, M.forecast_har, h)
        c7 = D.centers(x, M.forecast_combo, h) if name == "D1b" else None
        dist = D.DENSITY[name](D.contesto(x, t, h, c1, c7))
        q = dict(zip(D.Q_REPORT, np.exp(dist.quantiles(np.array(D.Q_REPORT)))))
        out["orizzonti"][str(h)] = {
            "modello": name, "quantili_riordinati": bool(dist.crossed), "mediana": float(q[0.50]),
            "quantili": {f"q{int(100 * k):02d}": float(v) for k, v in q.items()},
            "intervalli": {str(lev): [float(q[a]), float(q[b])] for lev, (a, b) in D.INTERVALS.items()},
            "probabilita": {e: dist.prob_above(D.event_threshold(e, x[-1])) for e in D.events_for(h)},
        }
    return out


def run() -> None:
    df = K.load_dev()
    info = json.loads((OUT / "a_info.json").read_text())
    scelta = {int(h): m for h, m in info["scelta_per_orizzonte"].items()}
    s = scheda(df, len(df) - 1, scelta)
    s["nota"] = ("Esempio all'ultima data del campione di sviluppo. Nessun confronto con il valore realizzato: "
                 "le date successive appartengono all'holdout bloccato.")
    (OUT / "b_scheda_esempio.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with K.guardia_dati():
        run()
