"""VIX9D: lettura della situazione all'origine t. Solo descrizione: nessun modello. Usa solo le settimane 0..t."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import modelli as M

ANNI5 = 260
SOGLIA_DIREZIONE = 0.10          # variazione log a 4 settimane oltre ±0,10: «in rialzo» / «in calo»
MIN_CASI = 20


def _pct(v: np.ndarray, x: float) -> float:
    return float(100 * (v <= x).mean())


def _direzione(d4: float) -> str:
    return "in rialzo" if d4 > SOGLIA_DIREZIONE else "in calo" if d4 < -SOGLIA_DIREZIONE else "stabile"


def lettura(df: pd.DataFrame, t: int) -> dict:
    d = df.iloc[: t + 1]
    c, x = d["close"].to_numpy(dtype=float), np.log(d["close"].to_numpy(dtype=float))
    vix = d["vix"].to_numpy(dtype=float)
    q = M.quintile(c[-1], c)
    out = {"data_origine": str(pd.Timestamp(d["date"].iloc[-1]).date()), "prima_data": str(pd.Timestamp(d["date"].iloc[0]).date()), "settimane_di_storia": int(t + 1),
           "livello": {"chiusura": float(c[-1]), "percentile_storia": _pct(c, c[-1]), "percentile_5_anni": _pct(c[-ANNI5:], c[-1]), "quintile": q, "mediana_storia": float(np.median(c)),
                       "media_12_settimane": float(c[-12:].mean())},
           "dinamica": {f"variazione_{k}_settimane": {"punti": float(c[-1] - c[-1 - k]), "pct": float(100 * (c[-1] / c[-1 - k] - 1))} for k in (1, 4, 12)}}
    if np.isfinite(vix[-1]):
        r = c / vix
        ok = np.isfinite(r)
        out["vix"] = {"chiusura_vix": float(vix[-1]), "rapporto": float(r[-1]), "distanza_punti": float(c[-1] - vix[-1]), "percentile_rapporto": _pct(r[ok], r[-1]), "rapporto_mediano_storico": float(np.median(r[ok]))}
    else:
        out["vix"] = None
    # precedenti: stesso quintile di livello (soglie della storia fino all'origine) e stessa direzione a 4 settimane, con esito a 2 settimane noto
    soglie = np.quantile(c, [0.2, 0.4, 0.6, 0.8])
    quint = np.searchsorted(soglie, c, side="right") + 1
    d4 = np.concatenate([np.full(4, np.nan), x[4:] - x[:-4]])
    dir0 = _direzione(d4[-1])
    i = np.arange(4, t - 1)
    i = i[(quint[i] == q) & (np.array([_direzione(v) for v in d4[i]]) == dir0)]
    prec = {"quintile": q, "direzione": dir0, "variazione_log_4_settimane": float(d4[-1]), "n": int(len(i)), "meno_di_20_casi": bool(len(i) < MIN_CASI)}
    for h in (1, 2):
        if len(i):
            v = c[i + h] - c[i]
            prec[f"dopo_{h}"] = {"quota_piu_alto": float((v > 0).mean()), "mediana_punti": float(np.median(v)), "p10_punti": float(np.quantile(v, 0.10)), "p90_punti": float(np.quantile(v, 0.90))}
    out["precedenti"] = prec
    return out
