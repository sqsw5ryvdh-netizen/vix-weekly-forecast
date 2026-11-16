"""Fase 2b: previsione a 1-4 settimane di chiusura, massimo e minimo, e costruzione delle candele previste.

Scelte fissate (nessun modello nuovo, nessun parametro scelto sui risultati):
- mediana e fasce della chiusura: D4 (regressione quantilica sulle variabili dell'HAR);
- probabilita' di superare 20, 25 e 30: D1 (M1 + errori passati con esito noto);
- massimo e minimo della settimana t+h: stessa regressione diretta di M1, stesse variabili (calcolate sulle
  chiusure), con obiettivo log(high) e log(low);
- riferimento semplice per massimo e minimo: ultima chiusura x rapporto mediano storico.
Tutte le funzioni ricevono un DataFrame e un indice di origine t e usano solo le righe 0..t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import baselines as B
from ..fase1 import modelli as M
from ..fase2 import densita as D

ORIZZONTI = (1, 2, 3, 4)
SOGLIE = (20.0, 25.0, 30.0)
FASCE = D.INTERVALS                      # {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95)}


def forecast_hl(x: np.ndarray, target_log: np.ndarray, h: int) -> float:
    """Regressione diretta di target_log[i + h] su [1, x_i, media4, media12] (come M1); previsione per t + h."""
    F = B.har_features(x)
    y = target_log[11 + h:]
    beta = np.linalg.lstsq(F[: len(y)], y, rcond=None)[0]
    return float(F[-1] @ beta)


def riferimento_hl(close: np.ndarray, target: np.ndarray, h: int) -> float:
    """Riferimento: ultima chiusura x mediana storica di target[i + h] / close[i] (solo casi con esito noto)."""
    return float(close[-1] * np.median(target[h:] / close[:-h]))


def candele(chiusura_origine: float, mediane, massimi, minimi) -> list[dict]:
    """Candele previste. Apertura = chiusura prevista della settimana precedente (la prima: ultima chiusura reale).

    Vincoli: massimo >= max(apertura, chiusura), minimo <= min(apertura, chiusura); le correzioni sono segnalate.
    """
    out, prev = [], float(chiusura_origine)
    for med, hi, lo in zip(mediane, massimi, minimi):
        top, bot = max(prev, med), min(prev, med)
        out.append({"apertura": prev, "chiusura": float(med), "massimo": float(max(hi, top)), "minimo": float(min(lo, bot)),
                    "massimo_grezzo": float(hi), "minimo_grezzo": float(lo),
                    "massimo_corretto": bool(hi < top), "minimo_corretto": bool(lo > bot)})
        prev = float(med)
    return out


def origine_minima() -> int:
    """Prima origine utilizzabile: D1 richiede 104 errori passati con esito noto anche all'orizzonte piu' lungo."""
    return D.first_origin(max(ORIZZONTI))


def prevedi(df: pd.DataFrame, t: int) -> dict:
    """Previsione all'origine t (indice di riga). Usa solo df.iloc[: t + 1]."""
    if t < origine_minima():
        raise ValueError(f"origine troppo vicina all'inizio dei dati: servono almeno {origine_minima() + 1} settimane di storia")
    d = df.iloc[: t + 1]
    close = d["close"].to_numpy(dtype=float)
    x, lh, ll = np.log(close), np.log(d["high"].to_numpy(dtype=float)), np.log(d["low"].to_numpy(dtype=float))
    righe = []
    for h in ORIZZONTI:
        c1 = D.centers(x, M.forecast_har, h)
        ctx = D.contesto(x, t, h, c1, None)
        d4 = D.d4(ctx)
        q = np.exp(d4.quantiles(np.array(D.Q_REPORT)))
        d1 = D.d1(ctx)
        righe.append({"h": h, "quantili": {f"q{int(100 * k):02d}": float(v) for k, v in zip(D.Q_REPORT, q)},
                      "quantili_riordinati": bool(d4.crossed), "m1": float(np.exp(c1[-1])),
                      "probabilita": {f"VIX>{int(s)}": float(d1.prob_above(np.log(s))) for s in SOGLIE},
                      "n_errori_d1": int(len(ctx.past(ctx.c1)[1])),
                      "massimo_previsto": float(np.exp(forecast_hl(x, lh, h))), "minimo_previsto": float(np.exp(forecast_hl(x, ll, h))),
                      "massimo_riferimento": riferimento_hl(close, d["high"].to_numpy(dtype=float), h),
                      "minimo_riferimento": riferimento_hl(close, d["low"].to_numpy(dtype=float), h)})
    cand = candele(close[-1], [r["quantili"]["q50"] for r in righe], [r["massimo_previsto"] for r in righe], [r["minimo_previsto"] for r in righe])
    data0 = pd.Timestamp(d["date"].iloc[-1])
    for r, c in zip(righe, cand):
        r |= c
        r["data"] = str((data0 + pd.Timedelta(days=7 * r["h"])).date())
        r["fasce"] = {str(lev): [r["quantili"][f"q{int(100 * a):02d}"], r["quantili"][f"q{int(100 * b):02d}"]] for lev, (a, b) in FASCE.items()}
    return {"data_origine": str(data0.date()), "indice_origine": int(t), "chiusura_origine": float(close[-1]),
            "settimane_di_storia": int(t + 1), "settimane": righe}
