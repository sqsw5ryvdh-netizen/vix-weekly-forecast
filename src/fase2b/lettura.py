"""Fase 2b, Parte B: lettura della situazione all'origine t. Solo descrizione: nessun modello.

Ogni percentile, media, soglia e tabella usa solo le righe 0..t. Fasce e direzione: definizioni della Fase 1.
I «casi simili» usano lo stato di M6 ma NON sono il modello M6: sono scelti con una distanza minima di 8
settimane tra un caso e l'altro e servono solo a illustrare.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..fase1 import comune as K
from ..fase1 import modelli as M

ANNI5 = 260
MIN_CASI = 20
DISTANZA_CASI, MAX_CASI, ESITO = 8, 20, 4


def _pct_rank(v: np.ndarray, x: float) -> float:
    return float(100 * (v <= x).mean())


def casi_simili(x: np.ndarray, t: int) -> list[int]:
    """Indici delle situazioni passate piu' simili (stato di M6), con esito a 4 settimane noto e distanza >= 8 settimane."""
    S = M.analog_states(x[: t + 1])                    # riga j <-> indice 12 + j
    Z = (S - S.mean(axis=0)) / S.std(axis=0)
    n_cand = len(S) - ESITO
    if n_cand < 1:
        return []
    dist = np.sqrt(((Z[:n_cand] - Z[-1]) ** 2).sum(axis=1))
    scelti: list[int] = []
    for j in np.argsort(dist, kind="stable"):
        i = int(j) + 12
        if all(abs(i - k) >= DISTANZA_CASI for k in scelti):
            scelti.append(i)
            if len(scelti) == MAX_CASI:
                break
    return scelti


def lettura(df: pd.DataFrame, t: int) -> dict:
    d = df.iloc[: t + 1]
    c = d["close"].to_numpy(dtype=float)
    x = np.log(c)
    dates = pd.to_datetime(d["date"]).to_numpy()
    now = float(c[-1])
    dist = lambda ref: {"valore": float(ref), "distanza_punti": now - float(ref), "distanza_pct": 100 * (now / float(ref) - 1)}

    L = {"data_origine": str(pd.Timestamp(dates[-1]).date()), "settimane_di_storia": int(t + 1),
         "prima_data": str(pd.Timestamp(dates[0]).date())}
    L["livello"] = {"chiusura": now, "percentile_storia": _pct_rank(c, now), "percentile_5_anni": _pct_rank(c[-ANNI5:], now),
                    "n_settimane_5_anni": int(min(ANNI5, len(c))), "fascia": K.BAND7_LABELS[int(K.band7(now))]}
    L["riferimenti"] = {"media_4_settimane": dist(c[-4:].mean()), "media_12_settimane": dist(c[-12:].mean()),
                        "mediana_5_anni": dist(np.median(c[-ANNI5:])), "mediana_storia": dist(np.median(c))}
    pc = np.abs(c[1:] / c[:-1] - 1) * 100
    L["dinamica"] = {f"variazione_{k}_settimane": {"punti": now - float(c[-1 - k]), "pct": 100 * (now / float(c[-1 - k]) - 1)} for k in (1, 4, 12)}
    L["dinamica"] |= {"massimo_chiusure_12_settimane": float(c[-12:].max()), "minimo_chiusure_12_settimane": float(c[-12:].min()),
                      "ampiezza_mediana_recente_pct": float(np.median(pc[-12:])), "ampiezza_mediana_storica_pct": float(np.median(pc)),
                      "percentile_ampiezza_recente": _pct_rank(pd.Series(pc).rolling(12).median().dropna().to_numpy(), float(np.median(pc[-12:])))}
    # contesto: ultimo episodio con chiusura >= 30 (stessa regola di unione della Fase 0: distanza <= 4 settimane)
    ctx = {"settimane_dall_ultima_chiusura_sopra_30": None}
    over = np.flatnonzero(c >= 30.0)
    if len(over):
        j = int(over[-1])
        s = j
        for k in over[::-1]:
            if s - k - 1 <= K.MERGE_GAP:
                s = int(k)
            else:
                break
        p = s + int(np.argmax(c[s:j + 1]))
        ctx = {"settimane_dall_ultima_chiusura_sopra_30": int(t - j), "data_ultima_chiusura_sopra_30": str(pd.Timestamp(dates[j]).date()),
               "picco_episodio": float(c[p]), "data_picco": str(pd.Timestamp(dates[p]).date()), "settimane_dal_picco": int(t - p)}
    below = c < K.CALM_LEVEL
    run = int(len(c) - 1 - np.flatnonzero(~below)[-1]) if (~below).any() else int(len(c))
    ctx["settimane_consecutive_sotto_15"] = run if below[-1] else 0
    ctx["periodo_calmo"] = bool(below[-1] and run >= K.CALM_MIN_WEEKS)
    L["contesto"] = ctx
    # precedenti per fascia x direzione (definizioni della Fase 1)
    i = np.arange(4, t + 1)
    band, dirn = K.band7(c[i]), K.direction(x[i] - x[i - 4])
    b0, d0 = int(band[-1]), int(dirn[-1])
    prec = {"fascia": K.BAND7_LABELS[b0], "direzione": K.DIR_LABELS[d0], "variazione_log_4_settimane": float(x[t] - x[t - 4])}
    for h in (1, 4):
        m = (band == b0) & (dirn == d0) & (i + h <= t)
        ch = c[i[m] + h] - c[i[m]]
        prec[f"dopo_{h}"] = ({"n": int(m.sum()), "quota_piu_alto": float((ch > 0).mean()), "mediana_punti": float(np.median(ch)),
                              "p10_punti": float(np.quantile(ch, 0.10)), "p90_punti": float(np.quantile(ch, 0.90)),
                              "meno_di_20_casi": bool(m.sum() < MIN_CASI)} if m.any() else {"n": 0, "meno_di_20_casi": True})
    L["precedenti"] = prec
    casi = []
    for k in casi_simili(x, t):
        path = c[k + 1:k + ESITO + 1]
        casi.append({"data": str(pd.Timestamp(dates[k]).date()), "livello": float(c[k]),
                     "variazioni_pct": [float(100 * (v / c[k] - 1)) for v in path], "chiusura_dopo_4": float(path[-1]),
                     "variazione_4_pct": float(100 * (path[-1] / c[k] - 1)), "massimo_chiusure_4": float(path.max()), "minimo_chiusure_4": float(path.min())})
    v4 = np.array([k["variazione_4_pct"] for k in casi])
    L["casi_simili"] = {"nota": "selezione illustrativa: stato di M6, almeno 8 settimane tra un caso e l'altro, esito a 4 settimane già noto; non è il modello M6",
                        "n": len(casi), "meno_di_20_casi": bool(len(casi) < MIN_CASI), "casi": casi,
                        "quota_piu_alto_dopo_4": float((v4 > 0).mean()) if len(casi) else None,
                        "mediana_variazione_4_pct": float(np.median(v4)) if len(casi) else None}
    return L
