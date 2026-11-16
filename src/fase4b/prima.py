"""Fase 4b: componenti della PRIMA candela prevista (settimana +1). Le settimane +2…+4 restano K2.

K4 (Parte C) — costruzione del corpo dalla direzione dell'etichetta:
  apertura = chiusura all'origine x rapporto di apertura (come K2);
  etichetta «ribasso»: chiusura = chiusura all'origine x exp(-d), d = media delle variazioni log assolute delle
                       settimane in calo tra le ultime 260 variazioni settimanali note all'origine;
  etichetta «rialzo»:  chiusura = chiusura all'origine x exp(+u), u = media delle variazioni log delle settimane in
                       rialzo tra le stesse 260;
  etichetta «neutra»:  chiusura = chiusura centrale della versione;
  ombre come in K2 (ampiezze della variante scelta nella Fase 4), sopra e sotto il corpo.
M1q (Parte D) — minimo: regressione sulla mediana di ln(minimo della settimana i + 1 / chiusura della settimana i) sulle
  variabili di M1, stimata sulle ultime 520 coppie con esito noto. Il minimo mostrato non puo' stare sopra il corpo:
  se la regressione lo mette piu' in alto di apertura o chiusura, vale il piu' basso dei due (caso segnalato).
Parametri fissati dalla consegna (260 e 520 settimane). Ogni funzione usa solo le settimane 0..t.
"""
from __future__ import annotations

import numpy as np

from .. import baselines as B
from ..fase2 import densita as D

FINESTRA_MOSSE = 260
FINESTRA_MINIMO = 520
COSTRUZIONI = ("K2", "K4")
MINIMI = ("M0", "M1q")


def mosse(x: np.ndarray) -> tuple[float, float]:
    """(d, u): ampiezza media delle settimane in calo e in rialzo tra le ultime 260 variazioni di x = log chiusura (0..t)."""
    dx = np.diff(x)[-FINESTRA_MOSSE:]
    giu, su = dx[dx < 0], dx[dx > 0]
    return (float(-giu.mean()) if len(giu) else 0.0), (float(su.mean()) if len(su) else 0.0)


def chiusura_k4(chiusura_origine: float, direzione: str, d: float, u: float, centrale: float) -> float:
    if direzione == "ribasso":
        return float(chiusura_origine * np.exp(-d))
    if direzione == "rialzo":
        return float(chiusura_origine * np.exp(u))
    return float(centrale)


def candela_k4(chiusura_origine: float, rapporto_apertura: float, direzione: str, d: float, u: float, centrale: float, ombra_sup: float, ombra_inf: float) -> dict:
    ap = float(chiusura_origine) * rapporto_apertura
    ch = chiusura_k4(chiusura_origine, direzione, d, u, centrale)
    return {"apertura": ap, "chiusura": ch, "massimo": max(ap, ch) * float(np.exp(ombra_sup)), "minimo": min(ap, ch) * float(np.exp(-ombra_inf))}


def minimo_m1q(x: np.ndarray, low: np.ndarray) -> float:
    """Minimo previsto della settimana t + 1 (punti VIX): chiusura_t x exp(mediana condizionata di ln(low_{i+1} / close_i))."""
    F = B.har_features(x)                               # riga j <-> indice 11 + j
    i = np.arange(11, len(x) - 1)[-FINESTRA_MINIMO:]    # coppie con esito noto: i + 1 <= t
    beta = D.quantreg(F[i - 11], np.log(low[i + 1]) - x[i], [0.5])[0]
    return float(np.exp(x[-1] + F[-1] @ beta))


def con_minimo(candela: dict, minimo: float) -> dict:
    """Sostituisce il minimo della candela; il minimo non puo' superare il corpo."""
    fondo = min(candela["apertura"], candela["chiusura"])
    return candela | {"minimo": float(min(minimo, fondo)), "minimo_limitato_dal_corpo": bool(minimo > fondo)}
