"""Fase 4c, Parte B: previsione incrociata con l'S&P 500 — legame B1 e versione reale B3.

B1  legame: regressione sulla mediana dell'obiettivo del VIX (massimo, minimo o chiusura della settimana i + 1, in
    rapporto log alla chiusura della settimana i) sulle variabili di S1 piu' rendimento ed escursione dell'S&P 500
    DELLA STESSA settimana i + 1. Finestra: ultime 520 coppie valide (minimo 156). Per le coppie di stima rendimento
    ed escursione della settimana i + 1 sono gia' noti all'origine (i + 1 <= t).
B3  versione reale: all'origine t rendimento ed escursione dell'S&P 500 della settimana t + 1 non sono noti; si
    prevedono con dati fino a t e si inseriscono nel legame B1:
      rendimento: regressione ai minimi quadrati su rendimento a 1 e a 4 settimane e ln(VIX);
      escursione: regressione ai minimi quadrati su escursione dell'ultima settimana, media a 4 e a 12 settimane e ln(VIX);
    stessa finestra (ultime 520 coppie valide, minimo 156).
Il tetto teorico B2 (che usa l'S&P 500 realmente avvenuto nella settimana +1) NON e' qui: sta in `tetto.py`, usato solo
dalla valutazione e mai dal programma.
Ogni funzione riceve solo le settimane 0..t.
"""
from __future__ import annotations

import numpy as np

from . import estremi as ES
from . import variabili as V

COL_S1 = list(V.S1)
PRED_REND = ["s3_rend_spx", "b_rend4"]
PRED_ESC = ["s3_escursione_spx", "b_escursione_spx_4", "b_escursione_spx_12"]
SERIE = ("spx",)


def _ols_finestra(X: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> float:
    """Minimi quadrati sulle ultime 520 righe valide (X e y finiti); NaN con meno di 156 righe o x_new non valido."""
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    j = np.flatnonzero(ok)[-ES.FINESTRA:]
    if len(j) < ES.MINIMO_COPPIE or not np.isfinite(x_new).all():
        return float("nan")
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(j)), X[j]]), y[j], rcond=None)[0]
    return float(np.concatenate([[1.0], x_new]) @ beta)


def prevedi_spx(x: np.ndarray, VV) -> tuple[float, float]:
    """(rendimento, escursione) dell'S&P 500 previsti per la settimana t + 1. `VV`: variabili settimanali 0..t (DataFrame)."""
    t = len(x) - 1
    rend, esc = VV["s3_rend_spx"].to_numpy(), VV["s3_escursione_spx"].to_numpy()
    Xr = np.column_stack([VV[PRED_REND].to_numpy(), x])
    Xe = np.column_stack([VV[PRED_ESC].to_numpy(), x])
    return _ols_finestra(Xr[:t], rend[1:t + 1], Xr[t]), _ols_finestra(Xe[:t], esc[1:t + 1], Xe[t])


def matrice_legame(VV) -> np.ndarray:
    """Riga i = [S1_i, rendimento S&P della settimana i + 1, escursione S&P della settimana i + 1]; l'ultima riga (i = t)
    ha i due valori futuri mancanti: non entra nella stima e va sostituita con la previsione (B3)."""
    s1 = VV[COL_S1].to_numpy()
    dopo = lambda c: np.concatenate([VV[c].to_numpy()[1:], [np.nan]])
    return np.column_stack([s1, dopo("s3_rend_spx"), dopo("s3_escursione_spx")])


def b3(x: np.ndarray, livelli: np.ndarray, VV) -> float:
    """Previsione reale del livello (massimo, minimo o chiusura) della settimana t + 1 con il legame B1 e l'S&P 500 previsto."""
    t = len(x) - 1
    r, e = prevedi_spx(x, VV)
    return ES.mediana_diretta(x, livelli, matrice_legame(VV), np.concatenate([VV[COL_S1].to_numpy()[t], [r, e]]))
