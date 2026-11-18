"""Fase 4c, Parte A: previsione DIRETTA di massimo e minimo della settimana +1.

Obiettivo: ln(estremo della settimana i + 1 / chiusura della settimana i). Regressione sulla mediana (stesso stimatore
di D4) su [1, x_i, media a 4 settimane, media a 12 settimane, variabili dell'insieme], stimata sulle ultime 520 coppie
valide con esito noto all'origine; servono almeno 156 coppie, altrimenti la previsione non esiste. Le variabili
dell'insieme sono standardizzate nella finestra di stima. Con l'insieme S0 e l'estremo «minimo» e' esattamente il
modello M1q della Fase 4b.
Riferimento «regola fissa»: chiusura all'origine x mediana storica (espandibile) del rapporto estremo / chiusura precedente.
Ogni funzione riceve solo le settimane 0..t.
"""
from __future__ import annotations

import numpy as np

from .. import baselines as B
from ..fase2 import densita as D

FINESTRA = 520
MINIMO_COPPIE = 156
ESTREMI = ("massimo", "minimo")


def coppie(n: int, Z: np.ndarray) -> np.ndarray:
    """Indici i delle coppie di stima all'origine t = n - 1: 11 <= i <= t - 1, variabili valide a i; le ultime 520."""
    i = np.arange(11, n - 1)
    if Z.shape[1]:
        i = i[np.isfinite(Z[i]).all(axis=1)]
    return i[-FINESTRA:]


def mediana_diretta(x: np.ndarray, livelli: np.ndarray, Z: np.ndarray, z_new: np.ndarray | None = None) -> float:
    """Previsione (in punti VIX) del livello della settimana t + 1: chiusura_t x exp(mediana condizionata di
    ln(livelli[i + 1] / chiusura_i)). `Z[i]` = variabili dell'insieme per la coppia i; `z_new` = variabili per
    l'origine (se assente: Z[t]). NaN se le coppie valide sono meno di 156 o le variabili dell'origine mancano."""
    t = len(x) - 1
    z_new = Z[t] if z_new is None else np.asarray(z_new, dtype=float)
    i = coppie(len(x), Z)
    if len(i) < MINIMO_COPPIE or not np.isfinite(z_new).all():
        return float("nan")
    F = B.har_features(x)                               # riga j <-> indice 11 + j
    if Z.shape[1]:
        mu, sd = Z[i].mean(axis=0), Z[i].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        X, nuovo = np.column_stack([F[i - 11], (Z[i] - mu) / sd]), np.concatenate([F[-1], (z_new - mu) / sd])
    else:
        X, nuovo = F[i - 11], F[-1]
    beta = D.quantreg(X, np.log(livelli[i + 1]) - x[i], [0.5])[0]
    return float(np.exp(x[t] + nuovo @ beta))


def regola_fissa(close: np.ndarray, livelli: np.ndarray) -> float:
    """Riferimento: chiusura all'origine x mediana storica di livelli[i + 1] / chiusura[i] (solo esiti noti)."""
    return float(close[-1] * np.median(livelli[1:] / close[:-1]))
