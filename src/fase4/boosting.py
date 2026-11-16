"""Fase 4, Parte B — N2 e N3: gradient boosting (scikit-learn), iperparametri FISSI.

N2  solo storico del VIX: variabili di M1 (ultima chiusura, media a 4 e a 12 settimane, in log), variabili della
    candela della Fase 2c (escursione, posizione della chiusura, ombra superiore e inferiore, variazione log a 1 e a
    4 settimane) e variabili giornaliere G1.
N3  come N2 con in piu' le variabili X1-X4, sul campione in cui esistono tutte.

Iperparametri (dalla consegna): 200 alberi, profondita' 3, tasso di apprendimento 0,05, sottocampionamento 0,8,
seed fisso (quello del progetto). Ristima ogni 13 settimane a partire dalla prima origine ammessa; tra una ristima e
l'altra il modello resta fermo e viene applicato alle variabili dell'origine.
Scelte fissate prima della valutazione, dove la consegna lasciava margine:
- T1: l'obiettivo dell'albero e' la VARIAZIONE x_{t+h} - x_t (gli alberi non estrapolano i livelli); la previsione della
  chiusura e' x_t + variazione prevista. Perdita quadratica.
- T2: classificatore con gli stessi iperparametri sull'evento x_{t+h} > x_t; si usa la probabilita' prevista.
- origini ammesse e coppie di stima: stesse regole della Fase 3 (`fase3.modelli`); la settimana con apertura
  incoerente (8 novembre 2004) ha le variabili della candela mancanti e non entra nelle stime come origine.
Convenzione anti look-ahead: le funzioni ricevono solo x[0..t] e Z[0..t].
"""
from __future__ import annotations

import numpy as np

from .. import baselines as B
from .. import config as C
from ..fase3 import modelli as M

PARAMETRI = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "random_state": C.SEED}
PASSO_STIMA = 13
NOMI_CANDELA = ("escursione", "posizione", "ombra_sup", "ombra_inf", "dlog1", "dlog4")


def _variabili(x: np.ndarray, Z: np.ndarray, i) -> np.ndarray:
    """Righe [x_i, media4_i, media12_i, Z_i] per gli indici i (>= 11)."""
    F = B.har_features(x)
    return np.column_stack([F[np.asarray(i) - 11][:, 1:], Z[np.asarray(i)]])


def stima(x: np.ndarray, Z: np.ndarray, h: int, obiettivo: str):
    """Stima sulla storia fino all'origine t = len(x) - 1: solo coppie con esito noto (i + h <= t)."""
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    i, _ = M._campione(x, Z, h)
    X = _variabili(x, Z, i)
    if obiettivo == "T1":
        return GradientBoostingRegressor(**PARAMETRI).fit(X, x[i + h] - x[i])
    return GradientBoostingClassifier(**PARAMETRI).fit(X, (x[i + h] > x[i]).astype(int))


def previsione(modello, x: np.ndarray, Z: np.ndarray, obiettivo: str) -> float:
    """Previsione all'origine t = len(x) - 1 con un modello gia' stimato (su dati non successivi a t)."""
    X = _variabili(x, Z, [len(x) - 1])
    if obiettivo == "T1":
        return float(x[-1] + modello.predict(X)[0])
    return float(modello.predict_proba(X)[0, list(modello.classes_).index(1)])


def walk(x: np.ndarray, Z: np.ndarray, h: int, obiettivo: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Walk-forward con ristima ogni 13 settimane. Al modello arrivano COPIE di x[: ts + 1], Z[: ts + 1] (stima) e
    di x[: t + 1], Z[: t + 1] (previsione). Restituisce (origini, previsioni, indici di ristima)."""
    o = M.origini(Z, h)
    out, ristime, modello, corrente = [], [], None, None
    for t in o.tolist():
        ts = int(o[0]) + PASSO_STIMA * ((t - int(o[0])) // PASSO_STIMA)
        if ts != corrente:
            modello, corrente = stima(x[: ts + 1].copy(), Z[: ts + 1].copy(), h, obiettivo), ts
            ristime.append(ts)
        out.append(previsione(modello, x[: t + 1].copy(), Z[: t + 1].copy(), obiettivo))
    return o, np.array(out), ristime
