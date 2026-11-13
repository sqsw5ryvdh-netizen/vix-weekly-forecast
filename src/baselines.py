"""Modelli baseline e motore walk-forward.

Convenzione anti look-ahead: ogni funzione di previsione riceve SOLO la storia
`x[: t + 1]` (log-VIX osservato fino all'origine t inclusa) e restituisce la
previsione di x[t + h]. La previsione per la data-obiettivo s = t + h usa quindi
dati fino a s - h (per h = 1: fino a s - 1). Il motore `walk_forward` non passa mai
al modello valori successivi all'origine: lo verifica tests/test_no_lookahead.py.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import config as C

Forecaster = Callable[[np.ndarray, int], float]


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def forecast_rw(hist: np.ndarray, h: int) -> float:
    """Random walk: ultimo valore osservato."""
    return float(hist[-1])


def forecast_mean(hist: np.ndarray, h: int) -> float:
    """Media storica di lungo periodo (finestra espandibile)."""
    return float(hist.mean())


def forecast_ar1(hist: np.ndarray, h: int) -> float:
    """AR(1) su log-VIX stimato con OLS; previsione iterata: mu + phi^h (x_t - mu)."""
    a, phi = _ols(np.column_stack([np.ones(len(hist) - 1), hist[:-1]]), hist[1:])
    f = hist[-1]
    for _ in range(h):
        f = a + phi * f
    return float(f)


def har_features(hist: np.ndarray) -> np.ndarray:
    """Per ogni indice i >= 11: [1, x_i, media(x_{i-3..i}), media(x_{i-11..i})]. Usa solo valori <= i."""
    cs = np.concatenate([[0.0], np.cumsum(hist)])
    idx = np.arange(11, len(hist))
    m4 = (cs[idx + 1] - cs[idx - 3]) / 4
    m12 = (cs[idx + 1] - cs[idx - 11]) / 12
    return np.column_stack([np.ones(len(idx)), hist[idx], m4, m12])


def forecast_har(hist: np.ndarray, h: int) -> float:
    """Tipo HAR, previsione diretta: x_{i+h} = b0 + b1 x_i + b2 media4_i + b3 media12_i.

    Le coppie di stima sono solo quelle il cui obiettivo i + h cade dentro la storia.
    """
    F = har_features(hist)               # riga j <-> indice i = 11 + j
    y = hist[11 + h:]                    # obiettivi x_{i+h} per i = 11 .. len-1-h
    beta = _ols(F[: len(y)], y)
    return float(F[-1] @ beta)


MODELS: dict[str, Forecaster] = {
    "random_walk": forecast_rw,
    "media_storica": forecast_mean,
    "ar1": forecast_ar1,
    "har": forecast_har,
}


def walk_forward(x: np.ndarray, model: Forecaster, h: int, min_train: int = C.MIN_TRAIN_WEEKS):
    """Finestra espandibile con ristima a ogni passo.

    Origini t = min_train-1 .. len(x)-h-1. Restituisce (origini, previsioni di x[t+h]).
    Al modello viene passata una COPIA di x[:t+1]: non puo' ne' leggere ne' alterare il futuro.
    """
    origins = np.arange(min_train - 1, len(x) - h)
    preds = np.array([model(x[: t + 1].copy(), h) for t in origins])
    return origins, preds


# ------------------------------------------------------------------ baseline probabilistica

def band_index(level, thresholds=C.LEVEL_THRESHOLDS) -> np.ndarray:
    return np.searchsorted(np.asarray(thresholds), np.asarray(level), side="right")


def prob_markov(hist_level: np.ndarray, h: int, threshold: float = C.EVENT_THRESHOLD) -> float:
    """P(VIX_{t+h} > soglia) dalla matrice di transizione empirica a 1 passo stimata sulla storia, elevata a h.

    Stati = fasce di livello (soglie 15/20/25/30). Nessun lisciamento; una riga senza
    osservazioni viene sostituita dalla frequenza incondizionata degli stati.
    """
    th = np.asarray(C.LEVEL_THRESHOLDS)
    st = band_index(hist_level)
    k = len(th) + 1
    cnt = np.zeros((k, k))
    np.add.at(cnt, (st[:-1], st[1:]), 1)
    rows = cnt.sum(axis=1, keepdims=True)
    uncond = np.bincount(st, minlength=k) / len(st)
    P = np.where(rows > 0, cnt / np.where(rows > 0, rows, 1), uncond)
    ph = np.linalg.matrix_power(P, h)[st[-1]]
    assert threshold in th, "la soglia dell'evento deve coincidere con un confine di fascia"
    event_states = np.arange(k) >= np.searchsorted(th, threshold, side="left") + 1
    return float(ph[event_states].sum())


def prob_climatology(hist_level: np.ndarray, h: int, threshold: float = C.EVENT_THRESHOLD) -> float:
    """Frequenza storica incondizionata dell'evento (finestra espandibile)."""
    return float((hist_level > threshold).mean())


def prob_persistence(hist_level: np.ndarray, h: int, threshold: float = C.EVENT_THRESHOLD) -> float:
    """Persistenza: 1 se oggi il VIX e' sopra la soglia, 0 altrimenti."""
    return float(hist_level[-1] > threshold)


PROB_MODELS: dict[str, Forecaster] = {
    "markov_fasce": prob_markov,
    "climatologia": prob_climatology,
    "persistenza": prob_persistence,
}
