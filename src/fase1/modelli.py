"""Fase 1, Parte B: elenco CHIUSO dei modelli (M0-M7). Parametri fissati prima di vedere i risultati.

Ogni funzione riceve solo la storia `hist = x[: t + 1]` (log-VIX fino all'origine t) e restituisce la
previsione di x[t + h]. Soglie, medie, standardizzazioni e tabelle sono calcolate dentro la funzione,
quindi con la sola storia disponibile all'origine.
"""
from __future__ import annotations

import hashlib
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.arima.model import ARIMA

from .. import baselines as B
from . import comune as K

ROLL_WINDOW = 520        # M4
MIN_CELL = 20            # M5
K_ANALOG = 30            # M6 (15 e 60 solo come sensibilita')
MAX_H = max(K.HORIZONS)

ARMA_STATS = {"stime": 0, "non_convergenti": 0, "fallite": 0}
_arma_cache: dict[bytes, np.ndarray] = {}


def forecast_rw(hist: np.ndarray, h: int) -> float:
    """M0. Random walk."""
    return float(hist[-1])


def forecast_har(hist: np.ndarray, h: int) -> float:
    """M1. Tipo HAR della Fase 0 (stessa funzione): riferimento da battere."""
    return B.forecast_har(hist, h)


def _arma_path(hist: np.ndarray) -> np.ndarray:
    """Stima ARMA(1,1) con costante a massima verosimiglianza esatta e previsione iterata a 1..12 passi.

    La stima dipende solo da `hist`; la memoria serve a non ristimare lo stesso modello per h = 1, 4, 12
    e per la combinazione M7 (chiave = contenuto esatto della storia).
    """
    key = hashlib.sha1(hist.tobytes()).digest()
    if key not in _arma_cache:
        ARMA_STATS["stime"] += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(hist, order=(1, 0, 1), trend="c").fit()
            if not res.mle_retvals.get("converged", True):
                ARMA_STATS["non_convergenti"] += 1
            path = np.asarray(res.forecast(MAX_H), dtype=float)
            if not np.isfinite(path).all():
                raise FloatingPointError
        except Exception:                      # ripiego dichiarato: AR(1) OLS iterato
            ARMA_STATS["fallite"] += 1
            path = np.array([B.forecast_ar1(hist, k) for k in range(1, MAX_H + 1)])
        _arma_cache[key] = path
    return _arma_cache[key]


def forecast_arma(hist: np.ndarray, h: int) -> float:
    """M2. ARMA(1,1) su log-VIX, ristimato a ogni origine."""
    return float(_arma_path(hist)[h - 1])


def forecast_har_ext(hist: np.ndarray, h: int) -> float:
    """M3. HAR esteso: variabili di M1 + ultima variazione settimanale divisa in parte positiva e negativa."""
    F = B.har_features(hist)                           # riga j <-> indice i = 11 + j
    d = hist[11:] - hist[10:-1]                        # variazione x_i - x_{i-1}
    X = np.column_stack([F, np.maximum(d, 0.0), np.minimum(d, 0.0)])
    y = hist[11 + h:]
    beta = np.linalg.lstsq(X[: len(y)], y, rcond=None)[0]
    return float(X[-1] @ beta)


def forecast_har_roll(hist: np.ndarray, h: int) -> float:
    """M4. M1 stimato sulle ultime 520 settimane (tutta la storia finche' e' piu' corta)."""
    return B.forecast_har(hist[-ROLL_WINDOW:], h)


def forecast_table(hist: np.ndarray, h: int) -> float:
    """M5. Valore attuale + mediana storica della variazione a h nella cella fascia x direzione.

    Cella con meno di 20 casi -> sola fascia. Fascia senza alcun caso passato -> variazione nulla.
    Sono usati solo i casi i con esito gia' noto (i + h <= t).
    """
    t = len(hist) - 1
    i = np.arange(4, t - h + 1)
    if len(i) == 0:
        return float(hist[-1])
    band = K.band7(np.exp(hist[i]))
    dirn = K.direction(hist[i] - hist[i - 4])
    ch = hist[i + h] - hist[i]
    b0 = K.band7(np.exp(hist[t]))
    d0 = K.direction(hist[t] - hist[t - 4])
    cell = (band == b0) & (dirn == d0)
    if cell.sum() >= MIN_CELL:
        return float(hist[t] + np.median(ch[cell]))
    only = band == b0
    return float(hist[t] + (np.median(ch[only]) if only.any() else 0.0))


def analog_states(hist: np.ndarray) -> np.ndarray:
    """Stato per ogni indice i >= 12: [x_i, x_i - x_{i-4}, x_i - x_{i-12}, dev. std. delle 12 variazioni fino a i]."""
    i = np.arange(12, len(hist))
    sd = sliding_window_view(np.diff(hist), 12).std(axis=1, ddof=1)      # elemento j <-> i = 12 + j
    return np.column_stack([hist[i], hist[i] - hist[i - 4], hist[i] - hist[i - 12], sd])


def _analog(hist: np.ndarray, h: int, k: int) -> float:
    t = len(hist) - 1
    S = analog_states(hist)
    Z = (S - S.mean(axis=0)) / S.std(axis=0)           # standardizzazione sulla storia disponibile
    n_cand = len(S) - h                                # candidati i = 12 .. t - h: esito x_{i+h} gia' noto
    if n_cand < 1:
        return float(hist[-1])
    dist = np.sqrt(((Z[:n_cand] - Z[-1]) ** 2).sum(axis=1))
    idx = np.argsort(dist, kind="stable")[: min(k, n_cand)] + 12
    return float(hist[t] + np.median(hist[idx + h] - hist[idx]))


def forecast_analog(hist: np.ndarray, h: int) -> float:
    """M6. Analogia con le 30 situazioni passate piu' simili."""
    return _analog(hist, h, K_ANALOG)


def forecast_analog_k15(hist: np.ndarray, h: int) -> float:
    return _analog(hist, h, 15)


def forecast_analog_k60(hist: np.ndarray, h: int) -> float:
    return _analog(hist, h, 60)


def forecast_combo(hist: np.ndarray, h: int) -> float:
    """M7. Media semplice delle previsioni di M2, M3 e M6."""
    return float(np.mean([forecast_arma(hist, h), forecast_har_ext(hist, h), forecast_analog(hist, h)]))


MODELS = {"M0": forecast_rw, "M1": forecast_har, "M2": forecast_arma, "M3": forecast_har_ext, "M4": forecast_har_roll,
          "M5": forecast_table, "M6": forecast_analog, "M7": forecast_combo}
SENSITIVITY = {"M6_k15": forecast_analog_k15, "M6_k60": forecast_analog_k60}
NAMES = {"M0": "M0 Random walk", "M1": "M1 Tipo HAR (riferimento)", "M2": "M2 ARMA(1,1)", "M3": "M3 HAR esteso",
         "M4": "M4 HAR finestra 520", "M5": "M5 Tabella condizionata", "M6": "M6 Analogia (k = 30)", "M7": "M7 Combinazione",
         "M6_k15": "M6 con k = 15", "M6_k60": "M6 con k = 60"}
# Annidamento rispetto a M1: M0 e' un caso particolare di M1; M1 e' un caso particolare di M3.
NESTED_IN_M1 = ("M0",)       # modello piu' piccolo di M1 -> Clark-West con M0 parsimonioso
NESTS_M1 = ("M3",)           # modello piu' grande di M1 -> Clark-West con M1 parsimonioso
