"""Fase 3: modelli (elenco CHIUSO). Ogni modello = variabili di M1 + variabili di un blocco.

Obiettivi:  T1 chiusura, log(VIX) a t + h (regressione diretta ai minimi quadrati, come M1);
            T2 direzione, 1 se la chiusura a t + h e' sopra la chiusura a t (logistica con la penalita' leggera di E1);
            T3 fasce (regressione quantilica come D4, griglia di 19 quantili).
Blocchi:    G1 (VIX giornaliero), X1-X4 (dati di mercato).
Combinazioni: X6 = tutte le variabili di G1 e X1-X4 con penalita' ridge; X7 = media semplice delle previsioni di G1, X1-X4.

Convenzione anti look-ahead: ogni funzione riceve SOLO la storia fino all'origine t — `x` = log-VIX settimanale
x[0..t], `Z` = variabili del blocco per le settimane 0..t (riga t = valori noti alla chiusura dell'ultima seduta della
settimana t) — e restituisce la previsione per t + h. Le coppie di stima sono solo quelle con esito noto (i + h <= t).

Parametri fissati prima della valutazione:
- un'origine e' ammessa se le variabili del blocco sono valide a t e se le settimane valide fino a t (indice >= 11)
  sono almeno 249: per un blocco senza buchi e' la stessa regola di M1 (260 settimane di storia);
- X6: variabili del blocco standardizzate nella finestra di stima; penalita' solo sulle variabili dei blocchi (le
  variabili di M1 e l'intercetta non sono penalizzate); intensita' = g x numero di coppie, con g scelto nella griglia
  GRIGLIA_RIDGE: si stima sulle coppie il cui esito precede le ultime 104 coppie della finestra e si sceglie il g
  con l'errore piu' basso su quelle 104 (errore quadratico per T1, Brier per T2); poi si ristima su tutta la finestra;
- T3 per le combinazioni (solo se scelte dalla regola): X6 = regressione quantilica con tutte le variabili, senza
  penalita'; X7 = media dei quantili dei cinque blocchi.
"""
from __future__ import annotations

import numpy as np

from .. import baselines as B
from .. import config as C
from ..fase2 import densita as D
from . import variabili as V

ORIZZONTI = (1, 4)
BLOCCHI = tuple(V.BLOCCHI)                       # G1, X1, X2, X3, X4
COMBINAZIONI = ("X6", "X7")
MODELLI = BLOCCHI + COMBINAZIONI
NOMI = {"G1": "G1 VIX giornaliero", "X1": "X1 struttura a termine (VIX3M)", "X2": "X2 future (VX1, VX2)", "X3": "X3 VVIX", "X4": "X4 S&P 500",
        "X6": "X6 tutte le variabili, ridge", "X7": "X7 media delle previsioni"}
MIN_VALIDE = C.MIN_TRAIN_WEEKS - 11              # 249
GRIGLIA_RIDGE = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
VALIDAZIONE = 104
RIDGE_LOGIT = D.LOGIT_RIDGE                      # 1.0, come E1
NOMI_M1 = ("x_t", "media4", "media12")


def colonne(modello: str) -> tuple[str, ...]:
    """Variabili del blocco (oltre a quelle di M1). X6 e X7 richiedono tutte le variabili."""
    return V.BLOCCHI[modello] if modello in V.BLOCCHI else V.TUTTE


def valide(Z: np.ndarray) -> np.ndarray:
    ok = np.isfinite(Z).all(axis=1) if Z.shape[1] else np.ones(len(Z), bool)
    ok[:11] = False
    return ok


def origini(Z: np.ndarray, h: int) -> np.ndarray:
    """Origini ammesse (dipende, per ogni t, solo dalle righe 0..t di Z)."""
    ok = valide(Z)
    t = np.arange(C.MIN_TRAIN_WEEKS - 1, len(Z) - h)
    return t[ok[t] & (np.cumsum(ok)[t] >= MIN_VALIDE)]


def _campione(x: np.ndarray, Z: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Indici i (11 <= i <= t - h, variabili valide) delle coppie di stima e variabili di M1 (riga j <-> indice 11 + j)."""
    i = np.arange(11, len(x) - h)
    return i[valide(Z)[i]], B.har_features(x)


def _std(Z: np.ndarray, i: np.ndarray) -> np.ndarray:
    mu, sd = Z[i].mean(axis=0), Z[i].std(axis=0)
    return (Z - mu) / np.where(sd > 0, sd, 1.0)


def _sigmoide(v):
    return 1 / (1 + np.exp(-np.clip(v, -30, 30)))


# ------------------------------------------------------------------ T1: chiusura

def t1(x: np.ndarray, Z: np.ndarray, h: int, usa=None, coef: bool = False):
    """Regressione diretta di x_{i+h} su [1, x_i, media4, media12, variabili del blocco]. Con Z senza colonne e' M1.

    `usa`: colonne di Z da mettere nel modello (la validita' delle righe si giudica comunque su tutta Z): con
    `usa=[]` si ottiene M1 stimato sulla stessa finestra del blocco.
    """
    i, F = _campione(x, Z, h)
    Zu = Z if usa is None else Z[:, usa]
    X, y = np.column_stack([F[i - 11], Zu[i]]), x[i + h]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    f = float(np.concatenate([F[-1], Zu[-1]]) @ beta)
    return (f, beta[1:] * X[:, 1:].std(axis=0) / y.std()) if coef else f


def _ridge(Xb: np.ndarray, Xz: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    A = np.column_stack([Xb, Xz])
    pen = np.zeros(A.shape[1])
    pen[Xb.shape[1]:] = lam
    return np.linalg.solve(A.T @ A + np.diag(pen), A.T @ y)


def _divisione(i: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Dentro la finestra di stima: ultime 104 coppie per scegliere la penalita', stima sulle coppie con esito precedente."""
    val = i[-VALIDAZIONE:]
    return i[i + h <= val[0]], val


def t1_ridge(x: np.ndarray, Z: np.ndarray, h: int, dettagli: bool = False):
    """X6 per T1: penalita' ridge sulle variabili dei blocchi, intensita' scelta dentro la finestra di stima."""
    i, F = _campione(x, Z, h)
    Zs = _std(Z, i)
    tr, val = _divisione(i, h)
    err = []
    for g in GRIGLIA_RIDGE:
        b = _ridge(F[tr - 11], Zs[tr], x[tr + h], g * len(tr))
        err.append(float(((x[val + h] - np.column_stack([F[val - 11], Zs[val]]) @ b) ** 2).mean()))
    g = GRIGLIA_RIDGE[int(np.argmin(err))]
    b = _ridge(F[i - 11], Zs[i], x[i + h], g * len(i))
    f = float(np.concatenate([F[-1], Zs[-1]]) @ b)
    return (f, g, b[1:] * np.concatenate([F[i - 11][:, 1:].std(axis=0), np.ones(Z.shape[1])]) / x[i + h].std()) if dettagli else f


# ------------------------------------------------------------------ T2: direzione

def _logit(Xs: np.ndarray, y: np.ndarray, pen: np.ndarray) -> np.ndarray:
    """Logistica con penalita' L2 (una per coefficiente) su variabili standardizzate; stesso schema di E1."""
    Zm = np.column_stack([np.ones(len(Xs)), Xs])
    P = np.diag(np.concatenate([[0.0], pen]))
    b = np.zeros(Zm.shape[1])
    b[0] = np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3))
    for _ in range(50):
        p = _sigmoide(Zm @ b)
        step = np.linalg.solve((Zm * (p * (1 - p))[:, None]).T @ Zm + P, Zm.T @ (y - p) - P @ b)
        b = b + step
        if np.abs(step).max() < 1e-9:
            break
    return b


def _xy_direzione(x: np.ndarray, Z: np.ndarray, h: int, usa=None):
    i, F = _campione(x, Z, h)
    Zu = Z if usa is None else Z[:, usa]
    A = np.column_stack([np.vstack([F[i - 11][:, 1:], F[-1][1:]]), np.vstack([Zu[i], Zu[-1]])])      # ultima riga = origine
    mu, sd = A[:-1].mean(axis=0), A[:-1].std(axis=0)
    As = (A - mu) / np.where(sd > 0, sd, 1.0)
    return i, As[:-1], As[-1], (x[i + h] > x[i]).astype(float)


def t2(x: np.ndarray, Z: np.ndarray, h: int, usa=None, coef: bool = False):
    """P(x_{t+h} > x_t): logistica su [x_i, media4, media12, variabili del blocco]. Con Z senza colonne e' la logistica di base."""
    _, Xs, x_new, y = _xy_direzione(x, Z, h, usa)
    b = _logit(Xs, y, np.full(Xs.shape[1], RIDGE_LOGIT))
    p = float(_sigmoide(np.concatenate([[1.0], x_new]) @ b))
    return (p, b[1:]) if coef else p


def t2_frequenza(x: np.ndarray, h: int) -> float:
    """Riferimento: frequenza storica (finestra espandibile) dei casi con chiusura a i + h sopra la chiusura a i."""
    return float((x[h:] > x[:-h]).mean())


def t2_ridge(x: np.ndarray, Z: np.ndarray, h: int, dettagli: bool = False):
    """X6 per T2: penalita' 1 sulle variabili di M1 (come la logistica di base), g x n sulle variabili dei blocchi."""
    i, Xs, x_new, y = _xy_direzione(x, Z, h)
    tr, val = _divisione(i, h)
    m_tr, m_val = np.isin(i, tr), np.isin(i, val)
    k = Z.shape[1]
    pen = lambda g, n: np.concatenate([np.full(3, RIDGE_LOGIT), np.full(k, g * n)])
    err = []
    for g in GRIGLIA_RIDGE:
        b = _logit(Xs[m_tr], y[m_tr], pen(g, int(m_tr.sum())))
        err.append(float(((_sigmoide(np.column_stack([np.ones(int(m_val.sum())), Xs[m_val]]) @ b) - y[m_val]) ** 2).mean()))
    g = GRIGLIA_RIDGE[int(np.argmin(err))]
    b = _logit(Xs, y, pen(g, len(i)))
    p = float(_sigmoide(np.concatenate([[1.0], x_new]) @ b))
    return (p, g, b[1:]) if dettagli else p


# ------------------------------------------------------------------ T3: fasce

def t3(x: np.ndarray, Z: np.ndarray, h: int) -> tuple[np.ndarray, bool]:
    """Quantili (griglia D.TAUS) di x_{t+h}: regressione quantilica su [1, x_i, media4, media12, variabili del blocco]
    (variabili del blocco standardizzate nella finestra). Con Z senza colonne e' D4. Quantili incrociati: riordinati e segnalati."""
    i, F = _campione(x, Z, h)
    Zs = _std(Z, i) if Z.shape[1] else Z
    beta = D.quantreg(np.column_stack([F[i - 11], Zs[i]]), x[i + h], D.TAUS)
    q = beta @ np.concatenate([F[-1], Zs[-1]])
    return np.sort(q), bool((np.diff(q) < 0).any())


# ------------------------------------------------------------------ motore

def walk(x: np.ndarray, Z: np.ndarray, h: int, fn, **kw) -> tuple[np.ndarray, list]:
    """Walk-forward, finestra espandibile, ristima a ogni passo. Al modello arrivano COPIE di x[: t + 1] e Z[: t + 1]."""
    o = origini(Z, h)
    return o, [fn(x[: t + 1].copy(), Z[: t + 1].copy(), h, **kw) for t in o.tolist()]
