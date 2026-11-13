"""Fase 2, Parte A: elenco CHIUSO dei modelli di densita' (D1-D6, D1b) e del modello per eventi E1.

Ogni modello riceve un `Contesto` costruito dal motore con la sola informazione disponibile all'origine t:
la storia x[: t + 1] e le previsioni puntuali (centri) fatte alle origini <= t. Gli errori passati sono
calcolabili solo per le origini i con i + h <= t, cioe' con esito gia' noto: gli altri non esistono nel contesto.
Parametri fissati prima della valutazione: MIN_ERR = 104, MIN_CELL = 20, K_ANALOG = 30, griglia di 19 quantili.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .. import baselines as B
from .. import config as C
from ..fase1 import comune as K
from ..fase1 import modelli as M

TAUS = np.round(np.arange(0.05, 0.951, 0.05), 2)           # griglia comune: 0.05, 0.10, ..., 0.95
Q_REPORT = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)      # quantili richiesti (sottoinsieme della griglia)
INTERVALS = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95)}
MIN_ERR = 104            # numero minimo di errori passati (con esito noto) per emettere una densita'
MIN_CELL = 20            # regola dei 20 casi (come M5)
K_ANALOG = M.K_ANALOG    # 30
O0 = C.MIN_TRAIN_WEEKS - 1                                 # prima origine con un centro disponibile
LOGIT_RIDGE = 1.0        # penalita' L2 di E1 sui coefficienti standardizzati (evita la separazione perfetta)
GARCH_SCALE = 10.0       # i residui sono moltiplicati per 10 prima della stima (solo condizionamento numerico)
LEVEL_EVENTS = (20.0, 25.0, 30.0)
RISE = 0.20              # rialzo settimanale oltre +20% (variazione semplice), solo h = 1
D5_HORIZONS = (1,)       # scelta fissata prima: D5 e' limitato a h = 1 (nessuna simulazione multi-passo)

STATS = {"incroci": {}, "garch_fallite": 0, "garch_stime": 0}


def first_origin(h: int) -> int:
    """Prima origine valutata: servono MIN_ERR errori con esito noto (origini O0 .. t - h)."""
    return O0 + MIN_ERR + h - 1


@dataclass
class Contesto:
    hist: np.ndarray           # x[0..t]
    h: int
    c1: np.ndarray             # centri M1 alle origini O0..t (l'ultimo e' il centro corrente)
    c7: np.ndarray | None      # centri M7 alle origini O0..t (solo per D1b)

    @property
    def t(self) -> int:
        return len(self.hist) - 1

    def past(self, centers: np.ndarray):
        """Origini i = O0 .. t - h (esito noto) e relativi errori x[i + h] - centro_i."""
        i = np.arange(O0, self.t - self.h + 1)
        return i, self.hist[i + self.h] - centers[: len(i)]


# ------------------------------------------------------------------ distribuzioni

class Empirica:
    """Centro + campione di scarti: quantili empirici (interpolazione lineare) e frequenze empiriche."""
    crossed = False

    def __init__(self, center: float, dev: np.ndarray):
        self.v = center + np.asarray(dev, dtype=float)

    def quantiles(self, taus=TAUS) -> np.ndarray:
        return np.quantile(self.v, taus)

    def prob_above(self, thr: float) -> float:
        return float((self.v > thr).mean())


class Griglia:
    """Distribuzione nota solo sulla griglia di quantili (D4). Quantili incrociati -> riordinati e segnalati."""

    def __init__(self, q: np.ndarray):
        self.crossed = bool((np.diff(q) < 0).any())
        self.q = np.sort(q)

    def quantiles(self, taus=TAUS) -> np.ndarray:
        return np.interp(taus, TAUS, self.q)

    def prob_above(self, thr: float) -> float:
        """CDF lineare tra i punti della griglia; nelle code prosegue con la pendenza dell'ultimo intervallo."""
        q, t = self.q, TAUS
        if thr <= q[0]:
            slope = (t[1] - t[0]) / max(q[1] - q[0], 1e-9)
            cdf = t[0] - slope * (q[0] - thr)
        elif thr >= q[-1]:
            slope = (t[-1] - t[-2]) / max(q[-1] - q[-2], 1e-9)
            cdf = t[-1] + slope * (thr - q[-1])
        else:
            cdf = np.interp(thr, q, t)
        return float(1 - np.clip(cdf, 0.0, 1.0))


class SkewT:
    """Centro + sigma x t asimmetrica standardizzata di Hansen (D5)."""
    crossed = False

    def __init__(self, center, sigma, dist, params):
        self.c, self.s, self.dist, self.par = center, sigma, dist, np.asarray(params)

    def quantiles(self, taus=TAUS) -> np.ndarray:
        return self.c + self.s * self.dist.ppf(np.asarray(taus), self.par)

    def prob_above(self, thr: float) -> float:
        return float(1 - self.dist.cdf(np.array([(thr - self.c) / self.s]), self.par)[0])


# ------------------------------------------------------------------ modelli di densita'

def _state(hist: np.ndarray, i):
    return K.band7(np.exp(hist[i])), K.direction(hist[i] - hist[np.asarray(i) - 4])


def d1(ctx: Contesto):
    """D1. M1 + quantili empirici incondizionati degli errori passati (riferimento)."""
    _, e = ctx.past(ctx.c1)
    return Empirica(ctx.c1[-1], e)


def d2(ctx: Contesto):
    """D2. Errori della stessa fascia di livello dell'origine; sotto 20 casi -> tutti gli errori (D1)."""
    i, e = ctx.past(ctx.c1)
    band, _ = _state(ctx.hist, i)
    b0, _ = _state(ctx.hist, ctx.t)
    m = band == b0
    return Empirica(ctx.c1[-1], e[m] if m.sum() >= MIN_CELL else e)


def d3(ctx: Contesto):
    """D3. Errori della stessa cella fascia x direzione; sotto 20 casi -> sola fascia; sotto 20 -> tutti."""
    i, e = ctx.past(ctx.c1)
    band, dirn = _state(ctx.hist, i)
    b0, d0 = _state(ctx.hist, ctx.t)
    cell, only = (band == b0) & (dirn == d0), band == b0
    use = cell if cell.sum() >= MIN_CELL else (only if only.sum() >= MIN_CELL else np.ones(len(e), bool))
    return Empirica(ctx.c1[-1], e[use])


def quantreg(X: np.ndarray, y: np.ndarray, taus, n_iter: int = 150, tol: float = 1e-8) -> np.ndarray:
    """Regressione quantilica con minimi quadrati ripesati (stesso schema di statsmodels QuantReg)."""
    b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    out = np.empty((len(taus), X.shape[1]))
    for k, tau in enumerate(taus):
        b = b_ols.copy()
        for _ in range(n_iter):
            r = y - X @ b
            r = np.where(np.abs(r) < 1e-6, np.where(r < 0, -1e-6, 1e-6), r)
            Xw = X * (np.where(r < 0, 1 - tau, tau) / np.abs(r))[:, None]
            bn = np.linalg.solve(X.T @ Xw, Xw.T @ y)
            done = np.abs(bn - b).max() < tol
            b = bn
            if done:
                break
        out[k] = b
    return out


def d4(ctx: Contesto):
    """D4. Regressione quantilica diretta di x_{i+h} su [1, x_i, media4, media12], un quantile alla volta."""
    F = B.har_features(ctx.hist)
    y = ctx.hist[11 + ctx.h:]
    beta = quantreg(F[: len(y)], y, TAUS)
    return Griglia(beta @ F[-1])


def d5(ctx: Contesto):
    """D5. M1 + GARCH(1,1) a media nulla sui residui in campione di M1, innovazioni t asimmetriche. Solo h = 1."""
    if ctx.h not in D5_HORIZONS:
        return None
    from arch import arch_model
    F = B.har_features(ctx.hist)
    y = ctx.hist[11 + ctx.h:]
    X = F[: len(y)]
    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    STATS["garch_stime"] += 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(res * GARCH_SCALE, mean="Zero", vol="GARCH", p=1, q=1, dist="skewt")
            fit = am.fit(disp="off", show_warning=False)
            sigma = float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0])) / GARCH_SCALE
        par = fit.params.to_numpy()[-2:]
        d = SkewT(ctx.c1[-1], sigma, am.distribution, par)
        q = d.quantiles()
        if not (np.isfinite(q).all() and np.isfinite(sigma) and sigma > 0):
            raise FloatingPointError
        return d
    except Exception:                   # ripiego dichiarato: quantili empirici dei residui in campione
        STATS["garch_fallite"] += 1
        return Empirica(ctx.c1[-1], res)


def d6(ctx: Contesto):
    """D6. Valore attuale + variazioni a h osservate dopo le 30 situazioni passate piu' simili (stato di M6)."""
    hist, h, t = ctx.hist, ctx.h, ctx.t
    S = M.analog_states(hist)
    Z = (S - S.mean(axis=0)) / S.std(axis=0)
    n_cand = len(S) - h
    dist = np.sqrt(((Z[:n_cand] - Z[-1]) ** 2).sum(axis=1))
    idx = np.argsort(dist, kind="stable")[: min(K_ANALOG, n_cand)] + 12
    return Empirica(hist[t], hist[idx + h] - hist[idx])


def d1b(ctx: Contesto):
    """D1b. Come D1 ma con centro M7 (solo controllo)."""
    _, e = ctx.past(ctx.c7)
    return Empirica(ctx.c7[-1], e)


DENSITY = {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6, "D1b": d1b}
SELECTABLE = ("D1", "D2", "D3", "D4", "D5", "D6")       # D1b e' solo un controllo: non entra nella scheda
NAMES = {"D1": "D1 M1 + errori incondizionati (riferimento)", "D2": "D2 M1 + errori per fascia",
         "D3": "D3 M1 + errori per fascia × direzione", "D4": "D4 Regressione quantilica HAR",
         "D5": "D5 M1 + GARCH(1,1) t asimmetrica", "D6": "D6 Quantili dalle 30 analogie", "D1b": "D1b M7 + errori incondizionati (controllo)",
         "E1": "E1 Logistica HAR", "RIF": "Riferimento"}


# ------------------------------------------------------------------ eventi

def events_for(h: int) -> list[str]:
    return [f"VIX>{int(c)}" for c in LEVEL_EVENTS] + (["rialzo>20%"] if h == 1 else [])


def event_threshold(event: str, x0: float) -> float:
    """Soglia in log: livello fisso oppure, per il rialzo, valore attuale + ln(1,2)."""
    return x0 + np.log(1 + RISE) if event.startswith("rialzo") else np.log(float(event.split(">")[1]))


def reference_prob(hist: np.ndarray, h: int, event: str) -> float:
    """Catena di Markov della Fase 0 per le soglie di livello; frequenza storica espandibile per il rialzo."""
    if event.startswith("rialzo"):
        return float((np.diff(hist) > np.log(1 + RISE)).mean())
    return B.prob_markov(np.exp(hist), h, float(event.split(">")[1]))


def e1(hist: np.ndarray, h: int, event: str) -> float:
    """E1. Logistica (ridge leggero) dell'evento sulle variabili dell'HAR, stimata sui casi con esito noto."""
    F = B.har_features(hist)[:, 1:]                       # x_i, media4, media12 per i = 11..t
    i = np.arange(11, len(hist) - h)                      # esito x_{i+h} noto
    X, x0 = F[: len(i)], hist[i]
    yb = (hist[i + h] > np.array([event_threshold(event, v) for v in x0])).astype(float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    z_new = np.concatenate([[1.0], (F[-1] - mu) / sd])
    pen = np.diag([0.0] + [LOGIT_RIDGE] * (Z.shape[1] - 1))
    b = np.zeros(Z.shape[1])
    b[0] = np.log((yb.mean() + 1e-3) / (1 - yb.mean() + 1e-3))
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30)))
        g = Z.T @ (yb - p) - pen @ b
        H = (Z * (p * (1 - p))[:, None]).T @ Z + pen
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-9:
            break
    return float(1 / (1 + np.exp(-np.clip(z_new @ b, -30, 30))))


# ------------------------------------------------------------------ motore

def centers(x: np.ndarray, fn, h: int) -> np.ndarray:
    """Previsioni puntuali a tutte le origini O0..len(x)-1 (ognuna usa solo x[: i + 1])."""
    return np.array([fn(x[: i + 1].copy(), h) for i in range(O0, len(x))])


def contesto(x: np.ndarray, t: int, h: int, c1: np.ndarray, c7: np.ndarray | None) -> Contesto:
    """Ritaglia cio' che e' disponibile all'origine t. I centri oltre t non vengono passati."""
    return Contesto(x[: t + 1].copy(), h, c1[: t - O0 + 1].copy(), None if c7 is None else c7[: t - O0 + 1].copy())


def density_walk(x: np.ndarray, name: str, h: int, c1: np.ndarray, c7: np.ndarray | None = None):
    """Walk-forward di un modello di densita'. Restituisce (origini, lista di distribuzioni)."""
    origins = np.arange(first_origin(h), len(x) - h)
    return origins, [DENSITY[name](contesto(x, t, h, c1, c7)) for t in origins]
