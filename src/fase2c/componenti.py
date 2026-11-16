"""Fase 2c, Parte A: componenti delle candele previste (apertura, direzione del corpo, ampiezza del corpo, ombre).

Tutte le funzioni ricevono la serie e un indice di origine t e usano solo le settimane 0..t.
Scelte fissate prima della valutazione:
- variabili di base: quelle dell'HAR (ultima chiusura, media a 4 e a 12 settimane, in log);
- variabili aggiuntive, dalla candela dell'ultima settimana e dalla dinamica recente: escursione ln(high/low),
  posizione della chiusura nell'escursione, ombra superiore, ombra inferiore, variazione log a 1 e a 4 settimane;
- le settimane con apertura incoerente (open fuori dall'intervallo low-high: nel file e' la sola settimana
  dell'8 novembre 2004, open = 2,58) sono escluse da ogni stima, come obiettivo e come variabile;
- logistiche con la stessa penalita' leggera di E1 (Fase 2); regressioni sulla mediana con lo stesso stimatore di D4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import baselines as B
from ..fase2 import densita as D

ORIZZONTI = (1, 2, 3, 4)
MODELLI_DIREZIONE = ("L0", "L1", "L2")
RIDGE = D.LOGIT_RIDGE
NOMI_EXTRA = ("escursione", "posizione", "ombra_sup", "ombra_inf", "dlog1", "dlog4")


@dataclass
class Serie:
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    date: np.ndarray

    @property
    def bad(self) -> np.ndarray:
        """Settimane con apertura incoerente (fuori dall'intervallo minimo-massimo): escluse dalle stime."""
        return (self.o < self.l - 1e-9) | (self.o > self.h + 1e-9)


def serie(df: pd.DataFrame) -> Serie:
    f = lambda k: df[k].to_numpy(dtype=float)
    return Serie(f("open"), f("high"), f("low"), f("close"), pd.to_datetime(df["date"]).to_numpy())


def grandezze(S: Serie, t: int) -> dict:
    """Grandezze settimanali per gli indici 0..t (nessun valore successivo a t viene letto)."""
    o, h, l, c = S.o[: t + 1], S.h[: t + 1], S.l[: t + 1], S.c[: t + 1]
    x = np.log(c)
    top, bot = np.maximum(o, c), np.minimum(o, c)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = np.where(h > l, (c - l) / np.where(h > l, h - l, 1.0), 0.5)
        g = np.concatenate([[np.nan], o[1:] / c[:-1]])
    d1 = np.concatenate([[np.nan], x[1:] - x[:-1]])
    d4 = np.concatenate([np.full(4, np.nan), x[4:] - x[:-4]])
    return {"x": x, "bad": S.bad[: t + 1], "gap": g, "su": (c > o).astype(float), "corpo": np.abs(np.log(c / o)),
            "ombra_sup": np.log(h / top), "ombra_inf": np.log(bot / l),
            "extra": np.column_stack([np.log(h / l), pos, np.log(h / top), np.log(bot / l), d1, d4]), "har": B.har_features(x)}


def logistica(X: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> float:
    """Logistica con penalita' L2 leggera sui coefficienti standardizzati (stesso schema di E1)."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    z_new = np.concatenate([[1.0], (x_new - mu) / sd])
    pen = np.diag([0.0] + [RIDGE] * X.shape[1])
    b = np.zeros(Z.shape[1])
    b[0] = np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3))
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30)))
        step = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + pen, Z.T @ (y - p) - pen @ b)
        b = b + step
        if np.abs(step).max() < 1e-9:
            break
    return float(1 / (1 + np.exp(-np.clip(z_new @ b, -30, 30))))


def mediana_condizionata(X: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> float:
    """Regressione sulla mediana (variabili standardizzate); la previsione non puo' essere negativa."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta = D.quantreg(Z, y, [0.5])[0]
    return float(max(0.0, np.concatenate([[1.0], (x_new - mu) / sd]) @ beta))


def componenti(S: Serie, t: int, h: int) -> dict:
    """Componenti della candela della settimana t + h, stimate sulla storia fino a t."""
    G = grandezze(S, t)
    bad = G["bad"]
    i = np.arange(11, t - h + 1)                       # origini passate con esito noto (i + h <= t)
    ok = ~bad[i] & ~bad[i + h]
    i = i[ok]
    Xb, Xe = G["har"][i - 11][:, 1:], G["extra"][i]
    Xa = np.column_stack([Xb, Xe])
    xb_new, xe_new = G["har"][-1][1:], G["extra"][t].copy()
    if bad[t]:                                         # origine con apertura incoerente: le ombre usano la mediana storica
        xe_new[2:4] = np.median(Xe[:, 2:4], axis=0)
    xa_new = np.concatenate([xb_new, xe_new])
    j = np.arange(1, t + 1)
    j = j[~bad[j]]
    out = {"rapporto_apertura": float(np.median(G["gap"][j])), "n_stima": int(len(i)),
           "p_L0": float(G["su"][j].mean()),
           "p_L1": logistica(Xb, G["su"][i + h], xb_new), "p_L2": logistica(Xa, G["su"][i + h], xa_new)}
    for nome in ("corpo", "ombra_sup", "ombra_inf"):
        y = G[nome][i + h]
        out[nome] = mediana_condizionata(Xa, y, xa_new)
        out[f"{nome}_rif"] = float(np.median(y))
    return out
