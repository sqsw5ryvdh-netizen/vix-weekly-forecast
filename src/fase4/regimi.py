"""Fase 4, Parte B — N1: modello autoregressivo a 2 regimi markoviani sul log-VIX settimanale.

Modello (Hamilton):  x_t - mu_{S_t} = phi (x_{t-1} - mu_{S_{t-1}}) + sigma_{S_t} eps_t,  S_t catena di Markov a 2 stati:
media e varianza diverse per regime, phi comune. Stima di massima verosimiglianza con statsmodels
(MarkovAutoregression, impostazioni predefinite). Scelte fissate prima della valutazione:
- ristima ogni 26 settimane a partire dalla prima origine (indice 259), sulla storia fino a quell'origine;
- tra una ristima e l'altra i parametri restano fermi e le probabilita' filtrate di regime sono ricalcolate a ogni
  origine sulla storia fino all'origine;
- una stima e' valida se l'ottimizzazione converge e i parametri sono ammissibili (varianze > 0, probabilita' in
  (0, 1)); se non lo e' si usa l'ultima stima valida; se non ne esiste ancora nessuna si usa un AR(1) a un solo
  regime; i casi vengono contati;
- previsione a h passi dalle probabilita' filtrate, esatta: somma sui percorsi di regime S_t ... S_{t+h};
  T1 = media della distribuzione prevista, T2 = probabilita' che x_{t+h} superi x_t (mistura di normali).
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
from scipy import stats

from .. import config as C

PASSO_STIMA = 26
PRIMA_ORIGINE = C.MIN_TRAIN_WEEKS - 1


def stima(x: np.ndarray) -> dict | None:
    """Stima sulla storia x. Restituisce None se la stima non e' valida."""
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mod = MarkovAutoregression(np.asarray(x, dtype=float), k_regimes=2, order=1, trend="c", switching_ar=False, switching_variance=True)
            res = mod.fit(disp=False)
        except Exception:
            return None
    p = dict(zip(mod.param_names, np.asarray(res.params, dtype=float)))
    par = {"params": np.asarray(res.params, dtype=float), "P": np.array([[p["p[0->0]"], 1 - p["p[0->0]"]], [p["p[1->0]"], 1 - p["p[1->0]"]]]),
           "mu": np.array([p["const[0]"], p["const[1]"]]), "s2": np.array([p["sigma2[0]"], p["sigma2[1]"]]), "phi": float(p["ar.L1"]), "n": int(len(x))}
    ok = (bool(res.mle_retvals.get("converged", False)) and np.isfinite(par["params"]).all() and (par["s2"] > 1e-8).all()
          and ((par["P"] > 1e-6) & (par["P"] < 1 - 1e-6)).all() and abs(par["phi"]) < 1)
    return par if ok else None


def filtrate(x: np.ndarray, par: dict) -> np.ndarray:
    """P(S_t = j | x_0..x_t) con i parametri dati (nessuna stima)."""
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = MarkovAutoregression(np.asarray(x, dtype=float), k_regimes=2, order=1, trend="c", switching_ar=False, switching_variance=True)
        return np.asarray(mod.filter(par["params"]).filtered_marginal_probabilities[-1], dtype=float)


def distribuzione(x_t: float, pi: np.ndarray, par: dict, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mistura di normali di x_{t+h} dato lo stato filtrato: (pesi, medie, deviazioni standard), una per percorso di regime."""
    P, mu, s2, phi = par["P"], par["mu"], par["s2"], par["phi"]
    w, m, s = [], [], []
    for path in itertools.product((0, 1), repeat=h + 1):
        prob = pi[path[0]] * np.prod([P[path[k - 1], path[k]] for k in range(1, h + 1)])
        w.append(prob)
        m.append(mu[path[h]] + phi ** h * (x_t - mu[path[0]]))
        s.append(np.sqrt(sum(phi ** (2 * (h - k)) * s2[path[k]] for k in range(1, h + 1))))
    return np.array(w), np.array(m), np.array(s)


def previsione(x: np.ndarray, par: dict, h: int, pi: np.ndarray | None = None) -> tuple[float, float]:
    """(media prevista di x_{t+h}, probabilita' che x_{t+h} > x_t) all'origine t = len(x) - 1."""
    pi = filtrate(x, par) if pi is None else pi
    w, m, s = distribuzione(float(x[-1]), pi, par, h)
    return float(w @ m), float(w @ stats.norm.sf((x[-1] - m) / s))


def ar1_distribuzione(x: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ripiego quando non esiste ancora una stima valida: AR(1) a un solo regime, errori normali (mistura con una sola componente)."""
    a, phi = np.linalg.lstsq(np.column_stack([np.ones(len(x) - 1), x[:-1]]), x[1:], rcond=None)[0]
    s2 = float(np.var(x[1:] - a - phi * x[:-1]))
    f = x[-1]
    for _ in range(h):
        f = a + phi * f
    return np.array([1.0]), np.array([float(f)]), np.array([float(np.sqrt(s2 * sum(phi ** (2 * k) for k in range(h))))])


def ar1(x: np.ndarray, h: int) -> tuple[float, float]:
    w, m, s = ar1_distribuzione(x, h)
    return float(w @ m), float(w @ stats.norm.sf((x[-1] - m) / s))


def parametri_all_origine(x: np.ndarray) -> dict | None:
    """Parametri in uso all'origine t = len(x) - 1: l'ultima stima valida tra quelle delle date di ristima <= t
    (ognuna fatta sulla storia fino alla propria data). None se non ne esiste nessuna. Stessa regola di `walk`."""
    ts = origine_di_stima(len(x) - 1)
    while ts >= PRIMA_ORIGINE:
        par = stima(x[: ts + 1].copy())
        if par is not None:
            return par
        ts -= PASSO_STIMA
    return None


def mistura_all_origine(x: np.ndarray, h: int, par: dict | None, pi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distribuzione prevista di x_{t+h} all'origine t = len(x) - 1 con i parametri in uso (AR(1) se non ce ne sono)."""
    if par is None:
        return ar1_distribuzione(x, h)
    return distribuzione(float(x[-1]), filtrate(x, par) if pi is None else pi, par, h)


def origine_di_stima(t: int) -> int:
    """Data di ristima in vigore all'origine t (indice di settimana): la piu' recente tra 259, 285, 311, ... <= t."""
    return PRIMA_ORIGINE + PASSO_STIMA * ((t - PRIMA_ORIGINE) // PASSO_STIMA)


def walk(x: np.ndarray, orizzonti=(1, 4)) -> dict:
    """Walk-forward. Per ogni origine t >= 259: parametri dell'ultima ristima valida (fatta su x[: ts + 1], ts <= t),
    probabilita' filtrate su x[: t + 1], previsioni per ogni h. Restituisce previsioni, stime e diagnostica."""
    n = len(x)
    stime, valida = {}, None
    out = {h: {"origini": [], "media": [], "p_su": []} for h in orizzonti}
    prob, righe, stato = [], [], {}
    for t in range(PRIMA_ORIGINE, n - min(orizzonti)):
        ts = origine_di_stima(t)
        if ts not in stime:
            stime[ts] = stima(x[: ts + 1].copy())
            if stime[ts] is not None:
                valida = stime[ts]
            righe.append({"idx_stima": ts, "valida": stime[ts] is not None, "usata_idx": None if valida is None else valida["n"] - 1})
        xs = x[: t + 1].copy()
        pi = None if valida is None else filtrate(xs, valida)
        alto = None if valida is None else int(np.argmax(valida["mu"]))
        stato[t] = (valida, pi)
        prob.append({"idx_origine": t, "idx_stima_usata": None if valida is None else valida["n"] - 1, "p_regime_alto": np.nan if pi is None else float(pi[alto])})
        for h in orizzonti:
            if t + h >= n:
                continue
            f, p = ar1(xs, h) if valida is None else previsione(xs, valida, h, pi)
            out[h]["origini"].append(t)
            out[h]["media"].append(f)
            out[h]["p_su"].append(p)
    return {"previsioni": {h: {k: np.array(v) for k, v in d.items()} for h, d in out.items()}, "stime": stime, "ristime": righe, "probabilita": prob, "stato": stato}


def descrizione(par: dict) -> dict:
    """Regimi ordinati per livello medio: «basso» e «alto»."""
    b, a = int(np.argmin(par["mu"])), int(np.argmax(par["mu"]))
    return {"livello_basso": float(np.exp(par["mu"][b])), "livello_alto": float(np.exp(par["mu"][a])), "sigma_basso": float(np.sqrt(par["s2"][b])),
            "sigma_alto": float(np.sqrt(par["s2"][a])), "phi": par["phi"], "p_resta_basso": float(par["P"][b, b]), "p_resta_alto": float(par["P"][a, a]),
            "durata_media_basso": float(1 / (1 - par["P"][b, b])), "durata_media_alto": float(1 / (1 - par["P"][a, a]))}
