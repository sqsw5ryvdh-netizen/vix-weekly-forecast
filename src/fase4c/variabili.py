"""Fase 4c: variabili degli insiemi S1-S5 (elenco CHIUSO) per massimo e minimo della settimana +1, e variabili
dell'S&P 500 per la previsione incrociata. Una riga per settimana: la riga t usa solo le settimane 0..t del VIX e i
valori giornalieri con data non successiva all'ultima seduta della settimana t.

S1  candela del VIX: escursione ln(high/low) dell'ultima settimana, sua media su 4 settimane, posizione della chiusura
    nell'escursione.
S2  S1 + VVIX: ln(VVIX) e ln(VVIX) meno la sua media a 52 settimane.
S3  S1 + ampiezza dell'S&P 500 dalle chiusure giornaliere: volatilita' realizzata sulle ultime 5 e 20 sedute
    (annualizzata), escursione dell'ultima settimana ln(chiusura massima / chiusura minima), rendimento dell'ultima settimana.
S4  S1 + VIX3M / VIX - 1.
S5  S1 + tutte le variabili di S2, S3 e S4.

Regola di allineamento: quella della Fase 3 (ultimo valore dentro la settimana; se nella settimana non ce n'e'
nessuno la variabile e' mancante; nessun valore portato avanti). Precisazioni fissate prima della valutazione:
- media a 52 settimane del VVIX = media dei ln(VVIX) giornalieri negli ultimi 364 giorni fino all'ultima seduta della
  settimana, con almeno 200 valori;
- volatilita' realizzata = 100 x radice(252 x media dei quadrati dei rendimenti log giornalieri) sulle ultime 5 e 20
  sedute fino all'ultima seduta della settimana;
- escursione settimanale dell'S&P 500 = ln(massimo / minimo) delle chiusure giornaliere dentro la settimana;
- rendimento settimanale = ln(ultima chiusura della settimana / ultima chiusura della settimana precedente).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..fase3 import variabili as V3
from ..fase4 import versioni as V4

S1 = ("s1_escursione", "s1_escursione_4", "s1_posizione")
S2 = ("s2_ln_vvix", "s2_ln_vvix_meno_media52")
S3 = ("s3_volreal5", "s3_volreal20", "s3_escursione_spx", "s3_rend_spx")
S4 = ("s4_vix3m_su_vix",)
INSIEMI = {"S0": (), "S1": S1, "S2": S1 + S2, "S3": S1 + S3, "S4": S1 + S4, "S5": S1 + S2 + S3 + S4}
SERIE = {"S0": (), "S1": (), "S2": ("vvix",), "S3": ("spx",), "S4": ("vix", "vix3m"), "S5": ("vvix", "spx", "vix", "vix3m")}
DESCRIZIONI = {"S0": "S0 variabili di M1", "S1": "S1 = S0 + candela del VIX", "S2": "S2 = S1 + VVIX", "S3": "S3 = S1 + ampiezza dell'S&P 500", "S4": "S4 = S1 + VIX3M/VIX",
               "S5": "S5 = S1 + VVIX, S&P 500 e VIX3M/VIX"}
INCROCIATA = ("b_rend4", "b_escursione_spx_4", "b_escursione_spx_12")           # servono solo alla Parte B (previsione dell'S&P 500)
TUTTE = S1 + S2 + S3 + S4 + INCROCIATA
MIN_VVIX_52 = 200


def candela_vix(high, low, close) -> dict[str, np.ndarray]:
    high, low, close = (np.asarray(v, dtype=float) for v in (high, low, close))
    esc = np.log(high / low)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = np.where(high > low, (close - low) / np.where(high > low, high - low, 1.0), 0.5)
    return {"s1_escursione": esc, "s1_escursione_4": pd.Series(esc).rolling(4).mean().to_numpy(), "s1_posizione": pos}


def _vvix(vv: pd.Series, date: np.ndarray) -> dict[str, np.ndarray]:
    k, ok = V3._ultimo_nella_settimana(vv, date)
    lv = np.log(vv.to_numpy())
    ln = V3._campiona(lv, k, ok)
    d = vv.index.to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(lv)])
    media = np.full(len(date), np.nan)
    for t in np.flatnonzero(ok):
        a = int(np.searchsorted(d, date[t] - np.timedelta64(364, "D"), side="right"))
        if k[t] + 1 - a >= MIN_VVIX_52:
            media[t] = (cs[k[t] + 1] - cs[a]) / (k[t] + 1 - a)
    return {"s2_ln_vvix": ln, "s2_ln_vvix_meno_media52": ln - media}


def _spx(s: pd.Series, date: np.ndarray) -> dict[str, np.ndarray]:
    n = len(date)
    k, ok = V3._ultimo_nella_settimana(s, date)
    ls = np.log(s.to_numpy()) if len(s) else np.array([])
    d = s.index.to_numpy()
    r2 = np.concatenate([[0.0], np.cumsum(np.concatenate([[0.0], np.diff(ls) ** 2]))]) if len(s) else np.array([0.0])   # r2[j + 1] = somma dei quadrati fino alla seduta j
    inizio = np.concatenate([[date[0] - np.timedelta64(7, "D")], date[:-1]])
    w = V3._campiona(ls, k, ok)                                    # log dell'ultima chiusura della settimana
    v5, v20, esc = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for t in np.flatnonzero(ok):
        j = k[t]
        if j >= 5:
            v5[t] = 100 * np.sqrt(252 * (r2[j + 1] - r2[j - 4]) / 5)
        if j >= 20:
            v20[t] = 100 * np.sqrt(252 * (r2[j + 1] - r2[j - 19]) / 20)
        a = int(np.searchsorted(d, inizio[t], side="right"))       # prima seduta della settimana
        esc[t] = ls[a:j + 1].max() - ls[a:j + 1].min()
    rend, rend4 = np.full(n, np.nan), np.full(n, np.nan)
    rend[1:], rend4[4:] = w[1:] - w[:-1], w[4:] - w[:-4]
    e = pd.Series(esc)
    return {"s3_volreal5": v5, "s3_volreal20": v20, "s3_escursione_spx": esc, "s3_rend_spx": rend, "b_rend4": rend4,
            "b_escursione_spx_4": e.rolling(4).mean().to_numpy(), "b_escursione_spx_12": e.rolling(12).mean().to_numpy()}


def costruisci(df: pd.DataFrame, giorn: pd.DataFrame | None) -> pd.DataFrame:
    """Variabili settimanali di tutti gli insiemi. `df`: VIX settimanale (date, high, low, close); `giorn`: dati
    giornalieri con indice = data (colonne tra vix, vix3m, vvix, spx; quelle assenti danno variabili mancanti)."""
    date = np.asarray(pd.to_datetime(df["date"]), dtype="datetime64[ns]")
    out = candela_vix(df["high"], df["low"], df["close"])
    vuoto = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    col = lambda c: giorn[c].dropna() if giorn is not None and c in giorn.columns else vuoto
    out |= _vvix(col("vvix"), date) | _spx(col("spx"), date)
    if giorn is not None and {"vix", "vix3m"} <= set(giorn.columns):
        out["s4_vix3m_su_vix"] = V4.x1_settimanale(date, giorn)
    else:
        out["s4_vix3m_su_vix"] = np.full(len(date), np.nan)
    return pd.DataFrame({c: out[c] for c in TUTTE})
