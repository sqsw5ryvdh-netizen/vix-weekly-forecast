"""Fase 3: variabili dei blocchi (elenco CHIUSO). Una riga per settimana, calcolata con i soli valori giornalieri
disponibili entro la chiusura dell'ultima seduta di quella settimana.

Regola di allineamento, fissata prima della valutazione:
- la «settimana t» va dal giorno dopo la data settimanale precedente fino alla data settimanale t (ultima seduta);
- di ogni serie si prende l'ultimo valore dentro la settimana t; per i rapporti tra due serie, l'ultima seduta della
  settimana in cui TUTTE le serie del rapporto hanno un valore (numeratore e denominatore dello stesso giorno);
- se nella settimana non c'e' nessun valore, la variabile e' mancante: quell'origine non entra ne' nella stima ne'
  nella valutazione del blocco. Nessun valore viene portato avanti da una settimana all'altra.
Scelte dove il testo della consegna lasciava margine (dichiarate nel report):
- variazioni = differenze di logaritmo; «massimo delle ultime 10 sedute» = massimo delle chiusure giornaliere;
- «massima variazione giornaliera delle ultime 5 sedute» = la piu' alta delle 5 variazioni con il loro segno;
- volatilita' realizzata = 100 x radice(252 x media dei quadrati dei rendimenti giornalieri delle ultime 4 settimane),
  con almeno 15 rendimenti; massimo a 52 settimane = massimo delle chiusure giornaliere negli ultimi 364 giorni.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BLOCCHI: dict[str, tuple[str, ...]] = {
    "G1": ("g_var1", "g_var3", "g_sd5", "g_sd20", "g_chiusura_su_media", "g_sedute_dal_max10", "g_max_var5"),
    "X1": ("x1_vix3m_su_vix",),
    "X2": ("x2_vx1_su_vix", "x2_vx2_su_vx1"),
    "X3": ("x3_log_vvix",),
    "X4": ("x4_rend1", "x4_rend4", "x4_volreal4", "x4_vix_meno_volreal", "x4_dist_max52"),
}
DESCRIZIONI = {
    "g_var1": "VIX: variazione (log) dell'ultima seduta", "g_var3": "VIX: variazione (log) delle ultime 3 sedute",
    "g_sd5": "VIX: deviazione standard delle variazioni giornaliere, ultime 5 sedute", "g_sd20": "VIX: deviazione standard delle variazioni giornaliere, ultime 20 sedute",
    "g_chiusura_su_media": "VIX: chiusura settimanale rispetto alla media delle chiusure giornaliere della settimana (log)",
    "g_sedute_dal_max10": "VIX: sedute trascorse dal massimo delle ultime 10 sedute", "g_max_var5": "VIX: massima variazione giornaliera (log) delle ultime 5 sedute",
    "x1_vix3m_su_vix": "VIX3M / VIX − 1", "x2_vx1_su_vix": "VX1 / VIX − 1", "x2_vx2_su_vx1": "VX2 / VX1 − 1", "x3_log_vvix": "log(VVIX)",
    "x4_rend1": "S&P 500: rendimento (log) a 1 settimana", "x4_rend4": "S&P 500: rendimento (log) a 4 settimane",
    "x4_volreal4": "S&P 500: volatilità realizzata a 4 settimane, annualizzata (%)", "x4_vix_meno_volreal": "VIX − volatilità realizzata a 4 settimane",
    "x4_dist_max52": "S&P 500: distanza (log) dal massimo a 52 settimane",
}
SERIE_PER_BLOCCO = {"G1": ("vix",), "X1": ("vix3m", "vix"), "X2": ("vx1", "vx2", "vix"), "X3": ("vvix",), "X4": ("spx",)}
TUTTE = tuple(v for b in BLOCCHI.values() for v in b)
MIN_RENDIMENTI = 15


def _ultimo_nella_settimana(s: pd.Series, date: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per ogni settimana t: (posizione in `s` dell'ultimo valore con data in (data[t-1], data[t]], valido?)."""
    d = s.index.to_numpy()
    if len(d) == 0:                                    # serie non ancora iniziata
        return np.full(len(date), -1), np.zeros(len(date), bool)
    k = np.searchsorted(d, date, side="right") - 1
    inizio = np.concatenate([[date[0] - np.timedelta64(7, "D")], date[:-1]])
    ok = (k >= 0) & (d[np.maximum(k, 0)] > inizio)
    return k, ok


def _campiona(valori: np.ndarray, k: np.ndarray, ok: np.ndarray) -> np.ndarray:
    if len(valori) == 0:
        return np.full(len(k), np.nan)
    return np.where(ok, valori[np.maximum(k, 0)], np.nan)


def _g1(v: pd.Series, date: np.ndarray, close: np.ndarray) -> dict[str, np.ndarray]:
    lv = np.log(v)
    dv = lv.diff()
    k, ok = _ultimo_nella_settimana(v, date)
    dal_max = pd.Series(v.to_numpy()).rolling(10).apply(lambda w: float(np.argmax(w[::-1])), raw=True)   # 0 = oggi; a parita', il massimo piu' recente
    out = {"g_var1": _campiona(dv.to_numpy(), k, ok), "g_var3": _campiona((lv - lv.shift(3)).to_numpy(), k, ok),
           "g_sd5": _campiona(dv.rolling(5).std().to_numpy(), k, ok), "g_sd20": _campiona(dv.rolling(20).std().to_numpy(), k, ok),
           "g_sedute_dal_max10": _campiona(dal_max.to_numpy(), k, ok), "g_max_var5": _campiona(dv.rolling(5).max().to_numpy(), k, ok)}
    d, val = v.index.to_numpy(), v.to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(val)])
    inizio = np.concatenate([[date[0] - np.timedelta64(7, "D")], date[:-1]])
    a = np.searchsorted(d, inizio, side="right")                 # prima seduta della settimana
    n_sed = np.maximum(k + 1 - a, 1)
    media = (cs[np.maximum(k, 0) + 1] - cs[np.minimum(a, np.maximum(k, 0))]) / n_sed       # media delle chiusure giornaliere della settimana
    out["g_chiusura_su_media"] = np.where(ok & (k + 1 - a >= 1), np.log(close / media), np.nan)
    return out


def _x4(s: pd.Series, date: np.ndarray, close: np.ndarray) -> dict[str, np.ndarray]:
    k, ok = _ultimo_nella_settimana(s, date)
    ls = np.log(s.to_numpy())
    w = _campiona(ls, k, ok)                                       # log S&P 500 a fine settimana
    n = len(date)
    r1, r4 = np.full(n, np.nan), np.full(n, np.nan)
    r1[1:], r4[4:] = w[1:] - w[:-1], w[4:] - w[:-4]
    d = s.index.to_numpy()
    r2 = np.concatenate([[0.0], np.cumsum(np.concatenate([[0.0], np.diff(ls) ** 2]))])     # r2[j + 1] = somma dei quadrati dei rendimenti fino alla seduta j
    vol, dist = np.full(n, np.nan), np.full(n, np.nan)
    for t in range(4, n):
        if not ok[t]:
            continue
        a = int(np.searchsorted(d, date[t - 4], side="right"))     # prima seduta dopo la data settimanale t - 4
        m = k[t] - a + 1                                           # rendimenti con data in (data[t-4], data[t]]
        if a >= 1 and m >= MIN_RENDIMENTI:
            vol[t] = 100 * np.sqrt(252 * (r2[k[t] + 1] - r2[a]) / m)
        b = int(np.searchsorted(d, date[t] - np.timedelta64(364, "D"), side="right"))
        dist[t] = ls[k[t]] - ls[b:k[t] + 1].max()
    return {"x4_rend1": r1, "x4_rend4": r4, "x4_volreal4": vol, "x4_vix_meno_volreal": close - vol, "x4_dist_max52": dist}


def costruisci(date_sett, close_sett, giorn: pd.DataFrame) -> pd.DataFrame:
    """Variabili settimanali di tutti i blocchi. `giorn`: dati giornalieri con indice = data di disponibilita'.

    Ogni riga t dipende solo dai dati giornalieri con data <= date_sett[t] e dalle chiusure settimanali <= t.
    """
    date = np.asarray(pd.to_datetime(date_sett), dtype="datetime64[ns]")
    close = np.asarray(close_sett, dtype=float)
    out: dict[str, np.ndarray] = {}
    out |= _g1(giorn["vix"].dropna(), date, close)
    j = giorn[["vix3m", "vix"]].dropna()
    k, ok = _ultimo_nella_settimana(j["vix"], date)
    out["x1_vix3m_su_vix"] = _campiona((j["vix3m"] / j["vix"] - 1).to_numpy(), k, ok)
    j = giorn[["vx1", "vx2", "vix"]].dropna()
    k, ok = _ultimo_nella_settimana(j["vix"], date)
    out["x2_vx1_su_vix"] = _campiona((j["vx1"] / j["vix"] - 1).to_numpy(), k, ok)
    out["x2_vx2_su_vx1"] = _campiona((j["vx2"] / j["vx1"] - 1).to_numpy(), k, ok)
    vv = giorn["vvix"].dropna()
    k, ok = _ultimo_nella_settimana(vv, date)
    out["x3_log_vvix"] = _campiona(np.log(vv.to_numpy()), k, ok)
    out |= _x4(giorn["spx"].dropna(), date, close)
    return pd.DataFrame({v: out[v] for v in TUTTE})


def inventario(V: pd.DataFrame, date_sett) -> pd.DataFrame:
    date = pd.to_datetime(pd.Series(date_sett)).reset_index(drop=True)
    rows = []
    for b, cols in BLOCCHI.items():
        for c in cols:
            ok = np.isfinite(V[c].to_numpy())
            first = int(np.argmax(ok))
            rows.append({"blocco": b, "variabile": c, "descrizione": DESCRIZIONI[c], "prima_settimana_valida": str(date[first].date()),
                         "settimane_valide": int(ok.sum()), "settimane_mancanti_dopo_l_inizio": int((~ok[first:]).sum()),
                         "media": float(V[c].mean()), "dev_std": float(V[c].std()), "minimo": float(V[c].min()), "massimo": float(V[c].max())})
    return pd.DataFrame(rows)
