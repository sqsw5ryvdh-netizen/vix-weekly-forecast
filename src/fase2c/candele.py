"""Fase 2c, Parte B: costruzione delle candele previste. Elenco CHIUSO: K0, K1, K2, K3.

K0  candele della Fase 2b (riferimento): corpo = spostamento della mediana D4.
K1  concatenata: apertura = chiusura della candela precedente x rapporto di apertura (la prima: ultima chiusura reale);
    chiusura = apertura x exp(segno x ampiezza); ombre dalle regressioni dedicate.
K2  ancorata al percorso centrale: come K1, ma l'apertura della settimana h parte dalla mediana D4 della settimana h-1.
K3  caso simile mediano: tra le 30 situazioni passate piu' simili (stato di M6, esito noto), quella con variazione a
    4 settimane piu' vicina alla mediana; le sue 4 candele reali, riportate al livello attuale in proporzione.
Ogni candela e' coerente per costruzione: minimo <= apertura, chiusura <= massimo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..fase1 import modelli as M
from ..fase2b import modello as MO
from . import componenti as CP

COSTRUZIONI = ("K0", "K1", "K2", "K3")
NOMI = {"K0": "K0 Fase 2b (mediane)", "K1": "K1 concatenata", "K2": "K2 ancorata alla mediana", "K3": "K3 caso simile mediano"}
K_SIMILI = M.K_ANALOG          # 30


def _candela(apertura: float, p_su: float, corpo: float, ombra_sup: float, ombra_inf: float) -> dict:
    su = p_su >= 0.5                                  # a parita' (0,5) si disegna il rialzo
    chiusura = apertura * float(np.exp(corpo if su else -corpo))
    return {"apertura": float(apertura), "chiusura": chiusura, "massimo": max(apertura, chiusura) * float(np.exp(ombra_sup)),
            "minimo": min(apertura, chiusura) * float(np.exp(-ombra_inf))}


def k1(chiusura0: float, comp: dict[int, dict], pdir: dict[int, float]) -> list[dict]:
    out, prev = [], float(chiusura0)
    for h in sorted(comp):
        c = comp[h]
        out.append(_candela(prev * c["rapporto_apertura"], pdir[h], c["corpo"], c["ombra_sup"], c["ombra_inf"]))
        prev = out[-1]["chiusura"]
    return out


def k2(chiusura0: float, mediane: dict[int, float], comp: dict[int, dict], pdir: dict[int, float]) -> list[dict]:
    out = []
    for h in sorted(comp):
        c = comp[h]
        base = chiusura0 if h == 1 else mediane[h - 1]
        out.append(_candela(base * c["rapporto_apertura"], pdir[h], c["corpo"], c["ombra_sup"], c["ombra_inf"]))
    return out


def caso_mediano(S: CP.Serie, t: int) -> int | None:
    """Indice del caso simile con variazione a 4 settimane piu' vicina alla mediana dei 30 piu' simili.

    Candidati: esito a 4 settimane gia' noto (i + 4 <= t) e nessuna candela incoerente nelle 4 settimane successive.
    """
    x = np.log(S.c[: t + 1])
    St = M.analog_states(x)                                # riga j <-> indice 12 + j
    Z = (St - St.mean(axis=0)) / St.std(axis=0)
    n_cand = len(St) - 4
    if n_cand < 1:
        return None
    idx = np.arange(12, 12 + n_cand)
    bad = S.bad[: t + 1]
    valido = np.array([not bad[i + 1:i + 5].any() for i in idx])
    dist = np.sqrt(((Z[:n_cand] - Z[-1]) ** 2).sum(axis=1))
    ordine = [int(idx[j]) for j in np.argsort(dist, kind="stable") if valido[j]][:K_SIMILI]
    if not ordine:
        return None
    var4 = np.array([x[i + 4] - x[i] for i in ordine])
    return ordine[int(np.argmin(np.abs(var4 - np.median(var4))))]      # a parita', il piu' simile (argmin prende il primo)


def k3(S: CP.Serie, t: int) -> tuple[list[dict], int | None]:
    i = caso_mediano(S, t)
    if i is None:
        return [], None
    f = S.c[t] / S.c[i]
    return [{"apertura": float(S.o[k] * f), "chiusura": float(S.c[k] * f), "massimo": float(S.h[k] * f), "minimo": float(S.l[k] * f)}
            for k in range(i + 1, i + 5)], i


def prevedi_candele(df: pd.DataFrame, t: int, costruzione: str, modello_dir: dict[int, str]) -> tuple[dict, list[dict]]:
    """Previsione completa all'origine t: (previsione Fase 2b con mediane, fasce e probabilita'; candele della costruzione scelta).

    Usa solo df.iloc[: t + 1]. Accanto a ogni candela: probabilita' della direzione dal modello di direzione scelto.
    """
    prev = MO.prevedi(df, t)
    S = CP.serie(df.iloc[: t + 1])
    comp = {h: CP.componenti(S, t, h) for h in CP.ORIZZONTI}
    pdir = {h: comp[h][f"p_{modello_dir[h]}"] for h in CP.ORIZZONTI}
    med = {s["h"]: s["chiusura"] for s in prev["settimane"]}
    caso = None
    if costruzione == "K0":
        cand = [{k: s[k] for k in ("apertura", "chiusura", "massimo", "minimo")} for s in prev["settimane"]]
    elif costruzione == "K1":
        cand = k1(prev["chiusura_origine"], comp, pdir)
    elif costruzione == "K2":
        cand = k2(prev["chiusura_origine"], med, comp, pdir)
    elif costruzione == "K3":
        cand, caso = k3(S, t)
    else:
        raise ValueError(costruzione)
    out = []
    for h, c in zip(CP.ORIZZONTI, cand):
        su = c["chiusura"] >= c["apertura"]
        out.append(c | {"h": h, "costruzione": costruzione, "p_rialzo": pdir[h], "modello_direzione": modello_dir[h],
                        "direzione": "rialzo" if su else "ribasso", "prob_direzione": pdir[h] if su else 1 - pdir[h],
                        "mediana_d4": med[h], "rapporto_apertura": comp[h]["rapporto_apertura"]})
    if caso is not None:
        for c in out:
            c["caso_simile_data"] = str(pd.Timestamp(S.date[caso]).date())
    return prev, out
