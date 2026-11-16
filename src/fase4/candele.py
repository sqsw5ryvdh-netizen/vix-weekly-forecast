"""Fase 4, Parte A: varianti (elenco CHIUSO) delle ampiezze delle candele K2 — corpo e due ombre.

A0  attuale (Fase 3): media storica espandibile; regressione, stimata su tutta la storia, per il corpo a +1 e +2.
A1  media sulle ultime 260 settimane; regressione stimata sulle ultime 520.
A2  media sulle ultime 104 settimane; regressione stimata sulle ultime 520.

Che cosa NON cambia tra le varianti (deciso prima): quale componente usa la regressione e quale la media (corpo a +1
e +2: regressione; tutto il resto: media, come misurato nella Fase 3), il rapporto di apertura (mediana espandibile),
la direzione del corpo (frequenza storica L0), la costruzione K2 e l'etichetta della chiusura.
«Ultime N settimane» = le ultime N coppie (origine i, esito i + h) con esito noto all'origine t.
Ogni funzione riceve la serie e un indice di origine t e usa solo le settimane 0..t.
"""
from __future__ import annotations

import numpy as np

from ..fase2c import componenti as CP
from ..fase3 import candele as K3

VARIANTI = {"A0": {"media": None, "regressione": None}, "A1": {"media": 260, "regressione": 520}, "A2": {"media": 104, "regressione": 520}}
DESCRIZIONI = {"A0": "A0 media espandibile, regressione su tutta la storia (Fase 3)", "A1": "A1 media su 260 settimane, regressione su 520",
               "A2": "A2 media su 104 settimane, regressione su 520"}
ASSEGNAZIONE = {"corpo": {1: "regressione", 2: "regressione", 3: "media", 4: "media"},
                "ombra_sup": {h: "media" for h in (1, 2, 3, 4)}, "ombra_inf": {h: "media" for h in (1, 2, 3, 4)}}
COMPONENTI = K3.COMPONENTI
ORIZZONTI = K3.ORIZZONTI
CAMPI = ("apertura", "massimo", "minimo", "chiusura")


def costruzione(variante: str) -> str:
    """Nome scritto nel registro: A0 e' la costruzione della Fase 3."""
    return K3.COSTRUZIONE if variante == "A0" else f"{K3.COSTRUZIONE}-{variante}"


def componenti(S: CP.Serie, t: int, h: int, variante: str) -> dict:
    """Componenti della candela della settimana t + h con le finestre della variante. Per ogni ampiezza restituisce il
    valore usato (`corpo`, `ombra_sup`, `ombra_inf`) secondo ASSEGNAZIONE."""
    v = VARIANTI[variante]
    G = CP.grandezze(S, t)
    bad = G["bad"]
    i = np.arange(11, t - h + 1)                       # origini passate con esito noto (i + h <= t)
    i = i[~bad[i] & ~bad[i + h]]
    xe_new = G["extra"][t].copy()
    if bad[t]:                                         # origine con apertura incoerente: come nelle Fasi 2c e 3
        xe_new[2:4] = np.median(G["extra"][i][:, 2:4], axis=0)
    xa_new = np.concatenate([G["har"][-1][1:], xe_new])
    i_m = i if v["media"] is None else i[-v["media"]:]
    i_r = i if v["regressione"] is None else i[-v["regressione"]:]
    j = np.arange(1, t + 1)
    j = j[~bad[j]]
    out = {"rapporto_apertura": float(np.median(G["gap"][j])), "p_L0": float(G["su"][j].mean()), "n_media": int(len(i_m)), "n_regressione": int(len(i_r))}
    for nome in COMPONENTI:
        if ASSEGNAZIONE[nome][h] == "regressione":
            out[nome] = K3.media_condizionata(np.column_stack([G["har"][i_r - 11][:, 1:], G["extra"][i_r]]), G[nome][i_r + h], xa_new)
        else:
            out[nome] = float(G[nome][i_m + h].mean())
    return out


def candela(base: float, comp: dict) -> dict:
    """Candela K2: apertura = base x rapporto di apertura; direzione L0; ampiezze gia' scelte in `comp`."""
    su = comp["p_L0"] >= 0.5
    ap = float(base) * comp["rapporto_apertura"]
    ch = ap * float(np.exp(comp["corpo"] if su else -comp["corpo"]))
    return {"apertura": ap, "chiusura": ch, "massimo": max(ap, ch) * float(np.exp(comp["ombra_sup"])), "minimo": min(ap, ch) * float(np.exp(-comp["ombra_inf"]))}


def candele(S: CP.Serie, t: int, chiusura_origine: float, centrali: dict[int, float], variante: str) -> list[dict]:
    """Le quattro candele previste all'origine t, ancorate alle chiusure centrali `centrali[h]` (K2): l'apertura della
    settimana h parte dalla chiusura centrale della settimana h - 1 (la prima: ultima chiusura reale)."""
    out = []
    for h in ORIZZONTI:
        comp = componenti(S, t, h, variante)
        c = candela(chiusura_origine if h == 1 else centrali[h - 1], comp)
        su = c["chiusura"] >= c["apertura"]
        out.append(c | {"h": h, "costruzione": costruzione(variante), "variante_ampiezze": variante, "modello_direzione": "L0", "p_rialzo": comp["p_L0"],
                        "direzione": "rialzo" if su else "ribasso", "prob_direzione": comp["p_L0"] if su else 1 - comp["p_L0"],
                        "rapporto_apertura": comp["rapporto_apertura"]})
    return out
