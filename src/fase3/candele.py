"""Fase 3, Parte 0: nuova definizione delle candele previste (costruzione K2 confermata).

Decisioni (prese prima di guardare i risultati di questa fase):
- costruzione K2: l'apertura della settimana h parte dalla mediana D4 della settimana h-1 (la prima: ultima chiusura reale);
- ampiezza del corpo e ombre: MEDIA condizionata (regressione ai minimi quadrati sulle stesse variabili della Fase 2c),
  non piu' mediana. Se la regressione non batte la media storica, si usa la media storica. «Batte» e' fissato qui:
  errore quadratico medio piu' basso nel walk-forward di sviluppo E test di Diebold-Mariano con p < 0,05
  (per ogni componente e per ogni orizzonte, separatamente);
- direzione del corpo: frequenza storica (L0);
- etichetta: direzione della CHIUSURA rispetto alla chiusura dell'origine (per la settimana +1 e' la chiusura della
  settimana prima), con la probabilita' del modello D1 gia' usato per le probabilita' (nessun modello nuovo);
  tra 40% e 60% l'etichetta e' «neutra».
Ogni funzione riceve la serie e un indice di origine t e usa solo le settimane 0..t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..fase1 import modelli as M
from ..fase2 import densita as D
from ..fase2b import modello as MO
from ..fase2c import componenti as CP

COSTRUZIONE = "K2m"
NOME = "K2 ancorata alla mediana, ampiezze medie"
COMPONENTI = ("corpo", "ombra_sup", "ombra_inf")
NOMI_COMPONENTI = {"corpo": "ampiezza del corpo", "ombra_sup": "ombra superiore", "ombra_inf": "ombra inferiore"}
ORIZZONTI = CP.ORIZZONTI
NEUTRA = (0.40, 0.60)             # tra 40% e 60% l'etichetta della chiusura e' «neutra»
P_BATTE = 0.05                    # soglia del test DM per dire che la regressione batte la media storica


def media_condizionata(X: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> float:
    """Regressione ai minimi quadrati (variabili standardizzate); la previsione non puo' essere negativa."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta = np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(max(0.0, np.concatenate([[1.0], (x_new - mu) / sd]) @ beta))


def componenti_medie(S: CP.Serie, t: int, h: int) -> dict:
    """Componenti della candela della settimana t + h stimate sulla storia fino a t: per corpo e ombre sia la
    regressione sulla media (`*_reg`) sia la media storica (`*_media`). Stesso campione di stima della Fase 2c."""
    G = CP.grandezze(S, t)
    bad = G["bad"]
    i = np.arange(11, t - h + 1)                       # origini passate con esito noto (i + h <= t)
    i = i[~bad[i] & ~bad[i + h]]
    Xa = np.column_stack([G["har"][i - 11][:, 1:], G["extra"][i]])
    xe_new = G["extra"][t].copy()
    if bad[t]:                                         # origine con apertura incoerente: ombre dell'origine sostituite dalla mediana storica
        xe_new[2:4] = np.median(G["extra"][i][:, 2:4], axis=0)
    xa_new = np.concatenate([G["har"][-1][1:], xe_new])
    j = np.arange(1, t + 1)
    j = j[~bad[j]]
    out = {"rapporto_apertura": float(np.median(G["gap"][j])), "p_L0": float(G["su"][j].mean()), "n_stima": int(len(i))}
    for nome in COMPONENTI:
        y = G[nome][i + h]
        out[f"{nome}_reg"] = media_condizionata(Xa, y, xa_new)
        out[f"{nome}_media"] = float(y.mean())
    return out


def scelte_predefinite() -> dict[str, dict[int, str]]:
    return {c: {h: "media" for h in ORIZZONTI} for c in COMPONENTI}


def candela(base: float, comp: dict, scelte: dict, h: int) -> dict:
    """Candela della settimana h: apertura = base x rapporto di apertura; direzione L0; ampiezze secondo `scelte`."""
    v = {c: comp[f"{c}_{'reg' if scelte[c][h] == 'regressione' else 'media'}"] for c in COMPONENTI}
    su = comp["p_L0"] >= 0.5
    ap = float(base) * comp["rapporto_apertura"]
    ch = ap * float(np.exp(v["corpo"] if su else -v["corpo"]))
    return {"apertura": ap, "chiusura": ch, "massimo": max(ap, ch) * float(np.exp(v["ombra_sup"])), "minimo": min(ap, ch) * float(np.exp(-v["ombra_inf"]))}


def etichetta(p_su: float) -> tuple[str, str]:
    """(direzione, testo) per la chiusura rispetto all'origine: «neutra» se la probabilita' e' tra 40% e 60%."""
    if p_su >= NEUTRA[1]:
        return "rialzo", f"rialzo {round(100 * p_su)}%"
    if p_su <= NEUTRA[0]:
        return "ribasso", f"ribasso {round(100 * (1 - p_su))}%"
    return "neutra", "neutra"


def prob_chiusura_su(x: np.ndarray, t: int, h: int, c1: np.ndarray | None = None) -> float:
    """P(chiusura a t + h > chiusura a t) dal modello D1 (M1 + errori passati con esito noto). Usa solo x[: t + 1]."""
    if c1 is None:
        c1 = D.centers(x[: t + 1], M.forecast_har, h)
    return float(D.d1(D.contesto(x, t, h, c1, None)).prob_above(float(x[t])))


def prevedi_candele(df: pd.DataFrame, t: int, scelte: dict | None = None) -> tuple[dict, list[dict]]:
    """Previsione completa all'origine t: (previsione Fase 2b con mediane, fasce e probabilita'; candele K2 con ampiezze medie).

    Usa solo df.iloc[: t + 1]. `scelte[componente][h]` vale «regressione» oppure «media» (misurato da run_phase3).
    """
    scelte = scelte or scelte_predefinite()
    prev = MO.prevedi(df, t)
    S = CP.serie(df.iloc[: t + 1])
    x = np.log(S.c)
    med = {s["h"]: s["chiusura"] for s in prev["settimane"]}
    out = []
    for h in ORIZZONTI:
        comp = componenti_medie(S, t, h)
        c = candela(prev["chiusura_origine"] if h == 1 else med[h - 1], comp, scelte, h)
        su = c["chiusura"] >= c["apertura"]
        p_su = prob_chiusura_su(x, t, h)
        d, testo = etichetta(p_su)
        out.append(c | {"h": h, "costruzione": COSTRUZIONE, "modello_direzione": "L0", "p_rialzo": comp["p_L0"],
                        "direzione": "rialzo" if su else "ribasso", "prob_direzione": comp["p_L0"] if su else 1 - comp["p_L0"],
                        "p_chiusura_su": p_su, "direzione_chiusura": d, "etichetta": testo,
                        "ampiezze": {k: scelte[k][h] for k in COMPONENTI}, "mediana_d4": med[h], "rapporto_apertura": comp["rapporto_apertura"]})
    return prev, out
