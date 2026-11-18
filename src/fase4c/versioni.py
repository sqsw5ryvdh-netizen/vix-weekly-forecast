"""Fase 4c: le due versioni che andranno all'esame, F1 e FS, con la prima candela comandata da massimo e minimo.

Tutto resta come nella Fase 4b (chiusura centrale, fasce, probabilita', etichette, candele +2…+4), tranne la prima
candela (settimana +1):
  massimo e minimo  previsione diretta (regressione sulla mediana, `estremi.py`) con l'insieme di variabili scelto dalla
                    regola della Parte A: per F1 tra S0 e S1 (solo storico del VIX), per FS tra tutti gli insiemi;
                    se per FS l'insieme scelto non e' calcolabile all'origine (serie mancante o piu' vecchia
                    dell'origine) si usa l'insieme di F1 e lo si segnala;
  corpo             K2 oppure K5 (regola della Parte C), tagliato dentro l'intervallo minimo-massimo;
  colore            secondo l'etichetta della chiusura, grigio se neutra.
`scelte` = scelte della Fase 4b + {"insieme_massimo_f1", "insieme_minimo_f1", "insieme_massimo_fs", "insieme_minimo_fs", "corpo"}.
`VV` = variabili settimanali (`variabili.costruisci`) per le settimane 0..t, costruite con i soli dati fino all'origine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..fase2c import componenti as CP
from ..fase4 import candele as KA
from ..fase4b import prima as PR
from ..fase4b import versioni as V4B
from . import candela as KC
from . import estremi as ES
from . import incrociata as IN
from . import variabili as V

VERSIONI = V4B.VERSIONI
NOMI = V4B.NOMI
CANDIDATI = V4B.CANDIDATI
CARTELLA = V4B.CARTELLA
NonDisponibile = V4B.NonDisponibile
INSIEMI_F1 = ("S0", "S1")                        # F1 resta solo storico del VIX
INSIEMI_FS = tuple(V.INSIEMI) + ("B3",)
SORGENTI_CONGELATI = V4B.SORGENTI_CONGELATI + ("src/fase4c/variabili.py", "src/fase4c/estremi.py", "src/fase4c/incrociata.py", "src/fase4c/candela.py", "src/fase4c/versioni.py")


def serie_richieste(insieme: str) -> tuple[str, ...]:
    return IN.SERIE if insieme == "B3" else V.SERIE[insieme]


def scelte_4b(scelte: dict) -> dict:
    """La parte delle scelte che riguarda la Fase 4b: candela K2 con il suo minimo, poi sostituita qui."""
    return {"variante": scelte["variante"], "prima_candela": "K2", "minimo": "M0", "candidato": scelte["candidato"]}


def nome_costruzione(scelte: dict) -> str:
    return f"{scelte['corpo']}+estremi/{KA.costruzione(scelte['variante'])}"


def estremo(x: np.ndarray, livelli: np.ndarray, VV: pd.DataFrame, insieme: str) -> float:
    """Massimo o minimo previsto per la settimana t + 1 con l'insieme dato (NaN se non calcolabile)."""
    if insieme == "B3":
        return IN.b3(x, livelli, VV)
    return ES.mediana_diretta(x, livelli, VV[list(V.INSIEMI[insieme])].to_numpy())


def prima_candela(df: pd.DataFrame, t: int, cand1: dict, VV: pd.DataFrame, insiemi: tuple[str, str], ripiego: tuple[str, str] | None, scelte: dict) -> dict:
    """Prima candela: estremi dagli insiemi (massimo, minimo) dati, corpo K2 o K5 tagliato dentro gli estremi."""
    S = CP.serie(df.iloc[: t + 1])
    x, VVt = np.log(S.c), VV.iloc[: t + 1]
    hi, lo = estremo(x, S.h, VVt, insiemi[0]), estremo(x, S.l, VVt, insiemi[1])
    usati, da_ripiego = insiemi, False
    if not (np.isfinite(hi) and np.isfinite(lo)) and ripiego is not None:
        hi, lo, usati, da_ripiego = estremo(x, S.h, VVt, ripiego[0]), estremo(x, S.l, VVt, ripiego[1]), ripiego, True
    if not (np.isfinite(hi) and np.isfinite(lo)):
        raise NonDisponibile("massimo e minimo della settimana +1 non calcolabili: storia insufficiente")
    ch, extra = cand1["chiusura"], {}
    if scelte["corpo"] == "K5":
        d, u = PR.mosse(x)
        ch, extra = KC.chiusura_k5(float(S.c[-1]), cand1["direzione_chiusura"], d, u, cand1["chiusura"]), {"mossa_ribasso": d, "mossa_rialzo": u}
    k = KC.componi(cand1["apertura"], ch, hi, lo)
    su = k["chiusura"] >= k["apertura"]
    return cand1 | k | extra | {"prima_candela": scelte["corpo"], "minimo_da": usati[1], "massimo_da": usati[0], "estremi_da_ripiego": da_ripiego, "colore": cand1["direzione_chiusura"],
                                "direzione": "rialzo" if su else "ribasso", "prob_direzione": cand1["p_rialzo"] if su else 1 - cand1["p_rialzo"]}


def _finisci(df, t, prev, cand, VV, insiemi, ripiego, scelte) -> tuple[dict, list[dict]]:
    nome = nome_costruzione(scelte)
    cand = [c | {"costruzione": nome} for c in cand]
    cand[0] = prima_candela(df, t, cand[0], VV, insiemi, ripiego, scelte) | {"costruzione": nome}
    return prev, cand


def prevedi_f1(df: pd.DataFrame, t: int, scelte: dict, VV: pd.DataFrame | None = None) -> tuple[dict, list[dict]]:
    """F1 all'origine t. `VV` puo' mancare: F1 usa solo variabili del VIX settimanale (S0 o S1)."""
    prev, cand = V4B.prevedi_f1(df, t, scelte_4b(scelte))
    VV = V.costruisci(df.iloc[: t + 1], None) if VV is None else VV
    return _finisci(df, t, prev, cand, VV, (scelte["insieme_massimo_f1"], scelte["insieme_minimo_f1"]), None, scelte)


def prevedi_fs(df: pd.DataFrame, t: int, z: np.ndarray | None, VV: pd.DataFrame, scelte: dict) -> tuple[dict, list[dict]]:
    """Sfidante all'origine t. `z` = X1 settimanale per la costruzione di base; `VV` = variabili settimanali costruite con i
    dati giornalieri disponibili fino all'origine. Se gli estremi dell'insieme di FS non sono calcolabili si usa l'insieme di F1."""
    prev, cand = V4B.prevedi_fs(df, t, z, scelte_4b(scelte))
    return _finisci(df, t, prev, cand, VV, (scelte["insieme_massimo_fs"], scelte["insieme_minimo_fs"]), (scelte["insieme_massimo_f1"], scelte["insieme_minimo_f1"]), scelte)


def configurazione(versione: str, scelte: dict) -> dict:
    """Parametri, finestre e regole della versione (senza data e commit: li aggiunge `congela`)."""
    cfg = V4B.configurazione(versione, scelte_4b(scelte))
    ins = (scelte[f"insieme_massimo_{versione}"], scelte[f"insieme_minimo_{versione}"])
    descr = lambda s: {"insieme": s, "variabili": list(V4B.V4.M.NOMI_M1) + (list(V.S1) + ["rendimento ed escursione dell'S&P 500 della settimana +1, previsti"] if s == "B3" else list(V.INSIEMI[s])),
                       "serie_giornaliere": list(serie_richieste(s))}
    cfg["candele"] = cfg["candele"] | {"prima_candela": {
        "settimana": 1, "apertura": "chiusura all'origine x rapporto di apertura (come K2)",
        "massimo": descr(ins[0]), "minimo": descr(ins[1]),
        "modello_degli_estremi": {"obiettivo": "ln(estremo della settimana +1 / chiusura all'origine)", "stima": "regressione sulla mediana", "finestra_settimane": ES.FINESTRA,
                                  "minimo_coppie": ES.MINIMO_COPPIE, "variabili_standardizzate_nella_finestra": True},
        "ripiego_estremi": None if versione == "f1" else {"se": "una serie giornaliera dell'insieme manca o è più vecchia dell'origine", "insieme_massimo": scelte["insieme_massimo_f1"],
                                                           "insieme_minimo": scelte["insieme_minimo_f1"]},
        "corpo": scelte["corpo"], "chiusura_k5": {"ribasso": "chiusura all'origine x exp(-d)", "rialzo": "chiusura all'origine x exp(+u)", "neutra": "come K2",
                                                  "finestra_mosse_settimane": PR.FINESTRA_MOSSE} if scelte["corpo"] == "K5" else None,
        "taglio": "apertura e chiusura sono riportate dentro l'intervallo minimo-massimo previsto; gli estremi non si spostano",
        "colore": "secondo l'etichetta della chiusura; grigio se neutra", "nome_nel_registro": nome_costruzione(scelte)}}
    cfg["file_sorgenti_congelati"] = list(SORGENTI_CONGELATI)
    return cfg
