"""Fase 4b: le due versioni che andranno all'esame — F1 e la sfidante FS — con la nuova prima candela.

F1  come nella Fase 4 (D4, D1, candele K2 con ampiezze della variante scelta), con la prima candela della Fase 4b.
FS  sfidante. E' costruita attorno al candidato scelto dalla regola della Parte B (il Brier piu' basso a h = 1 tra
    P2, P3, P4, P5): la probabilita' che la chiusura +1 superi quella dell'origine e' quella del candidato (media
    semplice delle probabilita' delle versioni che lo compongono); chiusura centrale, fasce, probabilita' di livello,
    etichette e candele delle settimane +2…+4 vengono dalla costruzione di F2 se il candidato contiene P2,
    altrimenti da quella di F3. Se la costruzione di base non e' calcolabile (VIX3M mancante) FS non e'
    disponibile e si ripiega su F1.
Prima candela (entrambe le versioni): costruzione K4 oppure K2 e minimo M1q oppure M0, secondo le regole delle Parti
C e D; il colore segue l'etichetta (grigio se neutra).
`scelte` = {"variante": A0|A1|A2, "prima_candela": K2|K4, "minimo": M0|M1q, "candidato": P2|P3|P4|P5}.
Tutte le funzioni usano solo le settimane 0..t (e, per X1, solo i dati giornalieri fino all'ultima seduta della settimana t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..fase2c import componenti as CP
from ..fase3 import candele as K3
from ..fase4 import candele as KA
from ..fase4 import versioni as V4
from . import prima as PR

VERSIONI = ("f1", "fs")
NOMI = {"f1": "F1 solo storico del VIX", "fs": "FS sfidante"}
CANDIDATI = {"P1": ("f1",), "P2": ("f2",), "P3": ("f3",), "P4": ("f1", "f2", "f3"), "P5": ("f2", "f3")}
SFIDANTI = ("P2", "P3", "P4", "P5")
DESCRIZIONI = {"P1": "P1 F1", "P2": "P2 F2", "P3": "P3 F3", "P4": "P4 media di F1, F2 e F3", "P5": "P5 media di F2 e F3"}
CARTELLA = V4.CARTELLA
SORGENTI_CONGELATI = V4.SORGENTI_CONGELATI + ("src/fase4b/prima.py", "src/fase4b/versioni.py")
NonDisponibile = V4.NonDisponibile


def base_di(candidato: str) -> str:
    """Costruzione usata per chiusura centrale, fasce e settimane +2…+4: F2 se il candidato contiene P2, altrimenti F3."""
    return "f2" if "f2" in CANDIDATI[candidato] else "f3"


def nome_costruzione(scelte: dict) -> str:
    return f"{scelte['prima_candela']}{'+M1q' if scelte['minimo'] == 'M1q' else ''}/{KA.costruzione(scelte['variante'])}"


def prima_candela(df: pd.DataFrame, t: int, cand1: dict, centrale: float, scelte: dict) -> dict:
    """Nuova candela della settimana +1 a partire da quella K2 (`cand1`, che porta etichetta e probabilita' della versione)."""
    d0 = df.iloc[: t + 1]
    S = CP.serie(d0)
    x, c0 = np.log(S.c), float(S.c[-1])
    comp = KA.componenti(S, t, 1, scelte["variante"])
    if scelte["prima_candela"] == "K4":
        dn, up = PR.mosse(x)
        k = PR.candela_k4(c0, comp["rapporto_apertura"], cand1["direzione_chiusura"], dn, up, centrale, comp["ombra_sup"], comp["ombra_inf"])
        extra = {"mossa_ribasso": dn, "mossa_rialzo": up, "modello_direzione": "etichetta"}
    else:
        k, extra = {c: cand1[c] for c in KA.CAMPI}, {}
    if scelte["minimo"] == "M1q":
        k = PR.con_minimo(k, PR.minimo_m1q(x, S.l))
    su = k["chiusura"] >= k["apertura"]
    return cand1 | k | extra | {"costruzione": nome_costruzione(scelte), "prima_candela": scelte["prima_candela"], "minimo_da": scelte["minimo"], "colore": cand1["direzione_chiusura"],
                                "direzione": "rialzo" if su else "ribasso", "prob_direzione": cand1["p_rialzo"] if su else 1 - cand1["p_rialzo"]}


def _finisci(df: pd.DataFrame, t: int, prev: dict, cand: list[dict], scelte: dict, versione: str) -> tuple[dict, list[dict]]:
    nome = nome_costruzione(scelte)
    cand = [c | {"versione": versione, "costruzione": nome} for c in cand]
    cand[0] = prima_candela(df, t, cand[0], prev["settimane"][0]["chiusura"], scelte)
    return prev | {"versione": versione}, cand


def prevedi_f1(df: pd.DataFrame, t: int, scelte: dict) -> tuple[dict, list[dict]]:
    prev, cand = V4.prevedi_f1(df, t, scelte["variante"])
    return _finisci(df, t, prev, cand, scelte, "f1")


def prevedi_fs(df: pd.DataFrame, t: int, z: np.ndarray | None, scelte: dict) -> tuple[dict, list[dict]]:
    """Sfidante all'origine t. `z` = X1 settimanale (serve solo se il candidato contiene F2). Solleva NonDisponibile se manca."""
    membri = CANDIDATI[scelte["candidato"]]
    if "f2" in membri and z is None:
        raise NonDisponibile("i dati giornalieri di VIX e VIX3M non sono disponibili")
    ris = {}
    for m in membri:
        ris[m] = (V4.prevedi_f1(df, t, scelte["variante"]) if m == "f1" else V4.prevedi_f2(df, t, z, scelte["variante"]) if m == "f2"
                  else V4.prevedi_f3(df, t, scelte["variante"]))
    prev, cand = ris[base_di(scelte["candidato"])]
    p1 = float(np.mean([ris[m][1][0]["p_chiusura_su"] for m in membri]))            # probabilita' del candidato per la settimana +1
    d, testo = K3.etichetta(p1)
    cand = [dict(c) for c in cand]
    cand[0] |= {"p_chiusura_su": p1, "direzione_chiusura": d, "etichetta": testo, "p_chiusura_su_membri": {m: ris[m][1][0]["p_chiusura_su"] for m in membri}}
    prev = prev | {"candidato": scelte["candidato"], "costruzione_di_base": base_di(scelte["candidato"])}
    return _finisci(df, t, prev, cand, scelte, "fs")


def configurazione(versione: str, scelte: dict) -> dict:
    """Parametri, finestre e regole della versione (senza data e commit: li aggiunge `congela`)."""
    prima = {"settimana": 1, "costruzione": scelte["prima_candela"], "apertura": "chiusura all'origine x rapporto di apertura",
             "chiusura": ({"ribasso": "chiusura all'origine x exp(-d), d = media delle variazioni log assolute delle settimane in calo",
                           "rialzo": "chiusura all'origine x exp(+u), u = media delle variazioni log delle settimane in rialzo",
                           "neutra": "chiusura centrale della versione", "finestra_settimane": PR.FINESTRA_MOSSE} if scelte["prima_candela"] == "K4"
                          else "come K2 (corpo dalla regressione, direzione dalla frequenza storica)"),
             "ombre": "come K2, sopra e sotto il corpo", "minimo": scelte["minimo"],
             "minimo_m1q": {"obiettivo": "ln(minimo della settimana +1 / chiusura all'origine)", "modello": "regressione sulla mediana sulle variabili di M1",
                            "finestra_settimane": PR.FINESTRA_MINIMO, "vincolo": "il minimo non supera il corpo"} if scelte["minimo"] == "M1q" else None,
             "colore": "secondo l'etichetta della chiusura; grigio se neutra", "nome_nel_registro": nome_costruzione(scelte)}
    if versione == "f1":
        cfg = V4.configurazione("f1", scelte["variante"])
    else:
        cand = scelte["candidato"]
        base = V4.configurazione(base_di(cand), scelte["variante"])
        cfg = {"versione": "fs", "nome": NOMI["fs"], "candidato": {"sigla": cand, "descrizione": DESCRIZIONI[cand], "componenti": list(CANDIDATI[cand]),
                                                                  "regola": "probabilità che la chiusura +1 superi quella dell'origine = media semplice delle probabilità delle versioni componenti"},
               "costruzione_di_base": base_di(cand), "ripiego": "se la costruzione di base o un componente non è calcolabile (VIX3M mancante), si usa F1 e lo si segnala",
               "componenti": {m: {k: v for k, v in V4.configurazione(m, scelte["variante"]).items() if k not in ("candele", "file_sorgenti_congelati")} for m in CANDIDATI[cand]}}
        cfg |= {k: v for k, v in base.items() if k not in ("versione", "nome", "ripiego")}
    cfg["versione"], cfg["nome"] = versione, NOMI[versione]
    cfg["candele"] = cfg["candele"] | {"prima_candela": prima, "settimane_2_4": "K2, come nella Fase 4"}
    cfg["file_sorgenti_congelati"] = list(SORGENTI_CONGELATI)
    return cfg
