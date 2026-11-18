"""Fase 4c, Parti C e D: prima candela (settimana +1) comandata da massimo e minimo previsti.

Massimo e minimo vengono dai modelli diretti (`estremi.py`); il corpo e' K2 (attuale) oppure K5 (ibrida):
  K5  se l'etichetta della chiusura non e' neutra, chiusura = chiusura all'origine x exp(-d) («ribasso») oppure
      x exp(+u) («rialzo»), con d e u le mosse medie della K4 (ultime 260 settimane); se e' neutra, chiusura di K2.
In entrambe il corpo e' tagliato dentro l'intervallo minimo-massimo previsto: se apertura o chiusura escono
dall'intervallo si accorcia il corpo, non si spostano gli estremi.
Se i due modelli danno un minimo sopra il massimo (caso anomalo, contato), i due valori vengono scambiati.
"""
from __future__ import annotations

import numpy as np

CORPI = ("K2", "K5")
CAMPI = ("apertura", "massimo", "minimo", "chiusura")


def chiusura_k5(chiusura_origine, etichetta: str, d: float, u: float, chiusura_k2: float) -> float:
    if etichetta == "ribasso":
        return float(chiusura_origine * np.exp(-d))
    if etichetta == "rialzo":
        return float(chiusura_origine * np.exp(u))
    return float(chiusura_k2)


def componi(apertura: float, chiusura: float, massimo: float, minimo: float) -> dict:
    """Candela con gli estremi dati; il corpo viene tagliato dentro [minimo, massimo]."""
    lo, hi = (minimo, massimo) if minimo <= massimo else (massimo, minimo)
    ap, ch = float(np.clip(apertura, lo, hi)), float(np.clip(chiusura, lo, hi))
    return {"apertura": ap, "chiusura": ch, "massimo": float(hi), "minimo": float(lo), "apertura_tagliata": bool(ap != apertura), "chiusura_tagliata": bool(ch != chiusura),
            "estremi_scambiati": bool(minimo > massimo)}
