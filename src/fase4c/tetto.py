"""Fase 4c, Parte B — B2: TETTO TEORICO. NON E' UNA PREVISIONE UTILIZZABILE.

Usa rendimento ed escursione dell'S&P 500 REALMENTE AVVENUTI nella settimana t + 1, cioe' informazione che all'origine
t non esiste. Serve solo a misurare quanto varrebbe conoscere l'S&P 500 della settimana prevista. Questo modulo e'
importato soltanto dalla valutazione di sviluppo (`valuta.py`): non entra nel programma di previsione ne' nelle
versioni congelate (lo verifica un test).
"""
from __future__ import annotations

import numpy as np

from . import estremi as ES
from . import incrociata as IN


def b2_tetto_teorico(x: np.ndarray, livelli: np.ndarray, VV_fino_a_t, spx_futuro: tuple[float, float]) -> float:
    """Livello della settimana t + 1 dal legame B1 (stimato sui dati fino a t) con l'S&P 500 FUTURO `spx_futuro` =
    (rendimento, escursione) della settimana t + 1. L'unico dato successivo all'origine e' `spx_futuro`, passato in modo esplicito."""
    t = len(x) - 1
    return ES.mediana_diretta(x, livelli, IN.matrice_legame(VV_fino_a_t), np.concatenate([VV_fino_a_t[IN.COL_S1].to_numpy()[t], np.asarray(spx_futuro, dtype=float)]))
