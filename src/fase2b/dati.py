"""Fase 2b: accesso ai dati di sviluppo per il walk-forward e per i grafici di verifica.

A differenza di `fase1.comune.load_dev()`, qui la colonna `open` viene conservata: serve SOLO a disegnare
le candele reali nei grafici. Nessun modello la usa (lo verifica un test).
"""
from __future__ import annotations

import pandas as pd

from .. import config as C

COLONNE = ["date", "open", "high", "low", "close", "log_close"]


def load_dev_candele() -> pd.DataFrame:
    return pd.read_csv(C.DEV_FILE, parse_dates=["date"])[COLONNE]
