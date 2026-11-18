"""VIX9D: accesso ai dati di sviluppo e guardia sull'holdout.

`guardia()` aggiunge alla guardia del progetto (holdout del VIX, originali e holdout dei dati esterni) il blocco dei
file del VIX9D che contengono settimane successive al confine: il file completo, il VIX settimanale completo da Cboe e
il file di holdout. La guardia del progetto non viene modificata: questa ne aggiunge una propria.
`load_dev()` e' l'unico punto di lettura per le misure: VIX9D settimanale di sviluppo, con accanto la chiusura
settimanale del VIX (parte di sviluppo) della stessa settimana.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .. import config as C
from ..fase1 import comune as K
from . import config as G

_VIETATI = {G.FILE_COMPLETO.name, G.FILE_VIX_COMPLETO.name, G.FILE_HOLDOUT.name}
_stato = {"attiva": False, "installata": False}


def _hook(event, args):
    if _stato["attiva"] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if os.path.basename(path) in _VIETATI:
            raise PermissionError(f"accesso vietato a {path} (VIX9D: dati successivi al confine di sviluppo)")


@contextmanager
def guardia():
    """Dentro il blocco: guardia del progetto + blocco dei file del VIX9D successivi al confine."""
    if not _stato["installata"]:
        sys.addaudithook(_hook)
        _stato["installata"] = True
    prima, _stato["attiva"] = _stato["attiva"], True
    try:
        with K.guardia_dati():
            yield
    finally:
        _stato["attiva"] = prima


def unisci_vix(w9: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge a ogni settimana del VIX9D la chiusura settimanale del VIX della stessa settimana (NaN se manca) e
    la colonna `candela` (True se tutte le sedute hanno apertura, massimo e minimo veri)."""
    v = vix[["week_start", "date", "close"]].rename(columns={"date": "date_vix", "close": "vix"})
    d = w9.merge(v, on="week_start", how="left")
    d["candela"] = d["sedute_solo_chiusura"] == 0
    d["vix_stessa_seduta"] = d["date_vix"] == d["date"]
    return d.drop(columns=["date_vix"])


def load_dev() -> pd.DataFrame:
    w9 = pd.read_csv(G.FILE_DEV, parse_dates=["date", "week_start"])
    vix = pd.read_csv(C.DEV_FILE, parse_dates=["date", "week_start"])
    d = unisci_vix(w9, vix)
    d["log_close"] = np.log(d["close"])
    return d
