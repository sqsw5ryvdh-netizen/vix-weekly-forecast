"""Costanti, accesso ai dati e bootstrap condivisi dai moduli di Fase 1.

Unico punto di accesso ai dati: `load_dev()`. La colonna `open` viene scartata alla lettura
(decisione di Fase 1: non usarla). `guardia_dati()` blocca a runtime l'apertura dei file
holdout / clean / raw: `python -m src.run_phase1` gira interamente sotto questa guardia.
"""
from __future__ import annotations

import os
import sys
import zlib
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .. import config as C

OUT = C.REPORTS_DIR / "fase1"
FIG = C.FIG_DIR / "fase1"

HORIZONS = (1, 4, 12)                 # 1 e 4 principali, 12 secondario
B_BOOT = 2000                         # repliche bootstrap
CI = (2.5, 97.5)                      # intervallo percentile al 95%
BLOCK = 26                            # settimane per blocco nel bootstrap a blocchi (dati settimanali dipendenti)
SPLIT_DATE = "2013-01-01"             # A5: prima meta' 2000-2012, seconda 2013-2024

BANDS7 = (12.0, 15.0, 20.0, 25.0, 30.0, 40.0)
BAND7_LABELS = ("<12", "12-15", "15-20", "20-25", "25-30", "30-40", ">=40")
DIR_THR = 0.10                        # "oltre 10%": |variazione log a 4 settimane| > 0,10
DIR_LABELS = ("in calo", "stabile", "in rialzo")

PEAK_CLASSES = ((30.0, 40.0, "30-40"), (40.0, 50.0, "40-50"), (50.0, np.inf, ">=50"))
EP_DEFS = {"soglia30": "chiusura >= 30 (Fase 0)", "relativa": "chiusura >= 1,5 x mediana delle 26 settimane precedenti"}
MERGE_GAP = 4
PRE_WIN = 26                          # settimane per il livello di partenza
MIN_RISE = 3.0                        # salita minima (punti VIX) perche' la quota recuperata sia definita
TRAJ_WEEKS = 26
REBOUND = 0.20                        # rialzo settimanale "oltre +20%" (variazione percentuale semplice)
CALM_LEVEL, CALM_MIN_WEEKS = 15.0, 8
CALM_RANGE_WIN, CALM_RANGE_Q, CALM_RANGE_MIN_HIST = 12, 0.20, 104

NOME_FILE_DATI = C.DEV_FILE.name
_FORBIDDEN = {C.HOLDOUT_FILE.name, C.CLEAN_FILE.name, C.RAW_FILE.name}
_guard = {"attiva": False, "installata": False}


# Dalla Fase 2: sono vietati anche i file originali dei dati esterni e le loro parti di holdout.
_FORBIDDEN_DIRS = tuple(str(C.ROOT / "data" / "external" / d) + os.sep for d in ("raw", "holdout"))


def _hook(event, args):
    if _guard["attiva"] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if os.path.basename(path) in _FORBIDDEN or os.path.abspath(path).startswith(_FORBIDDEN_DIRS):
            raise PermissionError(f"accesso vietato a {path} (holdout o dati non ammessi in questa fase)")


@contextmanager
def guardia_dati():
    """Dentro il blocco, qualunque apertura dei file holdout / clean / raw solleva PermissionError."""
    if not _guard["installata"]:
        sys.addaudithook(_hook)
        _guard["installata"] = True
    prev, _guard["attiva"] = _guard["attiva"], True
    try:
        yield
    finally:
        _guard["attiva"] = prev


def load_dev() -> pd.DataFrame:
    df = pd.read_csv(C.DEV_FILE, parse_dates=["date"])
    return df.drop(columns=["open", "week_start"])


def band7(level) -> np.ndarray:
    return np.searchsorted(np.asarray(BANDS7), np.asarray(level), side="right")


def direction(d4) -> np.ndarray:
    """0 = in calo oltre 10%, 1 = stabile, 2 = in rialzo oltre 10% (variazione log a 4 settimane)."""
    d4 = np.asarray(d4)
    return 1 + (d4 > DIR_THR).astype(int) - (d4 < -DIR_THR).astype(int)


# ------------------------------------------------------------------ bootstrap

def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng([C.SEED, zlib.crc32(tag.encode())])


def _ci(stats) -> tuple[float, float]:
    s = np.asarray(stats, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 0.5 * B_BOOT:
        return float("nan"), float("nan")
    lo, hi = np.percentile(s, CI)
    return float(lo), float(hi)


def boot_iid(n: int, stat, tag: str) -> tuple[float, float]:
    """Bootstrap i.i.d. su n unita' (episodi, periodi). `stat(idx)` riceve gli indici ricampionati."""
    if n < 2:
        return float("nan"), float("nan")
    rng = _rng(tag)
    return _ci([stat(rng.integers(0, n, n)) for _ in range(B_BOOT)])


def boot_cluster(clusters, stat, tag: str) -> tuple[float, float]:
    """Bootstrap per gruppi: si ricampionano interi gruppi di righe (blocchi di tempo o periodi)."""
    clusters = np.asarray(clusters)
    groups = [np.flatnonzero(clusters == g) for g in np.unique(clusters)]
    G = len(groups)
    if G < 2:
        return float("nan"), float("nan")
    rng = _rng(tag)
    return _ci([stat(np.concatenate([groups[g] for g in rng.integers(0, G, G)])) for _ in range(B_BOOT)])


def blocks(index) -> np.ndarray:
    """Identificativo di blocco (26 settimane consecutive) a partire dalla posizione nella serie."""
    return np.asarray(index) // BLOCK
