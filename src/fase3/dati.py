"""Fase 3: accesso ai dati. Solo parti di sviluppo: il VIX settimanale e i dati esterni giornalieri fino all'8 novembre 2024.

I dati esterni sono quelli scaricati nella Fase 2 da `python -m src.fetch_external` (cartella data/external/dev,
non pubblicata nel repository). Gli originali e le parti di holdout restano bloccati dalla guardia.
Serie usate: solo quelle dell'elenco chiuso della Fase 3. L'indice di condizioni finanziarie e i differenziali di
credito sono esclusi per decisione e non vengono nemmeno letti.
"""
from __future__ import annotations

import json

import pandas as pd

from .. import config as C

ESTERNI_DEV = C.ROOT / "data" / "external" / "dev" / "esterni_giornalieri_dev.csv"
INVENTARIO = C.REPORTS_DIR / "fase2" / "c_inventario.json"
SERIE = ("vix", "vix3m", "vvix", "spx", "vx1", "vx2")


def esterni_disponibili() -> bool:
    return ESTERNI_DEV.exists()


def load_settimanale() -> pd.DataFrame:
    return pd.read_csv(C.DEV_FILE, parse_dates=["date"])[["date", "close", "log_close"]]


def ritardi() -> dict[str, int]:
    """Ritardo di pubblicazione (giorni) di ogni serie, dall'inventario della Fase 2."""
    inv = json.loads(INVENTARIO.read_text())["serie"]
    return {s: int(inv[s]["ritardo_giorni_usato"]) for s in SERIE}


def load_giornaliero() -> pd.DataFrame:
    """Dati giornalieri di sviluppo, indice = data in cui il valore e' disponibile (data di osservazione + ritardo)."""
    if not esterni_disponibili():
        raise SystemExit(f"Manca {ESTERNI_DEV}: eseguire prima `python -m src.fetch_external` (i dati esterni non sono nel repository).")
    g = pd.read_csv(ESTERNI_DEV, usecols=["data_osservazione", *SERIE], parse_dates=["data_osservazione"]).set_index("data_osservazione").sort_index()
    rit = ritardi()
    if any(rit.values()):                              # nessuna delle serie usate ha un ritardo; se ne avesse, la data diventa quella di disponibilita'
        g = pd.concat([g[s].dropna().rename(lambda d, k=rit[s]: d + pd.Timedelta(days=k)) for s in SERIE], axis=1).sort_index()
    return g
