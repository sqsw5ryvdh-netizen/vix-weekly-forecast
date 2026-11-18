"""Fase 4c: serie giornaliere che servono alla sfidante FS nel programma di previsione.

USATO SOLO DA `src.forecast` (mai da `run_phase4c`, che legge i soli dati di sviluppo sotto la guardia). `aggiorna`
scarica dal sito Cboe le serie richieste (storia completa) nella cartella data/external/forecast (non pubblicata);
`leggi` le tronca all'ultima seduta della settimana di origine e dice, per ciascuna, se c'e' ed e' aggiornata.
Le variabili settimanali vengono poi costruite da `variabili.costruisci`, lo stesso codice delle misure di sviluppo.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .. import config as C
from . import versioni as VS

CARTELLA = C.ROOT / "data" / "external" / "forecast"
CODICI = {"vix": "VIX", "vix3m": "VIX3M", "vvix": "VVIX", "spx": "SPX"}
INDIRIZZO = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
FILE = {k: f"{v}_History.csv" for k, v in CODICI.items()}


def serie_di_fs(scelte: dict) -> tuple[str, ...]:
    """Serie giornaliere che servono a FS: quelle della costruzione di base e quelle degli insiemi di massimo e minimo."""
    s = set(VS.serie_richieste(scelte["insieme_massimo_fs"])) | set(VS.serie_richieste(scelte["insieme_minimo_fs"]))
    if "f2" in VS.CANDIDATI[scelte["candidato"]]:
        s |= {"vix", "vix3m"}
    return tuple(k for k in CODICI if k in s)


def aggiorna(cartella: Path, serie: tuple[str, ...], scarica=None) -> dict:
    """Scarica le serie richieste e le salva. Restituisce byte e impronta di ogni file."""
    if scarica is None:
        from ..fetch_external import scarica
    cartella.mkdir(parents=True, exist_ok=True)
    out = {}
    for nome in serie:
        url = INDIRIZZO.format(CODICI[nome])
        raw = scarica(url)
        (cartella / FILE[nome]).write_bytes(raw)
        out[nome] = {"indirizzo": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return out


def _chiusure(path: Path, nome: str) -> pd.Series:
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    d["date"] = pd.to_datetime(d["date"], format="%m/%d/%Y")
    col = "close" if "close" in d.columns else (nome if nome in d.columns else d.columns[-1])
    return d.set_index("date").sort_index()[col].astype(float).dropna()


def leggi(cartella: Path | None, serie: tuple[str, ...], origine: pd.Timestamp, inizio_settimana: pd.Timestamp) -> tuple[pd.DataFrame | None, dict]:
    """Dati giornalieri delle serie richieste, troncati all'origine. Per ogni serie: presente? aggiornata (almeno un
    valore dentro la settimana di origine)? Le serie assenti non compaiono tra le colonne."""
    stato, colonne = {}, {}
    for nome in serie:
        path = None if cartella is None else cartella / FILE[nome]
        if path is None or not path.exists():
            stato[nome] = {"presente": False, "aggiornata": False, "ultima_data_entro_origine": None}
            continue
        s = _chiusure(path, nome)
        s = s[s.index <= origine]                                   # nulla di successivo all'ultima seduta della settimana di origine
        ultima = s.index.max() if len(s) else None
        stato[nome] = {"presente": True, "aggiornata": bool(ultima is not None and ultima > inizio_settimana), "ultima_data_entro_origine": None if ultima is None else str(ultima.date()),
                       "ultimo_valore": None if ultima is None else float(s.iloc[-1]), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        colonne[nome] = s
    return (pd.concat(colonne, axis=1, sort=True) if colonne else None), stato
