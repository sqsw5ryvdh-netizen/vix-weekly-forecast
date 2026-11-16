"""Fase 4: il registro delle previsioni salva la versione usata (f1, f2, f3).

Una previsione e' identificata da origine + versione: la stessa coppia non viene ne' duplicata ne' riscritta. Le righe
scritte prima della Fase 4 non hanno la versione e non vengono toccate. `verifica` confronta le previsioni con le
settimane nel frattempo completate, separatamente per versione. Il registro serve a misurare dal vivo: NON serve a
scegliere modelli.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..fase3 import registro as RG3

SENZA_VERSIONE = "prima della Fase 4"


def righe(prev: dict, candele: list[dict], data_esecuzione: str) -> pd.DataFrame:
    r = RG3.righe(prev, candele, data_esecuzione)
    r.insert(2, "versione", prev["versione"])
    return r


def aggiungi(path: Path, nuove: pd.DataFrame) -> tuple[bool, str]:
    """Aggiunge le righe se la coppia (origine, versione) non e' gia' registrata. Restituisce (aggiunta, messaggio)."""
    origine, versione = str(nuove["data_origine"].iloc[0]), str(nuove["versione"].iloc[0])
    if path.exists():
        reg = pd.read_csv(path, dtype={"data_origine": str, "data_esecuzione": str})
        if "versione" not in reg.columns:
            reg["versione"] = pd.NA
        gia = (reg["data_origine"] == origine) & (reg["versione"] == versione)
        if gia.any():
            return False, f"Registro: la previsione {versione} con origine {origine} è già registrata (il {reg.loc[gia, 'data_esecuzione'].iloc[0]}); non viene duplicata né modificata."
        pd.concat([reg, nuove], ignore_index=True).to_csv(path, index=False, float_format="%.4f")
    else:
        nuove.to_csv(path, index=False, float_format="%.4f")
    return True, f"Registro: aggiunta la previsione {versione} con origine {origine} ({len(nuove)} settimane) in {path}."


def verifica(reg: pd.DataFrame, storia: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dettaglio e pagella dal vivo, per versione e orizzonte."""
    reg = reg.copy()
    reg["versione"] = (reg["versione"] if "versione" in reg.columns else pd.Series(pd.NA, index=reg.index)).fillna(SENZA_VERSIONE)
    det, pag = [], []
    for v, sub in reg.groupby("versione", sort=False):
        d, p = RG3.verifica(sub.drop(columns="versione"), storia)
        if not d.empty:
            det.append(d.assign(versione=v))
            pag.append(p.assign(versione=v))
    if not det:
        return pd.DataFrame(), pd.DataFrame()
    prima = lambda t: t[["versione"] + [c for c in t.columns if c != "versione"]]
    return prima(pd.concat(det, ignore_index=True)), prima(pd.concat(pag, ignore_index=True))
