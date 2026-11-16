"""Fase 2c: registro delle previsioni e pagella dal vivo.

Ogni esecuzione del programma (senza --origine) aggiunge al registro le candele e le fasce previste, con data di
origine e data di esecuzione. Una previsione con la stessa origine non viene ne' duplicata ne' riscritta.
`verifica` confronta le previsioni registrate con le settimane nel frattempo completate. Il registro serve a
misurare dal vivo: NON serve a scegliere modelli.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CAMPI = ("apertura", "massimo", "minimo", "chiusura")
REALI = {"apertura": "open", "massimo": "high", "minimo": "low", "chiusura": "close"}
MIN_CASI = 20


def righe(prev: dict, candele: list[dict], data_esecuzione: str) -> pd.DataFrame:
    rows = []
    for s, c in zip(prev["settimane"], candele):
        r = {"data_origine": prev["data_origine"], "data_esecuzione": data_esecuzione, "chiusura_origine": prev["chiusura_origine"], "h": s["h"],
             "data_prevista": s["data"], "costruzione": c["costruzione"], "modello_direzione": c["modello_direzione"]}
        r |= {k: c[k] for k in CAMPI} | {"direzione": c["direzione"], "prob_direzione": c["prob_direzione"], "mediana_d4": s["quantili"]["q50"]}
        for lev in ("50", "80", "90"):
            r |= {f"fascia_{lev}_inf": s["fasce"][lev][0], f"fascia_{lev}_sup": s["fasce"][lev][1]}
        r |= {f"p_vix_sopra_{k.split('>')[1]}": v for k, v in s["probabilita"].items()}
        rows.append(r)
    return pd.DataFrame(rows)


def aggiungi(path: Path, nuove: pd.DataFrame) -> tuple[bool, str]:
    """Aggiunge le righe se l'origine non e' gia' registrata. Restituisce (aggiunta, messaggio)."""
    origine = str(nuove["data_origine"].iloc[0])
    if path.exists():
        reg = pd.read_csv(path, dtype={"data_origine": str})
        if (reg["data_origine"] == origine).any():
            quando = reg.loc[reg["data_origine"] == origine, "data_esecuzione"].iloc[0]
            return False, f"Registro: la previsione con origine {origine} è già registrata (il {quando}); non viene duplicata né modificata."
        pd.concat([reg, nuove], ignore_index=True).to_csv(path, index=False, float_format="%.4f")
    else:
        nuove.to_csv(path, index=False, float_format="%.4f")
    return True, f"Registro: aggiunta la previsione con origine {origine} ({len(nuove)} settimane) in {path}."


def verifica(reg: pd.DataFrame, storia: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Confronta le previsioni registrate con le settimane completate. Restituisce (dettaglio, pagella per orizzonte)."""
    pos = {str(pd.Timestamp(d).date()): i for i, d in enumerate(storia["date"])}
    rows = []
    for r in reg.itertuples():
        i0 = pos.get(str(r.data_origine))
        if i0 is None or i0 + r.h >= len(storia):
            continue                                     # origine non presente nel file o settimana non ancora completata
        w = storia.iloc[i0 + r.h]
        d = {"data_origine": r.data_origine, "h": r.h, "data_reale": str(pd.Timestamp(w["date"]).date()), "data_esecuzione": r.data_esecuzione,
             "registrata_prima_dell_esito": str(r.data_esecuzione) <= str(pd.Timestamp(w["date"]).date()), "costruzione": r.costruzione}
        for k, col in REALI.items():
            d |= {f"{k}_prevista": getattr(r, k), f"{k}_reale": float(w[col]), f"errore_{k}": abs(getattr(r, k) - float(w[col]))}
        d["errore_mediana_d4"] = abs(r.mediana_d4 - float(w["close"]))
        d["direzione_corpo_ok"] = bool((r.chiusura >= r.apertura) == (w["close"] >= w["open"]))
        d["direzione_chiusura_ok"] = bool(np.sign(r.chiusura - r.chiusura_origine) == np.sign(w["close"] - r.chiusura_origine))
        inter = max(0.0, min(r.massimo, w["high"]) - max(r.minimo, w["low"]))
        d["sovrapposizione"] = inter / (max(r.massimo, w["high"]) - min(r.minimo, w["low"]))
        d["chiusura_nel_corpo"] = bool(min(r.apertura, r.chiusura) <= w["close"] <= max(r.apertura, r.chiusura))
        d["chiusura_nell_escursione"] = bool(r.minimo <= w["close"] <= r.massimo)
        for lev in ("50", "80", "90"):
            d[f"chiusura_nella_fascia_{lev}"] = bool(getattr(r, f"fascia_{lev}_inf") <= w["close"] <= getattr(r, f"fascia_{lev}_sup"))
        rows.append(d)
    det = pd.DataFrame(rows)
    if det.empty:
        return det, pd.DataFrame()
    cols = [c for c in det.columns if c.startswith(("errore_", "direzione_", "sovrapposizione", "chiusura_nel", "registrata"))]
    pag = det.groupby("h")[cols].mean(numeric_only=False).astype(float)
    pag.insert(0, "n", det.groupby("h").size())
    pag["meno_di_20_casi"] = pag["n"] < MIN_CASI
    return det, pag.reset_index()
