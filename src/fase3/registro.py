"""Fase 3, Parte 0: il registro delle previsioni conserva anche l'etichetta della chiusura.

Le righe gia' scritte non vengono toccate (regola della Fase 2c): le previsioni registrate prima della Fase 3 restano
con la costruzione e l'etichetta di allora, e le colonne nuove per loro sono vuote.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..fase2c import registro as RG


def righe(prev: dict, candele: list[dict], data_esecuzione: str) -> pd.DataFrame:
    r = RG.righe(prev, candele, data_esecuzione)
    r["p_chiusura_sopra_origine"] = [c["p_chiusura_su"] for c in candele]
    r["etichetta_chiusura"] = [c["direzione_chiusura"] for c in candele]
    return r


def verifica(reg: pd.DataFrame, storia: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Come la verifica della Fase 2c, piu' l'esito dell'etichetta della chiusura (solo righe che la contengono e non neutre)."""
    det, pag = RG.verifica(reg, storia)
    if det.empty or "etichetta_chiusura" not in reg.columns:
        return det, pag
    chiave = reg.assign(data_origine=reg["data_origine"].astype(str)).set_index(["data_origine", "h"])
    et = [chiave.loc[(str(r.data_origine), r.h), "etichetta_chiusura"] for r in det.itertuples()]
    c0 = [float(chiave.loc[(str(r.data_origine), r.h), "chiusura_origine"]) for r in det.itertuples()]
    su = det["chiusura_reale"].to_numpy() > np.array(c0)
    det["etichetta_chiusura"] = et
    det["etichetta_non_neutra"] = [e in ("rialzo", "ribasso") for e in et]
    det["etichetta_corretta"] = [(s if e == "rialzo" else not s) if e in ("rialzo", "ribasso") else np.nan for e, s in zip(et, su)]
    g = det.groupby("h")
    extra = pd.DataFrame({"n_etichette_non_neutre": g["etichetta_non_neutra"].sum().astype(int),
                          "etichette_corrette": g["etichetta_corretta"].apply(lambda v: float(pd.to_numeric(v, errors="coerce").mean()))}).reset_index()
    return det, pag.merge(extra, on="h", how="left")
