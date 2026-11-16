"""Fase 4, Parte A: pagella delle candele per le tre varianti delle ampiezze (A0, A1, A2) e regola di scelta.

Stesse origini delle Fasi 2c e 3: origini, valori reali e mediana D4 vengono da reports/fase3/p0_previsioni_dev.csv
(prodotto da `run_phase3`). Legge il file di sviluppo sotto la guardia.
Regola fissata prima: si usa la variante con il rapporto corpo previsto / reale piu' vicino a 1 a h = 1, purche' il
suo errore quadratico sulle ampiezze non sia significativamente peggiore di A0 (Diebold-Mariano, p < 0,05);
altrimenti resta A0. «Errore quadratico sulle ampiezze» = somma, per ogni origine, degli errori quadratici (in log)
di corpo, ombra superiore e ombra inferiore a h = 1, cioe' allo stesso orizzonte del criterio.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C
from ..fase1 import comune as K
from ..fase2 import valuta as V2
from ..fase2b import dati
from ..fase2c import componenti as CP
from ..fase2c import valuta as VC
from ..fase3 import valuta_candele as VK3
from . import candele as KA

OUT = C.REPORTS_DIR / "fase4"
FIG = C.FIG_DIR / "fase4"
P_PEGGIORE = 0.05
FASCIA_EVIDENZA = (13.0, 22.0)
GRUPPI = ("fascia_origine", "sottoperiodo", "rialzo_forte", "fascia_13_22")


def walk(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    S = CP.serie(df)
    pos = {pd.Timestamp(d): i for i, d in enumerate(df["date"])}
    p3 = pd.read_csv(VK3.OUT / "p0_previsioni_dev.csv", parse_dates=["data_origine", "data_obiettivo"])
    tieni = ["h", "data_origine", "data_obiettivo", "chiusura_origine", "mediana_d4", "fascia_origine", "rialzo_forte", "sottoperiodo",
             "p_chiusura_su", "chiusura_su_reale"] + [f"{c}_reale" for c in KA.CAMPI + KA.COMPONENTI]
    frames = {}
    for h in KA.ORIZZONTI:
        f = p3.loc[p3["h"] == h, tieni].reset_index(drop=True)
        o = np.array([pos[d] for d in f["data_origine"]])
        f.insert(1, "idx_origine", o)
        f["fascia_13_22"] = (f["chiusura_origine"] >= FASCIA_EVIDENZA[0]) & (f["chiusura_origine"] <= FASCIA_EVIDENZA[1])
        for v in KA.VARIANTI:
            comp = pd.DataFrame([KA.componenti(S, t, h, v) for t in o.tolist()])
            for c in ("rapporto_apertura", "p_L0") + KA.COMPONENTI:
                f[f"{v}_{c}"] = comp[c].to_numpy()
        frames[h] = f
    med = {h: dict(zip(f["idx_origine"].tolist(), f["mediana_d4"].tolist())) for h, f in frames.items()}
    for h, f in frames.items():                      # candele K2 ancorate alla mediana D4 della settimana precedente
        for v in KA.VARIANTI:
            cols = {c: [] for c in KA.CAMPI}
            for r in f.to_dict("records"):
                base = r["chiusura_origine"] if h == 1 else med[h - 1][r["idx_origine"]]
                k = KA.candela(base, {c: r[f"{v}_{c}"] for c in ("rapporto_apertura", "p_L0") + KA.COMPONENTI})
                for c in KA.CAMPI:
                    cols[c].append(k[c])
            for c in KA.CAMPI:
                f[f"{v}_{c}"] = cols[c]
    return frames


def perdita(f: pd.DataFrame, v: str) -> np.ndarray:
    """Somma degli errori quadratici (log) delle tre ampiezze, per origine."""
    return sum((f[f"{c}_reale"].to_numpy() - f[f"{v}_{c}"].to_numpy()) ** 2 for c in KA.COMPONENTI)


def ampiezze_tabella(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for h, f in frames.items():
        for c in KA.COMPONENTI + ("somma",):
            y = None if c == "somma" else f[f"{c}_reale"].to_numpy()
            e0 = perdita(f, "A0") if c == "somma" else (y - f[f"A0_{c}"].to_numpy()) ** 2
            for v in KA.VARIANTI:
                e = perdita(f, v) if c == "somma" else (y - f[f"{v}_{c}"].to_numpy()) ** 2
                r = {"h": h, "componente": c, "variante": v, "n": len(f), "metodo": "" if c == "somma" else KA.ASSEGNAZIONE[c][h],
                     "media_reale": np.nan if c == "somma" else y.mean(), "previsione_media": np.nan if c == "somma" else f[f"{v}_{c}"].mean(), "mse": e.mean()}
                if v != "A0":
                    r["mse_vs_a0"] = e.mean() / e0.mean() - 1
                    r["dm_stat"], r["p"] = V2.dm_loss(e0, e, h)           # positivo = la variante ha errore minore di A0
                rows.append(r)
    return pd.DataFrame(rows)


def pagella(frames: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    main, grp = [], []
    for h, f in frames.items():
        for v in KA.VARIANTI:
            r = VC.pagella_riga(f, v, f"f4iou|{v}|{h}")
            o, c = f[f"{v}_apertura"].to_numpy(), f[f"{v}_chiusura"].to_numpy()
            r |= {"corpo_medio_previsto": float(np.abs(c - o).mean()), "corpo_medio_reale": float(np.abs(f["chiusura_reale"] - f["apertura_reale"]).mean()),
                  "escursione_media_prevista": float((f[f"{v}_massimo"] - f[f"{v}_minimo"]).mean()),
                  "escursione_media_reale": float((f["massimo_reale"] - f["minimo_reale"]).mean())}
            main.append({"variante": v, "descrizione": KA.DESCRIZIONI[v], "h": h} | r)
            for col in GRUPPI:
                for key, g in f.groupby(col, sort=False):
                    grp.append({"gruppo": col, "valore": str(key), "variante": v, "h": h} | VC.pagella_riga(g, v))
    return pd.DataFrame(main), pd.DataFrame(grp)


def scegli_variante(pag: pd.DataFrame, f1: pd.DataFrame) -> dict:
    """Applica la regola. `f1` = previsioni a h = 1 (per il test sull'errore quadratico delle ampiezze)."""
    rap = {v: float(pag[(pag["variante"] == v) & (pag["h"] == 1)]["rapporto_corpo"].iloc[0]) for v in KA.VARIANTI}
    cand = min(KA.VARIANTI, key=lambda v: abs(rap[v] - 1))                   # a parita' vince la prima dell'elenco (A0, A1, A2)
    out = {"rapporto_corpo_h1": rap, "candidata": cand, "soglia_p": P_PEGGIORE}
    if cand == "A0":
        return out | {"scelta": "A0", "motivo": "A0 ha già il rapporto del corpo più vicino a 1 a h = 1"}
    dm, p = V2.dm_loss(perdita(f1, "A0"), perdita(f1, cand), 1)
    peggiore = bool(dm < 0 and p < P_PEGGIORE)
    mse0, mse1 = float(perdita(f1, "A0").mean()), float(perdita(f1, cand).mean())
    return out | {"scelta": "A0" if peggiore else cand, "dm_stat": dm, "p": p, "mse_a0": mse0, "mse_candidata": mse1, "significativamente_peggiore": peggiore,
                  "motivo": (f"{cand} ha il rapporto più vicino a 1 ma un errore quadratico sulle ampiezze significativamente peggiore di A0: resta A0" if peggiore
                             else f"{cand} ha il rapporto del corpo più vicino a 1 a h = 1 e un errore quadratico sulle ampiezze non significativamente peggiore di A0")}


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = dati.load_dev_candele()
    frames = walk(df)
    amp = ampiezze_tabella(frames)
    pag, grp = pagella(frames)
    scelta = scegli_variante(pag, frames[1])
    for name, t in (("a_ampiezze", amp), ("a_pagella", pag), ("a_pagella_gruppi", grp)):
        t.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")
    pd.concat([f.drop(columns="idx_origine") for f in frames.values()], ignore_index=True).to_csv(
        OUT / "a_previsioni_dev.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    info = {"varianti": {v: KA.VARIANTI[v] | {"descrizione": KA.DESCRIZIONI[v]} for v in KA.VARIANTI},
            "assegnazione": {c: {str(h): m for h, m in d.items()} for c, d in KA.ASSEGNAZIONE.items()}, "scelta": scelta,
            "per_orizzonte": {str(h): {"n_previsioni": int(len(f)), "prima_data_obiettivo": str(f["data_obiettivo"].iloc[0].date()),
                                       "ultima_data_obiettivo": str(f["data_obiettivo"].iloc[-1].date()), "n_fascia_13_22": int(f["fascia_13_22"].sum())} for h, f in frames.items()}}
    (OUT / "a_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with K.guardia_dati():
        run()
