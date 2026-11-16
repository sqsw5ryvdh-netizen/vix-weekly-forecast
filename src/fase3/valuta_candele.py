"""Fase 3, Parte 0: misura delle nuove candele sul campione di sviluppo (stesse origini della Fase 2c).

Legge il file di sviluppo (sotto la guardia) e reports/fase2c/previsioni_dev.csv (prodotto da `run_phase2c`): da li'
vengono le origini, le candele K2 della Fase 2c (per il confronto) e la mediana D4 a cui le candele sono ancorate.
Decide, con la regola fissata in `candele.py`, se per corpo e ombre vale la regressione o la media storica.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C
from ..fase1 import comune as K
from ..fase1 import modelli as M
from ..fase2 import densita as D
from ..fase2 import valuta as V2
from ..fase2b import dati, grafici
from ..fase2b import valuta as VB
from ..fase2c import componenti as CP
from ..fase2c import valuta as VC
from . import candele as KA

OUT = C.REPORTS_DIR / "fase3"
FIG = C.FIG_DIR / "fase3"
CAMPI = VC.CAMPI


def walk(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Per ogni orizzonte e ogni origine della Fase 2c: componenti (regressione e media storica) e probabilita' D1
    che la chiusura a t + h sia sopra la chiusura a t."""
    S = CP.serie(df)
    x = np.log(S.c)
    pos = {pd.Timestamp(d): i for i, d in enumerate(df["date"])}
    p2c = pd.read_csv(VC.OUT / "previsioni_dev.csv", parse_dates=["data_origine", "data_obiettivo"])
    tieni = ["h", "data_origine", "data_obiettivo", "chiusura_origine", "mediana_d4", "fascia_origine", "rialzo_forte", "sottoperiodo"] \
        + [f"{c}_reale" for c in CAMPI + KA.COMPONENTI] + [f"K2_{c}" for c in CAMPI]
    frames = {}
    for h in KA.ORIZZONTI:
        f = p2c.loc[p2c["h"] == h, tieni].reset_index(drop=True)
        o = np.array([pos[d] for d in f["data_origine"]])
        f.insert(1, "idx_origine", o)
        comp = pd.DataFrame([KA.componenti_medie(S, t, h) for t in o.tolist()])
        f = pd.concat([f, comp], axis=1)
        c1 = D.centers(x, M.forecast_har, h)
        f["p_chiusura_su"] = [KA.prob_chiusura_su(x, t, h, c1) for t in o.tolist()]
        f["chiusura_su_reale"] = (S.c[o + h] > S.c[o]).astype(float)
        f["freq_chiusura_su"] = [float((x[h:t + 1] > x[:t + 1 - h]).mean()) for t in o.tolist()]     # frequenza storica, esiti noti all'origine
        frames[h] = f
    return frames


def scegli_ampiezze(frames: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Regola fissata prima: la regressione vale solo se ha errore quadratico medio piu' basso della media storica
    e il test di Diebold-Mariano ha p < 0,05; altrimenti media storica."""
    rows, scelte = [], {c: {} for c in KA.COMPONENTI}
    for h, f in frames.items():
        bl = K.blocks(f["idx_origine"].to_numpy())
        for c in KA.COMPONENTI:
            y = f[f"{c}_reale"].to_numpy()
            e_r, e_m = (y - f[f"{c}_reg"].to_numpy()) ** 2, (y - f[f"{c}_media"].to_numpy()) ** 2
            dm, p = V2.dm_loss(e_m, e_r, h)
            lo, hi = K.boot_cluster(bl, lambda i: 1 - e_r[i].mean() / e_m[i].mean(), f"f3amp|{h}|{c}")
            batte = bool(e_r.mean() < e_m.mean() and p < KA.P_BATTE)
            scelte[c][h] = "regressione" if batte else "media"
            rows.append({"componente": c, "h": h, "n": len(y), "media_reale": y.mean(), "previsione_media_regressione": f[f"{c}_reg"].mean(),
                         "previsione_media_storica": f[f"{c}_media"].mean(), "mse_regressione": e_r.mean(), "mse_media_storica": e_m.mean(),
                         "guadagno_pct": 1 - e_r.mean() / e_m.mean(), "guadagno_ci_lo": lo, "guadagno_ci_hi": hi, "dm_stat": dm, "p": p,
                         "scelta": scelte[c][h]})
    return pd.DataFrame(rows), scelte


def aggiungi_candele(frames: dict[int, pd.DataFrame], scelte: dict) -> None:
    med = {h: dict(zip(f["idx_origine"].tolist(), f["mediana_d4"].tolist())) for h, f in frames.items()}
    for h, f in frames.items():
        cols = {c: [] for c in CAMPI}
        for r in f.to_dict("records"):
            base = r["chiusura_origine"] if h == 1 else med[h - 1][r["idx_origine"]]
            k = KA.candela(base, r, scelte, h)
            for c in CAMPI:
                cols[c].append(k[c])
        for c in CAMPI:
            f[f"{KA.COSTRUZIONE}_{c}"] = cols[c]


def pagella(frames: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    main, grp = [], []
    for h, f in frames.items():
        for k, nome in (("K2", "K2 Fase 2c (mediane, direzione L1)"), (KA.COSTRUZIONE, "K2 Fase 3 (medie, direzione L0)")):
            r = VC.pagella_riga(f, k, f"f3iou|{k}|{h}")
            o, c, ro, rc = f[f"{k}_apertura"].to_numpy(), f[f"{k}_chiusura"].to_numpy(), f["apertura_reale"].to_numpy(), f["chiusura_reale"].to_numpy()
            hi, lo = f[f"{k}_massimo"].to_numpy(), f[f"{k}_minimo"].to_numpy()
            r |= {"corpo_medio_previsto": float(np.abs(c - o).mean()), "corpo_medio_reale": float(np.abs(rc - ro).mean()),
                  "escursione_media_prevista": float((hi - lo).mean()), "escursione_media_reale": float((f["massimo_reale"] - f["minimo_reale"]).mean())}
            main.append({"costruzione": k, "descrizione": nome, "h": h} | r)
            for col in ("fascia_origine", "sottoperiodo", "rialzo_forte"):
                for key, g in f.groupby(col, sort=False):
                    grp.append({"gruppo": col, "valore": str(key), "costruzione": k, "h": h} | VC.pagella_riga(g, k))
    return pd.DataFrame(main), pd.DataFrame(grp)


def etichetta_tabella(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Misura dell'etichetta mostrata nel grafico: direzione della chiusura rispetto all'origine, probabilita' D1."""
    rows = []
    for h, f in frames.items():
        p, y, fr = f["p_chiusura_su"].to_numpy(), f["chiusura_su_reale"].to_numpy(), f["freq_chiusura_su"].to_numpy()
        su, giu = p >= KA.NEUTRA[1], p <= KA.NEUTRA[0]
        nn = su | giu
        ok = np.where(su, y == 1, y == 0)
        bl = K.blocks(f["idx_origine"].to_numpy())
        lo, hi = K.boot_cluster(bl, lambda i: ok[i][nn[i]].mean() if nn[i].any() else np.nan, f"f3et|{h}")
        dm, pv = V2.dm_loss((fr - y) ** 2, (p - y) ** 2, h)
        rows.append({"h": h, "n": len(y), "frequenza_chiusura_su": y.mean(), "prob_media": p.mean(), "accuratezza": float(((p >= 0.5) == (y == 1)).mean()),
                     "accuratezza_frequenza_storica": float(((fr >= 0.5) == (y == 1)).mean())} | V2.brier_parts(p, y)
                    | {"brier_frequenza_storica": float(((fr - y) ** 2).mean()), "brier_skill_vs_frequenza": 1 - ((p - y) ** 2).mean() / ((fr - y) ** 2).mean(),
                       "dm_stat": dm, "p": pv, "quota_neutra": float((~nn).mean()), "quota_rialzo": float(su.mean()), "quota_ribasso": float(giu.mean()),
                       "n_non_neutre": int(nn.sum()), "accuratezza_non_neutre": float(ok[nn].mean()) if nn.any() else np.nan,
                       "accuratezza_non_neutre_ci_lo": lo, "accuratezza_non_neutre_ci_hi": hi,
                       "n_rialzo": int(su.sum()), "accuratezza_rialzo": float((y[su] == 1).mean()) if su.any() else np.nan,
                       "n_ribasso": int(giu.sum()), "accuratezza_ribasso": float((y[giu] == 0).mean()) if giu.any() else np.nan})
    return pd.DataFrame(rows)


def pannelli_g3(df: pd.DataFrame, scelte: dict) -> tuple[list[dict], pd.DataFrame]:
    """Stesse 12 origini della Fase 2b; candele della nuova definizione, fasce D4 e linea della mediana."""
    n = len(df)
    out, rows = [], []
    for k in range(VB.G3_N):
        t = n - 1 - VB.G3_PRIMA - VB.G3_PASSO * k
        prev, cand = KA.prevedi_candele(df, t, scelte)
        vista = prev | {"settimane": [s | {c: x[c] for c in CAMPI} for s, x in zip(prev["settimane"], cand)]}
        reali = df.iloc[t + 1:t + 5]
        s4, y4, c0 = prev["settimane"][-1], float(reali["close"].iloc[-1]), prev["chiusura_origine"]
        d50, d80 = s4["fasce"]["50"][0] <= y4 <= s4["fasce"]["50"][1], s4["fasce"]["80"][0] <= y4 <= s4["fasce"]["80"][1]
        et = [(x["direzione_chiusura"], (r.close > c0) if x["direzione_chiusura"] == "rialzo" else (r.close <= c0)) for x, r in zip(cand, reali.itertuples())]
        out.append({"previsione": vista, "storia": df.iloc[t - VB.G3_STORIA + 1:t + 1], "reali": reali, "dentro_50": d50, "dentro_80": d80})
        rows.append({"data_origine": prev["data_origine"], "chiusura_origine": c0, "mediana_d4_4": s4["chiusura"], "candela_chiusura_4": cand[-1]["chiusura"],
                     "chiusura_reale_4": y4, "dentro_50": d50, "dentro_80": d80,
                     "etichette_non_neutre": int(sum(d != "neutra" for d, _ in et)), "etichette_corrette": int(sum(ok for d, ok in et if d != "neutra")),
                     "chiusura_reale_4_nell_escursione_prevista": bool(cand[-1]["minimo"] <= y4 <= cand[-1]["massimo"])})
    return out[::-1], pd.DataFrame(rows[::-1])


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = dati.load_dev_candele()
    frames = walk(df)
    amp, scelte = scegli_ampiezze(frames)
    aggiungi_candele(frames, scelte)
    pag, grp = pagella(frames)
    et = etichetta_tabella(frames)
    for name, t in (("p0_ampiezze", amp), ("p0_pagella", pag), ("p0_pagella_gruppi", grp), ("p0_etichetta", et)):
        t.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")
    pd.concat([f.drop(columns="idx_origine") for f in frames.values()], ignore_index=True).to_csv(
        OUT / "p0_previsioni_dev.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    pan, tg3 = pannelli_g3(df, scelte)
    tg3.to_csv(OUT / "p0_g3_pannelli.csv", index=False, float_format="%.4f")
    n50, n80 = int(tg3["dentro_50"].sum()), int(tg3["dentro_80"].sum())
    nn, okk = int(tg3["etichette_non_neutre"].sum()), int(tg3["etichette_corrette"].sum())
    riga = (f"Chiusura reale a 4 settimane dentro la fascia al 50%: {n50} pannelli su {len(tg3)}; dentro la fascia all'80%: {n80} su {len(tg3)}. "
            f"Etichette della chiusura non neutre: {nn} su {4 * len(tg3)}, corrette {okk}.")
    grafici.g3(pan, FIG / "g3_verifica_passato.png", riga, mediana=True)
    f1 = frames[1]
    info = {"costruzione": KA.COSTRUZIONE, "nome": KA.NOME, "modello_direzione": "L0", "scelte": {c: {str(h): v for h, v in d.items()} for c, d in scelte.items()},
            "regola_ampiezze": f"regressione solo se errore quadratico medio più basso della media storica e Diebold-Mariano con p < {KA.P_BATTE}",
            "etichetta": {"modello": "D1", "neutra_tra": list(KA.NEUTRA), "riferimento": "chiusura all'origine"},
            "rapporto_apertura_ultimo": float(f1["rapporto_apertura"].iloc[-1]), "frequenza_rialzo_corpo_ultima": float(f1["p_L0"].iloc[-1]),
            "frequenza_rialzo_corpo_min_max": [float(f1["p_L0"].min()), float(f1["p_L0"].max())],
            "per_orizzonte": {str(h): {"n_previsioni": int(len(f)), "prima_data_obiettivo": str(f["data_obiettivo"].iloc[0].date()),
                                       "ultima_data_obiettivo": str(f["data_obiettivo"].iloc[-1].date())} for h, f in frames.items()},
            "g3": {"riga": riga, "dentro_50": n50, "dentro_80": n80, "etichette_non_neutre": nn, "etichette_corrette": okk, "n": int(len(tg3)),
                   "regola": f"{VB.G3_N} origini, una ogni {VB.G3_PASSO} settimane a ritroso a partire da {VB.G3_PRIMA} settimane prima dell'ultima data di sviluppo"}}
    (OUT / "p0_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with K.guardia_dati():
        run()
