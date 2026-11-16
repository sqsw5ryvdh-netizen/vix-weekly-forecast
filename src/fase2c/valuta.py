"""Fase 2c: walk-forward di sviluppo per le componenti (Parte A), pagella delle candele (Parte C), regola di scelta, G3.

Legge il file di sviluppo (sotto la guardia) e, per K0 e per la mediana D4, le previsioni gia' prodotte da
`python -m src.run_phase2b` (reports/fase2b/previsioni_dev.csv): stesse origini, nessun ricalcolo.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import baselines as B
from .. import config as C
from ..fase1 import comune as K
from ..fase1.valuta import holm
from ..fase2 import valuta as V2
from ..fase2b import dati, grafici
from ..fase2b import valuta as VB
from . import candele as KA
from . import componenti as CP

OUT = C.REPORTS_DIR / "fase2c"
FIG = C.FIG_DIR / "fase2c"
RIALZO_FORTE = 0.30                 # "rialzo oltre il 30% in 4 settimane": chiusura_t / chiusura_{t-4} - 1 > 0,30
VINCOLO = (0.8, 1.2)                # regola di scelta: rapporto del corpo e dell'escursione a h = 1
CAMPI = ("apertura", "massimo", "minimo", "chiusura")
REALI = {"apertura": "open", "massimo": "high", "minimo": "low", "chiusura": "close"}


# ------------------------------------------------------------------ walk-forward

def walk(df: pd.DataFrame) -> tuple[dict[int, pd.DataFrame], dict]:
    S = CP.serie(df)
    pos = {pd.Timestamp(d): i for i, d in enumerate(df["date"])}
    p2b = pd.read_csv(VB.OUT / "previsioni_dev.csv", parse_dates=["data_origine", "data_obiettivo"])
    comp: dict[int, dict[int, dict]] = {}
    frames = {}
    for h in CP.ORIZZONTI:
        g = p2b[p2b["h"] == h].reset_index(drop=True)
        o = np.array([pos[d] for d in g["data_origine"]])
        rows = []
        for t in o.tolist():
            c = CP.componenti(S, t, h)
            comp.setdefault(t, {})[h] = c
            rows.append(c)
        f = pd.DataFrame(rows)
        f.insert(0, "idx_origine", o)
        f.insert(0, "h", h)
        f["data_origine"], f["data_obiettivo"] = g["data_origine"], g["data_obiettivo"]
        f["chiusura_origine"] = S.c[o]
        for k, col in REALI.items():
            f[f"{k}_reale"] = getattr(S, col[0])[o + h]
        f["su_reale"] = (f["chiusura_reale"] > f["apertura_reale"]).astype(float)
        f["corpo_reale"] = np.abs(np.log(f["chiusura_reale"] / f["apertura_reale"]))
        f["ombra_sup_reale"] = np.log(f["massimo_reale"] / np.maximum(f["apertura_reale"], f["chiusura_reale"]))
        f["ombra_inf_reale"] = np.log(np.minimum(f["apertura_reale"], f["chiusura_reale"]) / f["minimo_reale"])
        f["mediana_d4"] = g["mediana_d4"]
        f["K0_apertura"], f["K0_massimo"], f["K0_minimo"], f["K0_chiusura"] = g["apertura_prevista"], g["massimo_previsto"], g["minimo_previsto"], g["mediana_d4"]
        f["fascia_origine"] = np.array(C.BAND_LABELS)[B.band_index(f["chiusura_origine"])]
        f["rialzo_forte"] = S.c[o] / S.c[o - 4] - 1 > RIALZO_FORTE
        yr = f["data_obiettivo"].dt.year
        f["sottoperiodo"] = ""
        for a, b in C.SUBPERIODS:
            f.loc[(yr >= a) & (yr <= b), "sottoperiodo"] = f"{a}-{b}"
        frames[h] = f
    return frames, comp


def scegli_direzione(frames: dict[int, pd.DataFrame]) -> dict[int, str]:
    """Per ogni orizzonte, il modello di direzione con il Brier piu' basso nel walk-forward di sviluppo."""
    return {h: min(CP.MODELLI_DIREZIONE, key=lambda m: float(((f[f"p_{m}"] - f["su_reale"]) ** 2).mean())) for h, f in frames.items()}


def aggiungi_candele(df: pd.DataFrame, frames: dict[int, pd.DataFrame], comp: dict, scelta_dir: dict[int, str]) -> None:
    S = CP.serie(df)
    med = {h: dict(zip(f["idx_origine"].tolist(), f["mediana_d4"].tolist())) for h, f in frames.items()}
    cache: dict[int, dict[str, list[dict]]] = {}
    for h, f in frames.items():
        cols = {f"{k}_{c}": [] for k in ("K1", "K2", "K3") for c in CAMPI}
        for t in f["idx_origine"].tolist():
            if t not in cache:
                hs = sorted(comp[t])
                cp = {k: comp[t][k] for k in hs}
                pd_ = {k: cp[k][f"p_{scelta_dir[k]}"] for k in hs}
                cache[t] = {"K1": KA.k1(S.c[t], cp, pd_), "K2": KA.k2(S.c[t], {k: med[k][t] for k in hs if t in med[k]}, cp, pd_),
                            "K3": KA.k3(CP.Serie(S.o[: t + 1], S.h[: t + 1], S.l[: t + 1], S.c[: t + 1], S.date[: t + 1]), t)[0]}
            for k in ("K1", "K2", "K3"):
                for c in CAMPI:
                    cols[f"{k}_{c}"].append(cache[t][k][h - 1][c])
        for name, v in cols.items():
            f[name] = v
        f["p_scelta"] = f[f"p_{scelta_dir[h]}"]


# ------------------------------------------------------------------ metriche

def direzione_tabelle(frames: dict, scelta: dict) -> dict[str, pd.DataFrame]:
    main, grp, cal = [], [], []
    for h, f in frames.items():
        y = f["su_reale"].to_numpy()
        l0 = (f["p_L0"].to_numpy() - y) ** 2
        for m in CP.MODELLI_DIREZIONE:
            p = f[f"p_{m}"].to_numpy()
            row = {"h": h, "modello": m, "n": len(y), "frequenza_rialzo": y.mean(), "prob_media": p.mean(),
                   "accuratezza": float(((p >= 0.5) == (y == 1)).mean())} | V2.brier_parts(p, y)
            row["brier_skill_vs_L0"] = 1 - row["brier"] / l0.mean()
            row["dm_stat"], row["p"] = (np.nan, np.nan) if m == "L0" else V2.dm_loss(l0, (p - y) ** 2, h)
            row["scelto"] = scelta[h] == m
            main.append(row)
            b = np.minimum((p * 10).astype(int), 9)
            for k in np.unique(b):
                mk = b == k
                cal.append({"h": h, "modello": m, "classe": f"{k / 10:.1f}-{(k + 1) / 10:.1f}", "n": int(mk.sum()), "prob_media": p[mk].mean(), "freq_osservata": y[mk].mean()})
            for col in ("fascia_origine", "rialzo_forte", "sottoperiodo"):
                for key, g in f.groupby(col, sort=False):
                    pp, yy = g[f"p_{m}"].to_numpy(), g["su_reale"].to_numpy()
                    b0 = ((g["p_L0"].to_numpy() - yy) ** 2).mean()
                    grp.append({"h": h, "gruppo": col, "valore": str(key), "modello": m, "n": len(g), "frequenza_rialzo": yy.mean(),
                                "accuratezza": float(((pp >= 0.5) == (yy == 1)).mean()), "brier": ((pp - yy) ** 2).mean(),
                                "brier_skill_vs_L0": 1 - ((pp - yy) ** 2).mean() / b0 if b0 > 0 else np.nan})
    main = pd.DataFrame(main)
    t = main["p"].notna()
    main["p_holm"] = np.nan
    main.loc[t, "p_holm"] = holm(main.loc[t, "p"].to_numpy())
    return {"direzione": main, "direzione_gruppi": pd.DataFrame(grp), "direzione_calibrazione": pd.DataFrame(cal)}


def ampiezze_tabella(frames: dict) -> pd.DataFrame:
    rows = []
    for h, f in frames.items():
        bl = K.blocks(f["idx_origine"].to_numpy())
        for nome in ("corpo", "ombra_sup", "ombra_inf"):
            y = f[f"{nome}_reale"].to_numpy()
            e_m, e_r = np.abs(y - f[nome].to_numpy()), np.abs(y - f[f"{nome}_rif"].to_numpy())
            lo, hi = K.boot_cluster(bl, lambda i: e_r[i].mean() - e_m[i].mean(), f"f2camp|{h}|{nome}")
            dm, p = V2.dm_loss(e_r, e_m, h)
            rows.append({"h": h, "componente": nome, "n": len(y), "media_reale": y.mean(), "mediana_reale": float(np.median(y)), "previsione_media": f[nome].mean(),
                         "mae_modello": e_m.mean(), "mae_riferimento": e_r.mean(), "guadagno": e_r.mean() - e_m.mean(), "guadagno_ci_lo": lo, "guadagno_ci_hi": hi,
                         "guadagno_pct": 1 - e_m.mean() / e_r.mean(), "dm_stat": dm, "p": p})
    return pd.DataFrame(rows)


def pagella_riga(g: pd.DataFrame, k: str, tag: str | None = None) -> dict:
    o, hi, lo, c = (g[f"{k}_{x}"].to_numpy() for x in CAMPI)
    ro, rh, rl, rc = (g[f"{x}_reale"].to_numpy() for x in CAMPI)
    c0 = g["chiusura_origine"].to_numpy()
    r = {"n": len(g)}
    okb = rc != ro
    r["direzione_corpo"] = float((np.sign(c - o)[okb] == np.sign(rc - ro)[okb]).mean())
    okc = rc != c0
    r["direzione_chiusura_vs_origine"] = float((np.sign(c - c0)[okc] == np.sign(rc - c0)[okc]).mean())
    for nome, p, re in zip(CAMPI, (o, hi, lo, c), (ro, rh, rl, rc)):
        r[f"mae_{nome}"] = float(np.abs(p - re).mean())
        r[f"mae_pct_{nome}"] = float((np.abs(p - re) / re).mean())
    r["rapporto_corpo"] = float(np.abs(c - o).mean() / np.abs(rc - ro).mean())
    r["rapporto_escursione"] = float((hi - lo).mean() / (rh - rl).mean())
    iou = np.maximum(0.0, np.minimum(hi, rh) - np.maximum(lo, rl)) / (np.maximum(hi, rh) - np.minimum(lo, rl))
    r["sovrapposizione"] = float(iou.mean())
    r["chiusura_nel_corpo"] = float(((rc >= np.minimum(o, c)) & (rc <= np.maximum(o, c))).mean())
    r["chiusura_nell_escursione"] = float(((rc >= lo) & (rc <= hi)).mean())
    r["entro_10pct"], r["entro_20pct"] = float((np.abs(c / rc - 1) <= 0.10).mean()), float((np.abs(c / rc - 1) <= 0.20).mean())
    if tag:
        bl = K.blocks(g["idx_origine"].to_numpy())
        r["sovrapposizione_ci_lo"], r["sovrapposizione_ci_hi"] = K.boot_cluster(bl, lambda i: iou[i].mean(), tag)
    return r


def pagella_tabelle(frames: dict) -> dict[str, pd.DataFrame]:
    main, grp, entro = [], [], []
    for h, f in frames.items():
        for k in KA.COSTRUZIONI:
            main.append({"costruzione": k, "h": h} | pagella_riga(f, k, f"f2ciou|{k}|{h}"))
            for col in ("fascia_origine", "rialzo_forte", "sottoperiodo"):
                for key, g in f.groupby(col, sort=False):
                    grp.append({"gruppo": col, "valore": str(key), "costruzione": k, "h": h} | pagella_riga(g, k))
        rc = f["chiusura_reale"].to_numpy()
        for nome, col in [("mediana D4", "mediana_d4")] + [(k, f"{k}_chiusura") for k in KA.COSTRUZIONI]:
            p = f[col].to_numpy()
            entro.append({"h": h, "serie": nome, "n": len(f), "entro_10pct": float((np.abs(p / rc - 1) <= 0.10).mean()),
                          "entro_20pct": float((np.abs(p / rc - 1) <= 0.20).mean()), "mae_punti": float(np.abs(p - rc).mean())})
    return {"pagella": pd.DataFrame(main), "pagella_gruppi": pd.DataFrame(grp), "chiusura_entro": pd.DataFrame(entro)}


def scegli_costruzione(pag: pd.DataFrame) -> dict:
    """Regola fissata prima: tra le costruzioni con rapporto del corpo e dell'escursione in [0,8; 1,2] a h = 1,
    quella con la sovrapposizione media piu' alta su h = 1-4; se nessuna rispetta il vincolo, quella con il rapporto
    del corpo (a h = 1) piu' vicino a 1."""
    h1 = pag[pag["h"] == 1].set_index("costruzione")
    iou = pag.groupby("costruzione")["sovrapposizione"].mean()
    ok = [k for k in KA.COSTRUZIONI if VINCOLO[0] <= h1.loc[k, "rapporto_corpo"] <= VINCOLO[1] and VINCOLO[0] <= h1.loc[k, "rapporto_escursione"] <= VINCOLO[1]]
    spareggio = None
    if ok:
        scelta, motivo = max(ok, key=lambda k: iou[k]), "vincolo rispettato; sovrapposizione media più alta su h = 1-4 tra le costruzioni ammesse"
    else:
        dist = {k: abs(h1.loc[k, "rapporto_corpo"] - 1) for k in KA.COSTRUZIONI}
        pari = [k for k in KA.COSTRUZIONI if dist[k] - min(dist.values()) < 1e-9]
        scelta = max(pari, key=lambda k: iou[k])            # parita' esatta: decide il criterio principale della regola (sovrapposizione media)
        motivo = "nessuna costruzione rispetta il vincolo: scelta quella con il rapporto del corpo a h = 1 più vicino a 1"
        if len(pari) > 1:
            spareggio = {"in_parita": pari, "criterio": "sovrapposizione media su h = 1-4 (criterio principale della regola)", "scelta": scelta}
    return {"costruzione": scelta, "motivo": motivo, "ammesse": ok, "spareggio": spareggio, "vincolo": list(VINCOLO),
            "rapporto_corpo_h1": {k: float(h1.loc[k, "rapporto_corpo"]) for k in KA.COSTRUZIONI},
            "rapporto_escursione_h1": {k: float(h1.loc[k, "rapporto_escursione"]) for k in KA.COSTRUZIONI},
            "sovrapposizione_media": {k: float(iou[k]) for k in KA.COSTRUZIONI}}


# ------------------------------------------------------------------ figure

def figura_pagella(pag: pd.DataFrame) -> None:
    with plt.rc_context(grafici.STILE):
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
        col = {"K0": "#7f8c8d", "K1": "#b03a2e", "K2": "#1f4e79", "K3": "#d68910"}
        for ax, (m, tit, ref) in zip(axes, (("rapporto_corpo", "Corpo previsto / corpo reale", 1.0), ("rapporto_escursione", "Escursione prevista / reale", 1.0),
                                            ("sovrapposizione", "Sovrapposizione media (0-1)", None), ("mae_chiusura", "Errore assoluto medio sulla chiusura (punti)", None))):
            for j, k in enumerate(KA.COSTRUZIONI):
                g = pag[pag["costruzione"] == k]
                ax.bar(g["h"] + (j - 1.5) * 0.2, g[m], 0.2, color=col[k], label=KA.NOMI[k])
            if ref:
                ax.axhline(ref, color="black", lw=0.8, ls="--")
                ax.axhspan(VINCOLO[0], VINCOLO[1], color="#8e7cc3", alpha=0.12, lw=0)
            ax.set_xticks([1, 2, 3, 4], [f"+{h}" for h in (1, 2, 3, 4)])
            ax.set_title(tit, fontsize=9)
            ax.set_xlabel("settimana")
        axes[0].legend(frameon=False, fontsize=7.5)
        fig.tight_layout()
        fig.savefig(FIG / "pagella.png")
        plt.close(fig)


def pannelli_g3(df: pd.DataFrame, costruzione: str, modello_dir: dict[int, str]) -> tuple[list[dict], pd.DataFrame]:
    """Stesse 12 origini della Fase 2b; candele della costruzione scelta, fasce D4 e linea della mediana."""
    n = len(df)
    out, rows = [], []
    for k in range(VB.G3_N):
        t = n - 1 - VB.G3_PRIMA - VB.G3_PASSO * k
        prev, cand = KA.prevedi_candele(df, t, costruzione, modello_dir)
        vista = prev | {"settimane": [s | {c: x[c] for c in CAMPI} | {"direzione": x["direzione"], "prob_direzione": x["prob_direzione"]}
                                      for s, x in zip(prev["settimane"], cand)]}
        reali = df.iloc[t + 1:t + 5]
        s4, y4 = prev["settimane"][-1], float(reali["close"].iloc[-1])
        d50, d80 = s4["fasce"]["50"][0] <= y4 <= s4["fasce"]["50"][1], s4["fasce"]["80"][0] <= y4 <= s4["fasce"]["80"][1]
        dirs = [(x["chiusura"] >= x["apertura"]) == (r.close >= r.open) for x, r in zip(cand, reali.itertuples())]
        out.append({"previsione": vista, "storia": df.iloc[t - VB.G3_STORIA + 1:t + 1], "reali": reali, "dentro_50": d50, "dentro_80": d80})
        rows.append({"data_origine": prev["data_origine"], "chiusura_origine": prev["chiusura_origine"], "mediana_d4_4": s4["chiusura"], "candela_chiusura_4": cand[-1]["chiusura"],
                     "chiusura_reale_4": y4, "dentro_50": d50, "dentro_80": d80, "direzioni_indovinate_su_4": int(sum(dirs)),
                     "chiusura_reale_4_nell_escursione_prevista": bool(cand[-1]["minimo"] <= y4 <= cand[-1]["massimo"])})
    return out[::-1], pd.DataFrame(rows[::-1])


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = dati.load_dev_candele()
    frames, comp = walk(df)
    scelta_dir = scegli_direzione(frames)
    aggiungi_candele(df, frames, comp, scelta_dir)
    tabs = direzione_tabelle(frames, scelta_dir) | {"ampiezze": ampiezze_tabella(frames)} | pagella_tabelle(frames)
    for name, t in tabs.items():
        t.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")
    pd.concat([f.drop(columns="idx_origine") for f in frames.values()], ignore_index=True).to_csv(
        OUT / "previsioni_dev.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    scelta = scegli_costruzione(tabs["pagella"])
    figura_pagella(tabs["pagella"])
    pan, tg3 = pannelli_g3(df, scelta["costruzione"], scelta_dir)
    tg3.to_csv(OUT / "g3_pannelli.csv", index=False, float_format="%.4f")
    n50, n80, nd = int(tg3["dentro_50"].sum()), int(tg3["dentro_80"].sum()), int(tg3["direzioni_indovinate_su_4"].sum())
    riga = (f"Chiusura reale a 4 settimane dentro la fascia al 50%: {n50} pannelli su {len(tg3)}; dentro la fascia all'80%: {n80} su {len(tg3)}. "
            f"Direzione del corpo indovinata: {nd} candele su {4 * len(tg3)}.")
    grafici.g3(pan, FIG / "g3_verifica_passato.png", riga, mediana=True)
    S = CP.serie(df)
    f1 = frames[1]
    info = {"campione": [str(df["date"].min().date()), str(df["date"].max().date())], "n_settimane": int(len(df)),
            "settimane_escluse_dalle_stime": [{"data": str(pd.Timestamp(d).date()), "open": float(o), "low": float(l), "high": float(h)}
                                              for d, o, l, h in zip(S.date[S.bad], S.o[S.bad], S.l[S.bad], S.h[S.bad])],
            "modello_direzione": {str(h): m for h, m in scelta_dir.items()}, "regola_direzione": "per ogni orizzonte, Brier più basso nel walk-forward di sviluppo",
            "scelta": scelta, "rapporto_apertura": {"min": float(f1["rapporto_apertura"].min()), "max": float(f1["rapporto_apertura"].max()), "ultimo": float(f1["rapporto_apertura"].iloc[-1])},
            "rialzo_forte_soglia": RIALZO_FORTE, "n_rialzo_forte": {str(h): int(f["rialzo_forte"].sum()) for h, f in frames.items()},
            "per_orizzonte": {str(h): {"n_previsioni": int(len(f)), "prima_data_obiettivo": str(f["data_obiettivo"].iloc[0].date()),
                                       "ultima_data_obiettivo": str(f["data_obiettivo"].iloc[-1].date())} for h, f in frames.items()},
            "g3": {"riga": riga, "dentro_50": n50, "dentro_80": n80, "direzioni_indovinate": nd, "n_candele": int(4 * len(tg3)), "n": int(len(tg3)), "regola": VB.G3_N and
                   f"{VB.G3_N} origini, una ogni {VB.G3_PASSO} settimane a ritroso a partire da {VB.G3_PRIMA} settimane prima dell'ultima data di sviluppo"}}
    (OUT / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with K.guardia_dati():
        run()
