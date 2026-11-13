"""Fase 2, Parte A: valutazione walk-forward dei modelli di densita' e delle probabilita' degli eventi.

Tutti i modelli sono valutati sullo stesso insieme di origini per ciascun orizzonte (da `first_origin(h)`).
CRPS = 2 x media della perdita quantilica sulla griglia comune di 19 quantili (0,05 ... 0,95).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config as C
from ..evaluate import diebold_mariano
from ..fase1 import comune as K
from ..fase1 import modelli as M
from ..fase1.valuta import add_labels, holm
from . import densita as D

OUT = C.REPORTS_DIR / "fase2"
FIG = C.FIG_DIR / "fase2"
REF_D, REF_E = "D1", "RIF"
QI = [int(np.argmin(np.abs(D.TAUS - q))) for q in D.Q_REPORT]          # posizioni dei 7 quantili nella griglia
PIT_EDGES = (0.0,) + D.Q_REPORT + (1.0,)
PIT_LABELS = [f"{int(100 * a)}-{int(100 * b)}%" for a, b in zip(PIT_EDGES[:-1], PIT_EDGES[1:])]
N_BINS = 10


def pinball(y: np.ndarray, q: np.ndarray, taus) -> np.ndarray:
    d = y[:, None] - q
    return np.maximum(np.asarray(taus) * d, (np.asarray(taus) - 1) * d)


def dm_loss(loss_ref: np.ndarray, loss_mod: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano su perdite non negative (positivo = il modello ha perdita minore del riferimento)."""
    return diebold_mariano(np.sqrt(loss_ref), np.sqrt(loss_mod), h)


def brier_parts(p: np.ndarray, y: np.ndarray) -> dict:
    """Brier e scomposizione di Murphy su 10 classi di probabilita': Brier ~ affidabilita' - risoluzione + incertezza."""
    b = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
    ob = y.mean()
    rel = res = 0.0
    for k in np.unique(b):
        m = b == k
        rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2
        res += m.sum() * (y[m].mean() - ob) ** 2
    return {"brier": float(((p - y) ** 2).mean()), "affidabilita": rel / len(y), "risoluzione": res / len(y),
            "incertezza": float(ob * (1 - ob))}


# ------------------------------------------------------------------ previsioni

def build(df: pd.DataFrame, horizons=K.HORIZONS, models=tuple(D.DENSITY)) -> dict:
    """Esegue il walk-forward di tutti i modelli. Restituisce per ogni h: base, quantili (griglia), eventi."""
    x, close, dates = df["log_close"].to_numpy(), df["close"].to_numpy(), df["date"].to_numpy()
    res = {}
    for h in horizons:
        c1 = D.centers(x, M.forecast_har, h)
        c7 = D.centers(x, M.forecast_combo, h) if "D1b" in models else None
        o = np.arange(D.first_origin(h), len(x) - h)
        base = pd.DataFrame({"h": h, "idx_origine": o, "data_origine": dates[o], "data_obiettivo": dates[o + h],
                             "x0": x[o], "vix0": close[o], "y": x[o + h]})
        evs = D.events_for(h)
        thr = {e: np.array([D.event_threshold(e, v) for v in x[o]]) for e in evs}
        Q, P, crossed = {}, {}, {}
        for name in models:
            _, ds = D.density_walk(x, name, h, c1, c7)
            if ds[0] is None:
                continue
            Q[name] = np.array([d.quantiles() for d in ds])
            crossed[name] = int(sum(d.crossed for d in ds))
            P[name] = {e: np.array([d.prob_above(th) for d, th in zip(ds, thr[e])]) for e in evs}
        P["E1"] = {e: np.array([D.e1(x[: t + 1], h, e) for t in o]) for e in evs}
        P[REF_E] = {e: np.array([D.reference_prob(x[: t + 1], h, e) for t in o]) for e in evs}
        outcome = {e: (x[o + h] > thr[e]).astype(float) for e in evs}
        res[h] = {"base": base, "Q": Q, "P": P, "esito": outcome, "incroci": crossed}
    return res


# ------------------------------------------------------------------ metriche

def density_row(q: np.ndarray, g: pd.DataFrame, with_ci: bool = False, tag: str = "") -> dict:
    y = g["y"].to_numpy()
    pb = pinball(y, q, D.TAUS)
    r = {"n": len(y), "crps": 2 * pb.mean(), "perdita_quantilica_media": pb[:, QI].mean()}
    for j, tau in zip(QI, D.Q_REPORT):
        r[f"perdita_q{int(100 * tau):02d}"] = pb[:, j].mean()
    bl = K.blocks(g["idx_origine"].to_numpy())
    for lev, (a, b) in D.INTERVALS.items():
        lo, hi = q[:, int(np.argmin(np.abs(D.TAUS - a)))], q[:, int(np.argmin(np.abs(D.TAUS - b)))]
        inside = (y >= lo) & (y <= hi)
        r |= {f"copertura_{lev}": inside.mean(), f"sopra_{lev}": (y > hi).mean(), f"sotto_{lev}": (y < lo).mean(),
              f"larghezza_log_{lev}": (hi - lo).mean(), f"larghezza_vix_{lev}": (np.exp(hi) - np.exp(lo)).mean()}
        if with_ci:
            r[f"copertura_{lev}_ci_lo"], r[f"copertura_{lev}_ci_hi"] = K.boot_cluster(bl, lambda i: inside[i].mean(), f"{tag}|{lev}")
    med = q[:, int(np.argmin(np.abs(D.TAUS - 0.5)))]
    r |= {"mediana_mae_log": np.abs(y - med).mean(), "mediana_mae_vix": np.abs(np.exp(y) - np.exp(med)).mean(),
          "mediana_rmse_vix": np.sqrt(((np.exp(y) - np.exp(med)) ** 2).mean())}
    return r


def density_tables(res: dict, lab: dict) -> dict[str, pd.DataFrame]:
    main, pit, groups = [], [], {"sottoperiodo": [], "fascia_origine": [], "dopo_picco": [], "periodo_calmo": []}
    for h, r in res.items():
        g_all = lab[h]
        crps = {m: 2 * pinball(g_all["y"].to_numpy(), q, D.TAUS).mean(axis=1) for m, q in r["Q"].items()}
        for m, q in r["Q"].items():
            row = {"h": h, "modello": m} | density_row(q, g_all, True, f"f2cov|{h}|{m}") | {"quantili_incrociati": r["incroci"][m]}
            row["crps_vs_d1"] = 1 - row["crps"] / (2 * pinball(g_all["y"].to_numpy(), r["Q"][REF_D], D.TAUS).mean())
            row["dm_stat"], row["p"] = (np.nan, np.nan) if m == REF_D else dm_loss(crps[REF_D], crps[m], h)
            main.append(row)
            y = g_all["y"].to_numpy()
            cnt = (y[:, None] > q[:, QI]).sum(axis=1)                 # 0..7: classe PIT
            for k, labk in enumerate(PIT_LABELS):
                pit.append({"h": h, "modello": m, "classe": labk, "frequenza": (cnt == k).mean(),
                            "attesa": PIT_EDGES[k + 1] - PIT_EDGES[k]})
            for col, rows in groups.items():
                for key, gg in g_all.groupby(col, sort=False):
                    pos = gg.index.to_numpy()
                    rr = {"h": h, col: key, "modello": m} | density_row(q[pos], gg)
                    rr["crps_vs_d1"] = 1 - crps[m][pos].mean() / crps[REF_D][pos].mean()
                    rr["dm_stat"], rr["p"] = (np.nan, np.nan) if m == REF_D else dm_loss(crps[REF_D][pos], crps[m][pos], h)
                    rows.append(rr)
    main = pd.DataFrame(main)
    t = main["p"].notna()
    main["p_holm"] = np.nan
    main.loc[t, "p_holm"] = holm(main.loc[t, "p"].to_numpy())
    out = {"a_densita_metriche": main, "a_pit": pd.DataFrame(pit)}
    for col, rows in groups.items():
        out[f"a_densita_{col}"] = pd.DataFrame(rows)
    return out


def event_tables(res: dict, lab: dict) -> dict[str, pd.DataFrame]:
    main, cal, groups = [], [], {"sottoperiodo": [], "fascia_origine": [], "dopo_picco": [], "periodo_calmo": []}
    for h, r in res.items():
        g_all = lab[h]
        for e, y in r["esito"].items():
            ref = r["P"][REF_E][e]
            for m, pe in r["P"].items():
                p = pe[e]
                row = {"evento": e, "h": h, "modello": m, "n": len(y), "frequenza_evento": y.mean(), "prob_media": p.mean()} | brier_parts(p, y)
                row["brier_skill_vs_rif"] = 1 - row["brier"] / ((ref - y) ** 2).mean()
                row["dm_stat"], row["p"] = (np.nan, np.nan) if m == REF_E else dm_loss((ref - y) ** 2, (p - y) ** 2, h)
                main.append(row)
                b = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
                for k in np.unique(b):
                    mk = b == k
                    cal.append({"evento": e, "h": h, "modello": m, "classe": f"{k / N_BINS:.1f}-{(k + 1) / N_BINS:.1f}",
                                "n": int(mk.sum()), "prob_media": p[mk].mean(), "freq_osservata": y[mk].mean()})
                for col, rows in groups.items():
                    for key, gg in g_all.groupby(col, sort=False):
                        pos = gg.index.to_numpy()
                        bs, bs_ref = ((p[pos] - y[pos]) ** 2).mean(), ((ref[pos] - y[pos]) ** 2).mean()
                        rows.append({"evento": e, "h": h, col: key, "modello": m, "n": len(pos), "frequenza_evento": y[pos].mean(),
                                     "prob_media": p[pos].mean(), "brier": bs, "brier_skill_vs_rif": 1 - bs / bs_ref if bs_ref > 0 else np.nan})
    main = pd.DataFrame(main)
    t = main["p"].notna()
    main["p_holm"] = np.nan
    main.loc[t, "p_holm"] = holm(main.loc[t, "p"].to_numpy())
    out = {"a_eventi_metriche": main, "a_calibrazione": pd.DataFrame(cal)}
    for col, rows in groups.items():
        out[f"a_eventi_{col}"] = pd.DataFrame(rows)
    return out


def choose(main: pd.DataFrame) -> dict[int, str]:
    """Regola fissata prima: per ogni orizzonte il modello (tra D1-D6) con il CRPS medio piu' basso."""
    m = main[main["modello"].isin(D.SELECTABLE)]
    return {int(h): g.sort_values("crps").iloc[0]["modello"] for h, g in m.groupby("h")}


# ------------------------------------------------------------------ figure

def figures(res: dict, tabs: dict, scelta: dict) -> None:
    pit, main, cal = tabs["a_pit"], tabs["a_densita_metriche"], tabs["a_calibrazione"]
    models = [m for m in D.DENSITY]
    fig, axes = plt.subplots(len(res), len(models), figsize=(2.3 * len(models), 2.3 * len(res)), sharey=True)
    for i, h in enumerate(res):
        for j, m in enumerate(models):
            ax = axes[i, j]
            g = pit[(pit["h"] == h) & (pit["modello"] == m)]
            if g.empty:
                ax.set_axis_off()
                continue
            ax.bar(range(len(g)), g["frequenza"] / g["attesa"], color="#1f4e79", alpha=0.85)
            ax.axhline(1, color="#b03a2e", lw=0.8, ls="--")
            ax.set_xticks(range(len(g)), [c.split("-")[0] for c in g["classe"]], fontsize=6)
            ax.set_title(f"{m}, h = {h}", fontsize=8)
            if j == 0:
                ax.set_ylabel("freq. osservata / attesa", fontsize=7)
    fig.suptitle("Istogramma PIT per classi di quantile (1 = calibrazione perfetta; etichetta = limite inferiore della classe, %)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "a_pit.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(res), figsize=(4.6 * len(res), 3.8), sharey=True)
    w = 0.8 / len(models)
    for ax, h in zip(axes, res):
        g = main[main["h"] == h].set_index("modello")
        for j, m in enumerate(models):
            if m not in g.index:
                continue
            for k, lev in enumerate(D.INTERVALS):
                v = g.loc[m, f"copertura_{lev}"]
                ax.bar(k + (j - len(models) / 2 + 0.5) * w, v, w, color=plt.cm.tab10(j), label=m if k == 0 else None,
                       yerr=[[v - g.loc[m, f"copertura_{lev}_ci_lo"]], [g.loc[m, f"copertura_{lev}_ci_hi"] - v]], ecolor="#555", capsize=1.5)
        for k, lev in enumerate(D.INTERVALS):
            ax.plot([k - 0.45, k + 0.45], [lev / 100] * 2, color="black", lw=1.2, ls="--")
        ax.set_xticks(range(3), [f"{lev}%" for lev in D.INTERVALS])
        ax.set_title(f"Copertura degli intervalli, h = {h} (tratteggio = nominale)", fontsize=9)
        ax.set_ylim(0.3, 1.0)
    axes[0].legend(frameon=False, ncol=4, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "a_copertura.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(res), figsize=(4.8 * len(res), 3.6))
    for ax, (h, r) in zip(axes, res.items()):
        y = r["base"]["y"].to_numpy()
        ref = 2 * pinball(y, r["Q"][REF_D], D.TAUS).mean(axis=1)
        for j, (m, q) in enumerate(r["Q"].items()):
            if m == REF_D:
                continue
            ax.plot(pd.to_datetime(r["base"]["data_obiettivo"]), (ref - 2 * pinball(y, q, D.TAUS).mean(axis=1)).cumsum(),
                    label=m, lw=1, color=plt.cm.tab10(models.index(m)))
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"Σ (CRPS_D1 − CRPS_modello), h = {h} (sopra zero: meglio di D1)", fontsize=9)
    axes[0].legend(frameon=False, ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "a_crps_cumulato_vs_d1.png")
    plt.close(fig)

    panels = [(e, h) for h in (1, 4) for e in D.events_for(h)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, (e, h) in zip(axes.ravel(), panels):
        ax.plot([0, 1], [0, 1], color="#7f8c8d", ls="--", lw=0.8)
        for m, col in ((REF_E, "#7f8c8d"), ("D1", "#1f4e79"), (scelta[h], "#d68910"), ("E1", "#b03a2e")):
            g = cal[(cal["evento"] == e) & (cal["h"] == h) & (cal["modello"] == m) & (cal["n"] >= 5)]
            ax.plot(g["prob_media"], g["freq_osservata"], "o-", ms=3, lw=1, color=col, label=D.NAMES.get(m, m).split(" ")[0] if m != REF_E else "riferimento")
        ax.set_title(f"{e}, h = {h}", fontsize=9)
        ax.set_xlabel("probabilità prevista")
        ax.set_ylabel("frequenza osservata")
        ax.legend(frameon=False, fontsize=7)
    for ax in axes.ravel()[len(panels):]:
        ax.set_axis_off()
    fig.suptitle("Curve di calibrazione degli eventi (classi con almeno 5 previsioni)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "a_calibrazione_eventi.png")
    plt.close(fig)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    df = K.load_dev()
    res = build(df)
    lab = {h: add_labels(r["base"], df).reset_index(drop=True) for h, r in res.items()}
    tabs = density_tables(res, lab) | event_tables(res, lab)
    for name, t in tabs.items():
        t.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")
    scelta = choose(tabs["a_densita_metriche"])

    wide, ev = [], []
    for h, r in res.items():
        b = r["base"].drop(columns="idx_origine").copy()
        y = b["y"].to_numpy()
        for m, q in r["Q"].items():
            for j, tau in zip(QI, D.Q_REPORT):
                b[f"{m}_q{int(100 * tau):02d}"] = q[:, j]
            b[f"{m}_crps"] = 2 * pinball(y, q, D.TAUS).mean(axis=1)
        wide.append(b)
        for e, yb in r["esito"].items():
            t = r["base"][["h", "data_origine", "data_obiettivo"]].copy()
            t["evento"], t["esito"] = e, yb
            for m, pe in r["P"].items():
                t[m] = pe[e]
            ev.append(t)
    pd.concat(wide, ignore_index=True).to_csv(OUT / "a_quantili.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    pd.concat(ev, ignore_index=True).to_csv(OUT / "a_eventi_previsioni.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    figures(res, tabs, scelta)

    info = {"centro": "M1 (tipo HAR)", "griglia_quantili": [float(t) for t in D.TAUS], "quantili_riportati": list(D.Q_REPORT),
            "crps": "2 x media della perdita quantilica sulla griglia di 19 quantili", "min_errori_passati": D.MIN_ERR, "regola_casi": D.MIN_CELL,
            "d5_orizzonti": list(D.D5_HORIZONS), "garch": {"stime": D.STATS["garch_stime"], "fallite": D.STATS["garch_fallite"]},
            "arma_m7": dict(M.ARMA_STATS), "logit_ridge": D.LOGIT_RIDGE,
            "scelta_per_orizzonte": {str(h): m for h, m in scelta.items()},
            "regola_scelta": "CRPS medio piu' basso tra D1-D6 sull'intero walk-forward del campione di sviluppo",
            "per_orizzonte": {str(h): {"prima_origine": str(r["base"]["data_origine"].iloc[0])[:10],
                                       "prima_data_obiettivo": str(r["base"]["data_obiettivo"].iloc[0])[:10],
                                       "ultima_data_obiettivo": str(r["base"]["data_obiettivo"].iloc[-1])[:10],
                                       "n_previsioni": int(len(r["base"])), "incroci": r["incroci"],
                                       "n_dopo_picco": int(lab[h]["dopo_picco"].sum()), "n_periodo_calmo": int(lab[h]["periodo_calmo"].sum()),
                                       "eventi": D.events_for(h)} for h, r in res.items()}}
    (OUT / "a_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with K.guardia_dati():
        run()
