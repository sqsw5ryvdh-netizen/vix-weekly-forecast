"""Fase 3: walk-forward di sviluppo dei blocchi (G1, X1-X4, X6, X7) per T1, T2 e T3, valutazione e regole di scelta.

Ogni modello e' confrontato con il riferimento sulle STESSE origini. Il riferimento principale e' quello vero, stimato
su tutta la storia disponibile (M1 per T1; frequenza storica e logistica di base per T2; D4 per T3). Per T1 e' riportato
anche, come controllo, il confronto con M1 stimato sulla stessa finestra del blocco.
Regole di scelta fissate prima (ALFA = 5% dopo Holm):
- chiusura: tra i modelli con Clark-West significativo dopo Holm a h = 1 oppure h = 4, quello con R² medio (h = 1 e 4)
  rispetto a M1 piu' alto; se nessuno, resta M1;
- direzione: stesso criterio con il test di Diebold-Mariano sul Brier contro la logistica di base;
- T3 si esegue solo per i modelli scelti da una delle due regole.
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
from ..evaluate import diebold_mariano
from ..fase1 import comune as K
from ..fase1.valuta import clark_west, holm
from ..fase2 import densita as D
from ..fase2 import valuta as V2
from ..fase2b import grafici
from . import dati
from . import modelli as M
from . import variabili as V

OUT = C.REPORTS_DIR / "fase3"
FIG = C.FIG_DIR / "fase3"
ALFA = 0.05
SOGLIE_PROB = (0.60, 0.65, 0.70)          # fissate nella consegna
FASCIA_EVIDENZA = (13.0, 22.0)            # «fascia 13-22»: chiusura all'origine tra 13 e 22, estremi inclusi
RIALZO_FORTE = 0.30                       # chiusura_t / chiusura_{t-4} - 1 > 0,30 (come nella Fase 2c)
MIN_COPPIE_SOTTOPERIODO = 104             # per la stabilita' dei segni: regressioni separate per sottoperiodo
GRUPPI = ("sottoperiodo", "fascia_origine", "fascia_13_22", "dopo_rialzo_30")
RIF_T2 = ("FREQ", "BASE")
NOMI_RIF = {"M1": "M1 (riferimento)", "FREQ": "frequenza storica (riferimento)", "BASE": "logistica con le variabili di M1 (riferimento)", "D4": "D4 (riferimento)"}


# ------------------------------------------------------------------ previsioni

def _colonna(o_tutte: np.ndarray, o: np.ndarray, valori) -> np.ndarray:
    col = np.full(len(o_tutte), np.nan)
    col[np.searchsorted(o_tutte, o)] = valori
    return col


def previsioni(w: pd.DataFrame, VV: pd.DataFrame) -> tuple[dict[int, pd.DataFrame], list[dict], dict]:
    """Walk-forward di tutti i modelli. Restituisce per ogni h la tabella larga delle previsioni, i coefficienti
    standardizzati a ogni origine e le penalita' scelte da X6."""
    x, close, dates = w["log_close"].to_numpy(), w["close"].to_numpy(), w["date"].to_numpy()
    Z0 = np.empty((len(x), 0))
    frames, coefs, penalita = {}, [], {}
    for h in M.ORIZZONTI:
        o = M.origini(Z0, h)
        f = pd.DataFrame({"h": h, "idx_origine": o, "data_origine": dates[o], "data_obiettivo": dates[o + h], "x0": x[o], "vix0": close[o],
                          "y": x[o + h], "su": (x[o + h] > x[o]).astype(float)})
        f["f_M1"] = M.walk(x, Z0, h, M.t1)[1]
        f["p_FREQ"] = [M.t2_frequenza(x[: t + 1], h) for t in o.tolist()]
        f["p_BASE"] = M.walk(x, Z0, h, M.t2)[1]
        for m in M.BLOCCHI:
            cols = list(M.colonne(m))
            Z = VV[cols].to_numpy()
            om, r = M.walk(x, Z, h, M.t1, coef=True)
            f[f"f_{m}"] = _colonna(o, om, [a for a, _ in r])
            coefs.append({"obiettivo": "T1", "h": h, "modello": m, "variabili": list(M.NOMI_M1) + cols, "origini": om, "valori": np.array([b for _, b in r])})
            f[f"f_M1fin_{m}"] = _colonna(o, om, M.walk(x, Z, h, M.t1, usa=[])[1])
            r = M.walk(x, Z, h, M.t2, coef=True)[1]
            f[f"p_{m}"] = _colonna(o, om, [a for a, _ in r])
            coefs.append({"obiettivo": "T2", "h": h, "modello": m, "variabili": list(M.NOMI_M1) + cols, "origini": om, "valori": np.array([b for _, b in r])})
        Z = VV[list(V.TUTTE)].to_numpy()
        om, r = M.walk(x, Z, h, M.t1_ridge, dettagli=True)
        f["f_X6"] = _colonna(o, om, [a for a, _, _ in r])
        f["f_M1fin_X6"] = f["f_M1fin_X7"] = _colonna(o, om, M.walk(x, Z, h, M.t1, usa=[])[1])
        coefs.append({"obiettivo": "T1", "h": h, "modello": "X6", "variabili": list(M.NOMI_M1) + list(V.TUTTE), "origini": om, "valori": np.array([b for _, _, b in r])})
        r2 = M.walk(x, Z, h, M.t2_ridge, dettagli=True)[1]
        f["p_X6"] = _colonna(o, om, [a for a, _, _ in r2])
        coefs.append({"obiettivo": "T2", "h": h, "modello": "X6", "variabili": list(M.NOMI_M1) + list(V.TUTTE), "origini": om, "valori": np.array([b for _, _, b in r2])})
        penalita[h] = {"T1": [g for _, g, _ in r], "T2": [g for _, g, _ in r2]}
        comuni = f["f_X6"].notna()
        for pre in ("f", "p"):
            f[f"{pre}_X7"] = f[[f"{pre}_{m}" for m in M.BLOCCHI]].mean(axis=1, skipna=False).where(comuni)
        f["fascia_origine"] = np.array(C.BAND_LABELS)[B.band_index(f["vix0"])]
        yr = pd.to_datetime(f["data_obiettivo"]).dt.year
        f["sottoperiodo"] = ""
        for a, b in C.SUBPERIODS:
            f.loc[(yr >= a) & (yr <= b), "sottoperiodo"] = f"{a}-{b}"
        f["fascia_13_22"] = (f["vix0"] >= FASCIA_EVIDENZA[0]) & (f["vix0"] <= FASCIA_EVIDENZA[1])
        f["dopo_rialzo_30"] = close[o] / close[o - 4] - 1 > RIALZO_FORTE
        frames[h] = f
    return frames, coefs, penalita


# ------------------------------------------------------------------ metriche

def t1_riga(g: pd.DataFrame, m: str, h: int, tag: str | None = None) -> dict:
    g = g[g[f"f_{m}"].notna()]
    y, f, r = g["y"].to_numpy(), g[f"f_{m}"].to_numpy(), g["f_M1"].to_numpy()
    e, er = y - f, y - r
    ev, evr = np.exp(y) - np.exp(f), np.exp(y) - np.exp(r)
    rel, relr = np.abs(np.exp(f - y) - 1), np.abs(np.exp(r - y) - 1)
    row = {"n": len(g), "prima_data_obiettivo": str(pd.Timestamp(g["data_obiettivo"].iloc[0]).date()) if len(g) else "",
           "rmse_log": np.sqrt((e ** 2).mean()), "mae_log": np.abs(e).mean(), "rmse_vix": np.sqrt((ev ** 2).mean()), "mae_vix": np.abs(ev).mean(),
           "rmse_log_m1": np.sqrt((er ** 2).mean()), "mae_log_m1": np.abs(er).mean(), "rmse_vix_m1": np.sqrt((evr ** 2).mean()), "mae_vix_m1": np.abs(evr).mean(),
           "entro_10": (rel <= 0.10).mean(), "entro_20": (rel <= 0.20).mean(), "entro_10_m1": (relr <= 0.10).mean(), "entro_20_m1": (relr <= 0.20).mean()}
    if m == "M1":
        return row
    row["r2_vs_m1"] = 1 - (e ** 2).sum() / (er ** 2).sum()
    ef = y - g[f"f_M1fin_{m}"].to_numpy()
    row["r2_vs_m1_stessa_finestra"] = 1 - (e ** 2).sum() / (ef ** 2).sum()
    row["cw_stat"], row["p"] = clark_west(er, e, r, f, h)
    row["dm_stat"], row["dm_p"] = diebold_mariano(er, e, h)
    if tag:
        row["r2_ci_lo"], row["r2_ci_hi"] = K.boot_cluster(K.blocks(g["idx_origine"].to_numpy()), lambda i: 1 - (e[i] ** 2).sum() / (er[i] ** 2).sum(), tag)
    return row


def t2_riga(g: pd.DataFrame, m: str, h: int, tag: str | None = None) -> dict:
    g = g[g[f"p_{m}"].notna()]
    y, p, pf, pb = g["su"].to_numpy(), g[f"p_{m}"].to_numpy(), g["p_FREQ"].to_numpy(), g["p_BASE"].to_numpy()
    l, lf, lb = (p - y) ** 2, (pf - y) ** 2, (pb - y) ** 2
    acc = lambda q: float(((q >= 0.5) == (y == 1)).mean())
    row = {"n": len(g), "prima_data_obiettivo": str(pd.Timestamp(g["data_obiettivo"].iloc[0]).date()) if len(g) else "", "frequenza_rialzo": y.mean(), "prob_media": p.mean(),
           "accuratezza": acc(p), "accuratezza_base": acc(pb), "accuratezza_freq": acc(pf)} | V2.brier_parts(p, y) | {"brier_base": lb.mean(), "brier_freq": lf.mean()}
    conf, ok = np.maximum(p, 1 - p), (p >= 0.5) == (y == 1)
    for s in SOGLIE_PROB:
        k = conf >= s
        row |= {f"n_{int(100 * s)}": int(k.sum()), f"quota_{int(100 * s)}": float(k.mean()), f"accuratezza_{int(100 * s)}": float(ok[k].mean()) if k.any() else np.nan}
    if m != "FREQ":
        row["skill_vs_freq"] = 1 - l.mean() / lf.mean()
        row["dm_stat_freq"], row["p_freq"] = V2.dm_loss(lf, l, h)
    if m not in RIF_T2:
        row["skill_vs_base"] = 1 - l.mean() / lb.mean()
        row["dm_stat_base"], row["p_base"] = V2.dm_loss(lb, l, h)
        if tag:
            row["skill_ci_lo"], row["skill_ci_hi"] = K.boot_cluster(K.blocks(g["idx_origine"].to_numpy()), lambda i: 1 - l[i].mean() / lb[i].mean(), tag)
    return row


def _holm(t: pd.DataFrame, col_p: str, col_out: str) -> None:
    ok = t[col_p].notna()
    t[col_out] = np.nan
    t.loc[ok, col_out] = holm(t.loc[ok, col_p].to_numpy())


def tabelle(frames: dict[int, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    t1, t2, g1, g2, cal = [], [], [], [], []
    for h, f in frames.items():
        t1.append({"h": h, "modello": "M1"} | t1_riga(f, "M1", h))
        for m in RIF_T2:
            t2.append({"h": h, "modello": m} | t2_riga(f, m, h))
        for m in M.MODELLI:
            t1.append({"h": h, "modello": m} | t1_riga(f, m, h, f"f3t1|{m}|{h}"))
            t2.append({"h": h, "modello": m} | t2_riga(f, m, h, f"f3t2|{m}|{h}"))
        for m in RIF_T2 + M.MODELLI:
            fm = f[f[f"p_{m}"].notna()]
            p, y = fm[f"p_{m}"].to_numpy(), fm["su"].to_numpy()
            b = np.minimum((p * 10).astype(int), 9)
            for k in np.unique(b):
                cal.append({"h": h, "modello": m, "classe": f"{k / 10:.1f}-{(k + 1) / 10:.1f}", "n": int((b == k).sum()), "prob_media": p[b == k].mean(), "freq_osservata": y[b == k].mean()})
        for col in GRUPPI:
            for key, g in f.groupby(col, sort=False):
                for m in M.MODELLI:
                    if g[f"f_{m}"].notna().sum() == 0:
                        continue
                    tag = f"|{col}|{key}|{m}|{h}" if col in ("fascia_13_22", "dopo_rialzo_30") and key else None
                    g1.append({"h": h, "gruppo": col, "valore": str(key), "modello": m} | t1_riga(g, m, h, tag and "f3g1" + tag))
                    g2.append({"h": h, "gruppo": col, "valore": str(key), "modello": m} | t2_riga(g, m, h, tag and "f3g2" + tag))
    t1, t2 = pd.DataFrame(t1), pd.DataFrame(t2)
    _holm(t1, "p", "p_holm")
    _holm(t2, "p_base", "p_base_holm")
    _holm(t2, "p_freq", "p_freq_holm")
    return {"t1_metriche": t1, "t2_metriche": t2, "t1_gruppi": pd.DataFrame(g1), "t2_gruppi": pd.DataFrame(g2), "t2_calibrazione": pd.DataFrame(cal)}


def scegli(t: pd.DataFrame, col_stat: str, col_p: str, col_skill: str) -> dict:
    """Tra i modelli significativamente migliori del riferimento dopo Holm a h = 1 oppure h = 4, quello con il
    guadagno medio (sui due orizzonti) piu' alto. Se nessuno, resta il riferimento."""
    t = t[t["modello"].isin(M.MODELLI)]
    sig = [m for m in M.MODELLI if ((t["modello"] == m) & (t[col_p] < ALFA) & (t[col_stat] > 0)).any()]
    medio = {m: float(t.loc[t["modello"] == m, col_skill].mean()) for m in M.MODELLI}
    scelto = max(sig, key=lambda m: medio[m]) if sig else None
    return {"scelto": scelto, "significativi": sig, "guadagno_medio": medio,
            "significativo_per_orizzonte": {m: [int(h) for h in t.loc[(t["modello"] == m) & (t[col_p] < ALFA) & (t[col_stat] > 0), "h"]] for m in sig}}


# ------------------------------------------------------------------ coefficienti

def coefficienti(coefs: list[dict], w: pd.DataFrame, VV: pd.DataFrame) -> pd.DataFrame:
    """Coefficienti standardizzati: valore all'ultima origine, quota di origini con lo stesso segno, e segno in
    regressioni separate per sottoperiodo (solo T1, blocchi singoli) per giudicare la stabilita' nel tempo."""
    x, yr = w["log_close"].to_numpy(), pd.to_datetime(w["date"]).dt.year.to_numpy()
    F = B.har_features(x)
    rows = []
    for c in coefs:
        val, fin = c["valori"], c["valori"][-1]
        sub = {}
        if c["obiettivo"] == "T1" and c["modello"] in M.BLOCCHI:
            h, Z = c["h"], VV[list(M.colonne(c["modello"]))].to_numpy()
            i = np.arange(11, len(x) - h)
            i = i[M.valide(Z)[i]]
            for a, b in C.SUBPERIODS:
                j = i[(yr[i + h] >= a) & (yr[i + h] <= b)]
                if len(j) >= MIN_COPPIE_SOTTOPERIODO:
                    X, y = np.column_stack([F[j - 11], Z[j]]), x[j + h]
                    sub[f"{a}-{b}"] = np.linalg.lstsq(X, y, rcond=None)[0][1:] * X[:, 1:].std(axis=0) / y.std()
        for k, nome in enumerate(c["variabili"]):
            r = {"obiettivo": c["obiettivo"], "h": c["h"], "modello": c["modello"], "variabile": nome, "di_m1": nome in M.NOMI_M1, "n_origini": len(val),
                 "coef_std_finale": fin[k], "coef_std_medio": val[:, k].mean(), "quota_origini_stesso_segno": float((np.sign(val[:, k]) == np.sign(fin[k])).mean())}
            if sub:
                r |= {"n_sottoperiodi": len(sub), "sottoperiodi_stesso_segno": int(sum(np.sign(v[k]) == np.sign(fin[k]) for v in sub.values())),
                      "coef_per_sottoperiodo": "; ".join(f"{p}: {v[k]:+.3f}" for p, v in sub.items())}
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ T3

def t3_quantili(x: np.ndarray, VV: pd.DataFrame, m: str, h: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Quantili walk-forward del modello m (origini ammesse per m). X7 = media dei quantili dei cinque blocchi."""
    Z = VV[list(M.colonne(m))].to_numpy()
    o = M.origini(Z, h)
    if m == "X7":
        parti = []
        for b in M.BLOCCHI:
            Zb = VV[list(M.colonne(b))].to_numpy()
            parti.append(np.array([M.t3(x[: t + 1].copy(), Zb[: t + 1].copy(), h)[0] for t in o.tolist()]))
        return o, np.mean(parti, axis=0), 0
    r = [M.t3(x[: t + 1].copy(), Z[: t + 1].copy(), h) for t in o.tolist()]
    return o, np.array([q for q, _ in r]), int(sum(c for _, c in r))


def t3_tabelle(w: pd.DataFrame, VV: pd.DataFrame, frames: dict[int, pd.DataFrame], modelli: list[str]) -> dict[str, pd.DataFrame]:
    x = w["log_close"].to_numpy()
    Z0 = np.empty((len(x), 0))
    main, grp = [], []
    for m in modelli:
        for h in M.ORIZZONTI:
            o, Q, inc = t3_quantili(x, VV, m, h)
            r = [M.t3(x[: t + 1].copy(), Z0[: t + 1].copy(), h) for t in o.tolist()]
            Qr, incr = np.array([q for q, _ in r]), int(sum(c for _, c in r))
            g = frames[h].set_index("idx_origine").loc[o].reset_index()
            y = g["y"].to_numpy()
            crps, crps_r = 2 * V2.pinball(y, Q, D.TAUS).mean(axis=1), 2 * V2.pinball(y, Qr, D.TAUS).mean(axis=1)
            for nome, q, k in ((m, Q, inc), ("D4", Qr, incr)):
                row = {"confronto": m, "h": h, "modello": nome} | V2.density_row(q, g, True, f"f3t3|{m}|{nome}|{h}") | {"quantili_incrociati": k}
                if nome != "D4":
                    row["crps_vs_d4"] = 1 - crps.mean() / crps_r.mean()
                    row["dm_stat"], row["p"] = V2.dm_loss(crps_r, crps, h)
                    row["crps_ci_lo"], row["crps_ci_hi"] = K.boot_cluster(K.blocks(o), lambda i: 1 - crps[i].mean() / crps_r[i].mean(), f"f3t3c|{m}|{h}")
                main.append(row)
            for col in GRUPPI:
                for key, gg in g.groupby(col, sort=False):
                    pos = gg.index.to_numpy()
                    for nome, q in ((m, Q), ("D4", Qr)):
                        rr = {"confronto": m, "h": h, "gruppo": col, "valore": str(key), "modello": nome} | V2.density_row(q[pos], gg)
                        if nome != "D4":
                            rr["crps_vs_d4"] = 1 - crps[pos].mean() / crps_r[pos].mean()
                        grp.append(rr)
    return {"t3_metriche": pd.DataFrame(main), "t3_gruppi": pd.DataFrame(grp)}


# ------------------------------------------------------------------ figure

def figure(frames: dict, t1: pd.DataFrame, t2: pd.DataFrame) -> None:
    col = {"G1": "#b03a2e", "X1": "#1f4e79", "X2": "#5d8fc2", "X3": "#117a65", "X4": "#d68910", "X6": "#7d3c98", "X7": "#000000"}
    with plt.rc_context(grafici.STILE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
        for ax, (t, v, lo, hi, tit) in zip(axes, ((t1, "r2_vs_m1", "r2_ci_lo", "r2_ci_hi", "T1 chiusura: R² fuori campione rispetto a M1 (%)"),
                                                  (t2, "skill_vs_base", "skill_ci_lo", "skill_ci_hi", "T2 direzione: guadagno sul Brier rispetto alla logistica di base (%)"))):
            for j, h in enumerate(M.ORIZZONTI):
                g = t[(t["h"] == h) & t["modello"].isin(M.MODELLI)].set_index("modello").loc[list(M.MODELLI)]
                xs = np.arange(len(g)) + (j - 0.5) * 0.36
                ax.bar(xs, 100 * g[v], 0.36, color=["#1f4e79", "#9db9d9"][j], label=f"h = {h}")
                ax.errorbar(xs, 100 * g[v], yerr=[100 * (g[v] - g[lo]), 100 * (g[hi] - g[v])], fmt="none", ecolor="#333", lw=0.9, capsize=2)
            ax.axhline(0, color="black", lw=0.7)
            ax.set_xticks(range(len(M.MODELLI)), M.MODELLI)
            ax.set_title(tit + "\nsopra zero: meglio del riferimento; barre: intervallo bootstrap al 95%", fontsize=9)
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "guadagni_vs_riferimento.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
        for ax, h in zip(axes, M.ORIZZONTI):
            f = frames[h]
            for m in M.MODELLI:
                g = f[f[f"f_{m}"].notna()]
                ax.plot(pd.to_datetime(g["data_obiettivo"]), ((g["y"] - g["f_M1"]) ** 2 - (g["y"] - g[f"f_{m}"]) ** 2).cumsum(), color=col[m], lw=1.1, label=m)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_title(f"T1, h = {h}: Σ (e²_M1 − e²_modello) dall'inizio del campione di ogni modello (sopra zero: meglio di M1)", fontsize=9)
        axes[0].legend(frameon=False, ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "t1_errore_cumulato_vs_m1.png")
        plt.close(fig)


# ------------------------------------------------------------------ esecuzione

def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    w, g = dati.load_settimanale(), dati.load_giornaliero()
    VV = V.costruisci(w["date"], w["close"], g)
    inv = V.inventario(VV, w["date"])
    inv.to_csv(OUT / "variabili.csv", index=False, float_format="%.6g")
    frames, coefs, penalita = previsioni(w, VV)
    tabs = tabelle(frames)
    tabs["coefficienti"] = coefficienti(coefs, w, VV)
    s1 = scegli(tabs["t1_metriche"], "cw_stat", "p_holm", "r2_vs_m1")
    s2 = scegli(tabs["t2_metriche"], "dm_stat_base", "p_base_holm", "skill_vs_base")
    da_t3 = [m for m in dict.fromkeys([s1["scelto"], s2["scelto"]]) if m]
    if da_t3:
        tabs |= t3_tabelle(w, VV, frames, da_t3)
    else:                                              # nessun modello scelto: T3 non si esegue; eventuali file di una esecuzione precedente vengono tolti
        for nome in ("t3_metriche", "t3_gruppi"):
            (OUT / f"{nome}.csv").unlink(missing_ok=True)
    for name, t in tabs.items():
        t.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")
    pd.concat([f.drop(columns="idx_origine") for f in frames.values()], ignore_index=True).to_csv(
        OUT / "previsioni_dev.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    figure(frames, tabs["t1_metriche"], tabs["t2_metriche"])
    quota = lambda v: {str(k): float(np.mean(np.array(v) == k)) for k in M.GRIGLIA_RIDGE}
    senza = {}
    for s_ in dati.SERIE:                               # settimane (dopo l'inizio della serie) in cui l'ultima seduta del VIX non ha un valore della serie
        d = g[s_].dropna().index
        m = w.loc[(w["date"] >= d.min()) & ~w["date"].isin(d), "date"]
        senza[s_] = {"n": int(len(m)), "date": [str(x.date()) for x in m.iloc[:3]]}
    info = {"campione": [str(w["date"].min().date()), str(w["date"].max().date())], "n_settimane": int(len(w)),
            "serie_usate": list(dati.SERIE), "ritardi_giorni": dati.ritardi(), "ultima_data_giornaliera": str(g.index.max().date()),
            "ultima_seduta_senza_valore": senza,
            "riferimento_t1": "M1 stimato su tutta la storia, valutato sulle stesse origini del modello", "alfa": ALFA,
            "n_test_holm": {"t1": int(tabs["t1_metriche"]["p"].notna().sum()), "t2_base": int(tabs["t2_metriche"]["p_base"].notna().sum()),
                            "t2_freq": int(tabs["t2_metriche"]["p_freq"].notna().sum())},
            "scelta_chiusura": s1, "scelta_direzione": s2, "t3_eseguito_per": da_t3,
            "x6_penalita": {str(h): {k: quota(v) for k, v in d.items()} for h, d in penalita.items()},
            "griglia_ridge": list(M.GRIGLIA_RIDGE), "validazione_ridge": M.VALIDAZIONE, "soglie_probabilita": list(SOGLIE_PROB),
            "fascia_evidenza": list(FASCIA_EVIDENZA), "rialzo_forte": RIALZO_FORTE,
            "per_orizzonte": {str(h): {"n_origini_m1": int(len(f)), "n_fascia_13_22": int(f["fascia_13_22"].sum()), "n_dopo_rialzo_30": int(f["dopo_rialzo_30"].sum()),
                                       "n_per_modello": {m: int(f[f"f_{m}"].notna().sum()) for m in M.MODELLI}} for h, f in frames.items()}}
    (OUT / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with K.guardia_dati():
        run()
