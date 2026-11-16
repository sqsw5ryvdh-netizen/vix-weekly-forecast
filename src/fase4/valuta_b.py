"""Fase 4, Parte B: walk-forward di sviluppo di N1 (regimi), N2 e N3 (gradient boosting) per T1 e T2, con le stesse
tabelle della Fase 3 (sottoperiodi, fasce di partenza, fascia 13-22, casi dopo un rialzo oltre il 30%).

Riferimenti sulle stesse origini: M1 stimato su tutta la storia (T1); frequenza storica e logistica di base (T2).
Test: i tre modelli non contengono M1 come caso particolare, quindi — come nella Fase 1 per i modelli non annidati —
si usa Diebold-Mariano (due code) e non Clark-West. Regola (come nella Fase 3, ALFA = 5% dopo Holm): passa un
modello significativamente migliore del riferimento a h = 1 oppure h = 4; tra quelli che passano, guadagno medio
piu' alto. Se nessuno passa, lo si dichiara e nessuno entra tra i finalisti.
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
from ..fase2b import dati as dati2b
from ..fase2b import grafici
from ..fase2c import componenti as CP
from ..fase3 import dati as dati3
from ..fase3 import modelli as M
from ..fase3 import valuta as VA
from ..fase3 import variabili as V
from . import boosting as GB
from . import regimi as RE

OUT = C.REPORTS_DIR / "fase4"
FIG = C.FIG_DIR / "fase4"
MODELLI = ("N1", "N2", "N3")
NOMI = {"N1": "N1 autoregressivo a 2 regimi", "N2": "N2 gradient boosting, solo VIX", "N3": "N3 gradient boosting, VIX + mercato"}
ALFA = VA.ALFA
ORIZZONTI = M.ORIZZONTI


def variabili_boosting(df: pd.DataFrame, VV: pd.DataFrame) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Matrici delle variabili aggiuntive (oltre a quelle di M1) per N2 e N3. Riga i = valori noti alla fine della settimana i."""
    S = CP.serie(df)
    extra = CP.grandezze(S, len(df) - 1)["extra"].copy()
    extra[S.bad] = np.nan                              # apertura incoerente: variabili della candela mancanti
    g1 = list(V.BLOCCHI["G1"])
    resto = [c for c in V.TUTTE if c not in g1]
    return {"N2": (np.column_stack([extra, VV[g1].to_numpy()]), list(GB.NOMI_CANDELA) + g1),
            "N3": (np.column_stack([extra, VV[g1 + resto].to_numpy()]), list(GB.NOMI_CANDELA) + g1 + resto)}


def previsioni(w: pd.DataFrame, df: pd.DataFrame, VV: pd.DataFrame) -> tuple[dict[int, pd.DataFrame], dict]:
    x, close, dates = w["log_close"].to_numpy(), w["close"].to_numpy(), w["date"].to_numpy()
    Z0 = np.empty((len(x), 0))
    ZZ = variabili_boosting(df, VV)
    n1 = RE.walk(x, ORIZZONTI)
    frames, extra = {}, {"n1": n1, "ristime_gb": {}, "importanze": {}}
    for h in ORIZZONTI:
        o = M.origini(Z0, h)
        f = pd.DataFrame({"h": h, "idx_origine": o, "data_origine": dates[o], "data_obiettivo": dates[o + h], "x0": x[o], "vix0": close[o],
                          "y": x[o + h], "su": (x[o + h] > x[o]).astype(float)})
        f["f_M1"] = M.walk(x, Z0, h, M.t1)[1]
        f["p_FREQ"] = [M.t2_frequenza(x[: t + 1], h) for t in o.tolist()]
        f["p_BASE"] = M.walk(x, Z0, h, M.t2)[1]
        p = n1["previsioni"][h]
        f["f_N1"], f["p_N1"], f["f_M1fin_N1"] = VA._colonna(o, p["origini"], p["media"]), VA._colonna(o, p["origini"], p["p_su"]), f["f_M1"]
        for m in ("N2", "N3"):
            Z = ZZ[m][0]
            om, v1, rs = GB.walk(x, Z, h, "T1")
            _, v2, _ = GB.walk(x, Z, h, "T2")
            f[f"f_{m}"], f[f"p_{m}"] = VA._colonna(o, om, v1), VA._colonna(o, om, v2)
            f[f"f_M1fin_{m}"] = VA._colonna(o, om, M.walk(x, Z, h, M.t1, usa=[])[1])
            extra["ristime_gb"][f"{m}|{h}"] = len(rs)
            ultimo = GB.stima(x[: om[-1] + 1], Z[: om[-1] + 1], h, "T1")           # importanza delle variabili all'ultima origine (solo descrizione)
            imp = pd.Series(ultimo.feature_importances_, index=list(M.NOMI_M1) + ZZ[m][1]).sort_values(ascending=False)
            extra["importanze"][f"{m}|{h}"] = {k: float(v) for k, v in imp.iloc[:5].items()}
        f["fascia_origine"] = np.array(C.BAND_LABELS)[B.band_index(f["vix0"])]
        yr = pd.to_datetime(f["data_obiettivo"]).dt.year
        f["sottoperiodo"] = ""
        for a, b in C.SUBPERIODS:
            f.loc[(yr >= a) & (yr <= b), "sottoperiodo"] = f"{a}-{b}"
        f["fascia_13_22"] = (f["vix0"] >= VA.FASCIA_EVIDENZA[0]) & (f["vix0"] <= VA.FASCIA_EVIDENZA[1])
        f["dopo_rialzo_30"] = close[o] / close[o - 4] - 1 > VA.RIALZO_FORTE
        frames[h] = f
    return frames, extra


def _t1(g: pd.DataFrame, m: str, h: int, tag: str | None = None) -> dict:
    """Riga T1 della Fase 3, con il test di Diebold-Mariano al posto di Clark-West (modelli non annidati)."""
    r = VA.t1_riga(g, m, h, tag)
    r.pop("cw_stat", None)
    r.pop("p", None)
    r["p"] = r.pop("dm_p")
    return r


def tabelle(frames: dict[int, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    t1, t2, g1, g2, cal = [], [], [], [], []
    for h, f in frames.items():
        for m in MODELLI:
            t1.append({"h": h, "modello": m} | _t1(f, m, h, f"f4t1|{m}|{h}"))
            t2.append({"h": h, "modello": m} | VA.t2_riga(f, m, h, f"f4t2|{m}|{h}"))
            fm = f[f[f"p_{m}"].notna()]
            p, y = fm[f"p_{m}"].to_numpy(), fm["su"].to_numpy()
            b = np.minimum((p * 10).astype(int), 9)
            for k in np.unique(b):
                cal.append({"h": h, "modello": m, "classe": f"{k / 10:.1f}-{(k + 1) / 10:.1f}", "n": int((b == k).sum()), "prob_media": p[b == k].mean(), "freq_osservata": y[b == k].mean()})
        for col in VA.GRUPPI:
            for key, g in f.groupby(col, sort=False):
                for m in MODELLI:
                    if g[f"f_{m}"].notna().sum() == 0:
                        continue
                    tag = f"|{col}|{key}|{m}|{h}" if col in ("fascia_13_22", "dopo_rialzo_30") and key else None
                    g1.append({"h": h, "gruppo": col, "valore": str(key), "modello": m} | _t1(g, m, h, tag and "f4g1" + tag))
                    g2.append({"h": h, "gruppo": col, "valore": str(key), "modello": m} | VA.t2_riga(g, m, h, tag and "f4g2" + tag))
    t1, t2 = pd.DataFrame(t1), pd.DataFrame(t2)
    VA._holm(t1, "p", "p_holm")
    VA._holm(t2, "p_base", "p_base_holm")
    VA._holm(t2, "p_freq", "p_freq_holm")
    return {"b_t1_metriche": t1, "b_t2_metriche": t2, "b_t1_gruppi": pd.DataFrame(g1), "b_t2_gruppi": pd.DataFrame(g2), "b_t2_calibrazione": pd.DataFrame(cal)}


def scegli(t: pd.DataFrame, col_stat: str, col_p: str, col_skill: str) -> dict:
    """Stessa regola della Fase 3, sull'elenco N1-N3."""
    sig = [m for m in MODELLI if ((t["modello"] == m) & (t[col_p] < ALFA) & (t[col_stat] > 0)).any()]
    medio = {m: float(t.loc[t["modello"] == m, col_skill].mean()) for m in MODELLI}
    return {"scelto": max(sig, key=lambda m: medio[m]) if sig else None, "significativi": sig, "guadagno_medio": medio}


def regimi_tabelle(n1: dict, w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = w["date"].to_numpy()
    st = []
    for r in n1["ristime"]:
        par = n1["stime"][r["idx_stima"]]
        st.append({"data_stima": str(pd.Timestamp(dates[r["idx_stima"]]).date()), "settimane_usate": r["idx_stima"] + 1, "valida": r["valida"]} | (RE.descrizione(par) if par is not None else {}))
    pr = pd.DataFrame(n1["probabilita"])
    pr.insert(0, "data_origine", [str(pd.Timestamp(d).date()) for d in dates[pr["idx_origine"].to_numpy()]])
    pr["chiusura"] = w["close"].to_numpy()[pr["idx_origine"].to_numpy()]
    return pd.DataFrame(st), pr.drop(columns=["idx_origine", "idx_stima_usata"])


def figura_regimi(pr: pd.DataFrame) -> None:
    with plt.rc_context(grafici.STILE):
        fig, (a, b) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
        d = pd.to_datetime(pr["data_origine"])
        a.plot(d, pr["chiusura"], color=grafici.GIU, lw=0.9)
        a.set_ylabel("VIX (chiusura settimanale)")
        a.set_title("N1: chiusura del VIX e probabilità filtrata del regime con livello medio più alto (calcolata a ogni origine con i soli dati fino all'origine)", fontsize=9.5)
        b.fill_between(d, 0, pr["p_regime_alto"], color="#b03a2e", alpha=0.55, lw=0)
        b.set_ylim(0, 1)
        b.set_ylabel("P(regime alto)")
        fig.tight_layout()
        fig.savefig(FIG / "b_regimi_probabilita.png")
        plt.close(fig)


def figura_guadagni(t1: pd.DataFrame, t2: pd.DataFrame) -> None:
    with plt.rc_context(grafici.STILE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (t, v, lo, hi, tit) in zip(axes, ((t1, "r2_vs_m1", "r2_ci_lo", "r2_ci_hi", "T1 chiusura: R² fuori campione rispetto a M1 (%)"),
                                                  (t2, "skill_vs_base", "skill_ci_lo", "skill_ci_hi", "T2 direzione: guadagno sul Brier rispetto alla logistica di base (%)"))):
            for j, h in enumerate(ORIZZONTI):
                g = t[t["h"] == h].set_index("modello").loc[list(MODELLI)]
                xs = np.arange(len(g)) + (j - 0.5) * 0.36
                ax.bar(xs, 100 * g[v], 0.36, color=["#1f4e79", "#9db9d9"][j], label=f"h = {h}")
                ax.errorbar(xs, 100 * g[v], yerr=[100 * (g[v] - g[lo]), 100 * (g[hi] - g[v])], fmt="none", ecolor="#333", lw=0.9, capsize=2)
            ax.axhline(0, color="black", lw=0.7)
            ax.set_xticks(range(len(MODELLI)), MODELLI)
            ax.set_title(tit + "\nsopra zero: meglio del riferimento; barre: intervallo bootstrap al 95%", fontsize=9)
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "b_guadagni_vs_riferimento.png")
        plt.close(fig)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    w, g, df = dati3.load_settimanale(), dati3.load_giornaliero(), dati2b.load_dev_candele()
    VV = V.costruisci(w["date"], w["close"], g)
    frames, extra = previsioni(w, df, VV)
    tabs = tabelle(frames)
    s1 = scegli(tabs["b_t1_metriche"], "dm_stat", "p_holm", "r2_vs_m1")
    s2 = scegli(tabs["b_t2_metriche"], "dm_stat_base", "p_base_holm", "skill_vs_base")
    st, pr = regimi_tabelle(extra["n1"], w)
    tabs |= {"b_regimi_stime": st, "b_regimi_probabilita": pr}
    for name, t in tabs.items():
        t.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")
    pd.concat([f.drop(columns="idx_origine") for f in frames.values()], ignore_index=True).to_csv(
        OUT / "b_previsioni_dev.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    figura_regimi(pr)
    figura_guadagni(tabs["b_t1_metriche"], tabs["b_t2_metriche"])
    valide = st[st["valida"]]
    info = {"modelli": NOMI, "alfa": ALFA, "test": "Diebold-Mariano a due code (modelli non annidati in M1)",
            "n_test_holm": {"t1": int(tabs["b_t1_metriche"]["p"].notna().sum()), "t2_base": int(tabs["b_t2_metriche"]["p_base"].notna().sum()),
                            "t2_freq": int(tabs["b_t2_metriche"]["p_freq"].notna().sum())},
            "scelta_chiusura": s1, "scelta_direzione": s2, "passa": sorted({m for m in (s1["scelto"], s2["scelto"]) if m}),
            "n1": {"passo_stima": RE.PASSO_STIMA, "n_ristime": int(len(st)), "n_non_valide": int((~st["valida"]).sum()),
                   "date_non_valide": st.loc[~st["valida"], "data_stima"].tolist(),
                   "origini_con_ripiego_ar1": int(pr["p_regime_alto"].isna().sum()), "n_origini": int(len(pr)),
                   "ultima_stima": valide.iloc[-1].drop(labels=["valida"]).to_dict(),
                   "intervallo_tra_le_stime": {c: [float(valide[c].min()), float(valide[c].max())] for c in ("livello_basso", "livello_alto", "sigma_basso", "sigma_alto", "durata_media_basso", "durata_media_alto", "phi")},
                   "quota_origini_regime_alto_oltre_50": float((pr["p_regime_alto"] > 0.5).mean())},
            "boosting": {"parametri": GB.PARAMETRI, "passo_stima": GB.PASSO_STIMA, "n_ristime": extra["ristime_gb"], "obiettivo_t1": "variazione x_{t+h} - x_t",
                         "importanze_ultima_origine": extra["importanze"]},
            "per_orizzonte": {str(h): {"n_origini_m1": int(len(f)), "n_per_modello": {m: int(f[f"f_{m}"].notna().sum()) for m in MODELLI},
                                       "prima_data_obiettivo": {m: str(pd.Timestamp(f.loc[f[f"f_{m}"].notna(), "data_obiettivo"].iloc[0]).date()) for m in MODELLI}} for h, f in frames.items()}}
    (OUT / "b_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with K.guardia_dati():
        run()
