"""Genera reports/fase3_report.md. Tutti i numeri sono letti dai file reports/fase3/*.csv e *.json (e, per il
confronto delle candele, dai file della Fase 2c): nessun numero e' scritto a mano."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .. import config as C
from . import candele as KA
from . import modelli as M
from . import variabili as V

R = C.REPORTS_DIR / "fase3"
MIN_N = 20


def md(df: pd.DataFrame, fmt=".3f") -> str:
    return re.sub(r"\bnan\b", "  –", df.to_markdown(index=False, floatfmt=fmt, missingval="–"))


def n(v, d: int = 2) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def pct(v, d: int = 1) -> str:
    return f"{100 * v:+.{d}f}%".replace(".", ",")


def pc(v, d: int = 1) -> str:
    return f"{100 * v:.{d}f}%".replace(".", ",")


def ic(v, lo, hi, d: int = 1) -> str:
    return f"{pct(v, d)} [{pct(lo, d)}; {pct(hi, d)}]"


def giudizio(v: float, lo: float, hi: float, p_holm: float) -> str:
    if p_holm < 0.05 and v > 0:
        return "guadagno significativo dopo Holm" + (", ma con intervallo bootstrap che comprende lo zero" if lo <= 0 else "")
    if lo > 0:
        return "guadagno con intervallo sopra zero, ma non significativo dopo Holm"
    if hi < 0:
        return "peggioramento (intervallo tutto sotto zero)"
    return "nessuna differenza distinguibile da zero"


def n_casi(k) -> str:
    return f"{int(k)}" + (" (meno di 20)" if k < MIN_N else "")


def run() -> None:
    T = lambda k: pd.read_csv(R / f"{k}.csv")
    info, p0 = json.loads((R / "info.json").read_text()), json.loads((R / "p0_info.json").read_text())
    amp, pag, et, var = T("p0_ampiezze"), T("p0_pagella"), T("p0_etichetta"), T("variabili")
    t1, t2, g1, g2, cal, co = T("t1_metriche"), T("t2_metriche"), T("t1_gruppi"), T("t2_gruppi"), T("t2_calibrazione"), T("coefficienti")
    t3_ok = bool(info["t3_eseguito_per"])
    t3, g3t = (T("t3_metriche"), T("t3_gruppi")) if t3_ok else (pd.DataFrame(), pd.DataFrame())
    s1, s2 = info["scelta_chiusura"], info["scelta_direzione"]
    A1, A2 = t1.set_index(["modello", "h"]), t2.set_index(["modello", "h"])
    P, E = pag.set_index(["costruzione", "h"]), et.set_index("h")
    K_NEW, hz = KA.COSTRUZIONE, list(M.ORIZZONTI)

    # ------------------------------------------------------------------ Parte 0
    u = amp.copy()
    u["componente"] = u["componente"].map(KA.NOMI_COMPONENTI)
    u["guadagno della regressione sull'errore quadratico [IC 95%]"] = [ic(a, b, c) for a, b, c in zip(u["guadagno_pct"], u["guadagno_ci_lo"], u["guadagno_ci_hi"])]
    u["usata nel programma"] = u["scelta"].map({"regressione": "regressione", "media": "media storica"})
    t_amp = md(u[["componente", "h", "n", "media_reale", "previsione_media_regressione", "previsione_media_storica", "guadagno della regressione sull'errore quadratico [IC 95%]",
                  "dm_stat", "p", "usata nel programma"]].rename(columns={"media_reale": "media reale", "previsione_media_regressione": "previsione media, regressione",
                                                                          "previsione_media_storica": "previsione media, media storica", "dm_stat": "DM"}), ".4f")
    usa_reg = [f"{KA.NOMI_COMPONENTI[r.componente]} a h = {r.h}" for r in amp.itertuples() if r.scelta == "regressione"]
    q = pag.sort_values(["h", "costruzione"]).copy()
    q["sovrapposizione [IC 95%]"] = [f"{a:.3f} [{b:.3f}; {c:.3f}]" for a, b, c in zip(q["sovrapposizione"], q["sovrapposizione_ci_lo"], q["sovrapposizione_ci_hi"])]
    t_pag = md(q[["h", "descrizione", "n", "rapporto_corpo", "rapporto_escursione", "sovrapposizione [IC 95%]", "chiusura_nel_corpo", "chiusura_nell_escursione",
                  "mae_apertura", "mae_massimo", "mae_minimo", "mae_chiusura"]].rename(columns={
        "descrizione": "candele", "rapporto_corpo": "corpo previsto / reale", "rapporto_escursione": "escursione prevista / reale", "chiusura_nel_corpo": "chiusura reale nel corpo previsto",
        "chiusura_nell_escursione": "chiusura reale nell'escursione prevista", "mae_apertura": "errore medio apertura", "mae_massimo": "errore medio massimo",
        "mae_minimo": "errore medio minimo", "mae_chiusura": "errore medio chiusura della candela"}), ".3f")
    in_vincolo = lambda v: 0.8 <= v <= 1.2
    c1n, e1n = P.loc[(K_NEW, 1), "rapporto_corpo"], P.loc[(K_NEW, 1), "rapporto_escursione"]
    a1 = amp.set_index(["componente", "h"])
    sale = lambda col: all(P.loc[(K_NEW, h), col] > P.loc[("K2", h), col] for h in KA.ORIZZONTI)
    peggio_ch = all(P.loc[(K_NEW, h), "mae_chiusura"] > P.loc[("K2", h), "mae_chiusura"] for h in KA.ORIZZONTI)
    neutra_magg = all(E.loc[h, "quota_neutra"] > 0.5 for h in KA.ORIZZONTI)
    ribasso_meglio = all(E.loc[h, "accuratezza_ribasso"] > E.loc[h, "accuratezza_rialzo"] for h in KA.ORIZZONTI)
    ee = et.copy()
    ee["accuratezza delle etichette non neutre [IC 95%]"] = [f"{pc(a)} [{pc(b)}; {pc(c)}]" for a, b, c in zip(ee["accuratezza_non_neutre"], ee["accuratezza_non_neutre_ci_lo"], ee["accuratezza_non_neutre_ci_hi"])]
    t_et = md(ee[["h", "n", "frequenza_chiusura_su", "quota_neutra", "quota_rialzo", "quota_ribasso", "n_non_neutre", "accuratezza delle etichette non neutre [IC 95%]",
                  "n_rialzo", "accuratezza_rialzo", "n_ribasso", "accuratezza_ribasso", "brier", "brier_frequenza_storica", "brier_skill_vs_frequenza", "p"]].rename(columns={
        "frequenza_chiusura_su": "freq. chiusura sopra l'origine", "quota_neutra": "quota «neutra»", "quota_rialzo": "quota «rialzo»", "quota_ribasso": "quota «ribasso»",
        "n_non_neutre": "n non neutre", "n_rialzo": "n «rialzo»", "accuratezza_rialzo": "corrette «rialzo»", "n_ribasso": "n «ribasso»", "accuratezza_ribasso": "corrette «ribasso»",
        "brier": "Brier D1", "brier_frequenza_storica": "Brier frequenza storica", "brier_skill_vs_frequenza": "guadagno sul Brier", "p": "p (DM)"}), ".3f")

    # ------------------------------------------------------------------ variabili
    t_var = md(var[["blocco", "variabile", "descrizione", "prima_settimana_valida", "settimane_valide", "settimane_mancanti_dopo_l_inizio"]].rename(columns={
        "prima_settimana_valida": "prima settimana valida", "settimane_valide": "settimane valide", "settimane_mancanti_dopo_l_inizio": "mancanti dopo l'inizio"}))

    # ------------------------------------------------------------------ T1
    nome = lambda m: M.NOMI.get(m, {"M1": "M1 (riferimento)", "FREQ": "frequenza storica (riferimento)", "BASE": "logistica di base (riferimento)", "D4": "D4 (riferimento)"}.get(m, m))
    a = t1[t1["modello"] != "M1"].copy()
    a["modello "] = a["modello"].map(nome)
    a["R² vs M1 [IC 95%]"] = [ic(x, lo, hi) for x, lo, hi in zip(a["r2_vs_m1"], a["r2_ci_lo"], a["r2_ci_hi"])]
    a["R² vs M1 stimato sulla stessa finestra"] = [pct(x) for x in a["r2_vs_m1_stessa_finestra"]]
    a["significativo dopo Holm"] = np.where((a["p_holm"] < info["alfa"]) & (a["cw_stat"] > 0), "sì", "no")
    t_t1 = md(a[["h", "modello ", "n", "prima_data_obiettivo", "rmse_log", "rmse_log_m1", "mae_log", "mae_log_m1", "R² vs M1 [IC 95%]", "R² vs M1 stimato sulla stessa finestra",
                 "cw_stat", "p", "p_holm", "significativo dopo Holm", "dm_p"]].rename(columns={
        "modello ": "modello", "prima_data_obiettivo": "prima data prevista", "rmse_log": "RMSE log", "rmse_log_m1": "RMSE log M1", "mae_log": "MAE log", "mae_log_m1": "MAE log M1",
        "cw_stat": "Clark-West", "p": "p (una coda)", "p_holm": "p Holm", "dm_p": "p DM (due code)"}), ".4f")
    t_t1b = md(a[["h", "modello ", "n", "rmse_vix", "rmse_vix_m1", "mae_vix", "mae_vix_m1", "entro_10", "entro_10_m1", "entro_20", "entro_20_m1"]].rename(columns={
        "modello ": "modello", "rmse_vix": "RMSE punti", "rmse_vix_m1": "RMSE punti M1", "mae_vix": "MAE punti", "mae_vix_m1": "MAE punti M1", "entro_10": "entro il 10%",
        "entro_10_m1": "entro il 10%, M1", "entro_20": "entro il 20%", "entro_20_m1": "entro il 20%, M1"}), ".3f")
    m1 = t1[t1["modello"] == "M1"].set_index("h")

    # ------------------------------------------------------------------ T2
    b = t2.copy()
    b["modello "] = b["modello"].map(nome)
    b["guadagno sul Brier vs logistica di base [IC 95%]"] = [ic(x, lo, hi) if np.isfinite(x) else "–" for x, lo, hi in zip(b["skill_vs_base"], b["skill_ci_lo"], b["skill_ci_hi"])]
    b["significativo dopo Holm (vs base)"] = np.where(b["p_base"].isna(), "–", np.where((b["p_base_holm"] < info["alfa"]) & (b["dm_stat_base"] > 0), "sì", "no"))
    t_t2 = md(b[["h", "modello ", "n", "frequenza_rialzo", "accuratezza", "accuratezza_base", "brier", "brier_base", "brier_freq", "affidabilita", "risoluzione",
                 "guadagno sul Brier vs logistica di base [IC 95%]", "p_base", "p_base_holm", "significativo dopo Holm (vs base)", "skill_vs_freq", "p_freq", "p_freq_holm"]].rename(columns={
        "modello ": "modello", "frequenza_rialzo": "freq. rialzo", "accuratezza_base": "accuratezza base", "brier": "Brier", "brier_base": "Brier base", "brier_freq": "Brier frequenza",
        "affidabilita": "affidabilità (0 = calibrato)", "p_base": "p vs base", "p_base_holm": "p Holm vs base", "skill_vs_freq": "guadagno sul Brier vs frequenza",
        "p_freq": "p vs frequenza", "p_freq_holm": "p Holm vs frequenza"}), ".4f")
    cols_s = {}
    for s in info["soglie_probabilita"]:
        k = int(round(100 * s))
        b[f"prob. ≥ {k}%: quota (n) → accuratezza"] = [f"{pc(qq)} ({int(nn)}{', meno di 20' if nn < MIN_N else ''}) → {pc(ac) if np.isfinite(ac) else '–'}"
                                                      for qq, nn, ac in zip(b[f"quota_{k}"], b[f"n_{k}"], b[f"accuratezza_{k}"])]
        cols_s[k] = f"prob. ≥ {k}%: quota (n) → accuratezza"
    t_t2s = md(b[b["modello"] != "FREQ"][["h", "modello ", "n", "accuratezza"] + list(cols_s.values())].rename(columns={"modello ": "modello", "accuratezza": "accuratezza (tutte)"}), ".3f")
    migliore_t2 = max(M.MODELLI, key=lambda m: s2["guadagno_medio"][m])
    cc = cal[cal["modello"].isin(["BASE", migliore_t2])].copy()
    cc["modello"] = cc["modello"].map(nome)
    t_cal = md(cc.rename(columns={"classe": "classe di probabilità", "prob_media": "probabilità media", "freq_osservata": "frequenza osservata"}), ".3f")

    # ------------------------------------------------------------------ T3
    if t3_ok:
        c = t3.copy()
        c["modello "] = c["modello"].map(nome)
        c["guadagno sul CRPS vs D4 [IC 95%]"] = [ic(x, lo, hi) if np.isfinite(x) else "–" for x, lo, hi in zip(c["crps_vs_d4"], c["crps_ci_lo"], c["crps_ci_hi"])]
        t_t3 = md(c[["h", "modello ", "n", "crps", "guadagno sul CRPS vs D4 [IC 95%]", "p", "copertura_50", "copertura_80", "copertura_90", "larghezza_vix_50", "larghezza_vix_80",
                     "larghezza_vix_90", "mediana_mae_vix", "quantili_incrociati"]].rename(columns={
            "modello ": "modello", "crps": "CRPS", "p": "p (DM)", "copertura_50": "copertura 50%", "copertura_80": "copertura 80%", "copertura_90": "copertura 90%",
            "larghezza_vix_50": "larghezza 50% (punti)", "larghezza_vix_80": "larghezza 80% (punti)", "larghezza_vix_90": "larghezza 90% (punti)",
            "mediana_mae_vix": "errore medio della mediana (punti)", "quantili_incrociati": "previsioni con quantili incrociati"}), ".4f")
        B3 = t3.set_index(["confronto", "modello", "h"])
    else:
        t_t3 = "*T3 non è stato eseguito: nessuna regola ha scelto un modello.*"

    # ------------------------------------------------------------------ gruppi
    def gruppo_t1(col: str, val: str | None = None) -> str:
        g = g1[g1["gruppo"] == col]
        g = g[g["valore"] == val] if val else g
        g = g.assign(**{"modello ": g["modello"].map(nome), "casi": [n_casi(k) for k in g["n"]],
                        "R² vs M1 [IC 95%]": [ic(x, lo, hi) if np.isfinite(lo) else pct(x) for x, lo, hi in zip(g["r2_vs_m1"], g.get("r2_ci_lo", np.nan), g.get("r2_ci_hi", np.nan))]})
        return md(g[["h", "modello ", "casi", "rmse_log", "rmse_log_m1", "mae_vix", "mae_vix_m1", "R² vs M1 [IC 95%]", "cw_stat", "p", "entro_10", "entro_10_m1", "entro_20", "entro_20_m1"]].rename(columns={
            "modello ": "modello", "rmse_log": "RMSE log", "rmse_log_m1": "RMSE log M1", "mae_vix": "MAE punti", "mae_vix_m1": "MAE punti M1", "cw_stat": "Clark-West",
            "p": "p (una coda, non corretto)", "entro_10": "entro il 10%", "entro_10_m1": "entro il 10%, M1", "entro_20": "entro il 20%", "entro_20_m1": "entro il 20%, M1"}), ".4f")

    def gruppo_t2(col: str, val: str) -> str:
        g = g2[(g2["gruppo"] == col) & (g2["valore"] == val)]
        g = g.assign(**{"modello ": g["modello"].map(nome), "casi": [n_casi(k) for k in g["n"]],
                        "guadagno sul Brier vs base [IC 95%]": [ic(x, lo, hi) for x, lo, hi in zip(g["skill_vs_base"], g["skill_ci_lo"], g["skill_ci_hi"])],
                        "prob. ≥ 60%: quota (n) → accuratezza": [f"{pc(qq)} ({int(nn)}) → {pc(ac) if np.isfinite(ac) else '–'}" for qq, nn, ac in zip(g["quota_60"], g["n_60"], g["accuratezza_60"])]})
        return md(g[["h", "modello ", "casi", "frequenza_rialzo", "accuratezza", "accuratezza_base", "brier", "brier_base", "guadagno sul Brier vs base [IC 95%]",
                     "prob. ≥ 60%: quota (n) → accuratezza"]].rename(columns={"modello ": "modello", "frequenza_rialzo": "freq. rialzo", "accuratezza_base": "accuratezza base",
                                                                               "brier": "Brier", "brier_base": "Brier base"}), ".4f")

    def pivot(g: pd.DataFrame, col: str, val: str, ordine=None) -> str:
        g = g[g["gruppo"] == col]
        p = g.pivot_table(index=["h", "valore"], columns="modello", values=val, sort=False).reset_index()
        nn = g.pivot_table(index=["h", "valore"], columns="modello", values="n", sort=False).reset_index()
        if ordine is not None:
            p["valore"] = pd.Categorical(p["valore"], ordine, ordered=True)
            nn["valore"] = pd.Categorical(nn["valore"], ordine, ordered=True)
        p, nn = p.sort_values(["h", "valore"]), nn.sort_values(["h", "valore"])
        out = p[["h", "valore"]].copy()
        for m in M.MODELLI:
            out[m] = [f"{pct(v)} ({int(k)})" if np.isfinite(v) else "–" for v, k in zip(p[m], nn[m].fillna(0))]
        return md(out.rename(columns={"valore": {"sottoperiodo": "sottoperiodo", "fascia_origine": "fascia di partenza"}[col]}))

    def gruppo_t3(col: str, val: str | None = None) -> str:
        g = g3t[g3t["gruppo"] == col]
        g = g[g["valore"] == val] if val else g
        g = g.assign(**{"modello ": g["modello"].map(nome), "casi": [n_casi(k) for k in g["n"]], "guadagno sul CRPS vs D4": [pct(x) if np.isfinite(x) else "–" for x in g["crps_vs_d4"]]})
        cols = ["h"] + ([] if val else ["valore"]) + ["modello ", "casi", "crps", "guadagno sul CRPS vs D4", "copertura_50", "copertura_80", "copertura_90", "larghezza_vix_80"]
        return md(g[cols].rename(columns={"modello ": "modello", "valore": "gruppo", "crps": "CRPS", "copertura_50": "copertura 50%", "copertura_80": "copertura 80%",
                                          "copertura_90": "copertura 90%", "larghezza_vix_80": "larghezza 80% (punti)"}), ".4f")

    # ------------------------------------------------------------------ frasi
    def frase_t1(m: str) -> str:
        return "; ".join(f"h = {h}: R² rispetto a M1 {ic(A1.loc[(m, h), 'r2_vs_m1'], A1.loc[(m, h), 'r2_ci_lo'], A1.loc[(m, h), 'r2_ci_hi'])}, "
                         f"{giudizio(A1.loc[(m, h), 'r2_vs_m1'], A1.loc[(m, h), 'r2_ci_lo'], A1.loc[(m, h), 'r2_ci_hi'], A1.loc[(m, h), 'p_holm'])}" for h in hz)

    def frase_t2(m: str) -> str:
        return "; ".join(f"h = {h}: guadagno sul Brier {ic(A2.loc[(m, h), 'skill_vs_base'], A2.loc[(m, h), 'skill_ci_lo'], A2.loc[(m, h), 'skill_ci_hi'])}, "
                         f"{giudizio(A2.loc[(m, h), 'skill_vs_base'], A2.loc[(m, h), 'skill_ci_lo'], A2.loc[(m, h), 'skill_ci_hi'], A2.loc[(m, h), 'p_base_holm'])}" for h in hz)

    def esito(m: str, A, v: str, lo: str, hi: str, ph: str) -> str:
        g = {giudizio(A.loc[(m, h), v], A.loc[(m, h), lo], A.loc[(m, h), hi], A.loc[(m, h), ph]) for h in hz}
        if any(x.startswith("guadagno significativo") for x in g):
            return "un guadagno significativo dopo Holm su almeno un orizzonte"
        if any(x.startswith("guadagno") for x in g):
            return "un guadagno non significativo dopo Holm"
        if all(x.startswith("peggioramento") for x in g):
            return "un peggioramento"
        if any(x.startswith("peggioramento") for x in g):
            return "nessun guadagno (un peggioramento su un orizzonte)"
        return "nessun guadagno distinguibile da zero"

    mercato = [m for m in M.BLOCCHI if m != "G1"]
    C1 = co[(co["obiettivo"] == "T1") & (~co["di_m1"])]

    def frase_pesi(m: str) -> str:
        g = C1[C1["modello"] == m]
        peso = g.groupby("variabile")["coef_std_finale"].apply(lambda v: float(np.abs(v).mean())).sort_values(ascending=False)
        frasi = []
        for v in peso.index[:2]:
            r = g[g["variabile"] == v].set_index("h")
            segni = {int(np.sign(r.loc[h, "coef_std_finale"])) for h in hz}
            stab = all(r.loc[h, "sottoperiodi_stesso_segno"] == r.loc[h, "n_sottoperiodi"] for h in hz) and len(segni) == 1
            verso = ("valori più alti → VIX previsto più alto" if segni == {1} else "valori più alti → VIX previsto più basso" if segni == {-1} else "segno diverso ai due orizzonti")
            frasi.append(f"«{V.DESCRIZIONI[v]}» (coefficiente standardizzato {r.loc[hz[0], 'coef_std_finale']:+.3f} a h = {hz[0]} e {r.loc[hz[1], 'coef_std_finale']:+.3f} a h = {hz[1]}: {verso}; "
                         + "; ".join(f"stesso segno in {int(r.loc[h, 'sottoperiodi_stesso_segno'])} sottoperiodi su {int(r.loc[h, 'n_sottoperiodi'])} a h = {h}" for h in hz)
                         + f": **segno {'stabile' if stab else 'non stabile'} nel tempo**)")
        return f"- **{M.NOMI[m]}.** Pesa di più " + (frasi[0] if len(frasi) == 1 else frasi[0] + "; poi " + frasi[1]) + "."

    cc1 = C1[C1["modello"].isin(M.BLOCCHI)].copy()
    cc1["variabile"] = cc1["variabile"].map(V.DESCRIZIONI)
    t_co = md(cc1[["modello", "h", "variabile", "coef_std_finale", "coef_std_medio", "quota_origini_stesso_segno", "sottoperiodi_stesso_segno", "n_sottoperiodi", "coef_per_sottoperiodo"]].rename(columns={
        "modello": "blocco", "coef_std_finale": "coefficiente standardizzato, ultima origine", "coef_std_medio": "media sulle origini", "quota_origini_stesso_segno": "quota di origini con lo stesso segno",
        "sottoperiodi_stesso_segno": "sottoperiodi con lo stesso segno", "n_sottoperiodi": "sottoperiodi stimati", "coef_per_sottoperiodo": "coefficiente per sottoperiodo (regressioni separate)"}), ".3f")

    # ------------------------------------------------------------------ scelte
    def frase_scelta(s: dict, A, v: str, lo: str, hi: str, rif: str, cosa: str, extra) -> str:
        if not s["scelto"]:
            return f"**{cosa}: nessun modello è significativamente migliore del riferimento dopo Holm (né a h = 1 né a h = 4). Resta {rif}.**"
        m = s["scelto"]
        return (f"**{cosa}: la regola sceglie {M.NOMI[m]}** (significativo dopo Holm a h = {' e h = '.join(str(h) for h in s['significativo_per_orizzonte'][m])}; "
                f"guadagno medio sui due orizzonti {pct(s['guadagno_medio'][m])}). Modelli significativi: {', '.join(s['significativi'])}. "
                + "; ".join(f"h = {h}: {ic(A.loc[(m, h), v], A.loc[(m, h), lo], A.loc[(m, h), hi])}" for h in hz) + ". " + extra(m))

    def cautele_t1(m: str) -> str:
        out = []
        if any(A1.loc[(m, h), "r2_ci_lo"] <= 0 for h in hz):
            out.append("l'intervallo bootstrap del guadagno comprende lo zero a " + " e ".join(f"h = {h}" for h in hz if A1.loc[(m, h), "r2_ci_lo"] <= 0))
        if all(A1.loc[(m, h), "dm_p"] >= 0.05 for h in hz):
            out.append("il test di Diebold-Mariano (che non corregge per i parametri in più) non è significativo a nessun orizzonte")
        out.append(f"il campione di valutazione parte dal {pd.Timestamp(A1.loc[(m, 1), 'prima_data_obiettivo']).strftime('%d/%m/%Y')} ({int(A1.loc[(m, 1), 'n'])} previsioni a h = 1, contro le {int(m1.loc[1, 'n'])} di M1)")
        return "Cautele: " + "; ".join(out) + "."

    f_s1 = frase_scelta(s1, A1, "r2_vs_m1", "r2_ci_lo", "r2_ci_hi", "M1", "Chiusura", cautele_t1)
    f_s2 = frase_scelta(s2, A2, "skill_vs_base", "skill_ci_lo", "skill_ci_hi", "il riferimento (logistica con le variabili di M1)", "Direzione", lambda m: "")
    sp = g1[(g1["gruppo"] == "sottoperiodo")]

    def per_sottoperiodo(m: str) -> str:
        return "; ".join(f"h = {h}: " + ", ".join(f"{r.valore} {pct(r.r2_vs_m1)} ({n_casi(r.n)} casi)" for r in sp[(sp["modello"] == m) & (sp["h"] == h)].itertuples()) for h in hz)

    if t3_ok:
        def frase_t3(m: str) -> str:
            return "; ".join(f"h = {h}: CRPS {ic(B3.loc[(m, m, h), 'crps_vs_d4'], B3.loc[(m, m, h), 'crps_ci_lo'], B3.loc[(m, m, h), 'crps_ci_hi'])} rispetto a D4 (p = {n(B3.loc[(m, m, h), 'p'], 3)}), "
                             f"copertura della fascia all'80% {pc(B3.loc[(m, m, h), 'copertura_80'])} contro {pc(B3.loc[(m, 'D4', h), 'copertura_80'])} di D4, "
                             f"larghezza {n(B3.loc[(m, m, h), 'larghezza_vix_80'], 1)} punti contro {n(B3.loc[(m, 'D4', h), 'larghezza_vix_80'], 1)}" for h in hz)
        f_t3 = " ".join(f"{M.NOMI[m]}: {frase_t3(m)}." for m in info["t3_eseguito_per"])
        inc = "; ".join(f"{M.NOMI[m]} a h = {h}: {int(B3.loc[(m, m, h), 'quantili_incrociati'])} previsioni su {int(B3.loc[(m, m, h), 'n'])} (D4: {int(B3.loc[(m, 'D4', h), 'quantili_incrociati'])})"
                        for m in info["t3_eseguito_per"] for h in hz)
    else:
        f_t3, inc = "T3 non eseguito: nessun modello scelto dalle regole.", "non applicabile"

    ev13_1 = g1[(g1["gruppo"] == "fascia_13_22") & (g1["valore"] == "True")].set_index(["modello", "h"])
    ev13_2 = g2[(g2["gruppo"] == "fascia_13_22") & (g2["valore"] == "True")].set_index(["modello", "h"])
    pos13 = [f"{m} a h = {h}" for m in M.MODELLI for h in hz if ev13_1.loc[(m, h), "r2_ci_lo"] > 0]
    neg13 = [f"{m} a h = {h}" for m in M.MODELLI for h in hz if ev13_1.loc[(m, h), "r2_ci_hi"] < 0]
    pos13d = [f"{m} a h = {h}" for m in M.MODELLI for h in hz if ev13_2.loc[(m, h), "skill_ci_lo"] > 0]
    n13 = info["per_orizzonte"]["1"]["n_fascia_13_22"]
    pen = info["x6_penalita"]
    gmax = str(max(info["griglia_ridge"]))
    scelte_testo = "; ".join(f"{KA.NOMI_COMPONENTI[c]}: " + ", ".join(f"h = {h} {v}" for h, v in d.items()) for c, d in p0["scelte"].items())
    primo_x = {m: pd.Timestamp(A1.loc[(m, 1), "prima_data_obiettivo"]).strftime("%d/%m/%Y") for m in M.MODELLI}
    inizio_blocchi = var.groupby("blocco", sort=False)["prima_settimana_valida"].max().to_dict()
    mm = var[var["settimane_mancanti_dopo_l_inizio"] > 0]
    mancanti = ", ".join(f"{r.variabile} {r.settimane_mancanti_dopo_l_inizio}" for r in mm.itertuples()) if len(mm) else "nessuna"
    senza = "; ".join(f"{k}: {v['n']}" + (f" ({', '.join(v['date'])}{'…' if v['n'] > len(v['date']) else ''})" if v["n"] else "") for k, v in info["ultima_seduta_senza_valore"].items())

    # ------------------------------------------------------------------ sintesi (10 righe)
    best_m = max(mercato, key=lambda m: s1["guadagno_medio"][m])
    sintesi = [
        f"**Candele (Parte 0).** Con ampiezze medie al posto delle mediane, a 1 settimana il corpo medio previsto passa dal {pc(P.loc[('K2', 1), 'rapporto_corpo'])} "
        f"all'{pc(c1n)} di quello reale e l'escursione dal {pc(P.loc[('K2', 1), 'rapporto_escursione'])} al {pc(e1n)}; a 4 settimane corpo {pc(P.loc[(K_NEW, 4), 'rapporto_corpo'])}, "
        f"escursione {pc(P.loc[(K_NEW, 4), 'rapporto_escursione'])}. Restano sotto il 100% perché negli anni valutati corpi e ombre sono stati più ampi della media di tutta la storia.",
        f"**Regressione o media storica.** La regressione batte la media storica solo per: {', '.join(usa_reg) if usa_reg else 'nessuna componente'}. Per tutto il resto il programma usa la media storica; la direzione del corpo è la frequenza storica (L0).",
        f"**Etichetta della chiusura** (rispetto all'origine, probabilità D1, «neutra» tra 40% e 60%): è neutra nel {pc(E.loc[1, 'quota_neutra'], 0)} dei casi a 1 settimana e nel "
        f"{pc(E.loc[4, 'quota_neutra'], 0)} a 4; quando non è neutra è corretta nel {pc(E.loc[1, 'accuratezza_non_neutre'], 0)} dei casi a 1 settimana ({int(E.loc[1, 'n_non_neutre'])} casi) e nel "
        f"{pc(E.loc[4, 'accuratezza_non_neutre'], 0)} a 4 ({int(E.loc[4, 'n_non_neutre'])} casi).",
        f"**VIX giornaliero da solo (G1): {esito('G1', A1, 'r2_vs_m1', 'r2_ci_lo', 'r2_ci_hi', 'p_holm')} sulla chiusura** ({frase_t1('G1')}) e "
        f"{esito('G1', A2, 'skill_vs_base', 'skill_ci_lo', 'skill_ci_hi', 'p_base_holm')} sulla direzione.",
        f"**Dati di mercato.** Il blocco migliore sulla chiusura è {M.NOMI[best_m]}: {frase_t1(best_m)}. Gli altri blocchi: "
        + "; ".join(f"{m} {pct(A1.loc[(m, 1), 'r2_vs_m1'])} a h = 1 e {pct(A1.loc[(m, 4), 'r2_vs_m1'])} a h = 4" for m in mercato if m != best_m) + ".",
        "**Combinazioni.** " + "; ".join(f"{M.NOMI[m]}: {pct(A1.loc[(m, 1), 'r2_vs_m1'])} a h = 1 e {pct(A1.loc[(m, 4), 'r2_vs_m1'])} a h = 4 sulla chiusura" for m in M.COMBINAZIONI)
        + (": mettere insieme tutte le variabili non fa meglio del blocco migliore da solo." if max(s1["guadagno_medio"][m] for m in M.COMBINAZIONI) <= s1["guadagno_medio"][best_m]
           else ": una combinazione fa meglio del blocco migliore da solo."),
        f_s1,
        f_s2 + " Sulla direzione il blocco con il guadagno medio più alto è " + f"{M.NOMI[migliore_t2]} ({frase_t2(migliore_t2)}).",
        f"**Fasce (T3).** {f_t3}",
        f"**Fascia 13-22** ({n13} origini a h = 1 per i modelli con tutta la storia). Guadagno sulla chiusura con intervallo sopra zero: {', '.join(pos13) if pos13 else 'nessun modello'}; "
        f"sulla direzione: {', '.join(pos13d) if pos13d else 'nessun modello'}. **Nulla di tutto questo è entrato nel programma di previsione**, che cambia solo per la Parte 0.",
    ]

    out = f"""# Fase 3 — Informazioni nuove: VIX giornaliero e dati di mercato

*Generato da `python -m src.run_phase3`. Campione di sviluppo: {info['n_settimane']} settimane dal {info['campione'][0]} al {info['campione'][1]}; dati giornalieri fino al {info['ultima_data_giornaliera']}.
L'holdout (VIX ed esterni) non è stato letto: il comando gira sotto la guardia. Tutti i numeri vengono da `reports/fase3/`.*

## Sintesi

{chr(10).join(f'{i}. {s}' for i, s in enumerate(sintesi, start=1))}

## 1. Parte 0 — Nuova definizione delle candele

**Decisioni applicate** (prese prima di questa fase): costruzione K2 confermata; ampiezza del corpo e ombre previste come **media** condizionata, non più mediana; direzione del corpo dalla
frequenza storica (L0); ombre e ampiezza dalla media storica quando la regressione non la batte; etichetta sul grafico = direzione della chiusura rispetto alla chiusura dell'origine; linea di
riferimento principale = media a 12 settimane.

**Cambio di definizione, dichiarato.** Nella Fase 2c la candela mostrava «la settimana più probabile» (mediane). Ora mostra «una settimana di dimensioni medie»: corpo ed escursione sono medie,
quindi più grandi delle mediane (le ampiezze hanno una coda lunga a destra). È una scelta nostra, non il risultato di una selezione sui dati.

**Regola per «batte»** (fissata nel codice prima di guardare i risultati, perché la consegna non la precisava): la regressione è usata solo se nel walk-forward di sviluppo ha errore quadratico medio
più basso della media storica **e** il test di Diebold-Mariano ha p < {n(KA.P_BATTE, 2)}, separatamente per ogni componente e orizzonte. Esito: {scelte_testo}.

### 1.1 Regressione contro media storica

Ampiezze in logaritmi (0,10 ≈ 10%). La media storica è quella delle settimane note all'origine (finestra espandibile).

{t_amp}

Nota: la previsione media ({n(a1.loc[('corpo', 1), 'previsione_media_storica'], 3)} per il corpo a 1 settimana) è più bassa della media reale delle settimane valutate ({n(a1.loc[('corpo', 1), 'media_reale'], 3)}):
le medie sono calcolate su tutta la storia fino all'origine, e negli anni valutati corpi e ombre sono stati più ampi che all'inizio del campione. È questo, e non più la differenza tra media e mediana,
a tenere i rapporti sotto 1.

### 1.2 Nuovi rapporti corpo ed escursione previsti / reali

Stesse origini della Fase 2c ({p0['per_orizzonte']['1']['n_previsioni']} previsioni a 1 settimana, date previste dal {p0['per_orizzonte']['1']['prima_data_obiettivo']} al {p0['per_orizzonte']['1']['ultima_data_obiettivo']}).
Errori medi in punti VIX.

{t_pag}

- A 1 settimana il rapporto del corpo è {n(c1n, 3)} ({'dentro l' if in_vincolo(c1n) else 'fuori dall'}'intervallo 0,8–1,2 usato come criterio di realismo nella Fase 2c) e quello dell'escursione è {n(e1n, 3)} ({'dentro l' if in_vincolo(e1n) else 'fuori dall'}'intervallo).
- Sovrapposizione con la candela reale: {'più alta a tutti gli orizzonti' if sale('sovrapposizione') else 'non più alta a tutti gli orizzonti'}. Quota di chiusure reali dentro l'escursione prevista: {'più alta a tutti gli orizzonti' if sale('chiusura_nell_escursione') else 'non più alta a tutti gli orizzonti'}.
- **Errore sulla chiusura della candela: {'più alto' if peggio_ch else 'non più alto'} a tutti gli orizzonti** ({n(P.loc[(K_NEW, 1), 'mae_chiusura'], 2)} punti contro {n(P.loc[('K2', 1), 'mae_chiusura'], 2)} a 1 settimana): un corpo più grande in una direzione fissa porta la chiusura della candela più lontano dal centro.
  La previsione centrale della chiusura resta la mediana D4, non la chiusura della candela.

### 1.3 Etichetta della chiusura

Sopra ogni candela prevista è scritta la direzione della chiusura rispetto alla chiusura dell'origine (per la settimana +1 coincide con la chiusura della settimana prima; le settimane +2 e +3 seguono
la stessa regola della +4). La probabilità è quella del modello D1, già usato per le probabilità di superare 20, 25 e 30: nessun modello nuovo. Tra 40% e 60% l'etichetta è «neutra».

{t_et}

L'etichetta è neutra {'nella maggior parte dei casi a tutti gli orizzonti' if neutra_magg else 'in una parte rilevante dei casi'}. Quando indica «ribasso» {'è corretta più spesso di quando indica «rialzo» a tutti gli orizzonti' if ribasso_meglio else 'non è sempre più corretta di quando indica «rialzo»'}; i casi «rialzo» sono {int(E.loc[1, 'n_rialzo'])} a 1 settimana e {int(E.loc[4, 'n_rialzo'])} a 4.

### 1.4 Grafico e programma

- Linea di riferimento principale: media a 12 settimane («livello recente»), spessa e prolungata fino alla settimana +4; la mediana storica è una linea grigia sottile.
- Verifica visiva sul passato rifatta con le nuove candele: {p0['g3']['riga']}

![Verifica sul passato](figures/fase3/g3_verifica_passato.png)

- Il registro non riscrive mai un'origine già registrata: le previsioni scritte prima di questa fase restano con le candele di allora (colonna `costruzione` = K2); le nuove hanno `costruzione` = {K_NEW}
  e due colonne in più (`p_chiusura_sopra_origine`, `etichetta_chiusura`). `--verifica` misura anche l'esito dell'etichetta.

## 2. Dati e variabili

Serie usate: {', '.join(info['serie_usate'])} (giornaliere, scaricate nella Fase 2). Ritardi di pubblicazione dall'inventario: {', '.join(f'{k} {v}' for k, v in info['ritardi_giorni'].items())} giorni.
Esclusi per decisione: indice delle condizioni finanziarie e differenziali di credito (non vengono nemmeno letti).

**Regola di allineamento** (fissata prima della valutazione). La settimana t va dal giorno dopo la data settimanale precedente fino all'ultima seduta della settimana t. Di ogni serie si prende
l'ultimo valore dentro la settimana; per i rapporti, l'ultima seduta della settimana in cui tutte le serie del rapporto hanno un valore. Se nella settimana non c'è nessun valore la variabile è
mancante e quell'origine esce dalla stima e dalla valutazione del blocco: nessun valore viene portato avanti.

{t_var}

## 3. Protocollo

- Ogni modello = variabili di M1 (ultima chiusura, media a 4 e a 12 settimane, in log) + variabili del blocco. Walk-forward con finestra espandibile e ristima a ogni passo.
- Un'origine è ammessa quando il blocco ha almeno {M.MIN_VALIDE} settimane valide: per i blocchi con tutta la storia è la regola di M1. Prime date previste a h = 1: {', '.join(f'{m} {d}' for m, d in primo_x.items())}.
- **Riferimento sulle stesse origini.** Il riferimento è quello vero, stimato su tutta la storia dal 2000 (M1; frequenza storica e logistica di base; D4), valutato sulle sole origini del modello.
  Per T1 è riportato anche il confronto con M1 stimato sulla stessa finestra più corta del blocco.
- T1: Clark-West a una coda (il modello contiene M1), Holm su {info['n_test_holm']['t1']} test. T2: Diebold-Mariano sul Brier, Holm su {info['n_test_holm']['t2_base']} test contro la logistica di base e su {info['n_test_holm']['t2_freq']} contro la frequenza storica.
- X6: penalità ridge solo sulle variabili dei blocchi, intensità scelta nella griglia {info['griglia_ridge']} (× numero di coppie) sulle ultime {info['validazione_ridge']} coppie della finestra di stima. La penalità massima
  ({gmax}) è stata scelta nel {pc(pen['1']['T1'][gmax], 0)} delle origini a h = 1 e nel {pc(pen['4']['T1'][gmax], 0)} a h = 4 per T1 (con la penalità massima X6 resta vicino a M1 stimato sulla stessa finestra).
- X7: media semplice delle previsioni dei cinque blocchi, sulle origini in cui esistono tutte.
- Intervalli di confidenza: bootstrap a blocchi di 26 settimane, 2000 repliche.

## 4. Tabella principale

### 4.1 T1 — Chiusura (log VIX)

Riferimento M1 su tutte le sue origini: RMSE log {n(m1.loc[1, 'rmse_log'], 4)} a h = 1 ({int(m1.loc[1, 'n'])} previsioni) e {n(m1.loc[4, 'rmse_log'], 4)} a h = 4 ({int(m1.loc[4, 'n'])}).
In ogni riga «M1» è il riferimento ricalcolato sulle origini del modello.

{t_t1}

In punti VIX e quota di previsioni entro il 10% e il 20% dal valore reale:

{t_t1b}

![Guadagni rispetto al riferimento](figures/fase3/guadagni_vs_riferimento.png)

![Errore cumulato rispetto a M1](figures/fase3/t1_errore_cumulato_vs_m1.png)

### 4.2 T2 — Direzione della chiusura rispetto all'origine

Evento: chiusura a t + h sopra la chiusura a t. Riferimenti: frequenza storica e logistica con le sole variabili di M1.

{t_t2}

Accuratezza e quota di settimane quando la probabilità della direzione prevista è almeno 60%, 65% e 70% (soglie fissate prima):

{t_t2s}

Calibrazione della logistica di base e del blocco con il guadagno medio più alto ({M.NOMI[migliore_t2]}); tutte le classi sono in `reports/fase3/t2_calibrazione.csv`:

{t_cal}

### 4.3 T3 — Fasce

Eseguito solo per i modelli scelti dalle regole ({', '.join(info['t3_eseguito_per']) if t3_ok else 'nessuno'}). Regressione quantilica come D4 con le variabili del blocco in più; riferimento D4 sulle stesse origini.

{t_t3}

## 5. Fascia 13-22 e casi dopo un rialzo oltre il 30%

Fascia 13-22: chiusura all'origine tra 13 e 22, estremi inclusi. Nelle tabelle di questa sezione i p-value non sono corretti per i confronti multipli: servono a descrivere, non a scegliere.

### 5.1 Fascia 13-22 — Chiusura

{gruppo_t1('fascia_13_22', 'True')}

### 5.2 Fascia 13-22 — Direzione

{gruppo_t2('fascia_13_22', 'True')}

{('### 5.3 Fascia 13-22 — Fasce' + chr(10) + chr(10) + gruppo_t3('fascia_13_22', 'True')) if t3_ok else ''}

Guadagno sulla chiusura con intervallo bootstrap tutto sopra zero: {', '.join(pos13) if pos13 else 'nessun modello'}. Con intervallo tutto sotto zero: {', '.join(neg13) if neg13 else 'nessun modello'}.

### 5.4 Dopo un rialzo oltre il 30% in 4 settimane — Chiusura

{gruppo_t1('dopo_rialzo_30', 'True')}

### 5.5 Dopo un rialzo oltre il 30% in 4 settimane — Direzione

{gruppo_t2('dopo_rialzo_30', 'True')}

I casi sono pochi ({info['per_orizzonte']['1']['n_dopo_rialzo_30']} con tutta la storia, meno per i blocchi che partono più tardi): gli intervalli sono larghi e nessuna conclusione è solida.

## 6. Sottoperiodi e fasce di partenza

R² rispetto a M1 (tra parentesi il numero di previsioni). Tutte le metriche sono in `reports/fase3/t1_gruppi.csv`, `t2_gruppi.csv`{' e `t3_gruppi.csv`' if t3_ok else ''}.

{pivot(g1, 'sottoperiodo', 'r2_vs_m1')}

{pivot(g1, 'fascia_origine', 'r2_vs_m1', list(C.BAND_LABELS))}

Guadagno sul Brier rispetto alla logistica di base:

{pivot(g2, 'sottoperiodo', 'skill_vs_base')}

{pivot(g2, 'fascia_origine', 'skill_vs_base', list(C.BAND_LABELS))}

{('Fasce (T3) per sottoperiodo:' + chr(10) + chr(10) + gruppo_t3('sottoperiodo')) if t3_ok else ''}

## 7. Quanto aggiungono le informazioni nuove

**VIX giornaliero da solo (G1).** Sulla chiusura: {frase_t1('G1')}. Sulla direzione: {frase_t2('G1')}.
In sintesi, rispetto alle tre variabili settimanali di M1, sapere come il VIX si è mosso dentro la settimana porta a: **{esito('G1', A1, 'r2_vs_m1', 'r2_ci_lo', 'r2_ci_hi', 'p_holm')}** sulla chiusura; **{esito('G1', A2, 'skill_vs_base', 'skill_ci_lo', 'skill_ci_hi', 'p_base_holm')}** sulla direzione.

**Dati di mercato.**

{chr(10).join(f"- {M.NOMI[m]}. Chiusura: {frase_t1(m)}. Direzione: {frase_t2(m)}." for m in mercato)}

**Combinazioni.**

{chr(10).join(f"- {M.NOMI[m]}. Chiusura: {frase_t1(m)}. Direzione: {frase_t2(m)}." for m in M.COMBINAZIONI)}

In punti VIX le differenze sono piccole in ogni caso: l'errore assoluto medio a h = 1 del blocco migliore sulla chiusura ({best_m}) è {n(A1.loc[(best_m, 1), 'mae_vix'], 2)} contro {n(A1.loc[(best_m, 1), 'mae_vix_m1'], 2)} di M1 sulle stesse origini,
e la quota di previsioni entro il 10% dal valore reale è {pc(A1.loc[(best_m, 1), 'entro_10'])} contro {pc(A1.loc[(best_m, 1), 'entro_10_m1'])}.

## 8. Quali variabili pesano

Coefficienti standardizzati (deviazioni standard dell'obiettivo per deviazione standard della variabile) della regressione T1. La stabilità del segno è giudicata con regressioni separate per
sottoperiodo (almeno {104} coppie): «stabile» solo se il segno è lo stesso in tutti i sottoperiodi e ai due orizzonti.

{chr(10).join(frase_pesi(m) for m in M.BLOCCHI)}

{t_co}

## 9. Modelli scelti dalle regole

- {f_s1}
- {f_s2}
- {'Per sottoperiodo il guadagno di ' + s1['scelto'] + ' sulla chiusura è: ' + per_sottoperiodo(s1['scelto']) + '.' if s1['scelto'] else 'Nessun modello sostituisce M1 per la chiusura.'}
- **Fasce.** {f_t3}
- **Nessuno di questi modelli è stato integrato in `src.forecast`**: in questa fase si misura soltanto.

## 10. Cosa servirebbe per portarli nel programma

1. **Aggiornamento dei dati a ogni esecuzione.** Oggi `src.forecast` legge solo il file settimanale del VIX. Per usare un blocco di mercato dovrebbe scaricare (o ricevere) a ogni esecuzione le serie giornaliere
   necessarie{' (per ' + s1['scelto'] + ': ' + ', '.join(V.SERIE_PER_BLOCCO.get(s1['scelto'], info['serie_usate'])) + ')' if s1['scelto'] else ''}, aggiornate fino all'ultima seduta della settimana di origine.
2. **Controllo di allineamento.** Se l'ultima data della serie esterna è precedente all'ultima seduta della settimana di origine, la variabile è mancante: il programma deve dirlo e ripiegare su M1/D4, non usare un valore vecchio.
3. **Permesso di lettura.** `src.forecast` è l'unico comando autorizzato a leggere il VIX successivo all'8 novembre 2024, e solo come storia. Lo stesso permesso andrebbe esteso ai dati esterni, con la stessa regola; la parte di holdout
   dei dati esterni resta chiusa fino alla valutazione finale.
4. **Coerenza tra i pezzi.** Se la chiusura centrale venisse dal blocco scelto, fasce e probabilità dovrebbero venire dallo stesso insieme di variabili (T3), altrimenti centro, fasce e probabilità userebbero informazioni diverse.
5. **Storia più corta.** Alcuni blocchi di mercato hanno meno storia di M1 (prima settimana valida: {', '.join(f'{b} {d}' for b, d in inizio_blocchi.items())}): le stime sono meno precise e i campioni di valutazione più recenti non contengono il 2008.
6. **Riproducibilità.** I dati esterni non sono nel repository e i fornitori possono rivederli: servirebbe salvare a ogni esecuzione i valori usati (come fa il registro per le previsioni).

## 11. Problemi e ambiguità

1. **«Batte» non era definito** per la scelta tra regressione e media storica (Parte 0): è stato fissato come errore quadratico più basso e Diebold-Mariano con p < {n(KA.P_BATTE, 2)}, per componente e orizzonte. Con un criterio meno severo
   (solo errore più basso) la regressione sarebbe usata in più casi: la tabella 1.1 permette di vederlo.
2. **Le candele restano più piccole di quelle reali** anche con le medie (corpo {pc(c1n, 0)}, escursione {pc(e1n, 0)} a 1 settimana), perché corpi e ombre sono cresciuti nel tempo e la media espandibile li insegue.
3. **Etichetta.** La consegna la definiva per le settimane +1 e +4; per +2 e +3 è stata usata la stessa regola (rispetto all'origine). La probabilità viene da D1, non dai modelli T2 di questa fase, che non sono nel programma.
4. **Registro.** Una previsione già registrata prima di questa fase resta con le candele della Fase 2c: registro e scheda corrente possono mostrare candele diverse per la stessa origine. Il registro non è stato modificato.
5. **VIX giornaliero.** Il file di sviluppo dei dati esterni contiene solo le chiusure giornaliere: «massimo delle ultime 10 sedute» è il massimo delle chiusure. «Massima variazione giornaliera» è stata letta come la più alta
   delle variazioni con il loro segno (non in valore assoluto). Tutte le variazioni sono differenze di logaritmo.
6. **Settimane senza valore.** Settimane mancanti dopo l'inizio della serie: {mancanti}. Settimane in cui l'ultima seduta del VIX non ha un valore dell'altra serie (dopo il suo inizio): {senza}.
   In quei casi si usa l'ultima seduta della settimana in cui ci sono tutti i valori necessari, oppure la variabile è mancante.
7. **Quale M1 come riferimento.** «Ricalcolato sulle stesse origini» è stato letto come M1 stimato su tutta la storia e valutato sulle origini del modello. Il confronto con M1 stimato sulla stessa finestra corta è in tabella e dà numeri vicini.
8. **Test.** Clark-West è usato per tutti i modelli, anche per X7 (una media di modelli, non un modello annidato in senso stretto). È un test a una coda che corregge per il rumore dei parametri in più: può essere significativo
   anche quando l'errore fuori campione è quasi uguale. Per questo sono riportati anche intervallo bootstrap e Diebold-Mariano.
9. **Campioni diversi.** I blocchi partono in date diverse ({', '.join(f'{m} {d}' for m, d in primo_x.items())}): i loro R² non sono confrontabili tra loro, solo ciascuno con M1 sulle stesse origini.
10. **Quantili incrociati in T3.** {inc}. I quantili vengono riordinati, come per D4.
11. **T3 «per il blocco scelto dalla regola».** Letto come: per ogni modello scelto da una delle due regole (chiusura o direzione).
12. **Dati esterni non pubblicati.** `python -m src.run_phase3` richiede `python -m src.fetch_external`; senza i dati esterni la Parte 0 è comunque rigenerabile da sola (`python -m src.fase3.valuta_candele`) e i test che li usano vengono saltati.
13. **Molti confronti sullo stesso campione.** Dopo cinque fasi il campione di sviluppo è stato usato per molti modelli: Holm corregge dentro questa fase, non tra le fasi. La conferma vera resta l'holdout, mai toccato.
14. **Domande ancora aperte dalle fasi precedenti:** se pubblicare `data/external/`; i file completi dei differenziali di credito (qui esclusi); l'uso in D1, nel programma, degli errori passati successivi all'8 novembre 2024 come dato d'ingresso.
"""
    (C.REPORTS_DIR / "fase3_report.md").write_text(re.sub(r"\n{3,}", "\n\n", out))


if __name__ == "__main__":
    run()
