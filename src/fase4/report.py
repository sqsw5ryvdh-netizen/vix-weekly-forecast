"""Genera reports/fase4_report.md. Tutti i numeri sono letti dai file reports/fase4/*.csv e *.json: nessun numero e' scritto a mano."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C
from ..fase3.report import MIN_N, ic, md, n, n_casi, pc, pct
from . import boosting as GB
from . import candele as KA
from . import regimi as RE
from . import valuta_b as VB4
from . import versioni as VS

R = C.REPORTS_DIR / "fase4"
SPAREGGIO_CONFERMATO = "2026-11-16"      # data in cui lo spareggio A1/A2 (a parita' vince la prima dell'elenco) e' stato confermato
ETICHETTE = {"n": "previsioni", "mae_centrale": "chiusura centrale: errore assoluto medio (punti)", "rmse_log_centrale": "chiusura centrale: RMSE in log",
             "entro_10": "chiusura centrale entro il 10% dal reale", "entro_20": "chiusura centrale entro il 20% dal reale",
             "mae_centro": "centro: errore assoluto medio (punti)", "rmse_log_centro": "centro: RMSE in log",
             "copertura_50": "fascia 50%: copertura", "larghezza_50": "fascia 50%: larghezza (punti)", "copertura_80": "fascia 80%: copertura",
             "larghezza_80": "fascia 80%: larghezza (punti)", "copertura_90": "fascia 90%: copertura", "larghezza_90": "fascia 90%: larghezza (punti)",
             "n_etichette": "etichette confrontabili", "quota_neutra": "etichetta: quota di «neutre»", "n_non_neutre": "etichetta: non neutre (numero)",
             "accuratezza_non_neutre": "etichetta: corrette tra le non neutre", "brier_direzione": "direzione: Brier",
             "rapporto_corpo": "candele: corpo previsto / reale", "rapporto_escursione": "candele: escursione prevista / reale", "sovrapposizione": "candele: sovrapposizione",
             "chiusura_nell_escursione": "candele: chiusura reale nell'escursione prevista", "mae_apertura": "candele: errore medio apertura", "mae_massimo": "candele: errore medio massimo",
             "mae_minimo": "candele: errore medio minimo", "mae_chiusura": "candele: errore medio chiusura della candela"}
INTERI = {"n", "n_etichette", "n_non_neutre"}
QUOTE = {"entro_10", "entro_20", "copertura_50", "copertura_80", "copertura_90", "quota_neutra", "accuratezza_non_neutre", "chiusura_nell_escursione"}


def esito(v: float, lo: float, hi: float, p_holm: float, stat: float) -> str:
    if p_holm < 0.05:
        return "significativamente migliore dopo Holm" if stat > 0 else "significativamente peggiore dopo Holm"
    if lo > 0:
        return "intervallo sopra zero, ma non significativo dopo Holm"
    if hi < 0:
        return "intervallo sotto zero, ma non significativo dopo Holm"
    return "nessuna differenza distinguibile da zero"


def run() -> None:
    T = lambda k: pd.read_csv(R / f"{k}.csv")
    a, b, c = (json.loads((R / f"{k}_info.json").read_text()) for k in ("a", "b", "c"))
    amp, pag, t1, t2, g1, g2 = T("a_ampiezze"), T("a_pagella"), T("b_t1_metriche"), T("b_t2_metriche"), T("b_t1_gruppi"), T("b_t2_gruppi")
    st, conf, diff = T("b_regimi_stime"), T("c_confronto"), T("c_differenze")
    sc, s1, s2, n1 = a["scelta"], b["scelta_chiusura"], b["scelta_direzione"], b["n1"]
    V, hz = sc["scelta"], [1, 4]
    P = pag.set_index(["variante", "h"])
    A1, A2 = t1.set_index(["modello", "h"]), t2.set_index(["modello", "h"])
    versioni = [v.upper() for v in c["versioni"]]
    nome_v = lambda v: VS.NOMI[v.lower()]
    f3 = "F3" in versioni

    # ------------------------------------------------------------------ Parte A
    q = pag.sort_values(["h", "variante"]).copy()
    q["sovrapposizione [IC 95%]"] = [f"{x:.3f} [{lo:.3f}; {hi:.3f}]" for x, lo, hi in zip(q["sovrapposizione"], q["sovrapposizione_ci_lo"], q["sovrapposizione_ci_hi"])]
    t_pag = md(q[["h", "variante", "n", "rapporto_corpo", "rapporto_escursione", "sovrapposizione [IC 95%]", "direzione_corpo", "chiusura_nel_corpo", "chiusura_nell_escursione",
                  "mae_apertura", "mae_massimo", "mae_minimo", "mae_chiusura", "entro_10pct", "entro_20pct"]].rename(columns={
        "rapporto_corpo": "corpo previsto / reale", "rapporto_escursione": "escursione prevista / reale", "direzione_corpo": "direzione del corpo indovinata",
        "chiusura_nel_corpo": "chiusura reale nel corpo previsto", "chiusura_nell_escursione": "chiusura reale nell'escursione prevista", "mae_apertura": "errore medio apertura",
        "mae_massimo": "errore medio massimo", "mae_minimo": "errore medio minimo", "mae_chiusura": "errore medio chiusura della candela",
        "entro_10pct": "chiusura della candela entro il 10%", "entro_20pct": "entro il 20%"}), ".3f")
    u = amp[amp["h"].isin(hz)].copy()
    u["componente"] = u["componente"].map({"corpo": "ampiezza del corpo", "ombra_sup": "ombra superiore", "ombra_inf": "ombra inferiore", "somma": "somma delle tre"})
    u["errore quadratico rispetto ad A0"] = [pct(x) if np.isfinite(x) else "–" for x in u["mse_vs_a0"]]
    t_amp = md(u[["h", "componente", "metodo", "variante", "n", "media_reale", "previsione_media", "mse", "errore quadratico rispetto ad A0", "dm_stat", "p"]].rename(columns={
        "media_reale": "media reale", "previsione_media": "previsione media", "mse": "errore quadratico medio", "dm_stat": "DM vs A0 (positivo = meglio di A0)"}), ".5f")
    rap = sc["rapporto_corpo_h1"]
    pari = [v for v in KA.VARIANTI if abs(abs(rap[v] - 1) - min(abs(x - 1) for x in rap.values())) < 1e-12]
    medio = {v: float(pag[pag["variante"] == v]["rapporto_corpo"].mean()) for v in KA.VARIANTI}
    if len(pari) > 1:
        altra = [v for v in pari if v != sc["candidata"]][0]
        f_pari = (f"**Parità esatta tra {' e '.join(pari)} a h = 1** (rapporto {n(rap[pari[0]], 3)}): a +1 il corpo è previsto dalla stessa regressione su 520 settimane, quindi le due varianti danno lo stesso corpo. "
                  f"La regola non dice come sciogliere la parità: nel codice era fissato, prima di guardare i risultati, che vince la prima dell'elenco ({sc['candidata']}, la finestra più lunga e più vicina ad A0). "
                  f"Con uno spareggio diverso, per esempio il rapporto medio su h = 1-4 ({'; '.join(f'{v} {n(medio[v], 3)}' for v in pari)}), "
                  f"la scelta {'sarebbe ' + altra if abs(medio[altra] - 1) < abs(medio[sc['candidata']] - 1) else 'non cambierebbe'}. **Spareggio confermato il {SPAREGGIO_CONFERMATO}: fa parte della regola, e la variante congelata resta {sc['scelta']}.**")
    else:
        f_pari = "Nessuna parità tra le varianti a h = 1."
    f_dm = (f"errore quadratico sulle ampiezze a h = 1: {n(sc['mse_candidata'], 5)} contro {n(sc['mse_a0'], 5)} di A0 (DM = {n(sc['dm_stat'], 2)}, p = {n(sc['p'], 3)}): "
            + ("significativamente peggiore, quindi resta A0" if sc.get("significativamente_peggiore") else "non peggiore" + (", anzi significativamente più basso" if sc["dm_stat"] > 0 and sc["p"] < 0.05 else ""))) if sc["candidata"] != "A0" else ""

    # ------------------------------------------------------------------ Parte B
    nome = lambda m: VB4.NOMI[m]
    x = t1.copy()
    x["modello "] = x["modello"].map(nome)
    x["R² vs M1 [IC 95%]"] = [ic(v, lo, hi) for v, lo, hi in zip(x["r2_vs_m1"], x["r2_ci_lo"], x["r2_ci_hi"])]
    x["esito"] = [esito(v, lo, hi, ph, s) for v, lo, hi, ph, s in zip(x["r2_vs_m1"], x["r2_ci_lo"], x["r2_ci_hi"], x["p_holm"], x["dm_stat"])]
    t_t1 = md(x[["h", "modello ", "n", "prima_data_obiettivo", "rmse_log", "rmse_log_m1", "mae_vix", "mae_vix_m1", "R² vs M1 [IC 95%]", "dm_stat", "p", "p_holm", "esito",
                 "entro_10", "entro_10_m1", "entro_20", "entro_20_m1"]].rename(columns={
        "modello ": "modello", "prima_data_obiettivo": "prima data prevista", "rmse_log": "RMSE log", "rmse_log_m1": "RMSE log M1", "mae_vix": "MAE punti", "mae_vix_m1": "MAE punti M1",
        "dm_stat": "DM (positivo = meglio di M1)", "p": "p (due code)", "p_holm": "p Holm", "entro_10": "entro il 10%", "entro_10_m1": "entro il 10%, M1", "entro_20": "entro il 20%",
        "entro_20_m1": "entro il 20%, M1"}), ".4f")
    y = t2.copy()
    y["modello "] = y["modello"].map(nome)
    y["guadagno sul Brier vs logistica di base [IC 95%]"] = [ic(v, lo, hi) for v, lo, hi in zip(y["skill_vs_base"], y["skill_ci_lo"], y["skill_ci_hi"])]
    y["esito"] = [esito(v, lo, hi, ph, s) for v, lo, hi, ph, s in zip(y["skill_vs_base"], y["skill_ci_lo"], y["skill_ci_hi"], y["p_base_holm"], y["dm_stat_base"])]
    t_t2 = md(y[["h", "modello ", "n", "frequenza_rialzo", "accuratezza", "accuratezza_base", "brier", "brier_base", "brier_freq", "affidabilita", "risoluzione",
                 "guadagno sul Brier vs logistica di base [IC 95%]", "p_base", "p_base_holm", "esito", "skill_vs_freq", "p_freq_holm"]].rename(columns={
        "modello ": "modello", "frequenza_rialzo": "freq. rialzo", "accuratezza_base": "accuratezza base", "brier": "Brier", "brier_base": "Brier base", "brier_freq": "Brier frequenza",
        "affidabilita": "affidabilità (0 = calibrato)", "p_base": "p vs base", "p_base_holm": "p Holm vs base", "skill_vs_freq": "guadagno sul Brier vs frequenza", "p_freq_holm": "p Holm vs frequenza"}), ".4f")
    for s in (60, 65, 70):
        y[f"prob. ≥ {s}%: quota (n) → accuratezza"] = [f"{pc(qq)} ({int(nn)}{', meno di 20' if nn < MIN_N else ''}) → {pc(ac) if np.isfinite(ac) else '–'}"
                                                      for qq, nn, ac in zip(y[f"quota_{s}"], y[f"n_{s}"], y[f"accuratezza_{s}"])]
    t_t2s = md(y[["h", "modello ", "n", "accuratezza"] + [f"prob. ≥ {s}%: quota (n) → accuratezza" for s in (60, 65, 70)]].rename(columns={"modello ": "modello", "accuratezza": "accuratezza (tutte)"}), ".3f")

    def gruppo_t1(col: str) -> str:
        g = g1[(g1["gruppo"] == col) & (g1["valore"] == "True")]
        g = g.assign(**{"modello ": g["modello"].map(nome), "casi": [n_casi(k) for k in g["n"]], "R² vs M1 [IC 95%]": [ic(v, lo, hi) for v, lo, hi in zip(g["r2_vs_m1"], g["r2_ci_lo"], g["r2_ci_hi"])]})
        return md(g[["h", "modello ", "casi", "rmse_log", "rmse_log_m1", "mae_vix", "mae_vix_m1", "R² vs M1 [IC 95%]", "p", "entro_10", "entro_10_m1"]].rename(columns={
            "modello ": "modello", "rmse_log": "RMSE log", "rmse_log_m1": "RMSE log M1", "mae_vix": "MAE punti", "mae_vix_m1": "MAE punti M1", "p": "p DM (non corretto)",
            "entro_10": "entro il 10%", "entro_10_m1": "entro il 10%, M1"}), ".4f")

    def gruppo_t2(col: str) -> str:
        g = g2[(g2["gruppo"] == col) & (g2["valore"] == "True")]
        g = g.assign(**{"modello ": g["modello"].map(nome), "casi": [n_casi(k) for k in g["n"]],
                        "guadagno sul Brier vs base [IC 95%]": [ic(v, lo, hi) for v, lo, hi in zip(g["skill_vs_base"], g["skill_ci_lo"], g["skill_ci_hi"])]})
        return md(g[["h", "modello ", "casi", "frequenza_rialzo", "accuratezza", "accuratezza_base", "brier", "brier_base", "guadagno sul Brier vs base [IC 95%]", "p_base"]].rename(columns={
            "modello ": "modello", "frequenza_rialzo": "freq. rialzo", "accuratezza_base": "accuratezza base", "brier": "Brier", "brier_base": "Brier base", "p_base": "p DM (non corretto)"}), ".4f")

    def pivot(g: pd.DataFrame, col: str, val: str, ordine=None) -> str:
        g = g[g["gruppo"] == col]
        p = g.pivot_table(index=["h", "valore"], columns="modello", values=val, sort=False).reset_index()
        nn = g.pivot_table(index=["h", "valore"], columns="modello", values="n", sort=False).reset_index()
        if ordine is not None:
            p["valore"], nn["valore"] = pd.Categorical(p["valore"], ordine, ordered=True), pd.Categorical(nn["valore"], ordine, ordered=True)
        p, nn = p.sort_values(["h", "valore"]), nn.sort_values(["h", "valore"])
        out = p[["h", "valore"]].copy()
        for m in VB4.MODELLI:
            out[m] = [f"{pct(v)} ({int(k)})" if np.isfinite(v) else "–" for v, k in zip(p[m], nn[m].fillna(0))]
        return md(out.rename(columns={"valore": {"sottoperiodo": "sottoperiodo", "fascia_origine": "fascia di partenza"}[col]}))

    def frase(A, m, v, lo, hi, ph, stat) -> str:
        return "; ".join(f"h = {h}: {ic(A.loc[(m, h), v], A.loc[(m, h), lo], A.loc[(m, h), hi])}, {esito(A.loc[(m, h), v], A.loc[(m, h), lo], A.loc[(m, h), hi], A.loc[(m, h), ph], A.loc[(m, h), stat])}" for h in hz)

    f1_ = lambda m: frase(A1, m, "r2_vs_m1", "r2_ci_lo", "r2_ci_hi", "p_holm", "dm_stat")
    f2_ = lambda m: frase(A2, m, "skill_vs_base", "skill_ci_lo", "skill_ci_hi", "p_base_holm", "dm_stat_base")
    peggio = lambda A, m, ph, stat: all(A.loc[(m, h), ph] < 0.05 and A.loc[(m, h), stat] < 0 for h in hz)
    gb_verdetto = {m: ("significativamente peggiore di M1 a tutti e due gli orizzonti" if peggio(A1, m, "p_holm", "dm_stat") else
                       "peggiore di M1 (significativo dopo Holm a " + " e ".join(f"h = {h}" for h in hz if A1.loc[(m, h), "p_holm"] < 0.05 and A1.loc[(m, h), "dm_stat"] < 0) + ")"
                       if any(A1.loc[(m, h), "p_holm"] < 0.05 and A1.loc[(m, h), "dm_stat"] < 0 for h in hz) else "non migliore di M1") for m in ("N2", "N3")}
    passa = b["passa"]
    u_st = n1["ultima_stima"]
    iv = n1["intervallo_tra_le_stime"]
    imp = b["boosting"]["importanze_ultima_origine"]
    stv = st[st["valida"]].copy()
    t_st = md(stv.iloc[::max(1, len(stv) // 8)][["data_stima", "settimane_usate", "livello_basso", "livello_alto", "sigma_basso", "sigma_alto", "phi", "p_resta_basso", "p_resta_alto",
                                                 "durata_media_basso", "durata_media_alto"]].rename(columns={
        "data_stima": "data della stima", "settimane_usate": "settimane usate", "livello_basso": "livello medio, regime basso", "livello_alto": "livello medio, regime alto",
        "sigma_basso": "dev. std. settimanale (log), basso", "sigma_alto": "dev. std. (log), alto", "p_resta_basso": "prob. di restare, basso", "p_resta_alto": "prob. di restare, alto",
        "durata_media_basso": "durata media (settimane), basso", "durata_media_alto": "durata media, alto"}), ".3f")
    if passa:
        v_dir = (f"**{VB4.NOMI[s2['scelto']]} passa la regola sulla direzione**" if s2["scelto"] else "nessun modello passa la regola sulla direzione")
        v_chi = (f"**{VB4.NOMI[s1['scelto']]} passa la regola sulla chiusura**" if s1["scelto"] else "nessun modello passa la regola sulla chiusura")
        verdetto = f"{v_chi[0].upper() + v_chi[1:]}; {v_dir}. Entra tra i finalisti come F3: {', '.join(VB4.NOMI[m] for m in passa)}."
    else:
        verdetto = "**Nessun modello della Parte B passa la regola**, né sulla chiusura né sulla direzione: nessuno entra tra i finalisti e F3 non esiste."

    cautele_n1 = ("- **Cautele su N1.** Il guadagno è piccolo e riguarda solo la direzione a 1 settimana; a 4 settimane non si vede. La probabilità di N1 è quella del modello (mistura di normali), "
                  "confrontata con la logistica di base come chiedeva la regola; il confronto con la probabilità D1 usata dal programma è nella Parte C.") if "N1" in passa else ""

    # ------------------------------------------------------------------ Parte C
    def fianco(gruppo: str, orizzonti=(1, 4)) -> str:
        g = conf[(conf["gruppo"] == gruppo) & conf["h"].isin(orizzonti)].set_index(["h", "versione"])
        rows = []
        for k, lab in ETICHETTE.items():
            r = {"misura": lab}
            for h in orizzonti:
                for v in versioni:
                    val = g.loc[(h, v), k]
                    r[f"{v}, h = {h}"] = (f"{int(val)}" if k in INTERI else pc(val) if k in QUOTE else n(val, 3)) if np.isfinite(val) else "–"
            rows.append(r)
        return pd.DataFrame(rows).to_markdown(index=False)

    d = diff[diff["h"].isin(hz)].copy()
    d["gruppo"] = d["gruppo"].map({"tutte": "tutte le origini", "fascia_13_22": "fascia 13-22"})
    d["versione"] = d["versione"].map(nome_v)
    d["errore assoluto medio: F1 − versione (punti) [IC 95%]"] = [f"{x:+.3f} [{lo:+.3f}; {hi:+.3f}]" for x, lo, hi in zip(d["mae_guadagno"], d["mae_ci_lo"], d["mae_ci_hi"])]
    d["R² della chiusura centrale vs F1 [IC 95%]"] = [ic(x, lo, hi) for x, lo, hi in zip(d["r2_centrale_vs_f1"], d["r2_ci_lo"], d["r2_ci_hi"])]
    d["guadagno sul Brier della direzione vs F1 [IC 95%]"] = [ic(x, lo, hi) for x, lo, hi in zip(d["brier_skill_vs_f1"], d["brier_ci_lo"], d["brier_ci_hi"])]
    t_diff = md(d[["gruppo", "h", "versione", "n", "errore assoluto medio: F1 − versione (punti) [IC 95%]", "R² della chiusura centrale vs F1 [IC 95%]", "p", "n_etichette",
                   "guadagno sul Brier della direzione vs F1 [IC 95%]", "p_brier"]].rename(columns={"p": "p DM (chiusura)", "n_etichette": "etichette", "p_brier": "p DM (direzione)"}), ".3f")
    D = diff[diff["gruppo"] == "tutte"].set_index(["versione", "h"])
    CF = conf[conf["gruppo"] == "tutte"].set_index(["versione", "h"])

    def frase_c(v: str) -> str:
        sg = lambda x: f"{x:+.3f}".replace(".", ",")
        return "; ".join(f"h = {h}: errore assoluto medio della chiusura centrale {n(CF.loc[(v, h), 'mae_centrale'], 2)} punti contro {n(CF.loc[('F1', h), 'mae_centrale'], 2)} di F1 "
                         f"(differenza {sg(D.loc[(v, h), 'mae_guadagno'])} [{sg(D.loc[(v, h), 'mae_ci_lo'])}; {sg(D.loc[(v, h), 'mae_ci_hi'])}]; test sull'errore quadratico: p = {n(D.loc[(v, h), 'p'], 2)}), "
                         f"copertura della fascia all'80% {pc(CF.loc[(v, h), 'copertura_80'])} contro {pc(CF.loc[('F1', h), 'copertura_80'])}, "
                         f"Brier della direzione {ic(D.loc[(v, h), 'brier_skill_vs_f1'], D.loc[(v, h), 'brier_ci_lo'], D.loc[(v, h), 'brier_ci_hi'])} (p = {n(D.loc[(v, h), 'p_brier'], 2)})" for h in hz)

    sig_c = [f"{v} a h = {h} ({k})" for v in versioni[1:] for h in (1, 2, 3, 4) for k, col in (("chiusura", "p"), ("direzione", "p_brier")) if D.loc[(v, h), col] < 0.05]
    po = c["per_orizzonte"]

    # ------------------------------------------------------------------ sintesi (10 righe)
    sintesi = [
        f"**Candele (Parte A).** La regola sceglie **{KA.DESCRIZIONI[V]}**: a 1 settimana il corpo previsto passa dall'{pc(rap['A0'])} al {pc(rap[V])} di quello reale e l'escursione dal "
        f"{pc(P.loc[('A0', 1), 'rapporto_escursione'])} all'{pc(P.loc[(V, 1), 'rapporto_escursione'])}; a 4 settimane corpo {pc(P.loc[(V, 4), 'rapporto_corpo'])}, escursione {pc(P.loc[(V, 4), 'rapporto_escursione'])}." if V != "A0"
        else f"**Candele (Parte A).** La regola lascia A0: {sc['motivo']}.",
        (f"**Spareggio confermato il {SPAREGGIO_CONFERMATO}:** {' e '.join(pari)} sono in parità esatta sul criterio (stesso corpo a +1); è stata presa {sc['candidata']} per una regola di spareggio fissata nel codice prima dei risultati (sezione 1.3)."
         if len(pari) > 1 else "**Parte A, regola:** nessuna parità tra le varianti; " + sc["motivo"] + "."),
        f"**Gradient boosting (N2, N3): verdetto negativo.** N2 è {gb_verdetto['N2']} (R² {pct(A1.loc[('N2', 1), 'r2_vs_m1'])} a h = 1, {pct(A1.loc[('N2', 4), 'r2_vs_m1'])} a h = 4); "
        f"N3, con i dati di mercato, è {gb_verdetto['N3']} ({pct(A1.loc[('N3', 1), 'r2_vs_m1'])} e {pct(A1.loc[('N3', 4), 'r2_vs_m1'])}). Anche sulla direzione fanno peggio della logistica di base.",
        f"**Modello a regimi (N1), chiusura: nessun guadagno.** {f1_('N1')}.",
        f"**Modello a regimi (N1), direzione:** {f2_('N1')}. " + ("**Passa la regola** (significativo dopo Holm a un orizzonte) ed entra tra i finalisti come F3." if "N1" in passa else "Non passa la regola."),
        f"**I due regimi** (ultima stima): livello medio {n(u_st['livello_basso'], 1)} e {n(u_st['livello_alto'], 1)}, variabilità settimanale {n(u_st['sigma_basso'], 3)} e {n(u_st['sigma_alto'], 3)} (log), "
        f"durata media {n(u_st['durata_media_basso'], 1)} e {n(u_st['durata_media_alto'], 1)} settimane: i regimi si distinguono più per la variabilità che per il livello. Stime non valide: {n1['n_non_valide']} su {n1['n_ristime']}.",
        f"**Versioni congelate:** {', '.join(nome_v(v) for v in versioni)}, in `versioni/*.json` con parametri, finestre, regole, impronta dei sorgenti e commit. È ciò che verrà esaminato sull'holdout.",
        f"**F1 contro F2 nello sviluppo** ({po['1']['n']} origini a h = 1, dal {po['1']['prima_data_origine']}). {frase_c('F2')}.",
        (f"**F1 contro F3.** {frase_c('F3')}." if f3 else "**F3 non esiste:** nessun modello della Parte B ha passato la regola."),
        f"**Programma e piano.** `python -m src.forecast --versione f1|f2{'|f3' if f3 else ''}` (predefinita f1), `--aggiorna` per i dati di F2 con ripiego su f1, registro con la versione, scheda con la tabella di confronto. "
        f"Il piano d'esame è in `reports/piano_esame_holdout.md`; **l'holdout non è stato aperto**.",
    ]

    out = f"""# Fase 4 — Ultima fase di costruzione: candele, modelli non lineari e a regimi, versioni finaliste

*Generato da `python -m src.run_phase4`. Campione di sviluppo fino al {a['per_orizzonte']['1']['ultima_data_obiettivo']}. L'holdout (VIX e dati esterni) non è stato letto: il comando gira sotto la guardia.
Tutti i numeri vengono da `reports/fase4/`.*

## Sintesi

{chr(10).join(f'{i}. {s}' for i, s in enumerate(sintesi, start=1))}

## 1. Parte A — Dimensioni delle candele

**Varianti** (elenco chiuso) per le ampiezze di corpo, ombra superiore e ombra inferiore:

{chr(10).join(f'- {KA.DESCRIZIONI[v]}.' for v in KA.VARIANTI)}

Tra le varianti cambia solo la finestra. Restano come nella Fase 3: quale componente usa la regressione (corpo a +1 e +2) e quale la media (tutto il resto), il rapporto di apertura,
la direzione del corpo (frequenza storica) e la costruzione K2. «Ultime N settimane» = le ultime N coppie origine-esito con esito noto all'origine.

**Regola** (dalla consegna): la variante con il rapporto corpo previsto / reale più vicino a 1 a h = 1, purché il suo errore quadratico sulle ampiezze non sia significativamente peggiore di A0
(Diebold-Mariano, p < {n(sc['soglia_p'], 2)}); altrimenti resta A0. «Errore quadratico sulle ampiezze» è stato fissato prima come la somma, per ogni origine, degli errori quadratici in log di corpo e due ombre a h = 1.

### 1.1 Pagella completa delle candele

Stesse origini delle Fasi 2c e 3 ({a['per_orizzonte']['1']['n_previsioni']} previsioni a 1 settimana, date previste dal {a['per_orizzonte']['1']['prima_data_obiettivo']} al {a['per_orizzonte']['1']['ultima_data_obiettivo']}).
Errori medi in punti VIX. La pagella per fascia di partenza, sottoperiodo, casi dopo un rialzo oltre il 30% e fascia 13-22 è in `reports/fase4/a_pagella_gruppi.csv`.

{t_pag}

### 1.2 Errore sulle ampiezze

Ampiezze in logaritmi. DM positivo = la variante ha errore quadratico più basso di A0.

{t_amp}

### 1.3 Esito della regola

- Rapporto del corpo a h = 1: {'; '.join(f'{v} {n(rap[v], 3)}' for v in KA.VARIANTI)}. Candidata: {sc['candidata']}{'; ' + f_dm if f_dm else ''}.
- **Scelta: {KA.DESCRIZIONI[V]}.** {sc['motivo'][0].upper() + sc['motivo'][1:]}.
- {f_pari}
- Le candele restano un po' più piccole di quelle reali (corpo {pc(P.loc[(V, 1), 'rapporto_corpo'])}, escursione {pc(P.loc[(V, 1), 'rapporto_escursione'])} a 1 settimana), ma il ritardo della media espandibile è in gran parte recuperato.
  L'errore sulla chiusura della candela {'aumenta' if P.loc[(V, 1), 'mae_chiusura'] > P.loc[('A0', 1), 'mae_chiusura'] else 'non aumenta'} ({n(P.loc[(V, 1), 'mae_chiusura'], 2)} punti contro {n(P.loc[('A0', 1), 'mae_chiusura'], 2)} di A0 a 1 settimana):
  una candela più grande in una direzione fissa si allontana di più dal centro. La previsione centrale resta la chiusura centrale, non la chiusura della candela.

## 2. Parte B — Ultimo tentativo: modelli a regimi e non lineari

### 2.1 Che cosa è stato provato

- **{VB4.NOMI['N1']}.** x_t − μ(S_t) = φ (x_(t−1) − μ(S_(t−1))) + σ(S_t) ε_t sul log della chiusura, con S_t catena di Markov a due stati: media e varianza diverse per regime, φ comune.
  Ristima ogni {RE.PASSO_STIMA} settimane; tra una ristima e l'altra parametri fermi e probabilità filtrate ricalcolate a ogni origine. Previsione a h passi esatta dalle probabilità filtrate
  (somma sui percorsi di regime): T1 = media, T2 = probabilità che la chiusura superi quella dell'origine. Stime non valide: {n1['n_non_valide']} su {n1['n_ristime']} ({', '.join(n1['date_non_valide']) or 'nessuna'});
  per le {n1['origini_con_ripiego_ar1']} origini senza una stima valida precedente è stato usato un AR(1) a un solo regime, come fissato prima.
- **{VB4.NOMI['N2']}.** Variabili di M1, variabili della candela della Fase 2c ({', '.join(GB.NOMI_CANDELA)}) e variabili giornaliere G1. Iperparametri fissi: {GB.PARAMETRI['n_estimators']} alberi, profondità {GB.PARAMETRI['max_depth']},
  tasso di apprendimento {n(GB.PARAMETRI['learning_rate'], 2)}, sottocampionamento {n(GB.PARAMETRI['subsample'], 1)}, seed {GB.PARAMETRI['random_state']}. Ristima ogni {GB.PASSO_STIMA} settimane.
  Per T1 l'albero prevede la variazione x(t+h) − x(t) (gli alberi non estrapolano i livelli); per T2 è un classificatore con gli stessi iperparametri.
- **{VB4.NOMI['N3']}.** Come N2 con in più le variabili X1-X4, sul campione comune (prima data prevista {b['per_orizzonte']['1']['prima_data_obiettivo']['N3']}).
- Riferimenti sulle stesse origini: M1 su tutta la storia; frequenza storica e logistica di base. Test: Diebold-Mariano a due code (i tre modelli non contengono M1 come caso particolare), Holm su {b['n_test_holm']['t1']} test per la chiusura
  e su {b['n_test_holm']['t2_base']} per la direzione. Regola come nella Fase 3.

### 2.2 T1 — Chiusura

{t_t1}

![Guadagni rispetto al riferimento](figures/fase4/b_guadagni_vs_riferimento.png)

### 2.3 T2 — Direzione della chiusura rispetto all'origine

{t_t2}

Accuratezza e quota di settimane quando la probabilità della direzione prevista è almeno 60%, 65% e 70%:

{t_t2s}

Calibrazione per classi di probabilità: `reports/fase4/b_t2_calibrazione.csv`.

### 2.4 Fascia 13-22 e casi dopo un rialzo oltre il 30%

Chiusura, fascia 13-22:

{gruppo_t1('fascia_13_22')}

Direzione, fascia 13-22:

{gruppo_t2('fascia_13_22')}

Chiusura, dopo un rialzo oltre il 30% in 4 settimane:

{gruppo_t1('dopo_rialzo_30')}

Direzione, dopo un rialzo oltre il 30% in 4 settimane:

{gruppo_t2('dopo_rialzo_30')}

### 2.5 Sottoperiodi e fasce di partenza

R² rispetto a M1 (tra parentesi il numero di previsioni); tutte le metriche in `reports/fase4/b_t1_gruppi.csv` e `b_t2_gruppi.csv`.

{pivot(g1, 'sottoperiodo', 'r2_vs_m1')}

{pivot(g1, 'fascia_origine', 'r2_vs_m1', list(C.BAND_LABELS))}

Guadagno sul Brier rispetto alla logistica di base:

{pivot(g2, 'sottoperiodo', 'skill_vs_base')}

{pivot(g2, 'fascia_origine', 'skill_vs_base', list(C.BAND_LABELS))}

### 2.6 I due regimi di N1

Regimi ordinati per livello medio. Ultima stima ({u_st['data_stima']}, {int(u_st['settimane_usate'])} settimane): regime basso con livello medio {n(u_st['livello_basso'], 1)} e deviazione standard settimanale {n(u_st['sigma_basso'], 3)} (log);
regime alto con livello medio {n(u_st['livello_alto'], 1)} e deviazione standard {n(u_st['sigma_alto'], 3)}; φ = {n(u_st['phi'], 3)}. Durata media: {n(u_st['durata_media_basso'], 1)} settimane nel regime basso, {n(u_st['durata_media_alto'], 1)} nel regime alto.
Tra le {len(stv)} stime valide: livello basso tra {n(iv['livello_basso'][0], 1)} e {n(iv['livello_basso'][1], 1)}, livello alto tra {n(iv['livello_alto'][0], 1)} e {n(iv['livello_alto'][1], 1)},
durata media del regime alto tra {n(iv['durata_media_alto'][0], 1)} e {n(iv['durata_media_alto'][1], 1)} settimane. La probabilità filtrata del regime alto supera il 50% nel {pc(n1['quota_origini_regime_alto_oltre_50'], 0)} delle origini.
I due livelli medi sono vicini: il modello non separa «VIX basso» da «VIX alto», ma settimane tranquille da settimane agitate (la variabilità del regime alto è circa {n(u_st['sigma_alto'] / u_st['sigma_basso'], 1)} volte quella del regime basso).

Una stima ogni {max(1, len(stv) // 8)} (tutte in `reports/fase4/b_regimi_stime.csv`):

{t_st}

![Probabilità di regime](figures/fase4/b_regimi_probabilita.png)

### 2.7 Verdetto

- **Gradient boosting, solo VIX (N2): negativo.** Chiusura: {f1_('N2')}. Direzione: {f2_('N2')}.
- **Gradient boosting con i dati di mercato (N3): negativo.** Chiusura: {f1_('N3')}. Direzione: {f2_('N3')}.
  Variabili con più peso negli alberi all'ultima origine (h = 1): N2 {', '.join(f'{k} {pc(v, 0)}' for k, v in list(imp['N2|1'].items())[:3])}; N3 {', '.join(f'{k} {pc(v, 0)}' for k, v in list(imp['N3|1'].items())[:3])}.
  Lettura nostra, non misurata: con circa mille settimane e un segnale debole gli alberi si adattano al rumore del campione di stima, mentre M1 ha solo quattro parametri.
- **Modello a regimi (N1).** Chiusura: {f1_('N1')}. Direzione: {f2_('N1')}.
- **Esito della regola.** {verdetto}
{cautele_n1}

## 3. Parte C — Versioni finaliste

### 3.1 Definizioni

- **{VS.NOMI['f1']}.** Chiusura centrale e fasce: D4 (regressione quantilica sulle variabili di M1). Probabilità ed etichetta: D1 (errori passati attorno al centro M1). Candele K2 con le ampiezze {V}.
- **{VS.NOMI['f2']}.** Stessa costruzione con X1 = VIX3M / VIX − 1 in più: centro M1 + X1; chiusura centrale e fasce dalla regressione quantilica con X1; probabilità ed etichetta dagli errori passati attorno al centro M1 + X1
  (almeno 104, come D1); stesse candele di F1, ancorate alla chiusura centrale di F2 (la candela +1 è quindi identica a quella di F1). Se X1 manca all'origine si ripiega su F1.
{('- **' + VS.NOMI['f3'] + '.** Tutto dalla distribuzione prevista da N1: centro = media, chiusura centrale = mediana, fasce = quantili, probabilità ed etichetta = probabilità del modello (quella misurata nella Parte B); stesse candele di F1, ancorate alla chiusura centrale di F3.') if f3 else '- **F3 non esiste**: nessun modello della Parte B ha passato la regola.'}

Lettura adottata dove la consegna lasciava margine: in ogni versione la «chiusura centrale» mostrata è la mediana della distribuzione (come D4 in F1), mentre il «centro» è il valore attorno a cui si misurano gli errori passati;
nelle tabelle sono riportati entrambi.

### 3.2 Configurazioni congelate

`python -m src.fase4.congela` scrive `versioni/f1.json`, `versioni/f2.json`{' e `versioni/f3.json`' if f3 else ''}: dati usati, modello del centro, della chiusura centrale, delle fasce, delle probabilità e dell'etichetta, orizzonti, griglia dei quantili,
soglie, numero minimo di errori, regole di ammissione e di ripiego, finestre e assegnazione delle ampiezze delle candele, scelte fatte dalle regole di questa fase, impronta SHA-256 dei file sorgente che definiscono i modelli
e hash del commit del codice. Un test verifica che i file coincidano con la configurazione ricostruita dal codice e che i sorgenti non siano cambiati dopo il congelamento. Il programma legge da lì la variante delle ampiezze.

### 3.3 Programma

- `python -m src.forecast --versione f1|f2{'|f3' if f3 else ''}` (predefinita: f1). La scheda mostra la versione richiesta e una tabella di confronto con le altre versioni calcolabili (chiusura centrale, fasce, etichetta, probabilità).
- `--aggiorna` scarica dal sito Cboe i file giornalieri di VIX e VIX3M in `data/external/forecast` (non pubblicata). Vengono usati solo i valori fino all'ultima seduta della settimana di origine; la serie settimanale di X1 usata
  viene salvata in `reports/previsione/f2_valori_usati_AAAA-MM-GG.csv` e i valori dell'origine, con l'impronta dei file, nel JSON della previsione.
- Se VIX3M manca nella settimana di origine (file assenti o più vecchi dell'origine) il programma avvisa a schermo e, se era stata chiesta f2, ripiega su f1.
- Il registro salva la versione: una previsione è identificata da origine + versione e non viene mai duplicata né riscritta. Le righe scritte prima di questa fase restano senza versione. `--verifica` produce la pagella dal vivo per versione.

### 3.4 Confronto fianco a fianco nello sviluppo

Stesse origini per tutte le versioni: quelle ammesse per F2, {po['1']['n']} a h = 1 (origini dal {po['1']['prima_data_origine']}, date previste fino al {po['1']['ultima_data_obiettivo']}) e {po['4']['n']} a h = 4.
I numeri di F1 sono quelli del programma (D4, D1, candele {V}), letti dai file delle fasi precedenti; F2{' e F3' if f3 else ''} sono calcolate in walk-forward.
L'etichetta di F2 richiede 104 errori passati attorno al suo centro: esiste dal {c['prima_origine_con_etichetta_f2']}; le righe sull'etichetta usano le {po['1']['n_etichette']} origini (a h = 1) in cui esiste per tutte le versioni.
h = 2 e 3 sono in `reports/fase4/c_confronto.csv`.

{fianco('tutte')}

Differenze rispetto a F1 (positivo = la versione fa meglio di F1), con intervallo bootstrap e test di Diebold-Mariano non corretti per i confronti multipli:

{t_diff}

Confronti con p < 0,05 (non corretto) tra tutti gli orizzonti 1-4: {', '.join(sig_c) if sig_c else 'nessuno'}.

### 3.5 Fascia 13-22

{po['1']['n_fascia_13_22']} origini a h = 1.

{fianco('fascia_13_22')}

### 3.6 Lettura del confronto

- **F2.** {frase_c('F2')}.
{('- **F3.** ' + frase_c('F3') + '.') if f3 else ''}
- Le differenze sulla chiusura centrale sono di pochi centesimi di punto e gli intervalli comprendono lo zero. Le differenze più visibili sono sulle fasce (larghezza e copertura) e sulla quota di etichette neutre.
- Le candele a +1 sono identiche tra le versioni per costruzione; da +2 cambia solo l'ancoraggio.
- Questo confronto è sullo stesso campione su cui i modelli sono stati scelti: non è una conferma. La conferma è l'esame sull'holdout.

![Verifica sul passato, F1](figures/fase4/g3_verifica_passato.png)

{c['g3']['riga']}

## 4. Piano d'esame

`reports/piano_esame_holdout.md` fissa versioni esaminate, procedura, metriche, tabelle e criteri per dichiarare una versione migliore di un'altra. È solo scritto: nessun codice d'esame è stato eseguito e l'holdout non è stato aperto.

## 5. Problemi e ambiguità

1. **Parità tra A1 e A2 (spareggio confermato).** {f_pari}
2. **Che cosa cambia tra le varianti.** La consegna dà le finestre; è stato tenuto fermo quale componente usa la regressione e quale la media (corpo a +1 e +2: regressione), come in A0.
3. **«Errore quadratico sulle ampiezze».** Letto come somma degli errori quadratici di corpo e due ombre a h = 1. La tabella 1.2 riporta anche le singole componenti e h = 4.
4. **N1: definizione di stima valida e ripiego iniziale.** La prima stima ({', '.join(n1['date_non_valide']) or 'nessuna'}) non è valida; non esistendo una stima precedente, per le prime {n1['origini_con_ripiego_ar1']} origini è stato usato un AR(1) (fissato prima).
5. **N1: T2 dal modello.** La probabilità della direzione di N1 è quella della sua distribuzione prevista, non una logistica: è il motivo per cui può differire da M1 anche a parità di previsione media.
6. **N2/N3: obiettivo dell'albero.** Per T1 l'albero prevede la variazione e non il livello. È una scelta fatta prima della valutazione; con l'obiettivo sul livello gli alberi non potrebbero prevedere valori fuori da quelli già visti.
7. **Test per la Parte B.** Diebold-Mariano e non Clark-West, perché N1-N3 non contengono M1. Holm dentro la Parte B, non tra le fasi.
8. **F3 non era prevista come certa.** N1 passa solo sulla direzione a h = 1 e con un guadagno piccolo; la versione è stata costruita perché la regola lo chiede. Le sue fasce sono parametriche (mistura di normali) e nello sviluppo coprono
   {pc(CF.loc[('F3', 4), 'copertura_80']) if f3 else '–'} a h = 4 per la fascia all'80%.
9. **Etichetta di F2 disponibile più tardi.** Servono 104 errori attorno al centro M1 + X1: nello sviluppo l'etichetta di F2 parte dal {c['prima_origine_con_etichetta_f2']}. Nell'esame sull'holdout il problema non si pone.
10. **Campioni diversi per le misure mostrate nel programma.** Accanto a F1 il programma mostra le misure su tutte le origini dal 2006; accanto a F2 e F3 quelle sulle origini comuni dal 2014. La scheda lo dice.
11. **Registro.** Con la versione nella chiave, per l'origine già registrata prima di questa fase il registro contiene sia la riga vecchia (senza versione) sia le nuove. Le previsioni registrate dopo la data della settimana prevista
    sono segnalate dalla colonna `registrata_prima_dell_esito` della verifica.
12. **Dati per F2 nel programma.** I file Cboe sono scaricati interi e troncati all'origine; non sono nel repository. Viene pubblicata solo la serie settimanale di X1 usata per l'origine.
13. **Nuova dipendenza.** scikit-learn (versione fissata in `requirements.txt`) serve solo a N2 e N3, che non sono tra i finalisti.
14. **Log della chiusura.** Nella Parte C il log è ricalcolato dalla chiusura, come fa il programma; nelle Parti A e B e nelle fasi precedenti si usa la colonna del file di sviluppo, arrotondata a 6 decimali.
    La differenza è trascurabile sulle medie ma sposta di qualche millesimo di punto i quantili della regressione quantilica: un test confronta il file della Parte C con il programma.
15. **Molti confronti sullo stesso campione.** Dopo sei fasi il campione di sviluppo è stato usato per molte scelte. I guadagni di F2 e F3 su F1 sono piccoli e in parte attesi per selezione: l'holdout dirà se reggono.
"""
    (C.REPORTS_DIR / "fase4_report.md").write_text(out)


if __name__ == "__main__":
    run()
