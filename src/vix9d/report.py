"""Genera reports/vix9d/fase1_report.md. Tutti i numeri sono letti dai file prodotti dal codice (reports/vix9d/*.csv e
*.json; per il confronto con il VIX, dai file gia' pubblicati delle fasi del VIX): nessun numero e' scritto a mano."""
from __future__ import annotations

import json

import pandas as pd

from .. import config as C
from ..fase3.report import ic, md, n, pc, pct
from . import config as G
from . import modelli as M
from . import versioni as VS

MIN_N = 20
EST = ("massimo", "minimo")


def pv(p: float) -> str:
    return "< 0,0001" if p < 1e-4 else n(p, 4)


def pe(p: float) -> str:
    return "p < 0,0001" if p < 1e-4 else f"p = {n(p, 4)}"


def casi(k) -> str:
    return f"{int(k)}" + (" (meno di 20)" if k < MIN_N else "")


def sett(h: int) -> str:
    return "1 settimana" if h == 1 else f"{h} settimane"


def vix_pubblicati() -> dict:
    """Numeri del VIX gia' pubblicati (versione F1 e modello di riferimento M1), letti dai file delle fasi del VIX."""
    sp = pd.read_csv(C.REPORTS_DIR / "fase4c" / "statistiche_programma.csv")
    am = pd.read_csv(C.REPORTS_DIR / "fase4c" / "a_misure.csv")
    pd4 = pd.read_csv(C.REPORTS_DIR / "fase4c" / "prima_e_dopo.csv")
    t1 = pd.read_csv(C.REPORTS_DIR / "fase3" / "t1_metriche.csv")
    f1 = {int(r["h"]): r for r in sp[sp["versione"] == "f1"].to_dict("records")}
    ch = am[(am["campione"] == "tutte le origini") & (am["gruppo"] == "tutte") & (am["modello"] == "S0") & (am["bersaglio"] == "chiusura")].iloc[0]
    tutte = pd4[(pd4["versione"] == "F1") & (pd4["campione"] == "tutte le origini") & (pd4["gruppo"] == "tutte") & (pd4["stato"] == "dopo")].iloc[0]
    rec = pd4[(pd4["versione"] == "F1") & (pd4["campione"] != "tutte le origini") & (pd4["gruppo"] == "tutte") & (pd4["stato"] == "dopo")].iloc[0]
    m1 = t1[(t1["modello"] == "M1") & (t1["h"] == 1)].iloc[0]
    return {"f1": f1, "chiusura": ch, "tutte": tutte, "recenti": rec, "m1": m1}


def tabella_confronto(ev: pd.DataFrame, b: pd.DataFrame, an: dict, V: dict) -> tuple[str, dict]:
    g = {int(r["h"]): r for r in ev[(ev["versione"] == "g1") & (ev["gruppo"] == "tutte")].to_dict("records")}
    r1 = b[(b["campione"] == "origini comuni") & (b["modello"] == "R1") & (b["h"] == 1)].iloc[0]
    f1, ch, tu, rc, m1 = V["f1"], V["chiusura"], V["tutte"], V["recenti"], V["m1"]
    np_ = "n.p."
    rows = [("Errore assoluto medio della chiusura centrale, +1 (punti)", n(f1[1]["mediana_mae_vix"]), np_, n(g[1]["mae_centrale"])),
            ("Errore assoluto medio della chiusura centrale, +2 (punti)", n(f1[2]["mediana_mae_vix"]), np_, n(g[2]["mae_centrale"])),
            (f"Radice dell'errore quadratico in log del modello di riferimento, +1 (M1 del VIX, n = {int(m1['n'])}; R1 del VIX9D)", n(m1["rmse_log"], 3), np_, n(r1["rmse_log"], 3)),
            ("Chiusura entro il 5% dal valore reale, +1", pc(ch["entro_5"]), np_, pc(g[1]["entro_5_chiusura"])),
            ("Chiusura entro il 10% dal valore reale, +1", pc(ch["entro_10"]), np_, pc(g[1]["entro_10_chiusura"])),
            ("Massimo entro il 5%, +1", pc(tu["massimo_entro_5"]), pc(rc["massimo_entro_5"]), pc(g[1]["massimo_entro_5"])),
            ("Massimo entro il 10%, +1", pc(tu["massimo_entro_10"]), pc(rc["massimo_entro_10"]), pc(g[1]["massimo_entro_10"])),
            ("Errore percentuale mediano del massimo, +1", pc(tu["massimo_errore_pct_mediano"]), pc(rc["massimo_errore_pct_mediano"]), pc(g[1]["massimo_errore_pct_mediano"])),
            ("Minimo entro il 5%, +1", pc(tu["minimo_entro_5"]), pc(rc["minimo_entro_5"]), pc(g[1]["minimo_entro_5"])),
            ("Minimo entro il 10%, +1", pc(tu["minimo_entro_10"]), pc(rc["minimo_entro_10"]), pc(g[1]["minimo_entro_10"])),
            ("Errore percentuale mediano del minimo, +1", pc(tu["minimo_errore_pct_mediano"]), pc(rc["minimo_errore_pct_mediano"]), pc(g[1]["minimo_errore_pct_mediano"])),
            ("Massimo e minimo entrambi entro il 5%, +1", pc(tu["entrambi_entro_5"]), pc(rc["entrambi_entro_5"]), pc(g[1]["entrambi_entro_5"])),
            ("Sovrapposizione tra candela prevista e reale, +1", n(tu["sovrapposizione"], 3), n(rc["sovrapposizione"], 3), n(g[1]["sovrapposizione"], 3))]
    for h in (1, 2):
        rows += [(f"Etichetta di direzione non neutra, +{h}: quota dei casi", pc(f1[h]["quota_non_neutre"]), np_, pc(g[h]["quota_non_neutre"])),
                 (f"Etichetta di direzione non neutra, +{h}: quota corretta (casi)", f"{pc(f1[h]['accuratezza_non_neutre'])} ({int(f1[h]['n_non_neutre'])})", np_,
                  f"{pc(g[h]['accuratezza_non_neutre'])} ({int(g[h]['n_non_neutre'])})")]
    for h in (1, 2):
        rows.append((f"Copertura delle fasce al 50% / 80% / 90%, +{h}", " / ".join(pc(f1[h][f"copertura_{k}"], 0) for k in (50, 80, 90)), np_, " / ".join(pc(g[h][f"copertura_{k}"], 0) for k in (50, 80, 90))))
    col = ["misura", f"VIX, versione F1: tutte le origini (n = {int(f1[1]['n'])})", f"VIX, versione F1: origini più recenti (n = {int(rc['n'])})", f"VIX9D, versione G1 (n = {int(g[1]['n'])})"]
    num = {"rapporto_rmse": float(r1["rmse_log"] / m1["rmse_log"]), "rapporto_dev_std": float(an["vix9d"]["variazione_log_settimanale"]["dev_std"] / an["vix_stesso_periodo"]["variazione_log_settimanale"]["dev_std"]),
           "vix": {"massimo_5": float(tu["massimo_entro_5"]), "minimo_5": float(tu["minimo_entro_5"]), "non_neutre": float(f1[1]["quota_non_neutre"]), "accuratezza": float(f1[1]["accuratezza_non_neutre"]),
                   "coperture": [float(f1[1][f"copertura_{k}"]) for k in (50, 80, 90)]}}
    return pd.DataFrame(rows, columns=col).to_markdown(index=False), num


def run() -> None:
    T = lambda k: pd.read_csv(G.OUT / f"{k}.csv")
    J = lambda k: json.loads((G.OUT / f"{k}.json").read_text())
    au, an, info, g3 = J("audit"), J("a_analisi"), J("info"), J("g3")
    dr, sp, b, cf, cs, cd, de, rg, ev = T("a_dopo_rapporto"), T("a_sottoperiodi"), T("b_chiusura"), T("c_fasce"), T("c_soglie"), T("c_direzione"), T("d_estremi"), T("regole"), T("e_versioni")
    sc, org = info["scelte"], info["origini"]
    gd, ws = au["giornaliero_sviluppo"], au["settimanale_sviluppo"]
    a9, av, k9, kv, rp = an["vix9d"], an["vix_stesso_periodo"], an["candele_vix9d"], an["candele_vix_stesso_periodo"], an["rapporto_vix9d_su_vix"]
    v9, vv = a9["variazione_log_settimanale"], av["variazione_log_settimanale"]
    ap = ws["apertura_su_chiusura_precedente"]
    B = lambda h, m, camp="origini comuni": b[(b["campione"] == camp) & (b["h"] == h) & (b["modello"] == m)].iloc[0]
    E = lambda v, h, gr="tutte": ev[(ev["versione"] == v) & (ev["h"] == h) & (ev["gruppo"] == gr)].iloc[0]
    D = lambda h, m: cd[(cd["h"] == h) & (cd["modello"] == m)].iloc[0]
    X = lambda h, ins, e, gr="tutte": de[(de["h"] == h) & (de["insieme"] == ins) & (de["estremo"] == e) & (de["gruppo"] == gr)].iloc[0]
    RG = lambda fam, chiave, cand: rg[(rg["famiglia"] == fam) & (rg["chiave"].astype(str) == str(chiave)) & (rg["candidato"] == cand)].iloc[0]
    d1, d10 = dr.iloc[0], dr.iloc[-1]

    # ---------- tabelle
    t_liv = pd.DataFrame([{"indice": nome, "settimane": s["n_settimane"], "livello mediano": n(s["livello"]["mediana"]), "5° – 95° percentile": f"{n(s['livello']['p05'], 1)} – {n(s['livello']['p95'], 1)}",
                           "massimo": n(s["livello"]["max"], 1), "dev. std. delle variazioni log settimanali": n(s["variazione_log_settimanale"]["dev_std"], 3),
                           "variazione assoluta mediana": pc(s["variazione_log_settimanale"]["mediana_assoluta"]),
                           "settimane con rialzo oltre il 20%": f"{pc(s['rialzi_oltre_20']['quota'])} ({s['rialzi_oltre_20']['n']})", "oltre il 30%": f"{pc(s['rialzi_oltre_30']['quota'])} ({s['rialzi_oltre_30']['n']})",
                           "ribasso oltre il 20%": f"{pc(s['ribassi_oltre_20']['quota'])} ({s['ribassi_oltre_20']['n']})", "autocorrelazione del livello a 1 / 4 / 12 settimane": " / ".join(n(s["acf_livello_log"][k]) for k in ("1", "4", "12")),
                           "mezza vita di uno scostamento (settimane)": n(s["ar1"]["mezza_vita_settimane"], 1)} for nome, s in (("VIX9D", a9), ("VIX, stesse settimane", av))]).set_index("indice").T.reset_index(names="misura").to_markdown(index=False)
    t_can = pd.DataFrame([{"indice": nome, "settimane con candela vera": k["n"], "corpo medio (log)": n(k["corpo_log"]["media"], 3), "escursione media (log)": n(k["escursione_log"]["media"], 3),
                           "ombra superiore media (log)": n(k["ombra_superiore_log"], 3), "ombra inferiore media (log)": n(k["ombra_inferiore_log"], 3),
                           "settimane che chiudono sopra l'apertura": pc(k["quota_chiusura_sopra_apertura"]), "massimo / chiusura precedente (mediana)": n(k["massimo_su_chiusura_precedente_mediana"], 3),
                           "minimo / chiusura precedente (mediana)": n(k["minimo_su_chiusura_precedente_mediana"], 3)} for nome, k in (("VIX9D", k9), ("VIX, stesse settimane", kv))]).set_index("indice").T.reset_index(names="misura").to_markdown(index=False)
    t_sp = pd.DataFrame([{"sottoperiodo": r.sottoperiodo, "settimane": r.n, "livello mediano VIX9D": n(r.vix9d_livello_mediano), "livello mediano VIX": n(r.vix_livello_mediano),
                          "dev. std. variazioni VIX9D": n(r.vix9d_dev_std_variazioni, 3), "dev. std. variazioni VIX": n(r.vix_dev_std_variazioni, 3), "rialzi oltre il 20%: VIX9D": pc(r.vix9d_rialzi_oltre_20),
                          "rialzi oltre il 20%: VIX": pc(r.vix_rialzi_oltre_20), "rapporto VIX9D / VIX mediano": n(r.rapporto_mediano, 3)} for r in sp.itertuples()]).to_markdown(index=False)
    t_dr = pd.DataFrame([{"decile del rapporto": int(r.decile), "rapporto VIX9D / VIX": f"{n(r.rapporto_min, 2)} – {n(r.rapporto_max, 2)}", "casi": int(r.n_1), "variazione mediana dopo 1 settimana": pct(r.variazione_log_mediana_1),
                          "VIX9D più alto dopo 1 settimana": pc(r.quota_rialzo_1, 0), "variazione mediana dopo 2 settimane": pct(r.variazione_log_mediana_2), "VIX9D più alto dopo 2 settimane": pc(r.quota_rialzo_2, 0)}
                         for r in dr.itertuples()]).to_markdown(index=False)

    def riga_b(r) -> dict:
        ha = pd.notna(r["r2_vs_r1"])
        return {"orizzonte": f"+{int(r['h'])}", "modello": M.NOMI[r["modello"]], "casi": int(r["n"]), "RMSE log": n(r["rmse_log"], 4), "MAE log": n(r["mae_log"], 4), "RMSE punti": n(r["rmse_punti"]), "MAE punti": n(r["mae_punti"]),
                "entro 5%": pc(r["entro_5"]), "entro 10%": pc(r["entro_10"]), "entro 20%": pc(r["entro_20"]), "R² fuori campione contro R0": "–" if pd.isna(r["r2_vs_r0"]) else pct(r["r2_vs_r0"]),
                "R² fuori campione contro R1 [IC 95%]": ic(r["r2_vs_r1"], r["r2_ci_lo"], r["r2_ci_hi"]) if ha else "–", "test": "–" if pd.isna(r["p"]) else f"{r['test']}: {pe(r['p'])}",
                "p dopo Holm sui 6 confronti con R1": pv(r["p_holm"]) if pd.notna(r["p_holm"]) else "–"}
    t_b = pd.DataFrame([riga_b(r) for r in b[b["campione"] == "origini comuni"].to_dict("records")]).to_markdown(index=False)
    t_b2 = pd.DataFrame([riga_b(r) for r in b[b["campione"] != "origini comuni"].to_dict("records")]).drop(columns=["p dopo Holm sui 6 confronti con R1"]).to_markdown(index=False)
    t_cf = pd.DataFrame([{"orizzonte": f"+{int(r.h)}", "variabili del modello": r.modello, "casi": int(r.n), "CRPS (7 quantili)": n(r.crps7, 4), "CRPS contro R1": "–" if pd.isna(r.crps_vs_r1) else f"{pct(r.crps_vs_r1)} ({pe(r.p)})",
                          "copertura 50%": pc(r.copertura_50), "larghezza 50% (punti)": n(r.larghezza_50, 1), "copertura 80%": pc(r.copertura_80), "larghezza 80%": n(r.larghezza_80, 1),
                          "copertura 90%": pc(r.copertura_90), "larghezza 90%": n(r.larghezza_90, 1)} for r in cf.itertuples()]).to_markdown(index=False)
    t_cs = pd.DataFrame([{"versione": r.versione.upper(), "orizzonte": f"+{int(r.h)}", "soglia": int(r.soglia), "casi": int(r.n), "frequenza dell'evento": pc(r.frequenza_evento), "probabilità media prevista": pc(r.prob_media),
                          "Brier": n(r.brier, 4), "Brier della frequenza storica": n(r.brier_frequenza, 4), "Brier di «resta dov'è»": n(r.brier_persistenza, 4)} for r in cs.itertuples()]).to_markdown(index=False)
    nomi_d = {"FREQ": "frequenza storica", "BASE": "logistica di base (variabili di R1)", "RAPP": "logistica con ln(VIX9D/VIX)"}

    def riga_d(r) -> dict:
        o = {"orizzonte": f"+{int(r['h'])}", "modello": nomi_d[r["modello"]], "casi": int(r["n"]), "frequenza dei rialzi": pc(r["frequenza_rialzo"]), "accuratezza": pc(r["accuratezza"]), "Brier": n(r["brier"], 4),
             "guadagno di Brier sulla frequenza": "–" if pd.isna(r["skill_vs_freq"]) else f"{pct(r['skill_vs_freq'])} ({pe(r['p_freq'])})",
             "guadagno di Brier sulla base [IC 95%]": "–" if pd.isna(r["skill_vs_base"]) else ic(r["skill_vs_base"], r["skill_ci_lo"], r["skill_ci_hi"])}
        for s in (60, 65, 70):
            k = int(r[f"n_non_neutre_{s}"])
            o[f"non neutre al {s}%: quota, corrette (casi)"] = "–" if k == 0 else f"{pc(r[f'quota_non_neutre_{s}'], 0)}, {pc(r[f'accuratezza_{s}'], 0)} ({casi(k)})"
        return o
    t_cd = pd.DataFrame([riga_d(r) for r in cd.to_dict("records")]).to_markdown(index=False)
    t_cd2 = pd.DataFrame([{"orizzonte": f"+{int(r['h'])}", "modello": nomi_d[r["modello"]], "soglia": f"{s}%", "etichette «rialzo» (casi)": casi(r[f"n_rialzo_{s}"]),
                           "«rialzo» corrette": "–" if r[f"n_rialzo_{s}"] == 0 else pc(r[f"accuratezza_rialzo_{s}"], 0), "etichette «ribasso» (casi)": casi(r[f"n_ribasso_{s}"]),
                           "«ribasso» corrette": "–" if r[f"n_ribasso_{s}"] == 0 else pc(r[f"accuratezza_ribasso_{s}"], 0)} for r in cd[cd["modello"] != "FREQ"].to_dict("records") for s in (60, 65, 70)]).to_markdown(index=False)
    nomi_s = {"S0": "S0 (variabili di R1)", "S1": "S1 (+ candela)", "S2": "S2 (+ ln rapporto col VIX)", "FISSA": "regola fissa"}

    def tab_estremi(gruppo: str) -> str:
        g = de[de["gruppo"] == gruppo]
        return pd.DataFrame([{"orizzonte": f"+{int(r.h)}", "estremo": r.estremo, "insieme": nomi_s[r.insieme], "casi": int(r.n), "entro 5%": pc(r.entro_5), "entro 10%": pc(r.entro_10),
                              "errore % mediano": pc(r.errore_pct_mediano), "errore mediano (punti)": n(r.errore_punti_mediano), "oltre il 5% per eccesso": pc(r.oltre_5_per_eccesso),
                              "oltre il 5% per difetto": pc(r.oltre_5_per_difetto), "massimo e minimo entrambi entro il 5%": pc(r.entrambi_entro_5)} for r in g.sort_values(["h", "estremo"], kind="stable").itertuples()]).to_markdown(index=False)
    q = de[de["gruppo"].str.startswith("quintile") & (de["insieme"] == "S0")]
    t_q = pd.DataFrame([{"orizzonte": f"+{int(h)}", "fascia di livello all'origine": gr, "casi": casi(g["n"].iloc[0])} | {f"{e}: entro 5%": pc(g.loc[g["estremo"] == e, "entro_5"].iloc[0]) for e in EST}
                        | {f"{e}: errore % mediano": pc(g.loc[g["estremo"] == e, "errore_pct_mediano"].iloc[0]) for e in EST} for (h, gr), g in q.groupby(["h", "gruppo"], sort=True)]).to_markdown(index=False)
    t_rg = pd.DataFrame([{"famiglia di confronti": r.famiglia, "componente / orizzonte": str(r.chiave).replace("|", ", +") if "|" in str(r.chiave) else f"+{r.chiave}", "riferimento": r.riferimento, "candidato": r.candidato,
                          "errore del riferimento": n(r.errore_riferimento, 4), "errore del candidato": n(r.errore, 4), "test": f"{r.test}: {pe(r.p)}", "p dopo Holm": pv(r.p_holm), "batte il riferimento": "sì" if r.batte else "no"}
                         for r in rg.itertuples()]).to_markdown(index=False)
    t_sc = pd.DataFrame([{"versione": VS.NOMI[v], "chiusura e fasce (+1 / +2)": " / ".join(sc["chiusura"][v][str(h)] for h in G.ORIZZONTI), "direzione (+1 / +2)": " / ".join(sc["direzione"][v][str(h)] for h in G.ORIZZONTI),
                          "massimo (+1 / +2)": " / ".join(sc["estremi"][v]["massimo"][str(h)] for h in G.ORIZZONTI), "minimo (+1 / +2)": " / ".join(sc["estremi"][v]["minimo"][str(h)] for h in G.ORIZZONTI)} for v in VS.VERSIONI]).to_markdown(index=False)

    def tab_versioni(gruppo: str) -> str:
        rows = []
        for r in ev[ev["gruppo"] == gruppo].itertuples():
            rows.append({"versione": r.versione.upper(), "candela": f"+{int(r.h)}", "casi": int(r.n), "errore medio chiusura centrale (punti)": n(r.mae_centrale), "chiusura entro 5% / 10%": f"{pc(r.entro_5_chiusura, 0)} / {pc(r.entro_10_chiusura, 0)}",
                         "copertura 50 / 80 / 90%": " / ".join(pc(getattr(r, f"copertura_{k}"), 0) for k in (50, 80, 90)), "etichetta non neutra: quota, corrette (casi)": f"{pc(r.quota_non_neutre, 0)}, {pc(r.accuratezza_non_neutre, 0)} ({casi(r.n_non_neutre)})",
                         "massimo entro 5% / 10%": f"{pc(r.massimo_entro_5, 0)} / {pc(r.massimo_entro_10, 0)}", "minimo entro 5% / 10%": f"{pc(r.minimo_entro_5, 0)} / {pc(r.minimo_entro_10, 0)}",
                         "errore % mediano massimo / minimo": f"{pc(r.massimo_errore_pct_mediano)} / {pc(r.minimo_errore_pct_mediano)}", "entrambi entro 5%": pc(r.entrambi_entro_5), "sovrapposizione": n(r.sovrapposizione, 3),
                         "chiusura reale dentro l'escursione prevista": pc(r.chiusura_nell_escursione, 0), "corpo previsto / reale": n(r.rapporto_corpo), "escursione prevista / reale": n(r.rapporto_escursione),
                         "errore medio apertura / massimo / minimo / chiusura (punti)": " / ".join(n(getattr(r, f"mae_{k}")) for k in ("apertura", "massimo", "minimo", "chiusura")), "corpo tagliato dagli estremi": pc(r.quota_chiusura_tagliata, 0)})
        t = pd.DataFrame(rows)
        t.index = t.pop("versione") + ", candela " + t.pop("candela")
        return t.T.reset_index(names="misura").to_markdown(index=False)

    # ---------- quanto aggiunge il VIX
    agg = []
    for h in G.ORIZZONTI:
        r, x = B(h, "R2"), RG("chiusura con VIX", h, "R2")
        agg.append({"componente": f"chiusura, +{h} (R2 contro R1)", "misura": "R² fuori campione contro R1 [IC 95%]", "guadagno": ic(r["r2_vs_r1"], r["r2_ci_lo"], r["r2_ci_hi"]), "p dopo Holm": pv(x["p_holm"]), "esito": "il VIX aggiunge" if x["batte"] else "nessun guadagno dimostrato"})
    for h in G.ORIZZONTI:
        r = cf[(cf["h"] == h) & (cf["modello"] == "R2")].iloc[0]
        agg.append({"componente": f"fasce, +{h} (variabili di R2 contro R1)", "misura": "riduzione del CRPS", "guadagno": pct(r["crps_vs_r1"]), "p dopo Holm": f"non nel criterio ({pe(r['p'])})",
                    "esito": "descrittivo: le fasce seguono il modello di chiusura"})
    for h in G.ORIZZONTI:
        r, x = D(h, "RAPP"), RG("direzione con VIX", h, "RAPP")
        agg.append({"componente": f"direzione, +{h} (logistica con il rapporto contro base)", "misura": "riduzione del Brier [IC 95%]", "guadagno": ic(r["skill_vs_base"], r["skill_ci_lo"], r["skill_ci_hi"]), "p dopo Holm": pv(x["p_holm"]),
                    "esito": "il VIX aggiunge" if x["batte"] else "nessun guadagno dimostrato"})
    for e in EST:
        for h in G.ORIZZONTI:
            x, s0, s2 = RG("estremi con VIX", f"{e}|{h}", "S2"), X(h, "S0", e), X(h, "S2", e)
            agg.append({"componente": f"{e}, +{h} (S2 contro S0)", "misura": "quota entro il 5%: S0 → S2 (errore log medio: S0 → S2)",
                        "guadagno": f"{pc(s0['entro_5'])} → {pc(s2['entro_5'])} ({n(x['errore_riferimento'], 4)} → {n(x['errore'], 4)})", "p dopo Holm": pv(x["p_holm"]), "esito": "il VIX aggiunge" if x["batte"] else "nessun guadagno dimostrato"})
    t_agg = pd.DataFrame(agg).to_markdown(index=False)
    si = [a["componente"].split(" (")[0] for a in agg if a["esito"] == "il VIX aggiunge"]
    no = [a["componente"].split(" (")[0] for a in agg if a["esito"] == "nessun guadagno dimostrato"]
    n_agg, n_reg = len(si), len(si) + len(no)
    elenco = lambda v: "; ".join(v) if v else "nessuno"
    soglie_ok = bool(((cs["brier"] < cs["brier_frequenza"]) & (cs["brier"] < cs["brier_persistenza"])).all())
    q_str = {h: org[str(h)]["strappi_tra_le_comuni"] / org[str(h)]["comuni"] for h in G.ORIZZONTI}
    g1_1, g1_2, gs_1, gs_2 = E("g1", 1), E("g1", 2), E("gs", 1), E("gs", 2)
    g1_1s = E("g1", 1, "senza strappi")
    t_conf, num = tabella_confronto(ev, b, an, vix_pubblicati())
    r1_1, r1_2, r2_1, r2_2, r3_1 = B(1, "R1"), B(2, "R1"), B(1, "R2"), B(2, "R2"), B(1, "R3")
    dB1, dB2, dR2 = D(1, "BASE"), D(2, "BASE"), D(2, "RAPP")
    fis_M, fis_m = X(1, "FISSA", "massimo"), X(1, "FISSA", "minimo")
    cf1, cf2 = cf[(cf["h"] == 1) & (cf["modello"] == "R1")].iloc[0], cf[(cf["h"] == 2) & (cf["modello"] == "R1")].iloc[0]
    batte_senza = bool(rg.loc[rg["famiglia"].isin(["chiusura senza VIX", "estremi senza VIX"]), "batte"].any())
    comp = info["componenti_ultima_origine"]

    md_out = f"""# Fase 9D-1 — VIX9D settimanale: fondamenta, modelli e programma

*Generato da `python -m src.vix9d.run_fase1`. Tutti i numeri vengono dai file in `reports/vix9d/`, prodotti dal codice sul solo campione di sviluppo
({info['campione'][0]} → {info['campione'][1]}, {info['n_settimane']} settimane). Le {au['n_settimane_holdout']} settimane successive al confine sono salvate a parte e bloccate dalla guardia: nessuna misura le usa.
Il lavoro sul VIX resta fermo: le versioni F1 e FS sono congelate e il loro esame sull'holdout non è stato eseguito.*

## Sintesi in 10 righe

1. **Dati.** VIX9D giornaliero ufficiale Cboe, candele settimanali con le convenzioni del VIX: {ws['n_settimane']} settimane di sviluppo dal {ws['prima_data']}. Fino al {gd['ultima_seduta_con_solo_chiusura']} il file Cboe ha solo le chiusure: le candele vere sono {ws['settimane_con_candela_vera']} (dal {ws['prima_settimana_con_candela_vera']}).
2. **Il VIX9D si muove molto più del VIX**: deviazione standard delle variazioni settimanali {n(v9['dev_std'], 3)} contro {n(vv['dev_std'], 3)}; chiude oltre +20% sulla settimana prima nel {pc(a9['rialzi_oltre_20']['quota'])} delle settimane (VIX: {pc(av['rialzi_oltre_20']['quota'])}) e torna verso la media più in fretta (mezza vita {n(a9['ar1']['mezza_vita_settimane'], 1)} settimane contro {n(av['ar1']['mezza_vita_settimane'], 1)}).
3. **Rapporto VIX9D / VIX**: mediana {n(rp['mediana'], 3)}, sopra 1 nel {pc(rp['quota_sopra_1'], 0)} delle settimane. Dopo gli estremi il VIX9D rientra: nel decile più basso del rapporto la variazione mediana della settimana dopo è {pct(d1['variazione_log_mediana_1'])}, nel più alto {pct(d10['variazione_log_mediana_1'])}.
4. **Chiusura.** Il modello tipo HAR (R1) batte il random walk: R² fuori campione {pct(r1_1['r2_vs_r0'])} a 1 settimana e {pct(r1_2['r2_vs_r0'])} a 2 ({int(r1_1['n'])} e {int(r1_2['n'])} previsioni). La candela del VIX9D (R3) non aggiunge nulla; il rapporto con il VIX (R2) sì: {pct(r2_1['r2_vs_r1'])} e {pct(r2_2['r2_vs_r1'])} su R1.
5. **Fasce.** Coperture vicine al nominale: {pc(cf1['copertura_50'], 0)} / {pc(cf1['copertura_80'], 0)} / {pc(cf1['copertura_90'], 0)} a 1 settimana e {pc(cf2['copertura_50'], 0)} / {pc(cf2['copertura_80'], 0)} / {pc(cf2['copertura_90'], 0)} a 2 (fasce al 50 / 80 / 90%), ma larghe: la fascia all'80% misura in media {n(cf1['larghezza_80'], 1)} punti a 1 settimana.
6. **Direzione.** La logistica di base indovina il verso nel {pc(dB1['accuratezza'], 0)} dei casi a 1 settimana e nel {pc(dB2['accuratezza'], 0)} a 2; l'etichetta è non neutra nel {pc(dB1['quota_non_neutre_60'], 0)} e nel {pc(dB2['quota_non_neutre_60'], 0)} dei casi, ed è allora corretta nel {pc(dB1['accuratezza_60'], 0)} e nel {pc(dB2['accuratezza_60'], 0)}.
7. **Massimo e minimo.** Entro il 5% dal reale: massimo {pc(g1_1['massimo_entro_5'], 0)} e minimo {pc(g1_1['minimo_entro_5'], 0)} per la candela +1; {pc(g1_2['massimo_entro_5'], 0)} e {pc(g1_2['minimo_entro_5'], 0)} per la +2. La regola fissa fa {pc(fis_M['entro_5'], 0)} e {pc(fis_m['entro_5'], 0)} sulla +1: i modelli la migliorano poco.
8. **Versioni.** G1 (solo VIX9D): {sc['chiusura']['g1']['1']} per chiusura e fasce, {sc['estremi']['g1']['massimo']['1']} per gli estremi, logistica di base. GS (con il VIX): {sc['chiusura']['gs']['1']} per la chiusura; direzione con il rapporto {'solo a 2 settimane' if sc['direzione']['gs']['1'] == 'BASE' and sc['direzione']['gs']['2'] == 'RAPP' else 'dove batte la base'}; minimo della +2 con {sc['estremi']['gs']['minimo']['2']}. Entrambe congelate in `versioni/vix9d/`.
9. **Quanto aggiunge il VIX**: {n_agg} confronti su {n_reg} superano la regola ({elenco(si)}); negli altri ({elenco(no)}) GS resta uguale a G1. In punti: l'errore medio della chiusura centrale passa da {n(g1_1['mae_centrale'])} a {n(gs_1['mae_centrale'])} (+1) e da {n(g1_2['mae_centrale'])} a {n(gs_2['mae_centrale'])} (+2).
10. **Rispetto al VIX** l'errore sulla chiusura in log è {n(num['rapporto_rmse'])} volte più grande, quasi quanto il rapporto tra le variabilità dei due indici ({n(num['rapporto_dev_std'])}); massimo e minimo della +1 entro il 5% riescono nel {pc(g1_1['massimo_entro_5'], 0)} e nel {pc(g1_1['minimo_entro_5'], 0)} dei casi, contro {pc(num['vix']['massimo_5'], 0)} e {pc(num['vix']['minimo_5'], 0)} sul VIX. Programma: `python -m src.vix9d.forecast`. Piano d'esame scritto e non eseguito.

## 1. Audit dei dati

| voce | valore |
|---|---|
| fonte | {au['fonte']} |
| sedute giornaliere nel campione di sviluppo | {gd['n_sedute']} ({gd['prima_data']} → {gd['ultima_data']}) |
| valori mancanti / non positivi / date duplicate | {gd['valori_mancanti']} / {gd['valori_non_positivi']} / {gd['date_duplicate']} |
| sedute con apertura, massimo, minimo e chiusura incoerenti | {gd['sedute_ohlc_incoerenti']} |
| sedute con un solo valore (apertura = massimo = minimo = chiusura) | {gd['sedute_con_solo_chiusura']}, l'ultima il {gd['ultima_seduta_con_solo_chiusura']} |
| giorni feriali senza seduta (festività) | {gd['giorni_feriali_senza_seduta']} |
| chiusura giornaliera minima / massima | {n(gd['chiusura_min_max'][0])} / {n(gd['chiusura_min_max'][1])} |
| variazioni giornaliere oltre il 30% in valore assoluto (log) | {gd['variazione_log_giornaliera']['n_oltre_30pct_in_valore_assoluto']}; oltre il 50%: {len(gd['variazione_log_giornaliera']['date_oltre_50pct'])} (elenco in `audit.json`) |
| settimane di sviluppo | {ws['n_settimane']} ({ws['prima_data']} → {ws['ultima_data']}), settimane mancanti: {ws['settimane_mancanti']} |
| settimane con meno di 5 sedute | {ws['settimane_con_meno_di_5_sedute']} |
| settimane con sole chiusure / con candela vera | {ws['settimane_con_solo_chiusure']} / {ws['settimane_con_candela_vera']} (prima candela vera: {ws['prima_settimana_con_candela_vera']}) |
| apertura della settimana / chiusura della settimana prima | mediana {n(ap['mediana'], 3)}; 10° – 90° percentile {n(ap['p10'], 3)} – {n(ap['p90'], 3)}; sopra 1 nel {pc(ap['quota_sopra_1'], 0)} delle {ap['n']} settimane |
| settimane senza VIX / con ultima seduta diversa da quella del VIX | {ws['settimane_senza_vix']} / {ws['settimane_con_ultima_seduta_diversa_dal_vix']} |
| settimane di holdout (bloccate) | {au['n_settimane_holdout']} |

- **Convenzioni**: settimana da lunedì a venerdì; apertura = apertura ufficiale della prima seduta; massimo e minimo = estremi delle sedute; chiusura = chiusura dell'ultima seduta; settimana incompleta esclusa. Sono quelle del file del VIX.
- **Niente correzioni.** Nessun valore è stato modificato o rimosso. Le variazioni giornaliere molto grandi sono tante perché l'indice è fatto così (vedi sezione 2), non perché i dati siano sbagliati: non risultano valori incoerenti.
- **Le candele mancano fino a novembre 2013.** Le chiusure sono usate dal 2011; tutto ciò che richiede apertura, massimo o minimo usa solo le settimane con candela vera (colonna `candela`). Conseguenza: i modelli con la candela e le previsioni di massimo e minimo
  hanno la prima stima dopo {info['min_settimane']} settimane con candela, cioè dal {org['1']['prima_origine_comune']}.
- **File**: `data/vix9d/vix9d_weekly_dev.csv` è l'unico file del VIX9D letto dalle misure. Il file completo, il VIX settimanale ricostruito da Cboe e il file di holdout sono bloccati dalla guardia (`src/vix9d/dati.py`) e letti solo dal programma di previsione.
- Sulla parte di sviluppo il VIX settimanale ricostruito dai dati giornalieri Cboe coincide con quello del progetto (lo verifica un test).

## 2. Analisi compatta e confronto con il VIX

![Livelli e rapporto](figures/a_livelli_e_rapporto.png)

**Livelli e variazioni** (chiusure settimanali, stesse {a9['n_settimane']} settimane per i due indici):

{t_liv}

- Il VIX9D sta in media un po' sotto il VIX (mediana {n(a9['livello']['mediana'])} contro {n(av['livello']['mediana'])}) ma ha code più lunghe: il suo massimo settimanale di chiusura è {n(a9['livello']['max'], 1)} contro {n(av['livello']['max'], 1)}.
- I due indici si muovono insieme (correlazione delle variazioni log settimanali {n(rp['correlazione_variazioni_log'])}), ma il VIX9D amplifica: la sua variazione tipica è {n(v9['dev_std'] / vv['dev_std'])} volte quella del VIX.
- Uno scostamento dalla media si dimezza in {n(a9['ar1']['mezza_vita_settimane'], 1)} settimane (VIX: {n(av['ar1']['mezza_vita_settimane'], 1)}). L'autocorrelazione delle variazioni a 1 settimana è {n(a9['acf_variazioni']['1'])} (VIX: {n(av['acf_variazioni']['1'])}): dopo un rialzo tende a seguire un ribasso, più che sul VIX.

**Candele** (solo settimane con candela vera):

{t_can}

- L'apertura ufficiale della settimana è di norma molto sopra la chiusura del venerdì precedente (mediana {pct(ap['mediana'] - 1)}), poi la settimana scende: solo il {pc(k9['quota_chiusura_sopra_apertura'], 0)} delle settimane chiude sopra la propria apertura (VIX: {pc(kv['quota_chiusura_sopra_apertura'], 0)}).
  Una spiegazione plausibile, non verificata qui, è il calendario: i 9 giorni dell'indice contengono due fine settimana il venerdì e uno solo il lunedì. Per la previsione conta il fatto, che è stabile; la candela prevista ne tiene conto (sezione 6).

**Per sottoperiodi:**

{t_sp}

**Che cosa succede dopo gli estremi del rapporto VIX9D / VIX** (decili del rapporto alla chiusura settimanale; variazioni log del VIX9D):

{t_dr}

![Variazioni e rapporto](figures/a_variazioni_e_rapporto.png)

- Il rapporto ha mediana {n(rp['mediana'], 3)} (5° – 95° percentile: {n(rp['p05'], 2)} – {n(rp['p95'], 2)}) ed è sopra 1 nel {pc(rp['quota_sopra_1'], 0)} delle settimane.
- Quando il VIX9D è molto sotto il VIX (primo decile, {int(d1['n_1'])} casi), la settimana dopo è più alto nel {pc(d1['quota_rialzo_1'], 0)} dei casi e dopo due nel {pc(d1['quota_rialzo_2'], 0)}. Quando è molto sopra (ultimo decile, {int(d10['n_1'])} casi) è più alto solo nel {pc(d10['quota_rialzo_1'], 0)} dei casi,
  con variazione mediana {pct(d10['variazione_log_mediana_1'])} a 1 settimana e {pct(d10['variazione_log_mediana_2'])} a 2. Questa tabella è descrittiva e usa tutto il campione: la misura fuori campione di quanto vale il rapporto è nelle sezioni 3-7.

## 3. Chiusura (Parte B)

Modelli della lista chiusa, stimati su log(VIX9D) con finestra espandibile e ristima a ogni origine; prima stima su almeno {info['min_settimane']} settimane. **Origini comuni** = origini in cui esistono tutti i candidati: {org['1']['comuni']} a 1 settimana
(dal {org['1']['prima_origine_comune']}) e {org['2']['comuni']} a 2. È il campione su cui si applicano le regole.

{t_b}

Per R0, R1 e R2, che non usano la candela, anche tutte le origini disponibili ({org['1']['tutte_chiusura']} a 1 settimana, dal {org['1']['prima_origine_chiusura']}):

{t_b2}

- **R1 batte R0** a entrambi gli orizzonti (Clark-West, {pe(r1_1['p'])} e {pe(r1_2['p'])}). In punti però l'errore medio cala poco ({n(B(1, 'R0')['mae_punti'])} → {n(r1_1['mae_punti'])} a 1 settimana) e la quota entro il 5% è più bassa di quella del random walk
  ({pc(r1_1['entro_5'])} contro {pc(B(1, 'R0')['entro_5'])}): il modello guadagna sulle settimane lontane dalla media, non su quelle tranquille.
- **R3 (candela del VIX9D) {'non batte' if not RG('chiusura senza VIX', 1, 'R3')['batte'] else 'batte'} R1**: R² contro R1 {pct(r3_1['r2_vs_r1'])} a 1 settimana e {pct(B(2, 'R3')['r2_vs_r1'])} a 2. Escursione e posizione della chiusura della settimana non aiutano a prevedere la chiusura successiva.
- **R2 (rapporto con il VIX) batte R1**: R² {ic(r2_1['r2_vs_r1'], r2_1['r2_ci_lo'], r2_1['r2_ci_hi'])} a 1 settimana e {ic(r2_2['r2_vs_r1'], r2_2['r2_ci_lo'], r2_2['r2_ci_hi'])} a 2; p dopo Holm nella famiglia della regola {pv(RG('chiusura con VIX', 1, 'R2')['p_holm'])} e {pv(RG('chiusura con VIX', 2, 'R2')['p_holm'])} (tabella della sezione 6).
  {'Anche R4 supera il test, ma con errore più alto di R2: la regola sceglie R2.' if RG('chiusura con VIX', 1, 'R4')['batte'] and B(1, 'R4')['rmse_log'] > r2_1['rmse_log'] else 'R4 non viene scelto: vedi la tabella delle regole.'}
  Nella tabella qui sopra la correzione di Holm è fatta su tutti e 6 i confronti con R1 (più severa di quella per famiglia): {'R2 resta sotto il 5% a entrambi gli orizzonti' if max(r2_1['p_holm'], r2_2['p_holm']) < 0.05 else 'R2 non resta sotto il 5% a entrambi gli orizzonti'}.
- Entro il 10% dal valore reale cade circa il {pc(r1_1['entro_10'], 0)} delle previsioni a 1 settimana e il {pc(r1_2['entro_10'], 0)} a 2: è la misura di quanto resta imprevedibile.

## 4. Fasce, probabilità e direzione (Parte C)

**Fasce**: regressione quantilica ai quantili 5, 10, 25, 50, 75, 90, 95% sulle variabili del modello di chiusura. La chiusura centrale delle versioni è il quantile al 50%.

{t_cf}

- Le coperture sono vicine al nominale per tutte le varianti. Il CRPS è calcolato sui 7 quantili (è un'approssimazione, la stessa usata per il VIX).
- Le fasce sono larghe perché l'indice lo è: quella al 90% misura in media {n(cf1['larghezza_90'], 1)} punti a 1 settimana, su un livello mediano di {n(a9['livello']['mediana'], 1)}.

**Probabilità di chiudere sopra 20, 25, 30**: centro del modello di chiusura più la distribuzione degli errori passati (almeno {info['min_errori']} errori con esito già noto).

{t_cs}

- {"Le probabilità hanno un Brier più basso sia della frequenza storica sia della regola «resta dov'è» a tutte le soglie e a entrambi gli orizzonti." if soglie_ok else "Le probabilità non battono entrambi i riferimenti a tutte le soglie: vedi tabella."} La probabilità media prevista è vicina alla frequenza osservata dell'evento.

**Direzione** della chiusura rispetto all'origine: logistica con penalizzazione, variabili di R1 (base) e con ln(VIX9D/VIX). Etichetta «neutra» tra {int(100 * G.NEUTRA[0])}% e {int(100 * G.NEUTRA[1])}%.

{t_cd}

Etichette separate per verso, alle tre soglie:

{t_cd2}

- Il VIX9D sale rispetto alla settimana prima nel {pc(dB1['frequenza_rialzo'], 0)} dei casi. La logistica di base batte la frequenza storica (Brier, {pe(dB1['p_freq'])} a 1 settimana e {pe(dB2['p_freq'])} a 2).
- **Tra le etichette non neutre, {int(dB1['n_ribasso_60'])} su {int(dB1['n_non_neutre_60'])} a 1 settimana sono «ribasso»** ({int(dB2['n_ribasso_60'])} su {int(dB2['n_non_neutre_60'])} a 2). Il modello si sbilancia quando l'indice è alto e ci si aspetta che rientri; quasi mai nel verso del rialzo.
- Alzando la soglia l'accuratezza sale ({pc(dB1['accuratezza_60'], 0)} → {pc(dB1['accuratezza_65'], 0)} → {pc(dB1['accuratezza_70'], 0)} a 1 settimana) e i casi calano ({int(dB1['n_non_neutre_60'])} → {int(dB1['n_non_neutre_65'])} → {int(dB1['n_non_neutre_70'])}).
- Con il rapporto, a 2 settimane l'etichetta è non neutra nel {pc(dR2['quota_non_neutre_60'], 0)} dei casi e corretta nel {pc(dR2['accuratezza_60'], 0)} ({int(dR2['n_non_neutre_60'])} casi).

## 5. Massimo e minimo (Parte D)

Regressione mediana del logaritmo di massimo (o minimo) della settimana prevista rispetto alla chiusura dell'origine, solo sulle settimane con candela vera. Riferimento: regola fissa (mediana storica del rapporto tra estremo e chiusura dell'origine).

**Tutte le settimane:**

{tab_estremi('tutte')}

**Senza le settimane di strappo** (chiusura reale oltre +{int(100 * info['strappo'])}% sull'origine: {org['1']['strappi_tra_le_comuni']} su {org['1']['comuni']} a 1 settimana, {org['2']['strappi_tra_le_comuni']} su {org['2']['comuni']} a 2; è un'etichetta nota solo dopo):

{tab_estremi('senza strappi')}

**Per fascia di livello** (quintile della chiusura all'origine sulla storia disponibile fino a quel momento; insieme S0):

{t_q}

- Il massimo è il più difficile: entro il 5% nel {pc(X(1, 'S0', 'massimo')['entro_5'], 0)} dei casi per la +1 e nel {pc(X(2, 'S0', 'massimo')['entro_5'], 0)} per la +2. Gli errori oltre il 5% sono divisi quasi a metà tra eccesso e difetto: la previsione è centrata, è la dispersione che è grande.
- **Le settimane di strappo sono il {pc(q_str[1], 0)} delle origini a 1 settimana** (nel campione intero il VIX9D chiude oltre +20% nel {pc(a9['rialzi_oltre_20']['quota'], 0)} delle settimane, il VIX nel {pc(av['rialzi_oltre_20']['quota'], 0)}): toglierle migliora le quote di pochi punti (massimo della +1: {pc(X(1, 'S0', 'massimo')['entro_5'], 0)} → {pc(X(1, 'S0', 'massimo', 'senza strappi')['entro_5'], 0)}). L'errore non viene solo dai picchi.
- Nessun insieme batte S0 senza il VIX; con il VIX solo il minimo della +2 (S2). Rispetto alla regola fissa, S0 guadagna sul minimo ({pc(fis_m['entro_5'], 0)} → {pc(X(1, 'S0', 'minimo')['entro_5'], 0)} entro il 5% sulla +1) e non sul massimo ({pc(fis_M['entro_5'], 0)} → {pc(X(1, 'S0', 'massimo')['entro_5'], 0)}).

## 6. Candele e versioni (Parte E)

**Regole applicate** (fissate prima dei risultati: scelta per componente e per orizzonte; Holm dentro ogni famiglia; errore quadratico in log per la chiusura, assoluto in log per gli estremi, Brier per la direzione):

{t_rg}

**Composizione delle versioni:**

{t_sc}

- **G1** usa solo lo storico del VIX9D. {'Nessun candidato senza VIX batte il riferimento: G1 è fatta di R1, S0 e logistica di base.' if not batte_senza else 'Almeno un candidato senza VIX batte il riferimento ed entra in G1.'}
- **GS** parte da G1 e sostituisce un componente solo dove il candidato con ln(VIX9D/VIX) batte quello di G1.
- **Candela**: apertura = chiusura precedente × rapporto mediano di apertura (ultima origine: {n(comp['rapporto_apertura'], 3)}); massimo e minimo previsti comandano; il corpo ha la direzione più frequente nella storia (chiusura sopra l'apertura nel {pc(comp['p_su'], 0)} dei casi: quindi verso il basso)
  e l'ampiezza media delle ultime {info['finestra_corpo']} settimane ({n(comp['corpo'], 3)} in log), tagliato dentro l'intervallo minimo–massimo. Il colore segue l'etichetta di direzione della chiusura rispetto all'origine; grigio se neutra.
  Per la candela +2 la chiusura precedente è la chiusura centrale prevista per la +1.

**Pagella delle due candele, tutte le settimane:**

{tab_versioni('tutte')}

**Senza le settimane di strappo:**

{tab_versioni('senza strappi')}

- La chiusura reale cade dentro l'escursione prevista circa una volta su due ({pc(g1_1['chiusura_nell_escursione'], 0)} sulla +1); l'escursione prevista è più stretta di quella reale ({n(g1_1['rapporto_escursione'])} volte), perché massimo e minimo sono mediane e non coprono le code.
- Il corpo viene tagliato dagli estremi nel {pc(g1_1['quota_chiusura_tagliata'], 0)} dei casi: la chiusura della candela non coincide con la chiusura centrale, che resta la previsione della chiusura (linea viola nel grafico).
- Senza strappi l'errore medio della chiusura centrale scende da {n(g1_1['mae_centrale'])} a {n(g1_1s['mae_centrale'])} punti sulla +1 ({int(g1_1['n'])} → {int(g1_1s['n'])} casi).

## 7. Quanto aggiunge il VIX

{t_agg}

- **Il VIX aggiunge** (regola superata) su: {elenco(si)}. È l'effetto visto nella sezione 2: quando il VIX9D è lontano dal VIX tende a tornargli vicino, e il rapporto lo misura meglio della sola storia del VIX9D.
- **Non aggiunge nulla di dimostrato** su: {elenco(no)}. In questi confronti GS usa lo stesso componente di G1, quindi la differenza tra le versioni è zero per costruzione.
- Sulle fasce (fuori dalla regola, perché seguono il modello di chiusura) la variazione del CRPS è {pct(cf[(cf['h'] == 1) & (cf['modello'] == 'R2')].iloc[0]['crps_vs_r1'])} a 1 settimana ({pe(cf[(cf['h'] == 1) & (cf['modello'] == 'R2')].iloc[0]['p'])}) e {pct(cf[(cf['h'] == 2) & (cf['modello'] == 'R2')].iloc[0]['crps_vs_r1'])} a 2 ({pe(cf[(cf['h'] == 2) & (cf['modello'] == 'R2')].iloc[0]['p'])}).
- **Il minimo della +2 va letto con cautela**: con S2 l'errore assoluto in log scende ({n(RG('estremi con VIX', 'minimo|2', 'S2')['errore_riferimento'], 4)} → {n(RG('estremi con VIX', 'minimo|2', 'S2')['errore'], 4)}), ed è la perdita della regola, ma la quota entro il 5% passa da {pc(X(2, 'S0', 'minimo')['entro_5'])} a {pc(X(2, 'S2', 'minimo')['entro_5'])} e quella entro il 10% da {pc(X(2, 'S0', 'minimo')['entro_10'])} a {pc(X(2, 'S2', 'minimo')['entro_10'])}: migliora gli errori grandi, non i colpi precisi.
- In punti il guadagno è piccolo: errore medio della chiusura centrale {n(g1_1['mae_centrale'])} → {n(gs_1['mae_centrale'])} (+1) e {n(g1_2['mae_centrale'])} → {n(gs_2['mae_centrale'])} (+2). La quota di chiusure entro il 5% passa da {pc(g1_1['entro_5_chiusura'])} a {pc(gs_1['entro_5_chiusura'])} sulla +1 e da {pc(g1_2['entro_5_chiusura'])} a {pc(gs_2['entro_5_chiusura'])} sulla +2.

## 8. VIX e VIX9D a confronto

Numeri del VIX già pubblicati (`reports/fase4c/statistiche_programma.csv`, `prima_e_dopo.csv`, `a_misure.csv`; `reports/fase3/t1_metriche.csv`), versione F1; numeri del VIX9D da `reports/vix9d/e_versioni.csv`, versione G1. «n.p.» = non pubblicato per quel campione.

{t_conf}

- **I campioni non coincidono**: il VIX è misurato dal 2006 (2008 compreso), il VIX9D dal {org['1']['prima_origine_comune'][:4]}. La colonna centrale riporta il VIX sulle sole origini più recenti dove il numero era già stato pubblicato: è il confronto più vicino allo stesso periodo.
- **La chiusura**: in log l'errore del VIX9D è {n(num['rapporto_rmse'])} volte quello del VIX; il VIX9D è {n(num['rapporto_dev_std'])} volte più variabile. La difficoltà in più è quasi tutta nella natura dell'indice, non nel modello.
- **Massimo e minimo entro il 5%** (+1): {pc(num['vix']['massimo_5'], 0)} e {pc(num['vix']['minimo_5'], 0)} sul VIX, {pc(g1_1['massimo_entro_5'], 0)} e {pc(g1_1['minimo_entro_5'], 0)} sul VIX9D. Con la soglia al 10% il VIX9D arriva a {pc(g1_1['massimo_entro_10'], 0)} e {pc(g1_1['minimo_entro_10'], 0)}.
- **Direzione** (+1): l'etichetta è non neutra nel {pc(g1_1['quota_non_neutre'], 0)} dei casi sul VIX9D e nel {pc(num['vix']['non_neutre'], 0)} sul VIX; quando lo è, è corretta nel {pc(g1_1['accuratezza_non_neutre'], 0)} e nel {pc(num['vix']['accuratezza'], 0)} dei casi. Il VIX9D rientra più in fretta dopo gli eccessi, e questo è ciò che l'etichetta coglie.
- **Fasce** (+1, 50 / 80 / 90%): {' / '.join(pc(g1_1[f'copertura_{k}'], 0) for k in (50, 80, 90))} sul VIX9D, {' / '.join(pc(x, 0) for x in num['vix']['coperture'])} sul VIX (F1).

## 9. Programma di previsione

`python -m src.vix9d.forecast [--versione g1|gs] [--aggiorna]` produce la scheda `reports/vix9d/previsione_corrente.md`: lettura della situazione (livello, percentile, distanza dal VIX, dinamica a 1, 4 e 12 settimane, precedenti in situazioni simili),
tabella delle settimane +1 e +2, grafico con le due candele previste, verifica visiva sul passato, avvertenze. Scrive il registro `reports/vix9d/registro_previsioni.csv`; `--verifica` confronta il registro con le settimane completate.
È l'unico comando del VIX9D che legge i dati successivi al confine, solo come storia da cui prevedere: non calcola errori su quel periodo. Se il VIX della settimana di origine manca, GS non è calcolabile e il programma ripiega su G1 avvisando.

**Verifica visiva sul passato** ({g3['regola']}; versione G1):

![Verifica sul passato](figures/g3_verifica_passato.png)

{g3['riga']} Dodici origini servono a vedere come si comporta la previsione, non a misurarla.

## 10. Problemi e ambiguità

1. **Candele assenti prima di novembre 2013.** Scelta: chiusure dal 2011, candele solo dove sono vere. Alternativa scartata: partire dal 2013 per tutto, perdendo {ws['settimane_con_solo_chiusure']} settimane di chiusure.
2. **Storia corta.** {ws['n_settimane']} settimane contro le 1297 del VIX; il periodo di valutazione ({org['1']['comuni']} origini) non contiene il 2008. Gli intervalli sono larghi e le regole hanno poca potenza.
3. **Apertura ufficiale molto sopra la chiusura precedente** (mediana {pct(ap['mediana'] - 1)}). Non è stata corretta né spiegata con un test: è trattata come un fatto dell'indice. Se l'apertura ufficiale non interessa a chi usa la previsione, la candela si può ridisegnare a partire dalla chiusura precedente: va deciso.
4. **Regola per «batte»: precisazioni mie.** La consegna non dice su quale orizzonte né come raggruppare i test. Ho fissato prima dei risultati: scelta per componente e per orizzonte; Holm dentro ogni famiglia di confronti (tabella della sezione 6). Con una sola famiglia per tutti i test passerebbero meno candidati.
5. **Versioni composte per orizzonte.** GS usa per il minimo della +2 un insieme diverso da quello del massimo: se massimo e minimo previsti si incrociassero vengono scambiati e il caso è segnalato nel programma.
6. **Chiusura centrale = quantile al 50%**, non la previsione dei minimi quadrati usata per i test della Parte B: sono due stime dello stesso centro e differiscono di poco. I test confrontano i minimi quadrati, le versioni pubblicano la mediana (come per il VIX).
7. **Strappi**: la soglia del +20% è quella del VIX; sul VIX9D riguarda il {pc(q_str[1], 0)} delle origini a 1 settimana e il {pc(q_str[2], 0)} a 2, quindi «senza strappi» toglie molto più campione che sul VIX.
8. **Fasce di livello**: quintili sulla storia disponibile all'origine; alcune celle hanno pochi casi e sono segnalate.
9. **Confronto con il VIX**: periodi diversi e, per alcune righe, numeri non pubblicati sullo stesso campione. Non ho ricalcolato nulla sul VIX, come richiesto.
10. **L'esame del VIX9D leggerà il VIX dell'holdout** come variabile di GS (non come bersaglio). Non è una valutazione dei modelli del VIX, ma va saputo prima di decidere l'ordine dei due esami: è scritto nel piano.
11. **Dati pubblicati nel repository**: `data/vix9d/` contiene le candele settimanali, comprese quelle dopo il confine (file di holdout e file completo), come per il VIX. Se preferisci non pubblicarle, si tolgono e si rigenerano con `python -m src.vix9d.prepara --scarica`.
12. **Registro**: riusa il formato del registro del VIX; le colonne `p_vix_sopra_*` contengono qui le probabilità che il VIX9D chiuda sopra le soglie.
13. **Nessun codice esistente è stato modificato.** Le fasi del VIX rieseguite danno gli stessi file (verificato su `run_phase4c`); i moduli del VIX9D riusano le funzioni del VIX importandole.

## 11. Proposte per il passo successivo

1. **Decidere l'apertura** (punto 3): candela dall'apertura ufficiale, come ora, oppure dalla chiusura precedente.
2. **Dati giornalieri del VIX9D dentro la settimana** (variazione dell'ultima seduta, variabilità delle sedute): è il blocco che per il VIX non ha dato nulla, ma qui l'indice è più rapido e la candela settimanale non è servita.
3. **Usare la previsione del VIX** (F1) come variabile per il VIX9D a 2 settimane, visto che il rapporto con il VIX è l'unica informazione che ha aggiunto qualcosa.
4. **Effetto non lineare del rapporto**: il rientro è forte soprattutto dopo gli estremi (decili 1 e 10); una variabile a soglie fissate prima potrebbe rendere di più della forma lineare.
5. **Soglia del 10% per massimo e minimo** come misura principale del VIX9D, lasciando il 5% come misura secondaria: al 5% il massimo della +1 riesce nel {pc(g1_1['massimo_entro_5'], 0)} dei casi, al 10% nel {pc(g1_1['massimo_entro_10'], 0)}.
6. **Esame sull'holdout**: il piano è in `reports/vix9d/piano_esame_holdout.md`; resta da decidere quando eseguirlo e in che ordine rispetto a quello del VIX.
"""
    (G.OUT / "fase1_report.md").write_text(md_out)


if __name__ == "__main__":
    from . import dati
    with dati.guardia():
        run()
