"""Genera reports/fase2_report.md. Tutti i numeri sono letti dai file reports/fase2/*.csv e *.json."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .. import config as C
from ..fase1 import comune as K
from . import densita as D

R = C.REPORTS_DIR / "fase2"
MIN_N = 30
ORDER = list(D.DENSITY)
E_ORDER = ["RIF"] + ORDER + ["E1"]


def J(name: str):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def T(name: str) -> pd.DataFrame:
    return pd.read_csv(R / f"{name}.csv")


def md(df: pd.DataFrame, fmt=".3f") -> str:
    return re.sub(r"\bnan\b", "  –", df.to_markdown(index=False, floatfmt=fmt, missingval="–"))


def n(v, d: int = 2) -> str:
    return "n.d." if v is None or not np.isfinite(v) else f"{v:.{d}f}".replace(".", ",")


def pct(v, d: int = 1) -> str:
    return "n.d." if v is None or not np.isfinite(v) else f"{100 * v:.{d}f}%".replace(".", ",")


def pv(p) -> str:
    return "< 0,001" if p < 0.001 else n(p, 3)


def run() -> None:
    AI = J("a_info")
    dm, pit = T("a_densita_metriche"), T("a_pit")
    em, cal = T("a_eventi_metriche"), T("a_calibrazione")
    grp = {k: T(f"a_densita_{k}") for k in ("sottoperiodo", "fascia_origine", "dopo_picco", "periodo_calmo")}
    egrp = {k: T(f"a_eventi_{k}") for k in ("sottoperiodo", "fascia_origine", "dopo_picco", "periodo_calmo")}
    SCH = J("b_scheda_esempio")
    INV, VER = J("c_inventario"), J("c_verifica_vix")
    Dm = dm.set_index(["h", "modello"])
    Em = em.set_index(["evento", "h", "modello"])
    scelta = {int(h): m for h, m in AI["scelta_per_orizzonte"].items()}
    po = AI["per_orizzonte"]

    # ---------------------------------------------------------------- Parte A: densita'
    t = dm.copy()
    t["modello"] = pd.Categorical(t["modello"], ORDER, ordered=True)
    t = t.sort_values(["h", "modello"])
    for lev in D.INTERVALS:
        t[f"cop. {lev}% [IC 95%]"] = [f"{100 * a:.1f} [{100 * b:.1f}; {100 * c:.1f}]" for a, b, c in
                                      zip(t[f"copertura_{lev}"], t[f"copertura_{lev}_ci_lo"], t[f"copertura_{lev}_ci_hi"])]
    t["CRPS vs D1 (%)"] = 100 * t["crps_vs_d1"]
    t_main = md(t[["h", "modello", "n", "crps", "CRPS vs D1 (%)", "perdita_quantilica_media", "cop. 50% [IC 95%]", "cop. 80% [IC 95%]",
                   "cop. 90% [IC 95%]", "quantili_incrociati", "dm_stat", "p", "p_holm"]].rename(
        columns={"crps": "CRPS", "perdita_quantilica_media": "perdita quantilica media", "quantili_incrociati": "incroci",
                 "dm_stat": "DM vs D1", "p_holm": "p Holm"}), ".4f")
    qcols = [f"perdita_q{int(100 * q):02d}" for q in D.Q_REPORT]
    t_q = md(t[["h", "modello"] + qcols].rename(columns={c: c.replace("perdita_q", "q") for c in qcols}), ".4f")
    rows = []
    for r in t.itertuples():
        row = {"h": r.h, "modello": r.modello}
        for lev in D.INTERVALS:
            row[f"{lev}%: sopra"] = 100 * getattr(r, f"sopra_{lev}")
            row[f"{lev}%: sotto"] = 100 * getattr(r, f"sotto_{lev}")
            row[f"{lev}%: larghezza (punti VIX)"] = getattr(r, f"larghezza_vix_{lev}")
        row["MAE mediana (punti VIX)"] = r.mediana_mae_vix
        rows.append(row)
    t_asym = md(pd.DataFrame(rows), ".2f")
    pv_ = pit.pivot_table(index=["h", "modello"], columns="classe", values="frequenza", sort=False)[list(pit["classe"].unique())]
    att = pit.drop_duplicates("classe").set_index("classe")["attesa"]
    pit_t = (100 * pv_).round(1).reset_index()
    pit_t["modello"] = pd.Categorical(pit_t["modello"], ORDER, ordered=True)
    t_pit = md(pit_t.sort_values(["h", "modello"]), ".1f")
    att_txt = ", ".join(f"{c}: {100 * v:.0f}%" for c, v in att.items())

    def gpivot(g: pd.DataFrame, col: str, val: str, scale: float = 100.0, order=None, hs=(1, 4)):
        g = g[(g["n"] >= MIN_N) & (g["h"].isin(hs))]
        p = g.pivot_table(index=["h", col], columns="modello", values=val, sort=False)
        p = (scale * p[[m for m in ORDER if m in p.columns]]).round(1)
        nn = g[g["modello"] == "D1"].set_index(["h", col])["n"]
        p.insert(0, "n", nn)
        p = p.reset_index()
        if order:
            p[col] = pd.Categorical(p[col], order, ordered=True)
            p = p.sort_values(["h", col])
        return md(p, ".1f")

    yn = {True: "sì", False: "no"}
    sub_p = grp["dopo_picco"].assign(dopo_picco=grp["dopo_picco"]["dopo_picco"].map({True: "entro 12 sett. dopo un picco", False: "altre origini"}))
    sub_c = grp["periodo_calmo"].assign(periodo_calmo=grp["periodo_calmo"]["periodo_calmo"].map({True: "dentro un periodo calmo", False: "altre origini"}))
    G = {k: v.set_index(["h", k, "modello"]) for k, v in grp.items()}
    best_h = {h: Dm.loc[(h, scelta[h])] for h in K.HORIZONS}
    sig_better = dm[(dm["crps_vs_d1"] > 0) & (dm["p_holm"] < 0.05)]
    sig_worse = dm[(dm["crps_vs_d1"] < 0) & (dm["p_holm"] < 0.05)]
    lst = lambda d: ", ".join(f"{r.modello} a h = {r.h}" for r in d.itertuples()) or "nessuno"
    cov_ok = dm[(dm["copertura_90_ci_lo"] <= 0.90) & (dm["copertura_90_ci_hi"] >= 0.90)]
    n_tests_d = int(dm["p"].notna().sum())
    cross = {h: po[str(h)]["incroci"].get("D4", 0) for h in K.HORIZONS}
    fb = lambda h, b, m, c: G["fascia_origine"].loc[(h, b, m), c]

    # ---------------------------------------------------------------- Parte A: eventi
    ev_list = [(e, h) for h in K.HORIZONS for e in D.events_for(h)]
    rows = []
    for e, h in ev_list:
        row = {"evento": e, "h": h, "n": int(Em.loc[(e, h, "RIF"), "n"]), "freq. evento": Em.loc[(e, h, "RIF"), "frequenza_evento"]}
        for m in E_ORDER:
            if (e, h, m) in Em.index:
                row[m] = Em.loc[(e, h, m), "brier"]
        rows.append(row)
    t_brier = md(pd.DataFrame(rows), ".4f")
    rows = []
    for e, h in ev_list:
        row = {"evento": e, "h": h}
        for m in E_ORDER[1:]:
            if (e, h, m) in Em.index:
                r = Em.loc[(e, h, m)]
                row[m] = f"{100 * r['brier_skill_vs_rif']:+.1f}" + ("*" if r["p_holm"] < 0.05 else "")
        rows.append(row)
    t_bss = md(pd.DataFrame(rows))
    dec = em[em["h"].isin([1, 4]) & em["modello"].isin(["RIF", "D1", "D4", "D5", "E1"])].copy()
    dec["modello"] = pd.Categorical(dec["modello"], E_ORDER, ordered=True)
    t_dec = md(dec.sort_values(["h", "evento", "modello"])[["evento", "h", "modello", "prob_media", "frequenza_evento", "brier", "affidabilita",
                                                           "risoluzione", "incertezza"]].rename(columns={"prob_media": "prob. media", "frequenza_evento": "freq. evento",
                                                                                                         "affidabilita": "affidabilità"}), ".4f")
    n_tests_e = int(em["p"].notna().sum())
    e_sig = em[(em["p_holm"] < 0.05) & (em["brier_skill_vs_rif"] > 0)]
    e_sig_txt = "; ".join(f"{e} a h = {h}: {', '.join(g['modello'])}" for (e, h), g in e_sig.groupby(["evento", "h"], sort=False)) or "nessuno"
    e_neg = em[(em["p_holm"] < 0.05) & (em["brier_skill_vs_rif"] < 0)]
    e_neg_txt = "; ".join(f"{r.modello} per {r.evento} a h = {r.h}" for r in e_neg.itertuples()) or "nessuno"
    rise = em[(em["evento"] == "rialzo>20%")].set_index("modello")
    eg = egrp["fascia_origine"]
    eg = eg[(eg["evento"] == "VIX>25") & (eg["h"].isin([1, 4])) & (eg["n"] >= MIN_N)]
    pe = (100 * eg.pivot_table(index=["h", "fascia_origine"], columns="modello", values="brier_skill_vs_rif", sort=False)[[m for m in E_ORDER[1:] if m in eg["modello"].unique()]]).round(1)
    pe.insert(0, "freq. evento (%)", (100 * eg[eg["modello"] == "RIF"].set_index(["h", "fascia_origine"])["frequenza_evento"]).round(1))
    pe.insert(0, "n", eg[eg["modello"] == "RIF"].set_index(["h", "fascia_origine"])["n"])
    pe = pe.reset_index()
    pe["fascia_origine"] = pd.Categorical(pe["fascia_origine"], C.BAND_LABELS, ordered=True)
    t_e_fascia = md(pe.sort_values(["h", "fascia_origine"]), ".1f")
    bss = lambda e, h, m: Em.loc[(e, h, m), "brier_skill_vs_rif"]
    bss_rng = lambda e, h: (min(bss(e, h, m) for m in ORDER + ["E1"] if (e, h, m) in Em.index), max(bss(e, h, m) for m in ORDER + ["E1"] if (e, h, m) in Em.index))

    # ---------------------------------------------------------------- Parte B: scheda
    rows = []
    for h, s in SCH["orizzonti"].items():
        row = {"orizzonte (settimane)": h, "modello": s["modello"], "mediana": s["mediana"]}
        for lev in ("50", "80", "90"):
            row[f"intervallo {lev}%"] = f"{s['intervalli'][lev][0]:.1f} – {s['intervalli'][lev][1]:.1f}"
        for e in ("VIX>20", "VIX>25", "VIX>30", "rialzo>20%"):
            row[f"P({e})"] = f"{100 * s['probabilita'][e]:.1f}%" if e in s["probabilita"] else "–"
        rows.append(row)
    t_scheda = md(pd.DataFrame(rows), ".1f")
    zero_p = [(h, e) for h, s in SCH["orizzonti"].items() for e, v in s["probabilita"].items() if v == 0.0]

    # ---------------------------------------------------------------- Parte C: dati esterni
    if INV is None:
        part_c = ("*L'inventario non è presente: eseguire `python -m src.fetch_external` e poi rigenerare il report con "
                  "`python -m src.fase2.report`.*\n")
        c_syn = "Parte C non eseguita."
        missing_txt, ver_txt, n_ok, n_tot, nfci_note, c_manual = "", "", 0, 0, "", "n.d."
    else:
        ser = INV["serie"]
        ok = {k: v for k, v in ser.items() if v["stato"] == "scaricata"}
        ko = {k: v for k, v in ser.items() if v["stato"] != "scaricata"}
        n_ok, n_tot = len(ok), len(ser)
        rows = []
        for k, v in ok.items():
            w = v["settimanale_dev"]
            rows.append({"serie": k, "descrizione": v.get("descrizione", ""), "fonte": v["fonte"], "frequenza": v["frequenza"],
                         "prima data": v["prima_data"], "ultima data": v["ultima_data"], "osservazioni": v["n_osservazioni_totali"],
                         "prima settimana dev con valore": w["prima_settimana_con_valore"],
                         "settimane dev con valore": f"{w['settimane_con_valore']} / {w['settimane_dev_totali']}",
                         "settimane mancanti dopo l'inizio (dev)": w["settimane_mancanti_dopo_l_inizio"],
                         "giorni feriali senza valore, festività comprese (dev)": v["giorni_feriali_senza_valore_dev"] if v["giorni_feriali_senza_valore_dev"] is not None else "–",
                         "ritardo usato (giorni)": v["ritardo_giorni_usato"]})
        t_inv = md(pd.DataFrame(rows))
        t_src = md(pd.DataFrame([{"serie": k, "indirizzo": v["indirizzo"], "data di scarico": v["data_scarico"],
                                  "ritardo di pubblicazione": v["ritardo_pubblicazione"],
                                  "avvertenze": v.get("avvertenza", "")} for k, v in ok.items()]))
        corte = {k: v for k, v in ok.items() if v.get("storia_insufficiente")}
        manual = [{"serie": k, "descrizione": v["descrizione"], "fonte": v["fonte"], "problema": "non scaricabile: " + v["motivo"],
                   "file da procurare a mano": v["file_da_procurare"]} for k, v in ko.items()]
        manual += [{"serie": k, "descrizione": v["descrizione"], "fonte": v["fonte"],
                    "problema": f"scaricata ma con storia insufficiente (dal {v['prima_data']}): {v['avvertenza']}",
                    "file da procurare a mano": v["file_da_procurare"]} for k, v in corte.items()]
        if manual:
            missing_txt = ("**Serie da procurare a mano** (non sostituite con altro):\n\n" + md(pd.DataFrame(manual)))
        else:
            missing_txt = "**Serie da procurare a mano:** nessuna. Tutte le serie richieste sono state scaricate con storia sufficiente."
        nf = ok.get("nfci")
        nfci_note = (f"L'NFCI scaricato si ferma al {nf['ultima_data']}, mesi prima della data di scarico ({INV['data_scarico']}): "
                     "l'indirizzo usato potrebbe non essere più aggiornato dalla fonte. " if nf and nf["ultima_data"] < INV["data_scarico"][:5] + "08" else "")
        short = [k for k, v in ok.items() if v["settimanale_dev"]["prima_settimana_con_valore"] and v["settimanale_dev"]["prima_settimana_con_valore"] > "2005-01-01"]
        if VER:
            hyp = VER["ipotesi_open"]
            nw = VER["settimane_confrontate"]
            og = VER["open_grezzo_meno_ufficiale"]
            inc = VER.get("settimane_ufficiali_ohlc_incoerenti", [])
            salto = VER["salto_apertura_ufficiale"]
            if VER["open"]["uguali"] >= 0.99 * nw:
                open_txt = (f"- **La colonna `open` del nostro file è l'apertura ufficiale Cboe della prima seduta della settimana**: coincide in {VER['open']['uguali']} settimane su {nw}. "
                            "Non è un errore del file.\n"
                            f"- **Le due stranezze segnalate in Fase 0 sono nella fonte ufficiale.** Il salto tra chiusura e apertura successiva, calcolato sui dati ufficiali, ha mediana {pct(salto['mediana'])} "
                            f"ed è positivo nel {pct(salto['quota_positivi'], 0)} delle settimane, come nel file. "
                            + (("Le settimane in cui le barre ufficiali sono incoerenti (apertura fuori dall'intervallo minimo–massimo) sono " + str(len(inc)) + ": "
                                + "; ".join(f"{r['week_start']} (apertura ufficiale {n(r['open_ufficiale'])}, minimo {n(r['low_ufficiale'])}, massimo {n(r['high_ufficiale'])})" for r in inc[:5])
                                + ". Il valore anomalo del 2004 è quindi presente tale e quale nei dati Cboe.\n") if inc else "Nessuna barra ufficiale è incoerente.\n")
                            + "- **Conclusione.** Il file è una trascrizione fedele dei dati ufficiali. È il valore di apertura pubblicato da Cboe a non essere confrontabile con la chiusura precedente "
                              "(una spiegazione plausibile, che non ho verificato, è che l'apertura dell'indice sia calcolata sulle prime quotazioni delle opzioni, con differenziali denaro-lettera più ampi). "
                              "La decisione di non usare `open` resta corretta, per un motivo diverso da quello ipotizzato in Fase 0: non un errore di trascrizione, ma un limite della grandezza stessa.")
            else:
                best_h = max(hyp, key=lambda k: hyp[k]["uguali"])
                open_txt = (f"- Il candidato che coincide più spesso è «{best_h}» ({hyp[best_h]['uguali']} settimane su {nw}, {pct(hyp[best_h]['uguali'] / nw, 0)}).\n"
                            f"- Rispetto all'apertura ufficiale del primo giorno, l'`open` del file è più alto nel {pct(og['quota_grezzo_maggiore'], 0)} delle settimane e più basso nel {pct(og['quota_grezzo_minore'], 0)} (differenza mediana {n(og['mediana'], 3)} punti).\n"
                            f"- Il salto tra chiusura e apertura successiva calcolato sui dati ufficiali ha mediana {pct(salto['mediana'])} ed è positivo nel {pct(salto['quota_positivi'], 0)} delle settimane.")
            all_eq = all(VER[c]["diversi"] == 0 for c in ("close", "high", "low"))
            ver_txt = f"""### 4.3 Verifica del nostro file con il VIX giornaliero ufficiale

Dal VIX giornaliero di Cboe ho ricostruito le barre settimanali (lunedì–domenica: chiusura dell'ultima seduta, massimo e minimo della settimana,
apertura della prima seduta) e le ho confrontate con il file grezzo, **solo sul periodo di sviluppo** ({VER['periodo'][0]} → {VER['periodo'][1]},
{nw} settimane; tolleranza {n(VER['tolleranza'], 3)} punti).

| colonna | settimane uguali | settimane diverse | differenza massima | differenza mediana tra le diverse |
|---|---|---|---|---|
""" + "\n".join(f"| `{c}` | {VER[c]['uguali']} | {VER[c]['diversi']} | {n(VER[c]['differenza_massima'])} | {n(VER[c]['differenza_mediana_tra_i_diversi'])} |"
                for c in ("close", "high", "low", "open")) + f"""

Ultima seduta della settimana uguale in {VER['last_trading_day_uguali']} settimane su {nw}; numero di sedute uguale in {VER['trading_days_uguali']}.
{'**Chiusura, massimo e minimo coincidono in tutte le settimane.**' if all_eq else 'Le settimane con differenze in chiusura, massimo o minimo sono elencate in `reports/fase2/c_verifica_vix_differenze.csv`.'}

**Cosa dice sulla colonna `open`.** Confronto della colonna `open` del file grezzo con cinque candidati ricavati dai dati ufficiali:

| candidato | settimane uguali (su {nw}) | differenza assoluta mediana |
|---|---|---|
""" + "\n".join(f"| {k} | {v['uguali']} | {n(v['differenza_mediana'], 3)} |" for k, v in hyp.items()) + f"""

{open_txt}
"""
        else:
            ver_txt = "### 4.3 Verifica del nostro file\n\n*Non eseguita: il VIX giornaliero ufficiale non è stato scaricato.*\n"
        part_c = f"""Comando separato: `python -m src.fetch_external` (ultimo scarico: {INV['data_scarico']}). Nessun modello, nessuna analisi, nessuna correlazione:
solo raccolta, documentazione e allineamento. **Serie scaricate: {n_ok} su {n_tot}.**

### 4.1 Inventario

{t_inv}

Le colonne «prima/ultima data» e «osservazioni» riguardano l'intera serie scaricata; tutti gli altri conteggi sono calcolati **solo fino al {INV['confine_dev_holdout']}**.
Nessun valore successivo a quella data è stampato o riportato.

Fonti, indirizzi e ritardi di pubblicazione:

{t_src}

{missing_txt}

### 4.2 File prodotti, allineamento e divisione dev/holdout

- **Originali**: `{INV['file_prodotti']['originali']}/`, salvati così come scaricati (impronta sha256 nell'inventario `reports/fase2/c_inventario.json`).
- **Allineamento settimanale**: {INV['regola_allineamento']}. Per ogni serie è salvata anche la data dell'osservazione usata (colonna `<serie>_data_oss`): un test verifica che non sia mai successiva alla data settimanale meno il ritardo.
- **Sviluppo**: {', '.join(f'`{f}`' for f in INV['file_prodotti']['dev'])} (date ≤ {INV['confine_dev_holdout']}).
- **Holdout**: {', '.join(f'`{f}`' for f in INV['file_prodotti']['holdout_protetto'])}. La guardia usata da `run_phase1` e `run_phase2` blocca l'apertura di tutto ciò che sta in `data/external/holdout/` e in `data/external/raw/` (gli originali contengono anche il periodo di holdout).
- **Serie con storia più corta del campione** (prima settimana con valore dopo il 2005): {', '.join(f"{k} (dal {ok[k]['settimanale_dev']['prima_settimana_con_valore']})" for k in short) or 'nessuna'}.
- **Pubblicazione nel repository**: `data/external/` è escluso dal repository (`.gitignore`): contiene dati di terzi (Cboe, ICE, S&P) la cui ridistribuzione pubblica non è una decisione che spetta a me. Nel repository ci sono il programma di scarico e l'inventario.

{ver_txt}"""
        c_syn = (f"scaricate {n_ok} serie su {n_tot}" + (f"; non scaricabili: {', '.join(ko)}" if ko else "")
                 + (f"; con storia insufficiente: {', '.join(corte)}" if corte else "")
                 + (f". Verifica del file: chiusura uguale ai dati ufficiali in {VER['close']['uguali']} settimane su {VER['settimane_confrontate']}, "
                    f"`open` uguale all'apertura ufficiale in {VER['open']['uguali']}" if VER else ""))
        c_manual = ", ".join(list(ko) + list(corte)) or "nessuna"

    mdl = lambda h: scelta[h]
    gq = {q: 1 - Dm.loc[(1, "D4"), f"perdita_q{int(100 * q):02d}"] / Dm.loc[(1, "D1"), f"perdita_q{int(100 * q):02d}"] for q in D.Q_REPORT}
    q_best = max(gq, key=gq.get)
    best_gain = [Dm.loc[(h, scelta[h]), "crps_vs_d1"] for h in K.HORIZONS]
    EF = egrp["fascia_origine"].set_index(["evento", "h", "fascia_origine", "modello"])["brier_skill_vs_rif"]
    n_sig12 = int(((em["h"] == 12) & (em["p_holm"] < 0.05)).sum())
    out = f"""# Fase 2 — Previsione probabilistica attorno a M1 e raccolta dei dati esterni

*Report generato da `python -m src.run_phase2` (Parti A e B) e `python -m src.fetch_external` (Parte C). Ogni numero è letto dai file
`reports/fase2/*.csv` e `*.json` prodotti dal codice. Parti A e B: solo `data/processed/vix_weekly_dev.csv`, sotto la guardia che blocca l'holdout.*

## 1. Sintesi

1. **Il modello di densità migliore è la regressione quantilica sulle variabili dell'HAR (D4)**, a tutti gli orizzonti: CRPS più basso di D1 del {pct(Dm.loc[(1, 'D4'), 'crps_vs_d1'], 2)} a h = 1, {pct(Dm.loc[(4, 'D4'), 'crps_vs_d1'], 2)} a h = 4, {pct(Dm.loc[(12, 'D4'), 'crps_vs_d1'], 2)} a h = 12. Dopo Holm la differenza è significativa per: {lst(sig_better)}.
2. **I guadagni sono piccoli.** Il margine del modello migliore sul riferimento D1 (M1 + errori passati incondizionati) va dal {pct(min(best_gain), 1)} al {pct(max(best_gain), 1)} del CRPS: anche per la distribuzione, con il solo VIX, c'è poco da aggiungere.
3. **Condizionare gli errori a fascia e direzione (D2, D3) e le analogie (D6) peggiorano**: meno casi per stimare le code. Significativamente peggiori di D1 dopo Holm: {lst(sig_worse)}.
4. **Tutti gli intervalli coprono meno del nominale.** L'intervallo al 90% copre l'{pct(Dm.loc[(1, 'D1'), 'copertura_90'])} con D1 e l'{pct(Dm.loc[(1, 'D4'), 'copertura_90'])} con D4 a h = 1. D4 vince sul CRPS perché è più stretto e si adatta allo stato, non perché sia meglio calibrato.
5. **D1 è mal calibrato per stato**: a h = 1 il suo intervallo al 90% copre il {pct(fb(1, '<15', 'D1', 'copertura_90'))} partendo da VIX < 15 e il {pct(fb(1, '>=30', 'D1', 'copertura_90'))} partendo da VIX ≥ 30. D4 e D5 riducono questo squilibrio ({pct(fb(1, '<15', 'D4', 'copertura_90'))} e {pct(fb(1, '>=30', 'D4', 'copertura_90'))} per D4).
6. **D5 (GARCH con t asimmetrica, solo h = 1)** migliora D1 del {pct(Dm.loc[(1, 'D5'), 'crps_vs_d1'], 2)} (p Holm {pv(Dm.loc[(1, 'D5'), 'p_holm'])}), con una copertura dell'intervallo al 90% pari al {pct(Dm.loc[(1, 'D5'), 'copertura_90'])}.
7. **Eventi di livello.** Tutte le densità battono la catena di Markov sul Brier: per P(VIX > 25) a h = 1 il guadagno va dal {pct(bss_rng('VIX>25', 1)[0])} all'{pct(bss_rng('VIX>25', 1)[1])}. Dopo Holm ({n_tests_e} confronti) restano significativi: {e_sig_txt}.
8. **Il rialzo oltre +20% non è prevedibile.** Nessun modello batte la frequenza storica (Brier skill tra {pct(rise['brier_skill_vs_rif'].min())} e {pct(rise['brier_skill_vs_rif'].max())}; risoluzione praticamente nulla). Coerente con la Fase 1: i picchi arrivano senza preavviso.
9. **Scheda di previsione.** La regola del CRPS più basso sceglie {', '.join(f'{mdl(h)} per h = {h}' for h in K.HORIZONS)}. Limite emerso: D4 conosce la distribuzione solo tra il 5° e il 95° percentile, quindi assegna probabilità zero agli eventi oltre la coda.
10. **Dati esterni.** {c_syn[0].upper() + c_syn[1:]}.

## 2. Parte A — Modelli di densità

### 2.1 Protocollo e scelte fissate prima della valutazione

- **Centro**: previsione di M1 (tipo HAR) in log; la previsione puntuale in punti VIX è la mediana della distribuzione.
- **Errori passati**: all'origine t si usano solo gli errori delle previsioni fatte alle origini i con i + h ≤ t (esito già noto). Numero minimo: **{AI['min_errori_passati']}**.
- **Stesso insieme di origini per tutti i modelli**: {', '.join(f"h = {h}: {po[str(h)]['n_previsioni']} previsioni, obiettivi dal {po[str(h)]['prima_data_obiettivo']} al {po[str(h)]['ultima_data_obiettivo']}" for h in K.HORIZONS)}. Il campione è più corto di quello della Fase 1 perché servono due anni di errori.
- **Quantili**: griglia comune di 19 livelli (0,05 … 0,95), che contiene i sette richiesti (5, 10, 25, 50, 75, 90, 95). **CRPS** = {AI['crps']}: è un'approssimazione che ignora le code oltre il 5° e il 95° percentile.
- **Regole di ripiego**: D2 usa gli errori della fascia di livello (7 fasce della Fase 1) se sono almeno {AI['regola_casi']}, altrimenti tutti; D3 usa la cella fascia × direzione, poi la fascia, poi tutti, con la stessa soglia.
- **D4**: una regressione quantilica diretta per ogni livello e orizzonte, ristimata a ogni origine. Quantili incrociati: riordinati e contati.
- **D5**: GARCH(1,1) a media nulla sui residui in campione di M1, innovazioni t asimmetrica di Hansen; **limitato a h = 1** (scelta fatta prima: nessuna simulazione multi-passo). Stime: {AI['garch']['stime']}, fallite: {AI['garch']['fallite']}.
- **D6**: valore attuale + variazioni osservate dopo le {D.K_ANALOG} situazioni più simili (non usa M1).
- **Probabilità degli eventi dalle densità**: frequenza empirica per D1, D2, D3, D6, D1b; funzione di ripartizione per D5; per D4 interpolazione lineare sulla griglia, proseguita nelle code con la pendenza dell'ultimo intervallo.
- **E1**: logistica sulle tre variabili dell'HAR, una per evento e orizzonte, con penalità L2 leggera ({n(AI['logit_ridge'], 1)} sui coefficienti standardizzati) per evitare la separazione perfetta.
- **Riferimenti per gli eventi**: catena di Markov della Fase 0 (soglie 20, 25, 30); frequenza storica espandibile per il rialzo oltre +20%.
- **Test**: Diebold-Mariano sul CRPS contro D1 e sul Brier contro il riferimento; Holm su {n_tests_d} confronti (densità) e {n_tests_e} (eventi).

### 2.2 Tabella principale delle densità

{t_main}

(CRPS vs D1: positivo = meglio di D1. Copertura in percentuale con intervallo di confidenza bootstrap a blocchi.)

![CRPS cumulato rispetto a D1](figures/fase2/a_crps_cumulato_vs_d1.png)

Perdita quantilica per ciascun quantile:

{t_q}

- **D4 è il migliore a ogni orizzonte.** A h = 1 il guadagno maggiore è sul quantile {int(100 * q_best)}% (perdita quantilica più bassa del {pct(gq[q_best])} rispetto a D1); sulla mediana è {pct(gq[0.50])}. Quantili incrociati (riordinati): {cross[1]} origini su {po['1']['n_previsioni']} a h = 1, {cross[4]} a h = 4, {cross[12]} a h = 12.
- **D5** segue a h = 1; **D1b** (centro M7) è indistinguibile da D1 ({pct(Dm.loc[(1, 'D1b'), 'crps_vs_d1'], 2)}, {pct(Dm.loc[(4, 'D1b'), 'crps_vs_d1'], 2)}, {pct(Dm.loc[(12, 'D1b'), 'crps_vs_d1'], 2)}): cambiare il centro non serve.
- **D2, D3 e D6 sono peggiori di D1 a tutti gli orizzonti.** A h = 12 la perdita è marcata ({pct(Dm.loc[(12, 'D2'), 'crps_vs_d1'], 1)}, {pct(Dm.loc[(12, 'D3'), 'crps_vs_d1'], 1)}, {pct(Dm.loc[(12, 'D6'), 'crps_vs_d1'], 1)}): suddividere pochi errori in celle rende instabili le code.

### 2.3 Copertura, larghezza e asimmetria degli errori

![Copertura](figures/fase2/a_copertura.png)

Quota di valori reali sopra il limite superiore e sotto il limite inferiore di ogni intervallo (in %; valore nominale per lato: 25, 10 e 5), larghezza media e
errore assoluto medio della mediana:

{t_asym}

- **Sottocopertura generale.** L'intervallo di confidenza della copertura al 90% contiene il 90% solo per: {lst(cov_ok)}. Una spiegazione plausibile: gli errori passati sottostimano la dispersione futura quando arrivano episodi più violenti di quelli già visti (il campione di valutazione comprende il 2008, il 2011 e il 2020).
- **Compromesso tra nitidezza e calibrazione.** D4 ha intervalli più stretti di D1 (all'80%, {n(Dm.loc[(1, 'D4'), 'larghezza_vix_80'])} punti contro {n(Dm.loc[(1, 'D1'), 'larghezza_vix_80'])} a h = 1) e copre meno ({pct(Dm.loc[(1, 'D4'), 'copertura_80'])} contro {pct(Dm.loc[(1, 'D1'), 'copertura_80'])}).
- **Asimmetria.** Per D1 gli sforamenti dell'intervallo al 90% sono quasi simmetrici ({pct(Dm.loc[(1, 'D1'), 'sopra_90'])} sopra, {pct(Dm.loc[(1, 'D1'), 'sotto_90'])} sotto a h = 1). Per D4 sono più frequenti verso il basso ({pct(Dm.loc[(1, 'D4'), 'sopra_90'])} sopra, {pct(Dm.loc[(1, 'D4'), 'sotto_90'])} sotto): il suo limite inferiore è più vicino al centro, e questo gli fa perdere copertura pur migliorando la perdita quantilica. Per D6 gli sforamenti sono i più numerosi su entrambi i lati ({pct(Dm.loc[(1, 'D6'), 'sopra_90'])} e {pct(Dm.loc[(1, 'D6'), 'sotto_90'])}).

### 2.4 Istogramma PIT

Con sette quantili il PIT si legge su otto classi: frequenza con cui il valore reale cade in ciascuna (in %). Frequenze attese: {att_txt}.

![PIT](figures/fase2/a_pit.png)

{t_pit}

Le classi estreme (sotto il 5° e sopra il 95° percentile) sono più popolate del dovuto per tutti i modelli: le code previste sono troppo corte.

### 2.5 Eventi

Brier score (più basso è meglio; RIF = riferimento):

{t_brier}

Brier skill rispetto al riferimento, in % (positivo = meglio; * = significativo al 5% dopo Holm):

{t_bss}

Scomposizione del Brier (h = 1 e 4; affidabilità: più bassa è meglio; risoluzione: più alta è meglio):

{t_dec}

![Calibrazione degli eventi](figures/fase2/a_calibrazione_eventi.png)

- **Soglie di livello.** Le probabilità ricavate dalle densità sono migliori di quelle della catena di Markov a h = 1 e h = 4 per tutte e tre le soglie. Il guadagno viene dalla risoluzione: una densità continua sa quanto il VIX è vicino alla soglia, la catena solo in quale fascia si trova.
- **Dopo Holm** restano significativi: {e_sig_txt}. Gli altri guadagni hanno lo stesso segno ma non superano la correzione su {n_tests_e} confronti.
- **E1 (logistica diretta) non fa meglio delle densità**: per P(VIX > 25) a h = 1 ha Brier {n(Em.loc[('VIX>25', 1, 'E1'), 'brier'], 4)} contro {n(Em.loc[('VIX>25', 1, 'D4'), 'brier'], 4)} di D4 e {n(Em.loc[('VIX>25', 1, 'D1'), 'brier'], 4)} di D1. Non serve un modello separato per gli eventi.
- **Rialzo oltre +20%.** Frequenza osservata {pct(rise.loc['RIF', 'frequenza_evento'])}. Nessun modello migliora la frequenza storica; significativamente peggiori dopo Holm: {e_neg_txt}. La risoluzione è prossima a zero per tutti: l'evento non è prevedibile con il solo VIX.
- **h = 12.** Confronti significativi dopo Holm: {n_sig12}. A questo orizzonte nessun modello si distingue in modo dimostrabile dalla catena di Markov.

### 2.6 Per sottoperiodo, fascia di partenza e sottoinsiemi a posteriori

Per ogni suddivisione: CRPS rispetto a D1 in % (positivo = meglio) e copertura dell'intervallo al 90% in %. h = 1 e 4; le tabelle complete, compreso h = 12 e
gli eventi, sono in `reports/fase2/a_densita_*.csv` e `a_eventi_*.csv`. Test non corretti per confronti multipli.

**Sottoperiodo (data dell'obiettivo)** — CRPS vs D1:

{gpivot(grp['sottoperiodo'], 'sottoperiodo', 'crps_vs_d1')}

Copertura al 90%:

{gpivot(grp['sottoperiodo'], 'sottoperiodo', 'copertura_90')}

**Fascia di partenza** — CRPS vs D1:

{gpivot(grp['fascia_origine'], 'fascia_origine', 'crps_vs_d1', order=list(C.BAND_LABELS))}

Copertura al 90%:

{gpivot(grp['fascia_origine'], 'fascia_origine', 'copertura_90', order=list(C.BAND_LABELS))}

**Dopo un picco** (etichetta assegnata a posteriori) — CRPS vs D1 e copertura al 90%:

{gpivot(sub_p, 'dopo_picco', 'crps_vs_d1')}

{gpivot(sub_p, 'dopo_picco', 'copertura_90')}

**Dentro un periodo calmo** (etichetta assegnata a posteriori) — CRPS vs D1 e copertura al 90%:

{gpivot(sub_c, 'periodo_calmo', 'crps_vs_d1')}

{gpivot(sub_c, 'periodo_calmo', 'copertura_90')}

P(VIX > 25): Brier skill rispetto alla catena di Markov per fascia di partenza (%):

{t_e_fascia}

- **D1 è calibrato in media ma non per stato.** Copre troppo quando il VIX è basso e troppo poco quando è alto: a h = 1, {pct(fb(1, '<15', 'D1', 'copertura_90'))} sotto 15 e {pct(fb(1, '>=30', 'D1', 'copertura_90'))} da 30 in su. Gli errori in log sono più ampi a livelli alti, e una distribuzione unica non può saperlo.
- **D4 guadagna di più dove D1 è più sbagliato**: CRPS migliore del {pct(fb(1, '>=30', 'D4', 'crps_vs_d1'))} partendo da VIX ≥ 30 e del {pct(fb(1, '25-30', 'D4', 'crps_vs_d1'))} da 25-30 (h = 1).
- **Periodi calmi.** Dentro un periodo calmo D1 copre il {pct(G['periodo_calmo'].loc[(1, True, 'D1'), 'copertura_90'])} (intervalli troppo larghi), D4 l'{pct(G['periodo_calmo'].loc[(1, True, 'D4'), 'copertura_90'])}; fuori, {pct(G['periodo_calmo'].loc[(1, False, 'D1'), 'copertura_90'])} e {pct(G['periodo_calmo'].loc[(1, False, 'D4'), 'copertura_90'])}.
- **Dopo un picco** nessun modello si distingue nettamente; D6 perde di più ({pct(G['dopo_picco'].loc[(1, True, 'D6'), 'crps_vs_d1'])} a h = 1).
- **Eventi per fascia.** Per P(VIX > 25) il guadagno sulla catena di Markov si vede solo partendo da 20 in su; sotto 20 l'evento è quasi impossibile e non c'è nulla da guadagnare. Nella fascia 20-25, dove la Fase 0 aveva trovato la catena non informativa, a h = 1 il Brier migliora del {pct(EF.loc[('VIX>25', 1, '20-25', 'D1')])} con D1 e del {pct(EF.loc[('VIX>25', 1, '20-25', 'D4')])} con D4.

### 2.7 Cosa implica per il modello di previsione

1. **La distribuzione va fatta dipendere dallo stato in modo continuo** (regressione quantilica o varianza condizionata), non per celle: D4 e D5 migliorano D1, D2 e D3 lo peggiorano.
2. **Gli intervalli vanno letti come un po' ottimistici**: la copertura reale è alcuni punti sotto la nominale per tutti i modelli. Una correzione di calibrazione (per esempio allargare gli intervalli in base alla copertura passata) è un'estensione naturale, da decidere prima di usarla.
3. **Un solo modello non basta per tutto.** D4 è il migliore sul CRPS ma non descrive le code oltre il 5° e il 95° percentile; D5 le descrive ma esiste solo a h = 1; le densità empiriche (D1) danno probabilità non nulle anche per gli eventi lontani dalla mediana.
4. **Gli eventi si ricavano dalla densità**: non serve un modello dedicato (E1 non aggiunge nulla).
5. **Il rialzo oltre +20% va comunicato come frequenza di base** (circa {pct(rise.loc['RIF', 'frequenza_evento'], 0)} delle settimane), non come previsione condizionata.

## 3. Parte B — Scheda di previsione

La funzione `src.fase2.scheda.scheda(df, t, scelta)` produce la scheda all'origine t usando solo i dati fino a t. Regola fissata prima: per ogni orizzonte
il modello con il CRPS medio più basso tra D1-D6 ({', '.join(f'h = {h}: {mdl(h)}' for h in K.HORIZONS)}). D1b è escluso perché è solo un controllo.

**Esempio: scheda al {SCH['data_origine']}** (ultima data del campione di sviluppo; VIX all'origine: {n(SCH['vix_origine'])}). Valori in punti VIX.
Nessun confronto con il valore realizzato: le settimane successive sono nell'holdout.

{t_scheda}

- La mediana è la previsione puntuale in punti VIX; gli intervalli sono asimmetrici verso l'alto, come ci si aspetta partendo da un VIX basso.
- **Limite della regola.** {('In questa scheda ' + str(len(zero_p)) + ' probabilità valgono esattamente zero (' + ', '.join(f'{e} a h = {h}' for h, e in zero_p) + '): ') if zero_p else ''}D4 conosce solo i quantili tra il 5° e il 95° e, oltre, prosegue in linea retta fino a zero. Una probabilità pari a zero per un evento possibile non è una previsione accettabile. La regola era fissata e non l'ho cambiata; nella sezione 6 propongo come correggerla.

**Cosa implica per il modello di previsione.** La scheda è pronta nella forma (mediana, tre intervalli, quattro probabilità), ma le probabilità degli eventi
lontani dalla mediana vanno prese da una densità con code definite. Va deciso prima dell'apertura dell'holdout.

## 4. Parte C — Raccolta dei dati esterni

{part_c}
**Cosa implica per il modello di previsione.** Nulla, per ora: i dati esterni non sono stati usati né analizzati. Per la Fase 3 contano tre vincoli
pratici emersi dall'inventario: le serie iniziano in date diverse (alcune ben dopo il 2000), quindi i modelli che le usano avranno campioni di valutazione
più corti; alcune serie hanno un ritardo di pubblicazione o vengono riviste; la colonna `open` del nostro file resta esclusa.

## 5. Problemi e ambiguità

1. **CRPS approssimato** sulla griglia 5–95%: non misura la qualità delle code estreme, che è proprio dove i modelli differiscono di più. La classifica potrebbe cambiare con un CRPS esatto.
2. **D4 e le code.** La regola di estrapolazione lineare oltre la griglia produce probabilità zero per eventi lontani. Era fissata prima e l'ho lasciata; incide sugli eventi (Brier di D4) e sulla scheda.
3. **D4, quantili incrociati**: {cross[1]}, {cross[4]} e {cross[12]} origini (h = 1, 4, 12), riordinati. La regressione quantilica è stimata con minimi quadrati ripesati (verificata contro statsmodels in un test) e ristimata a ogni origine.
4. **D5 solo a h = 1** e stimato sui residui in campione di M1, non sugli errori fuori campione usati da D1–D3: le due famiglie non hanno esattamente la stessa base informativa (entrambe usano solo dati fino all'origine).
5. **Regole di ripiego di D2 e D3** (sotto 20 casi) e **penalità di E1**: scelte mie, fissate prima della valutazione.
6. **Fasce**: D2 e D3 usano le 7 fasce della Fase 1; le tabelle per fascia di partenza usano le 5 fasce della Fase 0, per confrontabilità con i report precedenti.
7. **Campione di valutazione più corto** della Fase 1 (dal {po['1']['prima_data_obiettivo']}): i numeri di M1 e della catena di Markov qui non coincidono con quelli dei report precedenti.
8. **Copertura**: gli intervalli di confidenza sono bootstrap a blocchi di {K.BLOCK} settimane; a h = 4 e 12 gli errori sono sovrapposti.
9. **Holm** è applicato separatamente alle due tabelle principali ({n_tests_d} e {n_tests_e} confronti). Le tabelle per sottogruppo non sono corrette.
10. **Etichette a posteriori** («dopo un picco», «periodo calmo»): come in Fase 1, servono solo a suddividere la valutazione.
11. **Scelta del modello per la scheda**: fatta sul CRPS dell'intero walk-forward di sviluppo; all'ultima data di sviluppo tutti quegli esiti erano noti. Per date precedenti la scelta andrebbe rifatta con i soli esiti noti a quella data.
12. **Evento «rialzo oltre +20%»**: definito sulla chiusura settimanale (chiusura successiva / chiusura attuale − 1 > 0,20).
13. **Dati esterni: scarico.** Lo scarico usa `curl` di sistema, perché da Python la verifica dei certificati falliva in questo ambiente; la verifica TLS resta attiva. Durante lo sviluppo FRED non ha risposto ai primi tentativi; l'inventario riporta l'esito dell'ultimo scarico. Serie da procurare a mano: {c_manual}.
14. **Dati esterni: ritardi.** I ritardi di disponibilità usati nell'allineamento (0 giorni per prezzi e rendimenti, 1 per i differenziali di credito, 5 per l'NFCI) sono mie stime prudenti dei calendari di pubblicazione, da confermare.
15. **NFCI.** {nfci_note}Il file contiene i valori rivisti a oggi, non quelli noti in tempo reale: usarlo in un walk-forward introduce un look-ahead da revisione che il ritardo di 5 giorni non elimina.
16. **Future sul VIX.** Serie continue costruite da me a partire dai singoli contratti: due scadenze mensili più vicine, prezzo di regolamento, prezzi d'archivio divisi per 10 prima del 26/03/2007, scadenza dedotta dall'ultima data per i file 2004-2013. Per i contratti presenti in entrambe le fonti Cboe (archivio 2004-2013 ed elenco dal 2013) il file recente ha la precedenza e l'archivio colma i buchi. Come controllo di coerenza della scala ho verificato, sul solo periodo di sviluppo, che il rapporto tra prima scadenza e VIX resti vicino a 1 in ogni anno: è l'unico confronto con il VIX fatto in questa fase e riguarda la costruzione del dato, non la sua capacità previsiva. Regole da verificare in Fase 3 prima dell'uso.
17. **Dati esterni non pubblicati nel repository** (dati di terzi): chi clona il repository deve rieseguire lo scarico e può ottenere valori leggermente diversi per le serie soggette a revisione.
18. **Verifica del file**: fatta sul solo periodo di sviluppo; le settimane di holdout del file grezzo non sono state confrontate.

## 6. Proposte per la Fase 3 (dati esterni)

**Prima della Fase 3, due correzioni alla scheda da decidere:**
(a) probabilità degli eventi prese da una densità con code definite (propongo D1 per gli eventi e D4 per mediana e intervalli, oppure D5 a h = 1);
(b) eventuale ricalibrazione della larghezza degli intervalli sulla copertura passata.

**Protocollo proposto.** Stesso walk-forward; per ogni modello con dati esterni, valutazione sul campione in cui tutte le sue variabili esistono,
confrontato con M1 (punto) e con il migliore tra D1 e D4 (densità) **ricalcolati sulle stesse origini**; variabili allineate con i ritardi dell'inventario;
elenco chiuso; Holm; holdout aperto una sola volta alla fine, per il solo modello scelto.

**Elenco chiuso proposto** (ogni modello = M1 più un blocco di variabili, stessa regressione diretta per h = 1, 4, 12):

| sigla | variabili aggiunte a M1 | motivo | inizio dei dati |
|---|---|---|---|
| X1 | pendenza della struttura a termine degli indici: VIX3M / VIX − 1 | misura diretta di quanto il mercato si aspetta che il VIX rientri | 2009 |
| X2 | future: VX1 / VIX − 1 e VX2 / VX1 − 1 | stessa informazione, storia più lunga, prezzi negoziati | 2004 |
| X3 | VVIX (livello in log) | incertezza sul VIX stesso; candidato per la dispersione più che per il centro | 2006 |
| X4 | S&P 500: rendimento a 1 e 4 settimane, volatilità realizzata a 4 settimane, distanza dal massimo a 52 settimane | il VIX reagisce ai ribassi azionari; la volatilità realizzata àncora il livello | 2000 |
| X5 | tassi e credito: pendenza 10 anni − 2 anni, variazione a 4 settimane dei differenziali investment grade e high yield | condizioni macro-finanziarie lente | 2000 per i tassi; i differenziali solo se arriva la storia completa (sezione 4.1), altrimenti X5 usa i soli tassi |
| X6 | tutte le variabili di X1–X5 con penalità ridge, intensità scelta solo dentro la finestra di stima | verifica se i blocchi si sommano senza sovradattare | campione comune |
| X7 | media semplice delle previsioni di X1–X5 | controllo robusto, come M7 | campione comune |
| QX | regressione quantilica (come D4) con il blocco risultato migliore tra X1–X5 | porta l'informazione esterna nella distribuzione | come il blocco |
| EX | probabilità degli eventi ricavate da QX | coerenza con la Fase 2 | come il blocco |

Non propongo l'NFCI nell'elenco, per il problema delle revisioni (punto 15): entrerebbe solo con dati in tempo reale. VIX9D e VIX6M restano fuori
dall'elenco principale (storia corta il primo, ridondante con VIX3M il secondo) ma sono disponibili.

**Decisioni richieste prima della Fase 3:** (i) le due correzioni alla scheda; (ii) se pubblicare `data/external/` nel repository; (iii) conferma dei ritardi
di disponibilità; (iv) NFCI: escluso, oppure ammesso con l'avvertenza sulle revisioni; (v) elenco dei modelli: approvato, ridotto o modificato;
(vi) come trattare le serie non scaricate, se ce ne sono (sezione 4.1).

## 7. Riproducibilità

```
python -m src.run_phase0      # Fase 0 (invariata)
python -m src.run_phase1      # Fase 1 (invariata)
python -m src.run_phase2      # Fase 2, Parti A e B e questo report (circa 5 minuti)
python -m src.fetch_external  # Fase 2, Parte C: scarico dei dati esterni (richiede rete e curl; non deterministico)
python -m pytest              # test
```

Versioni e seed: `reports/fase2/run_manifest.json`. Quantili e CRPS per ogni origine: `reports/fase2/a_quantili.csv`; probabilità degli eventi:
`reports/fase2/a_eventi_previsioni.csv`; inventario dei dati esterni: `reports/fase2/c_inventario.json`.
"""
    (C.REPORTS_DIR / "fase2_report.md").write_text(out)


if __name__ == "__main__":
    run()
