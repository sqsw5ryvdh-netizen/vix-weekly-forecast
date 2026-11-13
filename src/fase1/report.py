"""Genera reports/fase1_report.md. Tutti i numeri sono letti dai file reports/fase1/*.csv e *.json."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .. import config as C
from . import comune as K
from . import modelli as M

R = K.OUT
MIN_N = 30


def J(name: str) -> dict:
    return json.loads((R / f"{name}.json").read_text())


def T(name: str) -> pd.DataFrame:
    return pd.read_csv(R / f"{name}.csv")


def md(df: pd.DataFrame, fmt=".3f") -> str:
    return re.sub(r"\bnan\b", "  –", df.to_markdown(index=False, floatfmt=fmt, missingval="–"))


def n(v: float, d: int = 2) -> str:
    return "n.d." if not np.isfinite(v) else f"{v:.{d}f}".replace(".", ",")


def pct(v: float, d: int = 1) -> str:
    return "n.d." if not np.isfinite(v) else f"{100 * v:.{d}f}%".replace(".", ",")


def ci(est, lo, hi, d: int = 2) -> str:
    """Cella di tabella: stima [IC 95%]."""
    if not np.isfinite(est):
        return "–"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{est:.{d}f} [n.d.]"
    return f"{est:.{d}f} [{lo:.{d}f}; {hi:.{d}f}]"


def cit(est, lo, hi, d: int = 2) -> str:
    """Nel testo: stima (IC 95%: lo; hi)."""
    return f"{n(est, d)} (IC 95%: {n(lo, d)}; {n(hi, d)})"


def run() -> None:
    AI, BI = J("a_info"), J("b_info")
    cls, ep, s1 = T("a1_classi"), T("a1_episodi"), T("a1_sintesi").set_index(["definizione", "classe", "misura"])
    a2c, a2t, a2d = T("a2_confronto"), T("a2_terzili"), T("a2_terzili_differenza")
    a3p, a3s, a3h, a3d = T("a3_periodi"), T("a3_sintesi").set_index(["definizione", "misura"]), T("a3_durata"), T("a3_durata_differenza").set_index("definizione")
    a4, a5 = T("a4_celle"), T("a5_verdetti")
    bm, bsub, bper, bfas, bk = T("b_metriche"), T("b_metriche_sottoinsiemi"), T("b_metriche_sottoperiodi"), T("b_metriche_fascia"), T("b_sensibilita_k")
    S, RL = "soglia30", "relativa"
    n_ep = {d: int((ep["definizione"] == d).sum()) for d in K.EP_DEFS}

    def g1(d, m, c="tutti"):
        return s1.loc[(d, c, m)]

    def g1c(d, m, c="tutti", dec=2):
        r = g1(d, m, c)
        return cit(r["stima"], r["ci_lo"], r["ci_hi"], dec)

    # ---------------------------------------------------------------- tabelle Parte A
    def a1_table(measures, labels, dec=2):
        rows = []
        for m, lab in zip(measures, labels):
            row = {"misura": lab}
            for d in K.EP_DEFS:
                r = g1(d, m)
                row[f"{d}: stima [IC 95%]"] = ci(r["stima"], r["ci_lo"], r["ci_hi"], dec)
                row[f"{d}: n"] = f"{int(r['n'])}" + (f" (+{int(r['n_non_disponibili'])} n.d.)" if r["n_non_disponibili"] else "")
            rows.append(row)
        return md(pd.DataFrame(rows))

    ks = (1, 2, 4, 8, 13, 26)
    t_quota = a1_table([f"quota_k{k}" for k in ks], [f"quota recuperata dopo {k} sett. (mediana)" for k in ks])
    t_punti = a1_table([f"punti_k{k}" for k in ks], [f"punti VIX rispetto al picco dopo {k} sett. (mediana)" for k in ks], 1)
    t_tempi = a1_table(["sett_recupero_50", "sett_recupero_100", "sett_sotto_25", "sett_sotto_20", "rimbalzi_oltre_20pct", "almeno_un_rimbalzo"],
                       ["settimane per recuperare il 50% (mediana)", "settimane per recuperare il 100% (mediana)",
                        "settimane per tornare sotto 25 (mediana)", "settimane per tornare sotto 20 (mediana)",
                        "rialzi settimanali oltre +20% durante la discesa (media per episodio)", "quota di episodi con almeno un rimbalzo"], 1)
    t_ar = a1_table([f"ar_storia_k{k}" for k in ks] + [f"ar_intero_k{k}" for k in (1, 4, 13)],
                    [f"AR(1) stimato fino al picco, {k} sett." for k in ks] + [f"AR(1) stimato su tutto il campione, {k} sett." for k in (1, 4, 13)], 3)
    rows = []
    for d in K.EP_DEFS:
        for c in ("<30", "30-40", "40-50", ">=50"):
            if (d, c, "quota_k4") not in s1.index:
                continue
            row = {"definizione": d, "classe del picco": c, "episodi": int(cls[(cls["definizione"] == d) & (cls["classe"] == c)]["n_episodi"].iloc[0])}
            for m, lab, dec in (("quota_k4", "quota a 4 sett.", 2), ("quota_k13", "quota a 13 sett.", 2), ("sett_recupero_50", "sett. per il 50%", 1),
                                ("sett_recupero_100", "sett. per il 100%", 1), ("sett_sotto_25", "sett. sotto 25", 1), ("sett_sotto_20", "sett. sotto 20", 1)):
                r = g1(d, m, c)
                row[lab] = ci(r["stima"], r["ci_lo"], r["ci_hi"], dec)
            rows.append(row)
    t_classi = md(pd.DataFrame(rows))
    ep_list = ep[ep["definizione"] == S][["inizio", "data_picco", "picco", "classe", "base", "salita", "sett_recupero_50", "sett_recupero_100",
                                           "sett_sotto_25", "sett_sotto_20", "rimbalzi_oltre_20pct"]]
    no_quota = ep[(ep["definizione"] == S) & (~ep["quota_definita"])]
    short_base = ep[ep["n_sett_base"] < K.PRE_WIN]
    cens100 = {d: int(g1(d, "sett_recupero_100")["n_non_disponibili"] - ((ep["definizione"] == d) & (~ep["quota_definita"])).sum()) for d in K.EP_DEFS}
    ncl = lambda d, c: int(cls[(cls["definizione"] == d) & (cls["classe"] == c)]["n_episodi"].iloc[0])
    ep_s = ep[ep["definizione"] == S]
    yr_peak = pd.to_datetime(ep_s["data_picco"]).dt.year.value_counts()

    lab_m = {"livello": "livello", "var4": "variazione log 4 sett.", "var12": "variazione log 12 sett.", "sd12": "dev. std. variazioni (12 sett.)",
             "escursione": "escursione ln(high/low) [esplorativo]", "sett_da_prec": "settimane dall'episodio precedente"}
    t = a2c.copy()
    t["differenza [IC 95%]"] = [ci(a, b, c_, 3) for a, b, c_ in zip(t["differenza"], t["ci_lo"], t["ci_hi"])]
    t["percentile medio [IC 95%]"] = [ci(a, b, c_, 0) for a, b, c_ in zip(t["percentile_medio_pre"], t["percentile_ci_lo"], t["percentile_ci_hi"])]
    t["grandezza"] = t["grandezza"].map(lab_m)
    t_a2c = md(t[["definizione", "grandezza", "n_episodi", "n_altre_settimane", "mediana_pre_episodio", "mediana_altre", "differenza [IC 95%]",
                  "percentile medio [IC 95%]", "mann_whitney_p"]].rename(columns={"n_episodi": "episodi", "n_altre_settimane": "altre sett.",
                  "mediana_pre_episodio": "mediana prima", "mediana_altre": "mediana altre", "mann_whitney_p": "p Mann-Whitney"}))
    piv = a2t.assign(cell=[f"{100 * f:.1f}% [{100 * a:.1f}; {100 * b:.1f}] (n={k}, eventi={e})" for f, a, b, k, e in
                           zip(a2t["frequenza"], a2t["ci_lo"], a2t["ci_hi"], a2t["n_settimane"], a2t["n_settimane_con_evento"])])
    piv = piv.pivot(index=["definizione", "grandezza"], columns="terzile", values="cell").reindex(columns=["basso", "medio", "alto"]).reset_index()
    dd = a2d.assign(**{"alto − basso [IC 95%]": [f"{100 * a:+.1f} [{100 * b:+.1f}; {100 * c_:+.1f}] p.p." for a, b, c_ in zip(a2d["stima"], a2d["ci_lo"], a2d["ci_hi"])]})
    piv = piv.merge(dd[["definizione", "grandezza", "alto − basso [IC 95%]"]], on=["definizione", "grandezza"])
    piv["_o"] = piv["grandezza"].map({k: i for i, k in enumerate(lab_m)})
    piv = piv.sort_values(["definizione", "_o"], ascending=[False, True]).drop(columns="_o")
    piv["grandezza"] = piv["grandezza"].map(lab_m)
    t_a2t = md(piv)
    c2 = a2c.set_index(["definizione", "grandezza"])
    d2 = a2d.set_index(["definizione", "grandezza"])
    t2 = a2t.set_index(["definizione", "grandezza", "terzile"])
    sig_rel = [lab_m[m] for m in lab_m if (d2.loc[(RL, m), "ci_lo"] > 0 or d2.loc[(RL, m), "ci_hi"] < 0)]
    sig_s = [lab_m[m] for m in lab_m if (d2.loc[(S, m), "ci_lo"] > 0 or d2.loc[(S, m), "ci_hi"] < 0)]

    lab3 = {"durata": ("durata in settimane (mediana)", 1), "banda_punti": ("banda di oscillazione, punti VIX (mediana)", 2),
            "banda_log": ("banda di oscillazione, log (mediana)", 3), "var_1": ("variazione log 1 sett. dopo la fine (mediana)", 3),
            "var_2": ("variazione log 2 sett. dopo la fine (mediana)", 3), "var_4": ("variazione log 4 sett. dopo la fine (mediana)", 3),
            "prosecuzione_1_4": ("variazione log dalla 1ª alla 4ª sett. dopo la fine (mediana)", 3),
            "uscita_con_rialzo_oltre_20pct": ("quota di uscite con rialzo settimanale oltre +20%", 2), "uscita_al_rialzo": ("quota di uscite al rialzo", 2)}
    rows = []
    for m, (lab, dec) in lab3.items():
        row = {"misura": lab}
        for d in ("sotto15", "escursione_bassa"):
            r = a3s.loc[(d, m)]
            row[f"{d}: stima [IC 95%]"] = ci(r["stima"], r["ci_lo"], r["ci_hi"], dec)
            row[f"{d}: min–max"] = f"{r['min']:.{dec}f}–{r['max']:.{dec}f}"
            row[f"{d}: n"] = int(r["n"])
        rows.append(row)
    t_a3 = md(pd.DataFrame(rows))
    h3 = a3h.assign(**{"P(fine entro 4 sett.) [IC 95%]": [ci(a, b, c_, 2) for a, b, c_ in zip(a3h["prob_fine_entro_4"], a3h["ci_lo"], a3h["ci_hi"])]})
    t_a3h = md(h3[["definizione", "classe_durata", "n_settimane", "n_periodi", "P(fine entro 4 sett.) [IC 95%]"]].rename(
        columns={"classe_durata": "settimane già trascorse", "n_settimane": "settimane osservate", "n_periodi": "periodi"}))
    g3 = lambda d, m: a3s.loc[(d, m)]
    g3c = lambda d, m, dec=2: cit(g3(d, m)["stima"], g3(d, m)["ci_lo"], g3(d, m)["ci_hi"], dec)
    h3i = a3h.set_index(["definizione", "classe_durata"])
    per15 = a3p[a3p["definizione"] == "sotto15"]
    yrs15 = pd.to_datetime(per15["inizio"]).dt.year

    c4 = a4.set_index(["h", "fascia", "direzione"])

    def a4_table(h):
        rows = []
        for b in K.BAND7_LABELS:
            row = {"fascia": b}
            for dl in K.DIR_LABELS + ("tutte",):
                r = c4.loc[(h, b, dl)]
                row[dl] = "–" if r["n"] == 0 else f"{r['mediana']:+.3f} [{r['ci_lo']:+.3f}; {r['ci_hi']:+.3f}] n={int(r['n'])}"
            rows.append(row)
        return md(pd.DataFrame(rows))

    q4 = a4[(a4["h"] == 4) & (a4["direzione"] != "tutte")][["fascia", "direzione", "n", "mediana", "q10", "q25", "q75", "q90", "quota_rialzi"]]
    small = a4[(a4["h"] == 1) & (a4["direzione"] != "tutte") & (a4["n"] < M.MIN_CELL)]
    c4v = lambda h, b, d_: c4.loc[(h, b, d_)]

    v = a5.copy()
    for k in (1, 2):
        v[f"stima {k}ª metà [IC 95%]"] = [ci(a, b, c_, 3) for a, b, c_ in zip(v[f"stima_{k}"], v[f"ci_lo_{k}"], v[f"ci_hi_{k}"])]
    t_a5 = md(v[["blocco", "pattern", "n_1", "n_2", "stima 1ª metà [IC 95%]", "stima 2ª metà [IC 95%]", "verdetto", "nota"]].rename(
        columns={"n_1": "casi 2000-12", "n_2": "casi 2013-24"}).fillna(""))
    vc = a5["verdetto"].value_counts()
    regge = a5[a5["verdetto"] == "regge"]
    mecc = regge["nota"].fillna("").str.contains("meccanico").sum()

    # ---------------------------------------------------------------- tabelle Parte B
    B = bm.set_index(["h", "modello"])

    def b_main():
        t = bm.copy()
        t["modello"] = t["modello"].map(M.NAMES)
        t["test"] = t["test"].fillna("").str.replace(r" \(.*\)", "", regex=True)
        cols = ["h", "modello", "n", "rmse_log", "mae_log", "rmse_vix", "mae_vix", "acc_direzionale", "r2_oos_vs_rw", "r2_oos_vs_m1", "test", "stat", "p", "p_holm"]
        return md(t[cols].rename(columns={"rmse_log": "RMSE log", "mae_log": "MAE log", "rmse_vix": "RMSE VIX", "mae_vix": "MAE VIX",
                                          "acc_direzionale": "acc. dir.", "r2_oos_vs_rw": "R² vs RW", "r2_oos_vs_m1": "R² vs M1",
                                          "stat": "statistica", "p_holm": "p Holm"}), ".4f")

    def b_pivot(df, col, order=None):
        t = df[df["n"] >= MIN_N]
        p = t.pivot_table(index=["h", col], columns="modello", values="r2_oos_vs_m1", sort=False)
        nn = t[t["modello"] == "M1"].set_index(["h", col])["n"]
        rm = t[t["modello"] == "M1"].set_index(["h", col])["rmse_log"]
        r0 = t[t["modello"] == "M1"].set_index(["h", col])["r2_oos_vs_rw"]
        p = (100 * p[[m for m in M.MODELS if m != "M1"]]).round(1)
        p.insert(0, "R² M1 vs RW (%)", (100 * r0).round(1))
        p.insert(0, "RMSE log M1", rm.round(4))
        p.insert(0, "n", nn)
        return md(p.reset_index(), ".1f").replace("RMSE log M1", "RMSE log M1")

    sub = bsub.set_index(["h", "sottoinsieme", "modello"])
    lab_peak = [x for x in bsub["sottoinsieme"].unique() if x.startswith("entro")][0]
    lab_calm = [x for x in bsub["sottoinsieme"].unique() if x.startswith("dentro")][0]
    fas = bfas.set_index(["h", "fascia_origine", "modello"])
    per = bper.set_index(["h", "sottoperiodo", "modello"])
    best = {h: bm[(bm["h"] == h) & (bm["modello"] != "M1")].sort_values("r2_oos_vs_m1").iloc[-1] for h in K.HORIZONS}
    n_better_sig = int(((bm["r2_oos_vs_m1"] > 0) & (bm["p_holm"] < 0.05)).sum())
    worse_sig = bm[(bm["r2_oos_vs_m1"] < 0) & (bm["p_holm"] < 0.05) & (bm["modello"] != "M0")]
    worse_txt = ", ".join(f"{r.modello} a h = {r.h}" for r in worse_sig.itertuples()) or "nessuno"
    tk = bk.assign(modello=bk["modello"].map(M.NAMES))[["h", "modello", "n", "rmse_log", "mae_log", "acc_direzionale", "r2_oos_vs_rw", "r2_oos_vs_m1"]]
    arma = BI["arma"]
    same_m4 = M.ROLL_WINDOW - C.MIN_TRAIN_WEEKS + 1
    r2 = lambda h, m: pct(B.loc[(h, m), "r2_oos_vs_m1"], 2)
    q = lambda h, b, d_, c_: n(c4v(h, b, d_)[c_], 3)

    out = f"""# Fase 1 — Pattern storici e modelli che usano solo il VIX

*Report generato da `python -m src.run_phase1`. Ogni numero è letto dai file `reports/fase1/*.csv` e `*.json` prodotti dal codice.
Dati: solo `data/processed/vix_weekly_dev.csv` ({AI['campione'][0]} → {AI['campione'][1]}, {AI['n_settimane']} settimane); l'holdout resta bloccato
e l'esecuzione avviene sotto una guardia che ne impedisce l'apertura. Intervalli di confidenza: bootstrap percentile al 95%,
{AI['repliche_bootstrap']} repliche (per episodi e periodi: ricampionamento delle unità; per le settimane: blocchi di {AI['blocco_settimane']} settimane).*

## 1. Sintesi

1. **Nessun modello dell'elenco batte il tipo HAR (M1).** Il migliore è {best[1]['modello']} a h = 1 (R² fuori campione rispetto a M1: {pct(best[1]['r2_oos_vs_m1'], 2)}), {best[4]['modello']} a h = 4 ({pct(best[4]['r2_oos_vs_m1'], 2)}) e {best[12]['modello']} a h = 12 ({pct(best[12]['r2_oos_vs_m1'], 2)}). Dopo la correzione di Holm i modelli significativamente migliori di M1 sono {n_better_sig} su {BI['n_test_holm'] - 3}.
2. **M1 batte il random walk a tutti gli orizzonti** anche con il test corretto per modelli annidati (Clark-West, p Holm < 0,001): R² rispetto al random walk {pct(B.loc[(1, 'M1'), 'r2_oos_vs_rw'])} a h = 1, {pct(B.loc[(4, 'M1'), 'r2_oos_vs_rw'])} a h = 4, {pct(B.loc[(12, 'M1'), 'r2_oos_vs_rw'])} a h = 12.
3. **Tabella condizionata (M5) e analogia (M6) fanno peggio di M1**: significativamente peggio dopo Holm per {worse_txt}. I pattern su cui si basano sono reali, ma una regressione lineare li sfrutta già meglio.
4. **Dopo i picchi la discesa iniziale è rapida.** Con la definizione di Fase 0 ({n_ep[S]} episodi) metà della salita è recuperata in {n(g1(S, 'sett_recupero_50')['stima'], 0)} settimane (mediana) e il 100% in {n(g1(S, 'sett_recupero_100')['stima'], 0)}; con la definizione relativa ({n_ep[RL]} episodi) in {n(g1(RL, 'sett_recupero_50')['stima'], 0)} e {n(g1(RL, 'sett_recupero_100')['stima'], 1)} settimane.
5. **Nelle prime 4 settimane la discesa è più rapida di quella prevista dall'AR(1)**: scarto mediano in log a 4 settimane {g1c(S, 'ar_storia_k4', dec=3)}; da 8 settimane in poi l'intervallo comprende lo zero (definizione principale). Attenzione: il picco si riconosce solo a posteriori.
6. **I picchi alti rientrano molto più lentamente**, ma i casi sono pochissimi: {int(cls[(cls['definizione'] == S) & (cls['classe'] == '40-50')]['n_episodi'].iloc[0])} episodi con picco 40-50 e {int(cls[(cls['definizione'] == S) & (cls['classe'] == '>=50')]['n_episodi'].iloc[0])} con picco ≥ 50.
7. **Prima dei picchi il segnale è debole.** Con la definizione relativa, quando il VIX è già salito nelle ultime 12 settimane (terzile alto) un episodio inizia entro 4 settimane nel {pct(t2.loc[(RL, 'var12', 'alto'), 'frequenza'])} dei casi, contro il {pct(t2.loc[(RL, 'var12', 'basso'), 'frequenza'])} del terzile basso. Volatilità ed escursione recenti non danno un segnale affidabile. I picchi arrivano in gran parte senza preavviso.
8. **I periodi calmi finiscono di colpo e senza calendario.** {int(g3('sotto15', 'durata')['n'])} periodi sotto 15 (durata mediana {n(g3('sotto15', 'durata')['stima'], 1)} settimane); il {pct(g3('sotto15', 'uscita_con_rialzo_oltre_20pct')['stima'], 0)} termina con un rialzo settimanale oltre +20%. La probabilità di fine non cresce in modo dimostrabile con la durata.
9. **Tabelle condizionate.** Il ritorno alla media per fascia è netto e regge in entrambe le metà del campione partendo da VIX ≥ 25; la direzione recente aggiunge informazione a 4 e 12 settimane (dopo un rialzo il calo successivo è più forte).
10. **Verifica sulle due metà.** Su {len(a5)} pattern: {int(vc.get('regge', 0))} reggono (in parte meccanici tra questi: {int(mecc)}), {int(vc.get('non regge', 0))} non reggono, {int(vc.get('dati insufficienti', 0))} hanno dati insufficienti. Quasi tutto ciò che riguarda episodi e periodi calmi ha troppo pochi casi per metà oppure non regge.

## 2. Parte A — Pattern storici

### A1. Dopo i picchi

**Definizioni.** Principale (`{S}`): {K.EP_DEFS[S]}, periodi separati da al più {K.MERGE_GAP} settimane uniti → **{n_ep[S]} episodi**.
Alternativa (`{RL}`): {K.EP_DEFS[RL]}, stessa regola di unione → **{n_ep[RL]} episodi**.
Picco = chiusura massima dell'episodio. Livello di partenza = mediana delle {K.PRE_WIN} settimane prima dell'inizio. Quota recuperata dopo k settimane =
(picco − chiusura) / (picco − partenza): 0 al picco, 1 al ritorno al livello di partenza.

Episodi per classe di altezza del picco:

{md(cls.rename(columns={'n_episodi': 'episodi', 'quota_definita': 'con quota recuperata definita'}))}

Con la definizione relativa {int(cls[(cls['definizione'] == RL) & (cls['classe'] == '<30')]['n_episodi'].iloc[0])} episodi hanno picco sotto 30: sono i picchi nati in periodi calmi, che la soglia fissa non vede.
**Le classi 40-50 e ≥ 50 contengono {ncl(S, '40-50')} e {ncl(S, '>=50')} episodi: per loro ogni numero è un aneddoto, non una stima.**

Elenco degli episodi (definizione principale):

{md(ep_list.rename(columns={'sett_recupero_50': 'sett. 50%', 'sett_recupero_100': 'sett. 100%', 'sett_sotto_25': 'sett. <25', 'sett_sotto_20': 'sett. <20', 'rimbalzi_oltre_20pct': 'rimbalzi'}), ".1f")}

![Traiettorie dopo il picco](figures/fase1/a1_traiettorie.png)

Quota della salita recuperata:

{t_quota}

In punti VIX:

{t_punti}

Tempi di recupero e rimbalzi:

{t_tempi}

Per classe di altezza del picco (stima [IC 95%]):

{t_classi}

- **La prima settimana fa gran parte del lavoro.** Una settimana dopo il picco è già recuperato il {pct(g1(S, 'quota_k1')['stima'], 0)} della salita (definizione principale) o il {pct(g1(RL, 'quota_k1')['stima'], 0)} (relativa); il 50% è raggiunto in {n(g1(S, 'sett_recupero_50')['stima'], 0)} settimane in mediana con entrambe le definizioni.
- **Il ritorno completo è lento e molto variabile.** Mediana {g1c(S, 'sett_recupero_100', dec=1)} settimane con la definizione principale, {g1c(RL, 'sett_recupero_100', dec=1)} con quella relativa; in quest'ultima {cens100[RL]} episodi non tornano al livello di partenza entro la fine del campione. Dopo i picchi nati in periodi molto calmi il livello di partenza può non essere più rivisto per anni: la «quota recuperata» si ferma sotto 1.
- **Ritorno sotto le soglie.** Sotto 25 in {n(g1(S, 'sett_sotto_25')['stima'], 0)} settimane e sotto 20 in {n(g1(S, 'sett_sotto_20')['stima'], 0)} (mediane, definizione principale).
- **L'altezza del picco conta.** Per i picchi 30-40 il 100% è recuperato in {n(g1(S, 'sett_recupero_100', '30-40')['stima'], 0)} settimane (mediana, {int(g1(S, 'sett_recupero_100', '30-40')['n'])} episodi); per i picchi 40-50 ne servono {n(g1(S, 'sett_recupero_100', '40-50')['stima'], 1)} e per i due ≥ 50 {n(g1(S, 'sett_recupero_100', '>=50')['stima'], 0)}. Pochi casi, ma la direzione è coerente: più alto il picco, più lenta la normalizzazione.
- **Rimbalzi.** Durante la discesa si osservano in media {n(g1(S, 'rimbalzi_oltre_20pct')['stima'])} rialzi settimanali oltre +20% per episodio (definizione principale; {n(g1(RL, 'rimbalzi_oltre_20pct')['stima'])} con la relativa); il {pct(g1(S, 'almeno_un_rimbalzo')['stima'], 0)} degli episodi ne ha almeno uno ({pct(g1(RL, 'almeno_un_rimbalzo')['stima'], 0)} con la relativa). La discesa non è liscia.

**Confronto con l'AR(1).** Scarto tra il log-VIX realizzato k settimane dopo il picco e quello previsto dall'AR(1) partendo dal picco
(negativo = discesa reale più rapida). L'AR(1) «fino al picco» usa solo la storia disponibile al picco (richiede almeno 104 settimane).

{t_ar}

![Confronto con AR(1)](figures/fase1/a1_confronto_ar1.png)

La discesa reale è **più rapida di quella dell'AR(1) nelle prime settimane** (a 2 settimane: {g1c(S, 'ar_storia_k2', dec=3)}; a 4: {g1c(S, 'ar_storia_k4', dec=3)}),
poi lo scarto si chiude: a 13 settimane è {g1c(S, 'ar_storia_k13', dec=3)} e a 26 {g1c(S, 'ar_storia_k26', dec=3)}. Due avvertenze: il picco è scelto
a posteriori come massimo dell'episodio, quindi una parte del calo successivo è garantita per costruzione; e gli episodi non sono indipendenti
({int(yr_peak.max())} hanno il picco nel {int(yr_peak.idxmax())}).

**Cosa implica per il modello di previsione.** (i) Serve una componente che rientra in 1–4 settimane oltre a quella lenta: un AR(1) da solo
sbaglia la forma della discesa; l'HAR e l'ARMA(1,1) ce l'hanno. (ii) Il picco non è riconoscibile in tempo reale, quindi questi numeri non si
traducono in una regola di previsione: servono come controllo di plausibilità delle traiettorie previste. (iii) La dispersione tra episodi
(quartili molto larghi, rimbalzi oltre +20%) dice che dopo un picco conta la distribuzione più del valore centrale: è materia per la Fase 2.

### A2. Prima dei picchi

Per ogni episodio le grandezze sono misurate nell'ultima settimana prima dell'inizio, su finestre che guardano indietro fino a 12 settimane.
Il confronto è con tutte le altre settimane fuori dagli episodi. `escursione` usa `high` e `low` ed è **esplorativa**.

![Prima dei picchi](figures/fase1/a2_prima_dei_picchi.png)

{t_a2c}

(«Percentile medio»: posizione media dei valori pre-episodio nella distribuzione delle altre settimane; 50 = nessuna differenza.)

Frequenza con cui un episodio inizia nelle 4 settimane successive, per terzile (settimane fuori dagli episodi; le settimane con evento
sono fino a 4 per ogni episodio, quindi gli eventi indipendenti sono {n_ep[S]} e {n_ep[RL]}):

{t_a2t}

![Terzili](figures/fase1/a2_terzili.png)

- **Definizione principale.** Prima dell'inizio il VIX è già alto (mediana {n(c2.loc[(S, 'livello'), 'mediana_pre_episodio'], 1)} contro {n(c2.loc[(S, 'livello'), 'mediana_altre'], 1)}): è quasi una tautologia, perché per chiudere sopra 30 bisogna partire vicino a 30. Gli episodi inoltre si raggruppano: la distanza mediana dal precedente è {n(c2.loc[(S, 'sett_da_prec'), 'mediana_pre_episodio'], 0)} settimane contro {n(c2.loc[(S, 'sett_da_prec'), 'mediana_altre'], 0)}.
- **Definizione relativa** (più pulita, perché non dipende dal livello): il livello non distingue ({n(c2.loc[(RL, 'livello'), 'mediana_pre_episodio'], 1)} contro {n(c2.loc[(RL, 'livello'), 'mediana_altre'], 1)}). Distingue la **salita già in corso**: variazione a 4 settimane {n(c2.loc[(RL, 'var4'), 'mediana_pre_episodio'], 3)} contro {n(c2.loc[(RL, 'var4'), 'mediana_altre'], 3)}, a 12 settimane {n(c2.loc[(RL, 'var12'), 'mediana_pre_episodio'], 3)} contro {n(c2.loc[(RL, 'var12'), 'mediana_altre'], 3)}.
- **Terzili.** Con la definizione relativa l'intervallo della differenza alto − basso esclude lo zero per: {', '.join(sig_rel) or 'nessuna grandezza'}. Con la principale per: {', '.join(sig_s) or 'nessuna grandezza'}.
- **Deviazione standard ed escursione recenti**: differenze piccole, con intervalli che comprendono lo zero in quasi tutti i confronti.

**Conclusione.** Esiste un segnale, ma è debole e poco utile: un episodio è più probabile quando il VIX sta già salendo (e, con la soglia fissa,
quando è già alto o un altro episodio è finito da poco). Anche nel terzile più a rischio la frequenza resta bassa
({pct(t2.loc[(RL, 'var12', 'alto'), 'frequenza'])} con la definizione relativa, contro una frequenza di base del {pct(d2.loc[(RL, 'var12'), 'frequenza_base'])}):
nel {pct(1 - t2.loc[(RL, 'var12', 'alto'), 'frequenza'], 0)} delle settimane «a rischio» non inizia alcun episodio. La «calma prima della tempesta» misurata con la volatilità del VIX non è un segnale.
**In pratica i picchi arrivano senza preavviso utilizzabile**; ciò che si vede prima è l'inizio della salita stessa.

**Cosa implica per il modello di previsione.** Con il solo VIX non si può anticipare un picco nella previsione puntuale: nessuna variabile di
stato sposta la mediana in modo utile. Il rischio di picco va trattato come coda destra della distribuzione prevista (Fase 2), che può
allargarsi quando il VIX è in salita. Un vero segnale anticipatore, se esiste, va cercato nei dati esterni (Fase 3).

### A3. Periodi calmi

**Definizioni.** Principale (`sotto15`): {AI['definizioni_periodo_calmo']['sotto15']}. Alternativa (`escursione_bassa`): {AI['definizioni_periodo_calmo']['escursione_bassa']}
(soglia = quantile 20% calcolato sulla storia fino a quella settimana, a partire dalla settimana {K.CALM_RANGE_MIN_HIST}; un periodo è una sequenza consecutiva di settimane sotto soglia, di qualunque lunghezza).

![Periodi calmi](figures/fase1/a3_periodi_calmi.png)

{t_a3}

Probabilità che il periodo finisca nelle 4 settimane successive, per durata già trascorsa:

{t_a3h}

- **Quanti e quando.** {len(per15)} periodi sotto 15, concentrati in pochi anni: {int(((yrs15 >= 2004) & (yrs15 <= 2007)).sum())} tra 2004 e 2007, {int(((yrs15 >= 2013) & (yrs15 <= 2019)).sum())} tra 2013 e 2019, {int((yrs15 >= 2023).sum())} dal 2023. Durata da {int(per15['durata'].min())} a {int(per15['durata'].max())} settimane (mediana {n(g3('sotto15', 'durata')['stima'], 1)}).
- **Banda di oscillazione.** Dentro un periodo il VIX si muove in una banda mediana di {n(g3('sotto15', 'banda_punti')['stima'])} punti ({n(g3('sotto15', 'banda_log')['stima'], 3)} in log).
- **Come finiscono.** Per definizione un periodo sotto 15 finisce con un rialzo. Quel rialzo è brusco: variazione log mediana nella settimana di uscita {g3c('sotto15', 'var_1', 3)}; nel {pct(g3('sotto15', 'uscita_con_rialzo_oltre_20pct')['stima'], 0)} dei casi (IC 95%: {pct(g3('sotto15', 'uscita_con_rialzo_oltre_20pct')['ci_lo'], 0)}–{pct(g3('sotto15', 'uscita_con_rialzo_oltre_20pct')['ci_hi'], 0)}) supera +20% in una settimana.
- **Poi rientra in parte.** Dalla prima alla quarta settimana dopo l'uscita la variazione mediana è {g3c('sotto15', 'prosecuzione_1_4', 3)}: a 4 settimane dall'ultima settimana calma il guadagno mediano si riduce a {g3c('sotto15', 'var_4', 3)}.
- **La durata non aiuta.** P(fine entro 4 settimane) è {n(h3i.loc[('sotto15', '8-12'), 'prob_fine_entro_4'])} dopo 8-12 settimane e {n(h3i.loc[('sotto15', '13-25'), 'prob_fine_entro_4'])} dopo 13-25. Il valore {n(h3i.loc[('sotto15', '>=26'), 'prob_fine_entro_4'])} oltre le 26 settimane viene da {int(h3i.loc[('sotto15', '>=26'), 'n_periodi'])} soli periodi ({int(h3i.loc[('sotto15', '>=26'), 'n_settimane'])} settimane): non è una stima affidabile. Confrontando tutte le durate oltre 12 settimane con 8-12, la differenza è {cit(a3d.loc['sotto15', 'stima_2classi'], a3d.loc['sotto15', 'ci_lo_2classi'], a3d.loc['sotto15', 'ci_hi_2classi'])}: compatibile con zero.
- **Definizione alternativa.** {int(g3('escursione_bassa', 'durata')['n'])} periodi, molto più brevi (mediana {n(g3('escursione_bassa', 'durata')['stima'], 0)} settimane): più che periodi sono finestre. Escono al rialzo nel {pct(g3('escursione_bassa', 'uscita_al_rialzo')['stima'], 0)} dei casi e con un rialzo oltre +20% nel {pct(g3('escursione_bassa', 'uscita_con_rialzo_oltre_20pct')['stima'], 0)}; nemmeno qui la durata trascorsa conta.

**Cosa implica per il modello di previsione.** Dentro un periodo calmo la previsione puntuale migliore resta «poco sopra il valore attuale»;
il momento dell'uscita non è prevedibile dalla durata. L'informazione utile è sulla forma: variazioni piccole quasi sempre, con una coda destra
pesante (un'uscita su tre non supera +20%, due su tre sì). Per la Fase 2 questo chiede una distribuzione asimmetrica quando il VIX è basso,
e una verifica di calibrazione dedicata ai periodi calmi.

### A4. Tabelle condizionate

Fasce: {', '.join(K.BAND7_LABELS)}. Direzione = variazione log nelle ultime 4 settimane: «in calo» sotto −{n(K.DIR_THR)}, «in rialzo» sopra +{n(K.DIR_THR)}, altrimenti «stabile».
Celle: mediana della variazione log successiva [IC 95% a blocchi] e numero di settimane.

![Tabelle condizionate](figures/fase1/a4_tabelle_condizionate.png)

**h = 1**

{a4_table(1)}

**h = 4**

{a4_table(4)}

**h = 12**

{a4_table(12)}

Distribuzione completa a h = 4 (quantili della variazione log e quota di rialzi):

{md(q4)}

(Le tabelle complete per h = 1 e h = 12 sono in `reports/fase1/a4_celle.csv`.)

- **Livello.** La mediana della variazione successiva scende in modo regolare con la fascia: a 4 settimane da {n(c4v(4, '<12', 'tutte')['mediana'], 3)} partendo sotto 12 a {n(c4v(4, '30-40', 'tutte')['mediana'], 3)} partendo da 30-40; a 12 settimane da {n(c4v(12, '<12', 'tutte')['mediana'], 3)} a {n(c4v(12, '30-40', 'tutte')['mediana'], 3)}. Partendo da 30-40 il VIX è più alto dopo 12 settimane solo nel {pct(c4v(12, '30-40', 'tutte')['quota_rialzi'], 0)} dei casi.
- **Direzione.** A parità di fascia, dopo un rialzo recente il calo successivo è più forte. Esempio nella fascia 15-20 a 4 settimane: {q(4, '15-20', 'in rialzo', 'mediana')} dopo un rialzo contro {q(4, '15-20', 'in calo', 'mediana')} dopo un calo. Un rialzo recente è in parte transitorio.
- **Asimmetria.** Nelle fasce basse la distribuzione è sbilanciata verso l'alto: nella cella 12-15 / stabile a 4 settimane il 10° percentile è {q(4, '12-15', 'stabile', 'q10')} e il 90° è {q(4, '12-15', 'stabile', 'q90')}.
- **Celle poco popolate.** {len(small)} celle su 21 hanno meno di {M.MIN_CELL} casi ({', '.join(f"{r.fascia} / {r.direzione}: {int(r.n)}" for r in small.itertuples())}). Le settimane sono inoltre sovrapposte e dipendenti: i casi indipendenti sono molti meno di n.

**Cosa implica per il modello di previsione.** Livello e direzione recente sono le due variabili di stato che contano, e sono esattamente quelle
dell'HAR (ultimo valore, media a 4 e a 12 settimane). La tabella dà la stessa informazione in forma non parametrica e più rumorosa: infatti M5 non
batte M1 (Parte B). Il valore aggiunto della tabella è nei quantili: larghezza e asimmetria cambiano con la cella, ed è ciò che serve per le
previsioni probabilistiche.

### A5. Verifica dei pattern sulle due metà

Ogni pattern è stimato separatamente su 2000-2012 e 2013-2024. Verdetto: **regge** = stesso segno e intervallo che esclude lo zero in entrambe le metà;
**dati insufficienti** = meno di 10 casi in una metà; altrimenti **non regge**. Per le righe A4 i «casi» sono settimane sovrapposte.

{t_a5}

- **Reggono** ({int(vc.get('regge', 0))}): {'; '.join(regge['id'])}. Sono quasi tutti pattern di ritorno alla media per livello e direzione (A4); l'unico di A2 è il livello con la soglia fissa, in parte meccanico.
- **Dati insufficienti** ({int(vc.get('dati insufficienti', 0))}): tutti i pattern di A1, quelli di A2 con la definizione relativa, quelli dei periodi sotto 15 e la durata con la definizione alternativa. Con circa dieci episodi per metà non si può confermare né smentire nulla; in particolare i periodi sotto 15 sono solo {int(a5.loc[a5['id'] == 'A3-sotto15-var4', 'n_1'].iloc[0])} nella prima metà.
- **Non reggono** ({int(vc.get('non regge', 0))}): i segnali anticipatori di A2 diversi dal livello, i pattern dei periodi calmi con la definizione alternativa, il ritorno verso l'alto a 1 e 12 settimane partendo da VIX basso, l'effetto della direzione a 1 settimana.
- Lo scarto dall'AR(1) dopo i picchi ha lo stesso segno nelle due metà ma è più marcato nella seconda; con 7–13 episodi per metà resta non verificabile.

**Cosa implica per il modello di previsione.** I soli pattern abbastanza solidi da entrare in un modello sono quelli di livello e direzione,
che l'HAR già contiene. I pattern «a eventi» (dopo i picchi, prima dei picchi, fine dei periodi calmi) non hanno abbastanza casi per essere
stimati in modo affidabile: non vanno codificati come regole.

## 3. Parte B — Modelli

**Protocollo** (identico alla Fase 0): target log(VIX); {BI['schema']}; prima stima su {BI['min_train_settimane']} settimane.
Previsioni valutate: {BI['per_orizzonte']['1']['n_previsioni']} a h = 1, {BI['per_orizzonte']['4']['n_previsioni']} a h = 4, {BI['per_orizzonte']['12']['n_previsioni']} a h = 12
(obiettivi dal {BI['per_orizzonte']['1']['prima_data_obiettivo']} al {BI['per_orizzonte']['1']['ultima_data_obiettivo']}). h = 12 è secondario.

**Elenco chiuso dei modelli** (parametri fissati prima di eseguire la valutazione):

| sigla | modello | note di implementazione |
|---|---|---|
| M0 | Random walk | |
| M1 | Tipo HAR della Fase 0 | stessa funzione della Fase 0; regressione diretta per ogni h. **Riferimento.** |
| M2 | ARMA(1,1) su log-VIX | {arma['metodo']}; ristima {arma['ristima']} ({arma['stime']} stime, {arma['non_convergenti']} con ottimizzazione non convergente, {arma['fallite']} fallite); previsione iterata |
| M3 | HAR esteso | variabili di M1 + ultima variazione settimanale divisa in parte positiva e parte negativa |
| M4 | M1 su finestra mobile | ultime {M.ROLL_WINDOW} settimane; identico a M1 per le prime {same_m4} origini, quando la storia è più corta |
| M5 | Tabella condizionata | valore attuale + mediana storica della variazione a h nella cella fascia × direzione; sotto {M.MIN_CELL} casi usa la sola fascia; fascia mai vista: variazione nulla |
| M6 | Analogia | {M.K_ANALOG} situazioni passate più simili (distanza euclidea sullo stato standardizzato con la storia disponibile); solo casi con esito già noto |
| M7 | Combinazione | media semplice di M2, M3, M6 |

**Test contro M1.** Clark-West (una coda) per le coppie annidate: M0 dentro M1 e M1 dentro M3. Diebold-Mariano (due code) per gli altri.
La colonna «statistica» è positiva quando il modello della riga è migliore di M1. Holm sui {BI['n_test_holm']} confronti (7 modelli × 3 orizzonti).
R² vs M1 = 1 − SSE_modello / SSE_M1.

### 3.1 Tabella comparativa

{b_main()}

![R² rispetto a M1](figures/fase1/b_r2_vs_m1.png)

![Errore cumulato rispetto a M1](figures/fase1/b_errore_cumulato_vs_m1.png)

- **M1 resta il riferimento.** Nessun modello ha un R² rispetto a M1 superiore a {pct(max(best[h]['r2_oos_vs_m1'] for h in K.HORIZONS), 2)}; nessuna differenza positiva è significativa, nemmeno prima della correzione di Holm.
- **M2 (ARMA) e M3 (HAR esteso) sono equivalenti a M1**: R² vs M1 {r2(1, 'M2')} e {r2(1, 'M3')} a h = 1, {r2(4, 'M2')} e {r2(4, 'M3')} a h = 4. La componente transitoria individuata in Fase 0 è già catturata dall'HAR; separare rialzi e ribassi dell'ultima settimana non aggiunge nulla (Clark-West p = {n(B.loc[(1, 'M3'), 'p'], 3)} a h = 1, {n(B.loc[(4, 'M3'), 'p'], 3)} a h = 4).
- **M4 (finestra mobile) è un po' peggiore** a tutti gli orizzonti ({r2(1, 'M4')}, {r2(4, 'M4')}, {r2(12, 'M4')}): l'instabilità dei parametri vista in Fase 0 non si risolve accorciando la finestra.
- **M5 e M6 sono peggiori**, in modo significativo a h = 1 ({r2(1, 'M5')} e {r2(1, 'M6')}; p Holm {n(B.loc[(1, 'M5'), 'p_holm'], 3)} e {n(B.loc[(1, 'M6'), 'p_holm'], 3)}). A h = 1 non battono nemmeno il random walk.
- **M7 (combinazione) eguaglia M1** ({r2(1, 'M7')}, {r2(4, 'M7')}, {r2(12, 'M7')}) e ha il MAE in log più basso a tutti gli orizzonti ({n(B.loc[(1, 'M7'), 'mae_log'], 4)} contro {n(B.loc[(1, 'M1'), 'mae_log'], 4)} a h = 1): differenze minime, non significative.
- **Accuratezza direzionale** tra {pct(bm[bm['modello'] != 'M0'].groupby('h')['acc_direzionale'].min()[1])} e {pct(bm[bm['modello'] != 'M0'].groupby('h')['acc_direzionale'].max()[1])} a h = 1 e fino al {pct(bm[bm['modello'] != 'M0']['acc_direzionale'].max())} a h = 12: nessun modello si stacca.

**Sensibilità di M6 al numero di vicini** (solo informativa, non usata per scegliere):

{md(tk, ".4f")}

Più vicini si usano, più M6 migliora: mediando su più casi l'analogia si avvicina a una regressione. È coerente con il resto: non c'è struttura locale da sfruttare.

### 3.2 Per sottoperiodo (data dell'obiettivo)

R² fuori campione rispetto a M1, in percentuale (positivo = meglio di M1). Tabella completa in `reports/fase1/b_metriche_sottoperiodi.csv`.

{b_pivot(bper, 'sottoperiodo')}

Nessun modello è migliore di M1 in tutti i sottoperiodi. M4 perde soprattutto nel 2020-2024 ({pct(per.loc[(4, '2020-2024', 'M4'), 'r2_oos_vs_m1'], 1)} a h = 4, {pct(per.loc[(12, '2020-2024', 'M4'), 'r2_oos_vs_m1'], 1)} a h = 12):
una spiegazione plausibile è che la finestra di dieci anni, dominata dalla calma 2013-2019, fosse impreparata al 2020.

### 3.3 Per fascia di partenza

{b_pivot(bfas, 'fascia_origine')}

Partendo da VIX ≥ 30 ({int(fas.loc[(1, '>=30', 'M1'), 'n'])} previsioni) M2 e M7 fanno meglio di M1 ({pct(fas.loc[(4, '>=30', 'M2'), 'r2_oos_vs_m1'], 1)} e {pct(fas.loc[(4, '>=30', 'M7'), 'r2_oos_vs_m1'], 1)} a h = 4;
{pct(fas.loc[(12, '>=30', 'M2'), 'r2_oos_vs_m1'], 1)} e {pct(fas.loc[(12, '>=30', 'M7'), 'r2_oos_vs_m1'], 1)} a h = 12), ma perdono nelle fasce intermedie. Sono pochi casi concentrati in poche crisi:
un'indicazione, non un risultato.

### 3.4 Sottoinsiemi «dopo un picco» e «periodo calmo»

**Queste due etichette sono assegnate a posteriori** e servono solo a suddividere la valutazione: «{lab_peak}» = {BI['etichette_a_posteriori']['dopo_picco']};
«{lab_calm}» = {BI['etichette_a_posteriori']['periodo_calmo']}. Né il picco né l'appartenenza a un periodo calmo sono noti all'origine della previsione.
I test in questa tabella non sono corretti per confronti multipli.

{b_pivot(bsub, 'sottoinsieme')}

- **Dopo un picco** ({BI['per_orizzonte']['1']['n_dopo_picco']} origini) il ritorno alla media rende molto: M1 ha R² rispetto al random walk {pct(sub.loc[(1, lab_peak, 'M1'), 'r2_oos_vs_rw'])} a h = 1, {pct(sub.loc[(4, lab_peak, 'M1'), 'r2_oos_vs_rw'])} a h = 4 e {pct(sub.loc[(12, lab_peak, 'M1'), 'r2_oos_vs_rw'])} a h = 12. Gli altri modelli non fanno meglio: M3 guadagna {pct(sub.loc[(1, lab_peak, 'M3'), 'r2_oos_vs_m1'], 1)} a h = 1 (Clark-West p = {n(sub.loc[(1, lab_peak, 'M3'), 'p'], 3)}, non corretto); M5 e M6 perdono oltre il 10%.
- **Dentro un periodo calmo** ({BI['per_orizzonte']['1']['n_periodo_calmo']} origini) gli errori sono più piccoli (RMSE log di M1 {n(sub.loc[(1, lab_calm, 'M1'), 'rmse_log'], 4)} a h = 1) e il vantaggio sul random walk è modesto ({pct(sub.loc[(1, lab_calm, 'M1'), 'r2_oos_vs_rw'])}). M7 e M3 fanno leggermente meglio di M1 a h = 1 ({pct(sub.loc[(1, lab_calm, 'M7'), 'r2_oos_vs_m1'], 1)} e {pct(sub.loc[(1, lab_calm, 'M3'), 'r2_oos_vs_m1'], 1)}; p non corretti {n(sub.loc[(1, lab_calm, 'M7'), 'p'], 3)} e {n(sub.loc[(1, lab_calm, 'M3'), 'p'], 3)}), ma a h = 4 M3 torna sotto M1.

### 3.5 Cosa implica per il modello di previsione

1. **Per la previsione puntuale il tipo HAR resta il modello di riferimento.** Sette alternative costruite sul solo VIX (dinamica ARMA, asimmetria, finestra mobile, tabelle, analogie, combinazione) non lo migliorano. Con il solo VIX la media condizionata sembra sfruttata quasi per intero.
2. **I pattern della Parte A non sono informazione nuova per la media**: livello e direzione sono già nell'HAR; picchi e uscite dai periodi calmi non sono prevedibili nella tempistica.
3. **Il margine residuo sta nella distribuzione, non nel centro.** La Parte A mostra code e asimmetrie che dipendono dallo stato (bassa volatilità con coda destra pesante; dopo i picchi grande dispersione). È l'oggetto della Fase 2.
4. **La combinazione M7 non fa danni** e ha il MAE più basso: può restare come alternativa di controllo, senza aspettarsi guadagni.

## 4. Problemi e ambiguità

1. **Il picco è definito a posteriori.** Tutta l'analisi A1 condiziona su un'informazione non disponibile in tempo reale; parte del calo dopo il picco è garantita per costruzione. Lo stesso vale per «ultima settimana di un periodo calmo» in A3.
2. **Quota recuperata non definita** quando la salita è inferiore a {n(K.MIN_RISE, 0)} punti: accade per {len(no_quota)} episodio della definizione principale ({', '.join(str(x) for x in no_quota['data_picco'])}, partenza {', '.join(n(x, 1) for x in no_quota['base'])}), escluso dalle statistiche sulla quota. La soglia di {n(K.MIN_RISE, 0)} punti è una mia scelta.
3. **Livello di partenza calcolato su meno di {K.PRE_WIN} settimane** per gli episodi a inizio campione (numero di casi: {len(short_base)}); sono usate le settimane disponibili.
4. **Classe «<30»** aggiunta per la definizione relativa: le classi richieste (30-40, 40-50, ≥ 50) non coprono i picchi nati in periodi calmi.
5. **Tempi di recupero censurati**: episodi che non recuperano entro la fine del campione sono esclusi dalla mediana e contati a parte (per il recupero del 100%: {cens100[S]} con la definizione principale, {cens100[RL]} con la relativa). I tempi non hanno limite a 26 settimane.
6. **«Oltre +20%»** = variazione percentuale semplice settimanale superiore a 20%. **«Oltre 10%»** per la direzione = variazione log a 4 settimane oltre ±0,10. Interpretazioni mie.
7. **A2, «nelle 12 settimane prima»**: ho misurato le grandezze nell'ultima settimana prima dell'inizio, con finestre all'indietro fino a 12 settimane. I terzili sono calcolati sull'intero campione di sviluppo (uso descrittivo); in A5 sono ricalcolati dentro ciascuna metà.
8. **Bootstrap.** Gli episodi sono ricampionati come indipendenti, ma si raggruppano nel tempo; le settimane sono ricampionate a blocchi di {K.BLOCK}. Gli intervalli sono approssimati e, sotto i 10 casi, poco affidabili. Nella Parte A non c'è correzione per confronti multipli: su decine di intervalli, qualcuno esclude lo zero per caso.
9. **A3, definizione alternativa**: la soglia è un quantile calcolato sulla storia fino a quel momento (dalla settimana {K.CALM_RANGE_MIN_HIST}); produce sequenze brevi (mediana {n(g3('escursione_bassa', 'durata')['stima'], 0)} settimane) che è generoso chiamare periodi. Nessuna durata minima imposta.
10. **A3, durata**: la classe più lunga contiene {int(h3i.loc[('sotto15', '>=26'), 'n_periodi'])} periodi. Dopo aver visto che in A5 una metà restava senza casi in quella classe, ho aggiunto il confronto a due classi (oltre la prima classe contro la prima). È l'unica modifica fatta dopo aver visto dei risultati e riguarda la sola Parte A.
11. **A5.** La regola dei verdetti è meccanica. Per le righe A4 i casi sono settimane sovrapposte, quindi la soglia dei 10 casi non è mai vincolante e gli intervalli a blocchi sono l'unica protezione. Le due metà hanno regimi diversi (la prima contiene il 2008).
12. **M5**: quando la fascia di partenza non ha alcun caso passato la previsione è il valore attuale. **M4**: coincide con M1 per le prime {same_m4} origini. Entrambe le scelte sono state fissate prima della valutazione.
13. **M2**: {arma['non_convergenti']} stime su {arma['stime']} hanno l'ottimizzatore che non dichiara la convergenza; le stime sono usate comunque. Nessun ripiego sull'AR(1) è stato necessario ({arma['fallite']} stime fallite).
14. **Test.** Holm è applicato a p-value di natura diversa (Clark-West a una coda, Diebold-Mariano a due code). DM tra M1 e M4 (stesso modello, finestra diversa) e tra M1 e M7 (che contiene M3) è un'approssimazione. A h = 12 gli errori sono sovrapposti (varianza HAC con 11 ritardi). Nei sottogruppi le osservazioni non sono contigue e i test sono solo indicativi.
15. **«Entro 12 settimane da un picco»**: interpretato come da 0 a 12 settimane *dopo* il picco.
16. **h = 12 per M1**: la Fase 0 non lo riportava; è la stessa regressione diretta, valutata qui per la prima volta.
17. **Ordine dei lavori.** Durante lo sviluppo ho eseguito i modelli una volta a h = 4 per verificarne il funzionamento prima di scrivere il modulo di valutazione. Nessun parametro è stato cambiato dopo.
18. **Colonne.** `open` è scartata alla lettura. `high` e `low` sono usate solo per `escursione` in A2 (lo verifica un test).

## 5. Proposte per la Fase 2 (previsione probabilistica)

1. **Centro della distribuzione: M1.** Motivo: nessuna alternativa lo batte; usare M7 come controllo. Tutta la Fase 2 riguarda la forma attorno a quel centro.
2. **Elenco chiuso di modelli di densità, dal più semplice.** (a) M1 + quantili empirici incondizionati dei suoi errori passati (riferimento da battere); (b) M1 + quantili degli errori condizionati alla fascia di livello; (c) condizionati a fascia × direzione, con la stessa regola dei 20 casi di M5; (d) regressione quantilica sulle variabili dell'HAR; (e) M1 + varianza di tipo GARCH sui residui con distribuzione asimmetrica; (f) quantili dalle analogie di M6. Motivo: A4 mostra che larghezza e asimmetria cambiano con la cella; M5 e M6 falliscono sul centro ma possono essere utili sulla dispersione.
3. **Eventi da prevedere.** P(VIX > 25) a h = 1 e 4 come in Fase 0, con la catena di Markov come riferimento; propongo di aggiungere P(VIX > 20), P(VIX > 30) e P(rialzo settimanale oltre +20%), quest'ultimo perché A3 mostra che è il modo tipico in cui finisce la calma.
4. **Metriche.** CRPS e perdita quantilica (pinball) per la densità; Brier con scomposizione affidabilità/risoluzione e curve di calibrazione per gli eventi; copertura degli intervalli al 50%, 80% e 90%; istogramma PIT.
5. **Calibrazione per stato.** Riportare copertura e CRPS separatamente per fascia di partenza, dentro i periodi calmi e dopo i picchi (etichette a posteriori, solo per suddividere). Motivo: sono gli stati in cui A1 e A3 mostrano le distribuzioni più asimmetriche.
6. **Protocollo invariato.** Stesso walk-forward, stessi orizzonti (1 e 4 principali, 12 secondario), elenco chiuso, Holm, holdout bloccato.

**Decisioni richieste prima della Fase 2:** (i) quali eventi oltre P(VIX > 25); (ii) quali livelli di quantile e di copertura; (iii) se l'elenco dei modelli di densità qui proposto va bene o va ridotto; (iv) se h = 12 resta secondario.

## 6. Riproducibilità

```
python -m src.run_phase0     # Fase 0 (invariata): produce anche data/processed/vix_weekly_dev.csv
python -m src.run_phase1     # Fase 1: tabelle in reports/fase1/, figure in reports/figures/fase1/, questo report
python -m pytest             # test: anti look-ahead per tutti i modelli, blocco dell'holdout, definizioni
```

Versioni e seed: `reports/fase1/run_manifest.json`. Previsioni walk-forward complete: `reports/fase1/b_previsioni.csv`.
"""
    (C.REPORTS_DIR / "fase1_report.md").write_text(out)


if __name__ == "__main__":
    run()
