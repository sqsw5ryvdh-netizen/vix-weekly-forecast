"""Genera reports/fase4c_report.md. Tutti i numeri sono letti dai file reports/fase4c/*.csv e *.json: nessun numero e' scritto a mano."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C
from ..fase3.report import MIN_N, ic, md, n, pc, pct
from . import estremi as ES
from . import valuta as VA
from . import variabili as V

R = C.REPORTS_DIR / "fase4c"
NOME = VA.NOMI
MISURE = {"n": "origini", "massimo_entro_5": "massimo entro il 5% dal reale", "massimo_entro_10": "massimo entro il 10%", "massimo_errore_pct_mediano": "massimo: errore percentuale mediano",
          "massimo_oltre_5_per_eccesso": "massimo oltre il 5% per eccesso (previsto troppo alto)", "massimo_oltre_5_per_difetto": "massimo oltre il 5% per difetto (previsto troppo basso)",
          "minimo_entro_5": "minimo entro il 5% dal reale", "minimo_entro_10": "minimo entro il 10%", "minimo_errore_pct_mediano": "minimo: errore percentuale mediano",
          "minimo_oltre_5_per_eccesso": "minimo oltre il 5% per eccesso", "minimo_oltre_5_per_difetto": "minimo oltre il 5% per difetto", "entrambi_entro_5": "massimo e minimo entrambi entro il 5%",
          "sovrapposizione": "sovrapposizione con la candela reale", "chiusura_nell_escursione": "chiusura reale dentro l'escursione prevista", "chiusura_nel_corpo": "chiusura reale nel corpo previsto",
          "rapporto_corpo": "corpo previsto / reale", "rapporto_escursione": "escursione prevista / reale", "mae_massimo": "massimo: errore medio (punti)", "mae_minimo": "minimo: errore medio (punti)",
          "mae_chiusura": "chiusura della candela: errore medio (punti)"}
RAPPORTI = {"sovrapposizione", "rapporto_corpo", "rapporto_escursione", "mae_massimo", "mae_minimo", "mae_chiusura"}


def val(k: str, v) -> str:
    if not np.isfinite(v):
        return "–"
    return f"{int(v)}" if k == "n" else n(v, 3) if k in RAPPORTI else pc(v)


def pv(p: float) -> str:
    return "< 0,0001" if p < 0.0001 else "= " + n(p, 4)


def run() -> None:
    T = lambda k: pd.read_csv(R / f"{k}.csv")
    info = json.loads((R / "info.json").read_text())
    ta, tr, tp, tb, spx, tc, te, tpd = T("a_misure"), T("a_regola"), T("a_confronti_sulle_proprie_origini"), T("b_incrociata"), T("b_spx"), T("c_corpo"), T("c_corpo_per_etichetta"), T("prima_e_dopo")
    sc, ins, pc_, camp = info["scelte"], info["insiemi"], info["parte_c"], info["campioni"]
    A = ta.set_index(["campione", "gruppo", "modello", "bersaglio"])
    P = tpd.set_index(["versione", "campione", "gruppo", "stato"])
    RF = tr[tr["famiglia"].str.startswith("FS")].set_index(["bersaglio", "modello"])
    R1 = tr[tr["famiglia"].str.startswith("F1")].set_index(["bersaglio", "modello"])
    CE = "origini con l'etichetta di FS"
    n_tutte, n_com = camp["tutte le origini"]["n"], camp["origini comuni"]["n"]

    # ------------------------------------------------------------------ tabella in evidenza
    def evidenza() -> str:
        rows = []
        for v, c in (("F1", "tutte le origini"), ("F1", CE), ("FS", CE)):
            for g in ("tutte", "senza strappi"):
                a, d = P.loc[(v, c, g, "prima")], P.loc[(v, c, g, "dopo")]
                rows.append({"versione": v, "origini": c, "settimane": g, "n": int(a["n"]), "massimo entro il 5%: prima": pc(a["massimo_entro_5"]), "massimo: dopo": pc(d["massimo_entro_5"]),
                             "minimo entro il 5%: prima": pc(a["minimo_entro_5"]), "minimo: dopo": pc(d["minimo_entro_5"]),
                             "errore % mediano massimo: prima → dopo": f"{pc(a['massimo_errore_pct_mediano'])} → {pc(d['massimo_errore_pct_mediano'])}",
                             "errore % mediano minimo: prima → dopo": f"{pc(a['minimo_errore_pct_mediano'])} → {pc(d['minimo_errore_pct_mediano'])}"})
        return pd.DataFrame(rows).to_markdown(index=False)

    def prima_dopo(v: str, c: str) -> str:
        cols = [(f"{g}, {s}", (v, c, g, s)) for g in ("tutte", "senza strappi", "fascia 13-22") for s in ("prima", "dopo")]
        return pd.DataFrame([{"misura": lab} | {nome: val(k, P.loc[ch, k]) for nome, ch in cols} for k, lab in MISURE.items()]).to_markdown(index=False)

    # ------------------------------------------------------------------ Parte A
    def tab_a(campione: str, gruppo: str, modelli: list[str]) -> str:
        rows = []
        for m in modelli:
            for b in ES.ESTREMI:
                if (campione, gruppo, m, b) not in A.index:
                    continue
                r = A.loc[(campione, gruppo, m, b)]
                rows.append({"modello": NOME[m], "estremo": b, "n": int(r["n"]), "entro il 5%": pc(r["entro_5"]), "entro il 10%": pc(r["entro_10"]), "errore % mediano": pc(r["errore_pct_mediano"]),
                             "errore % medio": pc(r["errore_pct_medio"]), "errore mediano (punti)": n(r["errore_punti_mediano"], 2), "errore medio (punti)": n(r["errore_punti_medio"], 2),
                             "oltre il 5% per eccesso": pc(r["oltre_5_per_eccesso"]), "oltre il 5% per difetto": pc(r["oltre_5_per_difetto"]), "entrambi entro il 5%": pc(r["entrambi_entro_5"])})
        return pd.DataFrame(rows).to_markdown(index=False)

    def pivot_gruppi(campione: str, gruppi: list[str], modelli: list[str]) -> str:
        rows = []
        for g in gruppi:
            for b in ES.ESTREMI:
                if (campione, g, "S0", b) not in A.index:
                    continue
                r = {"gruppo": g, "estremo": b, "casi": f"{int(A.loc[(campione, g, 'S0', b), 'n'])}" + (" (meno di 20)" if A.loc[(campione, g, "S0", b), "n"] < MIN_N else "")}
                rows.append(r | {m: pc(A.loc[(campione, g, m, b), "entro_5"]) for m in modelli if (campione, g, m, b) in A.index})
        return pd.DataFrame(rows).to_markdown(index=False)

    def tab_regola(t: pd.DataFrame) -> str:
        t = t[t["bersaglio"].isin(ES.ESTREMI)].copy()
        t["insieme"] = t["modello"].map(NOME)
        t["guadagno sull'errore in log [IC 95%]"] = [ic(x, lo, hi) for x, lo, hi in zip(t["guadagno"], t["guadagno_ci_lo"], t["guadagno_ci_hi"])]
        t["entro il 5%: insieme / S0"] = [f"{pc(a)} / {pc(b)}" for a, b in zip(t["entro_5"], t["entro_5_s0"])]
        t["ammesso dalla regola"] = t["ammesso"].map({True: "sì", False: "no"})
        return md(t[["bersaglio", "insieme", "n", "n_variabili", "errore_log", "errore_log_s0", "guadagno sull'errore in log [IC 95%]", "dm_stat", "p", "p_holm", "entro il 5%: insieme / S0",
                     "ammesso dalla regola"]].rename(columns={"bersaglio": "estremo", "n_variabili": "variabili in più", "errore_log": "errore assoluto in log", "errore_log_s0": "errore di S0",
                                                             "dm_stat": "DM (positivo = meglio di S0)", "p": "p", "p_holm": "p Holm"}), ".4f")

    tutti = list(VA.INSIEMI) + ["B3", "B2", "FISSA", "PROG"]
    gruppi_f = [f"fascia {b}" for b in C.BAND_LABELS] + ["fascia 13-22", "VVIX terzile basso", "VVIX terzile medio", "VVIX terzile alto"]
    ammessi_fs = {b: [m for m in VA.IN_REGOLA if RF.loc[(b, m), "ammesso"]] for b in ES.ESTREMI}

    def frase_ins(m: str) -> str:
        return "; ".join(f"{b}: entro il 5% {pc(RF.loc[(b, m), 'entro_5'])} contro {pc(RF.loc[(b, m), 'entro_5_s0'])} di S0, errore in log {ic(RF.loc[(b, m), 'guadagno'], RF.loc[(b, m), 'guadagno_ci_lo'], RF.loc[(b, m), 'guadagno_ci_hi'])} "
                         f"(p Holm = {n(RF.loc[(b, m), 'p_holm'], 2)})" for b in ES.ESTREMI)

    def verdetto(m: str) -> str:
        if any(RF.loc[(b, m), "ammesso"] for b in ES.ESTREMI):
            return "passa la regola per " + " e ".join(b for b in ES.ESTREMI if RF.loc[(b, m), "ammesso"])
        if any(RF.loc[(b, m), "guadagno_ci_lo"] > 0 for b in ES.ESTREMI):
            return "un guadagno con intervallo sopra zero su " + " e ".join(b for b in ES.ESTREMI if RF.loc[(b, m), "guadagno_ci_lo"] > 0) + ", non significativo dopo Holm: non passa la regola"
        if any(RF.loc[(b, m), "guadagno_ci_hi"] < 0 for b in ES.ESTREMI):
            return "nessun guadagno (peggiora il " + " e il ".join(b for b in ES.ESTREMI if RF.loc[(b, m), "guadagno_ci_hi"] < 0) + ")"
        return "nessun guadagno distinguibile da zero"

    # ------------------------------------------------------------------ Parte B
    B_ = tb.set_index(["gruppo", "bersaglio"])
    tbb = tb.copy()
    tbb["migliore della Parte A"] = tbb["migliore_parte_a"]
    for k, lab in (("s0", "S0"), ("migliore_a", "migliore della Parte A"), ("b3", "B3 reale"), ("b2_tetto", "B2 TETTO TEORICO")):
        tbb[f"entro il 5%: {lab}"] = [pc(x) for x in tbb[f"entro_5_{k}"]]
    tbb["guadagno del tetto (punti %)"] = [f"{100 * x:+.1f}".replace(".", ",") for x in tbb["guadagno_tetto"]]
    tbb["guadagno reale (punti %)"] = [f"{100 * x:+.1f}".replace(".", ",") for x in tbb["guadagno_reale"]]
    tbb["quota del guadagno che sopravvive"] = [pc(x, 0) if np.isfinite(x) else "–" for x in tbb["quota_che_sopravvive"]]
    t_b = md(tbb[["gruppo", "bersaglio", "n", "migliore della Parte A", "entro il 5%: S0", "entro il 5%: migliore della Parte A", "entro il 5%: B3 reale", "entro il 5%: B2 TETTO TEORICO",
                  "guadagno del tetto (punti %)", "guadagno reale (punti %)", "quota del guadagno che sopravvive"]].rename(columns={"gruppo": "settimane"}))
    sx = spx.set_index("campione")
    t_spx = md(spx[["campione", "n", "direzione_indovinata", "sempre_rialzo", "quota_previsioni_di_rialzo", "r2_rendimento", "r2_escursione"]].rename(columns={
        "direzione_indovinata": "direzione del rendimento indovinata", "sempre_rialzo": "«sempre rialzo»", "quota_previsioni_di_rialzo": "quota di previsioni di rialzo",
        "r2_rendimento": "R² fuori campione, rendimento", "r2_escursione": "R² fuori campione, escursione"}), ".3f")
    sopr = {b: B_.loc[("tutte", b), "quota_che_sopravvive"] for b in VA.BERSAGLI}
    pp = lambda v: f"{100 * v:+.1f}".replace(".", ",") + " punti"
    f_sopr = "; ".join(f"{b}: il tetto vale {pp(B_.loc[('tutte', b), 'guadagno_tetto'])} sulla quota entro il 5%, la versione reale {pp(B_.loc[('tutte', b), 'guadagno_reale'])}"
                       + (f" (sopravvive il {pc(sopr[b], 0)})" if sopr[b] > 0 else " (non sopravvive nulla)") for b in VA.BERSAGLI)

    # ------------------------------------------------------------------ Parte C
    x = tc.copy()
    x["sovrapposizione [IC 95%]"] = [f"{v:.3f} [{lo:.3f}; {hi:.3f}]" for v, lo, hi in zip(x["sovrapposizione"], x["sovrapposizione_ci_lo"], x["sovrapposizione_ci_hi"])]
    t_c = md(x[["versione", "campione", "corpo", "n", "mae_apertura", "mae_chiusura", "sovrapposizione [IC 95%]", "chiusura_nel_corpo", "chiusura_nell_escursione", "direzione_corpo", "rapporto_corpo",
                "rapporto_escursione", "quota_apertura_tagliata", "quota_chiusura_tagliata"]].rename(columns={
        "mae_apertura": "errore medio apertura", "mae_chiusura": "errore medio chiusura", "chiusura_nel_corpo": "chiusura reale nel corpo", "chiusura_nell_escursione": "chiusura reale nell'escursione",
        "direzione_corpo": "direzione del corpo", "rapporto_corpo": "corpo previsto / reale", "rapporto_escursione": "escursione prevista / reale", "quota_apertura_tagliata": "apertura tagliata",
        "quota_chiusura_tagliata": "chiusura tagliata"}), ".3f")
    y = te.copy()
    y["casi"] = [f"{k}" + (" (meno di 20)" if k < MIN_N else "") for k in y["n"]]
    t_ce = md(y[["etichetta", "corpo", "casi", "mae_chiusura", "chiusura_nel_corpo", "direzione_corpo", "rapporto_corpo"]].rename(columns={
        "mae_chiusura": "errore medio chiusura", "chiusura_nel_corpo": "chiusura reale nel corpo", "direzione_corpo": "direzione del corpo", "rapporto_corpo": "corpo previsto / reale"}), ".3f")
    tag = info["tagli"]["F1"][sc["corpo"]]
    esito_c = ("**K5 è adottata.**" if sc["corpo"] == "K5" else "**Resta K2**: l'errore sulla chiusura di K5 " + ("non è significativamente più basso" if not pc_["chiusura_significativamente_migliore"] else "è più basso, ma la sovrapposizione è significativamente più bassa") + ".")

    # ------------------------------------------------------------------ frasi di sintesi
    E = lambda v, c, g, s, k: P.loc[(v, c, g, s), k]
    stesso = ins["f1"] == ins["fs"]
    f_diverse = ("F1 e FS usano gli stessi insiemi." if stesso else
                 f"**F1 e FS finiscono con insiemi diversi** (F1: massimo {ins['f1']['massimo']}, minimo {ins['f1']['minimo']}; FS: massimo {ins['fs']['massimo']}, minimo {ins['fs']['minimo']}): "
                 f"per FS la regola confronta sei candidati sulle {n_com} origini comuni e Holm è più severo; S1 sul minimo lì ha p = {n(RF.loc[('minimo', 'S1'), 'p'], 3)} prima della correzione e {n(RF.loc[('minimo', 'S1'), 'p_holm'], 2)} dopo.")
    sintesi = [
        f"**Massimo, previsione diretta.** Nessun insieme batte S0 secondo la regola. Su tutte le {n_tutte} origini il massimo è entro il 5% dal reale nel {pc(E('F1', 'tutte le origini', 'tutte', 'dopo', 'massimo_entro_5'))} dei casi "
        f"(prima {pc(E('F1', 'tutte le origini', 'tutte', 'prima', 'massimo_entro_5'))}); senza strappi {pc(E('F1', 'tutte le origini', 'senza strappi', 'dopo', 'massimo_entro_5'))} (prima {pc(E('F1', 'tutte le origini', 'senza strappi', 'prima', 'massimo_entro_5'))}). "
        f"La regola fissa fa {pc(A.loc[('tutte le origini', 'tutte', 'FISSA', 'massimo'), 'entro_5'])}: **la previsione diretta del massimo non migliora quella che c'era**.",
        f"**Minimo.** Per F1 la regola adotta {ins['f1']['minimo']} (errore in log {ic(R1.loc[('minimo', 'S1'), 'guadagno'], R1.loc[('minimo', 'S1'), 'guadagno_ci_lo'], R1.loc[('minimo', 'S1'), 'guadagno_ci_hi'])} rispetto a S0, p = {n(R1.loc[('minimo', 'S1'), 'p'], 3)}): "
        f"minimo entro il 5% nel {pc(E('F1', 'tutte le origini', 'tutte', 'dopo', 'minimo_entro_5'))} dei casi (prima {pc(E('F1', 'tutte le origini', 'tutte', 'prima', 'minimo_entro_5'))}); "
        f"senza strappi {pc(E('F1', 'tutte le origini', 'senza strappi', 'dopo', 'minimo_entro_5'))} (prima {pc(E('F1', 'tutte le origini', 'senza strappi', 'prima', 'minimo_entro_5'))}).",
        f"**Scelte della regola.** F1: massimo {ins['f1']['massimo']}, minimo {ins['f1']['minimo']}. FS: massimo {ins['fs']['massimo']}, minimo {ins['fs']['minimo']}. {f_diverse}",
        f"**VVIX (S2): {verdetto('S2')}.** {frase_ins('S2')}.",
        f"**Ampiezza dell'S&P 500 (S3): {verdetto('S3')}.** {frase_ins('S3')}. VIX3M/VIX (S4): {verdetto('S4')}. Tutte insieme (S5): {verdetto('S5')}.",
        f"**Previsione incrociata, versione reale (B3): {verdetto('B3')}.** L'S&P 500 della settimana dopo non si prevede: direzione indovinata nel {pc(sx.loc['origini comuni', 'direzione_indovinata'])} dei casi contro il "
        f"{pc(sx.loc['origini comuni', 'sempre_rialzo'])} di «sempre rialzo», R² del rendimento {pct(sx.loc['origini comuni', 'r2_rendimento'])}; l'escursione invece sì (R² {pct(sx.loc['origini comuni', 'r2_escursione'])}), ma non basta.",
        f"**Tetto teorico (B2, non utilizzabile).** Conoscendo l'S&P 500 della settimana prevista: {f_sopr}.",
        f"**Corpo (Parte C).** {esito_c} Errore medio sulla chiusura della candela: K5 {n(pc_['mae_chiusura']['K5'], 3)} contro K2 {n(pc_['mae_chiusura']['K2'], 3)} punti (p {pv(pc_['p_chiusura'])}). "
        f"Con gli estremi che comandano, la chiusura della candela viene tagliata sul minimo previsto nel {pc(tag['chiusura'], 0)} dei casi.",
        f"**Prima candela, prima e dopo (F1, tutte le origini).** Entrambi gli estremi entro il 5%: {pc(E('F1', 'tutte le origini', 'tutte', 'dopo', 'entrambi_entro_5'))} (prima {pc(E('F1', 'tutte le origini', 'tutte', 'prima', 'entrambi_entro_5'))}); "
        f"errore sulla chiusura della candela {n(E('F1', 'tutte le origini', 'tutte', 'dopo', 'mae_chiusura'), 2)} punti (prima {n(E('F1', 'tutte le origini', 'tutte', 'prima', 'mae_chiusura'), 2)}); "
        f"chiusura reale dentro l'escursione prevista {pc(E('F1', 'tutte le origini', 'tutte', 'dopo', 'chiusura_nell_escursione'))} (prima {pc(E('F1', 'tutte le origini', 'tutte', 'prima', 'chiusura_nell_escursione'))}).",
        "**Congelamento e piano.** F1 e FS sono ricongelate (`versioni/f1.json`, `versioni/fs.json`); nel criterio D del piano d'esame entrano le quote entro il 5% di massimo e minimo. **L'holdout non è stato aperto.**",
    ]

    out = f"""# Fase 4c — Massimo e minimo della prima candela

*Generato da `python -m src.run_phase4c`. Campione di sviluppo fino al {info['ultima_data_obiettivo']}. L'holdout (VIX e dati esterni) non è stato letto: il comando gira sotto la guardia.
Tutti i numeri vengono da `reports/fase4c/`.*

**Conferme registrate:** criterio D1 del piano d'esame (perdita 0 / 1 / 0,5) confermato; K4 resta respinta e il corpo è stato rivalutato qui (Parte C); le altre voci «da sapere» della Fase 4b sono accettate.
**Decisioni applicate:** misura principale = quota di massimi e minimi previsti entro il 5% dal reale ed errore percentuale mediano (il 10% è secondario); ogni tabella anche senza le settimane di strappo;
il massimo è una previsione diretta; massimo e minimo comandano la candela; una sola previsione, alla chiusura settimanale; niente calendario degli eventi, niente modelli linguistici.

## Sintesi

{chr(10).join(f'{i}. {s}' for i, s in enumerate(sintesi, start=1))}

## 1. In evidenza: quote entro il 5%, prima e dopo, con e senza strappi

«Prima» = prima candela del programma dopo la Fase 4b (massimo della candela K2, minimo M1q). «Dopo» = massimo e minimo diretti scelti in questa fase.
«Senza strappi» = senza le settimane in cui la chiusura reale ha superato del {int(100 * info['strappo_soglia'])}% la chiusura all'origine ({info['n_strappi']['tutte le origini']} su {n_tutte} origini):
**è un'etichetta che si conosce solo a posteriori**, serve a leggere i risultati e non è usabile per prevedere.

{evidenza()}

## 2. Parte A — Massimo e minimo diretti

Obiettivi: ln(massimo della settimana +1 / chiusura all'origine) e ln(minimo della settimana +1 / chiusura all'origine). Regressione sulla mediana sulle ultime {info['finestra']} settimane (almeno {info['minimo_coppie']}).
Insiemi (elenco chiuso): {'; '.join(V.DESCRIZIONI.values())}. Per il minimo, S0 è il modello M1q della Fase 4b.
Riferimenti: «regola fissa» (chiusura all'origine × rapporto mediano storico) e «programma attuale» (Fase 4b).

Campioni (ogni confronto è sulle stesse origini): tutte le origini ({n_tutte}, dal {camp['tutte le origini']['prima_data_origine']}); origini con VVIX ({camp['origini con VVIX']['n']}, dal {camp['origini con VVIX']['prima_data_origine']});
origini con VIX3M e origini comuni a tutti gli insiemi ({n_com}, dal {camp['origini comuni']['prima_data_origine']}). Tutte le tabelle, per ogni campione e ogni gruppo, sono in `reports/fase4c/a_misure.csv`.

### 2.1 Tutte le origini

{tab_a('tutte le origini', 'tutte', tutti)}

Senza strappi:

{tab_a('tutte le origini', 'senza strappi', tutti)}

### 2.2 Origini comuni a tutti gli insiemi

{tab_a('origini comuni', 'tutte', tutti)}

Senza strappi:

{tab_a('origini comuni', 'senza strappi', tutti)}

### 2.3 Quota entro il 5% per fascia di partenza, fascia 13-22 e terzili del VVIX (origini comuni)

I terzili del VVIX sono calcolati sulle origini del campione (a posteriori). Le altre misure sono in `a_misure.csv`.

{pivot_gruppi('origini comuni', gruppi_f, list(VA.INSIEMI) + ['B3', 'B2'])}

### 2.4 Regola

Per ciascun estremo: tra gli insiemi significativamente migliori di S0 (Diebold-Mariano sull'errore assoluto in log, p < 0,05 dopo Holm) e con quota entro il 5% non più bassa di S0, quello con l'errore più basso;
se nessuno, S0; a parità meno variabili. B3 partecipa come un insieme in più. Fissato prima di guardare i risultati: per F1 si confronta S1 con S0 su tutte le origini; per FS si confrontano S1-S5 e B3 con S0 sulle origini comuni.

F1 (S1 contro S0, tutte le origini):

{tab_regola(tr[tr['famiglia'].str.startswith('F1')])}

FS (S1-S5 e B3 contro S0, origini comuni):

{tab_regola(tr[tr['famiglia'].str.startswith('FS')])}

- **F1: massimo {ins['f1']['massimo']}, minimo {ins['f1']['minimo']}. FS: massimo {ins['fs']['massimo']}, minimo {ins['fs']['minimo']}.** Insiemi ammessi per FS: massimo {', '.join(ammessi_fs['massimo']) or 'nessuno'}; minimo {', '.join(ammessi_fs['minimo']) or 'nessuno'}.
- {f_diverse}
- Ogni insieme contro S0 sul campione più lungo in cui esiste (solo descrizione): `reports/fase4c/a_confronti_sulle_proprie_origini.csv`.

## 3. Parte B — Previsione incrociata con l'S&P 500 (solo misura)

- **B1, legame.** Nella finestra di stima, regressione sulla mediana di massimo, minimo e chiusura del VIX della settimana +1 sulle variabili di S1 più rendimento ed escursione dell'S&P 500 della stessa settimana +1.
- **B2, TETTO TEORICO — non è una previsione utilizzabile.** Usa rendimento ed escursione dell'S&P 500 realmente avvenuti nella settimana prevista. Serve solo a misurare quanto varrebbe conoscerli. Non entra nel programma (lo verifica un test).
- **B3, versione reale.** Rendimento ed escursione dell'S&P 500 della settimana +1 sono previsti con dati fino all'origine (minimi quadrati; rendimento su rendimento a 1 e a 4 settimane e ln(VIX); escursione su escursione
  dell'ultima settimana, media a 4 e a 12 settimane e ln(VIX)) e inseriti nel legame B1.

Quanto è prevedibile l'S&P 500 della settimana dopo (R² rispetto alla media della finestra di stima):

{t_spx}

Quota entro il 5% dal reale, origini comuni ({n_com}). «Migliore della Parte A»: per massimo e minimo l'insieme scelto per FS; per la chiusura l'insieme S1-S5 con l'errore più basso (solo misura).

{t_b}

**Quanta parte del guadagno del tetto teorico sopravvive nella versione reale:** {f_sopr}.
Il legame tra VIX e S&P 500 dentro la stessa settimana è forte, ma il rendimento dell'S&P 500 non è prevedibile con una settimana di anticipo: senza quello, il legame non porta informazione.

## 4. Parte C — Corpo della prima candela

Con massimo e minimo scelti nella Parte A: **K2** (attuale) e **K5** (ibrida: se l'etichetta non è neutra, chiusura nella direzione dell'etichetta con l'ampiezza della K4; se è neutra, come K2). In entrambe il corpo è tagliato dentro
l'intervallo minimo-massimo previsto: se apertura o chiusura escono, si accorcia il corpo e gli estremi non si spostano.
Regola: K5 se l'errore sulla chiusura della candela è significativamente più basso (DM, p < 0,05) e la sovrapposizione non è significativamente più bassa; altrimenti K2. Applicata a F1 su tutte le origini, come nella Fase 4b.

{t_c}

Per etichetta (F1, tutte le origini):

{t_ce}

- {esito_c} Errore sulla chiusura: K5 {n(pc_['mae_chiusura']['K5'], 3)}, K2 {n(pc_['mae_chiusura']['K2'], 3)} (DM = {n(pc_['dm_chiusura'], 2)}, p {pv(pc_['p_chiusura'])}).
- La sovrapposizione è identica nelle due costruzioni ({n(pc_['sovrapposizione']['K2'], 3)}): con gli estremi che comandano dipende solo da massimo e minimo, non dal corpo.
- **Il taglio conta.** Con K2 la chiusura della candela esce sotto il minimo previsto e viene riportata sul minimo nel {pc(info['tagli']['F1']['K2']['chiusura'], 0)} dei casi (F1); l'apertura viene tagliata nel {pc(info['tagli']['F1']['K2']['apertura'], 1)}.
  In quei casi la candela non ha ombra inferiore. Minimo sopra il massimo: {info['tagli']['F1']['K2']['estremi_scambiati']} casi.
- Colore secondo l'etichetta, grigio se neutra, come nella Fase 4b.

![Verifica sul passato, F1](figures/fase4c/g3_verifica_passato.png)

{info['g3']['riga']}

## 5. Parte D — Pagella, integrazione, congelamento

### 5.1 Prima candela: prima e dopo — F1, tutte le origini

{prima_dopo('F1', 'tutte le origini')}

### 5.2 F1, origini in cui esiste l'etichetta di FS ({info['n_etichetta_fs']}, dal {info['prima_origine_etichetta_fs']})

{prima_dopo('F1', CE)}

### 5.3 FS, stesse origini

{prima_dopo('FS', CE)}

### 5.4 Integrazione

- **F1** resta solo storico del VIX: massimo con {ins['f1']['massimo']}, minimo con {ins['f1']['minimo']} (scelti tra S0 e S1). **FS**: massimo {ins['fs']['massimo']}, minimo {ins['fs']['minimo']} (scelti tra tutti).
- **`--aggiorna`** scarica le serie giornaliere che servono a FS (ora: {', '.join(info['serie_fs']) or 'nessuna'}) e le usa solo fino all'ultima seduta della settimana di origine. Se una serie dell'insieme degli estremi manca o è più vecchia dell'origine,
  il programma avvisa e per massimo e minimo ripiega sull'insieme di F1; se manca VIX3M, FS ripiega su F1 come prima. I valori usati vengono salvati accanto alla previsione.
- **Congelamento.** `python -m src.fase4c.congela` riscrive `versioni/f1.json` e `versioni/fs.json` (impronte dei sorgenti comprese quelle di `src/fase4c/`). Nome della costruzione nel registro: `{info['costruzione_nel_registro']}`.
- **Piano d'esame.** `reports/piano_esame_holdout.md` è aggiornato: nel criterio D della prima candela entrano la quota entro il 5% del massimo (D4) e del minimo (D5), e le tabelle sono previste con e senza strappi.
  I piani delle Fasi 4 e 4b sono in archivio (`reports/fase4/`, `reports/fase4b/`).

## 6. Quanto aggiungono VVIX, S&P 500 e previsione incrociata

Sulle origini comuni ({n_com}), rispetto a S0:

- **VVIX (S2): {verdetto('S2')}.** {frase_ins('S2')}.
- **Ampiezza dell'S&P 500 (S3): {verdetto('S3')}.** {frase_ins('S3')}.
- **VIX3M/VIX (S4): {verdetto('S4')}.** {frase_ins('S4')}.
- **Tutte insieme (S5): {verdetto('S5')}.** {frase_ins('S5')}.
- **Previsione incrociata reale (B3): {verdetto('B3')}.** {frase_ins('B3')}.
- **Candela del VIX (S1):** {frase_ins('S1')}. Su tutte le origini, per il minimo, passa la regola di F1.
- **Tetto teorico (B2):** {f_sopr}. È l'unico confronto in cui i numeri si muovono davvero, e non è realizzabile.

## 7. Problemi e ambiguità

1. **La previsione diretta del massimo non migliora.** È stata adottata perché lo chiede la decisione («anche il massimo diventa una previsione diretta»), ma S0 fa {pc(A.loc[('tutte le origini', 'tutte', 'S0', 'massimo'), 'entro_5'])} entro il 5% contro
   {pc(A.loc[('tutte le origini', 'tutte', 'PROG', 'massimo'), 'entro_5'])} del programma di prima e {pc(A.loc[('tutte le origini', 'tutte', 'FISSA', 'massimo'), 'entro_5'])} della regola fissa. Il massimo sbaglia soprattutto per difetto ({pc(A.loc[('tutte le origini', 'tutte', 'S0', 'massimo'), 'oltre_5_per_difetto'])} dei casi): i picchi dentro la settimana non si vedono dall'origine.
2. **F1 e FS con insiemi diversi (da decidere).** {f_diverse} È l'effetto della correzione per i confronti multipli su un campione più corto, non un segnale che S1 funzioni peggio con più dati. Un'alternativa ragionevole è far ereditare a FS l'insieme di F1; la regola è stata applicata come scritta.
3. **Su quali origini si applica la regola.** «Stesse origini» è stato letto così: F1 su tutte le origini (S0 e S1 esistono ovunque), FS sulle origini comuni a tutti gli insiemi. Holm è applicato dentro ogni famiglia e per ciascun estremo.
4. **Strappi.** Etichetta a posteriori ({info['n_strappi']['tutte le origini']} settimane su {n_tutte}); toglierle alza tutte le quote, soprattutto quella del massimo. Non è una misura di ciò che il modello sa fare in anticipo.
5. **Il corpo viene tagliato spesso.** Con il minimo che comanda, la chiusura K2 finisce sul minimo previsto nel {pc(info['tagli']['F1']['K2']['chiusura'], 0)} dei casi: corpo K2 (media delle settimane, verso il basso) e minimo (mediana) non sono coerenti tra loro.
   Il corpo previsto scende al {pc(E('F1', 'tutte le origini', 'tutte', 'dopo', 'rapporto_corpo'), 0)} del reale e la chiusura reale cade nell'escursione prevista nel {pc(E('F1', 'tutte le origini', 'tutte', 'dopo', 'chiusura_nell_escursione'))} dei casi.
6. **Definizioni dove la consegna lasciava margine.** Media a 52 settimane del VVIX = media dei valori giornalieri negli ultimi 364 giorni (almeno 200); escursione settimanale dell'S&P 500 = ln(max/min) delle chiusure giornaliere dentro la settimana;
   regressioni dell'S&P 500 in B3 = minimi quadrati, stessa finestra di 520 settimane; terzili del VVIX calcolati sul campione.
7. **«Migliore della Parte A» per la chiusura.** La Parte A riguarda massimo e minimo; per la chiusura la tabella B mostra l'insieme S1-S5 con l'errore più basso sulle origini comuni ({info['migliori_per_tabella_b']['chiusura']}), solo come misura.
8. **Regola del corpo applicata a F1.** Come nella Fase 4b. Sulle origini con l'etichetta di FS: K5 {n(info['parte_c_fs']['mae_chiusura']['K5'], 3)} contro K2 {n(info['parte_c_fs']['mae_chiusura']['K2'], 3)} (p {pv(info['parte_c_fs']['p_chiusura'])}).
9. **`--aggiorna` e insiemi scelti.** Gli insiemi scelti non usano VVIX né S&P 500, quindi oggi il programma scarica solo ciò che serviva già (VIX e VIX3M per FS). Il ripiego sugli insiemi di F1 è programmato e provato da un test, ma con queste scelte non scatta.
10. **Fasi precedenti.** Il piano della Fase 4b è ora scritto in `reports/fase4b/piano_esame_holdout_fase4b.md` (contenuto identico); nessun'altra modifica al codice delle fasi precedenti.
11. **Registro.** Le previsioni f1 e fs già registrate per un'origine non vengono riscritte: conservano la prima candela di allora (colonna `costruzione`).
12. **Molti confronti sullo stesso campione.** Sette fasi sullo stesso campione di sviluppo: i piccoli guadagni vanno letti con cautela, e l'holdout dirà se reggono.
"""
    (C.REPORTS_DIR / "fase4c_report.md").write_text(out)


if __name__ == "__main__":
    run()
