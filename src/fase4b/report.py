"""Genera reports/fase4b_report.md. Tutti i numeri sono letti dai file reports/fase4b/*.csv e *.json: nessun numero e' scritto a mano."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C
from ..fase3.report import MIN_N, ic, md, n, pc, pct
from . import prima as PR
from . import versioni as VS

R = C.REPORTS_DIR / "fase4b"
MISURE = {"n": "origini", "brier_direzione": "direzione: Brier", "quota_non_neutre": "etichetta: quota di non neutre", "n_non_neutre": "etichetta: non neutre (numero)",
          "accuratezza_non_neutre": "etichetta: corrette tra le non neutre", "n_rialzo": "etichette «rialzo» (numero)", "accuratezza_rialzo": "corrette tra i «rialzo»",
          "n_ribasso": "etichette «ribasso» (numero)", "accuratezza_ribasso": "corrette tra i «ribasso»", "mae_apertura": "apertura: errore medio (punti)",
          "mae_massimo": "massimo: errore medio (punti)", "massimo_reale_sopra_previsto": "massimo reale sopra il previsto", "mae_minimo": "minimo: errore medio (punti)",
          "mediana_errore_minimo": "minimo: errore mediano (punti)", "minimo_entro_5": "minimo entro il 5% dal reale", "minimo_entro_10": "minimo entro il 10% dal reale",
          "minimo_troppo_basso": "minimi previsti troppo bassi", "mae_chiusura": "chiusura della candela: errore medio (punti)", "mediana_errore_chiusura": "chiusura della candela: errore mediano (punti)",
          "sovrapposizione": "sovrapposizione con la candela reale", "chiusura_nel_corpo": "chiusura reale nel corpo previsto", "chiusura_nell_escursione": "chiusura reale nell'escursione prevista",
          "direzione_corpo": "direzione del corpo indovinata", "rapporto_corpo": "corpo previsto / reale", "rapporto_escursione": "escursione prevista / reale"}
INTERI = {"n", "n_non_neutre", "n_rialzo", "n_ribasso"}
QUOTE = {"quota_non_neutre", "accuratezza_non_neutre", "accuratezza_rialzo", "accuratezza_ribasso", "massimo_reale_sopra_previsto", "minimo_entro_5", "minimo_entro_10", "minimo_troppo_basso",
         "chiusura_nel_corpo", "chiusura_nell_escursione", "direzione_corpo"}


def pv(p: float) -> str:
    return "< 0,0001" if p < 0.0001 else "= " + n(p, 4)


def val(k: str, v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{int(v)}" if k in INTERI else pc(v) if k in QUOTE else n(v, 3)


def run() -> None:
    T = lambda k: pd.read_csv(R / f"{k}.csv")
    info = json.loads((R / "info.json").read_text())
    ta, cal, tc, te, td, pd_tab = T("a_candidati"), T("a_calibrazione"), T("c_pagella"), T("c_pagella_per_etichetta"), T("d_minimo"), T("prima_e_dopo")
    sc, b, c, d, alt = info["scelte"], info["parte_b"], info["parte_c"], info["parte_d"], info["parte_d_con_altra_costruzione"]
    cand, base, ks, mn = sc["candidato"], b["base"].upper(), sc["prima_candela"], sc["minimo"]
    camp = info["campioni"]
    A = ta[ta["gruppo"] == "tutte"].set_index("candidato")
    P = pd_tab.set_index(["campione", "versione", "stato"])
    sgn = lambda x, dd=3: f"{x:+.{dd}f}".replace(".", ",")

    # ------------------------------------------------------------------ prima e dopo
    def prima_dopo(colonne: list[tuple[str, tuple]]) -> str:
        rows = []
        for k, lab in MISURE.items():
            rows.append({"misura": lab} | {nome: val(k, P.loc[chiave, k]) for nome, chiave in colonne})
        return pd.DataFrame(rows).to_markdown(index=False)

    t_pd1 = prima_dopo([("F1 prima", ("tutte le origini", "F1", "prima")), ("F1 dopo", ("tutte le origini", "F1", "dopo"))])
    t_pd2 = prima_dopo([("F1 prima", ("origini comuni", "F1", "prima")), ("F1 dopo", ("origini comuni", "F1", "dopo")), (f"{base} prima (Fase 4)", ("origini comuni", base, "prima")),
                        ("FS dopo", ("origini comuni", "FS", "dopo"))])
    t_pd3 = prima_dopo([("F1 dopo", ("origini comuni, fascia 13-22", "F1", "dopo")), ("FS dopo", ("origini comuni, fascia 13-22", "FS", "dopo"))])
    F0, F1d = P.loc[("tutte le origini", "F1", "prima")], P.loc[("tutte le origini", "F1", "dopo")]
    C1, CS, CB = P.loc[("origini comuni", "F1", "dopo")], P.loc[("origini comuni", "FS", "dopo")], P.loc[("origini comuni", base, "prima")]

    # ------------------------------------------------------------------ Parte A
    def tab_a(g: pd.DataFrame, con_test: bool) -> str:
        g = g.copy()
        g["candidato "] = g["candidato"].map(VS.DESCRIZIONI)
        g["casi"] = [f"{k}" + (" (meno di 20)" if k < MIN_N else "") for k in g["n"]]
        g["non neutre: quota (n) → corrette"] = [f"{pc(q)} ({int(k)}) → {pc(a) if np.isfinite(a) else '–'}" for q, k, a in zip(g["quota_non_neutre"], g["n_non_neutre"], g["accuratezza_non_neutre"])]
        g["«rialzo»: n → corrette"] = [f"{int(k)} → {pc(a) if np.isfinite(a) else '–'}" for k, a in zip(g["n_rialzo"], g["accuratezza_rialzo"])]
        g["«ribasso»: n → corrette"] = [f"{int(k)} → {pc(a) if np.isfinite(a) else '–'}" for k, a in zip(g["n_ribasso"], g["accuratezza_ribasso"])]
        cols = ["candidato ", "casi", "brier", "affidabilita", "risoluzione", "non neutre: quota (n) → corrette", "«rialzo»: n → corrette", "«ribasso»: n → corrette"]
        if "gruppo " in g.columns:
            cols = ["gruppo "] + cols
        if con_test:
            g["guadagno sul Brier vs P1 [IC 95%]"] = [ic(x, lo, hi) if np.isfinite(x) and np.isfinite(lo) else (pct(x) if np.isfinite(x) else "–") for x, lo, hi in zip(g["brier_skill_vs_p1"], g["skill_ci_lo"], g["skill_ci_hi"])]
            cols += ["guadagno sul Brier vs P1 [IC 95%]", "dm_stat", "p"] + (["p_holm"] if g["p_holm"].notna().any() else [])
        else:
            g["guadagno sul Brier vs P1"] = [pct(x) if np.isfinite(x) else "–" for x in g["brier_skill_vs_p1"]]
            cols += ["guadagno sul Brier vs P1"]
        return md(g[cols].rename(columns={"candidato ": "candidato", "gruppo ": "fascia di partenza", "brier": "Brier", "affidabilita": "affidabilità (0 = calibrato)", "dm_stat": "DM vs P1",
                                          "p": "p (due code)", "p_holm": "p Holm"}), ".4f")

    t_a = tab_a(ta[ta["gruppo"] == "tutte"], True)
    t_a13 = tab_a(ta[ta["gruppo"] == "fascia 13-22"], True)
    fas = ta[ta["gruppo"].str.startswith("fascia ") & (ta["gruppo"] != "fascia 13-22")].copy()
    fas["gruppo "] = fas["gruppo"].str.replace("fascia ", "")
    t_af = tab_a(fas, False)
    cc = cal[cal["candidato"].isin(["P1", cand])].copy()
    cc["candidato"] = cc["candidato"].map(VS.DESCRIZIONI)
    t_cal = md(cc.rename(columns={"classe": "classe di probabilità", "prob_media": "probabilità media", "freq_osservata": "frequenza osservata"}), ".3f")
    sig_a = [k for k in VS.SFIDANTI if A.loc[k, "p_holm"] < 0.05 and A.loc[k, "dm_stat"] > 0]

    # ------------------------------------------------------------------ Parte C
    x = tc.copy()
    x["sovrapposizione [IC 95%]"] = [f"{v:.3f} [{lo:.3f}; {hi:.3f}]" for v, lo, hi in zip(x["sovrapposizione"], x["sovrapposizione_ci_lo"], x["sovrapposizione_ci_hi"])]
    t_c = md(x[["versione", "campione", "costruzione", "n", "mae_apertura", "mae_massimo", "mae_minimo", "mae_chiusura", "mediana_errore_chiusura", "sovrapposizione [IC 95%]", "chiusura_nel_corpo",
                "chiusura_nell_escursione", "direzione_corpo", "rapporto_corpo", "rapporto_escursione"]].rename(columns={
        "mae_apertura": "errore medio apertura", "mae_massimo": "errore medio massimo", "mae_minimo": "errore medio minimo", "mae_chiusura": "errore medio chiusura",
        "mediana_errore_chiusura": "errore mediano chiusura", "chiusura_nel_corpo": "chiusura reale nel corpo", "chiusura_nell_escursione": "chiusura reale nell'escursione",
        "direzione_corpo": "direzione del corpo", "rapporto_corpo": "corpo previsto / reale", "rapporto_escursione": "escursione prevista / reale"}), ".3f")
    y = te.copy()
    y["casi"] = [f"{k}" + (" (meno di 20)" if k < MIN_N else "") for k in y["n"]]
    t_ce = md(y[["etichetta", "costruzione", "casi", "mae_minimo", "mae_chiusura", "sovrapposizione", "chiusura_nel_corpo", "chiusura_nell_escursione", "direzione_corpo", "rapporto_corpo"]].rename(columns={
        "mae_minimo": "errore medio minimo", "mae_chiusura": "errore medio chiusura", "chiusura_nel_corpo": "chiusura reale nel corpo", "chiusura_nell_escursione": "chiusura reale nell'escursione",
        "direzione_corpo": "direzione del corpo", "rapporto_corpo": "corpo previsto / reale"}), ".3f")
    TC = tc.set_index(["versione", "campione", "costruzione"])
    fs_sov = {k: TC.loc[("FS", "origini comuni", k), "sovrapposizione"] for k in ("K2", "K4")}
    f1c_sov = {k: TC.loc[("F1", "origini comuni", k), "sovrapposizione"] for k in ("K2", "K4")}
    esito_c = ("**K4 è adottata** per la settimana +1." if ks == "K4" else
               "**Resta K2**, con il solo colore legato all'etichetta: " + ("la sovrapposizione media di K4 è più bassa di quella di K2" if not c["sovrapposizione_non_piu_bassa"] else "")
               + (" e " if not c["sovrapposizione_non_piu_bassa"] and c["chiusura_significativamente_peggiore"] else "")
               + ("l'errore sulla chiusura di K4 è significativamente peggiore" if c["chiusura_significativamente_peggiore"] else "") + ".")
    chius = ("più basso in modo significativo" if c["dm_stat"] > 0 and c["p"] < 0.05 else "significativamente peggiore" if c["chiusura_significativamente_peggiore"] else "non distinguibile")

    # ------------------------------------------------------------------ Parte D
    z = td.copy()
    t_d = md(z.rename(columns={"errore_assoluto_medio": "errore assoluto medio (punti)", "errore_assoluto_mediano": "errore assoluto mediano", "entro_5": "entro il 5%", "entro_10": "entro il 10%",
                               "quota_troppo_bassi": "previsti troppo bassi", "errore_medio_con_segno": "errore medio con segno (previsto − reale)",
                               "quota_limitati_dal_corpo": "limitati dal corpo"}), ".3f")
    D = td[td["campione"] == "tutte le origini"].set_index("variante")
    lim = D.loc["M1q", "quota_limitati_dal_corpo"]

    # ------------------------------------------------------------------ sintesi (10 righe)
    sintesi = [
        f"**Direzione della prima candela (Parte A).** Sulle {int(A.loc['P1', 'n'])} origini in cui esistono tutti i candidati (dal {camp['origini_comuni']['prima_data_origine']}), Brier: "
        + "; ".join(f"{k} {n(A.loc[k, 'brier'], 4)}" for k in VS.CANDIDATI) + f". Significativamente migliori di P1 dopo Holm: {', '.join(sig_a) if sig_a else 'nessuno'}.",
        f"**Sfidante (Parte B).** Il Brier più basso tra P2-P5 è di **{VS.DESCRIZIONI[cand]}** ({n(b['brier'][cand], 4)} contro {n(b['brier']['P1'], 4)} di P1, {ic(A.loc[cand, 'brier_skill_vs_p1'], A.loc[cand, 'skill_ci_lo'], A.loc[cand, 'skill_ci_hi'])}). "
        f"FS usa quella probabilità per la settimana +1 e la costruzione di {base} per chiusura centrale, fasce e settimane +2…+4.",
        f"**Etichette di FS a +1** (origini comuni): non neutre nel {pc(CS['quota_non_neutre'])} dei casi, corrette nel {pc(CS['accuratezza_non_neutre'])}; F1: {pc(C1['quota_non_neutre'])} e {pc(C1['accuratezza_non_neutre'])}; "
        f"{base} da sola: {pc(CB['quota_non_neutre'])} e {pc(CB['accuratezza_non_neutre'])}. I «rialzo» restano pochi ({int(CS['n_rialzo'])} per FS).",
        f"**Corpo della prima candela (Parte C).** {esito_c} Sovrapposizione: K4 {n(c['sovrapposizione']['K4'], 3)} contro K2 {n(c['sovrapposizione']['K2'], 3)}; "
        f"errore medio sulla chiusura: K4 {n(c['mae_chiusura']['K4'], 2)} contro K2 {n(c['mae_chiusura']['K2'], 2)} punti ({chius}, p {pv(c['p'])}).",
        (f"**Attenzione, da decidere:** la regola respinge K4 per la sola sovrapposizione, mentre sulla chiusura — la prima delle priorità dichiarate — K4 è migliore. "
         f"Con l'etichetta di FS sulle origini comuni K4 avrebbe anche la sovrapposizione più alta ({n(fs_sov['K4'], 3)} contro {n(fs_sov['K2'], 3)}). La regola è stata applicata come scritta, a F1 su tutte le origini."
         if ks == "K2" and c["dm_stat"] > 0 and c["p"] < 0.05 else f"**Parte C, regola:** applicata a F1 su tutte le sue origini ({camp['tutte_le_origini']['n']}); la stessa costruzione vale per FS."),
        f"**Minimo della prima candela (Parte D).** La regola sceglie **{mn}**: errore assoluto medio {n(d['mae']['M1q'], 2)} punti contro {n(d['mae']['M0'], 2)} di M0 (DM p {pv(d['p'])}); "
        f"minimi entro il 5% dal reale {pc(D.loc['M1q', 'entro_5'])} contro {pc(D.loc['M0', 'entro_5'])}; previsti troppo bassi {pc(D.loc['M1q', 'quota_troppo_bassi'])} contro {pc(D.loc['M0', 'quota_troppo_bassi'])}.",
        f"**Prezzo del nuovo minimo.** L'escursione prevista si stringe (rapporto con la reale da {n(F0['rapporto_escursione'], 3)} a {n(F1d['rapporto_escursione'], 3)}) e la chiusura reale cade dentro l'escursione prevista meno spesso "
        f"(dal {pc(F0['chiusura_nell_escursione'])} al {pc(F1d['chiusura_nell_escursione'])}). Nel {pc(lim, 0)} dei casi il minimo di M1q starebbe sopra il corpo e viene portato al fondo del corpo.",
        f"**Massimo.** Non ricalibrato, come deciso: il massimo reale supera quello previsto nel {pc(F1d['massimo_reale_sopra_previsto'])} dei casi; errore medio {n(F1d['mae_massimo'], 2)} punti.",
        f"**Versioni congelate di nuovo:** `versioni/f1.json` e `versioni/fs.json` (prima candela {ks}{' + ' + mn if mn == 'M1q' else ''}, colore secondo l'etichetta). "
        + ("Con K2 la prima candela ha gli stessi quattro valori in F1 e in FS: tra le due versioni cambiano solo etichetta e colore." if ks == "K2" else "Con K4 la prima candela dipende dall'etichetta e dalla chiusura centrale della versione."),
        "**Piano d'esame aggiornato:** due versioni (F1 e FS), i tre criteri di prima più un criterio dedicato alla prima candela (direzione quando non neutra, minimo, chiusura). **L'holdout non è stato aperto.**",
    ]

    out = f"""# Fase 4b — La prima candela: direzione, corpo, minimo; sfidante e nuovo congelamento

*Generato da `python -m src.run_phase4b`. Campione di sviluppo fino al {camp['tutte_le_origini']['ultima_data_obiettivo']}. L'holdout non è stato letto: il comando gira sotto la guardia.
Tutti i numeri vengono da `reports/fase4b/`. I modelli delle versioni non sono stati ricalcolati: probabilità, chiusure centrali e candele K2 vengono dai file della Fase 4.*

**Decisioni applicate:** la previsione si fa una sola volta, alla chiusura settimanale; priorità a chiusura, minimo e direzione della candela +1; il massimo non si ricalibra; all'esame vanno F1 e una sola sfidante.

## Sintesi

{chr(10).join(f'{i}. {s}' for i, s in enumerate(sintesi, start=1))}

## 1. Prima candela: prima e dopo

«Prima» = stato della Fase 4 (candela K2 con ampiezze {sc['variante']}, minimo della candela, etichetta della versione). «Dopo» = scelte di questa fase: costruzione {ks}, minimo {mn}, colore secondo l'etichetta.
Errori in punti VIX. Tutte le misure riguardano la settimana +1.

### 1.1 F1 su tutte le sue origini ({camp['tutte_le_origini']['n']}, date previste dal {camp['tutte_le_origini']['prima_data_obiettivo']} al {camp['tutte_le_origini']['ultima_data_obiettivo']})

{t_pd1}

### 1.2 Origini comuni a tutti i candidati ({camp['origini_comuni']['n']}, origini dal {camp['origini_comuni']['prima_data_origine']} al {camp['origini_comuni']['ultima_data_origine']})

La colonna «{base} prima» è la sfidante com'era nella Fase 4 (costruzione di {base} con la sua etichetta); «FS dopo» è la sfidante congelata ora.

{t_pd2}

### 1.3 Fascia 13-22, origini comuni ({camp['origini_comuni']['n_fascia_13_22']})

{t_pd3}

## 2. Parte A — Direzione della prima candela

Probabilità che la chiusura della settimana +1 sia sopra la chiusura all'origine. Candidati (elenco chiuso): {'; '.join(VS.DESCRIZIONI.values())}.
Stesse origini per tutti: quelle in cui esistono le tre probabilità ({camp['origini_comuni']['n']}; l'etichetta di F2 richiede 104 errori passati, quindi si parte dal {camp['origini_comuni']['prima_data_origine']}).
Etichetta non neutra quando la probabilità è almeno 60% in una direzione. Test di Diebold-Mariano sul Brier contro P1, Holm sui {info['n_test_holm_parte_a']} confronti.

{t_a}

Per fascia di partenza (numero di casi in colonna; con meno di 20 casi il dato è solo indicativo):

{t_af}

Fascia 13-22 (p non corretti per i confronti multipli):

{t_a13}

Calibrazione di P1 e del candidato scelto (tutte le classi in `reports/fase4b/a_calibrazione.csv`):

{t_cal}

## 3. Parte B — Scelta della sfidante

Regola fissata nella consegna: la sfidante è costruita attorno al candidato con il Brier più basso a h = 1 tra P2, P3, P4 e P5; per chiusura centrale, fasce e orizzonti +2…+4 usa la costruzione di F2 se il candidato contiene P2, altrimenti quella di F3.

- Brier: {'; '.join(f'{k} {n(b["brier"][k], 4)}' for k in VS.CANDIDATI)}.
- **Candidato scelto: {VS.DESCRIZIONI[cand]}.** Costruzione di base: {base}.
- **FS**: per la settimana +1 la probabilità che la chiusura superi quella dell'origine è la media semplice delle probabilità di {' e '.join(m.upper() for m in b['componenti'])}; chiusura centrale, fasce, probabilità di superare 20, 25 e 30,
  etichette e candele delle settimane +2…+4 sono quelle di {base}. Se VIX3M manca all'origine, FS non è calcolabile e il programma ripiega su F1.
- Il margine tra i candidati migliori è piccolo ({n(b['brier'][cand], 4)} contro {n(min(v for k, v in b['brier'].items() if k not in (cand, 'P1')), 4)} del secondo): la regola sceglie il più basso senza un test tra candidati.

## 4. Parte C — Corpo e colore della prima candela

**K4** (solo settimana +1): apertura = chiusura all'origine × rapporto di apertura; se l'etichetta è «ribasso» chiusura = chiusura all'origine × exp(−d), con d = media delle variazioni log assolute delle settimane in calo
nelle ultime {PR.FINESTRA_MOSSE} settimane; se è «rialzo» chiusura = chiusura all'origine × exp(+u), con u = media delle variazioni log delle settimane in rialzo; se è «neutra» chiusura = chiusura centrale della versione; ombre come ora.
Nel campione d è stato tra {pc(info['mosse']['d_min_max'][0])} e {pc(info['mosse']['d_min_max'][1])}, u tra {pc(info['mosse']['u_min_max'][0])} e {pc(info['mosse']['u_min_max'][1])}.

**Regola** (dalla consegna): K4 se la sovrapposizione media non è più bassa di quella di K2 e l'errore sulla chiusura della candela non è significativamente peggiore (DM, p < 0,05); altrimenti K2 con il solo colore legato all'etichetta.
Fissato prima di guardare i risultati: la regola si applica a F1 su tutte le sue origini (il campione più lungo) e la costruzione scelta vale anche per FS.

{t_c}

K4 e K2 per etichetta (F1, tutte le origini; etichette di F1: {', '.join(f'{k} {v}' for k, v in info['etichette_f1'].items())}):

{t_ce}

- **Esito.** {esito_c}
- Errore sulla chiusura della candela: K4 {n(c['mae_chiusura']['K4'], 3)} punti, K2 {n(c['mae_chiusura']['K2'], 3)} (DM = {n(c['dm_stat'], 2)}, p {pv(c['p'])}: {chius}).
- Sovrapposizione: K4 {n(c['sovrapposizione']['K4'], 3)}, K2 {n(c['sovrapposizione']['K2'], 3)}. K4 ha un corpo molto più piccolo ({n(TC.loc[('F1', 'tutte le origini', 'K4'), 'rapporto_corpo'], 2)} del reale contro {n(TC.loc[('F1', 'tutte le origini', 'K2'), 'rapporto_corpo'], 2)}):
  quando l'etichetta è neutra — la maggior parte dei casi — la chiusura va sulla chiusura centrale, vicina all'origine, e la candela copre meno della settimana reale.
- Sulle origini comuni: con l'etichetta di F1 la sovrapposizione è {n(f1c_sov['K4'], 3)} (K4) contro {n(f1c_sov['K2'], 3)} (K2); con l'etichetta di FS è {n(fs_sov['K4'], 3)} contro {n(fs_sov['K2'], 3)}.
- **Colore.** La prima candela prevista è piena e colorata secondo l'etichetta: rossa «rialzo», blu «ribasso», grigia «neutra», con bordo tratteggiato perché è una previsione. Le candele +2…+4 restano vuote e tratteggiate. La legenda è aggiornata.
  {'Con K2 il corpo resta disegnato verso il basso: una candela rossa (etichetta «rialzo») può avere la chiusura sotto l' + chr(39) + 'apertura. Il colore dice la direzione della chiusura rispetto all' + chr(39) + 'origine, non il verso del corpo.' if ks == 'K2' else ''}

![Verifica sul passato, F1](figures/fase4b/g3_verifica_passato.png)

{info['g3']['riga']}

## 5. Parte D — Minimo della prima candela

Varianti (elenco chiuso): **M0** = minimo della candela scelta nella Parte C ({ks}: fondo del corpo meno l'ombra inferiore media); **M1q** = regressione sulla mediana di ln(minimo della settimana +1 / chiusura all'origine)
sulle variabili di M1, stimata sulle ultime {PR.FINESTRA_MINIMO} settimane. Il minimo mostrato non può stare sopra il corpo: se M1q è più alto del fondo del corpo, vale il fondo del corpo («M1q senza vincolo» è riportato per confronto).
Regola: M1q se l'errore assoluto medio è più basso con DM p < 0,05; altrimenti M0. Applicata a F1 su tutte le sue origini.

{t_d}

- **Esito: {mn}.** Errore assoluto medio {n(d['mae']['M1q'], 3)} contro {n(d['mae']['M0'], 3)} (DM = {n(d['dm_stat'], 2)}, p {pv(d['p'])}).
- M0 mette il minimo troppo in basso nel {pc(D.loc['M0', 'quota_troppo_bassi'])} dei casi (errore medio con segno {sgn(D.loc['M0', 'errore_medio_con_segno'])} punti); M1q nel {pc(D.loc['M1q', 'quota_troppo_bassi'])} ({sgn(D.loc['M1q', 'errore_medio_con_segno'])}).
- Nel {pc(lim, 0)} dei casi M1q è limitato dal corpo: {'il corpo K2, disegnato verso il basso con ampiezza media, arriva più in basso del minimo previsto, e allora minimo e chiusura della candela coincidono (nessuna ombra inferiore).' if ks == 'K2' else 'il corpo arriva più in basso del minimo previsto, e allora minimo e chiusura della candela coincidono.'}
- Se la Parte C avesse scelto {alt['costruzione']}, la stessa regola avrebbe dato {alt['scelta']} (errore medio {n(alt['mae']['M1q'], 3)} contro {n(alt['mae']['M0'], 3)}, p {pv(alt['p'])}).

## 6. Scelte fatte dalle regole, congelamento, programma

| regola | scelta |
|---|---|
| Parte B — candidato della sfidante | {VS.DESCRIZIONI[cand]}; costruzione di base {base} |
| Parte C — prima candela | {ks}{' (K4 respinta)' if ks == 'K2' else ''}, colore secondo l'etichetta |
| Parte D — minimo | {mn} |
| Fase 4, Parte A — ampiezze delle candele | {sc['variante']} (invariata) |

- **Congelamento.** `python -m src.fase4b.congela` scrive `versioni/f1.json` e `versioni/fs.json` con parametri, finestre, regole, scelte, impronta SHA-256 dei sorgenti (compresi `src/fase4b/prima.py` e `src/fase4b/versioni.py`) e commit.
  I file `f2.json` e `f3.json` della Fase 4 sono stati tolti: F2 e F3 non vanno all'esame come versioni a sé, restano come componenti di FS (e nella storia del repository).
- **Programma.** `python -m src.forecast --versione f1|fs` (predefinita f1); `--aggiorna` scarica VIX e VIX3M giornalieri, che servono a FS; se VIX3M manca il programma avvisa e ripiega su f1. Nome della costruzione nel registro: `{info['costruzione_nel_registro']}`.
  La scheda mostra la versione richiesta e il confronto con l'altra. La previsione non viene aggiornata durante la settimana.
- **Piano d'esame.** `reports/piano_esame_holdout.md` è stato riscritto: due versioni, criteri A (chiusura), B (direzione), C (fasce) come prima, più il criterio D sulla prima candela. Il piano della Fase 4 è conservato in `reports/fase4/piano_esame_holdout_fase4.md`.

## 7. Problemi e ambiguità

1. **K4 respinta dalla regola pur essendo migliore sulla chiusura (da decidere).** La regola chiede due condizioni; K4 perde sulla sovrapposizione ({n(c['sovrapposizione']['K4'], 3)} contro {n(c['sovrapposizione']['K2'], 3)}) e {'vince' if c['dm_stat'] > 0 else 'non vince'} sull'errore di chiusura
   ({n(c['mae_chiusura']['K4'], 2)} contro {n(c['mae_chiusura']['K2'], 2)} punti). La chiusura è la prima priorità dichiarata: se si preferisce K4 basta dirlo e va rifatto il congelamento.
2. **A quale versione si applica la regola della Parte C.** Fissato prima: a F1 su tutte le origini. Con l'etichetta di FS sulle origini comuni K4 avrebbe sovrapposizione {n(fs_sov['K4'], 3)} contro {n(fs_sov['K2'], 3)}; con quella di F1 sulle stesse origini {n(f1c_sov['K4'], 3)} contro {n(f1c_sov['K2'], 3)}.
3. **Colore e verso del corpo.** {'Con K2 il corpo è sempre disegnato verso il basso; il colore segue l' + chr(39) + 'etichetta. È ciò che la regola prevede nel caso «resta K2», ma una candela rossa con chiusura sotto l' + chr(39) + 'apertura può confondere: la guida alla lettura lo spiega.' if ks == 'K2' else 'Con K4 colore e direzione della chiusura rispetto all' + chr(39) + 'origine coincidono.'}
4. **Il nuovo minimo stringe la candela.** M1q riduce l'errore sul minimo ma l'escursione prevista scende al {pc(F1d['rapporto_escursione'], 0)} della reale e la chiusura reale cade nell'escursione prevista nel {pc(F1d['chiusura_nell_escursione'])} dei casi (prima {pc(F0['chiusura_nell_escursione'])}).
   È coerente con le priorità (minimo sì, massimo no), ma la candela «contiene» meno spesso la settimana reale.
5. **M1q limitato dal corpo** nel {pc(lim, 0)} dei casi: è un effetto del corpo K2, non del modello del minimo.
6. **«M0 attuale».** Letto come il minimo della candela scelta nella Parte C. Con l'altra costruzione l'esito è riportato nella sezione 5.
7. **Campione della Parte A.** Solo {camp['origini_comuni']['n']} origini (dal {camp['origini_comuni']['prima_data_origine']}), perché l'etichetta di F2 richiede 104 errori passati. Le fasce di partenza alte hanno pochi casi.
8. **Margine tra i candidati.** P2, P4 e P5 sono vicini; la scelta di {cand} segue la regola del Brier più basso, non un test.
9. **FS oltre la settimana +1.** Etichette e probabilità di livello delle settimane +2…+4 sono quelle di {base}: la consegna fissava solo chiusura centrale, fasce e orizzonti.
10. **Prima candela uguale nelle due versioni.** {'Con K2 e M1q apertura, massimo, minimo e chiusura della candela +1 non dipendono dalla versione: F1 e FS differiscono, sulla prima candela, solo per etichetta e colore. Nel criterio D dell' + chr(39) + 'esame i test su minimo e chiusura non saranno quindi calcolabili; il piano lo prevede.' if ks == 'K2' else 'Con K4 la candela +1 dipende dalla versione.'}
11. **Criterio D1 del piano.** La perdita che dà 0, 1 oppure 0,5 all'etichetta (corretta, sbagliata, neutra) è una scelta nostra per avere un test sulla «direzione quando non neutra»: da confermare.
12. **Versioni F2 e F3 e registro.** F2 e F3 sono tolte dalle versioni congelate e dal programma; le righe f2 e f3 già scritte nel registro restano. Una previsione f1 registrata prima di questa fase
    non viene riscritta: per quell'origine il registro conserva la prima candela di allora (colonna `costruzione`).
13. **Fasi precedenti.** Unica modifica al codice di una fase precedente: il piano della Fase 4 viene ora scritto in `reports/fase4/piano_esame_holdout_fase4.md` (contenuto identico), perché il piano in vigore è quello di questa fase.
    Al modulo dei grafici sono stati aggiunti parametri facoltativi (colore, legenda) che non cambiano i grafici delle fasi precedenti.
14. **Regole applicate a F1 e usate per FS.** Costruzione e minimo sono scelti una volta sola: un solo programma, una sola prima candela.
15. **Molti confronti sullo stesso campione.** Come nelle fasi precedenti: i guadagni misurati qui sono in parte attesi per selezione, e l'holdout dirà se reggono.
"""
    (C.REPORTS_DIR / "fase4b_report.md").write_text(out)


if __name__ == "__main__":
    run()
