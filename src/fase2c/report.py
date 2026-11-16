"""Genera reports/fase2c_report.md. Tutti i numeri sono letti dai file reports/fase2c/*.csv e *.json."""
from __future__ import annotations

import json
import re

import pandas as pd

from .. import config as C
from . import candele as KA
from . import componenti as CP

R = C.REPORTS_DIR / "fase2c"
MIN_N = 20


def md(df: pd.DataFrame, fmt=".3f") -> str:
    return re.sub(r"\bnan\b", "  –", df.to_markdown(index=False, floatfmt=fmt, missingval="–"))


def n(v, d: int = 2) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def pct(v, d: int = 1) -> str:
    return f"{100 * v:.{d}f}%".replace(".", ",")


def run() -> None:
    T = lambda k: pd.read_csv(R / f"{k}.csv")
    dr, dg, am, pg, pgg, en, g3, pv = (T(k) for k in ("direzione", "direzione_gruppi", "ampiezze", "pagella", "pagella_gruppi", "chiusura_entro", "g3_pannelli", "previsioni_dev"))
    info = json.loads((R / "info.json").read_text())
    sc, mdir = info["scelta"], {int(h): m for h, m in info["modello_direzione"].items()}
    K = sc["costruzione"]
    D, P, A = dr.set_index(["h", "modello"]), pg.set_index(["costruzione", "h"]), am.set_index(["componente", "h"])
    esc = info["settimane_escluse_dalle_stime"]
    quota_su = pv.assign(su=pv["p_scelta"] >= 0.5).groupby("h")["su"].mean()
    cal = T("direzione_calibrazione")
    cal = cal[(cal["h"] == 1) & (cal["modello"] == mdir[1])]
    cal_min, cal_max = cal["classe"].iloc[0].split("-")[0].replace(".", ","), cal["classe"].iloc[-1].split("-")[1].replace(".", ",")
    cal_quota = cal.loc[cal["classe"] == "0.3-0.4", "n"].sum() / cal["n"].sum()

    t_dir = md(dr.assign(scelto=dr["scelto"].map({True: "sì", False: ""}))[["h", "modello", "n", "frequenza_rialzo", "prob_media", "accuratezza", "brier", "affidabilita", "risoluzione",
                                                                             "brier_skill_vs_L0", "dm_stat", "p", "p_holm", "scelto"]].rename(columns={
        "frequenza_rialzo": "freq. rialzo", "prob_media": "prob. media", "affidabilita": "affidabilità", "brier_skill_vs_L0": "Brier skill vs L0", "dm_stat": "DM vs L0", "p_holm": "p Holm"}), ".4f")
    nomi_g = {"fascia_origine": "fascia di partenza", "rialzo_forte": "rialzo oltre il 30% in 4 settimane", "sottoperiodo": "sottoperiodo"}

    def dir_pivot(col):
        g = dg[(dg["gruppo"] == col) & dg["h"].isin([1, 4])]
        p = g.pivot_table(index=["h", "valore", "n", "frequenza_rialzo"], columns="modello", values="accuratezza", sort=False).reset_index()
        b = g.pivot_table(index=["h", "valore"], columns="modello", values="brier_skill_vs_L0", sort=False).reset_index()
        p["Brier skill L1 (%)"], p["Brier skill L2 (%)"] = 100 * b["L1"].to_numpy(), 100 * b["L2"].to_numpy()
        if col == "fascia_origine":
            p["valore"] = pd.Categorical(p["valore"], C.BAND_LABELS, ordered=True)
            p = p.sort_values(["h", "valore"])
        if col == "rialzo_forte":
            p["valore"] = p["valore"].map({"True": "sì", "False": "no"})
        p["n"] = [f"{k}" + (" (meno di 20)" if k < MIN_N else "") for k in p["n"]]
        return md(p.rename(columns={"valore": nomi_g[col], "frequenza_rialzo": "freq. rialzo", "L0": "accuratezza L0", "L1": "accuratezza L1", "L2": "accuratezza L2"}), ".3f")

    u = am.copy()
    u["guadagno sul riferimento [IC 95%]"] = [f"{a:+.4f} [{b:+.4f}; {c:+.4f}]" for a, b, c in zip(u["guadagno"], u["guadagno_ci_lo"], u["guadagno_ci_hi"])]
    u["guadagno (%)"] = 100 * u["guadagno_pct"]
    u["componente"] = u["componente"].map({"corpo": "A3 ampiezza del corpo", "ombra_sup": "A4 ombra superiore", "ombra_inf": "A4 ombra inferiore"})
    t_amp = md(u.sort_values(["componente", "h"])[["componente", "h", "n", "media_reale", "mediana_reale", "previsione_media", "mae_modello", "mae_riferimento",
                                                    "guadagno sul riferimento [IC 95%]", "guadagno (%)", "dm_stat", "p"]].rename(columns={
        "media_reale": "media reale", "mediana_reale": "mediana reale", "previsione_media": "previsione media", "mae_modello": "errore assoluto medio, modello",
        "mae_riferimento": "errore assoluto medio, mediana storica", "dm_stat": "DM"}), ".4f")
    amp_sig = [f"{r.componente.replace('_', ' ')} a h = {r.h}" for r in am.itertuples() if r.guadagno_ci_lo > 0]
    amp_neg = [f"{r.componente.replace('_', ' ')} a h = {r.h}" for r in am.itertuples() if r.guadagno < 0]

    q = pg.copy()
    q["costruzione"] = pd.Categorical(q["costruzione"], KA.COSTRUZIONI, ordered=True)
    q = q.sort_values(["h", "costruzione"])
    q["sovrapposizione [IC 95%]"] = [f"{a:.3f} [{b:.3f}; {c:.3f}]" for a, b, c in zip(q["sovrapposizione"], q["sovrapposizione_ci_lo"], q["sovrapposizione_ci_hi"])]
    t_p1 = md(q[["h", "costruzione", "n", "direzione_corpo", "direzione_chiusura_vs_origine", "rapporto_corpo", "rapporto_escursione", "sovrapposizione [IC 95%]",
                 "chiusura_nel_corpo", "chiusura_nell_escursione"]].rename(columns={"direzione_corpo": "direzione del corpo indovinata", "direzione_chiusura_vs_origine": "direzione della chiusura vs origine",
                                                                                     "rapporto_corpo": "corpo previsto / reale", "rapporto_escursione": "escursione prevista / reale",
                                                                                     "chiusura_nel_corpo": "chiusura reale nel corpo previsto", "chiusura_nell_escursione": "chiusura reale nell'escursione prevista"}), ".3f")
    t_p2 = md(q[["h", "costruzione", "mae_apertura", "mae_massimo", "mae_minimo", "mae_chiusura"]].rename(columns=lambda c: c.replace("mae_", "errore ")), ".2f")
    q2 = q.copy()
    for c in ("apertura", "massimo", "minimo", "chiusura"):
        q2[f"errore % {c}"] = 100 * q2[f"mae_pct_{c}"]
    t_p3 = md(q2[["h", "costruzione", "errore % apertura", "errore % massimo", "errore % minimo", "errore % chiusura"]], ".1f")
    e = en.copy()
    e["serie"] = pd.Categorical(e["serie"], ["mediana D4"] + list(KA.COSTRUZIONI), ordered=True)
    e["entro il 10% (%)"], e["entro il 20% (%)"] = 100 * e["entro_10pct"], 100 * e["entro_20pct"]
    t_en = md(e.sort_values(["h", "serie"])[["h", "serie", "n", "entro il 10% (%)", "entro il 20% (%)", "mae_punti"]].rename(columns={"mae_punti": "errore assoluto medio (punti)"}), ".1f")

    def pag_pivot(col, metrica, scala=1.0, dec=3):
        g = pgg[(pgg["gruppo"] == col) & pgg["h"].isin([1, 4])]
        p = (scala * g.pivot_table(index=["h", "valore", "n"], columns="costruzione", values=metrica, sort=False)[list(KA.COSTRUZIONI)]).reset_index()
        if col == "fascia_origine":
            p["valore"] = pd.Categorical(p["valore"], C.BAND_LABELS, ordered=True)
            p = p.sort_values(["h", "valore"])
        if col == "rialzo_forte":
            p["valore"] = p["valore"].map({"True": "sì", "False": "no"})
        return md(p.rename(columns={"valore": nomi_g[col]}), f".{dec}f")

    t_g3 = md(g3.assign(**{c: g3[c].map({True: "sì", False: "no"}) for c in ("dentro_50", "dentro_80", "chiusura_reale_4_nell_escursione_prevista")}).rename(columns={
        "data_origine": "origine", "chiusura_origine": "chiusura all'origine", "mediana_d4_4": "mediana D4 a +4", "candela_chiusura_4": "chiusura della candela a +4",
        "chiusura_reale_4": "chiusura reale a +4", "dentro_50": "nella fascia 50%", "dentro_80": "nella fascia 80%", "direzioni_indovinate_su_4": "direzioni indovinate su 4",
        "chiusura_reale_4_nell_escursione_prevista": "chiusura reale nell'escursione della candela +4"}), ".2f")
    acc = lambda h, m: D.loc[(h, m), "accuratezza"]
    best_err = {h: pg[pg["h"] == h].sort_values("mae_chiusura").iloc[0]["costruzione"] for h in CP.ORIZZONTI}
    iou = sc["sovrapposizione_media"]
    sp = sc.get("spareggio")
    rs = info["n_rialzo_forte"]["1"]
    gain_dir = [acc(h, mdir[h]) - acc(h, "L0") for h in CP.ORIZZONTI]
    sig_dir = dr[(dr["p_holm"] < 0.05)]

    out = f"""# Fase 2c — Candele previste realistiche e misurabili

*Report generato da `python -m src.run_phase2c`. Ogni numero è letto dai file `reports/fase2c/*.csv` e `*.json`. Solo campione di sviluppo
({info['campione'][0]} → {info['campione'][1]}), sotto la guardia che blocca l'holdout. Previsione corrente: [previsione_corrente.md](previsione_corrente.md).*

## 1. Sintesi

1. **Costruzione scelta: {KA.NOMI[K]}.** Nessuna delle quattro costruzioni rispetta il vincolo di realismo (corpo ed escursione tra 0,8 e 1,2 volte quelli reali a h = 1): è stata applicata la regola di riserva (rapporto del corpo più vicino a 1){', con uno spareggio che va confermato (sezione 4)' if sp else ''}.
2. **Il problema di partenza è risolto solo in parte.** Il corpo medio delle candele passa dal {pct(P.loc[('K0', 1), 'rapporto_corpo'], 0)} di quello reale (K0, Fase 2b) al {pct(P.loc[(K, 1), 'rapporto_corpo'], 0)} ({K}) a 1 settimana, e dal {pct(P.loc[('K0', 4), 'rapporto_corpo'], 0)} al {pct(P.loc[(K, 4), 'rapporto_corpo'], 0)} a 4 settimane. Restano più piccole del reale perché l'ampiezza è una mediana, e la mediana dei corpi ({n(A.loc[('corpo', 1), 'mediana_reale'], 3)} in log) è più bassa della media ({n(A.loc[('corpo', 1), 'media_reale'], 3)}).
3. **La direzione del corpo non è prevedibile.** Il modello scelto ({mdir[1]}) indovina la direzione nel {pct(acc(1, mdir[1]))} dei casi a 1 settimana; chi prevede sempre la direzione più frequente (L0) nel {pct(acc(1, 'L0'))}. **Il guadagno è {'nullo' if max(abs(g) for g in gain_dir) < 0.005 else 'minimo'}** a tutti gli orizzonti (differenze di accuratezza: {', '.join(pct(g, 1) for g in gain_dir)}); nessun confronto con L0 è significativo dopo Holm ({len(sig_dir)} su {int(dr['p'].notna().sum())}).
4. **Perché le candele sono quasi sempre in ribasso.** L'apertura ufficiale è in mediana il {pct(info['rapporto_apertura']['ultimo'] - 1)} sopra la chiusura precedente e la settimana chiude sopra la propria apertura solo nel {pct(D.loc[(1, 'L0'), 'frequenza_rialzo'])} dei casi. Nel walk-forward la quota di candele disegnate in rialzo a h = 1 è {pct(quota_su[1])}.
5. **Ampiezza e ombre: guadagno quasi nullo sulla mediana storica.** Intervallo di confidenza che esclude lo zero solo per: {', '.join(amp_sig) or 'nessuna componente'}.
6. **K0 resta la migliore per errore e sovrapposizione**, come previsto: sovrapposizione media {n(iou['K0'], 3)} contro {n(iou[K], 3)} di {K}; errore sulla chiusura a h = 4 di {n(P.loc[('K0', 4), 'mae_chiusura'])} punti contro {n(P.loc[(K, 4), 'mae_chiusura'])}. Ma i suoi corpi sono irrealistici.
7. **K1 (concatenata) deriva**: concatenando quattro settimane «più probabili», tutte in ribasso, la chiusura si allontana dal centro. A h = 4 l'errore sulla chiusura è {n(P.loc[('K1', 4), 'mae_chiusura'])} punti contro {n(P.loc[('K2', 4), 'mae_chiusura'])} di K2.
8. **K3 (caso simile mediano)** ha l'escursione più realistica dopo K0 ({pct(P.loc[('K3', 1), 'rapporto_escursione'], 0)} a h = 1) ma gli errori più alti a 1 settimana ({n(P.loc[('K3', 1), 'mae_chiusura'])} punti sulla chiusura).
9. **La chiusura reale cade dentro l'escursione prevista meno di una volta su due**: {pct(P.loc[(K, 1), 'chiusura_nell_escursione'], 0)} a h = 1 e {pct(P.loc[(K, 4), 'chiusura_nell_escursione'], 0)} a h = 4 per {K}. La candela non è un intervallo di confidenza: per quello ci sono le fasce.
10. **Programma.** `src.forecast` disegna le candele {K} con la probabilità della direzione e la linea della mediana D4, tiene un registro delle previsioni e con `--verifica` le confronta con le settimane completate.

## 2. Componenti (Parte A)

Walk-forward a finestra espandibile, ristima a ogni origine, h = 1, 2, 3, 4, stesse origini della Fase 2b ({info['per_orizzonte']['1']['n_previsioni']} previsioni a h = 1,
obiettivi dal {info['per_orizzonte']['1']['prima_data_obiettivo']} al {info['per_orizzonte']['1']['ultima_data_obiettivo']}).
**Settimane escluse dalle stime** (apertura fuori dall'intervallo minimo–massimo): {'; '.join(f"{e_['data']} (open {n(e_['open'])}, minimo {n(e_['low'])}, massimo {n(e_['high'])})" for e_ in esc)}.
Sono escluse sia come obiettivo sia come variabile.

### A1. Apertura

Rapporto tra apertura ufficiale e chiusura precedente: la previsione è la mediana storica sulla storia disponibile. Nel walk-forward è rimasta tra
{n(info['rapporto_apertura']['min'], 4)} e {n(info['rapporto_apertura']['max'], 4)} (ultimo valore: {n(info['rapporto_apertura']['ultimo'], 4)}). L'errore sull'apertura prevista è nella pagella (sezione 3).

### A2. Direzione del corpo

P(chiusura > apertura) nella settimana t + h. L0 = frequenza storica; L1 = logistica sulle variabili dell'HAR; L2 = logistica su variabili dell'HAR e aggiuntive.
Regola: per ogni orizzonte si usa il modello con il Brier più basso. Test: Diebold-Mariano sul Brier contro L0, Holm su {int(dr['p'].notna().sum())} confronti.

{t_dir}

**Quanto migliora la direzione rispetto al riferimento: {'non migliora' if max(abs(g) for g in gain_dir) < 0.005 else 'migliora in modo trascurabile'}.**
- Accuratezza del modello scelto meno accuratezza di L0: {', '.join(f'h = {h}: {pct(g, 1)}' for h, g in zip(CP.ORIZZONTI, gain_dir))}.
- Brier skill di L1 su L0: {', '.join(f'h = {h}: {pct(D.loc[(h, "L1"), "brier_skill_vs_L0"], 2)}' for h in CP.ORIZZONTI)}; di L2: {', '.join(f'h = {h}: {pct(D.loc[(h, "L2"), "brier_skill_vs_L0"], 2)}' for h in CP.ORIZZONTI)}. Nessuno è significativo.
- L1 è scelto perché ha il Brier più basso, ma la differenza da L0 è dentro il rumore: le sue probabilità restano {'sempre' if quota_su.max() == 0 else 'quasi sempre'} sotto 0,5, quindi la direzione disegnata è {'sempre' if quota_su.max() == 0 else 'quasi sempre'} il ribasso (quota di candele in rialzo: {', '.join(f'{pct(quota_su[h])} a h = {h}' for h in CP.ORIZZONTI)}).
- Le variabili della candela dell'ultima settimana (L2) peggiorano leggermente: escursione, ombre e posizione della chiusura non dicono nulla sulla direzione della settimana successiva.

Per fascia di partenza (h = 1 e 4):

{dir_pivot('fascia_origine')}

Dopo un rialzo oltre il 30% in 4 settimane ({rs} origini su {info['per_orizzonte']['1']['n_previsioni']} a h = 1):

{dir_pivot('rialzo_forte')}

Per sottoperiodo:

{dir_pivot('sottoperiodo')}

I ribassi tendono a essere più frequenti quando il livello di partenza è alto. L1, che ha il livello tra le variabili, ne coglie una piccola parte (Brier skill positivo in alcune fasce), ma non abbastanza da cambiare la direzione prevista.
La calibrazione per classi di probabilità è in `reports/fase2c/direzione_calibrazione.csv`: a h = 1 le probabilità di rialzo di {mdir[1]} stanno tra {cal_min} e {cal_max}, per il {pct(cal_quota, 0)} nella classe 0,3-0,4.

### A3 e A4. Ampiezza del corpo e ombre

Regressione sulla mediana con variabili dell'HAR e aggiuntive; riferimento: mediana storica. Valori in log.

{t_amp}

- Guadagno con intervallo di confidenza che esclude lo zero: {', '.join(amp_sig) or 'nessuno'}. {('Casi in cui il riferimento è migliore: ' + ', '.join(amp_neg) + '.') if amp_neg else ''}
- La previsione media del corpo ({n(A.loc[('corpo', 1), 'previsione_media'], 3)} a h = 1) è vicina alla mediana reale ({n(A.loc[('corpo', 1), 'mediana_reale'], 3)}) e lontana dalla media reale ({n(A.loc[('corpo', 1), 'media_reale'], 3)}): è questo che tiene il rapporto del corpo sotto 0,8.

**Cosa implica.** Di una candela settimanale del VIX si prevede bene solo l'apertura. Direzione, ampiezza e ombre della settimana successiva sono, con il solo VIX, quasi indistinguibili dalle loro frequenze e mediane storiche.

## 3. Pagella delle quattro costruzioni (Parte C)

Costruzioni: {'; '.join(KA.NOMI.values())}. K1 e K2 usano la direzione di {mdir[1]}, l'ampiezza di A3 e le ombre di A4; a h = 1 coincidono per costruzione.

![Pagella](figures/fase2c/pagella.png)

Direzione, realismo, sovrapposizione (frazioni tra 0 e 1):

{t_p1}

Errore assoluto medio in punti VIX:

{t_p2}

Errore assoluto medio in percentuale del valore reale:

{t_p3}

Chiusura: quota di previsioni entro il 10% e entro il 20% dal valore reale:

{t_en}

- **Realismo.** Rapporto del corpo a h = 1: {', '.join(f'{k} {n(sc["rapporto_corpo_h1"][k])}' for k in KA.COSTRUZIONI)}. Rapporto dell'escursione: {', '.join(f'{k} {n(sc["rapporto_escursione_h1"][k])}' for k in KA.COSTRUZIONI)}. Nessuna costruzione ha entrambi i rapporti tra 0,8 e 1,2.
- **Sovrapposizione media su h = 1-4**: {', '.join(f'{k} {n(iou[k], 3)}' for k in KA.COSTRUZIONI)}.
- **Errore sulla chiusura più basso**: {', '.join(f'{best_err[h]} a h = {h}' for h in CP.ORIZZONTI)}. La chiusura di K0 è la mediana D4: era previsto che fosse la più precisa.
- **Direzione della chiusura rispetto all'origine**: K0 la indovina nel {pct(P.loc[('K0', 4), 'direzione_chiusura_vs_origine'])} dei casi a h = 4, {K} nel {pct(P.loc[(K, 4), 'direzione_chiusura_vs_origine'])}. Per sapere se il VIX sarà più alto o più basso di oggi vale la mediana, non la candela.
- **Direzione del corpo**: K1 e K2 la indovinano nel {pct(P.loc[('K2', 1), 'direzione_corpo'])} dei casi a h = 1, K0 nel {pct(P.loc[('K0', 1), 'direzione_corpo'])}, K3 nel {pct(P.loc[('K3', 1), 'direzione_corpo'])}. Il vantaggio di K1 e K2 è quello di disegnare quasi sempre la direzione più frequente.

Per fascia di partenza — sovrapposizione media (h = 1 e 4):

{pag_pivot('fascia_origine', 'sovrapposizione')}

Per fascia di partenza — direzione del corpo indovinata:

{pag_pivot('fascia_origine', 'direzione_corpo')}

Dopo un rialzo oltre il 30% in 4 settimane — sovrapposizione, direzione del corpo, errore sulla chiusura (punti):

{pag_pivot('rialzo_forte', 'sovrapposizione')}

{pag_pivot('rialzo_forte', 'direzione_corpo')}

{pag_pivot('rialzo_forte', 'mae_chiusura', dec=2)}

Per sottoperiodo — sovrapposizione media:

{pag_pivot('sottoperiodo', 'sovrapposizione')}

La pagella completa per ogni gruppo (tutte le misure, tutti gli orizzonti) è in `reports/fase2c/pagella_gruppi.csv`.

## 4. Costruzione scelta e perché

**Regola fissata prima:** tra le costruzioni con rapporto del corpo e rapporto dell'escursione entrambi tra {n(sc['vincolo'][0], 1)} e {n(sc['vincolo'][1], 1)} a h = 1, quella con la sovrapposizione media più alta su h = 1-4;
se nessuna rispetta il vincolo, quella con il rapporto del corpo più vicino a 1.

- Costruzioni che rispettano il vincolo: **{', '.join(sc['ammesse']) or 'nessuna'}**.
- Esito: {sc['motivo']}.
{('- **Spareggio da confermare.** ' + ' e '.join(sp['in_parita']) + ' hanno esattamente lo stesso rapporto del corpo a h = 1 (' + n(sc['rapporto_corpo_h1'][sp['in_parita'][0]], 4) + '): a un passo sono la stessa candela per costruzione. La regola non dice come risolvere la parità. Ho usato il criterio principale della regola stessa, la sovrapposizione media su h = 1-4 (' + ', '.join(f"{k} {n(iou[k], 3)}" for k in sp['in_parita']) + '), che dà **' + sp['scelta'] + '**. È una scelta mia fatta dopo aver visto i risultati: se preferisci un altro criterio, il programma cambia modificando una riga.') if sp else ''}
- **Scelta: {KA.NOMI[K]}.**

## 5. Verifica visiva (G3) e programma

Stesse 12 origini della Fase 2b, candele {K}, fasce D4 e linea della mediana.

![Verifica sul passato](figures/fase2c/g3_verifica_passato.png)

{t_g3}

**{info['g3']['riga']}** Sono pochi casi (12 origini): servono a vedere, non a misurare.

Novità del programma `python -m src.forecast`:

- usa la costruzione scelta e scrive accanto a ogni candela la probabilità della direzione;
- nel grafico aggiunge la linea della mediana D4; fasce e probabilità di D4 e D1 non cambiano;
- **registro**: ogni esecuzione senza `--origine` aggiunge a `reports/registro_previsioni.csv` candele e fasce previste, con data di origine e data di esecuzione; una previsione con la stessa origine non viene duplicata né riscritta;
- **`--verifica`**: confronta il registro con le settimane completate nel file e mostra la pagella dal vivo (errori, direzione, sovrapposizione, copertura delle fasce), con l'avviso quando i casi sono meno di 20. Il registro non serve a scegliere modelli;
- **settimana incompleta senza nota**: l'ultima riga è esclusa anche quando ha meno sedute di quelle attese e l'ultima seduta non è l'ultima attesa della settimana; la decisione è sempre annunciata a schermo.

## 6. Problemi e ambiguità

1. **Vincolo di realismo non rispettato da nessuna costruzione**: applicata la regola di riserva. {('Lo spareggio tra ' + ' e '.join(sp['in_parita']) + ' è una mia scelta (sezione 4).') if sp else ''}
2. **Perché il corpo resta piccolo.** A3 è una regressione sulla mediana, come richiesto; la distribuzione dei corpi è asimmetrica e la mediana è circa il {pct(A.loc[('corpo', 1), 'mediana_reale'] / A.loc[('corpo', 1), 'media_reale'], 0)} della media. Con questa componente il rapporto non può arrivare a 1: per arrivarci servirebbe prevedere la media, cosa fuori dall'elenco chiuso.
3. **Modello di direzione scelto sull'intero walk-forward** e poi usato nella pagella delle candele sullo stesso campione: la pagella di K1 e K2 è leggermente ottimistica. Poiché L1 non differisce da L0, l'effetto è trascurabile.
4. **Scelta della direzione per orizzonte**: ho applicato «il Brier più basso» separatamente a ogni h; ha scelto {', '.join(sorted(set(mdir.values())))} ovunque.
5. **Stesse righe di stima per L1 e L2**: entrambe escludono le settimane con apertura incoerente, per un confronto alla pari.
6. **Corpo ed escursione sono misurati in punti VIX** (|chiusura − apertura| e massimo − minimo); la sovrapposizione è intersezione su unione degli intervalli minimo–massimo.
7. **K3**: tra i 30 casi più simili, quello con variazione a 4 settimane più vicina alla mediana (con 30 casi la mediana cade tra due valori); sono esclusi i casi le cui 4 settimane successive contengono un'apertura incoerente. Il caso mediano tende a essere un caso tranquillo: per questo anche K3 ha corpi più piccoli del reale.
8. **K0 e mediana D4** sono lette da `reports/fase2b/previsioni_dev.csv`: `run_phase2c` richiede che `run_phase2b` sia stato eseguito.
9. **«Rialzo oltre il 30% in 4 settimane»**: variazione semplice della chiusura superiore a 0,30; {rs} origini a h = 1.
10. **Direzione del corpo e direzione del VIX sono cose diverse.** Una candela in ribasso (chiusura sotto l'apertura ufficiale) può accompagnare un VIX più alto della chiusura precedente, perché l'apertura è già più alta. La scheda lo spiega; resta una fonte di equivoci.
11. **Registro.** Registra solo le esecuzioni senza `--origine`; conserva la prima registrazione per ogni origine; la data di esecuzione viene dall'orologio del computer. La prima riga del registro (origine 6 novembre 2026) è stata scritta dopo la data della settimana +1: `--verifica` lo segnala nella colonna `registrata_prima_dell_esito`.
12. **`--verifica` calcola errori su dati successivi all'8 novembre 2024**: è lo scopo dichiarato del comando; riguarda solo previsioni già scritte nel registro.
13. **Settimana incompleta.** Il calendario delle sedute attese è a regole, senza chiusure straordinarie. Dai dati risulta che da maggio 2022 il VIX ha un valore anche in molte festività (settimane con 5 sedute): da quella data tolgo dalle sedute attese solo Venerdì Santo, Natale e Capodanno. È una regola dedotta dal file di sviluppo.
14. **Confronti multipli**: Holm solo sulla tabella della direzione; le tabelle di ampiezze e pagella non sono corrette.

## 7. Proposte per il passo successivo

1. **Decidere cosa deve rappresentare la candela.** Oggi è «la settimana più probabile»: per costruzione è più piccola della settimana media e quasi sempre in ribasso. Se si vuole un corpo di dimensione realistica, l'ampiezza va prevista come media (o come un quantile più alto della mediana, fissato prima): è una modifica di una riga, ma cambia il significato della candela.
2. **Due candele per settimana invece di una**: lo scenario di rialzo e quello di ribasso, ciascuno con la sua probabilità e la sua ampiezza tipica. Motivo: la direzione non è prevedibile ({pct(acc(1, mdir[1]))} contro {pct(acc(1, 'L0'))}), quindi mostrarne una sola nasconde un terzo abbondante dei casi.
3. **Togliere dal modello ciò che non guadagna**: direzione = frequenza storica (L0), ombre = mediana storica. Stesso risultato, programma più semplice e più onesto.
4. **Lasciare lavorare il registro.** Dopo almeno 20 settimane verificate, confrontare la pagella dal vivo con quella di sviluppo. È l'unica misura fuori campione che resta, visto che l'holdout è stato in parte guardato.
5. **Dati giornalieri del VIX** (già scaricati in Fase 2, non usati): escursione e direzione dentro la settimana dipendono da ciò che accade tra una seduta e l'altra. È l'estensione più promettente per le candele, ma è un dato nuovo e richiede una decisione.
6. **Fase 3 (dati esterni)**, rinviata: resta la strada per migliorare la chiusura centrale.
"""
    (C.REPORTS_DIR / "fase2c_report.md").write_text(out)


if __name__ == "__main__":
    run()
