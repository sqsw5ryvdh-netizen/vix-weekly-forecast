"""Genera reports/fase2b_report.md. Tutti i numeri sono letti dai file reports/fase2b/*.csv e *.json."""
from __future__ import annotations

import json
import re

import pandas as pd

from .. import config as C
from . import modello as MO

R = C.REPORTS_DIR / "fase2b"


def md(df: pd.DataFrame, fmt=".3f") -> str:
    return re.sub(r"\bnan\b", "  –", df.to_markdown(index=False, floatfmt=fmt, missingval="–"))


def n(v, d: int = 2) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def pct(v, d: int = 1) -> str:
    return f"{100 * v:.{d}f}%".replace(".", ",")


def run() -> None:
    ch, hl, pr = (pd.read_csv(R / f"{k}.csv") for k in ("chiusura_metriche", "massimo_minimo_metriche", "probabilita_metriche"))
    g3 = pd.read_csv(R / "g3_pannelli.csv")
    info = json.loads((R / "info.json").read_text())
    C_, H = ch.set_index("h"), hl.set_index(["obiettivo", "h"])
    t = ch.copy()
    for lev in (50, 80, 90):
        t[f"copertura {lev}% [IC 95%]"] = [f"{100 * a:.1f} [{100 * b:.1f}; {100 * c:.1f}]" for a, b, c in zip(t[f"copertura_{lev}"], t[f"copertura_{lev}_ci_lo"], t[f"copertura_{lev}_ci_hi"])]
    t_err = md(t[["h", "n", "mediana_mae_vix", "mae_m1_vix", "mediana_rmse_vix", "crps", "quantili_incrociati"]].rename(columns={
        "mediana_mae_vix": "errore assoluto medio, mediana D4 (punti)", "mae_m1_vix": "errore assoluto medio, M1 (punti)",
        "mediana_rmse_vix": "RMSE, mediana D4 (punti)", "crps": "CRPS (log)", "quantili_incrociati": "quantili riordinati"}), ".3f")
    t_cov = md(t[["h", "n", "larghezza_vix_50", "copertura 50% [IC 95%]", "larghezza_vix_80", "copertura 80% [IC 95%]", "larghezza_vix_90", "copertura 90% [IC 95%]"]].rename(columns={
        "larghezza_vix_50": "larghezza 50% (punti)", "larghezza_vix_80": "larghezza 80% (punti)", "larghezza_vix_90": "larghezza 90% (punti)"}), ".2f")
    t_as = md(pd.DataFrame([{"h": r.h} | {f"{lev}%: sopra / sotto": f"{100 * getattr(r, f'sopra_{lev}'):.1f} / {100 * getattr(r, f'sotto_{lev}'):.1f}" for lev in (50, 80, 90)}
                            for r in ch.itertuples()]))
    u = hl.copy()
    u["guadagno sul riferimento (punti) [IC 95%]"] = [f"{a:+.3f} [{b:+.3f}; {c:+.3f}]" for a, b, c in zip(u["guadagno_punti"], u["guadagno_ci_lo"], u["guadagno_ci_hi"])]
    u["guadagno (%)"] = 100 * u["guadagno_pct"]
    u["correzioni del vincolo"] = [f"{a} ({100 * b:.1f}%)" for a, b in zip(u["n_correzioni"], u["quota_correzioni"])]
    t_hl = md(u.sort_values(["obiettivo", "h"])[["obiettivo", "h", "n", "mae_modello", "mae_riferimento", "guadagno sul riferimento (punti) [IC 95%]", "guadagno (%)", "dm_stat", "p",
                                                  "correzioni del vincolo", "errore_medio_con_segno", "quota_reale_oltre_previsto"]].rename(columns={
        "mae_modello": "errore assoluto medio, modello", "mae_riferimento": "errore assoluto medio, riferimento", "dm_stat": "DM", "errore_medio_con_segno": "errore medio con segno (reale − previsto)",
        "quota_reale_oltre_previsto": "quota di settimane in cui il reale va oltre il previsto"}), ".3f")
    t_pr = md(pr[["h", "evento", "n", "frequenza_evento", "prob_media", "brier", "affidabilita", "risoluzione"]].rename(columns={
        "frequenza_evento": "frequenza osservata", "prob_media": "probabilità media", "affidabilita": "affidabilità"}), ".4f")
    t_g3 = md(g3.assign(dentro_50=g3["dentro_50"].map({True: "sì", False: "no"}), dentro_80=g3["dentro_80"].map({True: "sì", False: "no"})).rename(columns={
        "data_origine": "origine", "chiusura_origine": "chiusura all'origine", "chiusura_prevista_4": "chiusura prevista a +4", "chiusura_reale_4": "chiusura reale a +4",
        "dentro_50": "dentro la fascia 50%", "dentro_80": "dentro la fascia 80%"})[["origine", "chiusura all'origine", "chiusura prevista a +4", "fascia_50_inf", "fascia_50_sup",
                                                                                   "fascia_80_inf", "fascia_80_sup", "chiusura reale a +4", "dentro la fascia 50%", "dentro la fascia 80%"]], ".2f")
    sig = lambda ob: [int(h) for h in MO.ORIZZONTI if H.loc[(ob, h), "guadagno_ci_lo"] > 0]
    neg = [f"{ob} a h = {h}" for ob in ("massimo", "minimo") for h in MO.ORIZZONTI if H.loc[(ob, h), "guadagno_punti"] < 0]
    lst = lambda v: ", ".join(str(k) for k in v) if v else "nessuno"
    cov_ok = [f"{lev}% a h = {h}" for h in MO.ORIZZONTI for lev in (50, 80, 90) if C_.loc[h, f"copertura_{lev}_ci_lo"] <= lev / 100 <= C_.loc[h, f"copertura_{lev}_ci_hi"]]
    g = info["g3"]

    out = f"""# Fase 2b — Programma di previsione con grafico a candele

*Report generato da `python -m src.run_phase2b`. Ogni numero è letto dai file `reports/fase2b/*.csv` e `*.json`. Le misure di questo report usano solo
il campione di sviluppo ({info['campione'][0]} → {info['campione'][1]}, {info['n_settimane']} settimane), sotto la guardia che blocca l'holdout.
La previsione corrente è in [previsione_corrente.md](previsione_corrente.md) ed è prodotta da `python -m src.forecast`.*

## 1. Sintesi

1. **Il programma.** `python -m src.forecast` legge lo storico, esclude la settimana incompleta e produce per le settimane +1 … +4: candela prevista (apertura, massimo, minimo, chiusura), fasce della chiusura al 50/80/90%, P(VIX > 20/25/30), lettura della situazione, tabella e grafici. Nessun modello nuovo: M1, D4 e D1 delle fasi precedenti, più la regressione di M1 applicata a massimo e minimo.
2. **Errore sulla chiusura.** L'errore assoluto medio della mediana cresce da {n(C_.loc[1, 'mediana_mae_vix'])} punti a 1 settimana a {n(C_.loc[4, 'mediana_mae_vix'])} a 4 settimane ({n(C_.loc[2, 'mediana_mae_vix'])} e {n(C_.loc[3, 'mediana_mae_vix'])} a 2 e 3).
3. **Le fasce coprono meno del nominale a tutti gli orizzonti**: la fascia all'80% contiene la chiusura reale tra il {pct(ch['copertura_80'].min())} e il {pct(ch['copertura_80'].max())} delle volte, quella al 50% tra il {pct(ch['copertura_50'].min())} e il {pct(ch['copertura_50'].max())}. Il programma mostra questi numeri accanto a ogni fascia, senza ricalibrare.
4. **Minimo settimanale**: il modello batte il riferimento semplice a tutti gli orizzonti (errore più basso dal {pct(hl[hl['obiettivo'] == 'minimo']['guadagno_pct'].min())} al {pct(hl[hl['obiettivo'] == 'minimo']['guadagno_pct'].max())}).
5. **Massimo settimanale**: il guadagno è molto più piccolo (dal {pct(hl[hl['obiettivo'] == 'massimo']['guadagno_pct'].min())} al {pct(hl[hl['obiettivo'] == 'massimo']['guadagno_pct'].max())}); l'intervallo di confidenza esclude lo zero per h = {lst(sig('massimo'))}{'' if 1 in sig('massimo') else '; a 1 settimana modello e riferimento sono equivalenti'}.
6. **Verifica visiva (G3)**: {g['riga']}

## 2. Chiusura: errore, larghezza e copertura per h = 1, 2, 3, 4

Stesso protocollo della Fase 2: finestra espandibile, ristima a ogni origine, almeno {info['min_errori_passati']} errori passati con esito noto, stesso insieme di origini
per tutti gli orizzonti disponibili. Mediana e fasce: D4. Per h = 2 e 3 nessuna nuova scelta: sono gli stessi modelli, qui misurati per la prima volta.

{t_err}

Larghezza media delle fasce (punti VIX) e copertura misurata (%, con intervallo di confidenza bootstrap a blocchi):

{t_cov}

Quota di chiusure reali sopra il limite superiore / sotto il limite inferiore di ogni fascia (%; valore nominale per lato: 25, 10, 5):

{t_as}

- **La mediana di D4 è leggermente più precisa di M1** a tutti gli orizzonti ({n(C_.loc[1, 'mediana_mae_vix'])} contro {n(C_.loc[1, 'mae_m1_vix'])} punti a h = 1; {n(C_.loc[4, 'mediana_mae_vix'])} contro {n(C_.loc[4, 'mae_m1_vix'])} a h = 4).
- **Copertura.** {('Fasce il cui intervallo di confidenza contiene il valore nominale: ' + ', '.join(cov_ok) + '; per le altre la sottocopertura è statisticamente chiara.') if cov_ok else 'Per nessuna fascia e nessun orizzonte l’intervallo di confidenza della copertura contiene il valore nominale: la sottocopertura è statisticamente chiara ovunque.'} Gli sforamenti verso il basso sono un po' più frequenti di quelli verso l'alto.
- **h = 2 e 3 si comportano come h = 1 e 4**: nessuna anomalia negli orizzonti intermedi.
- I numeri per h = 1 e h = 4 coincidono con quelli della Fase 2 a meno di differenze numeriche minime (qui il logaritmo è calcolato dalla chiusura, in Fase 2 era letto dal file con 6 decimali).

**Misura aggiuntiva, non richiesta: probabilità di D1.** Poiché il programma mostra P(VIX > 20/25/30) anche a 2 e 3 settimane, dove non erano mai state misurate, riporto frequenza osservata, probabilità media e Brier. Nessuna scelta dipende da questa tabella.

{t_pr}

## 3. Massimo e minimo settimanali

Modello: la stessa regressione diretta di M1 (ultima chiusura, media a 4 e a 12 settimane delle chiusure) con obiettivo log(massimo) e log(minimo) della settimana t + h.
Riferimento: ultima chiusura × mediana storica del rapporto tra massimo (o minimo) della settimana t + h e chiusura della settimana t, calcolata sulla storia disponibile.
Gli errori del modello sono misurati sui valori finali della candela, cioè dopo i vincoli (massimo ≥ apertura e chiusura previste; minimo ≤ apertura e chiusura previste).

{t_hl}

(Guadagno positivo = il modello sbaglia meno del riferimento. DM = Diebold-Mariano sull'errore assoluto, non corretto per confronti multipli.)

- **Minimo**: il modello è migliore del riferimento a ogni orizzonte, con intervallo di confidenza che esclude lo zero per h = {lst(sig('minimo'))}.
- **Massimo**: guadagno piccolo; intervallo che esclude lo zero solo per h = {lst(sig('massimo'))}. {('Casi in cui il riferimento è migliore: ' + ', '.join(neg) + '.') if neg else 'In nessun caso il riferimento è migliore del modello.'}
- **Il massimo è più difficile del minimo**: errore assoluto medio di {n(H.loc[('massimo', 1), 'mae_modello'])} punti contro {n(H.loc[('minimo', 1), 'mae_modello'])} a h = 1, e di {n(H.loc[('massimo', 4), 'mae_modello'])} contro {n(H.loc[('minimo', 4), 'mae_modello'])} a h = 4. I picchi dentro la settimana non sono prevedibili.
- **Il massimo previsto è superato meno di una volta su due** ({pct(H.loc[('massimo', 1), 'quota_reale_oltre_previsto'], 0)} a h = 1), ma quando viene superato lo è di molto: l'errore medio con segno è positivo ({n(H.loc[('massimo', 1), 'errore_medio_con_segno'])} punti a h = 1, {n(H.loc[('massimo', 4), 'errore_medio_con_segno'])} a h = 4). L'ombra superiore della candela prevista è un valore tipico, non un limite.
- **Vincoli.** Correzioni del massimo previsto: {int(hl[hl['obiettivo'] == 'massimo']['n_correzioni'].sum())} in tutto il walk-forward; del minimo: {int(H.loc[('minimo', 1), 'n_correzioni'])} origini su {int(H.loc[('minimo', 1), 'n'])} a h = 1 e {int(H.loc[('minimo', 4), 'n_correzioni'])} a h = 4.

## 4. Verifica visiva sul passato (G3)

Regola fissa: {g['regola']}. Tutte le origini e gli esiti stanno nel campione di sviluppo.

![Verifica sul passato](figures/fase2b/g3_verifica_passato.png)

{t_g3}

**{g['riga']}** Dodici casi sono pochi (meno di 20): il conteggio serve a vedere, non a misurare. La misura è la copertura della sezione 2.

## 5. Problemi e ambiguità

1. **Dati successivi all'8 novembre 2024.** `src.forecast` li legge come storia. Il modello D1 usa, per costruzione, gli errori delle proprie previsioni passate fino all'origine: con un'origine nel 2026 questo comprende errori relativi al periodo 2024-2026, usati come ingrediente del modello e mai mostrati né riassunti. L'ho inteso compatibile con «solo come storia da cui prevedere»; se non lo è, D1 va stimato con errori fino al confine di sviluppo.
2. **L'holdout non è più "mai visto".** La scheda corrente e il grafico G1 mostrano le chiusure degli ultimi due anni e i casi simili possono cadere in quel periodo. Nessuna misura di errore è stata calcolata lì e nessuna scelta è stata fatta su quei dati, ma una valutazione finale sull'holdout non sarà più alla cieca per chi ha guardato la scheda.
3. **Settimana incompleta**: riconosciuta dalla colonna `note` valorizzata, come in Fase 0. Un file aggiornato in cui l'ultima settimana è parziale ma senza nota verrebbe trattato come completo.
4. **Date delle settimane previste**: origine + 7 giorni per settimana; sono indicative, perché l'ultima seduta della settimana può non essere un venerdì.
5. **Riferimento per massimo e minimo a h > 1**: ho usato il rapporto tra il massimo (minimo) della settimana t + h e la chiusura della settimana t. È la mia lettura di «chiusura precedente» per gli orizzonti oltre il primo.
6. **Fasce (D4) e probabilità (D1) vengono da modelli diversi**: possono non essere coerenti tra loro al decimale (per esempio una probabilità di superare 25 diversa da quella implicita nelle fasce).
7. **Corpo e ombre delle candele previste hanno natura diversa**: il corpo unisce due mediane successive, le ombre vengono da due regressioni separate. Una candela prevista non è la previsione di una candela «tipica», che avrebbe corpo più ampio: è un riassunto di quattro numeri.
8. **Lettura della situazione**: massimo e minimo delle ultime 12 settimane sono calcolati sulle chiusure; «ultimi 5 anni» = 260 settimane; la soglia dei 20 casi è segnalata nel testo quando non è raggiunta.
9. **Casi simili**: scelti in ordine di somiglianza scartando quelli a meno di 8 settimane da un caso già scelto; è una selezione illustrativa, diversa dal modello M6 (che usa i 30 più vicini senza distanza minima).
10. **G1**: i percorsi dei casi simili sono nel pannello di ingrandimento sotto il grafico a 2 anni, dove le quattro settimane di previsione sarebbero troppo strette per contenerli. **G2**: storia come linea delle chiusure e riquadro con le candele, perché a quella scala le candele non sono leggibili.
11. **Origine minima**: servono almeno {info['origine_minima_indice'] + 1} settimane di storia (circa 7 anni), altrimenti D1 non ha i {info['min_errori_passati']} errori richiesti e il comando si ferma con un messaggio.
12. **Quantili riordinati**: D4 ha prodotto quantili incrociati in {', '.join(f"{info['per_orizzonte'][str(h)]['incroci']} origini (h = {h})" for h in MO.ORIZZONTI)}; sono riordinati, come in Fase 2.
13. **Misura aggiuntiva**: la tabella sulle probabilità di D1 (sezione 2) non era richiesta; l'ho aggiunta perché il programma mostra quelle probabilità anche a 2 e 3 settimane.
14. **Test multipli**: i confronti con il riferimento sono otto e non sono corretti.

## 6. Riproducibilità

```
python -m src.run_phase2b     # misure di sviluppo per h = 1-4, G3 e questo report (circa 4 minuti)
python -m src.forecast        # previsione corrente: scheda, tabella, JSON, G1 e G2
python -m pytest              # test
```
"""
    (C.REPORTS_DIR / "fase2b_report.md").write_text(out)


if __name__ == "__main__":
    run()
