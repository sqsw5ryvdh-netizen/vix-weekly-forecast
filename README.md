# Analisi e previsione del VIX settimanale

Progetto universitario: analisi e previsione dell'indice di volatilità VIX su base settimanale.
Il lavoro procede per fasi; ogni fase si chiude con un report in `reports/`.

| fase | contenuto | report |
|---|---|---|
| 0 | audit dei dati, analisi esplorativa, baseline walk-forward | [reports/fase0_report.md](reports/fase0_report.md) |
| 1 | pattern storici (dopo e prima dei picchi, periodi calmi, tabelle condizionate) e modelli che usano solo il VIX | [reports/fase1_report.md](reports/fase1_report.md) |
| 2 | previsione probabilistica attorno al tipo HAR (quantili, intervalli, probabilità di eventi), scheda di previsione, raccolta dei dati esterni | [reports/fase2_report.md](reports/fase2_report.md) |

## Come riprodurre

Richiede Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.run_phase0   # Fase 0: dataset puliti, tabelle, figure e report
python -m src.run_phase1   # Fase 1: pattern storici, modelli M0-M7, report (circa 2 minuti)
python -m src.run_phase2   # Fase 2: modelli di densità, eventi, scheda di previsione, report (circa 5 minuti)
python -m pytest           # test (anti look-ahead, integrità dell'holdout, metriche)
```

I tre comandi sono deterministici: rieseguendoli si ottengono gli stessi file. `run_phase1` e `run_phase2`
richiedono `data/processed/vix_weekly_dev.csv`, prodotto da `run_phase0`. Versioni e seed sono in
`reports/run_manifest.json`, `reports/fase1/run_manifest.json` e `reports/fase2/run_manifest.json`.

### Dati esterni (Fase 2, Parte C)

```bash
python -m src.fetch_external                     # scarica le serie dalle fonti pubbliche (richiede rete e curl)
python -m src.fetch_external --riusa-originali   # ricostruisce i file a partire dagli originali già scaricati
python -m src.fase2.report                       # rigenera il report con l'inventario aggiornato
```

Lo scarico non è deterministico (dipende dal giorno) ed è separato da `run_phase2`. Produce
`data/external/raw/` (originali), `data/external/dev/` (serie giornaliere e settimanali fino al confine di
sviluppo) e `data/external/holdout/` (parte successiva, bloccata). **`data/external/` non è nel repository**:
contiene dati di terzi; nel repository ci sono il programma di scarico e l'inventario
(`reports/fase2/c_inventario.json`). In questa fase i dati esterni sono solo raccolti: non sono usati da
alcun modello.

## Struttura

```
data/raw/          file originale, mai modificato
data/processed/    dataset puliti generati dal codice (+ TRASFORMAZIONI.md)
src/
  config.py        percorsi e costanti (holdout, soglie, orizzonti, seed)
  data_audit.py    audit del file grezzo, dataset clean / dev / holdout
  eda.py           analisi esplorativa (solo campione di sviluppo)
  baselines.py     modelli baseline e motore walk-forward
  evaluate.py      metriche, test di Diebold-Mariano, Brier, calibrazione
  report.py        costruisce il report leggendo reports/*.csv e *.json
  run_phase0.py    esegue tutta la Fase 0 in sequenza
  run_phase1.py    esegue tutta la Fase 1 in sequenza, sotto la guardia che blocca l'holdout
  fase1/
    comune.py      costanti, unico accesso ai dati (load_dev), guardia sui file vietati, bootstrap
    pattern.py     Parte A: pattern storici A1-A5 con intervalli bootstrap
    modelli.py     Parte B: elenco chiuso dei modelli M0-M7
    valuta.py      walk-forward, Clark-West, Diebold-Mariano, Holm
    report.py      costruisce il report leggendo reports/fase1/*.csv e *.json
  run_phase2.py    esegue Parti A e B della Fase 2 e il report, sotto la guardia
  fase2/
    densita.py     elenco chiuso dei modelli di densità D1-D6, D1b e del modello per eventi E1
    valuta.py      walk-forward, CRPS, copertura, PIT, Brier, Diebold-Mariano, Holm
    scheda.py      scheda di previsione a una data origine (mediana, intervalli, probabilità)
    report.py      costruisce il report leggendo reports/fase2/*.csv e *.json
  fetch_external.py  Parte C: scarico, documentazione e allineamento dei dati esterni
tests/             test automatici
reports/           report, tabelle (.csv/.json) e figure; i risultati delle Fasi 1 e 2 sono in reports/fase1/, reports/fase2/ e nelle rispettive cartelle di figure
notebooks/         opzionale, vuota
```

## Regole del progetto

- **Holdout.** Le ultime 104 settimane complete stanno in `data/processed/vix_weekly_holdout.csv` e sono
  bloccate: nessun modulo di analisi le legge (lo verifica un test). Analisi e baseline usano solo
  `data/processed/vix_weekly_dev.csv`. In Fase 1 c'è anche un blocco a runtime: durante `run_phase1`
  l'apertura del file di holdout, del file clean o del file grezzo solleva un errore. Dalla Fase 2 la
  stessa guardia blocca anche gli originali dei dati esterni e la loro parte di holdout.
- **Nessun look-ahead.** Ogni previsione per la data t + h usa solo dati fino a t. Il motore walk-forward
  passa ai modelli esclusivamente la storia fino all'origine; `tests/test_no_lookahead.py` lo controlla
  stravolgendo i dati futuri e verificando che le previsioni passate non cambino; `tests/test_fase1.py`
  fa lo stesso per tutti i modelli della Fase 1, `tests/test_fase2.py` per i modelli di densità e per
  l'allineamento settimanale dei dati esterni (ogni valore usato deve essere disponibile entro la data).
- **Elenco chiuso dei modelli.** In Fase 1 i modelli e i loro parametri sono fissati in
  `src/fase1/modelli.py` prima della valutazione; un test ne controlla l'elenco. Lo stesso vale per i
  modelli di densità della Fase 2 (`src/fase2/densita.py`).
- **Colonne.** `open` non è usata (scartata alla lettura in Fase 1); `high` e `low` compaiono solo
  nell'analisi esplorativa A2.
- **Numeri del report.** Il report è generato da `src/report.py` a partire dai file prodotti dal codice;
  non contiene valori scritti a mano.
- **Dati grezzi.** `data/raw/` non viene mai modificato; anomalie e ambiguità sono segnalate nel report,
  non corrette in silenzio.
