# Analisi e previsione del VIX settimanale

Progetto universitario: analisi e previsione dell'indice di volatilità VIX su base settimanale.
Il lavoro procede per fasi; ogni fase si chiude con un report in `reports/`.

| fase | contenuto | report |
|---|---|---|
| 0 | audit dei dati, analisi esplorativa, baseline walk-forward | [reports/fase0_report.md](reports/fase0_report.md) |
| 1 | pattern storici (dopo e prima dei picchi, periodi calmi, tabelle condizionate) e modelli che usano solo il VIX | [reports/fase1_report.md](reports/fase1_report.md) |

## Come riprodurre

Richiede Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.run_phase0   # Fase 0: dataset puliti, tabelle, figure e report
python -m src.run_phase1   # Fase 1: pattern storici, modelli M0-M7, report (circa 2 minuti)
python -m pytest           # test (anti look-ahead, integrità dell'holdout, metriche)
```

Entrambi i comandi sono deterministici: rieseguendoli si ottengono gli stessi file. `run_phase1` richiede
`data/processed/vix_weekly_dev.csv`, prodotto da `run_phase0`. Versioni e seed sono in
`reports/run_manifest.json` e `reports/fase1/run_manifest.json`.

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
tests/             test automatici
reports/           report, tabelle (.csv/.json) e figure; i risultati della Fase 1 sono in reports/fase1/ e reports/figures/fase1/
notebooks/         opzionale, vuota
```

## Regole del progetto

- **Holdout.** Le ultime 104 settimane complete stanno in `data/processed/vix_weekly_holdout.csv` e sono
  bloccate: nessun modulo di analisi le legge (lo verifica un test). Analisi e baseline usano solo
  `data/processed/vix_weekly_dev.csv`. In Fase 1 c'è anche un blocco a runtime: durante `run_phase1`
  l'apertura del file di holdout, del file clean o del file grezzo solleva un errore.
- **Nessun look-ahead.** Ogni previsione per la data t + h usa solo dati fino a t. Il motore walk-forward
  passa ai modelli esclusivamente la storia fino all'origine; `tests/test_no_lookahead.py` lo controlla
  stravolgendo i dati futuri e verificando che le previsioni passate non cambino; `tests/test_fase1.py`
  fa lo stesso per tutti i modelli della Fase 1.
- **Elenco chiuso dei modelli.** In Fase 1 i modelli e i loro parametri sono fissati in
  `src/fase1/modelli.py` prima della valutazione; un test ne controlla l'elenco.
- **Colonne.** `open` non è usata (scartata alla lettura in Fase 1); `high` e `low` compaiono solo
  nell'analisi esplorativa A2.
- **Numeri del report.** Il report è generato da `src/report.py` a partire dai file prodotti dal codice;
  non contiene valori scritti a mano.
- **Dati grezzi.** `data/raw/` non viene mai modificato; anomalie e ambiguità sono segnalate nel report,
  non corrette in silenzio.
