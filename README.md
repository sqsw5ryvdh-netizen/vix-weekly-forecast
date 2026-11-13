# Analisi e previsione del VIX settimanale

Progetto universitario: analisi e previsione dell'indice di volatilità VIX su base settimanale.
Il lavoro procede per fasi; ogni fase si chiude con un report in `reports/`.

| fase | contenuto | report |
|---|---|---|
| 0 | audit dei dati, analisi esplorativa, baseline walk-forward | [reports/fase0_report.md](reports/fase0_report.md) |

## Come riprodurre

Richiede Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.run_phase0   # rigenera dataset puliti, tabelle, figure e report della Fase 0
python -m pytest           # test (anti look-ahead, integrità dell'holdout, metriche)
```

`run_phase0` è deterministico: rieseguendolo si ottengono gli stessi file. Versioni e seed sono in
`reports/run_manifest.json`.

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
  run_phase0.py    esegue tutto in sequenza
tests/             test automatici
reports/           report, tabelle (.csv/.json) e figure
notebooks/         opzionale, vuota
```

## Regole del progetto

- **Holdout.** Le ultime 104 settimane complete stanno in `data/processed/vix_weekly_holdout.csv` e sono
  bloccate: nessun modulo di analisi le legge (lo verifica un test). Analisi e baseline usano solo
  `data/processed/vix_weekly_dev.csv`.
- **Nessun look-ahead.** Ogni previsione per la data t + h usa solo dati fino a t. Il motore walk-forward
  passa ai modelli esclusivamente la storia fino all'origine; `tests/test_no_lookahead.py` lo controlla
  stravolgendo i dati futuri e verificando che le previsioni passate non cambino.
- **Numeri del report.** Il report è generato da `src/report.py` a partire dai file prodotti dal codice;
  non contiene valori scritti a mano.
- **Dati grezzi.** `data/raw/` non viene mai modificato; anomalie e ambiguità sono segnalate nel report,
  non corrette in silenzio.
