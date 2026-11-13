# Trasformazioni dal file grezzo ai dataset puliti

Sorgente: `data/raw/VIX_settimanale_2000-2026.csv` (sha256 `bf67b3f19ef38827dd84ba1f2a4008cc439651a93328b9a42b6aab10dcbb76b5`).
Generato da `python -m src.data_audit`.

- T1. Lettura del CSV grezzo senza modifiche; `week_start` e `last_trading_day` convertite in date.
- T2. Rimozione delle righe con `note` valorizzata che indicano una settimana incompleta (settimana in corso): la chiusura non e' una chiusura settimanale definitiva.
- T3. Colonna `date` = `last_trading_day` (data in cui la chiusura settimanale e' osservabile); e' la data di riferimento di ogni osservazione.
- T4. Colonne derivate: `log_close` = ln(close); `dlog_close` = differenza prima di `log_close`.
- T5. Colonne di segnalazione (nessun valore viene corretto): `flag_ohlc_incoerente` (high < max(open, close) oppure low > min(open, close)); `flag_settimana_corta` (trading_days <= 3).
- T6. Divisione temporale: le ultime 104 settimane complete -> holdout (file separato, bloccato); tutte le precedenti -> dev. Il file clean e' l'unione dei due.

| file | righe | da | a |
|---|---|---|---|
| `vix_weekly_clean.csv` | 1401 | 2000-01-07 | 2026-11-06 |
| `vix_weekly_dev.csv` | 1297 | 2000-01-07 | 2024-11-08 |
| `vix_weekly_holdout.csv` | 104 | 2024-11-15 | 2026-11-06 |

`vix_weekly_holdout.csv` e' bloccato: non va usato per analisi, stime o valutazioni in Fase 0.
