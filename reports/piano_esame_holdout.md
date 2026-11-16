# Piano d'esame sull'holdout

*Scritto nella Fase 4, prima di aprire l'holdout. Generato da `python -m src.run_phase4` (modulo `src/fase4/piano.py`). L'esame non è stato eseguito:
questo documento fissa che cosa verrà fatto e come verranno lette le risposte. Dopo questa fase i modelli sono chiusi.*

## 1. Che cosa viene esaminato

Le versioni congelate in `versioni/*.json` (parametri, finestre, regole, impronta dei file sorgente e commit del congelamento):

- **F1 solo storico del VIX** (`versioni/f1.json`).
- **F2 con struttura a termine (VIX3M)** (`versioni/f2.json`).
- **F3 modello a regimi (N1)** (`versioni/f3.json`).

Nessun parametro, nessuna finestra e nessuna regola potrà essere cambiata dopo l'apertura dell'holdout. Prima dell'esame si verifica che i file sorgente
elencati nelle configurazioni abbiano ancora la stessa impronta SHA-256 e che tutti i test passino; l'esito della verifica va scritto nel report d'esame.

## 2. Dati e procedura

- **Periodo d'esame:** le 104 settimane successive al 2024-11-08 (`data/processed/vix_weekly_holdout.csv`), mai lette finora da nessuna analisi.
  Per F2 anche la parte di holdout dei dati giornalieri di VIX e VIX3M (`data/external/holdout`), anch'essa mai letta.
- **Obiettivi:** la chiusura di ogni settimana dell'holdout, prevista a 1, 2, 3, 4 settimane di distanza. Per ogni orizzonte h le origini sono le settimane
  la cui settimana-obiettivo cade nell'holdout: 104 previsioni per orizzonte (le prime origini sono le ultime settimane del campione di sviluppo).
- **Procedura:** lo stesso walk-forward dello sviluppo. A ogni origine i modelli sono ristimati con tutta la storia fino a quell'origine (sviluppo più le settimane di
  holdout già trascorse), con le finestre e le cadenze di ristima congelate (per F3: ristima ogni 26 settimane alle date fisse già in uso). Nessuna informazione successiva all'origine entra nella previsione.
- **F2 quando manca VIX3M:** se X1 non è valida in una settimana di origine, F2 usa la previsione di F1 per quell'origine (è il ripiego del programma). Il numero di ripieghi viene riportato;
  le metriche di F2 sono calcolate sia su tutte le origini (ripieghi compresi) sia sulle sole origini senza ripiego.
- **Una sola esecuzione.** L'esame si esegue una volta, con un comando dedicato che scrive tutte le tabelle. Se dopo l'esecuzione emerge un errore di programma, lo si corregge
  dichiarandolo e riportando i risultati prima e dopo la correzione. I risultati si riportano comunque, qualunque sia l'esito.

## 3. Metriche

Per ogni versione e per ogni orizzonte (h = 1 e h = 4 in evidenza; h = 2 e 3 riportati):

| ambito | metriche |
|---|---|
| chiusura centrale | errore assoluto medio in punti; radice dell'errore quadratico medio in log; quota di previsioni entro il 10% e entro il 20% dal valore reale |
| centro (M1, M1 + X1, media del modello a regimi) | errore assoluto medio in punti; radice dell'errore quadratico medio in log |
| fasce | copertura e larghezza media (punti) delle fasce al 50%, 80% e 90%; CRPS sulla griglia di 19 quantili; perdita quantilica ai quantili riportati |
| direzione ed etichetta | Brier della probabilità che la chiusura superi quella dell'origine; quota di etichette «neutre» (probabilità tra 40% e 60%); accuratezza delle etichette non neutre, con il loro numero |
| probabilità di livello | Brier delle probabilità di chiudere sopra 20, 25, 30, con la frequenza osservata dell'evento |
| candele | la pagella della Fase 2c: rapporto corpo ed escursione previsti / reali, sovrapposizione, errori medi di apertura, massimo, minimo e chiusura, quota di chiusure reali dentro l'escursione prevista |

Riferimenti riportati accanto, per orientarsi: «nessuna variazione» (chiusura centrale = chiusura dell'origine) e la frequenza storica per la direzione.
Tutte le metriche anche per la **fascia 13-22** (chiusura all'origine tra 13 e 22, estremi inclusi), con il numero di casi; sotto 20 casi lo si scrive e non si commenta.
Intervalli di confidenza: bootstrap a blocchi (26 settimane, 2000 repliche, seed del progetto), come nello sviluppo.

## 4. Tabelle e figure che verranno prodotte

1. Chiusura centrale e centro: una riga per versione e orizzonte.
2. Fasce: copertura, larghezza, CRPS.
3. Direzione ed etichetta.
4. Probabilità di superare 20, 25, 30.
5. Pagella delle candele.
6. Confronti a coppie con F1 (sezione 5): differenze, intervalli bootstrap, test, p-value corretti.
7. Le stesse tabelle 1-5 per la fascia 13-22.
8. Diagnostica: numero di ripieghi di F2, stime non valide e probabilità di regime di F3, quantili incrociati, origini usate.
9. Figure: chiusura reale e chiusure centrali previste a 1 e a 4 settimane nel periodo d'esame; copertura delle fasce nel tempo.
10. Accanto a ogni numero dell'esame, lo stesso numero misurato nello sviluppo (`reports/fase4/c_confronto.csv` e fasi precedenti), per vedere quanto regge fuori campione.

## 5. Criteri fissati ora

F1 è la versione di riferimento: usa solo lo storico settimanale del VIX. Ogni altra versione (F2 con struttura a termine (VIX3M), F3 modello a regimi (N1)) è confrontata con F1 sulle stesse 104 previsioni per orizzonte.

**Tre criteri, ciascuno con il suo test** (Diebold-Mariano a due code con la correzione per piccoli campioni, sulle differenze di perdita tra F1 e la versione):

| criterio | perdita |
|---|---|
| A. chiusura | errore quadratico in log della chiusura centrale |
| B. direzione | Brier della probabilità che la chiusura superi quella dell'origine |
| C. fasce | CRPS sulla griglia di 19 quantili |

Per ogni criterio i test sono 4 (2 versioni × h = 1 e h = 4) e i p-value sono corretti con Holm dentro il criterio. Soglia: 5%.

**Come si leggono i risultati:**

- Una versione è dichiarata **migliore di F1 su un criterio** se a h = 1 oppure a h = 4 la sua perdita è più bassa con p corretto sotto il 5%, e all'altro orizzonte non è significativamente peggiore.
- È dichiarata **peggiore di F1 su un criterio** nel caso simmetrico.
- In tutti gli altri casi le due versioni sono dichiarate **non distinguibili** su quel criterio. Non verrà dichiarata un'equivalenza: con 104 settimane un test può non vedere differenze piccole come quelle
  misurate nello sviluppo (pochi punti percentuali), quindi «non distinguibili» è l'esito più probabile e non dimostra che le versioni siano uguali.
- **Verdetto complessivo.** Una versione è dichiarata migliore di F1 nel complesso solo se è migliore su almeno un criterio e non è peggiore su nessuno. Se nessuna versione lo è, **resta F1**, perché a parità di risultati
  usa meno dati e meno ipotesi. Se più versioni lo sono, si riportano entrambe con i criteri su cui prevalgono; tra loro non si fa una classifica oltre a quella descrittiva delle tabelle.
- **Coerenza con lo sviluppo.** Per ogni confronto si dice anche se il segno della differenza nell'esame è lo stesso misurato nello sviluppo.
- Copertura delle fasce, pagella delle candele, probabilità di livello e fascia 13-22 sono **descrittive**: si riportano con gli intervalli, ma non entrano nel verdetto.
- La decisione su quale versione usare nel programma dopo l'esame resta a chi dirige il progetto: questo piano fissa solo che cosa l'esame potrà affermare.

## 6. Che cosa non verrà fatto

- Nessuna ristima di iperparametri, nessuna variante nuova, nessuna ricalibrazione delle fasce dopo aver visto l'holdout.
- Nessun sottoperiodo dentro l'holdout oltre alla fascia 13-22: 104 settimane sono troppo poche.
- Nessuna selezione dei risultati: tutte le tabelle della sezione 4 vengono pubblicate.
