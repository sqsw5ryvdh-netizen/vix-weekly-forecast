# Piano d'esame sull'holdout

*Aggiornato nella Fase 4b, prima di aprire l'holdout. Generato da `python -m src.run_phase4b` (modulo `src/fase4b/piano.py`). Sostituisce il piano della Fase 4, conservato in
`reports/fase4/piano_esame_holdout_fase4.md`. L'esame non è stato eseguito: questo documento fissa che cosa verrà fatto e come verranno lette le risposte.*

## 1. Che cosa viene esaminato

Due versioni, congelate in `versioni/*.json` (parametri, finestre, regole, impronta dei file sorgente e commit del congelamento):

- **F1 solo storico del VIX** (`versioni/f1.json`): chiusura centrale e fasce D4, probabilità ed etichetta D1, candele K2 con ampiezze A1.
- **FS sfidante** (`versioni/fs.json`): costruita attorno al candidato P5 media di F2 e F3. Per la settimana +1 la probabilità che la chiusura superi quella dell'origine è la media semplice delle
  probabilità di F2 e F3; chiusura centrale, fasce, probabilità di livello ed etichette e candele delle settimane +2…+4 vengono dalla costruzione di F2.
- In entrambe, la **prima candela** (settimana +1) usa la costruzione K2 con il colore legato all'etichetta e il minimo M1q.

Le versioni F2 e F3 della Fase 4 non vanno all'esame come versioni a sé: entrano solo come componenti di FS.
Nessun parametro, nessuna finestra e nessuna regola potrà essere cambiata dopo l'apertura dell'holdout. Prima dell'esame si verifica che i file sorgente elencati nelle configurazioni abbiano ancora la stessa
impronta SHA-256 e che tutti i test passino; l'esito della verifica va scritto nel report d'esame.

## 2. Dati e procedura

- **Periodo d'esame:** le 104 settimane successive al 2024-11-08 (`data/processed/vix_weekly_holdout.csv`), mai lette finora da nessuna analisi. Per FS anche la parte di holdout dei dati giornalieri di VIX e VIX3M (`data/external/holdout`), anch'essa mai letta.
- **Una previsione per settimana, alla chiusura settimanale**, mai aggiornata durante la settimana.
- **Obiettivi:** la chiusura di ogni settimana dell'holdout, prevista a 1, 2, 3, 4 settimane di distanza, e per la settimana +1 anche apertura, massimo, minimo e chiusura (la candela).
  Per ogni orizzonte h le origini sono le settimane la cui settimana-obiettivo cade nell'holdout: 104 previsioni per orizzonte (le prime origini sono le ultime settimane del campione di sviluppo).
- **Procedura:** lo stesso walk-forward dello sviluppo. A ogni origine i modelli sono ristimati con tutta la storia fino a quell'origine (sviluppo più le settimane di holdout già trascorse), con le finestre e le
  cadenze di ristima congelate. Nessuna informazione successiva all'origine entra nella previsione.
- **FS quando manca VIX3M:** se la variabile VIX3M / VIX − 1 non è valida in una settimana di origine, FS usa la previsione di F1 per quell'origine (è il ripiego del programma). Il numero di ripieghi viene riportato;
  le metriche di FS sono calcolate sia su tutte le origini (ripieghi compresi) sia sulle sole origini senza ripiego. Il verdetto usa tutte le origini.
- **Una sola esecuzione.** L'esame si esegue una volta, con un comando dedicato che scrive tutte le tabelle. Se dopo l'esecuzione emerge un errore di programma, lo si corregge dichiarandolo e riportando i risultati
  prima e dopo la correzione. I risultati si riportano comunque, qualunque sia l'esito.

## 3. Metriche

Per ogni versione e per ogni orizzonte (h = 1 e h = 4 in evidenza; h = 2 e 3 riportati):

| ambito | metriche |
|---|---|
| chiusura centrale | errore assoluto medio in punti; radice dell'errore quadratico medio in log; quota di previsioni entro il 10% e entro il 20% dal valore reale |
| fasce | copertura e larghezza media (punti) delle fasce al 50%, 80% e 90%; CRPS sulla griglia di 19 quantili; perdita quantilica ai quantili riportati |
| direzione ed etichetta | Brier della probabilità che la chiusura superi quella dell'origine; quota di etichette non neutre (probabilità fuori da 40%-60%); accuratezza delle non neutre, separatamente per «rialzo» e «ribasso», con il loro numero |
| probabilità di livello | Brier delle probabilità di chiudere sopra 20, 25, 30, con la frequenza osservata dell'evento |
| prima candela (+1) | errore assoluto medio e mediano su apertura, massimo, minimo e chiusura; quota di minimi entro il 5% e il 10%; quota di minimi previsti troppo bassi; quota di massimi reali sopra il previsto (riportata, non giudicata); sovrapposizione; chiusura reale nel corpo e nell'escursione |
| candele +2…+4 | la pagella della Fase 2c |

Riferimenti riportati accanto, per orientarsi: «nessuna variazione» (chiusura centrale = chiusura dell'origine) e la frequenza storica per la direzione.
Tutte le metriche anche per la **fascia 13-22** (chiusura all'origine tra 13 e 22, estremi inclusi), con il numero di casi; sotto 20 casi lo si scrive e non si commenta.
Intervalli di confidenza: bootstrap a blocchi (26 settimane, 2000 repliche, seed del progetto), come nello sviluppo.

## 4. Tabelle e figure che verranno prodotte

1. Chiusura centrale: una riga per versione e orizzonte.
2. Fasce: copertura, larghezza, CRPS.
3. Direzione ed etichetta, con «rialzo» e «ribasso» separati.
4. Probabilità di superare 20, 25, 30.
5. Prima candela: tutte le misure della riga dedicata, per versione.
6. Pagella delle candele +2…+4.
7. Confronto FS contro F1 (sezione 5): differenze, intervalli bootstrap, test, p-value corretti, per i quattro criteri.
8. Le stesse tabelle 1-5 per la fascia 13-22.
9. Diagnostica: numero di ripieghi di FS, stime non valide del modello a regimi, quantili incrociati, minimi limitati dal corpo, origini usate.
10. Figure: chiusura reale e chiusure centrali previste a 1 e a 4 settimane nel periodo d'esame; minimo previsto e reale della settimana +1.
11. Accanto a ogni numero dell'esame, lo stesso numero misurato nello sviluppo (`reports/fase4b/`, `reports/fase4/c_confronto.csv` e fasi precedenti), per vedere quanto regge fuori campione.

## 5. Criteri fissati ora

F1 è la versione di riferimento: usa solo lo storico settimanale del VIX. FS è confrontata con F1 sulle stesse 104 previsioni per orizzonte.
I test sono di Diebold-Mariano a due code, con la correzione per piccoli campioni, sulle differenze di perdita tra F1 e FS. Soglia: 5% dopo la correzione di Holm dentro ogni criterio.

**Criteri A, B, C (gli stessi del piano precedente)**, ciascuno con due test (h = 1 e h = 4):

| criterio | perdita |
|---|---|
| A. chiusura | errore quadratico in log della chiusura centrale |
| B. direzione | Brier della probabilità che la chiusura superi quella dell'origine |
| C. fasce | CRPS sulla griglia di 19 quantili |

**Criterio D, dedicato alla prima candela** (settimana +1), con tre test:

| componente | perdita, per ogni origine |
|---|---|
| D1. direzione quando non neutra | 0 se l'etichetta non è neutra ed è corretta; 1 se non è neutra ed è sbagliata; 0,5 se è neutra |
| D2. minimo | errore assoluto in punti tra minimo previsto e minimo reale |
| D3. chiusura | errore assoluto in punti tra chiusura della candela e chiusura reale |

La perdita D1 dà mezzo punto all'etichetta neutra: un'etichetta non neutra conviene solo se è corretta più di una volta su due, e chi non si sbilancia mai non viene premiato né punito.
Accanto a D1 vengono riportate comunque quota e accuratezza delle etichette non neutre, per «rialzo» e per «ribasso». Il massimo non entra nel criterio: un massimo reale sopra il previsto è accettato.

**Come si leggono i risultati:**

- FS è dichiarata **migliore di F1 su un criterio** se in almeno uno dei test del criterio la sua perdita è più bassa con p corretto sotto il 5%, e in nessuno degli altri test dello stesso criterio è significativamente peggiore.
- È dichiarata **peggiore di F1 su un criterio** nel caso simmetrico.
- In tutti gli altri casi le due versioni sono dichiarate **non distinguibili** su quel criterio. Non verrà dichiarata un'equivalenza: con 104 settimane un test può non vedere differenze piccole come quelle
  misurate nello sviluppo (pochi punti percentuali), quindi «non distinguibili» è l'esito più probabile e non dimostra che le versioni siano uguali.
- **Verdetto complessivo.** FS è dichiarata migliore di F1 nel complesso solo se è migliore su almeno uno dei quattro criteri e non è peggiore su nessuno. Altrimenti **resta F1**, perché a parità di risultati usa meno dati e meno ipotesi.
- **Verdetto sulla prima candela.** Il criterio D viene riportato anche da solo, perché le priorità dichiarate per la prima candela sono chiusura, minimo e direzione.
  Dove le due versioni danno per costruzione lo stesso valore (per esempio lo stesso minimo), il test non è calcolabile e lo si scrive.
- **Coerenza con lo sviluppo.** Per ogni confronto si dice anche se il segno della differenza nell'esame è lo stesso misurato nello sviluppo.
- Copertura delle fasce, candele +2…+4, probabilità di livello, massimo e fascia 13-22 sono **descrittivi**: si riportano con gli intervalli, ma non entrano nel verdetto.
- La decisione su quale versione usare nel programma dopo l'esame resta a chi dirige il progetto: questo piano fissa solo che cosa l'esame potrà affermare.

## 6. Che cosa non verrà fatto

- Nessuna ristima di iperparametri, nessuna variante nuova, nessuna ricalibrazione delle fasce o del massimo dopo aver visto l'holdout.
- Nessun sottoperiodo dentro l'holdout oltre alla fascia 13-22: 104 settimane sono troppo poche.
- Nessuna selezione dei risultati: tutte le tabelle della sezione 4 vengono pubblicate.
