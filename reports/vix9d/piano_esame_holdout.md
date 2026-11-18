# Piano dell'esame del VIX9D sull'holdout

*Scritto nella Fase 9D-1 e **non eseguito**. Generato da `python -m src.vix9d.run_fase1` (modulo `src/vix9d/piano.py`), che non apre l'holdout.
Non esiste alcun comando che esegua l'esame: andrà scritto quando l'esame verrà deciso. L'esame del VIX (versioni F1 e FS, `reports/piano_esame_holdout.md`) è un'altra cosa e resta anch'esso da fare.*

## 1. Che cosa viene esaminato

- **Due versioni congelate**: G1 (solo storico del VIX9D) e GS (con il VIX), descritte in `versioni/vix9d/g1.json` e `versioni/vix9d/gs.json` con l'impronta dei sorgenti e il commit.
  GS differisce da G1 in: chiusura e fasce della +1; chiusura e fasce della +2; direzione della +2; minimo della +2.
- **Periodo d'esame**: le 104 settimane successive all'08/11/2024 (`data/vix9d/vix9d_weekly_holdout.csv`), mai usate per analisi, stime o scelte.
- **Bersagli**: le due candele previste, settimana +1 e settimana +2. Le previsioni esaminate sono quelle la cui settimana prevista cade nel periodo d'esame: 104 per orizzonte.

## 2. Procedura

- Walk-forward con finestra espandibile, come nello sviluppo: a ogni origine i modelli vengono ristimati con tutta la storia fino all'origine, comprese le settimane d'esame già trascorse. Nessuna informazione successiva all'origine entra nella previsione.
- Modelli, variabili, finestre e soglie sono quelli dei file congelati; nessuna scelta viene rifatta sui dati d'esame.
- **GS e il VIX**: GS usa la chiusura settimanale del VIX della settimana di origine. Nell'esame questo significa leggere le chiusure del VIX del periodo di holdout **come variabile esplicativa**, non come bersaglio: non si calcola alcun errore di previsione del VIX.
  Va deciso prima se questo è accettabile mentre l'esame del VIX è ancora da fare; in alternativa i due esami si eseguono insieme. Se il VIX di un'origine manca, GS usa la previsione di G1 (è il ripiego del programma) e il numero di ripieghi viene riportato.
- **Una sola esecuzione**, con un comando dedicato che scrive tutte le tabelle. Se dopo l'esecuzione emerge un errore di programma, lo si corregge dichiarandolo e riportando i risultati prima e dopo. I risultati si riportano comunque, qualunque sia l'esito.

## 3. Metriche

Per ogni versione e per ogni orizzonte (+1 e +2), con il numero di casi:

| ambito | metriche |
|---|---|
| chiusura centrale | errore assoluto medio in punti; radice dell'errore quadratico medio in log; quota di previsioni entro il 5%, il 10% e il 20% dal valore reale |
| fasce | copertura e larghezza media (punti) delle fasce al 50%, 80% e 90%; CRPS sui 7 quantili |
| direzione ed etichetta | Brier della probabilità che la chiusura superi quella dell'origine; quota di etichette non neutre (probabilità fuori da 40%-60%) e loro accuratezza, separatamente per «rialzo» e «ribasso», anche alle soglie del 65% e del 70% |
| probabilità di livello | Brier delle probabilità di chiudere sopra 20, 25, 30, con la frequenza osservata dell'evento |
| massimo e minimo | quota entro il 5% e entro il 10% dal valore reale; errore percentuale mediano; errore mediano in punti; quota di errori oltre il 5% per eccesso e per difetto; quota con entrambi gli estremi entro il 5% |
| candele | errore assoluto medio su apertura, massimo, minimo e chiusura; sovrapposizione; chiusura reale nel corpo e nell'escursione; corpo ed escursione previsti rispetto ai reali; quota di corpi tagliati dagli estremi |

Ogni tabella è riportata **su tutte le settimane e senza le settimane di strappo** (chiusura reale oltre +20% sulla chiusura all'origine): è un'etichetta nota solo a posteriori e non entra nei test.
Riferimenti riportati accanto: random walk per la chiusura, frequenza storica per la direzione, regola fissa per massimo e minimo. Sotto 20 casi lo si scrive e non si commenta.
Intervalli di confidenza: bootstrap a blocchi (26 settimane, 2000 repliche, seed del progetto). Accanto a ogni numero dell'esame, lo stesso numero misurato nello sviluppo (`reports/vix9d/e_versioni.csv`).

## 4. Criteri fissati ora

**Primo confronto: G1 contro i riferimenti semplici.** Dice se la previsione vale qualcosa fuori campione.

| criterio | perdita | riferimento |
|---|---|---|
| R-A. chiusura | errore quadratico in log della chiusura centrale | random walk (chiusura dell'origine) |
| R-B. direzione | Brier della probabilità di rialzo | frequenza storica dei rialzi |
| R-C. massimo e minimo | errore assoluto in log dell'estremo previsto | regola fissa |

**Secondo confronto: GS contro G1**, sulle stesse previsioni.

| criterio | perdita, per ogni origine |
|---|---|
| A. chiusura | errore quadratico in log della chiusura centrale |
| B. direzione | Brier della probabilità che la chiusura superi quella dell'origine |
| C. fasce | CRPS sui 7 quantili |
| D. massimo e minimo | errore assoluto in log dell'estremo previsto; in più, 0 se l'estremo previsto è entro il 5% dal reale e 1 altrimenti |

- Test di Diebold-Mariano a due code con la correzione per piccoli campioni, sulle differenze di perdita. Soglia: 5% dopo la correzione di Holm dentro ogni criterio (i test di un criterio sono quelli dei due orizzonti e, per massimo e minimo, dei due estremi).
- Dove le due versioni danno per costruzione lo stesso valore, il test non è calcolabile e lo si scrive. Con le scelte congelate questo riguarda: massimo della +1; massimo della +2; minimo della +1; direzione della +1.

**Come si leggono i risultati:**

- Una versione è dichiarata **migliore dell'altra (o del riferimento) su un criterio** se in almeno uno dei test del criterio la sua perdita è più bassa con p corretto sotto il 5%, e in nessuno degli altri test dello stesso criterio è significativamente peggiore. **Peggiore** nel caso simmetrico.
- In tutti gli altri casi le due sono dichiarate **non distinguibili** su quel criterio. Non verrà dichiarata un'equivalenza: con 104 settimane un test può non vedere differenze piccole come quelle misurate nello sviluppo, quindi «non distinguibili» è un esito probabile e non dimostra che le versioni siano uguali.
- **Verdetto complessivo.** GS è dichiarata migliore di G1 solo se è migliore su almeno uno dei criteri A-D e non è peggiore su nessuno. Altrimenti **resta G1**, perché a parità di risultati usa meno dati.
- **Coerenza con lo sviluppo.** Per ogni confronto si dice anche se il segno della differenza nell'esame è lo stesso misurato nello sviluppo.
- Copertura delle fasce, probabilità di livello, pagella delle candele, tabelle senza strappi e per fascia di livello sono **descrittive**: si riportano con gli intervalli ma non entrano nel verdetto.
- La decisione su quale versione usare dopo l'esame resta a chi dirige il progetto: questo piano fissa solo che cosa l'esame potrà affermare.

## 5. Che cosa non verrà fatto

- Nessuna nuova scelta di modelli, variabili, finestre o soglie dopo aver visto i dati d'esame.
- Nessuna seconda esecuzione «migliorata»: se dopo l'esame si vorrà cambiare qualcosa, sarà un lavoro nuovo, valutabile solo su dati ancora successivi.
- Nessun uso dei risultati dell'esame del VIX9D per ritoccare le versioni F1 e FS del VIX, né viceversa.
- L'esame non viene eseguito in questa fase: il periodo d'esame resta chiuso.

Versioni: G1 solo storico del VIX9D; GS con il VIX.
