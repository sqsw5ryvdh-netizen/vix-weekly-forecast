"""Genera reports/piano_esame_holdout.md: il piano dell'esame sull'holdout, SCRITTO E NON ESEGUITO.

Il piano fissa ora versioni, metriche, tabelle e criteri. Questo modulo non apre l'holdout: usa solo costanti del
progetto e i file di reports/fase4 (elenco delle versioni e dimensioni dei campioni di sviluppo).
"""
from __future__ import annotations

import json

from .. import config as C
from ..fase2 import densita as D
from ..fase2b import modello as MO
from ..fase3 import candele as K3
from . import regimi as RE
from . import versioni as VS

R = C.REPORTS_DIR / "fase4"
ALFA = 0.05
ORIZZONTI_PRINCIPALI = (1, 4)
FASCIA = (13, 22)


def run() -> None:
    a, b, c = (json.loads((R / f"{k}_info.json").read_text()) for k in ("a", "b", "c"))
    versioni = c["versioni"]
    altre = [v for v in versioni if v != "f1"]
    fine_dev = a["per_orizzonte"]["1"]["ultima_data_obiettivo"]
    n_test = len(altre) * len(ORIZZONTI_PRINCIPALI)
    f3 = "f3" in versioni
    out = f"""# Piano d'esame sull'holdout

*Scritto nella Fase 4, prima di aprire l'holdout. Generato da `python -m src.run_phase4` (modulo `src/fase4/piano.py`). L'esame non è stato eseguito:
questo documento fissa che cosa verrà fatto e come verranno lette le risposte. Dopo questa fase i modelli sono chiusi.*

## 1. Che cosa viene esaminato

Le versioni congelate in `versioni/*.json` (parametri, finestre, regole, impronta dei file sorgente e commit del congelamento):

{chr(10).join(f'- **{VS.NOMI[v]}** (`versioni/{v}.json`).' for v in versioni)}

Nessun parametro, nessuna finestra e nessuna regola potrà essere cambiata dopo l'apertura dell'holdout. Prima dell'esame si verifica che i file sorgente
elencati nelle configurazioni abbiano ancora la stessa impronta SHA-256 e che tutti i test passino; l'esito della verifica va scritto nel report d'esame.

## 2. Dati e procedura

- **Periodo d'esame:** le {C.HOLDOUT_WEEKS} settimane successive al {fine_dev} (`data/processed/vix_weekly_holdout.csv`), mai lette finora da nessuna analisi.
  Per F2 anche la parte di holdout dei dati giornalieri di VIX e VIX3M (`data/external/holdout`), anch'essa mai letta.
- **Obiettivi:** la chiusura di ogni settimana dell'holdout, prevista a {', '.join(str(h) for h in MO.ORIZZONTI)} settimane di distanza. Per ogni orizzonte h le origini sono le settimane
  la cui settimana-obiettivo cade nell'holdout: {C.HOLDOUT_WEEKS} previsioni per orizzonte (le prime origini sono le ultime settimane del campione di sviluppo).
- **Procedura:** lo stesso walk-forward dello sviluppo. A ogni origine i modelli sono ristimati con tutta la storia fino a quell'origine (sviluppo più le settimane di
  holdout già trascorse), con le finestre e le cadenze di ristima congelate{f" (per F3: ristima ogni {RE.PASSO_STIMA} settimane alle date fisse già in uso)" if f3 else ""}. Nessuna informazione successiva all'origine entra nella previsione.
- **F2 quando manca VIX3M:** se X1 non è valida in una settimana di origine, F2 usa la previsione di F1 per quell'origine (è il ripiego del programma). Il numero di ripieghi viene riportato;
  le metriche di F2 sono calcolate sia su tutte le origini (ripieghi compresi) sia sulle sole origini senza ripiego.
- **Una sola esecuzione.** L'esame si esegue una volta, con un comando dedicato che scrive tutte le tabelle. Se dopo l'esecuzione emerge un errore di programma, lo si corregge
  dichiarandolo e riportando i risultati prima e dopo la correzione. I risultati si riportano comunque, qualunque sia l'esito.

## 3. Metriche

Per ogni versione e per ogni orizzonte (h = {ORIZZONTI_PRINCIPALI[0]} e h = {ORIZZONTI_PRINCIPALI[1]} in evidenza; h = 2 e 3 riportati):

| ambito | metriche |
|---|---|
| chiusura centrale | errore assoluto medio in punti; radice dell'errore quadratico medio in log; quota di previsioni entro il 10% e entro il 20% dal valore reale |
| centro (M1, M1 + X1, media del modello a regimi) | errore assoluto medio in punti; radice dell'errore quadratico medio in log |
| fasce | copertura e larghezza media (punti) delle fasce al 50%, 80% e 90%; CRPS sulla griglia di {len(D.TAUS)} quantili; perdita quantilica ai quantili riportati |
| direzione ed etichetta | Brier della probabilità che la chiusura superi quella dell'origine; quota di etichette «neutre» (probabilità tra {int(100 * K3.NEUTRA[0])}% e {int(100 * K3.NEUTRA[1])}%); accuratezza delle etichette non neutre, con il loro numero |
| probabilità di livello | Brier delle probabilità di chiudere sopra {', '.join(str(int(s)) for s in MO.SOGLIE)}, con la frequenza osservata dell'evento |
| candele | la pagella della Fase 2c: rapporto corpo ed escursione previsti / reali, sovrapposizione, errori medi di apertura, massimo, minimo e chiusura, quota di chiusure reali dentro l'escursione prevista |

Riferimenti riportati accanto, per orientarsi: «nessuna variazione» (chiusura centrale = chiusura dell'origine) e la frequenza storica per la direzione.
Tutte le metriche anche per la **fascia {FASCIA[0]}-{FASCIA[1]}** (chiusura all'origine tra {FASCIA[0]} e {FASCIA[1]}, estremi inclusi), con il numero di casi; sotto 20 casi lo si scrive e non si commenta.
Intervalli di confidenza: bootstrap a blocchi (26 settimane, 2000 repliche, seed del progetto), come nello sviluppo.

## 4. Tabelle e figure che verranno prodotte

1. Chiusura centrale e centro: una riga per versione e orizzonte.
2. Fasce: copertura, larghezza, CRPS.
3. Direzione ed etichetta.
4. Probabilità di superare {', '.join(str(int(s)) for s in MO.SOGLIE)}.
5. Pagella delle candele.
6. Confronti a coppie con F1 (sezione 5): differenze, intervalli bootstrap, test, p-value corretti.
7. Le stesse tabelle 1-5 per la fascia {FASCIA[0]}-{FASCIA[1]}.
8. Diagnostica: numero di ripieghi di F2{', stime non valide e probabilità di regime di F3' if f3 else ''}, quantili incrociati, origini usate.
9. Figure: chiusura reale e chiusure centrali previste a 1 e a 4 settimane nel periodo d'esame; copertura delle fasce nel tempo.
10. Accanto a ogni numero dell'esame, lo stesso numero misurato nello sviluppo (`reports/fase4/c_confronto.csv` e fasi precedenti), per vedere quanto regge fuori campione.

## 5. Criteri fissati ora

F1 è la versione di riferimento: usa solo lo storico settimanale del VIX. Ogni altra versione ({', '.join(VS.NOMI[v] for v in altre)}) è confrontata con F1 sulle stesse {C.HOLDOUT_WEEKS} previsioni per orizzonte.

**Tre criteri, ciascuno con il suo test** (Diebold-Mariano a due code con la correzione per piccoli campioni, sulle differenze di perdita tra F1 e la versione):

| criterio | perdita |
|---|---|
| A. chiusura | errore quadratico in log della chiusura centrale |
| B. direzione | Brier della probabilità che la chiusura superi quella dell'origine |
| C. fasce | CRPS sulla griglia di {len(D.TAUS)} quantili |

Per ogni criterio i test sono {n_test} ({len(altre)} versioni × h = {ORIZZONTI_PRINCIPALI[0]} e h = {ORIZZONTI_PRINCIPALI[1]}) e i p-value sono corretti con Holm dentro il criterio. Soglia: {int(100 * ALFA)}%.

**Come si leggono i risultati:**

- Una versione è dichiarata **migliore di F1 su un criterio** se a h = {ORIZZONTI_PRINCIPALI[0]} oppure a h = {ORIZZONTI_PRINCIPALI[1]} la sua perdita è più bassa con p corretto sotto il {int(100 * ALFA)}%, e all'altro orizzonte non è significativamente peggiore.
- È dichiarata **peggiore di F1 su un criterio** nel caso simmetrico.
- In tutti gli altri casi le due versioni sono dichiarate **non distinguibili** su quel criterio. Non verrà dichiarata un'equivalenza: con {C.HOLDOUT_WEEKS} settimane un test può non vedere differenze piccole come quelle
  misurate nello sviluppo (pochi punti percentuali), quindi «non distinguibili» è l'esito più probabile e non dimostra che le versioni siano uguali.
- **Verdetto complessivo.** Una versione è dichiarata migliore di F1 nel complesso solo se è migliore su almeno un criterio e non è peggiore su nessuno. Se nessuna versione lo è, **resta F1**, perché a parità di risultati
  usa meno dati e meno ipotesi. Se più versioni lo sono, si riportano entrambe con i criteri su cui prevalgono; tra loro non si fa una classifica oltre a quella descrittiva delle tabelle.
- **Coerenza con lo sviluppo.** Per ogni confronto si dice anche se il segno della differenza nell'esame è lo stesso misurato nello sviluppo.
- Copertura delle fasce, pagella delle candele, probabilità di livello e fascia {FASCIA[0]}-{FASCIA[1]} sono **descrittive**: si riportano con gli intervalli, ma non entrano nel verdetto.
- La decisione su quale versione usare nel programma dopo l'esame resta a chi dirige il progetto: questo piano fissa solo che cosa l'esame potrà affermare.

## 6. Che cosa non verrà fatto

- Nessuna ristima di iperparametri, nessuna variante nuova, nessuna ricalibrazione delle fasce dopo aver visto l'holdout.
- Nessun sottoperiodo dentro l'holdout oltre alla fascia {FASCIA[0]}-{FASCIA[1]}: {C.HOLDOUT_WEEKS} settimane sono troppo poche.
- Nessuna selezione dei risultati: tutte le tabelle della sezione 4 vengono pubblicate.
"""
    (C.REPORTS_DIR / "piano_esame_holdout.md").write_text(out)


if __name__ == "__main__":
    run()
