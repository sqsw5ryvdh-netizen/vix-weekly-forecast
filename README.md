# Analisi e previsione del VIX settimanale

Progetto universitario: analisi e previsione dell'indice di volatilità VIX su base settimanale.
Il lavoro procede per fasi; ogni fase si chiude con un report in `reports/`.

| fase | contenuto | report |
|---|---|---|
| 0 | audit dei dati, analisi esplorativa, baseline walk-forward | [reports/fase0_report.md](reports/fase0_report.md) |
| 1 | pattern storici (dopo e prima dei picchi, periodi calmi, tabelle condizionate) e modelli che usano solo il VIX | [reports/fase1_report.md](reports/fase1_report.md) |
| 2 | previsione probabilistica attorno al tipo HAR (quantili, intervalli, probabilità di eventi), scheda di previsione, raccolta dei dati esterni | [reports/fase2_report.md](reports/fase2_report.md) |
| 2b | programma di previsione a 1-4 settimane con grafico a candele | [reports/fase2b_report.md](reports/fase2b_report.md) |
| 2c | candele previste realistiche e misurabili, registro delle previsioni | [reports/fase2c_report.md](reports/fase2c_report.md) |
| 3 | nuova definizione delle candele (Parte 0); quanto aggiungono il VIX giornaliero e i dati di mercato a chiusura, direzione e fasce | [reports/fase3_report.md](reports/fase3_report.md) |
| 4 | ultima fase di costruzione: dimensioni delle candele, modelli a regimi e gradient boosting, versioni finaliste | [reports/fase4_report.md](reports/fase4_report.md) |
| 4b | la prima candela (direzione, corpo e colore, minimo), scelta della sfidante | [reports/fase4b_report.md](reports/fase4b_report.md) |
| 4c | massimo e minimo della prima candela previsti direttamente (candela del VIX, VVIX, S&P 500, VIX3M, previsione incrociata), nuovo congelamento (`versioni/`), piano d'esame aggiornato | [reports/fase4c_report.md](reports/fase4c_report.md) · [piano d'esame](reports/piano_esame_holdout.md) · previsione corrente: [reports/previsione_corrente.md](reports/previsione_corrente.md) |
| 9D-1 | secondo indice, il **VIX9D**: fondamenta, modelli e programma per due candele settimanali | [reports/vix9d/fase1_report.md](reports/vix9d/fase1_report.md) · [piano d'esame](reports/vix9d/piano_esame_holdout.md) · previsione corrente: [reports/vix9d/previsione_corrente.md](reports/vix9d/previsione_corrente.md) |

**Stato:** i modelli sono chiusi. All'esame vanno due versioni, F1 e la sfidante FS, congelate in `versioni/f1.json` e `versioni/fs.json`; resta solo
l'esame sull'holdout, descritto in `reports/piano_esame_holdout.md` e non ancora eseguito. L'holdout non è mai stato aperto.
Il lavoro sul VIX è fermo qui. Per il VIX9D sono congelate le versioni G1 e GS (`versioni/vix9d/`); anche il loro esame è solo scritto.

## Il secondo indice: VIX9D (Fase 9D-1)

Stesso metodo, altro indice: il **VIX9D** (volatilità implicita a 9 giorni, dati giornalieri ufficiali Cboe), su base settimanale,
con previsione di **due candele** (settimane +1 e +2). Codice, dati, risultati e test stanno in cartelle proprie
(`src/vix9d/`, `data/vix9d/`, `reports/vix9d/`, `tests/vix9d/`) e riusano le funzioni del VIX senza modificarle.

| che cosa | dove |
|---|---|
| report della fase (audit, analisi, modelli, quanto aggiunge il VIX, confronto VIX / VIX9D) | [reports/vix9d/fase1_report.md](reports/vix9d/fase1_report.md) |
| piano d'esame sull'holdout (scritto, non eseguito) | [reports/vix9d/piano_esame_holdout.md](reports/vix9d/piano_esame_holdout.md) |
| previsione corrente | [reports/vix9d/previsione_corrente.md](reports/vix9d/previsione_corrente.md) |
| versioni congelate | `versioni/vix9d/g1.json` (solo VIX9D), `versioni/vix9d/gs.json` (con il VIX) |

```bash
python -m src.vix9d.run_fase1                  # candele settimanali e audit, analisi, walk-forward, regole, report e piano d'esame (circa 90 secondi)
python -m src.vix9d.congela                    # scrive versioni/vix9d/g1.json e gs.json: da eseguire dopo aver registrato il codice
python -m src.vix9d.forecast                   # previsione dall'ultima chiusura settimanale completa, versione g1
python -m src.vix9d.forecast --aggiorna        # come sopra, scaricando prima VIX9D e VIX giornalieri da Cboe
python -m src.vix9d.forecast --versione gs     # versione con il VIX come principale; se il VIX manca, avvisa e ripiega su g1
python -m src.vix9d.forecast --verifica        # confronta le previsioni registrate con le settimane completate
python -m src.vix9d.prepara --scarica          # riscarica i file giornalieri Cboe e ricostruisce data/vix9d/ (non deterministico: dipende dal giorno)
```

- **Dati.** Il file Cboe ha apertura, massimo e minimo solo da metà novembre 2013: le chiusure sono usate dal 2011, le candele solo dove sono vere.
  Stesso confine del VIX: sviluppo fino all'8 novembre 2024; le 104 settimane successive sono in `data/vix9d/vix9d_weekly_holdout.csv`, bloccate.
- **Versioni.** `g1` usa solo lo storico del VIX9D (modello tipo HAR per la chiusura, regressione quantilica per le fasce, logistica per la direzione,
  regressione sulla mediana per massimo e minimo). `gs` aggiunge ln(VIX9D / VIX) solo nei componenti in cui batte `g1` secondo la regola fissata prima.
- **Guardia.** `run_fase1` legge solo i file di sviluppo; `src.vix9d.forecast` è l'unico comando del VIX9D che legge i dati successivi al confine, solo
  come storia da cui prevedere. L'esame sull'holdout non è stato eseguito e non esiste un comando che lo esegua.

## Il programma di previsione

```bash
python -m src.forecast                                   # versione f1, dall'ultima chiusura settimanale completa
python -m src.forecast --aggiorna                        # come sopra, scaricando prima VIX e VIX3M giornalieri (servono al confronto con fs)
python -m src.forecast --versione fs --aggiorna          # sfidante come versione principale; se VIX3M manca, avvisa e ripiega su f1
python -m src.forecast --origine 2025-06-27              # previsione "come se fossimo" a quella data
python -m src.forecast --file percorso/del/file.csv      # con un file aggiornato (stesso formato del file grezzo)
python -m src.forecast --verifica                        # confronta le previsioni registrate con le settimane completate, per versione
```

La previsione si fa **una sola volta, alla chiusura settimanale**, e non si aggiorna durante la settimana. Le due versioni (congelate nella
Fase 4c, `versioni/*.json`):

| versione | informazioni usate | chiusura centrale e fasce | direzione della settimana +1 |
|---|---|---|---|
| `f1` (predefinita) | solo lo storico settimanale del VIX | regressione quantilica sulle variabili di M1 (D4) | errori passati attorno a M1 (D1) |
| `fs` (sfidante) | in più VIX3M / VIX − 1 (struttura a termine), dai file giornalieri Cboe | regressione quantilica con la variabile in più | media della probabilità con VIX3M/VIX e di quella di un modello a 2 regimi |

Nella **prima candela** (settimana +1) massimo e minimo sono previsti direttamente (regressione sulla mediana del rapporto con la chiusura
dell'origine) e comandano la candela: se il corpo uscirebbe dall'intervallo viene accorciato. La candela è colorata secondo l'etichetta della
chiusura (rossa «rialzo», blu «ribasso», grigia «neutra»). Le candele +2…+4 sono settimane di dimensioni medie.
`--aggiorna` scarica i due file giornalieri ufficiali in `data/external/forecast/` (non pubblicata); vengono usati solo i valori fino
all'ultima seduta della settimana di origine. Sul campione di sviluppo le differenze tra le versioni sono piccole: quale sia migliore lo
dirà l'esame sull'holdout.

Cosa fa: legge l'intero storico, esclude la settimana incompleta (riga con la colonna `note` valorizzata, oppure
ultima riga con meno sedute di quelle attese: in questo caso avvisa a schermo), stima i modelli sulla storia fino
all'origine e produce, per le settimane +1 … +4:

- una **candela prevista** (apertura, massimo, minimo, chiusura): la prima colorata secondo l'etichetta e con massimo e
  minimo previsti direttamente, le altre settimane di dimensioni medie con il corpo nella direzione storicamente più frequente;
- la **direzione della chiusura rispetto alla chiusura dell'origine**, con la sua probabilità: «rialzo», «ribasso»
  oppure «neutra» quando la probabilità è tra 40% e 60% (è la scritta sopra ogni candela nel grafico);
- la **chiusura centrale** (mediana), cioè il centro di tutti gli esiti possibili;
- le **fasce della chiusura** al 50%, 80% e 90%, ciascuna con la copertura misurata nel passato;
- le **probabilità** di chiudere sopra 20, 25 e 30;
- una **lettura della situazione** (livello, riferimenti, dinamica, contesto, precedenti, casi simili) con una sintesi in poche frasi.

Cosa scrive:

| file | contenuto |
|---|---|
| `reports/previsione_corrente.md` | scheda completa della versione f1: sintesi, lettura, tabella, confronto con le altre versioni, grafico, guida alla lettura (solo senza `--origine`) |
| `reports/previsione/previsione_AAAA-MM-GG.md` | la stessa scheda quando si indica `--origine`; con `--versione fs` i file hanno il suffisso `_fs` |
| `reports/previsione/confronto_AAAA-MM-GG.csv` | chiusura centrale, fasce, etichetta e probabilità delle due versioni |
| `reports/previsione/fs_valori_usati_AAAA-MM-GG.csv` | la serie settimanale di VIX3M / VIX − 1 usata da fs per quell'origine |
| `reports/previsione/fs_variabili_estremi_AAAA-MM-GG.csv` | solo se gli insiemi di fs usano altre serie giornaliere: le variabili usate per massimo e minimo della prima candela |
| `reports/previsione/tabella_AAAA-MM-GG.csv` | tabella della previsione (stampata anche a schermo) |
| `reports/previsione/previsione_AAAA-MM-GG.json` | lettura della situazione e previsione in formato JSON |
| `reports/previsione/g1_AAAA-MM-GG.png` | grafico principale: ultimi 2 anni a candele, 4 candele previste, fasce, casi simili |
| `reports/previsione/g2_AAAA-MM-GG.png` | lo stesso su tutto lo storico dal 2000 |
| `reports/registro_previsioni.csv` | registro: ogni esecuzione senza `--origine` aggiunge candele e fasce previste di ogni versione calcolata; la stessa coppia origine + versione non viene mai duplicata né riscritta |
| `reports/previsione/registro_verifica_*.csv` | con `--verifica`: confronto tra il registro e le settimane nel frattempo completate (pagella dal vivo) |

Per aggiornare la previsione basta aggiungere le nuove settimane al file (o passarne uno nuovo con `--file`) e
rieseguire il comando. Il file deve avere le colonne `week_start, last_trading_day, open, high, low, close,
trading_days, note`; servono almeno 7 anni circa di storia.

Come leggere il grafico: le candele piene sono settimane reali, quelle vuote con bordo tratteggiato sono previste.
**La prima candela prevista è piena e ha il colore dell'etichetta: rossa se la chiusura è prevista sopra quella
dell'origine, blu se sotto, grigia se «neutra»; la scritta sopra ogni candela dà la direzione e la sua probabilità; la
linea viola (chiusura centrale) è il centro degli esiti possibili; le fasce colorate indicano l'incertezza sulla
chiusura; la linea arancione spessa è il livello recente (media a 12 settimane), riferimento principale.** Le candele
+2…+4 sono vuote e blu perché l'apertura ufficiale del VIX è di norma più alta della chiusura precedente e la
settimana chiude più spesso sotto la propria apertura: il loro colore non dice dove chiuderà il VIX rispetto a oggi.
Nella prima candela il corpo resta disegnato verso il basso: il colore indica la direzione della chiusura rispetto
all'origine, non il verso del corpo. Le fasce coprono meno del valore nominale: la copertura reale è scritta in tabella.

Il registro serve a misurare le previsioni dal vivo, non a scegliere modelli: dopo ogni aggiornamento del file,
`python -m src.forecast` registra la nuova previsione e `python -m src.forecast --verifica` mostra come sono andate
quelle precedenti.

Limiti: non anticipa i picchi; è un progetto universitario, non un consiglio di investimento. `src.forecast` è
l'unico comando che legge i dati successivi all'8 novembre 2024 (il VIX e, per fs, VIX3M), e solo come storia da cui
prevedere: errori e coperture mostrati nella scheda sono quelli misurati sul campione di sviluppo e raccolti da
`python -m src.run_phase4c`. Solo `--verifica` confronta
previsioni e dati recenti, e solo per le previsioni già scritte nel registro. Le previsioni registrate prima della
Fase 4 restano nel registro senza versione, con le candele di allora (colonna `costruzione`).

## Come riprodurre

Richiede Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.run_phase0   # Fase 0: dataset puliti, tabelle, figure e report
python -m src.run_phase1   # Fase 1: pattern storici, modelli M0-M7, report (circa 2 minuti)
python -m src.run_phase2   # Fase 2: modelli di densità, eventi, scheda di previsione, report (circa 5 minuti)
python -m src.run_phase2b  # Fase 2b: misure per h = 1-4 su chiusura, massimo e minimo, grafico di verifica, report (circa 4 minuti)
python -m src.run_phase2c  # Fase 2c: componenti e pagella delle candele, regola di scelta, report (circa 1 minuto; richiede run_phase2b)
python -m src.run_phase3   # Fase 3: nuove candele, blocchi con VIX giornaliero e dati di mercato, report (circa 3 minuti; richiede run_phase2c e i dati esterni)
python -m src.run_phase4   # Fase 4: varianti delle candele, modelli a regimi e boosting, confronto tra le versioni, report e piano d'esame (circa 7 minuti; richiede run_phase3 e i dati esterni)
python -m src.run_phase4b  # Fase 4b: prima candela (direzione, corpo, minimo), sfidante, report e piano d'esame aggiornato (meno di un minuto; richiede run_phase4)
python -m src.run_phase4c  # Fase 4c: massimo e minimo diretti della prima candela, previsione incrociata con l'S&P 500, report e piano d'esame aggiornato (circa 2 minuti; richiede run_phase4b e i dati esterni)
python -m src.fase4c.congela  # scrive versioni/f1.json e versioni/fs.json (parametri, impronte dei sorgenti, commit): da eseguire dopo aver registrato il codice
python -m src.vix9d.run_fase1  # VIX9D, Fase 9D-1: candele e audit, analisi, modelli, versioni G1 e GS, report e piano d'esame (circa 90 secondi)
python -m src.vix9d.congela    # scrive versioni/vix9d/g1.json e gs.json: da eseguire dopo aver registrato il codice
python -m pytest           # test (anti look-ahead, integrità dell'holdout, metriche)
```

I comandi sono deterministici: rieseguendoli si ottengono gli stessi file. `run_phase1` e `run_phase2`
richiedono `data/processed/vix_weekly_dev.csv`, prodotto da `run_phase0`. Versioni e seed sono in
`reports/run_manifest.json`, `reports/fase1/run_manifest.json` e `reports/fase2/run_manifest.json`.

`run_phase3`, `run_phase4` e `run_phase4c` leggono anche `data/external/dev/esterni_giornalieri_dev.csv`, che **non è nel
repository**: va prodotto prima con `python -m src.fetch_external` (vedi sotto). Senza i dati esterni si possono
rigenerare le sole parti sulle candele (`python -m src.fase3.valuta_candele`, `python -m src.fase4.valuta_a`), e i
test che richiedono i dati esterni vengono saltati.

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
(`reports/fase2/c_inventario.json`). Nella Fase 2 i dati esterni sono solo raccolti. Dalla Fase 3 la parte di
sviluppo (VIX giornaliero, VIX3M, VVIX, S&P 500, future VX1 e VX2) è usata da `run_phase3` per misurare quanto
aggiunge a M1; l'indice delle condizioni finanziarie e i differenziali di credito sono esclusi. La parte di holdout
resta bloccata dalla guardia.

## Struttura

```
data/raw/          file originale, mai modificato
data/processed/    dataset puliti generati dal codice (+ TRASFORMAZIONI.md)
data/vix9d/        VIX9D: candele settimanali costruite dai dati giornalieri Cboe (sviluppo, holdout bloccato, file completi letti solo dal programma)
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
  forecast.py      programma di previsione (unico comando che legge l'intero file grezzo)
  run_phase2b.py   misure di sviluppo della Fase 2b, sotto la guardia
  fase2b/
    modello.py     previsione di chiusura, massimo e minimo; costruzione delle candele previste
    lettura.py     lettura della situazione e casi simili
    testo.py       tabella e testo della scheda
    grafici.py     grafici a candele G1, G2, G3
    valuta.py      walk-forward di sviluppo per h = 1-4 e grafico G3
    report.py      costruisce il report leggendo reports/fase2b/*.csv e *.json
  run_phase2c.py   misure di sviluppo della Fase 2c, sotto la guardia
  fase2c/
    componenti.py  apertura, direzione, ampiezza del corpo e ombre della candela (stima sulla storia fino all'origine)
    candele.py     elenco chiuso delle costruzioni K0-K3
    valuta.py      walk-forward, pagella delle candele, regola di scelta, grafico G3
    registro.py    registro delle previsioni e pagella dal vivo
    settimana.py   riconoscimento della settimana incompleta
    testo.py       tabella e testi della scheda
    report.py      costruisce il report leggendo reports/fase2c/*.csv e *.json
  run_phase3.py    misure di sviluppo della Fase 3, sotto la guardia
  fase3/
    candele.py         Parte 0: candele K2 con ampiezze medie, direzione L0, etichetta della chiusura (usato da forecast.py)
    valuta_candele.py  Parte 0: regressione contro media storica, nuova pagella, grafico G3
    testo.py           tabella e testi della scheda (usato da forecast.py)
    registro.py        registro con l'etichetta della chiusura (usato da forecast.py)
    dati.py            unico accesso ai dati esterni di sviluppo (solo le serie dell'elenco chiuso)
    variabili.py       variabili dei blocchi G1 e X1-X4, calcolate con i soli dati disponibili a fine settimana
    modelli.py         elenco chiuso: blocchi G1, X1-X4, combinazioni X6 (ridge) e X7 (media) per T1, T2, T3
    valuta.py          walk-forward, tabelle, regole di scelta, figure
    report.py          costruisce il report leggendo reports/fase3/*.csv e *.json
  run_phase4.py    misure di sviluppo della Fase 4, sotto la guardia
  fase4/
    candele.py         varianti A0-A2 delle ampiezze delle candele K2
    valuta_a.py        Parte A: pagella delle tre varianti e regola di scelta
    regimi.py          N1: autoregressivo a 2 regimi markoviani (ristima ogni 26 settimane)
    boosting.py        N2, N3: gradient boosting con iperparametri fissi (ristima ogni 13 settimane)
    valuta_b.py        Parte B: walk-forward, tabelle e verdetto sui modelli a regimi e non lineari
    versioni.py        versioni finaliste F1, F2, F3 e loro configurazione (usato da forecast.py)
    valuta_c.py        Parte C: confronto di sviluppo tra le versioni sulle stesse origini
    congela.py         scrive versioni/*.json con impronte dei sorgenti e commit
    esterni.py         dati giornalieri di VIX e VIX3M per f2 (usato solo da forecast.py)
    registro.py        registro delle previsioni con la versione (usato da forecast.py)
    testo.py           tabelle e testi della scheda, tabella di confronto (usato da forecast.py)
    piano.py           scrive il piano d'esame della Fase 4, ora in archivio (reports/fase4/piano_esame_holdout_fase4.md)
    report.py          costruisce il report leggendo reports/fase4/*.csv e *.json
  run_phase4b.py   misure di sviluppo della Fase 4b, sotto la guardia
  fase4b/
    prima.py           componenti della prima candela: mosse medie, costruzione K4, minimo M1q
    versioni.py        le due versioni che vanno all'esame, F1 e FS, e loro configurazione (usato da forecast.py)
    valuta.py          Parti A-D: candidati per la direzione, scelta della sfidante, K4 contro K2, M1q contro M0
    congela.py         congelamento della Fase 4b (superato da quello della Fase 4c)
    testo.py           testi della scheda
    piano.py           scrive il piano d'esame della Fase 4b, ora in archivio (reports/fase4b/piano_esame_holdout_fase4b.md)
    report.py          costruisce il report leggendo reports/fase4b/*.csv e *.json
  run_phase4c.py   misure di sviluppo della Fase 4c, sotto la guardia
  fase4c/
    variabili.py       variabili degli insiemi S1-S5 (candela del VIX, VVIX, S&P 500, VIX3M/VIX) e dell'S&P 500 per la previsione incrociata
    estremi.py         previsione diretta di massimo e minimo della settimana +1 (regressione sulla mediana, finestra di 520 settimane)
    incrociata.py      previsione incrociata con l'S&P 500: legame B1 e versione reale B3
    tetto.py           B2, tetto teorico con l'S&P 500 futuro: solo misura, mai nel programma
    candela.py         prima candela comandata dagli estremi: corpo K2 o K5, tagliato dentro l'intervallo minimo-massimo
    versioni.py        F1 e FS con la nuova prima candela e loro configurazione (usato da forecast.py)
    valuta.py          Parti A-D: insiemi, regole di scelta, previsione incrociata, corpo, pagella prima e dopo
    esterni.py         serie giornaliere che servono a FS nel programma (usato solo da forecast.py)
    congela.py         scrive versioni/f1.json e versioni/fs.json con impronte dei sorgenti e commit
    testo.py           testi della scheda (usato da forecast.py)
    piano.py           scrive reports/piano_esame_holdout.md (solo testo: l'holdout non viene aperto)
    report.py          costruisce il report leggendo reports/fase4c/*.csv e *.json
  vix9d/           secondo indice (Fase 9D-1); riusa le funzioni del VIX senza modificarle
    config.py          percorsi e costanti (confine, orizzonti, soglie, finestre)
    prepara.py         candele settimanali dai dati giornalieri Cboe, divisione sviluppo / holdout, audit
    dati.py            lettura dei dati di sviluppo e guardia sui file con settimane successive al confine
    analisi.py         Parte A: analisi compatta, confronto con il VIX, rapporto VIX9D / VIX
    modelli.py         liste chiuse R0-R4, S0-S2, logistiche; candela comandata dagli estremi
    valuta.py          Parti B-E: walk-forward, tabelle, regole di composizione delle versioni
    versioni.py        G1 e GS: previsione completa a un'origine e configurazione (usato da forecast.py)
    congela.py         scrive versioni/vix9d/*.json con impronte dei sorgenti e commit
    lettura.py         lettura della situazione (livello, distanza dal VIX, dinamica, precedenti)
    grafici.py         grafico principale e verifica visiva
    verifica.py        12 origini equidistanti: candele previste accanto alle reali
    forecast.py        programma di previsione del VIX9D (unico a leggere i dati dopo il confine)
    piano.py           scrive reports/vix9d/piano_esame_holdout.md (solo testo)
    report.py          costruisce il report leggendo reports/vix9d/*.csv e *.json
    run_fase1.py       esegue tutta la fase
versioni/          configurazioni congelate: f1.json e fs.json (VIX), vix9d/g1.json e vix9d/gs.json (VIX9D)
tests/             test automatici (tests/vix9d/ per il VIX9D)
reports/           report, tabelle (.csv/.json) e figure; i risultati delle Fasi 1 e 2 sono in reports/fase1/, reports/fase2/ e nelle rispettive cartelle di figure
notebooks/         opzionale, vuota
```

## Regole del progetto

- **Holdout.** Le ultime 104 settimane complete stanno in `data/processed/vix_weekly_holdout.csv` e sono
  bloccate: nessun modulo di analisi le legge (lo verifica un test). Analisi e baseline usano solo
  `data/processed/vix_weekly_dev.csv`. In Fase 1 c'è anche un blocco a runtime: durante `run_phase1`
  l'apertura del file di holdout, del file clean o del file grezzo solleva un errore. Dalla Fase 2 la
  stessa guardia blocca anche gli originali dei dati esterni e la loro parte di holdout.
- **VIX9D.** Stesse regole: `data/vix9d/vix9d_weekly_dev.csv` è l'unico file del VIX9D letto dalle misure; il file completo, il VIX
  settimanale ricostruito da Cboe e il file di holdout sono bloccati da una guardia propria (`src/vix9d/dati.py`), aggiunta a quella del
  progetto. `tests/vix9d/test_fase1.py` controlla costruzione delle candele, guardia, assenza di look-ahead in variabili, modelli e versioni,
  regole di composizione, configurazioni congelate e programma.
- **Nessun look-ahead.** Ogni previsione per la data t + h usa solo dati fino a t. Il motore walk-forward
  passa ai modelli esclusivamente la storia fino all'origine; `tests/test_no_lookahead.py` lo controlla
  stravolgendo i dati futuri e verificando che le previsioni passate non cambino; `tests/test_fase1.py`
  fa lo stesso per tutti i modelli della Fase 1, `tests/test_fase2.py` per i modelli di densità e per
  l'allineamento settimanale dei dati esterni (ogni valore usato deve essere disponibile entro la data).
  `tests/test_fase3.py` lo fa per tutte le variabili della Fase 3 (ogni riga settimanale non cambia se si tolgono o
  si stravolgono i dati giornalieri successivi all'ultima seduta della settimana) e per tutti i modelli dei blocchi;
  `tests/test_fase4.py` per il modello a regimi, il gradient boosting, le varianti delle candele e le tre versioni;
  `tests/test_fase4b.py` per i componenti della prima candela; `tests/test_fase4c.py` per le variabili degli insiemi
  S1-S5, i modelli diretti di massimo e minimo, la previsione incrociata e le due versioni che vanno all'esame. Il
  «tetto teorico» della Fase 4c usa per definizione un dato futuro (l'S&P 500 della settimana prevista): è solo una
  misura, e un test verifica che non entri nel programma né nelle versioni congelate.
- **Elenco chiuso dei modelli.** In Fase 1 i modelli e i loro parametri sono fissati in
  `src/fase1/modelli.py` prima della valutazione; un test ne controlla l'elenco. Lo stesso vale per i
  modelli di densità della Fase 2 (`src/fase2/densita.py`) e per i blocchi della Fase 3 (`src/fase3/modelli.py`,
  `src/fase3/variabili.py`): regole di scelta e parametri sono scritti nel codice prima della valutazione.
- **Versioni congelate.** `versioni/*.json` contiene, per ogni versione finalista, parametri, finestre, regole, impronta
  SHA-256 dei file sorgente che definiscono i modelli e commit del congelamento. Un test fallisce se uno di quei file
  cambia dopo il congelamento o se la configurazione non coincide più con il codice.
- **Colonne.** Nelle Fasi 1 e 2 `open` non è usata (scartata alla lettura) e `high` e `low` compaiono solo
  nell'analisi esplorativa A2; dalla Fase 2b `high` e `low` sono obiettivi di previsione e dalla Fase 2c `open`,
  `high` e `low` entrano nei modelli delle candele (esclusa l'apertura anomala dell'8 novembre 2004).
- **Numeri del report.** Il report è generato da `src/report.py` a partire dai file prodotti dal codice;
  non contiene valori scritti a mano.
- **Dati grezzi.** `data/raw/` non viene mai modificato; anomalie e ambiguità sono segnalate nel report,
  non corrette in silenzio.
