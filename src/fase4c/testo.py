"""Fase 4c: testi della scheda (prima candela comandata da massimo e minimo previsti)."""
from __future__ import annotations

from ..fase2b.testo import n, seg
from ..fase4b import testo as T4B
from . import variabili as V

tabella, tabella_md, confronto, confronto_md, da_dove = T4B.tabella, T4B.tabella_md, T4B.confronto, T4B.confronto_md, T4B.da_dove


def _ins(k: str) -> str:
    return V.DESCRIZIONI.get(k, "B3 previsione incrociata con l'S&P 500")


def sintesi(L: dict, prev: dict, candele: list[dict]) -> list[str]:
    f = T4B.sintesi(L, prev, candele)[:-1]
    c1 = candele[0]
    f.append(f"Prima candela (settimana +1): massimo previsto {n(c1['massimo'], 1)}, minimo previsto {n(c1['minimo'], 1)}, apertura {n(c1['apertura'], 1)}, chiusura {n(c1['chiusura'], 1)}; "
             f"colore secondo l'etichetta («{c1['direzione_chiusura']}»).")
    return f


def come_leggere(st: dict, cand: dict, versione: str, cfg: dict) -> str:
    """Guida alla lettura; i numeri citati vengono dalle misure di sviluppo."""
    centrale, prob = da_dove(versione, cfg)
    e1, e4, p1 = st["etichetta"][1], st["etichetta"][4], st["pagella"][1]
    prima = cfg[versione]["candele"]["prima_candela"]
    corpo = ("La chiusura segue l'etichetta quando non è neutra (mossa media delle settimane in calo o in rialzo), altrimenti è quella di una settimana media." if prima["corpo"] == "K5"
             else "Il corpo è quello di una settimana media, disegnato verso il basso (frequenza storica): **per la prima candela il colore dice la direzione prevista della chiusura rispetto all'origine, non il verso del corpo**.")
    return f"""- **Candele piene dal bordo continuo** = settimane reali (apertura ufficiale della prima seduta, massimo, minimo, chiusura). Rosso: chiusura sopra l'apertura; blu: chiusura sotto l'apertura.
- **La prima candela prevista (settimana +1) è colorata secondo l'etichetta della chiusura**: rossa se «rialzo», blu se «ribasso», **grigia se «neutra»**; il bordo tratteggiato ricorda che è una previsione.
- **Massimo e minimo della prima candela sono previsti direttamente** (regressione sulla mediana del rapporto con la chiusura dell'origine; variabili: {_ins(prima['massimo']['insieme'])} per il massimo, {_ins(prima['minimo']['insieme'])} per il minimo) **e comandano la candela**: se il corpo uscirebbe dall'intervallo, viene accorciato. {corpo}
  Nel walk-forward di sviluppo il massimo previsto è stato entro il 5% da quello reale nel {n(100 * p1['massimo_entro_5'], 0)}% dei casi e il minimo nel {n(100 * p1['minimo_entro_5'], 0)}%.
- **Candele vuote con bordo tratteggiato** = settimane +2, +3 e +4: una settimana di dimensioni medie (medie delle ultime settimane), con il corpo nella direzione storicamente più frequente. Sono blu perché l'apertura ufficiale del VIX è di norma più alta della chiusura precedente (mediana storica: {seg(100 * (cand['rapporto_apertura'] - 1), 1)}%) e la settimana chiude sotto la propria apertura nel {n(100 * (1 - cand['frequenza_rialzo']), 0)}% dei casi: il loro colore non dice dove chiuderà il VIX rispetto a oggi.
- **La scritta sopra ogni candela dice dove chiuderà il VIX rispetto alla chiusura dell'origine**: «rialzo» o «ribasso» con la sua probabilità ({prob}), oppure **«neutra» quando la probabilità è tra 40% e 60%**. Nel walk-forward di sviluppo l'etichetta è stata «neutra» nel {n(100 * e1['quota_neutra'], 0)}% dei casi a 1 settimana e nel {n(100 * e4['quota_neutra'], 0)}% a 4; quando non era neutra è stata corretta nel {n(100 * e1['accuratezza_non_neutre'], 0)}% dei casi a 1 settimana e nel {n(100 * e4['accuratezza_non_neutre'], 0)}% a 4.
- **La linea arancione spessa è il livello recente (media delle chiusure delle ultime 12 settimane): è il riferimento principale.** Il tratteggio la prolunga fino alla settimana +4. La mediana storica (linea grigia punteggiata) resta in secondo piano.
- **La linea viola con i punti è la chiusura centrale ({centrale}).** Dove la candela si allontana dalla linea si vede la differenza tra «una settimana di dimensioni medie» e «il centro di tutte le chiusure possibili».
- **Le fasce colorate sullo sfondo indicano l'incertezza sulla chiusura**: la più scura è la fascia al 50%, la più chiara quella all'80%. La copertura misurata nel passato è in tabella.
- **Linea tratteggiata verticale** = data di origine: a destra tutto è previsione. La previsione si fa una sola volta, alla chiusura settimanale, e non viene aggiornata durante la settimana.
- **Linee grigie sottili** (pannello di ingrandimento) = cosa è successo davvero nelle 4 settimane successive ai casi simili del passato, riportato al livello attuale. Non sono una previsione."""
