"""Fase 4b: testi della scheda per le versioni F1 e FS (prima candela colorata secondo l'etichetta, minimo dedicato)."""
from __future__ import annotations

import pandas as pd

from ..fase2b.testo import n, seg, sintesi as sintesi_2b
from ..fase3 import testo as T3
from ..fase4 import testo as T4
from . import versioni as VS

tabella, tabella_md, confronto = T4.tabella, T4.tabella_md, T4.confronto


def da_dove(versione: str, cfg: dict) -> tuple[str, str]:
    """(origine della chiusura centrale, origine della probabilita' dell'etichetta +1) per la versione."""
    if versione == "f1":
        return "mediana D4", "modello D1"
    cand = cfg["fs"]["candidato"]
    base = "regressione quantilica con VIX3M/VIX" if cfg["fs"]["costruzione_di_base"] == "f2" else "modello a regimi"
    return f"mediana della {base}" if base.startswith("regressione") else f"mediana del {base}", f"candidato {cand['descrizione']} per la settimana +1"


def confronto_md(tab: pd.DataFrame) -> str:
    rows = [{"settimana": r.settimana, "versione": VS.NOMI[r.versione], "chiusura centrale": n(r.chiusura_centrale),
             "fascia 50%": f"{n(r.fascia_50_inf, 1)} – {n(r.fascia_50_sup, 1)}", "fascia 80%": f"{n(r.fascia_80_inf, 1)} – {n(r.fascia_80_sup, 1)}",
             "chiusura rispetto all'origine": r.etichetta_chiusura, "probabilità che chiuda sopra l'origine": n(100 * r.p_chiusura_sopra_origine, 0) + "%",
             "P(VIX > 20)": n(100 * r.p_vix_sopra_20, 0) + "%", "P(VIX > 30)": n(100 * r.p_vix_sopra_30, 0) + "%"} for r in tab.itertuples()]
    return pd.DataFrame(rows).to_markdown(index=False)


def sintesi(L: dict, prev: dict, candele: list[dict]) -> list[str]:
    f = sintesi_2b(L, prev)[:3]
    s4, c1, c4 = prev["settimane"][-1], candele[0], candele[-1]
    f.append(f"Versione {VS.NOMI[prev['versione']]}: per la settimana +4 la chiusura centrale è {n(s4['chiusura'], 1)}, con fascia all'80% tra {n(s4['fasce']['80'][0], 1)} e "
             f"{n(s4['fasce']['80'][1], 1)}; la probabilità di una chiusura sopra 20 è {n(100 * s4['probabilita']['VIX>20'], 0)}%, sopra 30 è {n(100 * s4['probabilita']['VIX>30'], 0)}%.")
    f.append(f"Rispetto alla chiusura dell'origine ({n(prev['chiusura_origine'])}), la direzione della chiusura è {T3._frase_etichetta(c1)} per la settimana +1 "
             f"e {T3._frase_etichetta(c4)} per la settimana +4.")
    f.append(f"Prima candela (settimana +1): apertura {n(c1['apertura'], 1)}, chiusura {n(c1['chiusura'], 1)}, minimo previsto {n(c1['minimo'], 1)}, massimo {n(c1['massimo'], 1)}; "
             f"colore secondo l'etichetta («{c1['direzione_chiusura']}»).")
    return f


def come_leggere(st: dict, cand: dict, versione: str, cfg: dict) -> str:
    """Guida alla lettura; i numeri citati vengono dalle misure di sviluppo."""
    centrale, prob = da_dove(versione, cfg)
    e1, e4 = st["etichetta"][1], st["etichetta"][4]
    prima = cfg[versione]["candele"]["prima_candela"]
    k4 = prima["costruzione"] == "K4"
    corpo = ("La sua chiusura segue l'etichetta: mossa media delle settimane in calo se «ribasso», delle settimane in rialzo se «rialzo», chiusura centrale se «neutra»." if k4
             else "Il corpo resta di dimensioni medie e disegnato verso il basso (frequenza storica): **per la prima candela il colore dice la direzione prevista della chiusura rispetto all'origine, non il verso del corpo**.")
    minimo = ("Il suo **minimo** viene da un modello dedicato (regressione sulla mediana del rapporto tra minimo della settimana e chiusura dell'origine) e non sta mai sopra il corpo." if prima["minimo"] == "M1q"
              else "Il suo minimo è quello della candela di dimensioni medie.")
    return f"""- **Candele piene dal bordo continuo** = settimane reali (apertura ufficiale della prima seduta, massimo, minimo, chiusura). Rosso: chiusura sopra l'apertura; blu: chiusura sotto l'apertura.
- **La prima candela prevista (settimana +1) è colorata secondo l'etichetta della chiusura**: rossa se «rialzo», blu se «ribasso», **grigia se «neutra»**; il bordo tratteggiato ricorda che è una previsione. {corpo} {minimo}
- **Candele vuote con bordo tratteggiato** = settimane +2, +3 e +4: una settimana di dimensioni medie (medie delle ultime settimane), con il corpo nella direzione storicamente più frequente. Sono blu perché l'apertura ufficiale del VIX è di norma più alta della chiusura precedente (mediana storica: {seg(100 * (cand['rapporto_apertura'] - 1), 1)}%) e la settimana chiude sotto la propria apertura nel {n(100 * (1 - cand['frequenza_rialzo']), 0)}% dei casi: il loro colore non dice dove chiuderà il VIX rispetto a oggi.
- **La scritta sopra ogni candela dice dove chiuderà il VIX rispetto alla chiusura dell'origine**: «rialzo» o «ribasso» con la sua probabilità ({prob}), oppure **«neutra» quando la probabilità è tra 40% e 60%**. Nel walk-forward di sviluppo l'etichetta è stata «neutra» nel {n(100 * e1['quota_neutra'], 0)}% dei casi a 1 settimana e nel {n(100 * e4['quota_neutra'], 0)}% a 4; quando non era neutra è stata corretta nel {n(100 * e1['accuratezza_non_neutre'], 0)}% dei casi a 1 settimana e nel {n(100 * e4['accuratezza_non_neutre'], 0)}% a 4.
- **Il massimo non è ricalibrato**: nel passato il massimo reale della settimana +1 ha superato quello previsto in circa metà dei casi; è un limite accettato.
- **La linea arancione spessa è il livello recente (media delle chiusure delle ultime 12 settimane): è il riferimento principale.** Il tratteggio la prolunga fino alla settimana +4. La mediana storica (linea grigia punteggiata) resta in secondo piano.
- **La linea viola con i punti è la chiusura centrale ({centrale}).** Dove la candela si allontana dalla linea si vede la differenza tra «una settimana di dimensioni medie» e «il centro di tutte le chiusure possibili».
- **Le fasce colorate sullo sfondo indicano l'incertezza sulla chiusura**: la più scura è la fascia al 50%, la più chiara quella all'80%. La copertura misurata nel passato è in tabella.
- **Linea tratteggiata verticale** = data di origine: a destra tutto è previsione. La previsione si fa una sola volta, alla chiusura settimanale, e non viene aggiornata durante la settimana.
- **Linee grigie sottili** (pannello di ingrandimento) = cosa è successo davvero nelle 4 settimane successive ai casi simili del passato, riportato al livello attuale. Non sono una previsione."""
