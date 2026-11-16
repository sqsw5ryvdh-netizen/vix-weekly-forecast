"""Fase 3, Parte 0: tabella e testi della scheda con le candele della nuova definizione e l'etichetta della chiusura."""
from __future__ import annotations

import pandas as pd

from ..fase2b import modello as MO
from ..fase2b.testo import data_it, n, seg, sintesi as sintesi_2b


def tabella(prev: dict, candele: list[dict], st: dict) -> pd.DataFrame:
    rows = []
    for s, c in zip(prev["settimane"], candele):
        h = s["h"]
        ch, pg, et = st["chiusura"][h], st["pagella"][h], st["etichetta"][h]
        r = {"settimana": f"+{h}", "data": s["data"], "apertura": c["apertura"], "massimo": c["massimo"], "minimo": c["minimo"], "chiusura": c["chiusura"],
             "etichetta_chiusura": c["etichetta"], "p_chiusura_sopra_origine": c["p_chiusura_su"], "direzione_corpo": c["direzione"],
             "frequenza_storica_direzione_corpo": c["prob_direzione"], "mediana_d4": s["quantili"]["q50"]}
        for lev in ("50", "80", "90"):
            r |= {f"fascia_{lev}_inf": s["fasce"][lev][0], f"fascia_{lev}_sup": s["fasce"][lev][1], f"copertura_storica_{lev}": ch[f"copertura_{lev}"]}
        r |= {f"p_vix_sopra_{int(k)}": s["probabilita"][f"VIX>{int(k)}"] for k in MO.SOGLIE}
        r |= {"mae_storico_mediana_d4": ch["mediana_mae_vix"], "mae_storico_apertura": pg["mae_apertura"], "mae_storico_massimo": pg["mae_massimo"],
              "mae_storico_minimo": pg["mae_minimo"], "mae_storico_chiusura": pg["mae_chiusura"], "sovrapposizione_storica": pg["sovrapposizione"],
              "etichetta_quota_neutra_storica": et["quota_neutra"], "etichetta_accuratezza_storica_non_neutre": et["accuratezza_non_neutre"],
              "costruzione": c["costruzione"], "n_previsioni_walk_forward": int(pg["n"])}
        rows.append(r)
    return pd.DataFrame(rows)


def tabella_md(tab: pd.DataFrame) -> tuple[str, str]:
    a, b = [], []
    for r in tab.itertuples():
        a.append({"settimana": r.settimana, "data": data_it(r.data), "apertura": n(r.apertura), "massimo": n(r.massimo), "minimo": n(r.minimo), "chiusura": n(r.chiusura),
                  "chiusura rispetto all'origine": r.etichetta_chiusura, "probabilità che chiuda sopra l'origine": n(100 * r.p_chiusura_sopra_origine, 0) + "%",
                  "chiusura centrale (mediana D4)": n(r.mediana_d4),
                  "errore medio storico della candela: apertura / massimo / minimo / chiusura": f"{n(r.mae_storico_apertura, 1)} / {n(r.mae_storico_massimo, 1)} / {n(r.mae_storico_minimo, 1)} / {n(r.mae_storico_chiusura, 1)}"})
        row = {"settimana": r.settimana, "chiusura centrale (mediana D4)": n(r.mediana_d4)}
        for lev in ("50", "80", "90"):
            row[f"fascia {lev}% (copertura storica)"] = (f"{n(getattr(r, f'fascia_{lev}_inf'), 1)} – {n(getattr(r, f'fascia_{lev}_sup'), 1)} "
                                                          f"({n(100 * getattr(r, f'copertura_storica_{lev}'), 0)}%)")
        for k in MO.SOGLIE:
            row[f"P(VIX > {int(k)})"] = n(100 * getattr(r, f"p_vix_sopra_{int(k)}"), 1) + "%"
        row["errore medio storico della mediana"] = n(r.mae_storico_mediana_d4, 1)
        b.append(row)
    return pd.DataFrame(a).to_markdown(index=False), pd.DataFrame(b).to_markdown(index=False)


def _frase_etichetta(c: dict) -> str:
    if c["direzione_chiusura"] == "neutra":
        return f"neutra (probabilità di chiudere sopra l'origine {n(100 * c['p_chiusura_su'], 0)}%)"
    p = c["p_chiusura_su"] if c["direzione_chiusura"] == "rialzo" else 1 - c["p_chiusura_su"]
    return f"in {c['direzione_chiusura']} (probabilità {n(100 * p, 0)}%)"


def sintesi(L: dict, prev: dict, candele: list[dict]) -> list[str]:
    f = sintesi_2b(L, prev)
    c1, c4 = candele[0], candele[-1]
    f.append(f"Rispetto alla chiusura dell'origine ({n(prev['chiusura_origine'])}), la direzione della chiusura è {_frase_etichetta(c1)} per la settimana +1 "
             f"e {_frase_etichetta(c4)} per la settimana +4.")
    f.append(f"Candela prevista per la settimana +1: apertura {n(c1['apertura'], 1)}, chiusura {n(c1['chiusura'], 1)}, escursione attesa tra {n(c1['minimo'], 1)} e "
             f"{n(c1['massimo'], 1)} (ampiezze medie; il corpo è disegnato nella direzione storicamente più frequente).")
    return f


def come_leggere(st: dict) -> str:
    """Guida alla lettura; i numeri citati vengono dalle misure di sviluppo (run_phase2b e run_phase3)."""
    c, e1, e4 = st["candele"], st["etichetta"][1], st["etichetta"][4]
    return f"""- **Candele piene** = settimane reali (apertura ufficiale della prima seduta, massimo, minimo, chiusura). Rosso: chiusura sopra l'apertura; blu: chiusura sotto l'apertura.
- **Candele vuote con bordo tratteggiato** = le quattro settimane previste. La candela mostra una settimana di **dimensioni medie**: apertura attesa, corpo e ombre di ampiezza media. Il corpo è disegnato nella direzione che nel passato è stata più frequente (frequenza storica, non una previsione).
- **Perché le candele previste sono blu.** L'apertura ufficiale del VIX è di norma più alta della chiusura precedente (mediana storica: {seg(100 * (c['rapporto_apertura'] - 1), 1)}%) e nel passato la settimana ha chiuso sotto la propria apertura nel {n(100 * (1 - c['frequenza_rialzo']), 0)}% dei casi. **Il colore della candela non dice dove chiuderà il VIX rispetto a oggi.**
- **La scritta sopra ogni candela dice dove chiuderà il VIX rispetto alla chiusura dell'origine** (per la settimana +1 è la chiusura della settimana prima): «rialzo» o «ribasso» con la sua probabilità (modello D1), oppure **«neutra» quando la probabilità è tra 40% e 60%**. Nel walk-forward di sviluppo l'etichetta è stata «neutra» nel {n(100 * e1['quota_neutra'], 0)}% dei casi a 1 settimana e nel {n(100 * e4['quota_neutra'], 0)}% a 4; quando non era neutra è stata corretta nel {n(100 * e1['accuratezza_non_neutre'], 0)}% dei casi a 1 settimana e nel {n(100 * e4['accuratezza_non_neutre'], 0)}% a 4.
- **La linea arancione spessa è il livello recente (media delle chiusure delle ultime 12 settimane): è il riferimento principale.** Il tratteggio la prolunga fino alla settimana +4 per confrontarla con la previsione. La mediana storica (linea grigia punteggiata) resta in secondo piano.
- **La linea viola con i punti è la mediana D4, cioè il valore centrale della chiusura.** Dove la candela si allontana dalla linea si vede la differenza tra «una settimana di dimensioni medie» e «il centro di tutte le chiusure possibili».
- **Le fasce colorate sullo sfondo indicano l'incertezza sulla chiusura**: la più scura è la fascia al 50%, la più chiara quella all'80%. La copertura misurata nel passato è in tabella ed è più bassa del valore nominale.
- **Le ombre delle candele indicano l'escursione media dentro la settimana**, non un intervallo di confidenza: nel walk-forward di sviluppo la chiusura reale è caduta dentro l'escursione prevista nel {n(100 * st['pagella'][1]['chiusura_nell_escursione'], 0)}% dei casi a 1 settimana e nel {n(100 * st['pagella'][4]['chiusura_nell_escursione'], 0)}% a 4.
- **Linea tratteggiata verticale** = data di origine: a destra tutto è previsione.
- **Linee grigie sottili** (pannello di ingrandimento) = cosa è successo davvero nelle 4 settimane successive ai casi simili del passato, riportato al livello attuale. Non sono una previsione."""
