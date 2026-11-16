"""Fase 4: tabelle e testi della scheda per le versioni F1, F2, F3 e tabella di confronto tra le versioni."""
from __future__ import annotations

import pandas as pd

from ..fase2b import modello as MO
from ..fase2b.testo import data_it, n, seg, sintesi as sintesi_2b
from ..fase3 import testo as T3
from . import versioni as VS

DA_DOVE = {"f1": ("mediana D4", "modello D1"), "f2": ("mediana della regressione quantilica con VIX3M/VIX", "errori passati attorno al centro M1 + VIX3M/VIX"),
           "f3": ("mediana del modello a regimi", "modello a regimi")}


def tabella(prev: dict, candele: list[dict], st: dict) -> pd.DataFrame:
    """Tabella della previsione di una versione; `st` = misure di sviluppo di quella versione."""
    t = T3.tabella(prev, candele, st).rename(columns={"mediana_d4": "chiusura_centrale", "mae_storico_mediana_d4": "mae_storico_chiusura_centrale"})
    t.insert(0, "versione", prev["versione"])
    return t


def tabella_md(tab: pd.DataFrame) -> tuple[str, str]:
    a, b = [], []
    for r in tab.itertuples():
        a.append({"settimana": r.settimana, "data": data_it(r.data), "apertura": n(r.apertura), "massimo": n(r.massimo), "minimo": n(r.minimo), "chiusura": n(r.chiusura),
                  "chiusura rispetto all'origine": r.etichetta_chiusura, "probabilità che chiuda sopra l'origine": n(100 * r.p_chiusura_sopra_origine, 0) + "%",
                  "chiusura centrale": n(r.chiusura_centrale),
                  "errore medio storico della candela: apertura / massimo / minimo / chiusura": f"{n(r.mae_storico_apertura, 1)} / {n(r.mae_storico_massimo, 1)} / {n(r.mae_storico_minimo, 1)} / {n(r.mae_storico_chiusura, 1)}"})
        row = {"settimana": r.settimana, "chiusura centrale": n(r.chiusura_centrale)}
        for lev in ("50", "80", "90"):
            row[f"fascia {lev}% (copertura storica)"] = (f"{n(getattr(r, f'fascia_{lev}_inf'), 1)} – {n(getattr(r, f'fascia_{lev}_sup'), 1)} "
                                                          f"({n(100 * getattr(r, f'copertura_storica_{lev}'), 0)}%)")
        for k in MO.SOGLIE:
            row[f"P(VIX > {int(k)})"] = n(100 * getattr(r, f"p_vix_sopra_{int(k)}"), 1) + "%"
        row["errore medio storico della chiusura centrale"] = n(r.mae_storico_chiusura_centrale, 1)
        b.append(row)
    return pd.DataFrame(a).to_markdown(index=False), pd.DataFrame(b).to_markdown(index=False)


def confronto(risultati: dict) -> pd.DataFrame:
    """Una riga per settimana e versione: chiusura centrale, fasce, etichetta, probabilita'."""
    rows = []
    for v, (prev, cand) in risultati.items():
        for s, c in zip(prev["settimane"], cand):
            rows.append({"settimana": f"+{s['h']}", "versione": v, "chiusura_centrale": s["chiusura"], "fascia_50_inf": s["fasce"]["50"][0], "fascia_50_sup": s["fasce"]["50"][1],
                         "fascia_80_inf": s["fasce"]["80"][0], "fascia_80_sup": s["fasce"]["80"][1], "etichetta_chiusura": c["etichetta"],
                         "p_chiusura_sopra_origine": c["p_chiusura_su"]} | {f"p_vix_sopra_{int(k)}": s["probabilita"][f"VIX>{int(k)}"] for k in MO.SOGLIE})
    return pd.DataFrame(rows).sort_values(["settimana", "versione"], kind="stable").reset_index(drop=True)


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
    f.append(f"Candela prevista per la settimana +1: apertura {n(c1['apertura'], 1)}, chiusura {n(c1['chiusura'], 1)}, escursione attesa tra {n(c1['minimo'], 1)} e "
             f"{n(c1['massimo'], 1)} (ampiezze medie recenti; il corpo è disegnato nella direzione storicamente più frequente).")
    return f


def come_leggere(st: dict, cand: dict, versione: str) -> str:
    """Guida alla lettura; i numeri citati vengono dalle misure di sviluppo."""
    centrale, prob = DA_DOVE[versione]
    e1, e4 = st["etichetta"][1], st["etichetta"][4]
    return f"""- **Candele piene** = settimane reali (apertura ufficiale della prima seduta, massimo, minimo, chiusura). Rosso: chiusura sopra l'apertura; blu: chiusura sotto l'apertura.
- **Candele vuote con bordo tratteggiato** = le quattro settimane previste. La candela mostra una settimana di **dimensioni medie**: apertura attesa, corpo e ombre di ampiezza media (medie delle ultime settimane, non di tutta la storia). Il corpo è disegnato nella direzione che nel passato è stata più frequente (frequenza storica, non una previsione).
- **Perché le candele previste sono blu.** L'apertura ufficiale del VIX è di norma più alta della chiusura precedente (mediana storica: {seg(100 * (cand['rapporto_apertura'] - 1), 1)}%) e nel passato la settimana ha chiuso sotto la propria apertura nel {n(100 * (1 - cand['frequenza_rialzo']), 0)}% dei casi. **Il colore della candela non dice dove chiuderà il VIX rispetto a oggi.**
- **La scritta sopra ogni candela dice dove chiuderà il VIX rispetto alla chiusura dell'origine** (per la settimana +1 è la chiusura della settimana prima): «rialzo» o «ribasso» con la sua probabilità ({prob}), oppure **«neutra» quando la probabilità è tra 40% e 60%**. Nel walk-forward di sviluppo l'etichetta è stata «neutra» nel {n(100 * e1['quota_neutra'], 0)}% dei casi a 1 settimana e nel {n(100 * e4['quota_neutra'], 0)}% a 4; quando non era neutra è stata corretta nel {n(100 * e1['accuratezza_non_neutre'], 0)}% dei casi a 1 settimana e nel {n(100 * e4['accuratezza_non_neutre'], 0)}% a 4.
- **La linea arancione spessa è il livello recente (media delle chiusure delle ultime 12 settimane): è il riferimento principale.** Il tratteggio la prolunga fino alla settimana +4. La mediana storica (linea grigia punteggiata) resta in secondo piano.
- **La linea viola con i punti è la chiusura centrale ({centrale}).** Dove la candela si allontana dalla linea si vede la differenza tra «una settimana di dimensioni medie» e «il centro di tutte le chiusure possibili».
- **Le fasce colorate sullo sfondo indicano l'incertezza sulla chiusura**: la più scura è la fascia al 50%, la più chiara quella all'80%. La copertura misurata nel passato è in tabella.
- **Le ombre delle candele indicano l'escursione media dentro la settimana**, non un intervallo di confidenza: nel walk-forward di sviluppo la chiusura reale è caduta dentro l'escursione prevista nel {n(100 * st['pagella'][1]['chiusura_nell_escursione'], 0)}% dei casi a 1 settimana e nel {n(100 * st['pagella'][4]['chiusura_nell_escursione'], 0)}% a 4.
- **Linea tratteggiata verticale** = data di origine: a destra tutto è previsione.
- **Linee grigie sottili** (pannello di ingrandimento) = cosa è successo davvero nelle 4 settimane successive ai casi simili del passato, riportato al livello attuale. Non sono una previsione."""
