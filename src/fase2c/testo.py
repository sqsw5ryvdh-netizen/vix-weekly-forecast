"""Fase 2c: tabella e testi della scheda con le candele della costruzione scelta."""
from __future__ import annotations

import pandas as pd

from ..fase2b import modello as MO
from ..fase2b.testo import data_it, n, seg, sintesi as sintesi_2b


def tabella(prev: dict, candele: list[dict], st: dict) -> pd.DataFrame:
    rows = []
    for s, c in zip(prev["settimane"], candele):
        h = s["h"]
        ch, pg = st["chiusura"][h], st["pagella"][h]
        r = {"settimana": f"+{h}", "data": s["data"], "apertura": c["apertura"], "massimo": c["massimo"], "minimo": c["minimo"], "chiusura": c["chiusura"],
             "direzione": c["direzione"], "prob_direzione": c["prob_direzione"], "mediana_d4": s["quantili"]["q50"]}
        for lev in ("50", "80", "90"):
            r |= {f"fascia_{lev}_inf": s["fasce"][lev][0], f"fascia_{lev}_sup": s["fasce"][lev][1], f"copertura_storica_{lev}": ch[f"copertura_{lev}"]}
        r |= {f"p_vix_sopra_{int(k)}": s["probabilita"][f"VIX>{int(k)}"] for k in MO.SOGLIE}
        r |= {"mae_storico_mediana_d4": ch["mediana_mae_vix"], "mae_storico_apertura": pg["mae_apertura"], "mae_storico_massimo": pg["mae_massimo"],
              "mae_storico_minimo": pg["mae_minimo"], "mae_storico_chiusura": pg["mae_chiusura"], "direzione_corpo_storica": pg["direzione_corpo"],
              "sovrapposizione_storica": pg["sovrapposizione"], "costruzione": c["costruzione"], "n_previsioni_walk_forward": int(pg["n"])}
        rows.append(r)
    return pd.DataFrame(rows)


def tabella_md(tab: pd.DataFrame) -> tuple[str, str]:
    a, b = [], []
    for r in tab.itertuples():
        a.append({"settimana": r.settimana, "data": data_it(r.data), "apertura": n(r.apertura), "massimo": n(r.massimo), "minimo": n(r.minimo), "chiusura": n(r.chiusura),
                  "direzione (probabilità)": f"{r.direzione} {n(100 * r.prob_direzione, 0)}%", "chiusura centrale (mediana D4)": n(r.mediana_d4),
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


def sintesi(L: dict, prev: dict, candele: list[dict]) -> list[str]:
    f = sintesi_2b(L, prev)
    c1 = candele[0]
    f.append(f"La candela prevista per la settimana +1 è in {c1['direzione']} (probabilità {n(100 * c1['prob_direzione'], 0)}%): apertura {n(c1['apertura'], 1)}, "
             f"chiusura {n(c1['chiusura'], 1)}, escursione attesa tra {n(c1['minimo'], 1)} e {n(c1['massimo'], 1)}.")
    return f


def come_leggere(st: dict) -> str:
    """Guida alla lettura; i numeri citati vengono dalle misure di sviluppo."""
    c = st["candele"]
    return f"""- **Candele piene** = settimane reali (apertura ufficiale della prima seduta, massimo, minimo, chiusura). Rosso: chiusura sopra l'apertura; blu: chiusura sotto l'apertura.
- **Candele vuote con bordo tratteggiato** = le quattro settimane previste. **La candela mostra la settimana più probabile per direzione e ampiezza**: apertura attesa, direzione più probabile del corpo (con la sua probabilità scritta sopra), ampiezza tipica del corpo e delle ombre.
- **La linea viola con i punti è la mediana D4, cioè il valore centrale della chiusura: la media degli esiti possibili**, rialzi e ribassi insieme. Dove la candela si allontana dalla linea si vede la differenza tra «la settimana più probabile» e «il centro di tutte le settimane possibili».
- **Perché le candele previste sono quasi sempre blu.** L'apertura ufficiale del VIX è di norma più alta della chiusura precedente (mediana storica: {seg(100 * (c['rapporto_apertura'] - 1), 1)}%) e nel passato la settimana ha chiuso sotto la propria apertura nel {n(100 * (1 - c['frequenza_rialzo']), 0)}% dei casi. Una candela blu non significa che il VIX scenderà rispetto a oggi: guarda la linea centrale per questo.
- **Le fasce colorate sullo sfondo indicano l'incertezza sulla chiusura**: la più scura è la fascia al 50%, la più chiara quella all'80%. La copertura misurata nel passato è in tabella ed è più bassa del valore nominale.
- **Le ombre delle candele indicano l'escursione attesa dentro la settimana** (massimo e minimo tipici), non un intervallo di confidenza: nel walk-forward di sviluppo la chiusura reale è caduta dentro l'escursione prevista nel {n(100 * st['pagella'][1]['chiusura_nell_escursione'], 0)}% dei casi a 1 settimana e nel {n(100 * st['pagella'][4]['chiusura_nell_escursione'], 0)}% a 4.
- **Linea tratteggiata verticale** = data di origine: a destra tutto è previsione. **Linea arancione** = media mobile a 12 settimane; **linea punteggiata** = mediana storica.
- **Linee grigie sottili** (pannello di ingrandimento) = cosa è successo davvero nelle 4 settimane successive ai casi simili del passato, riportato al livello attuale. Non sono una previsione."""
