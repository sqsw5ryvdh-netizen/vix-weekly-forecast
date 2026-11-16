"""Fase 2b: tabella della previsione (Parte D) e testo della scheda (lettura della situazione, sintesi, guida al grafico).

Le frasi sono generate dai numeri; copertura ed errori «storici» sono quelli misurati nel walk-forward di sviluppo
(letti da reports/fase2b/), mai ricalcolati qui.
"""
from __future__ import annotations

import pandas as pd

from . import modello as MO


def n(v, d: int = 2) -> str:
    return "n.d." if v is None else f"{v:.{d}f}".replace(".", ",")


def seg(v: float, d: int = 2) -> str:
    return ("+" if v >= 0 else "−") + n(abs(v), d)


def data_it(s: str) -> str:
    return pd.Timestamp(s).strftime("%d/%m/%Y")


def tabella(prev: dict, st: dict) -> pd.DataFrame:
    """Parte D. `st` = statistiche del walk-forward di sviluppo: st['chiusura'][h], st['massimo'][h], st['minimo'][h]."""
    rows = []
    for s in prev["settimane"]:
        h = s["h"]
        c = st["chiusura"][h]
        r = {"settimana": f"+{h}", "data": s["data"], "apertura": s["apertura"], "massimo": s["massimo"], "minimo": s["minimo"], "chiusura": s["chiusura"]}
        for lev in ("50", "80", "90"):
            r |= {f"fascia_{lev}_inf": s["fasce"][lev][0], f"fascia_{lev}_sup": s["fasce"][lev][1], f"copertura_storica_{lev}": c[f"copertura_{lev}"]}
        r |= {f"p_vix_sopra_{int(k)}": s["probabilita"][f"VIX>{int(k)}"] for k in MO.SOGLIE}
        r |= {"mae_storico_chiusura": c["mediana_mae_vix"], "mae_storico_massimo": st["massimo"][h]["mae_modello"],
              "mae_storico_minimo": st["minimo"][h]["mae_modello"], "n_previsioni_walk_forward": int(c["n"]),
              "massimo_corretto_dal_vincolo": s["massimo_corretto"], "minimo_corretto_dal_vincolo": s["minimo_corretto"]}
        rows.append(r)
    return pd.DataFrame(rows)


def tabella_md(tab: pd.DataFrame) -> str:
    rows = []
    for r in tab.itertuples():
        row = {"settimana": r.settimana, "data": data_it(r.data), "apertura": n(r.apertura), "massimo": n(r.massimo), "minimo": n(r.minimo), "chiusura": n(r.chiusura)}
        for lev in ("50", "80", "90"):
            row[f"fascia {lev}% (copertura storica)"] = (f"{n(getattr(r, f'fascia_{lev}_inf'), 1)} – {n(getattr(r, f'fascia_{lev}_sup'), 1)} "
                                                          f"({n(100 * getattr(r, f'copertura_storica_{lev}'), 0)}%)")
        for k in MO.SOGLIE:
            row[f"P(VIX > {int(k)})"] = n(100 * getattr(r, f"p_vix_sopra_{int(k)}"), 1) + "%"
        row["errore medio storico: chiusura / massimo / minimo"] = f"{n(r.mae_storico_chiusura, 1)} / {n(r.mae_storico_massimo, 1)} / {n(r.mae_storico_minimo, 1)}"
        rows.append(row)
    return pd.DataFrame(rows).to_markdown(index=False)


def sintesi(L: dict, prev: dict) -> list[str]:
    """Tre o quattro frasi generate dai numeri, senza aggettivi."""
    lv, rf, dn, pr, cx = L["livello"], L["riferimenti"], L["dinamica"], L["precedenti"], L["contesto"]
    sopra = lambda v: "sopra" if v >= 0 else "sotto"
    v4 = dn["variazione_4_settimane"]["pct"]
    f = [f"Il VIX ha chiuso a {n(lv['chiusura'])} il {data_it(L['data_origine'])}: si colloca al percentile {n(lv['percentile_storia'], 0)} di tutta la storia "
         f"e al percentile {n(lv['percentile_5_anni'], 0)} degli ultimi 5 anni (fascia {lv['fascia']}).",
         f"È {n(abs(rf['mediana_storia']['distanza_punti']))} punti {sopra(rf['mediana_storia']['distanza_punti'])} la mediana storica ({n(rf['mediana_storia']['valore'])}) "
         f"e {n(abs(rf['media_12_settimane']['distanza_punti']))} punti {sopra(rf['media_12_settimane']['distanza_punti'])} la media a 12 settimane "
         f"({n(rf['media_12_settimane']['valore'])}); nelle ultime 4 settimane è {'salito' if v4 >= 0 else 'sceso'} del {n(abs(v4), 1)}%."]
    p4 = pr["dopo_4"]
    if p4["n"] > 0:
        f.append(f"Nei {p4['n']} precedenti con la stessa fascia e direzione ({pr['fascia']}, {pr['direzione']}) il VIX era più alto dopo 4 settimane nel "
                 f"{n(100 * p4['quota_piu_alto'], 0)}% dei casi, con variazione mediana di {seg(p4['mediana_punti'])} punti"
                 + (" (meno di 20 casi: indicazione debole)." if p4["meno_di_20_casi"] else "."))
    else:
        f.append(f"Non ci sono precedenti con la stessa fascia e direzione ({pr['fascia']}, {pr['direzione']}).")
    s4 = prev["settimane"][-1]
    f.append(f"Per la settimana +4 il modello indica una chiusura mediana di {n(s4['chiusura'], 1)}, con fascia all'80% tra {n(s4['fasce']['80'][0], 1)} e "
             f"{n(s4['fasce']['80'][1], 1)}; la probabilità di una chiusura sopra 20 è {n(100 * s4['probabilita']['VIX>20'], 0)}%, sopra 30 è {n(100 * s4['probabilita']['VIX>30'], 0)}%.")
    return f


def lettura_md(L: dict) -> str:
    lv, rf, dn, cx, pr, cs = L["livello"], L["riferimenti"], L["dinamica"], L["contesto"], L["precedenti"], L["casi_simili"]
    out = [f"**1. Livello.** Chiusura {n(lv['chiusura'])}. Percentile rispetto a tutta la storia ({L['settimane_di_storia']} settimane dal {data_it(L['prima_data'])}): "
           f"{n(lv['percentile_storia'], 0)}°; rispetto agli ultimi 5 anni ({lv['n_settimane_5_anni']} settimane): {n(lv['percentile_5_anni'], 0)}°. Fascia: **{lv['fascia']}**.", ""]
    nomi = {"media_4_settimane": "media a 4 settimane", "media_12_settimane": "media a 12 settimane", "mediana_5_anni": "mediana degli ultimi 5 anni", "mediana_storia": "mediana di tutta la storia"}
    out += ["**2. Riferimenti.**", "", "| riferimento | valore | distanza dell'attuale (punti) | distanza (%) |", "|---|---|---|---|"]
    out += [f"| {nomi[k]} | {n(v['valore'])} | {seg(v['distanza_punti'])} | {seg(v['distanza_pct'], 1)}% |" for k, v in rf.items()]
    out += ["", "**3. Dinamica.** Variazione della chiusura: "
            + "; ".join(f"a {k} settiman{'a' if k == 1 else 'e'} {seg(dn[f'variazione_{k}_settimane']['punti'])} punti ({seg(dn[f'variazione_{k}_settimane']['pct'], 1)}%)" for k in (1, 4, 12))
            + f". Nelle ultime 12 settimane le chiusure sono state tra {n(dn['minimo_chiusure_12_settimane'])} e {n(dn['massimo_chiusure_12_settimane'])}. "
            f"Ampiezza tipica dei movimenti settimanali (mediana della variazione assoluta): {n(dn['ampiezza_mediana_recente_pct'], 1)}% nelle ultime 12 settimane, "
            f"contro {n(dn['ampiezza_mediana_storica_pct'], 1)}% nell'intera storia ({n(dn['percentile_ampiezza_recente'], 0)}° percentile dei periodi di 12 settimane).", ""]
    if cx["settimane_dall_ultima_chiusura_sopra_30"] is None:
        c30 = "Nessuna chiusura a 30 o più nella storia disponibile."
    else:
        c30 = (f"L'ultima chiusura a 30 o più risale a {cx['settimane_dall_ultima_chiusura_sopra_30']} settimane fa ({data_it(cx['data_ultima_chiusura_sopra_30'])}); "
               f"il picco di quell'episodio fu {n(cx['picco_episodio'])} il {data_it(cx['data_picco'])}.")
    calm = (f" Il VIX chiude sotto 15 da {cx['settimane_consecutive_sotto_15']} settimane consecutive." if cx["periodo_calmo"]
            else (f" Il VIX chiude sotto 15 da {cx['settimane_consecutive_sotto_15']} settimane (meno delle 8 che definiscono un periodo calmo)." if cx["settimane_consecutive_sotto_15"] else ""))
    out += [f"**4. Contesto.** {c30}{calm}", ""]
    out += [f"**5. Precedenti per fascia × direzione.** Situazione attuale: fascia {pr['fascia']}, direzione «{pr['direzione']}» "
            f"(variazione log a 4 settimane {seg(pr['variazione_log_4_settimane'], 3)}; soglie ±0,10).", "",
            "| dopo | casi | quota con VIX più alto | variazione mediana (punti) | 10° – 90° percentile (punti) |", "|---|---|---|---|---|"]
    for h in (1, 4):
        p = pr[f"dopo_{h}"]
        if p["n"] == 0:
            out.append(f"| {h} settiman{'a' if h == 1 else 'e'} | 0 | – | – | – |")
        else:
            out.append(f"| {h} settiman{'a' if h == 1 else 'e'} | {p['n']}{' (meno di 20: indicazione debole)' if p['meno_di_20_casi'] else ''} | {n(100 * p['quota_piu_alto'], 0)}% | "
                       f"{seg(p['mediana_punti'])} | {seg(p['p10_punti'])} – {seg(p['p90_punti'])} |")
    out += ["", f"**6. Casi simili** ({cs['n']} casi{'; meno di 20: indicazione debole' if cs['meno_di_20_casi'] else ''}). Selezione **solo illustrativa**: le situazioni passate più vicine "
            "secondo lo stato di M6 (livello, variazione a 4 e a 12 settimane, variabilità a 12 settimane), con almeno 8 settimane tra un caso e l'altro "
            "e con esito a 4 settimane già noto. **Non è il modello M6** e non entra nella previsione.", ""]
    if cs["n"]:
        out += [f"Dopo 4 settimane il VIX era più alto nel {n(100 * cs['quota_piu_alto_dopo_4'], 0)}% dei casi; variazione mediana {seg(cs['mediana_variazione_4_pct'], 1)}%.", "",
                "| data | livello di allora | dopo 1 sett. | dopo 2 | dopo 3 | dopo 4 | chiusura dopo 4 sett. | minimo – massimo delle chiusure nelle 4 sett. |", "|---|---|---|---|---|---|---|---|"]
        out += [f"| {data_it(k['data'])} | {n(k['livello'])} | " + " | ".join(seg(v, 1) + "%" for v in k["variazioni_pct"])
                + f" | {n(k['chiusura_dopo_4'])} | {n(k['minimo_chiusure_4'])} – {n(k['massimo_chiusure_4'])} |" for k in cs["casi"]]
    return "\n".join(out)


COME_LEGGERE = """- **Candele piene** = settimane reali (apertura, massimo, minimo, chiusura del file). Rosso: chiusura sopra l'apertura; blu: chiusura sotto l'apertura.
- **Candele vuote con bordo tratteggiato** = le quattro settimane previste. Il corpo va dalla chiusura prevista della settimana precedente (per la prima: l'ultima chiusura reale) alla chiusura prevista; **le ombre indicano l'escursione attesa dentro la settimana** (massimo e minimo previsti), non un intervallo di confidenza.
- **Le fasce colorate sullo sfondo indicano l'incertezza sulla chiusura**: la più scura è la fascia al 50%, la più chiara quella all'80%. La chiusura reale è caduta dentro queste fasce con la frequenza riportata in tabella («copertura storica»), che è più bassa del valore nominale.
- **Ombre e fasce misurano cose diverse.** Un'ombra lunga dice che dentro la settimana il VIX può allontanarsi molto dalla chiusura; una fascia larga dice che la chiusura stessa è incerta.
- **Linea tratteggiata verticale** = data di origine: a destra tutto è previsione. **Linea arancione** = media mobile a 12 settimane; **linea punteggiata** = mediana storica.
- **Linee grigie sottili** (pannello di ingrandimento) = cosa è successo davvero nelle 4 settimane successive ai casi simili del passato, riportato al livello attuale come variazione percentuale. Servono a dare un'idea della varietà degli esiti; non sono una previsione."""
