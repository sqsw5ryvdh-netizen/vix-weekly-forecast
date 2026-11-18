"""Programma di previsione del VIX9D settimanale (2 candele: settimane +1 e +2).

    python -m src.vix9d.forecast [--versione g1|gs] [--aggiorna] [--origine AAAA-MM-GG]
    python -m src.vix9d.forecast --verifica

Legge l'intero storico settimanale del VIX9D (data/vix9d/VIX9D_settimanale.csv) e, per la versione GS, il VIX settimanale
(data/vix9d/VIX_settimanale_cboe.csv); esclude la settimana incompleta, parte dall'ultima chiusura settimanale completa
(o dall'origine indicata) e calcola le versioni congelate in versioni/vix9d/*.json:
  g1  solo storico del VIX9D (predefinita);
  gs  con il VIX: se il VIX della settimana di origine manca, avvisa e ripiega su g1.
`--aggiorna` scarica dal sito Cboe i file giornalieri di VIX9D e VIX e ricostruisce i due file settimanali completi
(non tocca i file di sviluppo e di holdout). La previsione si fa una sola volta, alla chiusura settimanale.

E' L'UNICO comando del VIX9D autorizzato a leggere i dati successivi al confine di sviluppo, e solo come storia da cui
prevedere: errori e coperture mostrati accanto alla previsione sono quelli misurati sul campione di sviluppo da
`run_fase1`, e qui vengono solo letti. Unica eccezione, esplicita: con `--verifica` le previsioni gia' scritte nel
registro vengono confrontate con le settimane nel frattempo completate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..fase2b.testo import data_it, n, seg
from ..fase2c import settimana as SE
from ..fase4 import registro as RG
from . import config as G
from . import dati
from . import grafici
from . import lettura as LE
from . import prepara
from . import versioni as VS

OUT_DIR = G.OUT / "previsione"
ESTERNI = C.ROOT / "data" / "external" / "forecast"
FILE_ESTERNI = {"vix9d": "VIX9D_History.csv", "vix": "VIX_History.csv"}


def leggi_settimanale(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Settimane complete di un file nel formato del VIX9D (o del VIX): esclude la riga con nota e l'ultima riga se incompleta."""
    raw = pd.read_csv(path, parse_dates=["week_start", "last_trading_day"]).sort_values("week_start").reset_index(drop=True)
    incomp = raw["note"].notna() & (raw["note"].astype(str).str.strip() != "")
    avvisi = []
    if len(raw) and not incomp.iloc[-1]:
        u = raw.iloc[-1]
        inc, msg = SE.ultima_riga_incompleta(u["week_start"], u["last_trading_day"], u["trading_days"], None)
        if msg:
            avvisi.append(msg)
        if inc:
            incomp.iloc[-1] = True
    d = raw.loc[~incomp].rename(columns={"last_trading_day": "date"}).reset_index(drop=True)
    if d["date"].duplicated().any() or (d[["open", "high", "low", "close"]] <= 0).any().any() or d[["open", "high", "low", "close"]].isna().any().any():
        raise SystemExit(f"Il file {path} contiene date duplicate o valori mancanti / non positivi")
    return d, avvisi


def leggi_storico(path9: Path, path_vix: Path | None) -> tuple[pd.DataFrame, dict]:
    if not path9.exists():
        raise SystemExit(f"Manca {path9}: eseguire `python -m src.vix9d.prepara` oppure `python -m src.vix9d.forecast --aggiorna`.")
    w9, avvisi = leggi_settimanale(path9)
    if path_vix is not None and path_vix.exists():
        vix, _ = leggi_settimanale(path_vix)
    else:
        vix = pd.DataFrame({"week_start": pd.Series(dtype="datetime64[ns]"), "date": pd.Series(dtype="datetime64[ns]"), "close": pd.Series(dtype=float)})
    df = dati.unisci_vix(w9[["date", "week_start", "open", "high", "low", "close", "trading_days", "sedute_solo_chiusura"]], vix)
    info = {"file": str(path9), "file_vix": None if path_vix is None else str(path_vix), "settimane_complete": int(len(df)), "prima_data": str(df["date"].iloc[0].date()),
            "ultima_data_completa": str(df["date"].iloc[-1].date()), "settimane_con_candela": int(df["candela"].sum()), "avvisi": avvisi}
    return df, info


def aggiorna(scarica=None, cartella: Path = ESTERNI, destinazione: Path | None = None) -> dict:
    """Scarica VIX9D e VIX giornalieri e ricostruisce i due file settimanali completi. Non tocca i file di sviluppo e di holdout."""
    if scarica is None:
        from ..fetch_external import scarica
    cartella.mkdir(parents=True, exist_ok=True)
    dest = destinazione or G.DATI
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, url in G.INDIRIZZI.items():
        raw = scarica(url)
        (cartella / FILE_ESTERNI[k]).write_bytes(raw)
        out[k] = len(raw)
    fmt = dict(index=False, float_format="%.4f", date_format="%Y-%m-%d")
    w9 = prepara.settimanali(prepara.leggi_cboe(cartella / FILE_ESTERNI["vix9d"]))
    wv = prepara.settimanali(prepara.leggi_cboe(cartella / FILE_ESTERNI["vix"]))
    w9.to_csv(dest / G.FILE_COMPLETO.name, **fmt)
    wv[wv["week_start"] >= w9["week_start"].iloc[0] - pd.Timedelta(days=7 * 60)].to_csv(dest / G.FILE_VIX_COMPLETO.name, **fmt)
    return out


def carica_versioni() -> dict:
    cfg = {v: json.loads((G.VERSIONI / f"{v}.json").read_text()) for v in VS.VERSIONI if (G.VERSIONI / f"{v}.json").exists()}
    if "g1" not in cfg:
        raise SystemExit("Mancano le versioni congelate (versioni/vix9d/g1.json): eseguire `python -m src.vix9d.run_fase1` e poi `python -m src.vix9d.congela`.")
    return cfg


def statistiche_sviluppo() -> dict:
    """Misure del walk-forward di sviluppo per versione e orizzonte (prodotte da run_fase1). Qui vengono solo lette."""
    try:
        t = pd.read_csv(G.OUT / "e_versioni.csv")
        g3 = json.loads((G.OUT / "g3.json").read_text())
    except FileNotFoundError:
        raise SystemExit("Mancano le misure del walk-forward di sviluppo: eseguire prima `python -m src.vix9d.run_fase1`.")
    t = t[t["gruppo"] == "tutte"]
    return {"versioni": {v: {int(r["h"]): r for r in g.to_dict("records")} for v, g in t.groupby("versione")}, "g3": g3}


def indice_origine(df: pd.DataFrame, origine: str | None) -> int:
    if origine is None:
        return len(df) - 1
    ok = np.flatnonzero(df["date"].to_numpy() <= np.datetime64(pd.Timestamp(origine)))
    if len(ok) == 0:
        raise SystemExit(f"Nessuna settimana completa con data <= {origine}")
    return int(ok[-1])


def calcola(df: pd.DataFrame, t: int, cfg: dict) -> tuple[dict, dict]:
    ris, note = {}, {}
    for v in cfg:
        try:
            ris[v] = VS.prevedi(df, t, v, cfg[v]["scelte"])
        except VS.NonDisponibile as e:
            note[v] = str(e)
    return ris, note


def tabella(prev: dict, cand: list[dict], st: dict) -> pd.DataFrame:
    rows = []
    for s, c in zip(prev["settimane"], cand):
        m = st[s["h"]]
        r = {"versione": prev["versione"], "settimana": f"+{s['h']}", "data": s["data"], "apertura": c["apertura"], "massimo": c["massimo"], "minimo": c["minimo"], "chiusura": c["chiusura"],
             "etichetta_chiusura": c["etichetta"], "p_chiusura_sopra_origine": c["p_chiusura_su"], "chiusura_centrale": s["chiusura"]}
        for lev in ("50", "80", "90"):
            r |= {f"fascia_{lev}_inf": s["fasce"][lev][0], f"fascia_{lev}_sup": s["fasce"][lev][1], f"copertura_storica_{lev}": m[f"copertura_{lev}"]}
        r |= {f"p_sopra_{int(k)}": s["probabilita"][f"VIX>{int(k)}"] for k in G.SOGLIE}
        r |= {"mae_storico_chiusura_centrale": m["mae_centrale"], "massimo_entro_5_storico": m["massimo_entro_5"], "minimo_entro_5_storico": m["minimo_entro_5"],
              "etichetta_quota_non_neutre_storica": m["quota_non_neutre"], "etichetta_accuratezza_storica": m["accuratezza_non_neutre"], "n_previsioni_walk_forward": int(m["n"])}
        rows.append(r)
    return pd.DataFrame(rows)


def tabella_md(tab: pd.DataFrame) -> tuple[str, str]:
    a = [{"settimana": r.settimana, "data": data_it(r.data), "apertura": n(r.apertura), "massimo": n(r.massimo), "minimo": n(r.minimo), "chiusura": n(r.chiusura),
          "chiusura rispetto all'origine": r.etichetta_chiusura, "probabilità che chiuda sopra l'origine": n(100 * r.p_chiusura_sopra_origine, 0) + "%", "chiusura centrale": n(r.chiusura_centrale),
          "massimo / minimo entro il 5% nel passato": f"{n(100 * r.massimo_entro_5_storico, 0)}% / {n(100 * r.minimo_entro_5_storico, 0)}%"} for r in tab.itertuples()]
    b = []
    for r in tab.itertuples():
        row = {"settimana": r.settimana, "chiusura centrale": n(r.chiusura_centrale)}
        for lev in ("50", "80", "90"):
            row[f"fascia {lev}% (copertura misurata)"] = f"{n(getattr(r, f'fascia_{lev}_inf'), 1)} – {n(getattr(r, f'fascia_{lev}_sup'), 1)} ({n(100 * getattr(r, f'copertura_storica_{lev}'), 0)}%)"
        for k in G.SOGLIE:
            row[f"P(VIX9D > {int(k)})"] = n(100 * getattr(r, f"p_sopra_{int(k)}"), 1) + "%"
        row["errore medio storico della chiusura centrale"] = n(r.mae_storico_chiusura_centrale, 1)
        b.append(row)
    return pd.DataFrame(a).to_markdown(index=False), pd.DataFrame(b).to_markdown(index=False)


def confronto(ris: dict) -> pd.DataFrame:
    rows = [{"settimana": f"+{s['h']}", "versione": v, "chiusura_centrale": s["chiusura"], "fascia_80_inf": s["fasce"]["80"][0], "fascia_80_sup": s["fasce"]["80"][1], "massimo": c["massimo"],
             "minimo": c["minimo"], "etichetta_chiusura": c["etichetta"], "p_chiusura_sopra_origine": c["p_chiusura_su"]} for v, (p, cd) in ris.items() for s, c in zip(p["settimane"], cd)]
    return pd.DataFrame(rows).sort_values(["settimana", "versione"], kind="stable").reset_index(drop=True)


def lettura_md(L: dict) -> str:
    lv, dn, pr = L["livello"], L["dinamica"], L["precedenti"]
    out = [f"**Livello.** Chiusura {n(lv['chiusura'])}: percentile {n(lv['percentile_storia'], 0)} di tutta la storia ({L['settimane_di_storia']} settimane dal {data_it(L['prima_data'])}) e "
           f"{n(lv['percentile_5_anni'], 0)} degli ultimi 5 anni; quintile {lv['quintile']} su 5. Mediana storica {n(lv['mediana_storia'])}; livello recente (media a 12 settimane) {n(lv['media_12_settimane'])}.", ""]
    if L["vix"]:
        v = L["vix"]
        out += [f"**Distanza dal VIX.** VIX {n(v['chiusura_vix'])}: il VIX9D è {n(abs(v['distanza_punti']))} punti {'sopra' if v['distanza_punti'] >= 0 else 'sotto'} (rapporto {n(v['rapporto'], 3)}; "
                f"mediana storica del rapporto {n(v['rapporto_mediano_storico'], 3)}; percentile del rapporto {n(v['percentile_rapporto'], 0)}).", ""]
    else:
        out += ["**Distanza dal VIX.** Il VIX della settimana di origine non è disponibile.", ""]
    out += ["**Dinamica.** Variazione della chiusura: " + "; ".join(f"a {k} settiman{'a' if k == 1 else 'e'} {seg(dn[f'variazione_{k}_settimane']['punti'])} punti ({seg(dn[f'variazione_{k}_settimane']['pct'], 1)}%)" for k in (1, 4, 12)) + ".", ""]
    out += [f"**Precedenti in situazioni simili** (stesso quintile di livello, {pr['quintile']}, e stessa direzione a 4 settimane, «{pr['direzione']}»): {pr['n']} casi"
            + (" (meno di 20: indicazione debole)" if pr["meno_di_20_casi"] else "") + "."]
    if pr["n"]:
        out += ["", "| dopo | quota con VIX9D più alto | variazione mediana (punti) | 10° – 90° percentile (punti) |", "|---|---|---|---|"]
        out += [f"| {h} settiman{'a' if h == 1 else 'e'} | {n(100 * pr[f'dopo_{h}']['quota_piu_alto'], 0)}% | {seg(pr[f'dopo_{h}']['mediana_punti'])} | {seg(pr[f'dopo_{h}']['p10_punti'])} – {seg(pr[f'dopo_{h}']['p90_punti'])} |" for h in (1, 2)]
    return "\n".join(out)


def sintesi(L: dict, prev: dict, cand: list[dict]) -> list[str]:
    lv, f = L["livello"], []
    f.append(f"Il VIX9D ha chiuso a {n(lv['chiusura'])} il {data_it(L['data_origine'])}: percentile {n(lv['percentile_storia'], 0)} della storia, "
             f"{n(abs(lv['chiusura'] - lv['media_12_settimane']))} punti {'sopra' if lv['chiusura'] >= lv['media_12_settimane'] else 'sotto'} il livello recente ({n(lv['media_12_settimane'])}).")
    if L["vix"]:
        f.append(f"Rispetto al VIX ({n(L['vix']['chiusura_vix'])}) è {n(abs(L['vix']['distanza_punti']))} punti {'sopra' if L['vix']['distanza_punti'] >= 0 else 'sotto'}: rapporto {n(L['vix']['rapporto'], 3)}, percentile {n(L['vix']['percentile_rapporto'], 0)} della sua storia.")
    for s, c in zip(prev["settimane"], cand):
        d = c["direzione_chiusura"]
        p = c["p_chiusura_su"] if d == "rialzo" else 1 - c["p_chiusura_su"]
        et = "neutra" if d == "neutra" else f"in {d} (probabilità {n(100 * p, 0)}%)"
        f.append(f"Settimana +{s['h']} ({VS.NOMI[prev['versione']]}): chiusura centrale {n(s['chiusura'], 1)}, fascia all'80% tra {n(s['fasce']['80'][0], 1)} e {n(s['fasce']['80'][1], 1)}; "
                 f"massimo previsto {n(c['massimo'], 1)}, minimo {n(c['minimo'], 1)}; direzione della chiusura rispetto all'origine: {et}.")
    return f


def scrivi_md(path: Path, v: str, ris: dict, note: dict, L: dict, tab: pd.DataFrame, conf: pd.DataFrame, st: dict, info: dict, g1: Path, g3: Path) -> None:
    rel = lambda p: os.path.relpath(p, path.parent)
    prev, cand = ris[v]
    m1, m2 = st["versioni"][v][1], st["versioni"][v][2]
    t_cand, t_fasce = tabella_md(tab)
    if len(ris) > 1:
        c = conf.copy()
        s_conf = pd.DataFrame([{"settimana": r.settimana, "versione": VS.NOMI[r.versione], "chiusura centrale": n(r.chiusura_centrale), "fascia 80%": f"{n(r.fascia_80_inf, 1)} – {n(r.fascia_80_sup, 1)}",
                                "massimo": n(r.massimo), "minimo": n(r.minimo), "chiusura rispetto all'origine": r.etichetta_chiusura,
                                "probabilità che chiuda sopra l'origine": n(100 * r.p_chiusura_sopra_origine, 0) + "%"} for r in c.itertuples()]).to_markdown(index=False)
    else:
        s_conf = "L'altra versione non è calcolabile per questa origine" + ("".join(f" ({VS.NOMI[k]}: {m})" for k, m in note.items())) + "."
    md = f"""# Previsione del VIX9D settimanale — origine {data_it(prev['data_origine'])}

*Generato da `python -m src.vix9d.forecast`. Versione principale: **{VS.NOMI[v]}**. Storia usata: {L['settimane_di_storia']} settimane complete dal {data_it(L['prima_data'])} al {data_it(prev['data_origine'])}
({info['settimane_con_candela']} con candela vera: prima di metà novembre 2013 il file Cboe ha solo le chiusure). Tutto ciò che segue usa solo dati fino all'origine;
la previsione si fa una sola volta, alla chiusura settimanale. Progetto universitario: non è un consiglio di investimento.*

## In sintesi

{chr(10).join('- ' + f for f in sintesi(L, prev, cand))}

## 1. Lettura della situazione

{lettura_md(L)}

## 2. Previsione per le settimane +1 e +2 ({VS.NOMI[v]})

Valori in punti VIX9D. **Candele previste**: massimo e minimo sono previsti direttamente e comandano la candela; il corpo è quello di una settimana media, tagliato dentro l'intervallo;
il colore segue la direzione della chiusura rispetto all'origine («neutra» tra 40% e 60%).

{t_cand}

**Chiusura centrale, fasce e probabilità.** Tra parentesi la copertura misurata nel walk-forward di sviluppo ({int(m1['n'])} previsioni a 1 settimana).

{t_fasce}

- Etichetta della direzione nel walk-forward di sviluppo: non neutra nel {n(100 * m1['quota_non_neutre'], 0)}% dei casi a 1 settimana (corretta nel {n(100 * m1['accuratezza_non_neutre'], 0)}%) e nel {n(100 * m2['quota_non_neutre'], 0)}% a 2 (corretta nel {n(100 * m2['accuratezza_non_neutre'], 0)}%).
- Le date sono indicative (origine + 7 giorni per settimana).

## 3. Confronto tra le versioni

{s_conf}

Le versioni sono congelate (`versioni/vix9d/g1.json`, `versioni/vix9d/gs.json`); l'esame sull'holdout è descritto in `reports/vix9d/piano_esame_holdout.md` e non è stato eseguito.

## 4. Grafico

![Grafico principale]({rel(g1)})

- **Candele piene dal bordo continuo** = settimane reali. **Candele dal bordo tratteggiato** = le due settimane previste: rosse se la chiusura è prevista sopra quella dell'origine, blu se sotto, grigie se «neutra».
- **Fasce sullo sfondo** = incertezza sulla chiusura (50% più scura, 80% più chiara). **Linea viola** = chiusura centrale. **Linea arancione** = livello recente (media a 12 settimane). **Linea grigia** = VIX, in secondo piano.
- L'apertura ufficiale del VIX9D è di norma molto più alta della chiusura precedente (mediana storica: {seg(100 * (cand[0]['rapporto_apertura'] - 1), 1)}%) e la settimana chiude sotto la propria apertura nel {n(100 * (1 - cand[0]['p_rialzo']), 0)}% dei casi: per questo il corpo è disegnato verso il basso. Il colore indica la direzione della chiusura rispetto all'origine, non il verso del corpo.

## 5. Verifica visiva sul passato

![Verifica sul passato]({rel(g3)})

{st['g3']['regola'].capitalize()}, tutte nel campione di sviluppo, versione G1. {st['g3']['riga']} Dodici origini sono poche: servono a vedere come si comporta la previsione, non a misurarla.

## 6. Avvertenze

- **Il VIX9D si muove molto più del VIX.** Nel walk-forward di sviluppo il massimo previsto della settimana +1 è stato entro il 5% da quello reale nel {n(100 * m1['massimo_entro_5'], 0)}% dei casi e il minimo nel {n(100 * m1['minimo_entro_5'], 0)}%; a 2 settimane {n(100 * m2['massimo_entro_5'], 0)}% e {n(100 * m2['minimo_entro_5'], 0)}%.
- L'errore medio della chiusura centrale è stato {n(m1['mae_centrale'], 1)} punti a 1 settimana e {n(m2['mae_centrale'], 1)} a 2. Il modello non anticipa i picchi.
- Il corpo viene tagliato dagli estremi nel {n(100 * m1['quota_chiusura_tagliata'], 0)}% dei casi (misura di sviluppo, settimana +1): la chiusura della candela non è la previsione centrale, che è la linea viola.
- Ogni esecuzione senza `--origine` scrive le versioni calcolate in `reports/vix9d/registro_previsioni.csv`; `python -m src.vix9d.forecast --verifica` confronta il registro con le settimane completate.
"""
    path.write_text(md)


def esegui_verifica(df: pd.DataFrame, reg_path: Path, out: Path) -> dict:
    if not reg_path.exists():
        print(f"Il registro {reg_path} non esiste ancora: esegui prima `python -m src.vix9d.forecast`.")
        return {"dettaglio": pd.DataFrame(), "pagella": pd.DataFrame()}
    reg = pd.read_csv(reg_path, dtype={"data_origine": str, "data_esecuzione": str})
    det, pag = RG.verifica(reg, df)
    print(f"\nPAGELLA DAL VIVO (VIX9D) — registro: {reg['data_origine'].nunique()} origini, {len(reg)} righe; righe confrontabili con settimane completate: {len(det)}")
    if det.empty:
        print("Nessuna settimana prevista è ancora stata completata nel file: non c'è nulla da verificare.")
        return {"dettaglio": det, "pagella": pag}
    det.to_csv(out / "registro_verifica_dettaglio.csv", index=False, float_format="%.4f")
    pag.to_csv(out / "registro_verifica_pagella.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(pag.round(3).to_string(index=False))
    if pag["meno_di_20_casi"].any():
        print("\nAttenzione: meno di 20 casi per almeno una versione e un orizzonte. Sono troppo pochi per trarre conclusioni.")
    print("Il registro serve a misurare dal vivo, non a scegliere modelli.")
    return {"dettaglio": det, "pagella": pag}


def main(argv: list[str] | None = None, out_dir: Path | None = None, oggi: str | None = None, file9: Path | None = None, file_vix: Path | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Previsione del VIX9D settimanale: 2 candele (settimane +1 e +2), chiusura centrale, fasce, probabilità.")
    ap.add_argument("--versione", default="g1", choices=VS.VERSIONI, help="versione principale della scheda (predefinita: g1)")
    ap.add_argument("--aggiorna", action="store_true", help="scarica VIX9D e VIX giornalieri da Cboe e ricostruisce i file settimanali completi")
    ap.add_argument("--origine", default=None, help="data di origine AAAA-MM-GG (predefinito: ultima settimana completa)")
    ap.add_argument("--verifica", action="store_true", help="confronta le previsioni del registro con le settimane completate")
    a = ap.parse_args(argv)
    out = out_dir or OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    reg_path = (out_dir or G.OUT) / "registro_previsioni.csv"
    if a.aggiorna:
        try:
            agg = aggiorna()
            print("Dati giornalieri scaricati e file settimanali ricostruiti: " + "; ".join(f"{k} ({v} byte)" for k, v in agg.items()))
        except Exception as e:
            print(f"ATTENZIONE: aggiornamento non riuscito ({e}). Si usano i file già presenti.")
    df, info = leggi_storico(file9 or G.FILE_COMPLETO, (file_vix or G.FILE_VIX_COMPLETO) if (file9 is None or file_vix is not None) else None)
    for m in info["avvisi"]:
        print(m)
    if a.verifica:
        return esegui_verifica(df, reg_path, out) | {"info": info}
    cfg, st = carica_versioni(), statistiche_sviluppo()
    t = indice_origine(df, a.origine)
    ris, note = calcola(df, t, cfg)
    if "g1" not in ris:
        raise SystemExit(f"Previsione non calcolabile all'origine richiesta: {note.get('g1')}")
    v = a.versione
    if v not in ris:
        print(f"ATTENZIONE: la versione {v} non è calcolabile ({note.get(v, 'non congelata')}). Si ripiega su g1.")
        v = "g1"
    elif "gs" in cfg and "gs" not in ris:
        print(f"Confronto con gs non disponibile: {note['gs']}.")
    prev, cand = ris[v]
    L = LE.lettura(df, t)
    d0, suff = prev["data_origine"], "" if v == "g1" else f"_{v}"
    tab, conf = tabella(prev, cand, st["versioni"][v]), confronto(ris)
    tab.to_csv(out / f"tabella_{d0}{suff}.csv", index=False, float_format="%.4f")
    conf.to_csv(out / f"confronto_{d0}.csv", index=False, float_format="%.4f")
    frasi = sintesi(L, prev, cand)
    (out / f"previsione_{d0}{suff}.json").write_text(json.dumps({"file": info, "versione": v, "versione_richiesta": a.versione, "lettura": L, "sintesi": frasi,
                                                                  "versioni": {k: {"previsione": p, "candele": c} for k, (p, c) in ris.items()}, "note": note}, indent=2, ensure_ascii=False))
    g1 = out / f"g1_{d0}{suff}.png"
    grafici.principale(df.iloc[: t + 1], grafici.vista(prev, cand), g1)
    corrente = a.origine is None and out_dir is None and v == "g1"
    md = (G.OUT / "previsione_corrente.md") if corrente else (out / f"previsione_{d0}{suff}.md")
    scrivi_md(md, v, ris, note, L, tab, conf, st, info, g1, G.FIG / "g3_verifica_passato.png")
    msg, registrate = ["Registro: non aggiornato (con --origine la previsione non è una previsione dal vivo)."], []
    if a.origine is None:
        msg = []
        for k, (p, c) in ris.items():
            ok, m = RG.aggiungi(reg_path, RG.righe(p, c, oggi or datetime.now().strftime("%Y-%m-%d")))
            msg.append(m)
            if ok:
                registrate.append(k)
    print(f"\nPREVISIONE DEL VIX9D — origine {data_it(d0)} (chiusura {n(prev['chiusura_origine'])}); storia: {L['settimane_di_storia']} settimane; versione {VS.NOMI[v]}")
    print("\n" + "\n".join("  " + f for f in frasi) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab[["settimana", "data", "apertura", "massimo", "minimo", "chiusura", "etichetta_chiusura", "p_chiusura_sopra_origine", "chiusura_centrale", "fascia_50_inf", "fascia_50_sup",
                   "fascia_80_inf", "fascia_80_sup", "p_sopra_20", "p_sopra_25", "p_sopra_30"]].round(2).to_string(index=False))
        if len(ris) > 1:
            print("\nConfronto tra le versioni:")
            print(conf.round(2).to_string(index=False))
    print("\n" + "\n".join(msg))
    print(f"Scheda: {md}\nTabella: {out / f'tabella_{d0}{suff}.csv'}\nGrafico: {g1}")
    return {"versione": v, "versioni": ris, "note": note, "previsione": prev, "candele": cand, "lettura": L, "tabella": tab, "confronto": conf, "md": md, "info": info, "registrate": registrate}


if __name__ == "__main__":
    main(sys.argv[1:])
