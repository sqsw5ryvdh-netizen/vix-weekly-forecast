"""Programma di previsione del VIX settimanale.

    python -m src.forecast [--versione f1|fs] [--aggiorna] [--file percorso_csv] [--origine AAAA-MM-GG]
    python -m src.forecast --verifica [--file percorso_csv]

Legge l'intero storico (stesso formato del file grezzo), esclude la settimana incompleta, parte dall'ultima chiusura
settimanale completa (o dall'origine indicata) e calcola le due versioni congelate nella Fase 4b (versioni/*.json):
  f1  solo storico del VIX (predefinita);
  fs  sfidante: costruita attorno al candidato scelto nella Fase 4b; se usa la struttura a termine richiede i file
      giornalieri di VIX e VIX3M, che `--aggiorna` scarica e che vengono usati solo fino all'ultima seduta della
      settimana di origine; se VIX3M manca o e' piu' vecchio dell'origine il programma avvisa e ripiega su f1.
La previsione si fa una sola volta, alla chiusura settimanale, e non si aggiorna durante la settimana.
La scheda mostra la versione richiesta e una tabella di confronto con l'altra versione, se calcolabile.

E' L'UNICO comando autorizzato a leggere i dati successivi al confine di sviluppo (VIX e, per fs, VIX3M), e solo come
storia da cui prevedere. Errori e coperture «storici» mostrati accanto alla previsione sono quelli misurati sul
campione di sviluppo e raccolti da `run_phase4b`, e qui vengono solo letti. Unica eccezione, esplicita: con
`--verifica` le previsioni gia' scritte nel registro vengono confrontate con le settimane completate.
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

from . import config as C
from .fase2b import grafici
from .fase2b import lettura as LE
from .fase2b import modello as MO
from .fase2b import testo as T2B
from .fase2c import settimana as SE
from .fase4 import esterni as ES
from .fase4 import registro as RG
from .fase4b import testo
from .fase4b import versioni as VS

STAT_3, STAT_4B = C.REPORTS_DIR / "fase3", C.REPORTS_DIR / "fase4b"
OUT_DIR = C.REPORTS_DIR / "previsione"
COLONNE_FILE = ["week_start", "last_trading_day", "open", "high", "low", "close", "trading_days", "note"]
CAMPI = ("apertura", "massimo", "minimo", "chiusura")


def leggi_storico(path: Path) -> tuple[pd.DataFrame, dict]:
    """Legge il file nel formato grezzo ed esclude le settimane incomplete (nota, oppure ultima riga con sedute mancanti)."""
    raw = pd.read_csv(path)
    manca = [c for c in COLONNE_FILE if c not in raw.columns]
    if manca:
        raise SystemExit(f"Il file {path} non ha il formato atteso: mancano le colonne {manca}")
    raw = raw.sort_values("week_start").reset_index(drop=True)
    incomplete = raw["note"].notna() & (raw["note"].astype(str).str.strip() != "")
    avvisi = []
    if len(raw) and not incomplete.iloc[-1]:
        u = raw.iloc[-1]
        inc, msg = SE.ultima_riga_incompleta(u["week_start"], u["last_trading_day"], u["trading_days"], u["note"])
        if msg:
            avvisi.append(msg)
        if inc:
            incomplete.iloc[-1] = True
    df = raw.loc[~incomplete].copy()
    df["date"] = pd.to_datetime(df["last_trading_day"])
    df = df.sort_values("date").reset_index(drop=True)[["date", "open", "high", "low", "close"]]
    if df["date"].duplicated().any() or (df[["open", "high", "low", "close"]] <= 0).any().any() or df[["open", "high", "low", "close"]].isna().any().any():
        raise SystemExit(f"Il file {path} contiene date duplicate o valori mancanti / non positivi")
    passi = pd.to_datetime(raw.loc[~incomplete, "week_start"]).sort_values().diff().dt.days.dropna()
    info = {"file": str(path), "righe": int(len(raw)), "settimane_complete": int(len(df)), "settimane_incomplete_escluse": int(incomplete.sum()),
            "prima_data": str(df["date"].iloc[0].date()), "ultima_data_completa": str(df["date"].iloc[-1].date()),
            "settimane_mancanti_nel_file": int(((passi[passi > 7] / 7) - 1).sum()), "avvisi": avvisi}
    return df, info


def indice_origine(df: pd.DataFrame, origine: str | None) -> int:
    if origine is None:
        return len(df) - 1
    ok = np.flatnonzero(df["date"].to_numpy() <= np.datetime64(pd.Timestamp(origine)))
    if len(ok) == 0:
        raise SystemExit(f"Nessuna settimana completa con data <= {origine}")
    return int(ok[-1])


def carica_versioni() -> dict:
    """Configurazioni congelate (versioni/f1.json e versioni/fs.json). Senza f1 il programma non parte."""
    cfg = {v: json.loads((VS.CARTELLA / f"{v}.json").read_text()) for v in VS.VERSIONI if (VS.CARTELLA / f"{v}.json").exists()}
    if "f1" not in cfg:
        raise SystemExit("Mancano le versioni congelate (versioni/f1.json): eseguire `python -m src.run_phase4b` e poi `python -m src.fase4b.congela`.")
    return cfg


def statistiche_sviluppo() -> dict:
    """Misure del walk-forward di sviluppo, per versione e orizzonte (raccolte da run_phase4b). Qui vengono solo lette."""
    try:
        sp = pd.read_csv(STAT_4B / "statistiche_programma.csv")
        p0 = json.loads((STAT_3 / "p0_info.json").read_text())
        g3 = json.loads((STAT_4B / "info.json").read_text())["g3"]
    except FileNotFoundError:
        raise SystemExit("Mancano le misure del walk-forward di sviluppo: eseguire prima `python -m src.run_phase2b`, `python -m src.run_phase2c`, "
                         "`python -m src.run_phase3`, `python -m src.run_phase4` e `python -m src.run_phase4b`.")
    st = {}
    for v, g in sp.groupby("versione"):
        righe = {int(r["h"]): r | {"quota_neutra": 1 - r["quota_non_neutre"]} for r in g.to_dict("records")}
        st[v] = {"chiusura": righe, "pagella": righe, "etichetta": righe, "campione": g["campione"].iloc[0]}
    return {"versioni": st, "candele": {"rapporto_apertura": p0["rapporto_apertura_ultimo"], "frequenza_rialzo": p0["frequenza_rialzo_corpo_ultima"]}, "g3": g3}


def calcola(df: pd.DataFrame, t: int, cfg: dict, cartella_esterni: Path | None) -> tuple[dict, dict]:
    """Le versioni calcolabili all'origine t (solo df.iloc[: t + 1] e, per fs, dati giornalieri fino all'origine).

    Restituisce ({versione: (previsione, candele)}, {versione: informazioni o motivo dell'assenza})."""
    scelte = cfg["f1"]["scelte"]
    ris, note = {"f1": VS.prevedi_f1(df, t, scelte)}, {}
    if "fs" in cfg:
        z, info = None, {"disponibile": True, "avvisi": []}
        if "f2" in VS.CANDIDATI[cfg["fs"]["scelte"]["candidato"]]:
            z, info = ES.x1_per_previsione(df, t, cartella_esterni)
        note["fs"] = info
        if info["disponibile"]:
            try:
                ris["fs"] = VS.prevedi_fs(df, t, z, cfg["fs"]["scelte"])
            except VS.NonDisponibile as e:
                note["fs"] = info | {"disponibile": False, "motivo": str(e)}
    return ris, note


def vista_grafico(prev: dict, cand: list[dict]) -> dict:
    """Previsione con le candele K2 al posto di quelle della Fase 2b (fasce e chiusura centrale restano quelle della versione)."""
    return prev | {"settimane": [s | {k: c[k] for k in CAMPI} | {"etichetta": "chiusura:\n" + c["etichetta"]} | ({"colore": c["colore"]} if "colore" in c else {})
                                 for s, c in zip(prev["settimane"], cand)]}


def scrivi_md(path: Path, v: str, ris: dict, note: dict, L: dict, tab: pd.DataFrame, conf: pd.DataFrame, st: dict, cfg: dict, info: dict, g1: Path, g3: Path, nota_origine: str) -> None:
    rel = lambda p: os.path.relpath(p, path.parent)
    n = T2B.n
    prev, cand = ris[v]
    sv, k = st["versioni"][v], st["candele"]
    c4, e1, e4 = sv["chiusura"][4], sv["etichetta"][1], sv["etichetta"][4]
    centrale, prob = testo.da_dove(v, cfg)
    prima = cfg[v]["candele"]["prima_candela"]
    t_cand, t_fasce = testo.tabella_md(tab)
    if len(ris) > 1:
        s_conf = (f"In questa scheda la versione principale è **{VS.NOMI[v]}**. Le due versioni sono quelle che andranno all'esame sull'holdout.\n\n" + testo.confronto_md(conf))
    else:
        s_conf = f"L'altra versione non è calcolabile per questa origine: la scheda mostra solo **{VS.NOMI[v]}**."
    assenti = [f"- **{VS.NOMI[x]}** non è nel confronto: {note[x].get('motivo', 'non disponibile')}." for x in note if x not in ris]
    dettagli = ""
    if "fs" in ris:
        ps, c1 = ris["fs"][0], ris["fs"][1][0]
        membri = ", ".join(f"{m.upper()} {n(100 * q, 0)}%" for m, q in c1["p_chiusura_su_membri"].items())
        dettagli = (f"\n\nFS è costruita attorno al candidato {cfg['fs']['candidato']['descrizione']}: per la settimana +1 la probabilità che la chiusura superi quella dell'origine è la media di {membri}; "
                    f"chiusura centrale, fasce e settimane +2…+4 vengono dalla costruzione di {ps['costruzione_di_base'].upper()}.")
        if "x1_origine" in note["fs"]:
            u = note["fs"]["x1_origine"]
            dettagli += (f" Dati usati per l'origine: VIX3M {n(u['vix3m'])} e VIX {n(u['vix'])} del {T2B.data_it(u['data_osservazione'])} (VIX3M / VIX − 1 = {T2B.seg(u['x1'], 3)}); "
                         f"file Cboe aggiornati fino al {T2B.data_it(note['fs']['ultima_data_nei_file']['vix3m'])}, usati solo fino all'origine.")
    md = f"""# Previsione del VIX settimanale — origine {T2B.data_it(prev['data_origine'])}

*Generato da `python -m src.forecast`. Versione principale: **{VS.NOMI[v]}**. Storia usata: {L['settimane_di_storia']} settimane complete dal {T2B.data_it(L['prima_data'])} al {T2B.data_it(prev['data_origine'])}
(file `{Path(info['file']).name}`; settimane incomplete escluse: {info['settimane_incomplete_escluse']}).{nota_origine} Tutto ciò che segue usa solo dati fino all'origine;
la previsione si fa una sola volta, alla chiusura settimanale. Progetto universitario: non è un consiglio di investimento.*

## In sintesi

{chr(10).join('- ' + f for f in testo.sintesi(L, prev, cand))}

## 1. Lettura della situazione

{T2B.lettura_md(L)}

## 2. Previsione per le prossime 4 settimane ({VS.NOMI[v]})

Valori in punti VIX. Due livelli, da leggere insieme.

**Candele previste.** La prima (settimana +1) è colorata secondo l'etichetta della chiusura{' e il suo minimo viene da un modello dedicato' if prima['minimo'] == 'M1q' else ''}; le altre sono settimane di dimensioni medie.
La colonna «chiusura rispetto all'origine» è l'etichetta scritta sopra le candele nel grafico: «neutra» tra 40% e 60%.

{t_cand}

**Chiusura centrale, fasce e probabilità** (chiusura centrale: {centrale}; probabilità dell'etichetta: {prob}). Tra parentesi la copertura storica di ogni fascia,
cioè la frequenza con cui la chiusura reale vi è caduta nel walk-forward di sviluppo ({int(c4['n'])} previsioni a 4 settimane; {sv['campione']}). Le fasce non sono ricalibrate.

{t_fasce}

- L'**etichetta della chiusura** nel walk-forward di sviluppo è stata «neutra» nel {n(100 * e1['quota_neutra'], 0)}% dei casi a 1 settimana e nel {n(100 * e4['quota_neutra'], 0)}% a 4; quando non era neutra
  è stata corretta nel {n(100 * e1['accuratezza_non_neutre'], 0)}% dei casi a 1 settimana ({int(e1['n_non_neutre'])} casi) e nel {n(100 * e4['accuratezza_non_neutre'], 0)}% a 4 ({int(e4['n_non_neutre'])} casi).
- Gli errori storici sono errori assoluti medi in punti VIX misurati sul campione di sviluppo. Le date sono indicative (origine + 7 giorni per settimana).

## 3. Confronto tra le versioni

{s_conf}
{(chr(10) + chr(10).join(assenti)) if assenti else ''}{dettagli}

Le versioni sono congelate (`versioni/f1.json`, `versioni/fs.json`) e saranno confrontate sull'holdout secondo `reports/piano_esame_holdout.md`.
Sul campione di sviluppo le differenze tra le versioni sono piccole (`reports/fase4_report.md`, `reports/fase4b_report.md`).

## 4. Grafico

![Grafico principale]({rel(g1)})

In alto gli ultimi 2 anni; in basso l'ingrandimento sulle ultime 16 settimane, con i percorsi dei casi simili (nel grafico a 2 anni le quattro settimane
di previsione sono troppo strette per ospitarli).

### Come leggere il grafico

{testo.come_leggere(sv, k, v, cfg)}

## 5. Verifica visiva sul passato

![Verifica sul passato]({rel(g3)})

Dodici origini scelte con regola fissa ({st['g3']['regola']}), tutte nel campione di sviluppo, con la versione F1. {st['g3']['riga']}
Dodici casi sono pochi: servono a vedere come si comporta la previsione, non a misurarla. Le misure sono in `fase4b_report.md`.

## 6. Avvertenze

- Prima candela, misure di sviluppo a 1 settimana: errore medio sul minimo {n(sv['pagella'][1]['mae_minimo'], 2)} punti, sulla chiusura della candela {n(sv['pagella'][1]['mae_chiusura'], 2)}, sul massimo {n(sv['pagella'][1]['mae_massimo'], 2)}. Il massimo non è ricalibrato.
- La chiusura della candela **non** è la previsione centrale della chiusura: quella è la chiusura centrale (linea viola).
- Copertura misurata delle fasce a 4 settimane: quella al 50% ha contenuto la chiusura reale nel {n(100 * c4['copertura_50'], 0)}% dei casi, quella all'80% nel {n(100 * c4['copertura_80'], 0)}%, quella al 90% nel {n(100 * c4['copertura_90'], 0)}%.
- Candele, fasce e probabilità vengono da pezzi diversi del modello e possono non essere perfettamente coerenti tra loro.
- Il modello non anticipa i picchi.
- Ogni esecuzione senza `--origine` scrive le versioni calcolate in `reports/registro_previsioni.csv` (una riga per origine, versione e settimana); `python -m src.forecast --verifica` confronta il registro con le settimane completate.
"""
    path.write_text(md)


def esegui_verifica(df: pd.DataFrame, reg_path: Path, out: Path) -> dict:
    """Pagella dal vivo: confronto tra le previsioni del registro e le settimane completate nel file, per versione."""
    if not reg_path.exists():
        print(f"Il registro {reg_path} non esiste ancora: esegui prima `python -m src.forecast`.")
        return {"dettaglio": pd.DataFrame(), "pagella": pd.DataFrame()}
    reg = pd.read_csv(reg_path, dtype={"data_origine": str, "data_esecuzione": str})
    det, pag = RG.verifica(reg, df)
    print(f"\nPAGELLA DAL VIVO — registro: {reg['data_origine'].nunique()} origini, {len(reg)} righe; righe confrontabili con settimane completate: {len(det)}")
    if det.empty:
        print("Nessuna settimana prevista è ancora stata completata nel file: non c'è nulla da verificare.")
        return {"dettaglio": det, "pagella": pag}
    det.to_csv(out / "registro_verifica_dettaglio.csv", index=False, float_format="%.4f")
    pag.to_csv(out / "registro_verifica_pagella.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(pag.round(3).to_string(index=False))
    if pag["meno_di_20_casi"].any():
        print("\nAttenzione: meno di 20 casi per almeno una versione e un orizzonte. Sono troppo pochi per trarre conclusioni.")
    if not det["registrata_prima_dell_esito"].all():
        print("Attenzione: alcune previsioni sono state registrate dopo la data della settimana prevista (colonna registrata_prima_dell_esito).")
    print("Il registro serve a misurare dal vivo, non a scegliere modelli.")
    print(f"\nDettaglio: {out / 'registro_verifica_dettaglio.csv'}\nPagella: {out / 'registro_verifica_pagella.csv'}")
    return {"dettaglio": det, "pagella": pag}


def main(argv: list[str] | None = None, out_dir: Path | None = None, oggi: str | None = None, esterni_dir: Path | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Previsione del VIX settimanale a 1-4 settimane: candele, chiusura centrale, fasce, probabilità.")
    ap.add_argument("--file", default=str(C.RAW_FILE), help="CSV nel formato del file grezzo (predefinito: data/raw/...)")
    ap.add_argument("--origine", default=None, help="data di origine AAAA-MM-GG (predefinito: ultima settimana completa)")
    ap.add_argument("--versione", default="f1", choices=VS.VERSIONI, help="versione principale della scheda (predefinita: f1)")
    ap.add_argument("--aggiorna", action="store_true", help="scarica VIX e VIX3M giornalieri (servono a fs); vengono usati solo fino all'origine")
    ap.add_argument("--verifica", action="store_true", help="confronta le previsioni del registro con le settimane completate")
    a = ap.parse_args(argv)
    out = out_dir or OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    reg_path = (out_dir or C.REPORTS_DIR) / "registro_previsioni.csv"
    df, info = leggi_storico(Path(a.file))
    for m in info["avvisi"]:
        print(m)
    if a.verifica:
        return esegui_verifica(df, reg_path, out) | {"info": info}
    cfg = carica_versioni()
    if a.versione not in cfg:
        raise SystemExit(f"La versione {a.versione} non è tra quelle congelate ({', '.join(cfg)}).")
    st = statistiche_sviluppo()
    t = indice_origine(df, a.origine)
    cartella = esterni_dir if esterni_dir is not None else (None if out_dir is not None else ES.CARTELLA)     # nei test (out_dir) i dati esterni si passano esplicitamente
    if a.aggiorna:
        if cartella is None:
            raise SystemExit("--aggiorna richiede una cartella per i dati esterni")
        try:
            agg = ES.aggiorna(cartella)
            print("Dati giornalieri scaricati: " + "; ".join(f"{k} ({v['bytes']} byte)" for k, v in agg.items()) + f" in {cartella}")
        except Exception as e:                                  # rete assente o fonte non raggiungibile: si prosegue con i file gia' presenti, se ci sono
            print(f"ATTENZIONE: scarico dei dati giornalieri non riuscito ({e}). Si usano i file già presenti, se ci sono.")
    ris, note = calcola(df, t, cfg, cartella)
    for x in note.values():
        for m in x.get("avvisi", []):
            print(m)
    v = a.versione
    if v not in ris:
        print(f"ATTENZIONE: la versione {v} non è calcolabile ({note[v].get('motivo', 'dati non disponibili')}). Si ripiega su f1.")
        v = "f1"
    elif "fs" in cfg and "fs" not in ris:
        print(f"Confronto con fs non disponibile: {note['fs'].get('motivo', 'dati non disponibili')}.")
    prev, cand = ris[v]
    L = LE.lettura(df, t)
    d0 = prev["data_origine"]
    suff = "" if v == "f1" else f"_{v}"
    nota = ""
    if a.origine and d0 != str(pd.Timestamp(a.origine).date()):
        nota = f" Origine richiesta {T2B.data_it(a.origine)}: usata l'ultima chiusura settimanale disponibile a quella data."
    tab = testo.tabella(prev, cand, st["versioni"][v])
    conf = testo.confronto(ris)
    tab.to_csv(out / f"tabella_{d0}{suff}.csv", index=False, float_format="%.4f")
    conf.to_csv(out / f"confronto_{d0}.csv", index=False, float_format="%.4f")
    frasi = testo.sintesi(L, prev, cand)
    pulite = {k: {kk: vv for kk, vv in x.items() if kk != "valori_usati"} for k, x in note.items()}
    (out / f"previsione_{d0}{suff}.json").write_text(json.dumps({"file": info, "versione": v, "versione_richiesta": a.versione, "lettura": L, "sintesi": frasi,
                                                                  "versioni": {k: {"previsione": p, "candele": c} for k, (p, c) in ris.items()}, "note": pulite}, indent=2, ensure_ascii=False))
    if "fs" in ris and "valori_usati" in note["fs"]:            # valori di X1 usati da fs (tutta la serie settimanale fino all'origine)
        note["fs"]["valori_usati"].to_csv(out / f"fs_valori_usati_{d0}.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    storia, vista = df.iloc[: t + 1], vista_grafico(prev, cand)
    g1, g2 = out / f"g1_{d0}{suff}.png", out / f"g2_{d0}{suff}.png"
    etichetta_centrale = f"chiusura centrale ({testo.da_dove(v, cfg)[0]})"
    grafici.g1(storia, vista, L, g1, mediana=True, recente=True, centrale=etichetta_centrale, etichette=True)
    grafici.g2(storia, vista, g2, mediana=True, recente=True)
    corrente = a.origine is None and out_dir is None and v == "f1"
    md = (C.REPORTS_DIR / "previsione_corrente.md") if corrente else (out / f"previsione_{d0}{suff}.md")
    scrivi_md(md, v, ris, note, L, tab, conf, st, cfg, info, g1, C.FIG_DIR / "fase4b" / "g3_verifica_passato.png", nota)
    msg_reg, registrate = ["Registro: non aggiornato (con --origine la previsione non è una previsione dal vivo)."], []
    if a.origine is None:
        msg_reg = []
        for k, (p, c) in ris.items():
            ok, m = RG.aggiungi(reg_path, RG.righe(p, c, oggi or datetime.now().strftime("%Y-%m-%d")))
            msg_reg.append(m)
            if ok:
                registrate.append(k)

    print(f"\nPREVISIONE DEL VIX — origine {T2B.data_it(d0)} (chiusura {T2B.n(prev['chiusura_origine'])}); storia: {L['settimane_di_storia']} settimane; versione {VS.NOMI[v]}")
    print("\n" + "\n".join("  " + f for f in frasi) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab[["settimana", "data", "apertura", "massimo", "minimo", "chiusura", "etichetta_chiusura", "p_chiusura_sopra_origine", "chiusura_centrale", "fascia_50_inf", "fascia_50_sup",
                   "fascia_80_inf", "fascia_80_sup", "fascia_90_inf", "fascia_90_sup", "p_vix_sopra_20", "p_vix_sopra_25", "p_vix_sopra_30"]].round(2).to_string(index=False))
        if len(ris) > 1:
            print("\nConfronto tra le versioni:")
            print(conf.round(2).to_string(index=False))
        print(f"\nCandele: {cand[0]['costruzione']}. Etichetta: chiusura rispetto all'origine (neutra tra 40% e 60%). Misure storiche (walk-forward di sviluppo; {st['versioni'][v]['campione']}):")
        print(tab[["settimana", "copertura_storica_50", "copertura_storica_80", "copertura_storica_90", "mae_storico_chiusura_centrale", "mae_storico_apertura", "mae_storico_massimo",
                   "mae_storico_minimo", "mae_storico_chiusura", "etichetta_quota_neutra_storica", "etichetta_accuratezza_storica_non_neutre",
                   "n_previsioni_walk_forward"]].round(3).to_string(index=False))
    print("\n" + "\n".join(msg_reg))
    print(f"Scheda: {md}\nTabella: {out / f'tabella_{d0}{suff}.csv'}\nLettura e previsione (JSON): {out / f'previsione_{d0}{suff}.json'}\nGrafici: {g1}, {g2}")
    return {"versione": v, "versioni": ris, "note": note, "previsione": prev, "candele": cand, "lettura": L, "tabella": tab, "confronto": conf, "md": md, "info": info,
            "registrata": bool(registrate), "registrate": registrate}


if __name__ == "__main__":
    main(sys.argv[1:])
