"""Programma di previsione del VIX settimanale.

    python -m src.forecast [--file percorso_csv] [--origine AAAA-MM-GG]
    python -m src.forecast --verifica [--file percorso_csv]

Legge l'intero storico (stesso formato del file grezzo), esclude la settimana incompleta, parte dall'ultima chiusura
settimanale completa (o dall'origine indicata) e stima i modelli sulla storia fino a quella data.

E' L'UNICO comando autorizzato a leggere i dati successivi al confine di sviluppo, e solo come storia da cui
prevedere. Errori e coperture «storici» mostrati accanto alla previsione sono quelli misurati sul campione di
sviluppo da `run_phase2b` e `run_phase3` (Parte 0), e qui vengono solo letti. Unica eccezione, esplicita: con
`--verifica` le previsioni gia' scritte nel registro vengono confrontate con le settimane nel frattempo completate.

Dalla Fase 3 (Parte 0) le candele sono K2 con ampiezze medie e direzione del corpo dalla frequenza storica;
l'etichetta sopra ogni candela e' la direzione della chiusura rispetto all'origine (probabilita' D1). I modelli con
dati giornalieri e di mercato misurati nella Fase 3 NON sono usati qui.
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
from .fase2c import registro as RG
from .fase2c import settimana as SE
from .fase3 import candele as KA
from .fase3 import registro as RG3
from .fase3 import testo

STAT_2B, STAT_3 = C.REPORTS_DIR / "fase2b", C.REPORTS_DIR / "fase3"
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


def statistiche_sviluppo() -> dict:
    """Misure del walk-forward di sviluppo (prodotte da run_phase2b e run_phase3). Qui vengono solo lette."""
    try:
        ch = pd.read_csv(STAT_2B / "chiusura_metriche.csv").set_index("h")
        info = json.loads((STAT_3 / "p0_info.json").read_text())
        pg = pd.read_csv(STAT_3 / "p0_pagella.csv")
        et = pd.read_csv(STAT_3 / "p0_etichetta.csv").set_index("h")
    except FileNotFoundError:
        raise SystemExit("Mancano le misure del walk-forward di sviluppo: eseguire prima `python -m src.run_phase2b`, `python -m src.run_phase2c` "
                         "e `python -m src.run_phase3`.")
    pg = pg[pg["costruzione"] == info["costruzione"]].set_index("h")
    scelte = {c: {int(h): v for h, v in d.items()} for c, d in info["scelte"].items()}
    return {"chiusura": {h: ch.loc[h].to_dict() for h in MO.ORIZZONTI}, "pagella": {h: pg.loc[h].to_dict() for h in MO.ORIZZONTI},
            "etichetta": {h: et.loc[h].to_dict() for h in MO.ORIZZONTI},
            "candele": {"costruzione": info["costruzione"], "nome": info["nome"], "scelte": scelte, "rapporto_apertura": info["rapporto_apertura_ultimo"],
                        "frequenza_rialzo": info["frequenza_rialzo_corpo_ultima"]},
            "g3": info["g3"]}


def calcola(df: pd.DataFrame, t: int, st: dict) -> tuple[dict, list[dict], dict]:
    """Previsione (mediane, fasce, probabilita'), candele della nuova definizione e lettura: solo df.iloc[: t + 1]."""
    prev, cand = KA.prevedi_candele(df, t, st["candele"]["scelte"])
    return prev, cand, LE.lettura(df, t)


def vista_grafico(prev: dict, cand: list[dict]) -> dict:
    """Previsione con le candele scelte al posto di quelle della Fase 2b (fasce e mediane restano quelle di D4)."""
    return prev | {"settimane": [s | {k: c[k] for k in CAMPI} | {"etichetta": "chiusura:\n" + c["etichetta"]} for s, c in zip(prev["settimane"], cand)]}


def scrivi_md(path: Path, prev: dict, cand: list[dict], L: dict, tab: pd.DataFrame, st: dict, info: dict, g1: Path, g3: Path, nota_origine: str) -> None:
    rel = lambda p: os.path.relpath(p, path.parent)
    n = T2B.n
    c4, k, e1, e4 = st["chiusura"][4], st["candele"], st["etichetta"][1], st["etichetta"][4]
    amp = "; ".join(f"{KA.NOMI_COMPONENTI[c]}: " + ("regressione" if len(set(v.values())) == 1 and "regressione" in v.values() else
                                                  "media storica" if len(set(v.values())) == 1 else
                                                  "regressione per le settimane " + ", ".join(f"+{h}" for h, x in v.items() if x == "regressione") + ", media storica per le altre")
                    for c, v in k["scelte"].items())
    t_cand, t_fasce = testo.tabella_md(tab)
    md = f"""# Previsione del VIX settimanale — origine {T2B.data_it(prev['data_origine'])}

*Generato da `python -m src.forecast`. Storia usata: {L['settimane_di_storia']} settimane complete dal {T2B.data_it(L['prima_data'])} al {T2B.data_it(prev['data_origine'])}
(file `{Path(info['file']).name}`; settimane incomplete escluse: {info['settimane_incomplete_escluse']}).{nota_origine} Tutto ciò che segue usa solo dati fino all'origine.
Progetto universitario: non è un consiglio di investimento.*

## In sintesi

{chr(10).join('- ' + f for f in testo.sintesi(L, prev, cand))}

## 1. Lettura della situazione

{T2B.lettura_md(L)}

## 2. Previsione per le prossime 4 settimane

Valori in punti VIX. Due livelli, da leggere insieme.

**Candele previste** (costruzione {k['nome']}): una settimana di dimensioni medie. La colonna «chiusura rispetto all'origine»
è l'etichetta scritta sopra le candele nel grafico: direzione della chiusura rispetto alla chiusura dell'origine, «neutra» tra 40% e 60%.

{t_cand}

**Chiusura centrale, fasce e probabilità** (mediana e fasce: modello D4; probabilità: modello D1). Tra parentesi la copertura storica di ogni fascia,
cioè la frequenza con cui la chiusura reale vi è caduta nel walk-forward di sviluppo ({int(c4['n'])} previsioni a 4 settimane). Le fasce non sono ricalibrate.

{t_fasce}

- L'**etichetta della chiusura** usa la probabilità del modello D1 (lo stesso delle probabilità di superare 20, 25 e 30). Nel walk-forward di sviluppo è stata «neutra» nel
  {n(100 * e1['quota_neutra'], 0)}% dei casi a 1 settimana e nel {n(100 * e4['quota_neutra'], 0)}% a 4; quando non era neutra è stata corretta nel {n(100 * e1['accuratezza_non_neutre'], 0)}% dei casi a 1 settimana
  ({int(e1['n_non_neutre'])} casi) e nel {n(100 * e4['accuratezza_non_neutre'], 0)}% a 4 ({int(e4['n_non_neutre'])} casi).
- Il **corpo** della candela è disegnato nella direzione storicamente più frequente (chiusura sotto l'apertura nel {n(100 * (1 - k['frequenza_rialzo']), 0)}% delle settimane passate):
  è una frequenza, non una previsione. Ampiezze: {amp}.
- Gli errori storici sono errori assoluti medi in punti VIX misurati sul campione di sviluppo. Le date sono indicative (origine + 7 giorni per settimana).

## 3. Grafico

![Grafico principale]({rel(g1)})

In alto gli ultimi 2 anni; in basso l'ingrandimento sulle ultime 16 settimane, con i percorsi dei casi simili (nel grafico a 2 anni le quattro settimane
di previsione sono troppo strette per ospitarli).

### Come leggere il grafico

{testo.come_leggere(st)}

## 4. Verifica visiva sul passato

![Verifica sul passato]({rel(g3)})

Dodici origini scelte con regola fissa ({st['g3']['regola']}), tutte nel campione di sviluppo. {st['g3']['riga']}
Dodici casi sono pochi: servono a vedere come si comporta la previsione, non a misurarla. Le misure sono in `fase3_report.md` (Parte 0).

## 5. Avvertenze

- Le candele previste restano un po' più piccole di quelle reali: nel walk-forward, a 1 settimana, il rapporto tra corpo medio previsto e reale è {n(100 * st['pagella'][1]['rapporto_corpo'], 0)}% e quello tra escursione prevista e reale è {n(100 * st['pagella'][1]['rapporto_escursione'], 0)}%. Le ampiezze sono medie del passato, e negli ultimi anni corpi e ombre sono stati più ampi della media di tutta la storia.
- La chiusura della candela **non** è la previsione centrale della chiusura: quella è la mediana D4 (linea viola).
- Le fasce coprono meno del nominale: a 4 settimane quella al 50% ha contenuto la chiusura reale nel {n(100 * c4['copertura_50'], 0)}% dei casi, quella all'80% nel {n(100 * c4['copertura_80'], 0)}%, quella al 90% nel {n(100 * c4['copertura_90'], 0)}%.
- Candele, fasce (D4) e probabilità (D1) vengono da modelli diversi e possono non essere perfettamente coerenti tra loro.
- Il modello non anticipa i picchi e usa solo lo storico settimanale del VIX. I modelli con dati giornalieri e di mercato misurati nella Fase 3 non sono usati qui.
- Ogni esecuzione senza `--origine` viene scritta in `reports/registro_previsioni.csv`; `python -m src.forecast --verifica` confronta il registro con le settimane completate.
"""
    path.write_text(md)


def esegui_verifica(df: pd.DataFrame, reg_path: Path, out: Path) -> dict:
    """Pagella dal vivo: confronto tra le previsioni del registro e le settimane completate nel file."""
    if not reg_path.exists():
        print(f"Il registro {reg_path} non esiste ancora: esegui prima `python -m src.forecast`.")
        return {"dettaglio": pd.DataFrame(), "pagella": pd.DataFrame()}
    reg = pd.read_csv(reg_path, dtype={"data_origine": str, "data_esecuzione": str})
    det, pag = RG3.verifica(reg, df)
    print(f"\nPAGELLA DAL VIVO — registro: {reg['data_origine'].nunique()} origini, {len(reg)} settimane previste; settimane completate e confrontabili: {len(det)}")
    if det.empty:
        print("Nessuna settimana prevista è ancora stata completata nel file: non c'è nulla da verificare.")
        return {"dettaglio": det, "pagella": pag}
    det.to_csv(out / "registro_verifica_dettaglio.csv", index=False, float_format="%.4f")
    pag.to_csv(out / "registro_verifica_pagella.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(pag.round(3).to_string(index=False))
    if pag["meno_di_20_casi"].any():
        print("\nAttenzione: meno di 20 casi per almeno un orizzonte. Sono troppo pochi per trarre conclusioni.")
    if not det["registrata_prima_dell_esito"].all():
        print("Attenzione: alcune previsioni sono state registrate dopo la data della settimana prevista (colonna registrata_prima_dell_esito).")
    print("Il registro serve a misurare dal vivo, non a scegliere modelli.")
    print(f"\nDettaglio: {out / 'registro_verifica_dettaglio.csv'}\nPagella: {out / 'registro_verifica_pagella.csv'}")
    return {"dettaglio": det, "pagella": pag}


def main(argv: list[str] | None = None, out_dir: Path | None = None, oggi: str | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Previsione del VIX settimanale a 1-4 settimane: candele, chiusura centrale, fasce, probabilità.")
    ap.add_argument("--file", default=str(C.RAW_FILE), help="CSV nel formato del file grezzo (predefinito: data/raw/...)")
    ap.add_argument("--origine", default=None, help="data di origine AAAA-MM-GG (predefinito: ultima settimana completa)")
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
    st = statistiche_sviluppo()
    t = indice_origine(df, a.origine)
    prev, cand, L = calcola(df, t, st)
    d0 = prev["data_origine"]
    nota = ""
    if a.origine and d0 != str(pd.Timestamp(a.origine).date()):
        nota = f" Origine richiesta {T2B.data_it(a.origine)}: usata l'ultima chiusura settimanale disponibile a quella data."
    tab = testo.tabella(prev, cand, st)
    tab.to_csv(out / f"tabella_{d0}.csv", index=False, float_format="%.4f")
    frasi = testo.sintesi(L, prev, cand)
    (out / f"previsione_{d0}.json").write_text(json.dumps({"file": info, "lettura": L, "sintesi": frasi, "previsione": prev, "candele": cand}, indent=2, ensure_ascii=False))
    storia, vista = df.iloc[: t + 1], vista_grafico(prev, cand)
    g1, g2 = out / f"g1_{d0}.png", out / f"g2_{d0}.png"
    grafici.g1(storia, vista, L, g1, mediana=True, recente=True)
    grafici.g2(storia, vista, g2, mediana=True, recente=True)
    corrente = a.origine is None and out_dir is None
    md = (C.REPORTS_DIR / "previsione_corrente.md") if corrente else (out / f"previsione_{d0}.md")
    scrivi_md(md, prev, cand, L, tab, st, info, g1, C.FIG_DIR / "fase3" / "g3_verifica_passato.png", nota)
    msg_reg = "Registro: non aggiornato (con --origine la previsione non è una previsione dal vivo)."
    registrata = False
    if a.origine is None:
        registrata, msg_reg = RG.aggiungi(reg_path, RG3.righe(prev, cand, oggi or datetime.now().strftime("%Y-%m-%d")))

    print(f"\nPREVISIONE DEL VIX — origine {T2B.data_it(d0)} (chiusura {T2B.n(prev['chiusura_origine'])}); storia: {L['settimane_di_storia']} settimane")
    print("\n" + "\n".join("  " + f for f in frasi) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        vis = tab[["settimana", "data", "apertura", "massimo", "minimo", "chiusura", "etichetta_chiusura", "p_chiusura_sopra_origine", "mediana_d4", "fascia_50_inf", "fascia_50_sup",
                   "fascia_80_inf", "fascia_80_sup", "fascia_90_inf", "fascia_90_sup", "p_vix_sopra_20", "p_vix_sopra_25", "p_vix_sopra_30"]].round(2)
        print(vis.to_string(index=False))
        print(f"\nCandele: costruzione {st['candele']['nome']}. Etichetta: chiusura rispetto all'origine (neutra tra 40% e 60%). Misure storiche (walk-forward di sviluppo):")
        print(tab[["settimana", "copertura_storica_50", "copertura_storica_80", "copertura_storica_90", "mae_storico_mediana_d4", "mae_storico_apertura", "mae_storico_massimo",
                   "mae_storico_minimo", "mae_storico_chiusura", "etichetta_quota_neutra_storica", "etichetta_accuratezza_storica_non_neutre",
                   "n_previsioni_walk_forward"]].round(3).to_string(index=False))
    print(f"\n{msg_reg}")
    print(f"Scheda: {md}\nTabella: {out / f'tabella_{d0}.csv'}\nLettura e previsione (JSON): {out / f'previsione_{d0}.json'}\nGrafici: {g1}, {g2}")
    return {"previsione": prev, "candele": cand, "lettura": L, "tabella": tab, "md": md, "info": info, "registrata": registrata}


if __name__ == "__main__":
    main(sys.argv[1:])
