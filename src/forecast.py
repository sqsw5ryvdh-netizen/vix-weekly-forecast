"""Programma di previsione del VIX settimanale.

    python -m src.forecast [--file percorso_csv] [--origine AAAA-MM-GG]

Legge l'intero storico (stesso formato del file grezzo), esclude la settimana incompleta, parte dall'ultima chiusura
settimanale completa (o dall'origine indicata) e stima i modelli sulla storia fino a quella data.

E' L'UNICO comando autorizzato a leggere i dati successivi al confine di sviluppo, e solo come storia da cui
prevedere: non calcola ne' mostra errori o coperture su quel periodo. Errori e coperture «storici» riportati accanto
alla previsione sono quelli misurati nel walk-forward di sviluppo da `python -m src.run_phase2b` e vengono solo letti.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .fase2b import grafici, testo
from .fase2b import lettura as LE
from .fase2b import modello as MO

STAT_DIR = C.REPORTS_DIR / "fase2b"
OUT_DIR = C.REPORTS_DIR / "previsione"
COLONNE_FILE = ["week_start", "last_trading_day", "open", "high", "low", "close", "trading_days", "note"]


def leggi_storico(path: Path) -> tuple[pd.DataFrame, dict]:
    """Legge il file nel formato grezzo ed esclude le settimane incomplete (colonna `note` valorizzata)."""
    raw = pd.read_csv(path)
    manca = [c for c in COLONNE_FILE if c not in raw.columns]
    if manca:
        raise SystemExit(f"Il file {path} non ha il formato atteso: mancano le colonne {manca}")
    incomplete = raw["note"].notna() & (raw["note"].astype(str).str.strip() != "")
    df = raw.loc[~incomplete].copy()
    df["date"] = pd.to_datetime(df["last_trading_day"])
    df = df.sort_values("date").reset_index(drop=True)[["date", "open", "high", "low", "close"]]
    if df["date"].duplicated().any() or (df[["high", "low", "close"]] <= 0).any().any() or df[["high", "low", "close"]].isna().any().any():
        raise SystemExit(f"Il file {path} contiene date duplicate o valori mancanti / non positivi")
    passi = pd.to_datetime(raw.loc[~incomplete, "week_start"]).sort_values().diff().dt.days.dropna()
    info = {"file": str(path), "righe": int(len(raw)), "settimane_complete": int(len(df)), "settimane_incomplete_escluse": int(incomplete.sum()),
            "prima_data": str(df["date"].iloc[0].date()), "ultima_data_completa": str(df["date"].iloc[-1].date()),
            "settimane_mancanti_nel_file": int(((passi[passi > 7] / 7) - 1).sum())}
    return df, info


def indice_origine(df: pd.DataFrame, origine: str | None) -> int:
    if origine is None:
        return len(df) - 1
    d = pd.Timestamp(origine)
    ok = np.flatnonzero(df["date"].to_numpy() <= np.datetime64(d))
    if len(ok) == 0:
        raise SystemExit(f"Nessuna settimana completa con data <= {origine}")
    return int(ok[-1])


def statistiche_sviluppo() -> dict:
    """Errori e coperture misurati nel walk-forward di sviluppo (prodotti da run_phase2b). Qui vengono solo letti."""
    try:
        ch = pd.read_csv(STAT_DIR / "chiusura_metriche.csv").set_index("h")
        hl = pd.read_csv(STAT_DIR / "massimo_minimo_metriche.csv").set_index(["obiettivo", "h"])
        info = json.loads((STAT_DIR / "info.json").read_text())
    except FileNotFoundError:
        raise SystemExit("Mancano le statistiche del walk-forward di sviluppo: eseguire prima `python -m src.run_phase2b`.")
    return {"chiusura": {h: ch.loc[h].to_dict() for h in MO.ORIZZONTI}, "massimo": {h: hl.loc[("massimo", h)].to_dict() for h in MO.ORIZZONTI},
            "minimo": {h: hl.loc[("minimo", h)].to_dict() for h in MO.ORIZZONTI}, "info": info}


def calcola(df: pd.DataFrame, t: int) -> tuple[dict, dict]:
    """Previsione e lettura all'origine t: usano solo df.iloc[: t + 1]."""
    return MO.prevedi(df, t), LE.lettura(df, t)


def scrivi_md(path: Path, prev: dict, L: dict, tab: pd.DataFrame, st: dict, info: dict, g1: Path, g3: Path, nota_origine: str) -> None:
    rel = lambda p: os.path.relpath(p, path.parent)
    n = testo.n
    corr = [f"+{s['h']} ({'massimo' if s['massimo_corretto'] else ''}{' e ' if s['massimo_corretto'] and s['minimo_corretto'] else ''}{'minimo' if s['minimo_corretto'] else ''})"
            for s in prev["settimane"] if s["massimo_corretto"] or s["minimo_corretto"]]
    g = st["info"]["g3"]
    c4 = st["chiusura"][4]
    md = f"""# Previsione del VIX settimanale — origine {testo.data_it(prev['data_origine'])}

*Generato da `python -m src.forecast`. Storia usata: {L['settimane_di_storia']} settimane complete dal {testo.data_it(L['prima_data'])} al {testo.data_it(prev['data_origine'])}
(file `{Path(info['file']).name}`; settimane incomplete escluse: {info['settimane_incomplete_escluse']}).{nota_origine} Tutto ciò che segue usa solo dati fino all'origine.
Progetto universitario: non è un consiglio di investimento.*

## In sintesi

{chr(10).join('- ' + f for f in testo.sintesi(L, prev))}

## 1. Lettura della situazione

{testo.lettura_md(L)}

## 2. Previsione per le prossime 4 settimane

Valori in punti VIX. Chiusura = mediana del modello D4; fasce = quantili di D4; probabilità = modello D1; massimo e minimo = regressione dedicata.
Accanto a ogni fascia, tra parentesi, la **copertura storica**: la frequenza con cui la chiusura reale è caduta dentro la fascia nel walk-forward di sviluppo
({int(c4['n'])} previsioni a h = 4, obiettivi dal {testo.data_it(c4['prima_data_obiettivo'])} al {testo.data_it(c4['ultima_data_obiettivo'])}). Le fasce non sono ricalibrate: coprono meno del valore nominale.
Ultima colonna: errore assoluto medio storico (punti VIX) di chiusura, massimo e minimo, misurato nello stesso walk-forward.

{testo.tabella_md(tab)}

{('Candele corrette dai vincoli (massimo ≥ apertura e chiusura; minimo ≤ apertura e chiusura): settimane ' + ', '.join(corr) + '.') if corr else 'Nessuna candela ha richiesto la correzione dei vincoli su massimo e minimo.'}
Le date sono indicative (origine + 7 giorni per settimana): la chiusura settimanale è quella dell'ultima seduta della settimana.

## 3. Grafico

![Grafico principale]({rel(g1)})

Il grafico in alto mostra gli ultimi 2 anni. I percorsi dei casi simili sono nel pannello in basso (ingrandimento sulle ultime 16 settimane): nel grafico
a 2 anni le quattro settimane di previsione occupano troppo poco spazio per ospitarli senza coprire candele e fasce.

### Come leggere il grafico

{testo.COME_LEGGERE}

## 4. Verifica visiva sul passato

![Verifica sul passato]({rel(g3)})

Dodici origini scelte con regola fissa ({g['regola']}): tutte le origini e tutti gli esiti stanno nel campione di sviluppo. {g['riga']}
Dodici casi sono pochi: servono a vedere come si comporta la previsione, non a misurarla. Le misure sono nel report `fase2b_report.md`.

## 5. Avvertenze

- Le fasce coprono meno del nominale: a 4 settimane la fascia al 50% ha contenuto la chiusura reale nel {n(100 * c4['copertura_50'], 0)}% dei casi, quella all'80% nel {n(100 * c4['copertura_80'], 0)}%, quella al 90% nel {n(100 * c4['copertura_90'], 0)}%.
- Fasce (modello D4) e probabilità (modello D1) vengono da due modelli diversi e possono non essere perfettamente coerenti tra loro.
- Il modello non anticipa i picchi: nelle fasi precedenti nessuna variabile costruita sul solo VIX li ha previsti. Le probabilità di superare 25 e 30 sono frequenze condizionate al livello attuale, non segnali.
- La previsione usa solo lo storico del VIX. Nessun dato esterno.
"""
    path.write_text(md)


def main(argv: list[str] | None = None, out_dir: Path | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Previsione del VIX settimanale a 1-4 settimane (chiusura, massimo, minimo, fasce, probabilità).")
    ap.add_argument("--file", default=str(C.RAW_FILE), help="CSV nel formato del file grezzo (predefinito: data/raw/...)")
    ap.add_argument("--origine", default=None, help="data di origine AAAA-MM-GG (predefinito: ultima settimana completa)")
    a = ap.parse_args(argv)
    out = out_dir or OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    st = statistiche_sviluppo()
    df, info = leggi_storico(Path(a.file))
    t = indice_origine(df, a.origine)
    prev, L = calcola(df, t)
    d0 = prev["data_origine"]
    nota = ""
    if a.origine and d0 != str(pd.Timestamp(a.origine).date()):
        nota = f" Origine richiesta {testo.data_it(a.origine)}: usata l'ultima chiusura settimanale disponibile a quella data."
    tab = testo.tabella(prev, st)
    tab.to_csv(out / f"tabella_{d0}.csv", index=False, float_format="%.4f")
    (out / f"previsione_{d0}.json").write_text(json.dumps({"file": info, "lettura": L, "sintesi": testo.sintesi(L, prev), "previsione": prev}, indent=2, ensure_ascii=False))
    storia = df.iloc[: t + 1]
    g1, g2 = out / f"g1_{d0}.png", out / f"g2_{d0}.png"
    grafici.g1(storia, prev, L, g1)
    grafici.g2(storia, prev, g2)
    corrente = a.origine is None and out_dir is None
    md = (C.REPORTS_DIR / "previsione_corrente.md") if corrente else (out / f"previsione_{d0}.md")
    scrivi_md(md, prev, L, tab, st, info, g1, C.FIG_DIR / "fase2b" / "g3_verifica_passato.png", nota)

    print(f"\nPREVISIONE DEL VIX — origine {testo.data_it(d0)} (chiusura {testo.n(prev['chiusura_origine'])}); storia: {L['settimane_di_storia']} settimane")
    print("\n" + "\n".join("  " + f for f in testo.sintesi(L, prev)) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        vis = tab[["settimana", "data", "apertura", "massimo", "minimo", "chiusura", "fascia_50_inf", "fascia_50_sup", "fascia_80_inf", "fascia_80_sup",
                   "fascia_90_inf", "fascia_90_sup", "p_vix_sopra_20", "p_vix_sopra_25", "p_vix_sopra_30"]].round(2)
        print(vis.to_string(index=False))
        print("\nCopertura storica delle fasce ed errore assoluto medio storico (walk-forward di sviluppo):")
        print(tab[["settimana", "copertura_storica_50", "copertura_storica_80", "copertura_storica_90", "mae_storico_chiusura", "mae_storico_massimo",
                   "mae_storico_minimo", "n_previsioni_walk_forward"]].round(3).to_string(index=False))
    print(f"\nScheda: {md}\nTabella: {out / f'tabella_{d0}.csv'}\nLettura e previsione (JSON): {out / f'previsione_{d0}.json'}\nGrafici: {g1}, {g2}")
    return {"previsione": prev, "lettura": L, "tabella": tab, "md": md, "info": info}


if __name__ == "__main__":
    main(sys.argv[1:])
