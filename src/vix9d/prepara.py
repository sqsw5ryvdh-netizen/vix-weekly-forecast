"""VIX9D: costruzione delle candele settimanali dai dati giornalieri ufficiali Cboe e audit.   python -m src.vix9d.prepara [--scarica]

Convenzioni del file settimanale del VIX: settimana da lunedi' a venerdi'; apertura = apertura ufficiale della prima
seduta, massimo e minimo = estremi delle sedute, chiusura = chiusura dell'ultima seduta; l'ultima settimana, se
incompleta, viene marcata con una nota ed esclusa. Stesso confine del VIX: sviluppo fino all'8 novembre 2024, le
settimane successive vanno nel file di holdout, bloccato dalla guardia.
Fino a meta' novembre 2013 il file Cboe riporta per ogni seduta un solo valore (apertura = massimo = minimo = chiusura):
quelle settimane hanno solo le chiusure. La colonna `sedute_solo_chiusura` le segnala; non vengono corrette.
L'audit (reports/vix9d/audit.json) e' calcolato SOLO sulla parte di sviluppo: nessuna statistica sui valori successivi al confine.
Questo e' l'unico modulo (con il programma di previsione) che legge i file completi; non gira sotto la guardia.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd

from ..fase2c import settimana as SE
from . import config as G

COLONNE = ["week_start", "last_trading_day", "open", "high", "low", "close", "trading_days", "sedute_solo_chiusura", "note"]


def leggi_cboe(path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    d["date"] = pd.to_datetime(d["date"], format="%m/%d/%Y")
    return d.set_index("date").sort_index()[["open", "high", "low", "close"]].astype(float)


def settimanali(d: pd.DataFrame) -> pd.DataFrame:
    """Candele settimanali dalle sedute giornaliere (indice = data; colonne open, high, low, close)."""
    piatta = (d["open"] == d["high"]) & (d["high"] == d["low"]) & (d["low"] == d["close"])
    lun = (d.index - pd.to_timedelta(d.index.day_of_week, unit="D")).normalize()
    g = d.assign(piatta=piatta, giorno=d.index).groupby(lun)
    w = pd.DataFrame({"last_trading_day": g["giorno"].max(), "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
                      "trading_days": g.size(), "sedute_solo_chiusura": g["piatta"].sum().astype(int)})
    w.insert(0, "week_start", w.index)
    w["note"] = ""
    u = w.iloc[-1]
    incompleta, _ = SE.ultima_riga_incompleta(u["week_start"], u["last_trading_day"], u["trading_days"], None)
    if incompleta:
        w.iloc[-1, w.columns.get_loc("note")] = f"Settimana in corso (dati fino al {pd.Timestamp(u['last_trading_day']).date()})"
    return w.reset_index(drop=True)[COLONNE]


def dividi(w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sviluppo, holdout): solo settimane complete; il confine e' sulla data dell'ultima seduta. L'holdout e' il periodo
    d'esame del VIX: le prime 104 settimane dopo il confine (quelle ancora successive stanno solo nel file completo)."""
    c = w[w["note"] == ""].copy()
    c.insert(0, "date", pd.to_datetime(c["last_trading_day"]))
    c = c.drop(columns=["last_trading_day", "note"])
    dev = c[c["date"] <= pd.Timestamp(G.CONFINE)].reset_index(drop=True)
    return dev, c[c["date"] > pd.Timestamp(G.CONFINE)].head(G.SETTIMANE_HOLDOUT).reset_index(drop=True)


def audit(d: pd.DataFrame, dev: pd.DataFrame, vix_dev: pd.DataFrame) -> dict:
    """Audit della sola parte di sviluppo: sedute giornaliere fino al confine e settimane di sviluppo."""
    d = d[d.index <= pd.Timestamp(G.CONFINE)]
    piatta = (d["open"] == d["high"]) & (d["high"] == d["low"]) & (d["low"] == d["close"])
    feriali = pd.bdate_range(d.index.min(), d.index.max())
    mancanti = feriali.difference(d.index)
    inco = d[(d["open"] > d["high"] + 1e-9) | (d["open"] < d["low"] - 1e-9) | (d["close"] > d["high"] + 1e-9) | (d["close"] < d["low"] - 1e-9)]
    var = np.log(d["close"]).diff()
    vere = dev[dev["sedute_solo_chiusura"] == 0]
    gap = (dev["open"] / dev["close"].shift(1))[(dev["sedute_solo_chiusura"] == 0) & (dev["sedute_solo_chiusura"].shift(1) == 0)]
    passi = dev["week_start"].diff().dt.days.dropna()
    m = dev.merge(vix_dev[["week_start", "date", "close"]].rename(columns={"date": "date_vix", "close": "vix"}), on="week_start", how="left")
    return {"fonte": "Cboe Global Markets, VIX9D_History.csv (dati giornalieri ufficiali, OHLC)", "confine_sviluppo": G.CONFINE,
            "giornaliero_sviluppo": {"prima_data": str(d.index.min().date()), "ultima_data": str(d.index.max().date()), "n_sedute": int(len(d)), "valori_mancanti": int(d.isna().sum().sum()),
                                     "valori_non_positivi": int((d <= 0).sum().sum()), "date_duplicate": int(d.index.duplicated().sum()),
                                     "giorni_feriali_senza_seduta": int(len(mancanti)), "sedute_ohlc_incoerenti": int(len(inco)),
                                     "sedute_con_solo_chiusura": int(piatta.sum()), "ultima_seduta_con_solo_chiusura": str(d.index[piatta].max().date()) if piatta.any() else None,
                                     "prima_seduta_con_ohlc": str(d.index[~piatta].min().date()),
                                     "variazione_log_giornaliera": {"min": float(var.min()), "max": float(var.max()), "n_oltre_30pct_in_valore_assoluto": int((var.abs() > np.log(1.3)).sum()),
                                                                    "date_oltre_50pct": [str(x.date()) for x in var.index[var.abs() > np.log(1.5)]]},
                                     "chiusura_min_max": [float(d["close"].min()), float(d["close"].max())]},
            "settimanale_sviluppo": {"n_settimane": int(len(dev)), "prima_data": str(dev["date"].iloc[0].date()), "ultima_data": str(dev["date"].iloc[-1].date()),
                                     "settimane_mancanti": int(((passi[passi > 7] / 7) - 1).sum()), "settimane_con_meno_di_5_sedute": int((dev["trading_days"] < 5).sum()),
                                     "settimane_con_solo_chiusure": int((dev["sedute_solo_chiusura"] > 0).sum()), "settimane_con_candela_vera": int(len(vere)),
                                     "prima_settimana_con_candela_vera": str(vere["date"].iloc[0].date()),
                                     "apertura_su_chiusura_precedente": {"n": int(len(gap)), "mediana": float(gap.median()), "media": float(gap.mean()), "p10": float(gap.quantile(0.10)),
                                                                         "p90": float(gap.quantile(0.90)), "quota_sopra_1": float((gap > 1).mean())},
                                     "settimane_senza_vix": int(m["vix"].isna().sum()), "settimane_con_ultima_seduta_diversa_dal_vix": int((m["date_vix"].notna() & (m["date_vix"] != m["date"])).sum())},
            "nota": "nessuna statistica è calcolata sui valori successivi al confine di sviluppo"}


def _piu_recente(w9: pd.DataFrame) -> bool:
    """True se i file completi non esistono o finiscono prima dei dati appena letti (confronto sulle sole date, nessuna statistica)."""
    if not (G.FILE_COMPLETO.exists() and G.FILE_VIX_COMPLETO.exists()):
        return True
    ultima = pd.read_csv(G.FILE_COMPLETO, usecols=["last_trading_day"])["last_trading_day"].max()
    return str(w9["last_trading_day"].max().date()) >= str(ultima)


def main(argv: list[str] | None = None) -> bool:
    """Costruisce i file di data/vix9d dai file Cboe. Restituisce False (senza errore) se gli originali non ci sono ma i file esistono gia'."""
    argv = sys.argv[1:] if argv is None else argv
    if "--scarica" in argv:
        from ..fetch_external import scarica
        for k, url in G.INDIRIZZI.items():
            G.ORIGINALI[k].parent.mkdir(parents=True, exist_ok=True)
            G.ORIGINALI[k].write_bytes(scarica(url))
    if not all(p.exists() for p in G.ORIGINALI.values()):
        if G.FILE_DEV.exists():
            print(f"[vix9d] originali Cboe assenti: uso i file già presenti in {G.DATI}")
            return False
        raise SystemExit("Mancano i file giornalieri Cboe (data/external/raw/cboe): eseguire `python -m src.vix9d.prepara --scarica`.")
    G.DATI.mkdir(parents=True, exist_ok=True)
    G.OUT.mkdir(parents=True, exist_ok=True)
    d9 = leggi_cboe(G.ORIGINALI["vix9d"])
    w9, wv = settimanali(d9), settimanali(leggi_cboe(G.ORIGINALI["vix"]))
    fmt = dict(index=False, float_format="%.4f", date_format="%Y-%m-%d")
    if _piu_recente(w9):                                    # non sovrascrive file completi piu' aggiornati (li aggiorna `forecast --aggiorna`)
        w9.to_csv(G.FILE_COMPLETO, **fmt)
        wv[wv["week_start"] >= w9["week_start"].iloc[0] - pd.Timedelta(days=7 * 60)].to_csv(G.FILE_VIX_COMPLETO, **fmt)
    dev, hold = dividi(w9)
    dev.to_csv(G.FILE_DEV, **fmt)
    hold.to_csv(G.FILE_HOLDOUT, **fmt)
    from .. import config as C
    vix_dev = pd.read_csv(C.DEV_FILE, parse_dates=["date", "week_start"])
    a = audit(d9, dev, vix_dev) | {"sha256_originale": hashlib.sha256(G.ORIGINALI["vix9d"].read_bytes()).hexdigest(), "n_settimane_holdout": int(len(hold)),
                                   "file": {"sviluppo": str(G.FILE_DEV.relative_to(C.ROOT)), "holdout_bloccato": str(G.FILE_HOLDOUT.relative_to(C.ROOT)),
                                            "completo_solo_per_il_programma": str(G.FILE_COMPLETO.relative_to(C.ROOT))}}
    (G.OUT / "audit.json").write_text(json.dumps(a, indent=2, ensure_ascii=False))
    print(f"[vix9d] candele settimanali scritte in {G.DATI}: {len(dev)} settimane di sviluppo, {len(hold)} di holdout (bloccate)")
    return True


if __name__ == "__main__":
    main()
