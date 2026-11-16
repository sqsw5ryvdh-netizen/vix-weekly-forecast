"""Fase 4: dati giornalieri di VIX e VIX3M per la versione F2 del programma di previsione.

USATO SOLO DA `src.forecast` (mai da `run_phase4`, che legge i soli dati di sviluppo sotto la guardia). Come per il
VIX settimanale, il programma puo' leggere i valori successivi all'8 novembre 2024 soltanto come storia da cui
prevedere. `aggiorna` scarica i due file ufficiali Cboe nella cartella data/external/forecast (non pubblicata);
`x1_per_previsione` li tronca all'ultima seduta della settimana di origine e costruisce X1 = VIX3M / VIX - 1 con la
regola di allineamento congelata (fase4.versioni.x1_settimanale).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..fase3 import variabili as V
from . import versioni as VS

CARTELLA = C.ROOT / "data" / "external" / "forecast"
INDIRIZZI = {"vix": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
             "vix3m": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv"}
FILE = {"vix": "VIX_History.csv", "vix3m": "VIX3M_History.csv"}
TOLLERANZA_VIX = 0.011           # differenza massima tra chiusura giornaliera e settimanale dell'ultima seduta


def aggiorna(cartella: Path = CARTELLA, scarica=None) -> dict:
    """Scarica VIX e VIX3M giornalieri (storia completa) e li salva. Restituisce byte e impronta di ogni file."""
    if scarica is None:
        from ..fetch_external import scarica
    cartella.mkdir(parents=True, exist_ok=True)
    out = {}
    for nome, url in INDIRIZZI.items():
        raw = scarica(url)
        (cartella / FILE[nome]).write_bytes(raw)
        out[nome] = {"indirizzo": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return out


def _leggi(path: Path) -> pd.Series:
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    d["date"] = pd.to_datetime(d["date"], format="%m/%d/%Y")
    return d.set_index("date").sort_index()["close"].astype(float)


def x1_per_previsione(df: pd.DataFrame, t: int, cartella: Path | None) -> tuple[np.ndarray | None, dict]:
    """X1 settimanale per le settimane 0..t (solo dati giornalieri fino all'ultima seduta della settimana t).

    Restituisce (z, informazioni). z e' None se i file mancano. `informazioni` contiene i valori usati, le impronte dei
    file, gli avvisi e, se X1 non e' utilizzabile all'origine, il motivo.
    """
    if cartella is None or not all((cartella / f).exists() for f in FILE.values()):
        return None, {"disponibile": False, "motivo": "i file giornalieri di VIX e VIX3M non sono stati scaricati (usare --aggiorna)", "avvisi": []}
    origine = pd.Timestamp(df["date"].iloc[t])
    serie = {k: _leggi(cartella / f) for k, f in FILE.items()}
    ultima = {k: s.index.max() for k, s in serie.items()}
    g = pd.concat([serie["vix"].rename("vix"), serie["vix3m"].rename("vix3m")], axis=1).sort_index()
    g = g[g.index <= origine]                                   # nulla di successivo all'ultima seduta della settimana di origine
    date = np.asarray(df["date"].iloc[: t + 1], dtype="datetime64[ns]")
    z = VS.x1_settimanale(date, g)
    j = g[["vix3m", "vix"]].dropna()
    k, ok = V._ultimo_nella_settimana(j["vix"], date)
    oss = np.full(len(date), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    if len(j):
        oss[ok] = j.index.to_numpy()[k[ok]]
    usati = pd.DataFrame({"data_settimanale": pd.to_datetime(date), "data_osservazione": pd.to_datetime(oss),
                          "vix3m": V._campiona(j["vix3m"].to_numpy(), k, ok), "vix_giornaliero": V._campiona(j["vix"].to_numpy(), k, ok), "x1": z})
    usati = usati[usati["x1"].notna()].reset_index(drop=True)
    info = {"disponibile": bool(np.isfinite(z[t])), "avvisi": [], "ultima_data_nei_file": {k: str(v.date()) for k, v in ultima.items()},
            "sha256": {k: hashlib.sha256((cartella / f).read_bytes()).hexdigest() for k, f in FILE.items()}, "valori_usati": usati}
    if not info["disponibile"]:
        u = serie["vix3m"][serie["vix3m"].index <= origine]
        info["motivo"] = (f"VIX3M manca nella settimana di origine ({origine.date()}); ultima data di VIX3M disponibile entro l'origine: "
                          f"{u.index.max().date() if len(u) else 'nessuna'}")
        return z, info
    r = usati.iloc[-1]
    if pd.Timestamp(r["data_osservazione"]) < origine:
        info["avvisi"].append(f"ATTENZIONE: VIX3M e VIX giornalieri non hanno entrambi un valore il {origine.date()}: usata la seduta del {pd.Timestamp(r['data_osservazione']).date()} (stessa settimana).")
    elif abs(float(r["vix_giornaliero"]) - float(df["close"].iloc[t])) > TOLLERANZA_VIX:
        info["avvisi"].append(f"ATTENZIONE: la chiusura giornaliera del VIX del {origine.date()} ({r['vix_giornaliero']:.2f}) è diversa dalla chiusura settimanale del file ({float(df['close'].iloc[t]):.2f}).")
    info["x1_origine"] = {"data_osservazione": str(pd.Timestamp(r["data_osservazione"]).date()), "vix3m": float(r["vix3m"]), "vix": float(r["vix_giornaliero"]), "x1": float(r["x1"])}
    return z, info
