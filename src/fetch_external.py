"""Fase 2, Parte C: raccolta dei dati esterni.  python -m src.fetch_external

Solo raccolta e allineamento: nessun modello, nessuna analisi, nessuna correlazione con il VIX.
- I file originali sono salvati in data/external/raw senza modifiche.
- Le serie sono divise sullo stesso confine dev/holdout del VIX: la parte successiva va in
  data/external/holdout, protetta dalla stessa guardia che protegge l'holdout del VIX.
- Il programma non stampa ne' scrive nei report alcun valore o statistica successiva al confine:
  dell'intera serie riporta solo date estreme e numero di righe.
Non e' deterministico (dipende dal giorno di scarico): per questo e' un comando separato da run_phase2.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C

EXT = C.ROOT / "data" / "external"
RAW, DEV, HOLD = EXT / "raw", EXT / "dev", EXT / "holdout"
OUT = C.REPORTS_DIR / "fase2"
UA = "Mozilla/5.0 (compatible; research-data-fetch)"
STALE_DAYS = 14          # un valore piu' vecchio di 14 giorni rispetto alla data settimanale e' considerato mancante
VX_RESCALE_DATE = pd.Timestamp("2007-03-26")   # prima di questa data i future VX erano quotati a 10 volte l'indice

CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
INDICES = {  # nome -> (file CBOE, descrizione)
    "vix": ("VIX", "VIX, indice (OHLC)"), "vix9d": ("VIX9D", "VIX a 9 giorni (OHLC)"), "vix3m": ("VIX3M", "VIX a 3 mesi (OHLC)"),
    "vix6m": ("VIX6M", "VIX a 6 mesi (OHLC)"), "vvix": ("VVIX", "VVIX, volatilità del VIX"), "spx": ("SPX", "S&P 500, chiusura"),
}
TSY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{y}/all"
           "?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv")
NFCI_URL = "https://www.chicagofed.org/-/media/publications/nfci/nfci-data-series-csv.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRED_SERIES = {"oas_ig": ("BAMLC0A0CM", "Differenziale di credito investment grade (ICE BofA US Corporate OAS)"),
               "oas_hy": ("BAMLH0A0HYM2", "Differenziale di credito high yield (ICE BofA US High Yield OAS)")}
VX_LIST = "https://www.cboe.com/us/futures/market_statistics/historical_data/product/list/VX/"
VX_CDN = "https://cdn.cboe.com/"
VX_ARCHIVE = "https://cdn.cboe.com/resources/futures/archive/volume-and-price/CFE_{}{:02d}_VX.csv"
MONTH_CODES = "FGHJKMNQUVXZ"

# Ritardo di disponibilita' usato nell'allineamento (giorni di calendario tra data dell'osservazione e momento in cui e' nota)
LAGS = {"vix": 0, "vix9d": 0, "vix3m": 0, "vix6m": 0, "vvix": 0, "spx": 0, "vx1": 0, "vx2": 0, "dgs2": 0, "dgs10": 0,
        "nfci": 5, "oas_ig": 1, "oas_hy": 1}
LAG_NOTES = {
    "vix": "nessuno: valore di chiusura noto il giorno stesso", "vix9d": "nessuno", "vix3m": "nessuno", "vix6m": "nessuno",
    "vvix": "nessuno", "spx": "nessuno: chiusura nota il giorno stesso",
    "vx1": "nessuno: prezzo di regolamento pubblicato a fine seduta", "vx2": "nessuno: prezzo di regolamento pubblicato a fine seduta",
    "dgs2": "nessuno: curva pubblicata dal Tesoro la sera dello stesso giorno", "dgs10": "nessuno: curva pubblicata dal Tesoro la sera dello stesso giorno",
    "nfci": "5 giorni: il valore della settimana che termina il venerdì è pubblicato il mercoledì successivo; la serie viene inoltre rivista a ogni pubblicazione",
    "oas_ig": "1 giorno: pubblicato il giorno lavorativo successivo", "oas_hy": "1 giorno: pubblicato il giorno lavorativo successivo",
}


# ------------------------------------------------------------------ funzioni pure (testate)

def allinea_settimanale(serie: pd.Series, weekly_dates, lag_days: int = 0, stale_days: int = STALE_DAYS) -> pd.DataFrame:
    """Per ogni data settimanale d: ultimo valore con data_osservazione + lag_days <= d.

    Restituisce valore e data dell'osservazione usata. Usa solo osservazioni disponibili entro d;
    se l'ultima disponibile e' piu' vecchia di `stale_days` giorni il valore e' mancante.
    """
    s = serie.dropna().sort_index()
    wd = pd.DatetimeIndex(pd.to_datetime(weekly_dates))
    avail = s.index + pd.Timedelta(days=lag_days)
    pos = np.searchsorted(avail.values, wd.values, side="right") - 1
    val = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan) if len(s) else np.full(len(wd), np.nan)
    obs = np.where(pos >= 0, s.index.values[np.clip(pos, 0, None)], np.datetime64("NaT", "ns")) if len(s) else np.full(len(wd), np.datetime64("NaT", "ns"))
    obs = pd.DatetimeIndex(obs)
    stale = (wd - obs).days > stale_days
    val = np.where(stale | obs.isna(), np.nan, val)
    return pd.DataFrame({"valore": val, "data_osservazione": obs.where(~(stale | obs.isna()))}, index=wd)


def continui_futures(contratti: list[tuple[pd.Timestamp, pd.Series]]) -> pd.DataFrame:
    """Serie continue di prima e seconda scadenza dai singoli contratti mensili.

    `contratti` = lista di (data di scadenza, serie dei prezzi di regolamento per data di contrattazione).
    Per ogni giorno si considerano i contratti con scadenza successiva al giorno stesso, in ordine di scadenza.
    """
    rows = []
    for exp, s in contratti:
        s = s[(s > 0) & (s.index < exp)]
        rows.append(pd.DataFrame({"data": s.index, "scadenza": exp, "prezzo": s.to_numpy()}))
    allc = pd.concat(rows, ignore_index=True).sort_values(["data", "scadenza"])
    allc["ordine"] = allc.groupby("data").cumcount()
    w = allc[allc["ordine"] < 2].pivot(index="data", columns="ordine", values=["prezzo", "scadenza"])
    out = pd.DataFrame({"vx1": w[("prezzo", 0)].astype(float), "vx2": w[("prezzo", 1)].astype(float),
                        "vx1_scadenza": w[("scadenza", 0)], "vx2_scadenza": w[("scadenza", 1)]})
    return out


def settimanale_da_giornaliero(daily: pd.DataFrame) -> pd.DataFrame:
    """Barre settimanali (lunedì-domenica) da barre giornaliere OHLC indicizzate per data."""
    d = daily.sort_index()
    ws = (d.index - pd.to_timedelta(d.index.day_of_week, unit="D")).normalize()
    g = d.groupby(ws)
    return pd.DataFrame({"last_trading_day": g.apply(lambda t: t.index.max()), "open": g["open"].first(), "high": g["high"].max(),
                         "low": g["low"].min(), "close": g["close"].last(), "trading_days": g.size()})


def dividi(df: pd.DataFrame, confine: pd.Timestamp, date_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide sul confine dev/holdout (dev: date <= confine)."""
    d = pd.to_datetime(df[date_col]) if date_col else pd.to_datetime(df.index)
    m = np.asarray(d <= confine)
    return df[m], df[~m]


# ------------------------------------------------------------------ scarico

def scarica(url: str, tries: int = 3, timeout: int = 60) -> bytes:
    """Scarica con `curl` di sistema (usa i certificati del sistema operativo; la verifica TLS resta attiva).

    Se curl non e' installato si ripiega su urllib. Un errore HTTP o di rete dopo `tries` tentativi solleva RuntimeError.
    """
    last = "nessun tentativo"
    for k in range(tries):
        if shutil.which("curl"):
            r = subprocess.run(["curl", "-sSL", "--fail", "-m", str(timeout), "-A", UA, "-H", "Accept: text/csv,application/json,*/*", url],
                               capture_output=True)
            if r.returncode == 0 and r.stdout:
                return r.stdout
            last = f"curl codice {r.returncode}: {r.stderr.decode(errors='replace').strip()[:200]}"
        else:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/csv,application/json,*/*"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read()
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
        time.sleep(1.0 * (k + 1))
    raise RuntimeError(last)


RIUSA = {"attivo": False}       # con --riusa-originali i file gia' presenti in data/external/raw non vengono riscaricati


def ottieni(name: str, url: str, tries: int = 3, timeout: int = 60) -> tuple[bytes, dict]:
    """Scarica e salva l'originale; con --riusa-originali usa il file gia' salvato, se esiste."""
    path = RAW / name
    if RIUSA["attivo"] and path.exists() and path.stat().st_size > 0:
        raw = path.read_bytes()
        return raw, {"file": str(path.relative_to(C.ROOT)), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "riusato": True}
    raw = scarica(url, tries, timeout)
    return raw, salva_raw(name, raw)


def leggi_vx(content: bytes) -> pd.Series:
    """Prezzi di regolamento di un contratto VX per data. Lettura riga per riga: alcuni file hanno campi in piu' o righe di testo."""
    lines = content.decode("utf-8", errors="replace").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Trade Date"))
    head = [c.strip() for c in lines[start].split(",")]
    k = head.index("Settle")
    d, v = [], []
    for l in lines[start + 1:]:
        f = l.split(",")
        if len(f) <= k:
            continue
        try:
            dt = pd.to_datetime(f[0].strip())
            val = float(f[k])
        except (ValueError, TypeError):
            continue
        d.append(dt)
        v.append(val)
    return pd.Series(v, index=pd.DatetimeIndex(d), dtype=float).sort_index()


def salva_raw(name: str, content: bytes) -> dict:
    path = RAW / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"file": str(path.relative_to(C.ROOT)), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}


def leggi_cboe(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df.set_index("date").sort_index()


def fetch_all(log=print) -> tuple[dict[str, pd.Series], dict[str, dict], pd.DataFrame | None]:
    """Scarica tutto. Restituisce serie giornaliere, metadati per serie, VIX giornaliero OHLC."""
    oggi = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    series, meta, vix_ohlc = {}, {}, None

    def ok(name, s, fonte, url, freq, files, extra=None):
        series[name] = s.dropna().astype(float)
        meta[name] = {"stato": "scaricata", "fonte": fonte, "indirizzo": url, "data_scarico": oggi, "frequenza": freq,
                      "file_originali": files, "ritardo_pubblicazione": LAG_NOTES[name], "ritardo_giorni_usato": LAGS[name]} | (extra or {})
        log(f"  [ok] {name}: {len(series[name])} osservazioni, {series[name].index.min().date()} -> {series[name].index.max().date()}")

    def ko(name, descr, fonte, url, err, cosa_serve):
        meta[name] = {"stato": "non scaricabile", "descrizione": descr, "fonte": fonte, "indirizzo": url, "data_scarico": oggi,
                      "motivo": err, "file_da_procurare": cosa_serve}
        log(f"  [NO] {name}: {err}")

    for name, (code, descr) in INDICES.items():
        url = CBOE.format(code)
        try:
            raw, f = ottieni(f"cboe/{code}_History.csv", url)
            df = leggi_cboe(raw)
            col = "close" if "close" in df.columns else df.columns[0]
            ok(name, df[col], "Cboe Global Markets", url, "giornaliera", [f], {"descrizione": descr, "colonne_originali": list(df.columns)})
            if name == "vix":
                vix_ohlc = df[["open", "high", "low", "close"]]
        except Exception as e:
            ko(name, descr, "Cboe Global Markets", url, str(e), f"{code}_History.csv dal sito Cboe (Historical Data)")

    # Rendimenti dei titoli di Stato USA: curva giornaliera del Tesoro, un file per anno
    try:
        parts, files = [], []
        for y in range(1999, datetime.now(timezone.utc).year + 1):
            raw, f = ottieni(f"treasury/daily_treasury_yield_curve_{y}.csv", TSY_URL.format(y=y))
            files.append(f)
            t = pd.read_csv(io.BytesIO(raw))
            t["Date"] = pd.to_datetime(t["Date"], format="%m/%d/%Y")
            parts.append(t.set_index("Date")[["2 Yr", "10 Yr"]])
            time.sleep(0.2)
        tsy = pd.concat(parts).sort_index()
        for name, col in (("dgs2", "2 Yr"), ("dgs10", "10 Yr")):
            ok(name, pd.to_numeric(tsy[col], errors="coerce"), "U.S. Department of the Treasury, Daily Treasury Par Yield Curve Rates",
               TSY_URL.format(y="ANNO"), "giornaliera", [{"file": "data/external/raw/treasury/ (un file per anno)", "n_file": len(files),
                                                          "sha256_elenco": hashlib.sha256("".join(f["sha256"] for f in files).encode()).hexdigest()}],
               {"descrizione": f"Rendimento dei titoli di Stato USA, scadenza costante {col.replace(' Yr', ' anni')} (%)"})
    except Exception as e:
        for name in ("dgs2", "dgs10"):
            ko(name, "Rendimento dei titoli di Stato USA", "U.S. Department of the Treasury", TSY_URL.format(y="ANNO"), str(e),
               "Daily Treasury Par Yield Curve Rates (CSV per anno) da home.treasury.gov, oppure DGS2.csv e DGS10.csv da FRED")

    try:
        raw, f = ottieni("chicagofed/nfci-data-series-csv.csv", NFCI_URL)
        t = pd.read_csv(io.BytesIO(raw))
        t["d"] = pd.to_datetime(t["Friday_of_Week"], format="%m/%d/%Y")
        ok("nfci", t.set_index("d")["NFCI"], "Federal Reserve Bank of Chicago", NFCI_URL, "settimanale (venerdì)", [f],
           {"descrizione": "National Financial Conditions Index (NFCI)", "colonne_originali": list(t.columns[:-1]),
            "avvertenza": "serie rivista a ogni pubblicazione: il file scaricato contiene i valori rivisti a oggi, non quelli noti in tempo reale"})
    except Exception as e:
        ko("nfci", "National Financial Conditions Index", "Federal Reserve Bank of Chicago", NFCI_URL, str(e), "nfci-data-series-csv.csv dal sito della Chicago Fed")

    for name, (sid, descr) in FRED_SERIES.items():
        url = FRED_URL.format(sid)
        try:
            raw, f = ottieni(f"fred/{sid}.csv", url, tries=2, timeout=45)
            t = pd.read_csv(io.BytesIO(raw))
            t.columns = ["d", "v"]
            t["d"] = pd.to_datetime(t["d"])
            v = pd.to_numeric(t.set_index("d")["v"], errors="coerce")
            extra = {"descrizione": descr}
            if v.dropna().index.min() > pd.Timestamp("2010-01-01"):
                extra |= {"storia_insufficiente": True,
                          "avvertenza": "FRED distribuisce solo gli ultimi anni di questa serie (limite di licenza ICE): la storia scaricata copre una parte minima del campione",
                          "file_da_procurare": f"storia completa di {sid} (dal 1996) da ICE Data Indices o da un fornitore con licenza, in CSV con colonne data e valore, "
                                               f"da mettere in data/external/raw/fred/{sid}.csv"}
            ok(name, v, "ICE BofA via FRED (Federal Reserve Bank of St. Louis)", url, "giornaliera", [f], extra)
        except Exception as e:
            ko(name, descr, "ICE BofA via FRED (Federal Reserve Bank of St. Louis)", url, str(e),
               f"{sid}.csv scaricato a mano da https://fred.stlouisfed.org/series/{sid} (Download > CSV), da mettere in data/external/raw/fred/")

    # Future sul VIX: un file per contratto mensile
    try:
        contratti, files = {}, []          # chiave (anno, mese di scadenza) -> [scadenza, serie]
        lst = json.loads(ottieni("cboe_futures/VX_product_list.json", VX_LIST)[0])
        for year in sorted(lst):
            for c in lst[year]:
                if c.get("duration_type") != "M":
                    continue
                exp = pd.Timestamp(c["expire_date"])
                try:
                    raw, f = ottieni("cboe_futures/" + c["path"].split("/")[-1], VX_CDN + c["path"], tries=2, timeout=40)
                    s = leggi_vx(raw)
                except Exception:
                    continue                      # contratti futuri non ancora quotati, file assenti o illeggibili
                files.append(f)
                contratti[(exp.year, exp.month)] = [exp, s]
                time.sleep(0.05)
        for yy in range(4, 14):
            for code in MONTH_CODES:
                try:
                    raw, f = ottieni(f"cboe_futures/CFE_{code}{yy:02d}_VX.csv", VX_ARCHIVE.format(code, yy), tries=1, timeout=30)
                    s = leggi_vx(raw)
                except Exception:
                    continue                      # mese senza contratto o file illeggibile
                if s.empty:
                    continue
                exp = s.index.max()               # nei file d'archivio la scadenza non e' indicata: ultima data presente
                files.append(f)
                s = s.where(s.index >= VX_RESCALE_DATE, s / 10.0)       # quotazione a 10x l'indice prima del 26/03/2007
                key = (exp.year, exp.month)
                if key in contratti:              # contratto presente in entrambe le fonti: il file recente ha la precedenza, l'archivio colma i buchi
                    contratti[key][1] = contratti[key][1].where(contratti[key][1] > 0).combine_first(s)
                else:
                    contratti[key] = [exp + pd.Timedelta(days=1), s]    # l'ultima riga resta utilizzabile
                time.sleep(0.05)
        contratti = [tuple(v) for _, v in sorted(contratti.items())]
        cont = continui_futures(contratti)
        extra = {"n_contratti": len(contratti), "regola": "per ogni giorno, i due contratti mensili con scadenza successiva più vicina; prezzo di regolamento (Settle)",
                 "avvertenza": "prima del 26/03/2007 i prezzi d'archivio sono divisi per 10 (cambio di scala del contratto); "
                               "nei file d'archivio 2004-2013 la scadenza è dedotta dall'ultima data presente"}
        fl = [{"file": "data/external/raw/cboe_futures/ (un file per contratto)", "n_file": len(files),
               "sha256_elenco": hashlib.sha256("".join(f["sha256"] for f in files).encode()).hexdigest()}]
        for name, d in (("vx1", "prima"), ("vx2", "seconda")):
            ok(name, cont[name], "Cboe Futures Exchange (CFE)", VX_LIST, "giornaliera", fl, {"descrizione": f"Future sul VIX, {d} scadenza mensile, prezzo di regolamento"} | extra)
    except Exception as e:
        for name in ("vx1", "vx2"):
            ko(name, "Future sul VIX", "Cboe Futures Exchange (CFE)", VX_LIST, str(e), "file storici dei contratti VX dal sito Cboe (Futures Historical Data)")
    return series, meta, vix_ohlc


# ------------------------------------------------------------------ verifica del nostro file

def verifica_vix(vix_ohlc: pd.DataFrame, confine: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    """Ricostruisce le barre settimanali dal VIX giornaliero ufficiale e le confronta con il file grezzo (solo periodo dev)."""
    raw = pd.read_csv(C.RAW_FILE, parse_dates=["week_start", "last_trading_day"])
    raw = raw[raw["last_trading_day"] <= confine].set_index("week_start")           # solo periodo di sviluppo
    off = settimanale_da_giornaliero(vix_ohlc[vix_ohlc.index <= confine])
    j = raw.join(off.add_suffix("_uff"), how="left")
    tol = 0.006
    res = {"periodo": [str(raw.index.min().date()), str(confine.date())], "settimane_confrontate": int(len(j)),
           "settimane_senza_dato_ufficiale": int(j["close_uff"].isna().sum()), "tolleranza": tol}
    for c in ("close", "high", "low", "open"):
        d = (j[c] - j[f"{c}_uff"]).abs()
        res[c] = {"uguali": int((d <= tol).sum()), "diversi": int((d > tol).sum()), "differenza_massima": float(d.max()),
                  "differenza_mediana_tra_i_diversi": float(d[d > tol].median()) if (d > tol).any() else 0.0}
    res["last_trading_day_uguali"] = int((j["last_trading_day"] == j["last_trading_day_uff"]).sum())
    res["trading_days_uguali"] = int((j["trading_days"] == j["trading_days_uff"]).sum())
    # Ipotesi sulla colonna open del file grezzo
    prev_close = j["close_uff"].shift(1)
    first = vix_ohlc[vix_ohlc.index <= confine].copy()
    ws = (first.index - pd.to_timedelta(first.index.day_of_week, unit="D")).normalize()
    g = first.groupby(ws)
    cand = {"apertura ufficiale del primo giorno della settimana": j["open_uff"],
            "chiusura ufficiale della settimana precedente": prev_close,
            "massimo ufficiale del primo giorno": g["high"].first().reindex(j.index),
            "minimo ufficiale del primo giorno": g["low"].first().reindex(j.index),
            "chiusura ufficiale del primo giorno": g["close"].first().reindex(j.index)}
    res["ipotesi_open"] = {k: {"uguali": int(((j["open"] - v).abs() <= tol).sum()),
                               "differenza_mediana": float((j["open"] - v).abs().median())} for k, v in cand.items()}
    res["open_grezzo_meno_ufficiale"] = {"mediana": float((j["open"] - j["open_uff"]).median()),
                                         "quota_grezzo_maggiore": float((j["open"] > j["open_uff"] + tol).mean()),
                                         "quota_grezzo_minore": float((j["open"] < j["open_uff"] - tol).mean())}
    res["salto_apertura_ufficiale"] = {"descrizione": "ln(apertura ufficiale / chiusura ufficiale della settimana precedente)",
                                       "mediana": float(np.log(j["open_uff"] / prev_close).median()),
                                       "quota_positivi": float((np.log(j["open_uff"] / prev_close) > 0).mean())}
    inc = j[(j["high_uff"] < j[["open_uff", "close_uff"]].max(axis=1) - tol) | (j["low_uff"] > j[["open_uff", "close_uff"]].min(axis=1) + tol)]
    res["settimane_ufficiali_ohlc_incoerenti"] = [
        {"week_start": str(k.date()), "open_ufficiale": float(r["open_uff"]), "high_ufficiale": float(r["high_uff"]), "low_ufficiale": float(r["low_uff"]),
         "close_ufficiale": float(r["close_uff"]), "open_file_grezzo": float(r["open"])} for k, r in inc.iterrows()]
    diff = j.assign(**{f"diff_{c}": j[c] - j[f"{c}_uff"] for c in ("open", "high", "low", "close")})
    bad = diff[(diff[["diff_high", "diff_low", "diff_close"]].abs() > tol).any(axis=1)]
    cols = ["last_trading_day", "open", "open_uff", "high", "high_uff", "low", "low_uff", "close", "close_uff", "trading_days", "trading_days_uff"]
    return res, bad[cols].reset_index()


# ------------------------------------------------------------------ programma principale

def main() -> None:
    RIUSA["attivo"] = "--riusa-originali" in sys.argv
    for d in (RAW, DEV, HOLD, OUT):
        d.mkdir(parents=True, exist_ok=True)
    dev_dates = pd.read_csv(C.DEV_FILE, usecols=["date"], parse_dates=["date"])["date"]
    hold_dates = pd.read_csv(C.HOLDOUT_FILE, usecols=["date"], parse_dates=["date"])["date"]      # solo le date, nessun valore
    confine = dev_dates.max()
    print(f"[dati esterni] confine dev/holdout: {confine.date()}")
    series, meta, vix_ohlc = fetch_all()

    # Giornaliero: una colonna per serie; riga in dev se l'osservazione era DISPONIBILE entro il confine
    daily = pd.DataFrame(series).sort_index()
    daily.index.name = "data_osservazione"
    for name, s in series.items():
        disp = s.index + pd.Timedelta(days=LAGS[name])
        s_dev = s[disp <= confine]
        m = meta[name]
        m |= {"prima_data": str(s.index.min().date()), "ultima_data": str(s.index.max().date()), "n_osservazioni_totali": int(len(s)),
              "n_osservazioni_dev": int(len(s_dev)), "ultima_data_dev": str(s_dev.index.max().date()) if len(s_dev) else None}
    d_dev, d_hold = dividi(daily, confine)
    for name in series:                                # nel file dev un valore non ancora disponibile al confine va tolto
        late = d_dev.index + pd.Timedelta(days=LAGS[name]) > confine
        d_dev.loc[late, name] = np.nan
    d_dev.to_csv(DEV / "esterni_giornalieri_dev.csv", date_format="%Y-%m-%d", float_format="%.6f")
    d_hold.to_csv(HOLD / "esterni_giornalieri_holdout.csv", date_format="%Y-%m-%d", float_format="%.6f")

    # Settimanale allineato alla colonna `date` del nostro file
    all_dates = pd.concat([dev_dates, hold_dates], ignore_index=True)
    weekly = pd.DataFrame(index=pd.DatetimeIndex(all_dates, name="date"))
    for name, s in series.items():
        a = allinea_settimanale(s, all_dates, LAGS[name])
        weekly[name] = a["valore"].to_numpy()
        weekly[f"{name}_data_oss"] = a["data_osservazione"].to_numpy()
    w_dev, w_hold = dividi(weekly, confine)
    w_dev.to_csv(DEV / "esterni_settimanali_dev.csv", date_format="%Y-%m-%d", float_format="%.6f")
    w_hold.to_csv(HOLD / "esterni_settimanali_holdout.csv", date_format="%Y-%m-%d", float_format="%.6f")
    for name in series:                                # statistiche solo sul periodo di sviluppo
        v = w_dev[name]
        first = v.first_valid_index()
        after = v[v.index >= first] if first is not None else v
        gg = series[name][series[name].index <= confine]
        bdays = pd.bdate_range(gg.index.min(), gg.index.max()) if len(gg) else []
        meta[name] |= {"settimanale_dev": {"prima_settimana_con_valore": str(first.date()) if first is not None else None,
                                           "settimane_con_valore": int(v.notna().sum()), "settimane_dev_totali": int(len(v)),
                                           "settimane_mancanti_dopo_l_inizio": int(after.isna().sum()),
                                           "settimane_mancanti_prima_dell_inizio": int(len(v) - len(after))},
                       "giorni_feriali_senza_valore_dev": int(len(bdays) - bdays.isin(gg.index).sum()) if meta[name]["frequenza"] == "giornaliera" else None}
    ver, bad = (None, None)
    if vix_ohlc is not None:
        ver, bad = verifica_vix(vix_ohlc, confine)
        (OUT / "c_verifica_vix.json").write_text(json.dumps(ver, indent=2, ensure_ascii=False))
        bad.to_csv(OUT / "c_verifica_vix_differenze.csv", index=False, date_format="%Y-%m-%d", float_format="%.4f")
    inv = {"data_scarico": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "confine_dev_holdout": str(confine.date()),
           "regola_allineamento": f"per ogni data settimanale, ultimo valore con data di osservazione + ritardo <= data; mancante se più vecchio di {STALE_DAYS} giorni",
           "file_prodotti": {"dev": [str((DEV / f).relative_to(C.ROOT)) for f in ("esterni_settimanali_dev.csv", "esterni_giornalieri_dev.csv")],
                             "holdout_protetto": [str((HOLD / f).relative_to(C.ROOT)) for f in ("esterni_settimanali_holdout.csv", "esterni_giornalieri_holdout.csv")],
                             "originali": str(RAW.relative_to(C.ROOT))},
           "serie": meta}
    (OUT / "c_inventario.json").write_text(json.dumps(inv, indent=2, ensure_ascii=False))
    n_ok = sum(m["stato"] == "scaricata" for m in meta.values())
    print(f"[dati esterni] serie scaricate: {n_ok} su {len(meta)}; non scaricabili: {[k for k, m in meta.items() if m['stato'] != 'scaricata']}")
    print(f"[dati esterni] inventario: {OUT / 'c_inventario.json'}")


if __name__ == "__main__":
    main()
