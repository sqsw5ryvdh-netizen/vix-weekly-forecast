"""Audit del file grezzo e costruzione dei dataset puliti (clean / dev / holdout).

Il file grezzo non viene mai modificato. Ogni trasformazione e' elencata in
TRANSFORMATIONS e scritta in data/processed/TRASFORMAZIONI.md e nel JSON di audit.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import config as C

WEEKDAYS = ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]

TRANSFORMATIONS = [
    "T1. Lettura del CSV grezzo senza modifiche; `week_start` e `last_trading_day` convertite in date.",
    "T2. Rimozione delle righe con `note` valorizzata che indicano una settimana incompleta "
    "(settimana in corso): la chiusura non e' una chiusura settimanale definitiva.",
    "T3. Colonna `date` = `last_trading_day` (data in cui la chiusura settimanale e' osservabile); "
    "e' la data di riferimento di ogni osservazione.",
    "T4. Colonne derivate: `log_close` = ln(close); `dlog_close` = differenza prima di `log_close`.",
    "T5. Colonne di segnalazione (nessun valore viene corretto): `flag_ohlc_incoerente` "
    "(high < max(open, close) oppure low > min(open, close)); `flag_settimana_corta` (trading_days <= 3).",
    "T6. Divisione temporale: le ultime {n} settimane complete -> holdout (file separato, bloccato); "
    "tutte le precedenti -> dev. Il file clean e' l'unione dei due.".format(n=C.HOLDOUT_WEEKS),
]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(C.RAW_FILE)


def _dates(s: pd.Series) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.to_datetime(s)]


def audit_raw(raw: pd.DataFrame) -> dict:
    ws = pd.to_datetime(raw["week_start"])
    lt = pd.to_datetime(raw["last_trading_day"])
    o, h, l, c = (raw[k] for k in ("open", "high", "low", "close"))
    price_cols = ["open", "high", "low", "close"]

    gaps = ws.diff().dt.days
    missing_weeks = int(((gaps[gaps > 7] / 7) - 1).sum())
    ohlc_bad = (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)
    partial = raw["note"].notna()

    dlog = np.log(c).diff()
    med = dlog.median()
    mad = (dlog - med).abs().median() * 1.4826
    rz = (dlog - med) / mad
    extreme = rz.abs() > 5

    gap_open = np.log(o / c.shift())

    def rows(mask, extra=None):
        out = raw.loc[mask].copy()
        if extra is not None:
            out = out.assign(**{k: v[mask].round(4) for k, v in extra.items()})
        return json.loads(out.to_json(orient="records"))

    return {
        "file": C.RAW_FILE.name,
        "sha256": hashlib.sha256(C.RAW_FILE.read_bytes()).hexdigest(),
        "formato": "CSV, separatore virgola, intestazione alla prima riga, codifica ASCII",
        "colonne": {k: str(v) for k, v in raw.dtypes.items()},
        "tipo_valore": "OHLC settimanale (open, high, low, close) + numero di sedute della settimana",
        "n_righe": int(len(raw)),
        "valori_mancanti": {k: int(v) for k, v in raw.isna().sum().items()},
        "prima_week_start": ws.min().strftime("%Y-%m-%d"),
        "ultima_week_start": ws.max().strftime("%Y-%m-%d"),
        "primo_last_trading_day": lt.min().strftime("%Y-%m-%d"),
        "ultimo_last_trading_day": lt.max().strftime("%Y-%m-%d"),
        "giorno_week_start": {WEEKDAYS[k]: int(v) for k, v in ws.dt.day_of_week.value_counts().sort_index().items()},
        "giorno_last_trading_day": {WEEKDAYS[k]: int(v) for k, v in lt.dt.day_of_week.value_counts().sort_index().items()},
        "trading_days": {str(k): int(v) for k, v in raw["trading_days"].value_counts().sort_index().items()},
        "ordinato_per_data": bool(ws.is_monotonic_increasing),
        "duplicati_week_start": int(ws.duplicated().sum()),
        "duplicati_righe": int(raw.duplicated().sum()),
        "settimane_mancanti": missing_weeks,
        "passo_settimanale_giorni": {str(int(k)): int(v) for k, v in gaps.dropna().value_counts().items()},
        "valori_non_positivi": int((raw[price_cols] <= 0).sum().sum()),
        "decimali_max": int(max(len(str(v).split(".")[-1]) for v in raw[price_cols].to_numpy().ravel())),
        "min_max": {k: [float(raw[k].min()), float(raw[k].max())] for k in price_cols},
        "n_ohlc_incoerenti": int(ohlc_bad.sum()),
        "righe_ohlc_incoerenti": rows(ohlc_bad),
        "n_settimane_parziali": int(partial.sum()),
        "righe_settimane_parziali": rows(partial),
        "n_settimane_corte": int((raw["trading_days"] <= 3).sum()),
        "righe_settimane_corte": rows(raw["trading_days"] <= 3),
        "variazioni_estreme_regola": "|z robusto| > 5 su dlog(close), z = (x - mediana) / (1.4826 * MAD)",
        "n_variazioni_estreme": int(extreme.sum()),
        "righe_variazioni_estreme": rows(extreme, {"dlog_close": dlog, "z_robusto": rz}),
        "gap_apertura": {
            "descrizione": "ln(open_t / close_{t-1}): salto tra chiusura settimanale e apertura successiva",
            "mediana": float(gap_open.median()),
            "quota_positivi": float((gap_open > 0).mean()),
            "min": float(gap_open.min()),
            "max": float(gap_open.max()),
        },
    }


def build_clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["date"] = pd.to_datetime(df["last_trading_day"])
    df = df.loc[df["note"].isna()].reset_index(drop=True)
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    df["log_close"] = np.log(c)
    df["dlog_close"] = df["log_close"].diff()
    df["flag_ohlc_incoerente"] = ((h < np.maximum(o, c)) | (l > np.minimum(o, c))).astype(int)
    df["flag_settimana_corta"] = (df["trading_days"] <= 3).astype(int)
    cols = ["date", "week_start", "open", "high", "low", "close", "trading_days",
            "log_close", "dlog_close", "flag_ohlc_incoerente", "flag_settimana_corta"]
    return df[cols]


def split_dev_holdout(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(clean) - C.HOLDOUT_WEEKS
    return clean.iloc[:n].copy(), clean.iloc[n:].copy()


def run() -> dict:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    audit = audit_raw(raw)
    clean = build_clean(raw)
    dev, hold = split_dev_holdout(clean)
    for df, path in ((clean, C.CLEAN_FILE), (dev, C.DEV_FILE), (hold, C.HOLDOUT_FILE)):
        df.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.6f")

    # Dell'holdout si registrano solo estremi temporali e numerosita', mai i valori.
    audit["dataset"] = {
        "clean": {"file": C.CLEAN_FILE.name, "n": int(len(clean)),
                  "da": _dates(clean["date"])[0], "a": _dates(clean["date"])[-1]},
        "dev": {"file": C.DEV_FILE.name, "n": int(len(dev)),
                "da": _dates(dev["date"])[0], "a": _dates(dev["date"])[-1]},
        "holdout": {"file": C.HOLDOUT_FILE.name, "n": int(len(hold)),
                    "da": _dates(hold["date"])[0], "a": _dates(hold["date"])[-1]},
    }
    audit["trasformazioni"] = TRANSFORMATIONS
    (C.REPORTS_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    (C.PROCESSED_DIR / "TRASFORMAZIONI.md").write_text(
        "# Trasformazioni dal file grezzo ai dataset puliti\n\n"
        f"Sorgente: `data/raw/{C.RAW_FILE.name}` (sha256 `{audit['sha256']}`).\n"
        "Generato da `python -m src.data_audit`.\n\n" + "\n".join(f"- {t}" for t in TRANSFORMATIONS) + "\n\n"
        "| file | righe | da | a |\n|---|---|---|---|\n"
        + "\n".join(f"| `{v['file']}` | {v['n']} | {v['da']} | {v['a']} |" for v in audit["dataset"].values())
        + "\n\n`vix_weekly_holdout.csv` e' bloccato: non va usato per analisi, stime o valutazioni in Fase 0.\n"
    )
    return audit


if __name__ == "__main__":
    a = run()
    print(json.dumps(a["dataset"], indent=2))
