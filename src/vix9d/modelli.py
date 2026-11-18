"""VIX9D: modelli (elenco CHIUSO). Tutto su x = log(chiusura settimanale del VIX9D), orizzonti h = 1 e 2.

Chiusura (regressione diretta ai minimi quadrati, finestra espandibile):
  R0 random walk;  R1 tipo HAR (ultimo valore, media a 4 e a 12 settimane; riferimento);
  R2 = R1 + ln(VIX9D / VIX) all'origine;  R3 = R1 + candela del VIX9D (escursione ln(high/low) dell'ultima settimana, sua
  media su 4 settimane, posizione della chiusura nell'escursione);  R4 = R1 + variabili di R2 e R3.
Fasce: regressione quantilica (quantili 5, 10, 25, 50, 75, 90, 95) sulle variabili del modello di chiusura.
Probabilita' delle soglie: centro del modello di chiusura + errori passati con esito noto (almeno 52), come D1 per il VIX.
Direzione (chiusura a t + h sopra la chiusura a t): logistica con penalita' leggera sulle variabili di R1 (base) e
  con ln(VIX9D / VIX) in piu'.
Estremi: regressione sulla mediana di ln(massimo della settimana t + h / chiusura a t) e dello stesso per il minimo;
  insiemi S0 (variabili di R1), S1 (S0 + candela del VIX9D), S2 (S1 + ln(VIX9D / VIX)); riferimento: regola fissa.
Candele: apertura = chiusura precedente x rapporto mediano di apertura; corpo con direzione dalla frequenza storica e
  ampiezza media delle ultime 260 settimane; estremi che comandano (corpo tagliato dentro minimo-massimo).

Dati: prima di meta' novembre 2013 le settimane hanno solo le chiusure. Le variabili della candela, gli estremi,
il rapporto di apertura e il corpo usano solo le settimane con candela vera (colonna `candela`); le chiusure valgono da
sempre. Una previsione esiste solo con almeno 144 coppie (origine, esito) valide, cioe' 156 settimane di storia.
Convenzione anti look-ahead: ogni funzione riceve solo le settimane 0..t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import baselines as B
from ..fase2 import densita as D
from ..fase3 import modelli as M3
from . import config as G

CANDELA = ("escursione", "escursione_4", "posizione")
RAPPORTO = ("ln_rapporto_vix",)
CHIUSURA = {"R1": (), "R2": RAPPORTO, "R3": CANDELA, "R4": CANDELA + RAPPORTO}
ESTREMI = {"S0": (), "S1": CANDELA, "S2": CANDELA + RAPPORTO}
DIREZIONE = {"BASE": (), "RAPP": RAPPORTO}
SENZA_VIX = {"chiusura": ("R1", "R3"), "estremi": ("S0", "S1"), "direzione": ("BASE",)}
CON_VIX = {"chiusura": ("R2", "R4"), "estremi": ("S2",), "direzione": ("RAPP",)}
NOMI = {"R0": "R0 random walk", "R1": "R1 tipo HAR (riferimento)", "R2": "R2 = R1 + ln(VIX9D/VIX)", "R3": "R3 = R1 + candela del VIX9D", "R4": "R4 = R1 + candela + ln(VIX9D/VIX)",
        "S0": "S0 variabili di R1", "S1": "S1 = S0 + candela del VIX9D", "S2": "S2 = S1 + ln(VIX9D/VIX)", "BASE": "logistica di base (variabili di R1)", "RAPP": "logistica con ln(VIX9D/VIX)",
        "FISSA": "regola fissa", "FREQ": "frequenza storica"}
TAUS = np.array(G.TAUS)


def variabili(df: pd.DataFrame) -> pd.DataFrame:
    """Variabili settimanali (riga i = valori noti alla chiusura della settimana i). Le variabili della candela sono
    mancanti dove la settimana (o una delle 4 della media) ha solo le chiusure; ln(VIX9D/VIX) dove manca il VIX."""
    ok = df["candela"].to_numpy(bool)
    h, l, c = (df[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    esc = np.where(ok, np.log(h / l), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = np.where(ok, np.where(h > l, (c - l) / np.where(h > l, h - l, 1.0), 0.5), np.nan)
    return pd.DataFrame({"escursione": esc, "escursione_4": pd.Series(esc).rolling(4).mean().to_numpy(), "posizione": pos, "ln_rapporto_vix": np.log(c / df["vix"].to_numpy(dtype=float))})


def coppie(n: int, Z: np.ndarray, h: int, bersaglio_valido: np.ndarray | None = None) -> np.ndarray:
    """Indici i delle coppie di stima all'origine t = n - 1: 11 <= i <= t - h, variabili valide a i, esito valido a i + h."""
    i = np.arange(11, n - h)
    ok = np.ones(len(i), bool)
    if Z.shape[1]:
        ok &= np.isfinite(Z[i]).all(axis=1)
    if bersaglio_valido is not None:
        ok &= bersaglio_valido[i + h]
    return i[ok]


def _pronto(i: np.ndarray, Z: np.ndarray) -> bool:
    return len(i) >= G.MIN_COPPIE and bool(np.isfinite(Z[-1]).all())


def _matrice(x: np.ndarray, Z: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[1, x_i, media4, media12, variabili standardizzate nella finestra] per le coppie i e per l'origine."""
    F = B.har_features(x)
    if not Z.shape[1]:
        return F[i - 11], F[-1]
    mu, sd = Z[i].mean(axis=0), Z[i].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return np.column_stack([F[i - 11], (Z[i] - mu) / sd]), np.concatenate([F[-1], (Z[-1] - mu) / sd])


def chiusura(x: np.ndarray, Z: np.ndarray, h: int) -> float:
    """Previsione di x_{t+h} (log): minimi quadrati su [1, x_i, media4, media12, Z_i]. Con Z senza colonne e' R1. NaN se non pronta."""
    i = coppie(len(x), Z, h)
    if not _pronto(i, Z):
        return float("nan")
    X, nuovo = _matrice(x, Z, i)
    return float(nuovo @ np.linalg.lstsq(X, x[i + h], rcond=None)[0])


def quantili(x: np.ndarray, Z: np.ndarray, h: int) -> np.ndarray:
    """Quantili (5, 10, 25, 50, 75, 90, 95) di x_{t+h}, riordinati se incrociati. NaN se non pronta."""
    i = coppie(len(x), Z, h)
    if not _pronto(i, Z):
        return np.full(len(TAUS), np.nan)
    X, nuovo = _matrice(x, Z, i)
    return np.sort(D.quantreg(X, x[i + h], TAUS) @ nuovo)


def direzione(x: np.ndarray, Z: np.ndarray, h: int) -> float:
    """P(x_{t+h} > x_t): logistica con penalita' L2 leggera (come E1 del VIX) su [x_i, media4, media12, Z_i] standardizzate."""
    i = coppie(len(x), Z, h)
    if not _pronto(i, Z):
        return float("nan")
    F = B.har_features(x)[:, 1:]
    A = np.column_stack([np.vstack([F[i - 11], F[-1]]), np.vstack([Z[i], Z[-1]])]) if Z.shape[1] else np.vstack([F[i - 11], F[-1]])
    mu, sd = A[:-1].mean(axis=0), A[:-1].std(axis=0)
    As = (A - mu) / np.where(sd > 0, sd, 1.0)
    b = M3._logit(As[:-1], (x[i + h] > x[i]).astype(float), np.full(As.shape[1], M3.RIDGE_LOGIT))
    return float(1 / (1 + np.exp(-np.clip(np.concatenate([[1.0], As[-1]]) @ b, -30, 30))))


def frequenza_rialzo(x: np.ndarray, h: int) -> float:
    return float((x[h:] > x[:-h]).mean())


def estremo(x: np.ndarray, livelli: np.ndarray, Z: np.ndarray, h: int, candela: np.ndarray) -> float:
    """Livello (punti) dell'estremo della settimana t + h: chiusura_t x exp(mediana condizionata di ln(livelli[i + h] / chiusura_i)).
    Entrano solo le coppie il cui esito i + h e' una settimana con candela vera. NaN se non pronta."""
    i = coppie(len(x), Z, h, candela)
    if not _pronto(i, Z):
        return float("nan")
    X, nuovo = _matrice(x, Z, i)
    return float(np.exp(x[-1] + nuovo @ D.quantreg(X, np.log(livelli[i + h]) - x[i], [0.5])[0]))


def regola_fissa(x: np.ndarray, livelli: np.ndarray, h: int, candela: np.ndarray) -> float:
    """Riferimento: chiusura all'origine x mediana storica di livelli[i + h] / chiusura[i] (esiti con candela vera)."""
    i = np.arange(0, len(x) - h)
    i = i[candela[i + h]]
    return float(np.exp(x[-1]) * np.median(livelli[i + h] / np.exp(x[i]))) if len(i) >= G.MIN_COPPIE else float("nan")


def componenti_candela(o: np.ndarray, c: np.ndarray, candela: np.ndarray) -> dict:
    """Rapporto mediano di apertura (apertura / chiusura precedente), frequenza storica delle settimane con chiusura sopra
    l'apertura e ampiezza media del corpo |ln(chiusura / apertura)| sulle ultime 260 settimane con candela vera."""
    j = np.arange(1, len(c))
    jj = j[candela[j] & candela[j - 1]]
    k = np.flatnonzero(candela)
    if len(jj) < G.MIN_COPPIE:
        return {"rapporto_apertura": float("nan"), "p_su": float("nan"), "corpo": float("nan")}
    return {"rapporto_apertura": float(np.median(o[jj] / c[jj - 1])), "p_su": float((c[k] > o[k]).mean()), "corpo": float(np.abs(np.log(c[k] / o[k]))[-G.FINESTRA_CORPO:].mean())}


def candela_prevista(base: float, comp: dict, massimo: float, minimo: float) -> dict:
    """Candela: apertura = base x rapporto di apertura; corpo nella direzione piu' frequente con ampiezza media; massimo e
    minimo comandano: se apertura o chiusura escono dall'intervallo si accorcia il corpo, gli estremi non si spostano."""
    lo, hi = (minimo, massimo) if minimo <= massimo else (massimo, minimo)
    ap = base * comp["rapporto_apertura"]
    ch = ap * float(np.exp(comp["corpo"] if comp["p_su"] >= 0.5 else -comp["corpo"]))
    a2, c2 = float(np.clip(ap, lo, hi)), float(np.clip(ch, lo, hi))
    return {"apertura": a2, "chiusura": c2, "massimo": float(hi), "minimo": float(lo), "apertura_tagliata": bool(a2 != ap), "chiusura_tagliata": bool(c2 != ch),
            "estremi_scambiati": bool(minimo > massimo)}


def etichetta(p_su: float) -> tuple[str, str]:
    if p_su >= G.NEUTRA[1]:
        return "rialzo", f"rialzo {round(100 * p_su)}%"
    if p_su <= G.NEUTRA[0]:
        return "ribasso", f"ribasso {round(100 * (1 - p_su))}%"
    return "neutra", "neutra"


def quintile(livello: float, storia: np.ndarray) -> int:
    """Fascia di livello (1..5): quintile della chiusura all'origine rispetto alla storia disponibile fino all'origine."""
    return int(np.searchsorted(np.quantile(storia, [0.2, 0.4, 0.6, 0.8]), livello, side="right")) + 1
