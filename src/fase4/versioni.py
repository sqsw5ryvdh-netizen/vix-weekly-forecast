"""Fase 4, Parte C: versioni finaliste complete (chiusura centrale, fasce, probabilita', etichetta, candele).

F1  solo storico del VIX. Chiusura centrale e fasce: D4 (regressione quantilica sulle variabili di M1). Probabilita'
    ed etichetta: D1 (errori passati attorno al centro M1). Candele K2 con le ampiezze scelte nella Parte A.
F2  con la struttura a termine. Costruita allo stesso modo, con la variabile X1 = VIX3M / VIX - 1 in piu':
    centro = M1 + X1 (regressione diretta); chiusura centrale e fasce = regressione quantilica con X1 (mediana e
    quantili); probabilita' ed etichetta = errori passati attorno al centro M1 + X1 (almeno 104, come D1); stesse
    candele di F1, ancorate alla chiusura centrale di F2.
    Un'origine e' ammessa per F2 solo se X1 e' valida in quella settimana e ci sono almeno 249 settimane valide
    (regola della Fase 3); altrimenti F2 non e' disponibile e si ripiega su F1.

F3  attorno al modello a regimi N1 (presente solo perche' N1 ha passato la regola della Parte B, sulla direzione).
    Tutto dalla distribuzione prevista da N1 (mistura di normali sui percorsi di regime): centro = media; chiusura
    centrale = mediana; fasce = quantili; probabilita' ed etichetta = probabilita' della mistura (e' la probabilita'
    che e' stata misurata nella Parte B); stesse candele di F1, ancorate alla chiusura centrale di F3.
    Parametri ristimati ogni 26 settimane (date fisse a partire dalla settimana 260 del file), ultima stima valida.

Tutte le funzioni ricevono la storia e un indice di origine t e usano solo le settimane 0..t (e, per X1, solo i
dati giornalieri fino all'ultima seduta della settimana t). Nessun parametro e' stimato o scelto qui: le finestre e
le regole sono quelle delle fasi precedenti, elencate in `configurazione()` e congelate in versioni/*.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .. import config as C
from ..fase2 import densita as D
from ..fase2b import modello as MO
from ..fase2c import componenti as CP
from ..fase3 import candele as K3
from ..fase3 import modelli as M
from ..fase3 import variabili as V
from . import candele as KA
from . import regimi as RE

VERSIONI = ("f1", "f2", "f3")
NOMI = {"f1": "F1 solo storico del VIX", "f2": "F2 con struttura a termine (VIX3M)", "f3": "F3 modello a regimi (N1)"}
ORIZZONTI = MO.ORIZZONTI
X1 = V.BLOCCHI["X1"][0]
CARTELLA = C.ROOT / "versioni"
SORGENTI_CONGELATI = ("src/baselines.py", "src/fase1/modelli.py", "src/fase2/densita.py", "src/fase2b/modello.py", "src/fase2c/componenti.py",
                      "src/fase3/candele.py", "src/fase3/modelli.py", "src/fase3/variabili.py", "src/fase4/candele.py", "src/fase4/regimi.py", "src/fase4/versioni.py")


class NonDisponibile(Exception):
    """F2 non calcolabile all'origine richiesta (X1 mancante o storia insufficiente)."""


def x1_settimanale(date_sett, giorn: pd.DataFrame) -> np.ndarray:
    """X1 = VIX3M / VIX - 1 per ogni settimana, con la regola di allineamento della Fase 3: ultima seduta della
    settimana in cui VIX3M e VIX hanno entrambi un valore; mancante se nella settimana non ce n'e' nessuna."""
    date = np.asarray(pd.to_datetime(date_sett), dtype="datetime64[ns]")
    j = giorn[["vix3m", "vix"]].dropna()
    k, ok = V._ultimo_nella_settimana(j["vix"], date)
    return V._campiona((j["vix3m"] / j["vix"] - 1).to_numpy(), k, ok)


def _etichette(cand: list[dict], p_su: dict[int, float], centrali: dict[int, float], versione: str) -> list[dict]:
    out = []
    for c in cand:
        d, testo = K3.etichetta(p_su[c["h"]])
        out.append(c | {"p_chiusura_su": p_su[c["h"]], "direzione_chiusura": d, "etichetta": testo, "chiusura_centrale": centrali[c["h"]],
                        "mediana_d4": centrali[c["h"]], "versione": versione})
    return out


def prevedi_f1(df: pd.DataFrame, t: int, variante: str) -> tuple[dict, list[dict]]:
    """F1 all'origine t: (previsione con chiusura centrale D4, fasce D4, probabilita' D1; candele K2 della variante scelta)."""
    prev = MO.prevedi(df, t) | {"versione": "f1"}
    S = CP.serie(df.iloc[: t + 1])
    x = np.log(S.c)
    centrali = {s["h"]: s["chiusura"] for s in prev["settimane"]}
    cand = KA.candele(S, t, prev["chiusura_origine"], centrali, variante)
    return prev, _etichette(cand, {h: K3.prob_chiusura_su(x, t, h) for h in ORIZZONTI}, centrali, "f1")


def f2_ammessa(z: np.ndarray, t: int) -> bool:
    ok = M.valide(np.asarray(z, dtype=float)[: t + 1].reshape(-1, 1))
    return bool(t >= C.MIN_TRAIN_WEEKS - 1 and ok[t] and ok.sum() >= M.MIN_VALIDE)


def centri_f2(x: np.ndarray, Z: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Centri M1 + X1 a tutte le origini ammesse con esito noto (i + h <= t): (origini, centri). Usa solo x e Z fino a t."""
    o = M.origini(Z, h)
    return o, np.array([M.t1(x[: i + 1], Z[: i + 1], h) for i in o.tolist()])


def prevedi_f2(df: pd.DataFrame, t: int, z: np.ndarray, variante: str) -> tuple[dict, list[dict]]:
    """F2 all'origine t. `z[i]` = X1 della settimana i (NaN se mancante); vengono lette solo le settimane 0..t.

    Solleva NonDisponibile se l'origine non e' ammessa o se gli errori passati attorno al centro sono meno di 104.
    """
    z = np.asarray(z, dtype=float)
    if not f2_ammessa(z, t):
        raise NonDisponibile("X1 (VIX3M / VIX - 1) manca nella settimana di origine oppure la sua storia è insufficiente")
    d = df.iloc[: t + 1]
    S = CP.serie(d)
    x, Z = np.log(S.c), z[: t + 1].reshape(-1, 1).copy()
    data0, righe, p_su = pd.Timestamp(d["date"].iloc[-1]), [], {}
    for h in ORIZZONTI:
        o, c = centri_f2(x, Z, h)
        if len(o) < D.MIN_ERR:
            raise NonDisponibile(f"meno di {D.MIN_ERR} errori passati attorno al centro M1 + X1 (h = {h})")
        centro = M.t1(x, Z, h)
        q, incrociati = M.t3(x, Z, h)
        g, emp = D.Griglia(q), D.Empirica(centro, x[o + h] - c)
        qr = np.exp(g.quantiles(np.array(D.Q_REPORT)))
        r = {"h": h, "quantili": {f"q{int(100 * k):02d}": float(v) for k, v in zip(D.Q_REPORT, qr)}, "quantili_riordinati": bool(incrociati),
             "centro": float(np.exp(centro)), "probabilita": {f"VIX>{int(s)}": float(emp.prob_above(np.log(s))) for s in MO.SOGLIE},
             "n_errori": int(len(o)), "data": str((data0 + pd.Timedelta(days=7 * h)).date())}
        r["chiusura"] = r["quantili"]["q50"]
        r["fasce"] = {str(lev): [r["quantili"][f"q{int(100 * a):02d}"], r["quantili"][f"q{int(100 * b):02d}"]] for lev, (a, b) in MO.FASCE.items()}
        p_su[h] = float(emp.prob_above(float(x[t])))
        righe.append(r)
    prev = {"versione": "f2", "data_origine": str(data0.date()), "indice_origine": int(t), "chiusura_origine": float(S.c[-1]),
            "settimane_di_storia": int(t + 1), "x1_origine": float(z[t]), "settimane": righe}
    centrali = {r["h"]: r["chiusura"] for r in righe}
    return prev, _etichette(KA.candele(S, t, prev["chiusura_origine"], centrali, variante), p_su, centrali, "f2")


class Mistura:
    """Mistura di normali in log (distribuzione prevista da N1): quantili per bisezione, probabilita' esatte."""

    def __init__(self, w, m, s):
        self.w, self.m, self.s = np.asarray(w, dtype=float), np.asarray(m, dtype=float), np.asarray(s, dtype=float)

    @property
    def mean(self) -> float:
        return float(self.w @ self.m)

    def cdf(self, v) -> np.ndarray:
        v = np.atleast_1d(np.asarray(v, dtype=float))
        return (self.w * stats.norm.cdf((v[:, None] - self.m) / self.s)).sum(axis=1)

    def prob_above(self, thr: float) -> float:
        return float(self.w @ stats.norm.sf((thr - self.m) / self.s))

    def quantiles(self, taus) -> np.ndarray:
        taus = np.asarray(taus, dtype=float)
        lo, hi = np.full(len(taus), (self.m - 10 * self.s).min()), np.full(len(taus), (self.m + 10 * self.s).max())
        for _ in range(60):
            mid = (lo + hi) / 2
            sotto = self.cdf(mid) < taus
            lo, hi = np.where(sotto, mid, lo), np.where(sotto, hi, mid)
        return (lo + hi) / 2


def riga_da_mistura(mix: Mistura, h: int, x_t: float, data0: pd.Timestamp) -> tuple[dict, float]:
    """Riga della previsione (quantili, fasce, probabilita') e probabilita' di chiudere sopra l'origine, da una mistura."""
    qr = np.exp(mix.quantiles(np.array(D.Q_REPORT)))
    r = {"h": h, "quantili": {f"q{int(100 * k):02d}": float(v) for k, v in zip(D.Q_REPORT, qr)}, "quantili_riordinati": False, "centro": float(np.exp(mix.mean)),
         "probabilita": {f"VIX>{int(s)}": mix.prob_above(float(np.log(s))) for s in MO.SOGLIE}, "data": str((data0 + pd.Timedelta(days=7 * h)).date())}
    r["chiusura"] = r["quantili"]["q50"]
    r["fasce"] = {str(lev): [r["quantili"][f"q{int(100 * a):02d}"], r["quantili"][f"q{int(100 * b):02d}"]] for lev, (a, b) in MO.FASCE.items()}
    return r, mix.prob_above(x_t)


def prevedi_f3(df: pd.DataFrame, t: int, variante: str) -> tuple[dict, list[dict]]:
    """F3 all'origine t: tutto dalla distribuzione prevista dal modello a regimi N1. Usa solo df.iloc[: t + 1]."""
    if t < MO.origine_minima():
        raise NonDisponibile("origine troppo vicina all'inizio dei dati")
    d = df.iloc[: t + 1]
    S = CP.serie(d)
    x = np.log(S.c)
    par = RE.parametri_all_origine(x)
    pi = None if par is None else RE.filtrate(x, par)
    data0, righe, p_su = pd.Timestamp(d["date"].iloc[-1]), [], {}
    for h in ORIZZONTI:
        r, p_su[h] = riga_da_mistura(Mistura(*RE.mistura_all_origine(x, h, par, pi)), h, float(x[t]), data0)
        righe.append(r)
    regime = None if par is None else {"p_regime_alto": float(pi[int(np.argmax(par["mu"]))]), "stima_su_settimane": par["n"]} | RE.descrizione(par)
    prev = {"versione": "f3", "data_origine": str(data0.date()), "indice_origine": int(t), "chiusura_origine": float(S.c[-1]),
            "settimane_di_storia": int(t + 1), "regime": regime, "ripiego_ar1": par is None, "settimane": righe}
    centrali = {r["h"]: r["chiusura"] for r in righe}
    return prev, _etichette(KA.candele(S, t, prev["chiusura_origine"], centrali, variante), p_su, centrali, "f3")


def configurazione(versione: str, variante: str) -> dict:
    """Tutti i parametri, le finestre e le regole della versione (senza data e commit: li aggiunge `congela`)."""
    candele = {"costruzione": "K2: apertura della settimana h = chiusura centrale della settimana h - 1 (la prima: ultima chiusura reale) x rapporto di apertura",
               "nome_nel_registro": KA.costruzione(variante), "rapporto_apertura": "mediana espandibile di apertura / chiusura precedente",
               "direzione_del_corpo": "L0: frequenza storica espandibile delle settimane con chiusura sopra l'apertura; rialzo se >= 0,5",
               "variante_ampiezze": variante, "finestre_ampiezze": KA.VARIANTI[variante], "assegnazione_ampiezze": {c: {str(h): m for h, m in d.items()} for c, d in KA.ASSEGNAZIONE.items()},
               "variabili_regressione": list(M.NOMI_M1) + list(CP.NOMI_EXTRA), "esclusioni": "settimane con apertura fuori dall'intervallo minimo-massimo (8 novembre 2004)"}
    comuni = {"orizzonti_settimane": list(ORIZZONTI), "obiettivo": "chiusura settimanale del VIX", "min_settimane_di_storia": C.MIN_TRAIN_WEEKS,
              "ristima": "a ogni origine, finestra espandibile (salvo le finestre delle ampiezze delle candele)",
              "griglia_quantili": [float(v) for v in D.TAUS], "quantili_riportati": list(D.Q_REPORT), "fasce": {str(k): list(v) for k, v in D.INTERVALS.items()},
              "soglie_probabilita": list(MO.SOGLIE), "min_errori_passati": D.MIN_ERR,
              "etichetta": {"evento": "chiusura a t + h sopra la chiusura all'origine", "neutra_tra": list(K3.NEUTRA)}, "candele": candele,
              "file_sorgenti_congelati": list(SORGENTI_CONGELATI)}
    if versione == "f1":
        return {"versione": "f1", "nome": NOMI["f1"], "dati": ["VIX settimanale (apertura, massimo, minimo, chiusura)"],
                "centro": "M1: regressione diretta di x_{t+h} su [1, x_t, media a 4 settimane, media a 12 settimane], x = log della chiusura",
                "chiusura_centrale": "D4: mediana della regressione quantilica sulle variabili di M1",
                "fasce_da": "D4: quantili della stessa regressione quantilica (riordinati se incrociati)",
                "probabilita_ed_etichetta_da": "D1: centro M1 + errori passati con esito noto, quantili e frequenze empiriche",
                "origine_minima_indice": MO.origine_minima()} | comuni
    if versione == "f3":
        return {"versione": "f3", "nome": NOMI["f3"], "dati": ["VIX settimanale (apertura, massimo, minimo, chiusura)"],
                "modello": {"tipo": "autoregressivo di ordine 1 sul log della chiusura con 2 regimi markoviani; media e varianza diverse per regime, coefficiente autoregressivo comune",
                            "stima": "massima verosimiglianza, statsmodels MarkovAutoregression(k_regimes=2, order=1, trend='c', switching_ar=False, switching_variance=True), impostazioni predefinite",
                            "ristima_ogni_settimane": RE.PASSO_STIMA, "prima_data_di_ristima_indice": RE.PRIMA_ORIGINE,
                            "stima_valida_se": "ottimizzazione convergente, varianze > 1e-8, probabilità di transizione in (1e-6, 1 - 1e-6), |phi| < 1",
                            "se_non_valida": "ultima stima valida precedente; se non esiste, AR(1) a un solo regime con errori normali",
                            "probabilita_di_regime": "filtrate, ricalcolate a ogni origine sulla storia fino all'origine"},
                "centro": "media della distribuzione prevista (mistura di normali sui percorsi di regime da t a t + h)",
                "chiusura_centrale": "mediana della distribuzione prevista", "fasce_da": "quantili della distribuzione prevista",
                "probabilita_ed_etichetta_da": "probabilità della distribuzione prevista (nessun errore empirico)",
                "origine_minima_indice": MO.origine_minima()} | comuni
    return {"versione": "f2", "nome": NOMI["f2"], "dati": ["VIX settimanale (apertura, massimo, minimo, chiusura)", "VIX giornaliero (chiusura)", "VIX3M giornaliero (chiusura)"],
            "variabile_aggiuntiva": {"nome": X1, "definizione": "VIX3M / VIX - 1",
                                     "allineamento": "ultima seduta della settimana di origine in cui VIX3M e VIX hanno entrambi un valore; mancante se non ce n'è nessuna; nessun valore portato avanti",
                                     "ritardo_pubblicazione_giorni": 0},
            "centro": "M1 + X1: regressione diretta di x_{t+h} su [1, x_t, media a 4 settimane, media a 12 settimane, X1]",
            "chiusura_centrale": "mediana della regressione quantilica sulle variabili di M1 + X1 (X1 standardizzata nella finestra di stima)",
            "fasce_da": "quantili della stessa regressione quantilica (riordinati se incrociati)",
            "probabilita_ed_etichetta_da": "centro M1 + X1 alle origini ammesse + errori passati con esito noto, quantili e frequenze empiriche",
            "ammissione": {"x1_valida_all_origine": True, "min_settimane_valide": M.MIN_VALIDE},
            "ripiego": "se X1 manca all'origine o gli errori passati sono meno del minimo, si usa F1 e lo si segnala"} | comuni
