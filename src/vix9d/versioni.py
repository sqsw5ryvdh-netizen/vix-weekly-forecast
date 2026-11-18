"""VIX9D: le due versioni complete, G1 e GS, per le settimane +1 e +2.

G1  solo storico del VIX9D: per ogni componente il candidato senza VIX scelto dalle regole (chiusura R1 o R3, estremi S0 o
    S1, direzione dalla logistica di base).
GS  con il VIX: i candidati con ln(VIX9D / VIX) entrano dove battono il corrispondente di G1; altrimenti resta quello di G1.
    Se a un'origine il VIX della settimana manca, GS non e' calcolabile e il programma ripiega su G1.
Componenti, per ogni orizzonte h:
  centro               previsione del modello di chiusura scelto (minimi quadrati);
  chiusura centrale    mediana della regressione quantilica sulle stesse variabili; fasce = suoi quantili;
  probabilita' soglie  centro + errori passati dello stesso modello con esito noto (almeno 52);
  etichetta            probabilita' che la chiusura a t + h superi quella dell'origine, dalla logistica scelta;
  massimo e minimo     regressione sulla mediana con l'insieme scelto; comandano la candela;
  candela              apertura = chiusura precedente (per +2: chiusura centrale di +1) x rapporto mediano di apertura; corpo
                       con direzione dalla frequenza storica e ampiezza media delle ultime 260 settimane, tagliato dentro
                       minimo-massimo; colore secondo l'etichetta, grigio se neutra.
`scelte` = {"chiusura": {versione: {h: modello}}, "estremi": {versione: {estremo: {h: insieme}}}, "direzione": {versione: {h: modello}}}.
Tutte le funzioni usano solo le settimane 0..t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as G
from . import modelli as M

VERSIONI = ("g1", "gs")
NOMI = {"g1": "G1 solo storico del VIX9D", "gs": "GS con il VIX"}
CAMPI = ("apertura", "massimo", "minimo", "chiusura")
QN = [f"q{int(round(100 * t)):02d}" for t in G.TAUS]
SORGENTI_CONGELATI = ("src/baselines.py", "src/fase2/densita.py", "src/fase3/modelli.py", "src/vix9d/config.py", "src/vix9d/modelli.py", "src/vix9d/versioni.py")
COSTRUZIONE = "V9-estremi"


class NonDisponibile(Exception):
    """La versione non e' calcolabile all'origine richiesta (storia insufficiente o VIX mancante)."""


def usa_vix(scelte: dict, versione: str) -> bool:
    nomi = list(scelte["chiusura"][versione].values()) + list(scelte["direzione"][versione].values()) + [s for e in scelte["estremi"][versione].values() for s in e.values()]
    return any(k in M.CON_VIX["chiusura"] + M.CON_VIX["estremi"] + M.CON_VIX["direzione"] for k in nomi)


def probabilita_soglie(x: np.ndarray, Z: np.ndarray, h: int, centro: float) -> tuple[dict, int]:
    """P(VIX9D > soglia) dal centro + errori passati dello stesso modello (origini i con i + h <= t)."""
    t = len(x) - 1
    err = []
    for i in range(G.MIN_SETTIMANE - 1, t - h + 1):
        c = M.chiusura(x[: i + 1], Z[: i + 1], h)
        if np.isfinite(c):
            err.append(x[i + h] - c)
    if len(err) < G.MIN_ERRORI:
        raise NonDisponibile(f"meno di {G.MIN_ERRORI} errori passati per le probabilità delle soglie (h = {h})")
    e = np.array(err)
    return {f"VIX>{int(s)}": float((centro + e > np.log(s)).mean()) for s in G.SOGLIE}, len(e)


def prevedi(df: pd.DataFrame, t: int, versione: str, scelte: dict) -> tuple[dict, list[dict]]:
    """Previsione completa della versione all'origine t (indice di riga). `df`: date, open, high, low, close, candela, vix."""
    d = df.iloc[: t + 1]
    VV = M.variabili(d)
    x, cand = np.log(d["close"].to_numpy(dtype=float)), d["candela"].to_numpy(bool)
    o_, h_, l_, c_ = (d[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    Z = lambda cols: VV[list(cols)].to_numpy()
    comp = M.componenti_candela(o_, c_, cand)
    data0, righe, candele = pd.Timestamp(d["date"].iloc[-1]), [], []
    for h in G.ORIZZONTI:
        mc, md = scelte["chiusura"][versione][str(h)], scelte["direzione"][versione][str(h)]
        Zc = Z(M.CHIUSURA[mc])
        centro, q, p_su = M.chiusura(x, Zc, h), M.quantili(x, Zc, h), M.direzione(x, Z(M.DIREZIONE[md]), h)
        est = {e: M.estremo(x, L, Z(M.ESTREMI[scelte["estremi"][versione][e][str(h)]]), h, cand) for e, L in (("massimo", h_), ("minimo", l_))}
        if not (np.isfinite(centro) and np.isfinite(q).all() and np.isfinite(p_su) and all(np.isfinite(v) for v in est.values()) and np.isfinite(comp["corpo"])):
            raise NonDisponibile("storia insufficiente oppure VIX mancante nella settimana di origine")
        prob, n_err = probabilita_soglie(x, Zc, h, centro)
        qv = np.exp(q)
        r = {"h": h, "data": str((data0 + pd.Timedelta(days=7 * h)).date()), "quantili": {k: float(v) for k, v in zip(QN, qv)}, "centro": float(np.exp(centro)), "chiusura": float(qv[3]),
             "probabilita": prob, "n_errori": n_err, "modello_chiusura": mc, "modello_direzione": md}
        r["fasce"] = {str(lev): [r["quantili"][f"q{int(round(100 * a)):02d}"], r["quantili"][f"q{int(round(100 * b)):02d}"]] for lev, (a, b) in G.FASCE.items()}
        righe.append(r)
        base = float(c_[-1]) if h == 1 else righe[0]["chiusura"]
        k = M.candela_prevista(base, comp, est["massimo"], est["minimo"])
        direz, testo = M.etichetta(p_su)
        su = k["chiusura"] >= k["apertura"]
        candele.append(k | {"h": h, "versione": versione, "costruzione": COSTRUZIONE, "modello_direzione": "L0", "p_rialzo": comp["p_su"], "direzione": "rialzo" if su else "ribasso",
                            "prob_direzione": comp["p_su"] if su else 1 - comp["p_su"], "p_chiusura_su": float(p_su), "direzione_chiusura": direz, "etichetta": testo, "colore": direz,
                            "chiusura_centrale": r["chiusura"], "rapporto_apertura": comp["rapporto_apertura"],
                            "insieme_massimo": scelte["estremi"][versione]["massimo"][str(h)], "insieme_minimo": scelte["estremi"][versione]["minimo"][str(h)]})
    prev = {"versione": versione, "data_origine": str(data0.date()), "indice_origine": int(t), "chiusura_origine": float(c_[-1]), "settimane_di_storia": int(t + 1),
            "vix_origine": None if not np.isfinite(d["vix"].iloc[-1]) else float(d["vix"].iloc[-1]), "settimane": righe}
    return prev, candele


def configurazione(versione: str, scelte: dict) -> dict:
    """Parametri, finestre e regole della versione (senza data e commit: li aggiunge `congela`)."""
    return {"versione": versione, "nome": NOMI[versione], "indice": "VIX9D (Cboe, volatilità a 9 giorni), candele settimanali",
            "dati": ["VIX9D settimanale (apertura, massimo, minimo, chiusura)"] + (["VIX settimanale (chiusura)"] if usa_vix(scelte, versione) else []),
            "orizzonti_settimane": list(G.ORIZZONTI), "finestra": "espandibile, ristima a ogni origine", "min_settimane_di_storia": G.MIN_SETTIMANE, "min_coppie_valide": G.MIN_COPPIE,
            "settimane_senza_candela": "prima di metà novembre 2013 il file Cboe ha solo le chiusure: quelle settimane entrano nei modelli della chiusura e della direzione, non in estremi, apertura e corpo",
            "chiusura": {"modello_per_orizzonte": scelte["chiusura"][versione], "candidati": {k: ["x_t", "media4", "media12"] + list(v) for k, v in M.CHIUSURA.items()},
                         "centro": "regressione diretta ai minimi quadrati di x_{t+h}, x = log della chiusura",
                         "chiusura_centrale_e_fasce": "regressione quantilica sulle stesse variabili; quantili " + ", ".join(str(int(round(100 * a))) for a in G.TAUS) + "; riordinati se incrociati"},
            "fasce": {str(k): list(v) for k, v in G.FASCE.items()},
            "probabilita_soglie": {"soglie": list(G.SOGLIE), "metodo": "centro + errori passati dello stesso modello con esito noto", "min_errori": G.MIN_ERRORI},
            "direzione": {"modello_per_orizzonte": scelte["direzione"][versione], "evento": "chiusura a t + h sopra la chiusura all'origine", "stima": "logistica con penalità L2 = 1 su variabili standardizzate",
                          "variabili": {k: ["x_t", "media4", "media12"] + list(v) for k, v in M.DIREZIONE.items()}, "neutra_tra": list(G.NEUTRA)},
            "estremi": {"insieme_per_estremo_e_orizzonte": scelte["estremi"][versione], "obiettivo": "ln(estremo della settimana t + h / chiusura all'origine)", "stima": "regressione sulla mediana",
                        "insiemi": {k: ["x_t", "media4", "media12"] + list(v) for k, v in M.ESTREMI.items()}},
            "candele": {"apertura": "chiusura precedente (per +2: chiusura centrale di +1) x rapporto mediano storico di apertura / chiusura precedente",
                        "corpo": {"direzione": "frequenza storica delle settimane con chiusura sopra l'apertura; rialzo se >= 0,5", "ampiezza": "media di |ln(chiusura / apertura)|", "finestra_settimane": G.FINESTRA_CORPO},
                        "estremi": "massimo e minimo previsti comandano: il corpo è tagliato dentro l'intervallo", "colore": "secondo l'etichetta della chiusura; grigio se neutra", "nome_nel_registro": COSTRUZIONE},
            "ripiego": None if not usa_vix(scelte, versione) else "se il VIX della settimana di origine manca, si usa G1 e lo si segnala",
            "file_sorgenti_congelati": list(SORGENTI_CONGELATI)}
