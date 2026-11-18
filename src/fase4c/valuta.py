"""Fase 4c: misure di sviluppo su massimo e minimo della prima candela (settimana +1) e regole di scelta. Parti A-D.

Origini: le stesse della prima candela nelle Fasi 4 e 4b (lette da reports/fase4b/previsioni_dev.csv, che porta anche
la candela attuale del programma, le etichette e le mosse medie). Qui si calcolano, sulla storia fino a ogni origine,
le previsioni dirette di massimo, minimo e chiusura per gli insiemi S0-S5, la previsione incrociata B3 e il tetto
teorico B2. Legge solo dati di sviluppo (VIX settimanale e dati giornalieri esterni), sotto la guardia.

Precisazioni fissate prima di guardare i risultati:
- ogni confronto e' sulle stesse origini; i campioni sono: tutte le origini, le origini con VVIX, le origini con VIX3M
  e le origini comuni a tutti gli insiemi;
- «settimana di strappo» = chiusura reale della settimana +1 oltre +20% sulla chiusura all'origine. E' un'etichetta
  nota solo A POSTERIORI: serve a leggere i risultati, non e' usabile per prevedere;
- terzili del VVIX: calcolati sulle origini del campione in cui il VVIX esiste (anche questi a posteriori);
- regola della Parte A, per ciascun estremo: tra gli insiemi significativamente migliori di S0 (Diebold-Mariano
  sull'errore assoluto in log, p < 0,05 dopo Holm) e con quota entro il 5% non piu' bassa di S0, quello con l'errore
  assoluto in log piu' basso; se nessuno, S0; a parita' l'insieme con meno variabili. Per F1 la regola confronta S1
  con S0 su tutte le origini; per FS confronta S1-S5 e B3 con S0 sulle origini comuni a tutti;
- regola della Parte C: K5 se l'errore assoluto sulla chiusura della candela e' significativamente piu' basso
  (DM, p < 0,05) e la sovrapposizione non e' significativamente piu' bassa (DM, p < 0,05); altrimenti K2. Applicata a
  F1 su tutte le origini, come nella Fase 4b; la stessa scelta vale per FS.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import baselines as B
from .. import config as C
from ..fase1 import comune as K
from ..fase1.valuta import holm
from ..fase2 import valuta as V2
from ..fase2b import dati, grafici
from ..fase2b import valuta as VB
from ..fase2c import valuta as VC
from ..fase3 import dati as dati3
from . import candela as KC
from . import estremi as ES
from . import incrociata as IN
from . import tetto as TT
from . import variabili as V
from . import versioni as VS

OUT = C.REPORTS_DIR / "fase4c"
FIG = C.FIG_DIR / "fase4c"
F4B = C.REPORTS_DIR / "fase4b"
ALFA = 0.05
STRAPPO = 0.20
INSIEMI = tuple(V.INSIEMI)
IN_REGOLA = INSIEMI[1:] + ("B3",)
MODELLI = INSIEMI + ("B3", "B2")
BERSAGLI = {"massimo": "high", "minimo": "low", "chiusura": "close"}
N_VARIABILI = {s: len(V.INSIEMI[s]) for s in INSIEMI} | {"B3": len(V.S1) + 2 + len(IN.PRED_REND) + len(IN.PRED_ESC) + 2}
NOMI = V.DESCRIZIONI | {"B3": "B3 previsione incrociata (S&P 500 previsto)", "B2": "B2 TETTO TEORICO (S&P 500 futuro, non utilizzabile)", "FISSA": "regola fissa", "PROG": "programma attuale (Fase 4b)"}
FASCE = list(C.BAND_LABELS)


# ------------------------------------------------------------------ previsioni

def previsioni(df: pd.DataFrame, VV: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """Per ogni origine: massimo, minimo e chiusura della settimana +1 per S0-S5, B3, B2 e regola fissa; S&P 500 previsto e reale."""
    close = df["close"].to_numpy(dtype=float)
    x = np.log(close)
    liv = {b: df[c].to_numpy(dtype=float) for b, c in BERSAGLI.items()}
    Z = {s: VV[list(V.INSIEMI[s])].to_numpy() for s in INSIEMI}
    rend, esc = VV["s3_rend_spx"].to_numpy(), VV["s3_escursione_spx"].to_numpy()
    rows = []
    for t in idx.tolist():
        xt, VVt = x[: t + 1], VV.iloc[: t + 1]
        r = {"idx_origine": t}
        rh, eh = IN.prevedi_spx(xt, VVt)
        a = max(1, t - ES.FINESTRA + 1)
        r |= {"spx_rend_previsto": rh, "spx_esc_prevista": eh, "spx_rend_reale": rend[t + 1], "spx_esc_reale": esc[t + 1],
              "spx_rend_media": float(np.nanmean(rend[a:t + 1])), "spx_esc_media": float(np.nanmean(esc[a:t + 1]))}
        for b in BERSAGLI:
            L = liv[b][: t + 1]
            for s in INSIEMI:
                r[f"{s}_{b}"] = ES.mediana_diretta(xt, L, Z[s][: t + 1])
            r[f"B3_{b}"] = IN.b3(xt, L, VVt)
            r[f"B2_{b}"] = TT.b2_tetto_teorico(xt, L, VVt, (rend[t + 1], esc[t + 1]))      # TETTO TEORICO: usa l'S&P 500 della settimana t + 1
            r[f"FISSA_{b}"] = ES.regola_fissa(close[: t + 1], L)
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ misure

def metriche(p: np.ndarray, r: np.ndarray) -> dict:
    e = p / r - 1
    return {"n": int(len(p)), "entro_5": float((np.abs(e) <= 0.05).mean()), "entro_10": float((np.abs(e) <= 0.10).mean()),
            "errore_pct_mediano": float(np.median(np.abs(e))), "errore_pct_medio": float(np.abs(e).mean()),
            "errore_punti_mediano": float(np.median(np.abs(p - r))), "errore_punti_medio": float(np.abs(p - r).mean()),
            "oltre_5_per_eccesso": float((e > 0.05).mean()), "oltre_5_per_difetto": float((e < -0.05).mean()), "errore_log_medio": float(np.abs(np.log(p / r)).mean())}


def gruppi(f: pd.DataFrame) -> dict[str, np.ndarray]:
    g = {"tutte": np.ones(len(f), bool), "senza strappi": ~f["strappo"].to_numpy(bool)}
    g |= {f"fascia {b}": (f["fascia_origine"] == b).to_numpy() for b in FASCE}
    g["fascia 13-22"] = f["fascia_13_22"].to_numpy(bool)
    vv = f["ln_vvix"].to_numpy()
    ok = np.isfinite(vv)
    if ok.sum() >= 30:                                   # terzili sulle origini del campione in cui il VVIX esiste (a posteriori)
        q1, q2 = np.quantile(vv[ok], [1 / 3, 2 / 3])
        g |= {"VVIX terzile basso": ok & (vv <= q1), "VVIX terzile medio": ok & (vv > q1) & (vv <= q2), "VVIX terzile alto": ok & (vv > q2)}
    return g


def campioni(f: pd.DataFrame) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Insiemi di origini e, per ciascuno, i modelli che esistono su tutte le sue origini."""
    ok = {m: f[[f"{m}_massimo", f"{m}_minimo", f"{m}_chiusura"]].notna().all(axis=1).to_numpy() for m in MODELLI}
    masks = {"tutte le origini": ok["S0"], "origini con VVIX": ok["S2"], "origini con VIX3M": ok["S4"], "origini comuni": np.logical_and.reduce([ok[m] for m in MODELLI])}
    return {nome: (m, [k for k in MODELLI if ok[k][m].all()]) for nome, m in masks.items()}


def tabella_misure(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for camp, (mc, modelli) in campioni(f).items():
        fc = f[mc].reset_index(drop=True)
        for gruppo, mg in gruppi(fc).items():
            g = fc[mg]
            if len(g) == 0:
                continue
            for m in modelli + ["FISSA", "PROG"]:
                for b in BERSAGLI:
                    if f"{m}_{b}" not in g.columns:
                        continue
                    r = {"campione": camp, "gruppo": gruppo, "modello": m, "bersaglio": b} | metriche(g[f"{m}_{b}"].to_numpy(), g[f"{b}_reale"].to_numpy())
                    if b != "chiusura":
                        e = {k: np.abs(g[f"{m}_{k}"].to_numpy() / g[f"{k}_reale"].to_numpy() - 1) <= 0.05 for k in ES.ESTREMI}
                        r["entrambi_entro_5"] = float((e["massimo"] & e["minimo"]).mean())
                    rows.append(r)
    return pd.DataFrame(rows)


def confronti(f: pd.DataFrame, candidati: tuple[str, ...], tag: str) -> pd.DataFrame:
    """Ogni candidato contro S0 sulle origini date: errore assoluto in log, Diebold-Mariano, Holm per bersaglio."""
    rows = []
    bl = K.blocks(f["idx_origine"].to_numpy())
    for b in BERSAGLI:
        r = np.log(f[f"{b}_reale"].to_numpy())
        e0 = np.abs(np.log(f[f"S0_{b}"].to_numpy()) - r)
        q0 = float((np.abs(f[f"S0_{b}"].to_numpy() / f[f"{b}_reale"].to_numpy() - 1) <= 0.05).mean())
        for m in candidati:
            e = np.abs(np.log(f[f"{m}_{b}"].to_numpy()) - r)
            dm, p = V2.dm_loss(e0, e, 1)                 # positivo = il candidato ha errore minore di S0
            lo, hi = K.boot_cluster(bl, lambda i: 1 - e[i].mean() / e0[i].mean(), f"f4c|{tag}|{b}|{m}")
            q = float((np.abs(f[f"{m}_{b}"].to_numpy() / f[f"{b}_reale"].to_numpy() - 1) <= 0.05).mean())
            rows.append({"bersaglio": b, "modello": m, "n": len(f), "n_variabili": N_VARIABILI.get(m, np.nan), "errore_log": e.mean(), "errore_log_s0": e0.mean(),
                         "guadagno": 1 - e.mean() / e0.mean(), "guadagno_ci_lo": lo, "guadagno_ci_hi": hi, "dm_stat": dm, "p": p, "entro_5": q, "entro_5_s0": q0})
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for b in BERSAGLI:
        m = t["bersaglio"] == b
        t.loc[m, "p_holm"] = holm(t.loc[m, "p"].to_numpy())
    t["ammesso"] = (t["p_holm"] < ALFA) & (t["dm_stat"] > 0) & (t["entro_5"] >= t["entro_5_s0"])
    return t


def scegli_insieme(t: pd.DataFrame, bersaglio: str) -> str:
    """Regola della Parte A per un estremo: errore piu' basso tra gli ammessi; se nessuno, S0; a parita' meno variabili."""
    g = t[(t["bersaglio"] == bersaglio) & t["ammesso"]]
    if g.empty:
        return "S0"
    g = g.assign(err=g["errore_log"].round(12)).sort_values(["err", "n_variabili"], kind="stable")
    return str(g["modello"].iloc[0])


# ------------------------------------------------------------------ Parte B

def spx_prevedibilita(f: pd.DataFrame) -> dict:
    g = f[f[["spx_rend_previsto", "spx_esc_prevista", "spx_rend_reale", "spx_esc_reale"]].notna().all(axis=1)]
    rr, rp, er, ep = (g[c].to_numpy() for c in ("spx_rend_reale", "spx_rend_previsto", "spx_esc_reale", "spx_esc_prevista"))
    return {"n": int(len(g)), "direzione_indovinata": float((np.sign(rp) == np.sign(rr)).mean()), "sempre_rialzo": float((rr > 0).mean()), "quota_previsioni_di_rialzo": float((rp > 0).mean()),
            "r2_rendimento": float(1 - ((rr - rp) ** 2).sum() / ((rr - g["spx_rend_media"].to_numpy()) ** 2).sum()),
            "r2_escursione": float(1 - ((er - ep) ** 2).sum() / ((er - g["spx_esc_media"].to_numpy()) ** 2).sum())}


def tabella_incrociata(f: pd.DataFrame, migliori: dict[str, str]) -> pd.DataFrame:
    """Quota entro il 5% (e errore mediano) per S0, il migliore della Parte A, B3 e B2, sulle origini comuni; quanta parte
    del guadagno del tetto teorico sopravvive nella versione reale."""
    rows = []
    for b in BERSAGLI:
        r = {"bersaglio": b, "n": len(f), "migliore_parte_a": migliori[b]}
        for nome, m in (("s0", "S0"), ("migliore_a", migliori[b]), ("b3", "B3"), ("b2_tetto", "B2")):
            mt = metriche(f[f"{m}_{b}"].to_numpy(), f[f"{b}_reale"].to_numpy())
            r |= {f"entro_5_{nome}": mt["entro_5"], f"errore_pct_mediano_{nome}": mt["errore_pct_mediano"]}
        r["guadagno_tetto"] = r["entro_5_b2_tetto"] - r["entro_5_s0"]
        r["guadagno_reale"] = r["entro_5_b3"] - r["entro_5_s0"]
        r["quota_che_sopravvive"] = r["guadagno_reale"] / r["guadagno_tetto"] if r["guadagno_tetto"] > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Parti C e D

def candele(f: pd.DataFrame, col_max: str, col_min: str, ripiego: tuple[str, str] | None = None) -> pd.DataFrame:
    """Aggiunge le candele K2 e K5 comandate dagli estremi previsti (colonne K2c_*, K5c_*). `ripiego`: colonne da usare
    dove gli estremi scelti mancano (come nel programma: l'insieme di F1)."""
    f = f.copy()
    hi, lo = f[col_max].to_numpy(dtype=float), f[col_min].to_numpy(dtype=float)
    manca = ~(np.isfinite(hi) & np.isfinite(lo))
    if ripiego is not None:
        hi, lo = np.where(manca, f[ripiego[0]].to_numpy(dtype=float), hi), np.where(manca, f[ripiego[1]].to_numpy(dtype=float), lo)
    f["estremi_da_ripiego"] = manca
    for corpo in KC.CORPI:
        out = []
        for r, h, l in zip(f.itertuples(), hi, lo):
            ch = r.K2_chiusura if corpo == "K2" else KC.chiusura_k5(r.chiusura_origine, r.etichetta, r.mossa_ribasso, r.mossa_rialzo, r.K2_chiusura)
            out.append(KC.componi(r.K2_apertura, ch, h, l))
        for c in KC.CAMPI + ("apertura_tagliata", "chiusura_tagliata", "estremi_scambiati"):
            f[f"{corpo}c_{c}"] = [k[c] for k in out]
    return f


def _iou(f: pd.DataFrame, k: str) -> np.ndarray:
    hi, lo, rh, rl = (f[c].to_numpy() for c in (f"{k}_massimo", f"{k}_minimo", "massimo_reale", "minimo_reale"))
    return np.maximum(0.0, np.minimum(hi, rh) - np.maximum(lo, rl)) / (np.maximum(hi, rh) - np.minimum(lo, rl))


def scegli_corpo(f: pd.DataFrame) -> dict:
    e2, e5 = (np.abs(f[f"{k}c_chiusura"] - f["chiusura_reale"]).to_numpy() for k in KC.CORPI)
    s2, s5 = _iou(f, "K2c"), _iou(f, "K5c")
    dm_c, p_c = V2.dm_loss(e2, e5, 1)                    # positivo = K5 ha errore minore di K2
    dm_s, p_s = V2.dm_loss(1 - s2, 1 - s5, 1)            # positivo = K5 ha sovrapposizione maggiore di K2
    meglio = bool(dm_c > 0 and p_c < ALFA)
    sov_peggiore = bool(np.isfinite(dm_s) and dm_s < 0 and p_s < ALFA)
    return {"scelta": "K5" if meglio and not sov_peggiore else "K2", "mae_chiusura": {"K2": float(e2.mean()), "K5": float(e5.mean())}, "dm_chiusura": dm_c, "p_chiusura": p_c,
            "chiusura_significativamente_migliore": meglio, "sovrapposizione": {"K2": float(s2.mean()), "K5": float(s5.mean())}, "dm_sovrapposizione": dm_s, "p_sovrapposizione": p_s,
            "sovrapposizione_significativamente_peggiore": sov_peggiore}


def misure_candela(f: pd.DataFrame, k: str) -> dict:
    """Misure della prima candela con prefisso di colonna k (PROG = prima, DOPO = dopo)."""
    r = {"n": len(f)}
    for b in ES.ESTREMI:
        m = metriche(f[f"{k}_{b}"].to_numpy(), f[f"{b}_reale"].to_numpy())
        r |= {f"{b}_{c}": m[c] for c in ("entro_5", "entro_10", "errore_pct_mediano", "errore_punti_medio", "oltre_5_per_eccesso", "oltre_5_per_difetto")}
    e = {b: np.abs(f[f"{k}_{b}"].to_numpy() / f[f"{b}_reale"].to_numpy() - 1) <= 0.05 for b in ES.ESTREMI}
    pg = VC.pagella_riga(f, k)
    return r | {"entrambi_entro_5": float((e["massimo"] & e["minimo"]).mean()), "sovrapposizione": pg["sovrapposizione"], "chiusura_nell_escursione": pg["chiusura_nell_escursione"],
                "chiusura_nel_corpo": pg["chiusura_nel_corpo"], "rapporto_corpo": pg["rapporto_corpo"], "rapporto_escursione": pg["rapporto_escursione"],
                "mae_apertura": pg["mae_apertura"], "mae_massimo": pg["mae_massimo"], "mae_minimo": pg["mae_minimo"], "mae_chiusura": pg["mae_chiusura"],
                "direzione_corpo": pg["direzione_corpo"]}


def pannelli_g3(df: pd.DataFrame, VV: pd.DataFrame, scelte: dict) -> tuple[list[dict], pd.DataFrame]:
    n = len(df)
    out, rows = [], []
    for k in range(VB.G3_N):
        t = n - 1 - VB.G3_PRIMA - VB.G3_PASSO * k
        prev, cand = VS.prevedi_f1(df, t, scelte, VV.iloc[: t + 1])
        vista = prev | {"settimane": [s | {c: x[c] for c in KC.CAMPI} | ({"colore": x["colore"]} if "colore" in x else {}) for s, x in zip(prev["settimane"], cand)]}
        reali = df.iloc[t + 1:t + 5]
        s4, y4, r1, c1 = prev["settimane"][-1], float(reali["close"].iloc[-1]), reali.iloc[0], cand[0]
        d50, d80 = s4["fasce"]["50"][0] <= y4 <= s4["fasce"]["50"][1], s4["fasce"]["80"][0] <= y4 <= s4["fasce"]["80"][1]
        out.append({"previsione": vista, "storia": df.iloc[t - VB.G3_STORIA + 1:t + 1], "reali": reali, "dentro_50": d50, "dentro_80": d80})
        rows.append({"data_origine": prev["data_origine"], "massimo_previsto_1": c1["massimo"], "massimo_reale_1": float(r1["high"]), "minimo_previsto_1": c1["minimo"], "minimo_reale_1": float(r1["low"]),
                     "massimo_entro_5": bool(abs(c1["massimo"] / r1["high"] - 1) <= 0.05), "minimo_entro_5": bool(abs(c1["minimo"] / r1["low"] - 1) <= 0.05), "dentro_50": d50, "dentro_80": d80})
    return out[::-1], pd.DataFrame(rows[::-1])


# ------------------------------------------------------------------ esecuzione

def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df, giorn = dati.load_dev_candele(), dati3.load_giornaliero()
    VV = V.costruisci(df, giorn)
    pos = {pd.Timestamp(d): i for i, d in enumerate(df["date"])}
    p4b = pd.read_csv(F4B / "previsioni_dev.csv", parse_dates=["data_origine", "data_obiettivo"])
    s4b = json.loads((F4B / "info.json").read_text())["scelte"]
    base = p4b[p4b["versione"] == "F1"].reset_index(drop=True)
    base["idx_origine"] = [pos[d] for d in base["data_origine"]]
    f = pd.concat([base, previsioni(df, VV, base["idx_origine"].to_numpy()).drop(columns="idx_origine")], axis=1)
    f["PROG_massimo"], f["PROG_minimo"] = f["DOPO_massimo"], f["DOPO_minimo"]
    for c in KC.CAMPI:
        f[f"PROG_{c}"] = f[f"DOPO_{c}"]
    f["strappo"] = f["chiusura_reale"] > (1 + STRAPPO) * f["chiusura_origine"]
    f["fascia_origine"] = np.array(C.BAND_LABELS)[B.band_index(f["chiusura_origine"])]
    f["fascia_13_22"] = (f["chiusura_origine"] >= 13.0) & (f["chiusura_origine"] <= 22.0)
    f["ln_vvix"] = VV["s2_ln_vvix"].to_numpy()[f["idx_origine"].to_numpy()]

    # Parte A
    ta = tabella_misure(f)
    camp = campioni(f)
    tutte, comuni = f[camp["tutte le origini"][0]].reset_index(drop=True), f[camp["origini comuni"][0]].reset_index(drop=True)
    r_f1 = confronti(tutte, ("S1",), "f1").assign(famiglia="F1: S1 contro S0, tutte le origini")
    r_fs = confronti(comuni, IN_REGOLA, "fs").assign(famiglia="FS: S1-S5 e B3 contro S0, origini comuni")
    tr = pd.concat([r_f1, r_fs], ignore_index=True)
    ins = {"f1": {b: scegli_insieme(r_f1, b) for b in ES.ESTREMI}, "fs": {b: scegli_insieme(r_fs, b) for b in ES.ESTREMI}}
    proprie = []                                         # ogni insieme contro S0 sulle proprie origini (campione piu' lungo disponibile), solo descrizione
    for nome, (m, modelli) in camp.items():
        cand = tuple(k for k in IN_REGOLA if k in modelli)
        if cand:
            proprie.append(confronti(f[m].reset_index(drop=True), cand, f"p|{nome}").assign(campione=nome))
    tp = pd.concat(proprie, ignore_index=True)

    # Parte B
    spx = spx_prevedibilita(comuni) | {"campione": "origini comuni"}
    spx_t = spx_prevedibilita(tutte) | {"campione": "tutte le origini"}
    g_c = r_fs[(r_fs["bersaglio"] == "chiusura") & r_fs["modello"].isin(INSIEMI)].sort_values("errore_log")
    migliori = {b: ins["fs"][b] for b in ES.ESTREMI} | {"chiusura": str(g_c["modello"].iloc[0]) if g_c["errore_log"].iloc[0] < g_c["errore_log_s0"].iloc[0] else "S0"}
    tb = tabella_incrociata(comuni, migliori)
    tb_ns = tabella_incrociata(comuni[~comuni["strappo"]], migliori).assign(gruppo="senza strappi")
    tb = pd.concat([tb.assign(gruppo="tutte"), tb_ns], ignore_index=True)

    # Parte C: corpo, con gli estremi scelti (F1 su tutte le origini; FS sulle origini in cui esiste la sua etichetta)
    f1 = candele(f, f"{ins['f1']['massimo']}_massimo", f"{ins['f1']['minimo']}_minimo")
    sc = scegli_corpo(f1)
    corpo = sc["scelta"]
    fsb = p4b[p4b["versione"] == "FS"].reset_index(drop=True)
    pred = f.set_index("data_origine")
    for c in [k for k in f.columns if k not in fsb.columns]:
        fsb[c] = pred[c].reindex(fsb["data_origine"]).to_numpy()
    fs = candele(fsb, f"{ins['fs']['massimo']}_massimo", f"{ins['fs']['minimo']}_minimo", (f"{ins['f1']['massimo']}_massimo", f"{ins['f1']['minimo']}_minimo"))
    f1c = candele(f[f["data_origine"].isin(fsb["data_origine"])].reset_index(drop=True), f"{ins['f1']['massimo']}_massimo", f"{ins['f1']['minimo']}_minimo")
    tc = []
    for nome, x, campione in (("F1", f1, "tutte le origini"), ("F1", f1c, "origini con l'etichetta di FS"), ("FS", fs, "origini con l'etichetta di FS")):
        for k in KC.CORPI:
            pg = VC.pagella_riga(x, f"{k}c", f"f4c|c|{nome}|{campione}|{k}")
            tc.append({"versione": nome, "campione": campione, "corpo": k} | pg | {"quota_apertura_tagliata": float(x[f"{k}c_apertura_tagliata"].mean()),
                                                                                  "quota_chiusura_tagliata": float(x[f"{k}c_chiusura_tagliata"].mean())})
    sc_fs = scegli_corpo(fs)
    per_et = [{"versione": "F1", "etichetta": e, "corpo": k} | VC.pagella_riga(g, f"{k}c") for e, g in f1.groupby("etichetta", sort=True) for k in KC.CORPI]

    # Parte D: prima e dopo
    for x in (f1, f1c, fs):
        for c in KC.CAMPI:
            x[f"DOPO4C_{c}"] = x[f"{corpo}c_{c}"]
    pd_rows = []
    for nome, x, campione in (("F1", f1, "tutte le origini"), ("F1", f1c, "origini con l'etichetta di FS"), ("FS", fs, "origini con l'etichetta di FS")):
        for gruppo, m in (("tutte", np.ones(len(x), bool)), ("senza strappi", ~x["strappo"].to_numpy(bool)), ("fascia 13-22", x["fascia_13_22"].to_numpy(bool))):
            for stato, k in (("prima", "PROG"), ("dopo", "DOPO4C")):
                pd_rows.append({"versione": nome, "campione": campione, "gruppo": gruppo, "stato": stato} | misure_candela(x[m], k))
    tpd = pd.DataFrame(pd_rows)

    # misure mostrate dal programma (per versione e orizzonte): quelle della Fase 4b, con la prima candela aggiornata
    sp = pd.read_csv(F4B / "statistiche_programma.csv")
    P = tpd[(tpd["gruppo"] == "tutte") & (tpd["stato"] == "dopo")].set_index(["versione", "campione"])
    sp["massimo_entro_5"], sp["minimo_entro_5"] = np.nan, np.nan
    for v, chiave in (("f1", ("F1", "tutte le origini")), ("fs", ("FS", "origini con l'etichetta di FS"))):
        m = (sp["versione"] == v) & (sp["h"] == 1)
        for c in ("mae_apertura", "mae_massimo", "mae_minimo", "mae_chiusura", "sovrapposizione", "chiusura_nell_escursione", "rapporto_corpo", "rapporto_escursione"):
            sp.loc[m, c] = P.loc[chiave, c]
        sp.loc[m, "massimo_entro_5"], sp.loc[m, "minimo_entro_5"] = P.loc[chiave, "massimo_entro_5"], P.loc[chiave, "minimo_entro_5"]

    scelte = s4b | {"minimo": "diretto", "insieme_massimo_f1": ins["f1"]["massimo"], "insieme_minimo_f1": ins["f1"]["minimo"], "insieme_massimo_fs": ins["fs"]["massimo"],
                    "insieme_minimo_fs": ins["fs"]["minimo"], "corpo": corpo}
    pan, tg3 = pannelli_g3(df, VV, scelte)
    riga_g3 = (f"Prima candela: massimo entro il 5% dal reale in {int(tg3['massimo_entro_5'].sum())} pannelli su {len(tg3)}, minimo in {int(tg3['minimo_entro_5'].sum())}. "
               f"Chiusura reale a 4 settimane dentro la fascia all'80%: {int(tg3['dentro_80'].sum())} su {len(tg3)}.")
    grafici.g3(pan, FIG / "g3_verifica_passato.png", riga_g3, mediana=True, etichette=True)

    tabs = {"a_misure": ta, "a_regola": tr, "a_confronti_sulle_proprie_origini": tp, "b_incrociata": tb, "b_spx": pd.DataFrame([spx_t, spx]), "c_corpo": pd.DataFrame(tc),
            "c_corpo_per_etichetta": pd.DataFrame(per_et), "prima_e_dopo": tpd, "statistiche_programma": sp, "g3_pannelli": tg3}
    for name, t in tabs.items():
        t.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")
    tieni = ["data_origine", "data_obiettivo", "chiusura_origine", "strappo", "etichetta"] + [f"{b}_reale" for b in KC.CAMPI] + [f"{m}_{b}" for m in MODELLI + ("FISSA",) for b in BERSAGLI] \
        + ["PROG_massimo", "PROG_minimo", "spx_rend_previsto", "spx_esc_prevista", "spx_rend_reale", "spx_esc_reale"] + [f"{k}c_{c}" for k in KC.CORPI for c in KC.CAMPI]
    f1[tieni].to_csv(OUT / "previsioni_dev.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    n_c = {nome: {"n": int(m.sum()), "prima_data_origine": str(f.loc[m, "data_origine"].iloc[0].date()), "modelli": modelli} for nome, (m, modelli) in camp.items()}
    info = {"scelte": scelte, "prima_data_obiettivo": str(f["data_obiettivo"].iloc[0].date()), "ultima_data_obiettivo": str(f["data_obiettivo"].iloc[-1].date()), "insiemi": ins, "parte_c": sc, "parte_c_fs": sc_fs, "migliori_per_tabella_b": migliori, "strappo_soglia": STRAPPO,
            "n_strappi": {"tutte le origini": int(tutte["strappo"].sum()), "origini comuni": int(comuni["strappo"].sum())}, "campioni": n_c,
            "n_etichetta_fs": int(len(fs)), "prima_origine_etichetta_fs": str(fs["data_origine"].iloc[0].date()), "ripieghi_estremi_fs": int(fs["estremi_da_ripiego"].sum()),
            "tagli": {nome: {k: {"apertura": float(x[f"{k}c_apertura_tagliata"].mean()), "chiusura": float(x[f"{k}c_chiusura_tagliata"].mean()), "estremi_scambiati": int(x[f"{k}c_estremi_scambiati"].sum())}
                             for k in KC.CORPI} for nome, x in (("F1", f1), ("FS", fs))},
            "spx": {"tutte le origini": spx_t, "origini comuni": spx}, "finestra": ES.FINESTRA, "minimo_coppie": ES.MINIMO_COPPIE, "n_variabili": N_VARIABILI,
            "costruzione_nel_registro": VS.nome_costruzione(scelte),
            "serie_fs": sorted(set(VS.serie_richieste(ins["fs"]["massimo"])) | set(VS.serie_richieste(ins["fs"]["minimo"])) | ({"vix", "vix3m"} if "f2" in VS.CANDIDATI[scelte["candidato"]] else set())),
            "g3": {"riga": riga_g3, "n": int(len(tg3)), "regola": f"{VB.G3_N} origini, una ogni {VB.G3_PASSO} settimane a ritroso a partire da {VB.G3_PRIMA} settimane prima dell'ultima data di sviluppo"}}
    (OUT / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with K.guardia_dati():
        run()
