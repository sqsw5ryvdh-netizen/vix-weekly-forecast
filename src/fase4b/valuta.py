"""Fase 4b: misure di sviluppo sulla prima candela (settimana +1) e regole di scelta. Parti A, B, C, D.

Non ricalcola i modelli delle versioni: probabilita', chiusure centrali e candele K2 vengono dai file della Fase 4
(reports/fase4/c_previsioni_dev.csv e a_previsioni_dev.csv, prodotti da `run_phase4`). Qui si calcolano solo i
componenti nuovi (mosse medie d e u, minimo M1q) sulla storia fino a ogni origine. Legge il file di sviluppo sotto la guardia.

Regole fissate prima (dalla consegna; dove serviva una precisazione e' indicata):
A. candidati P1-P5 per la probabilita' che la chiusura +1 superi quella dell'origine, sulle origini in cui esistono
   tutti; test di Diebold-Mariano sul Brier contro P1, Holm sui quattro confronti.
B. sfidante = candidato con il Brier piu' basso tra P2, P3, P4, P5 (a parita' il primo dell'elenco).
C. K4 al posto di K2 per la settimana +1 se la sovrapposizione media non e' piu' bassa e l'errore assoluto sulla
   chiusura della candela non e' significativamente peggiore (DM, p < 0,05). La regola e' applicata a F1 su tutte le
   sue origini (il campione piu' lungo); la stessa costruzione vale poi per FS.
D. M1q al posto di M0 se l'errore assoluto medio sul minimo e' piu' basso con DM p < 0,05. M0 = minimo della candela
   scelta in C; M1q e' valutato come viene mostrato (non sopra il corpo). Anche questa regola e' applicata a F1.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import baselines as B
from .. import config as C
from ..fase1 import comune as K
from ..fase2 import valuta as V2
from ..fase2b import dati, grafici
from ..fase2b import valuta as VB
from ..fase2c import valuta as VC
from ..fase3 import candele as K3
from ..fase4 import candele as KA
from . import prima as PR
from . import versioni as VS

OUT = C.REPORTS_DIR / "fase4b"
FIG = C.FIG_DIR / "fase4b"
F4 = C.REPORTS_DIR / "fase4"
ALFA = 0.05
FASCE = list(C.BAND_LABELS)
NOMI_LIV = {"q25": "50", "q75": "50", "q10": "80", "q90": "80", "q05": "90", "q95": "90"}


# ------------------------------------------------------------------ Parte A e B

def candidati(c1: pd.DataFrame) -> pd.DataFrame:
    """Probabilita' dei cinque candidati sulle origini in cui esistono tutte (h = 1)."""
    c = c1[c1[["F1_p_su", "F2_p_su", "F3_p_su"]].notna().all(axis=1)].reset_index(drop=True).copy()
    for nome, membri in VS.CANDIDATI.items():
        c[nome] = c[[f"{m.upper()}_p_su" for m in membri]].mean(axis=1)
    c["fascia_origine"] = np.array(C.BAND_LABELS)[B.band_index(c["chiusura_origine"])]
    return c


def etichetta_metriche(p: np.ndarray, y: np.ndarray) -> dict:
    su, giu = p >= K3.NEUTRA[1], p <= K3.NEUTRA[0]
    nn = su | giu
    ok = np.where(su, y == 1, y == 0)
    return {"n": int(len(p)), "frequenza_rialzo": float(y.mean()), "quota_non_neutre": float(nn.mean()), "n_non_neutre": int(nn.sum()),
            "accuratezza_non_neutre": float(ok[nn].mean()) if nn.any() else np.nan, "n_rialzo": int(su.sum()), "accuratezza_rialzo": float((y[su] == 1).mean()) if su.any() else np.nan,
            "n_ribasso": int(giu.sum()), "accuratezza_ribasso": float((y[giu] == 0).mean()) if giu.any() else np.nan} | V2.brier_parts(p, y)


def parte_a(c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gruppi = [("tutte", np.ones(len(c), bool))] + [(f"fascia {b}", (c["fascia_origine"] == b).to_numpy()) for b in FASCE] + [("fascia 13-22", c["fascia_13_22"].to_numpy(bool))]
    y, bl = c["su"].to_numpy(), K.blocks(c["idx_origine"].to_numpy())
    rows, cal = [], []
    for gruppo, m in gruppi:
        if m.sum() == 0:
            continue
        for nome in VS.CANDIDATI:
            p, p1 = c[nome].to_numpy(), c["P1"].to_numpy()
            r = {"gruppo": gruppo, "candidato": nome} | etichetta_metriche(p[m], y[m])
            if nome != "P1":
                l, l1 = (p - y) ** 2, (p1 - y) ** 2
                r["brier_skill_vs_p1"] = 1 - l[m].mean() / l1[m].mean()
                r["dm_stat"], r["p"] = V2.dm_loss(l1[m], l[m], 1)
                if gruppo in ("tutte", "fascia 13-22"):
                    idx = np.flatnonzero(m)
                    r["skill_ci_lo"], r["skill_ci_hi"] = K.boot_cluster(bl[idx], lambda i: 1 - l[idx][i].mean() / l1[idx][i].mean(), f"f4b|a|{gruppo}|{nome}")
            rows.append(r)
            if gruppo == "tutte":
                b = np.minimum((p * 10).astype(int), 9)
                for k in np.unique(b):
                    cal.append({"candidato": nome, "classe": f"{k / 10:.1f}-{(k + 1) / 10:.1f}", "n": int((b == k).sum()), "prob_media": p[b == k].mean(), "freq_osservata": y[b == k].mean()})
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    tutte = (t["gruppo"] == "tutte") & t["p"].notna()
    from ..fase1.valuta import holm
    t.loc[tutte, "p_holm"] = holm(t.loc[tutte, "p"].to_numpy())
    return t, pd.DataFrame(cal)


def scegli_candidato(t: pd.DataFrame) -> dict:
    """Parte B: Brier piu' basso a h = 1 tra P2, P3, P4, P5 (a parita' il primo dell'elenco)."""
    g = t[t["gruppo"] == "tutte"].set_index("candidato")
    br = {k: float(g.loc[k, "brier"]) for k in VS.CANDIDATI}
    scelto = min(VS.SFIDANTI, key=lambda k: br[k])
    return {"candidato": scelto, "brier": br, "base": VS.base_di(scelto), "componenti": list(VS.CANDIDATI[scelto]),
            "significativo_vs_p1_dopo_holm": bool(g.loc[scelto, "p_holm"] < ALFA and g.loc[scelto, "dm_stat"] > 0)}


# ------------------------------------------------------------------ Parti C e D

def componenti_nuovi(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    """Per ogni origine: mosse medie d e u e minimo M1q (non vincolato), calcolati sulla sola storia fino all'origine."""
    x, low = np.log(df["close"].to_numpy(dtype=float)), df["low"].to_numpy(dtype=float)
    rows = []
    for t in idx.tolist():
        d, u = PR.mosse(x[: t + 1])
        rows.append({"mossa_ribasso": d, "mossa_rialzo": u, "m1q": PR.minimo_m1q(x[: t + 1], low[: t + 1])})
    return pd.DataFrame(rows)


def candele(f: pd.DataFrame, p: np.ndarray, centrale: np.ndarray, v: str) -> pd.DataFrame:
    """Aggiunge a `f` le candele +1: K2 (attuale, variante v) e K4 con l'etichetta data da `p` e la chiusura centrale data."""
    f = f.copy()
    f["p"] = p
    f["etichetta"] = [K3.etichetta(q)[0] for q in p]
    for c in KA.CAMPI:
        f[f"K2_{c}"] = f[f"{v}_{c}"]
    k4 = [PR.candela_k4(r.chiusura_origine, getattr(r, f"{v}_rapporto_apertura"), r.etichetta, r.mossa_ribasso, r.mossa_rialzo, cen, getattr(r, f"{v}_ombra_sup"), getattr(r, f"{v}_ombra_inf"))
          for r, cen in zip(f.itertuples(), centrale)]
    for c in KA.CAMPI:
        f[f"K4_{c}"] = [k[c] for k in k4]
    return f


def pagella(f: pd.DataFrame, k: str, tag: str | None = None) -> dict:
    r = VC.pagella_riga(f, k, tag)
    r["mediana_errore_chiusura"] = float(np.median(np.abs(f[f"{k}_chiusura"] - f["chiusura_reale"])))
    return r


def scegli_costruzione(f: pd.DataFrame, p2: dict, p4: dict) -> dict:
    e2, e4 = np.abs(f["K2_chiusura"] - f["chiusura_reale"]).to_numpy(), np.abs(f["K4_chiusura"] - f["chiusura_reale"]).to_numpy()
    dm, p = V2.dm_loss(e2, e4, 1)                                 # positivo = K4 ha errore minore di K2
    sov_ok, peggiore = bool(p4["sovrapposizione"] >= p2["sovrapposizione"]), bool(dm < 0 and p < ALFA)
    return {"scelta": "K4" if sov_ok and not peggiore else "K2", "sovrapposizione": {"K2": p2["sovrapposizione"], "K4": p4["sovrapposizione"]}, "sovrapposizione_non_piu_bassa": sov_ok,
            "mae_chiusura": {"K2": float(e2.mean()), "K4": float(e4.mean())}, "dm_stat": dm, "p": p, "chiusura_significativamente_peggiore": peggiore}


def minimi(f: pd.DataFrame, ks: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Parte D: M0 = minimo della candela scelta; M1q = regressione sulla mediana, non sopra il corpo."""
    fondo = np.minimum(f[f"{ks}_apertura"], f[f"{ks}_chiusura"]).to_numpy()
    reale = f["minimo_reale"].to_numpy()
    serie = {"M0": f[f"{ks}_minimo"].to_numpy(), "M1q": np.minimum(f["m1q"].to_numpy(), fondo), "M1q senza vincolo": f["m1q"].to_numpy()}
    rows = []
    for nome, v in serie.items():
        e = np.abs(v - reale)
        rows.append({"variante": nome, "n": len(f), "errore_assoluto_medio": e.mean(), "errore_assoluto_mediano": float(np.median(e)), "entro_5": float((e / reale <= 0.05).mean()),
                     "entro_10": float((e / reale <= 0.10).mean()), "quota_troppo_bassi": float((v < reale).mean()), "errore_medio_con_segno": float((v - reale).mean()),
                     "quota_limitati_dal_corpo": float((f["m1q"].to_numpy() > fondo).mean()) if nome == "M1q" else np.nan})
    return pd.DataFrame(rows), serie


def scegli_minimo(serie: dict, reale: np.ndarray) -> dict:
    e0, e1 = np.abs(serie["M0"] - reale), np.abs(serie["M1q"] - reale)
    dm, p = V2.dm_loss(e0, e1, 1)                                 # positivo = M1q ha errore minore di M0
    return {"scelta": "M1q" if (e1.mean() < e0.mean() and p < ALFA) else "M0", "mae": {"M0": float(e0.mean()), "M1q": float(e1.mean())}, "dm_stat": dm, "p": p}


def con_scelte(f: pd.DataFrame, ks: str, minimo: str) -> pd.DataFrame:
    """Colonne DOPO_*: la prima candela con le scelte delle Parti C e D."""
    f = f.copy()
    for c in KA.CAMPI:
        f[f"DOPO_{c}"] = f[f"{ks}_{c}"]
    if minimo == "M1q":
        f["DOPO_minimo"] = np.minimum(f["m1q"], np.minimum(f["DOPO_apertura"], f["DOPO_chiusura"]))
    return f


def misure(f: pd.DataFrame, k: str, p_col: str) -> dict:
    """Tutte le misure della prima candela per una versione e uno stato (prima / dopo)."""
    pg = pagella(f, k)
    e = np.abs(f[f"{k}_minimo"] - f["minimo_reale"]).to_numpy()
    et = etichetta_metriche(f[p_col].to_numpy(), f["su"].to_numpy())
    return {"n": len(f), "brier_direzione": et["brier"], "quota_non_neutre": et["quota_non_neutre"], "n_non_neutre": et["n_non_neutre"], "accuratezza_non_neutre": et["accuratezza_non_neutre"],
            "n_rialzo": et["n_rialzo"], "accuratezza_rialzo": et["accuratezza_rialzo"], "n_ribasso": et["n_ribasso"], "accuratezza_ribasso": et["accuratezza_ribasso"],
            "mae_apertura": pg["mae_apertura"], "mae_massimo": pg["mae_massimo"], "mae_minimo": pg["mae_minimo"], "mae_chiusura": pg["mae_chiusura"],
            "mediana_errore_chiusura": pg["mediana_errore_chiusura"], "mediana_errore_minimo": float(np.median(e)),
            "minimo_entro_5": float((e / f["minimo_reale"] <= 0.05).mean()), "minimo_entro_10": float((e / f["minimo_reale"] <= 0.10).mean()),
            "minimo_troppo_basso": float((f[f"{k}_minimo"] < f["minimo_reale"]).mean()), "massimo_reale_sopra_previsto": float((f["massimo_reale"] > f[f"{k}_massimo"]).mean()),
            "sovrapposizione": pg["sovrapposizione"], "chiusura_nel_corpo": pg["chiusura_nel_corpo"], "chiusura_nell_escursione": pg["chiusura_nell_escursione"],
            "direzione_corpo": pg["direzione_corpo"], "rapporto_corpo": pg["rapporto_corpo"], "rapporto_escursione": pg["rapporto_escursione"]}


def pannelli_g3(df: pd.DataFrame, scelte: dict) -> tuple[list[dict], pd.DataFrame]:
    n = len(df)
    out, rows = [], []
    for k in range(VB.G3_N):
        t = n - 1 - VB.G3_PRIMA - VB.G3_PASSO * k
        prev, cand = VS.prevedi_f1(df, t, scelte)
        vista = prev | {"settimane": [s | {c: x[c] for c in KA.CAMPI} | ({"colore": x["colore"]} if "colore" in x else {}) for s, x in zip(prev["settimane"], cand)]}
        reali = df.iloc[t + 1:t + 5]
        s4, y4, c0, r1 = prev["settimane"][-1], float(reali["close"].iloc[-1]), prev["chiusura_origine"], reali.iloc[0]
        d50, d80 = s4["fasce"]["50"][0] <= y4 <= s4["fasce"]["50"][1], s4["fasce"]["80"][0] <= y4 <= s4["fasce"]["80"][1]
        e1 = cand[0]["direzione_chiusura"]
        out.append({"previsione": vista, "storia": df.iloc[t - VB.G3_STORIA + 1:t + 1], "reali": reali, "dentro_50": d50, "dentro_80": d80})
        rows.append({"data_origine": prev["data_origine"], "chiusura_origine": c0, "etichetta_1": e1, "etichetta_1_corretta": (None if e1 == "neutra" else bool((r1["close"] > c0) == (e1 == "rialzo"))),
                     "chiusura_candela_1": cand[0]["chiusura"], "minimo_candela_1": cand[0]["minimo"], "chiusura_reale_1": float(r1["close"]), "minimo_reale_1": float(r1["low"]),
                     "chiusura_reale_4": y4, "dentro_50": d50, "dentro_80": d80})
    return out[::-1], pd.DataFrame(rows[::-1])


# ------------------------------------------------------------------ esecuzione

def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = dati.load_dev_candele()
    pos = {pd.Timestamp(d): i for i, d in enumerate(df["date"])}
    v = json.loads((F4 / "a_info.json").read_text())["scelta"]["scelta"]
    c_all = pd.read_csv(F4 / "c_previsioni_dev.csv", parse_dates=["data_origine", "data_obiettivo"])
    a_all = pd.read_csv(F4 / "a_previsioni_dev.csv", parse_dates=["data_origine", "data_obiettivo"])

    # Parte A e B
    c = c_all[c_all["h"] == 1].copy()
    c["idx_origine"] = [pos[d] for d in c["data_origine"]]
    c = candidati(c)
    ta, cal = parte_a(c)
    sb = scegli_candidato(ta)
    cand, base = sb["candidato"], sb["base"].upper()

    # Parte C: F1 su tutte le sue origini
    f = a_all[a_all["h"] == 1].reset_index(drop=True).copy()
    f["idx_origine"] = [pos[d] for d in f["data_origine"]]
    f = pd.concat([f, componenti_nuovi(df, f["idx_origine"].to_numpy())], axis=1)
    f["su"] = f["chiusura_su_reale"]
    f = candele(f, f["p_chiusura_su"].to_numpy(), f["mediana_d4"].to_numpy(), v)
    f["P1"] = f["p"]
    p2, p4 = pagella(f, "K2", "f4b|c|K2"), pagella(f, "K4", "f4b|c|K4")
    sc = scegli_costruzione(f, p2, p4)
    ks = sc["scelta"]
    tc = [{"versione": "F1", "campione": "tutte le origini", "costruzione": k} | r for k, r in (("K2", p2), ("K4", p4))]
    per_et = [{"versione": "F1", "etichetta": e, "costruzione": k} | pagella(g, k) for e, g in f.groupby("etichetta", sort=True) for k in ("K2", "K4")]

    # sfidante sulle origini comuni: etichetta del candidato, chiusura centrale della costruzione di base
    comp = f.set_index("data_origine")
    g = c[["data_origine", "data_obiettivo", "idx_origine", "chiusura_origine", "su", "fascia_origine", "fascia_13_22", "P1", cand, f"{base}_q50"]].copy()
    for col in [k for k in f.columns if k.endswith("_reale") or k.startswith(f"{v}_") or k in ("mossa_ribasso", "mossa_rialzo", "m1q", "mediana_d4")]:
        g[col] = comp[col].reindex(g["data_origine"]).to_numpy()
    fs = candele(g, g[cand].to_numpy(), g[f"{base}_q50"].to_numpy(), v)            # FS
    f1c = candele(g, g["P1"].to_numpy(), g["mediana_d4"].to_numpy(), v)             # F1 sulle stesse origini
    tc += [{"versione": nome, "campione": "origini comuni", "costruzione": k} | pagella(x, k, f"f4b|c|{nome}|{k}") for nome, x in (("F1", f1c), ("FS", fs)) for k in ("K2", "K4")]

    # Parte D
    td, serie = minimi(f, ks)
    sd = scegli_minimo(serie, f["minimo_reale"].to_numpy())
    td.insert(0, "versione", "F1")
    td.insert(1, "campione", "tutte le origini")
    altra = "K4" if ks == "K2" else "K2"                              # solo informazione: gli stessi minimi se la Parte C avesse scelto l'altra costruzione
    t_alt, s_alt = minimi(f, altra)
    t_alt.insert(0, "versione", "F1")
    t_alt.insert(1, "campione", f"tutte le origini, con {altra} (non scelta)")
    td = pd.concat([td, t_alt], ignore_index=True)
    for nome, x in (("F1", f1c), ("FS", fs)):
        t2, _ = minimi(x, ks)
        t2.insert(0, "versione", nome)
        t2.insert(1, "campione", "origini comuni")
        td = pd.concat([td, t2], ignore_index=True)

    # prima e dopo
    f, f1c, fs = (con_scelte(x, ks, sd["scelta"]) for x in (f, f1c, fs))
    righe = [{"campione": "tutte le origini", "versione": "F1", "stato": "prima"} | misure(f, "K2", "P1"), {"campione": "tutte le origini", "versione": "F1", "stato": "dopo"} | misure(f, "DOPO", "P1"),
             {"campione": "origini comuni", "versione": "F1", "stato": "prima"} | misure(f1c, "K2", "P1"), {"campione": "origini comuni", "versione": "F1", "stato": "dopo"} | misure(f1c, "DOPO", "P1"),
             {"campione": "origini comuni", "versione": base, "stato": "prima"} | misure(fs.assign(pb=c[f"{base}_p_su"].to_numpy()), "K2", "pb"),
             {"campione": "origini comuni", "versione": "FS", "stato": "dopo"} | misure(fs, "DOPO", cand)]
    for nome, x, pc in (("F1", f1c, "P1"), ("FS", fs, cand)):
        m13 = x[x["fascia_13_22"].astype(bool)]
        righe.append({"campione": "origini comuni, fascia 13-22", "versione": nome, "stato": "dopo"} | misure(m13, "DOPO", pc))
    pd_tab = pd.DataFrame(righe)

    # misure mostrate dal programma accanto alla previsione (per versione e orizzonte)
    ch = pd.read_csv(C.REPORTS_DIR / "fase2b" / "chiusura_metriche.csv").set_index("h")
    pg4 = pd.read_csv(F4 / "a_pagella.csv")
    pg4 = pg4[pg4["variante"] == v].set_index("h")
    et3 = pd.read_csv(C.REPORTS_DIR / "fase3" / "p0_etichetta.csv").set_index("h")
    cf = pd.read_csv(F4 / "c_confronto.csv")
    cf = cf[cf["gruppo"] == "tutte"].set_index(["versione", "h"])
    P = pd_tab.set_index(["campione", "versione", "stato"])
    campi_pg = ["mae_apertura", "mae_massimo", "mae_minimo", "mae_chiusura", "sovrapposizione", "chiusura_nell_escursione", "rapporto_corpo", "rapporto_escursione"]
    stat = []
    for h in (1, 2, 3, 4):
        r1 = {"versione": "f1", "h": h, "n": int(ch.loc[h, "n"]), "campione": "tutte le origini di sviluppo dal 2006", "mediana_mae_vix": ch.loc[h, "mediana_mae_vix"]} | {k: ch.loc[h, k] for k in ("copertura_50", "copertura_80", "copertura_90")}
        r1 |= {k: (P.loc[("tutte le origini", "F1", "dopo"), k] if h == 1 else pg4.loc[h, k]) for k in campi_pg}
        r1 |= {k: (1 - et3.loc[h, "quota_neutra"] if k == "quota_non_neutre" else et3.loc[h, k]) for k in ("quota_non_neutre", "n_non_neutre", "accuratezza_non_neutre")}
        rs = {"versione": "fs", "h": h, "n": int(cf.loc[(base, h), "n"]), "campione": "origini di sviluppo dal 2014 (etichetta e prima candela: dal 2016)", "mediana_mae_vix": cf.loc[(base, h), "mae_centrale"]}
        rs |= {k: cf.loc[(base, h), k] for k in ("copertura_50", "copertura_80", "copertura_90")}
        rs |= {k: (P.loc[("origini comuni", "FS", "dopo"), k] if h == 1 else cf.loc[(base, h), k]) for k in campi_pg}
        if h == 1:
            rs |= {k: P.loc[("origini comuni", "FS", "dopo"), k] for k in ("quota_non_neutre", "n_non_neutre", "accuratezza_non_neutre")}
        else:
            rs |= {"quota_non_neutre": 1 - cf.loc[(base, h), "quota_neutra"], "n_non_neutre": cf.loc[(base, h), "n_non_neutre"], "accuratezza_non_neutre": cf.loc[(base, h), "accuratezza_non_neutre"]}
        stat += [r1, rs]

    scelte = {"variante": v, "prima_candela": ks, "minimo": sd["scelta"], "candidato": cand}
    pan, tg3 = pannelli_g3(df, scelte)
    nn = int(tg3["etichetta_1"].ne("neutra").sum())
    okk = int(tg3.loc[tg3["etichetta_1"].ne("neutra"), "etichetta_1_corretta"].astype(bool).sum())
    n50, n80 = int(tg3["dentro_50"].sum()), int(tg3["dentro_80"].sum())
    riga_g3 = (f"Prima candela: etichetta non neutra in {nn} pannelli su {len(tg3)}, corretta in {okk}. "
               f"Chiusura reale a 4 settimane dentro la fascia al 50%: {n50} su {len(tg3)}; dentro la fascia all'80%: {n80} su {len(tg3)}.")
    grafici.g3(pan, FIG / "g3_verifica_passato.png", riga_g3, mediana=True, etichette=True)

    tabs = {"a_candidati": ta, "a_calibrazione": cal, "c_pagella": pd.DataFrame(tc), "c_pagella_per_etichetta": pd.DataFrame(per_et), "d_minimo": td, "prima_e_dopo": pd_tab,
            "statistiche_programma": pd.DataFrame(stat), "g3_pannelli": tg3}
    for name, t in tabs.items():
        t.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")
    tieni = ["data_origine", "data_obiettivo", "chiusura_origine", "su", "p", "etichetta", "mossa_ribasso", "mossa_rialzo", "m1q"] + [f"{k}_{x}" for k in ("K2", "K4", "DOPO") for x in KA.CAMPI] \
        + [f"{x}_reale" for x in KA.CAMPI]
    pd.concat([f.assign(versione="F1", campione="tutte le origini")[["versione", "campione"] + tieni], fs.assign(versione="FS", campione="origini comuni")[["versione", "campione"] + tieni]],
              ignore_index=True).to_csv(OUT / "previsioni_dev.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    info = {"scelte": scelte, "parte_b": sb, "parte_c": sc, "parte_d": sd, "parte_d_con_altra_costruzione": {"costruzione": altra} | scegli_minimo(s_alt, f["minimo_reale"].to_numpy()), "variante_ampiezze": v, "costruzione_nel_registro": VS.nome_costruzione(scelte),
            "campioni": {"tutte_le_origini": {"n": int(len(f)), "prima_data_obiettivo": str(f["data_obiettivo"].iloc[0].date()), "ultima_data_obiettivo": str(f["data_obiettivo"].iloc[-1].date())},
                         "origini_comuni": {"n": int(len(c)), "prima_data_origine": str(c["data_origine"].iloc[0].date()), "ultima_data_origine": str(c["data_origine"].iloc[-1].date()),
                                            "n_fascia_13_22": int(c["fascia_13_22"].sum())}},
            "mosse": {"finestra": PR.FINESTRA_MOSSE, "d_ultima": float(f["mossa_ribasso"].iloc[-1]), "u_ultima": float(f["mossa_rialzo"].iloc[-1]),
                      "d_min_max": [float(f["mossa_ribasso"].min()), float(f["mossa_ribasso"].max())], "u_min_max": [float(f["mossa_rialzo"].min()), float(f["mossa_rialzo"].max())]},
            "etichette_f1": {e: int(k) for e, k in f["etichetta"].value_counts().items()}, "etichette_fs": {e: int(k) for e, k in fs["etichetta"].value_counts().items()},
            "n_test_holm_parte_a": int(ta["p_holm"].notna().sum()),
            "g3": {"riga": riga_g3, "n": int(len(tg3)), "regola": f"{VB.G3_N} origini, una ogni {VB.G3_PASSO} settimane a ritroso a partire da {VB.G3_PRIMA} settimane prima dell'ultima data di sviluppo"}}
    (OUT / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with K.guardia_dati():
        run()
