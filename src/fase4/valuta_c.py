"""Fase 4, Parte C: confronto di sviluppo tra F1, F2 e (se un modello della Parte B ha passato la regola) F3, sulle
stesse origini: quelle ammesse per F2, da giugno 2014.

F1 non viene ricalcolata: chiusura centrale, fasce e probabilita' vengono da reports/fase2b/previsioni_dev.csv,
l'etichetta da reports/fase3/p0_previsioni_dev.csv, le candele (variante scelta nella Parte A) da
reports/fase4/a_previsioni_dev.csv — gli stessi numeri del programma. F2 e F3 vengono calcolate qui in walk-forward
(F3: stesso walk-forward di N1 nella Parte B, con ristima ogni 26 settimane).
L'etichetta di F2 richiede almeno 104 errori passati attorno al centro M1 + X1: esiste solo da circa due anni dopo
la prima origine; il confronto delle etichette e' fatto sulle origini in cui esiste per tutte le versioni.
Legge solo dati di sviluppo, sotto la guardia.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C
from ..fase1 import comune as K
from ..fase2 import densita as D
from ..fase2 import valuta as V2
from ..fase2b import dati as dati2b
from ..fase2b import grafici
from ..fase2b import modello as MO
from ..fase2b import valuta as VB
from ..fase2c import valuta as VC
from ..fase3 import candele as K3
from ..fase3 import dati as dati3
from ..fase3 import modelli as M
from ..fase3 import valuta_candele as VK3
from . import candele as KA
from . import regimi as RE
from . import valuta_a as VA4
from . import versioni as VS

OUT = C.REPORTS_DIR / "fase4"
FIG = C.FIG_DIR / "fase4"
QUANT = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")
LIVELLI = {"50": ("q25", "q75"), "80": ("q10", "q90"), "90": ("q05", "q95")}
GRUPPI = {"tutte": None, "fascia_13_22": True}


def walk(w: pd.DataFrame, z: np.ndarray, variante: str, con_f3: bool = False) -> dict[int, pd.DataFrame]:
    close, dates = w["close"].to_numpy(), w["date"].to_numpy()
    x = np.log(close)                                  # log esatto della chiusura, come nel programma (non la colonna arrotondata del file)
    Z = z.reshape(-1, 1)
    p2b = pd.read_csv(VB.OUT / "previsioni_dev.csv", parse_dates=["data_origine"])
    p3 = pd.read_csv(VK3.OUT / "p0_previsioni_dev.csv", parse_dates=["data_origine"])
    pa = pd.read_csv(VA4.OUT / "a_previsioni_dev.csv", parse_dates=["data_origine"])
    qi = [int(np.argmin(np.abs(D.TAUS - k))) for k in D.Q_REPORT]
    stato = RE.walk(x, VS.ORIZZONTI)["stato"] if con_f3 else {}
    frames = {}
    for h in VS.ORIZZONTI:
        o, c = VS.centri_f2(x, Z, h)                                       # tutte le origini ammesse per F2 con esito noto
        Q = np.array([M.t3(x[: t + 1].copy(), Z[: t + 1].copy(), h)[0] for t in o.tolist()])
        f = pd.DataFrame({"h": h, "idx_origine": o, "data_origine": dates[o], "data_obiettivo": dates[o + h], "chiusura_origine": close[o], "y": x[o + h]})
        f["F2_centro"] = np.exp(c)
        for k, j in zip(QUANT, qi):
            f[f"F2_{k}"] = np.exp(Q[:, j])
        p_su = np.full(len(o), np.nan)
        err = x[o + h] - c
        for k, t in enumerate(o.tolist()):
            noti = o + h <= t                                              # errori con esito noto all'origine t
            if noti.sum() >= D.MIN_ERR:
                p_su[k] = float((c[k] + err[noti] > x[t]).mean())
        f["F2_p_su"] = p_su
        if con_f3:                                                         # F3: distribuzione prevista da N1 con i parametri e le probabilita' filtrate dell'origine
            righe = []
            for t in o.tolist():
                mix = VS.Mistura(*RE.mistura_all_origine(x[: t + 1], h, *stato[t]))
                righe.append([np.exp(mix.mean), *np.exp(mix.quantiles(np.array(D.Q_REPORT))), mix.prob_above(float(x[t]))])
            righe = np.array(righe)
            f["F3_centro"], f["F3_p_su"] = righe[:, 0], righe[:, -1]
            for j, k in enumerate(QUANT):
                f[f"F3_{k}"] = righe[:, 1 + j]
        a = p2b[p2b["h"] == h].set_index("data_origine").reindex(f["data_origine"])
        f["F1_centro"] = a["m1"].to_numpy()
        for k in QUANT:
            f[f"F1_{k}"] = a["mediana_d4" if k == "q50" else k].to_numpy()
        b = p3[p3["h"] == h].set_index("data_origine").reindex(f["data_origine"])
        f["F1_p_su"] = b["p_chiusura_su"].to_numpy()
        for k in KA.CAMPI + KA.COMPONENTI:
            f[f"{k}_reale"] = b[f"{k}_reale"].to_numpy()
        f["su"] = (f["chiusura_reale"] > f["chiusura_origine"]).astype(float)
        e = pa[pa["h"] == h].set_index("data_origine").reindex(f["data_origine"])
        for k in ("rapporto_apertura", "p_L0") + KA.COMPONENTI:
            f[f"comp_{k}"] = e[f"{variante}_{k}"].to_numpy()
        for k in KA.CAMPI:
            f[f"F1_{k}"] = e[f"{variante}_{k}"].to_numpy()
        f["fascia_13_22"] = (f["chiusura_origine"] >= VA4.FASCIA_EVIDENZA[0]) & (f["chiusura_origine"] <= VA4.FASCIA_EVIDENZA[1])
        assert f[["F1_q50", "F1_p_su", "F1_chiusura", "chiusura_reale"]].notna().all().all(), "origini di F2 non coperte dai file delle fasi precedenti"
        frames[h] = f
    for v in ("F2", "F3") if con_f3 else ("F2",):                          # stesse candele di F1, ancorate alla chiusura centrale della versione
        med = {h: dict(zip(f["idx_origine"].tolist(), f[f"{v}_q50"].tolist())) for h, f in frames.items()}
        for h, f in frames.items():
            cols = {k: [] for k in KA.CAMPI}
            for r in f.to_dict("records"):
                base = r["chiusura_origine"] if h == 1 else med[h - 1][r["idx_origine"]]
                cnd = KA.candela(base, {k: r[f"comp_{k}"] for k in ("rapporto_apertura", "p_L0") + KA.COMPONENTI})
                for k in KA.CAMPI:
                    cols[k].append(cnd[k])
            for k in KA.CAMPI:
                f[f"{v}_{k}"] = cols[k]
    return frames


def _versioni(f: pd.DataFrame) -> tuple[str, ...]:
    return ("F1", "F2", "F3") if "F3_q50" in f.columns else ("F1", "F2")


def _con_etichetta(g: pd.DataFrame) -> pd.DataFrame:
    """Origini in cui la probabilita' della direzione esiste per tutte le versioni."""
    return g[g[[f"{v}_p_su" for v in _versioni(g)]].notna().all(axis=1)]


def _etichetta(p: np.ndarray, y: np.ndarray) -> dict:
    su, giu = p >= K3.NEUTRA[1], p <= K3.NEUTRA[0]
    nn = su | giu
    ok = np.where(su, y == 1, y == 0)
    return {"n_etichette": int(len(p)), "quota_neutra": float((~nn).mean()) if len(p) else np.nan, "n_non_neutre": int(nn.sum()),
            "accuratezza_non_neutre": float(ok[nn].mean()) if nn.any() else np.nan, "brier_direzione": float(((p - y) ** 2).mean()) if len(p) else np.nan}


def riga(g: pd.DataFrame, v: str) -> dict:
    rc, y = g["chiusura_reale"].to_numpy(), g["y"].to_numpy()
    cen, ctr = g[f"{v}_q50"].to_numpy(), g[f"{v}_centro"].to_numpy()
    rel = np.abs(cen / rc - 1)
    r = {"n": len(g), "mae_centrale": float(np.abs(cen - rc).mean()), "rmse_log_centrale": float(np.sqrt(((np.log(cen) - y) ** 2).mean())),
         "entro_10": float((rel <= 0.10).mean()), "entro_20": float((rel <= 0.20).mean()),
         "mae_centro": float(np.abs(ctr - rc).mean()), "rmse_log_centro": float(np.sqrt(((np.log(ctr) - y) ** 2).mean()))}
    for lev, (a, b) in LIVELLI.items():
        lo, hi = g[f"{v}_{a}"].to_numpy(), g[f"{v}_{b}"].to_numpy()
        r |= {f"copertura_{lev}": float(((rc >= lo) & (rc <= hi)).mean()), f"larghezza_{lev}": float((hi - lo).mean())}
    e = _con_etichetta(g)
    r |= _etichetta(e[f"{v}_p_su"].to_numpy(), e["su"].to_numpy())
    p = VC.pagella_riga(g, v)
    return r | {k: p[k] for k in ("rapporto_corpo", "rapporto_escursione", "sovrapposizione", "chiusura_nel_corpo", "chiusura_nell_escursione",
                                  "mae_apertura", "mae_massimo", "mae_minimo", "mae_chiusura")}


def differenze(g: pd.DataFrame, h: int, v: str, tag: str) -> dict:
    """Versione v rispetto a F1 sulle stesse origini: guadagni (positivo = v meglio di F1) con intervallo bootstrap a
    blocchi e test di Diebold-Mariano (due code)."""
    rc, y = g["chiusura_reale"].to_numpy(), g["y"].to_numpy()
    bl = K.blocks(g["idx_origine"].to_numpy())
    a1, a2 = np.abs(g["F1_q50"].to_numpy() - rc), np.abs(g[f"{v}_q50"].to_numpy() - rc)
    s1, s2 = (np.log(g["F1_q50"].to_numpy()) - y) ** 2, (np.log(g[f"{v}_q50"].to_numpy()) - y) ** 2
    r = {"n": len(g), "mae_guadagno": float(a1.mean() - a2.mean()), "r2_centrale_vs_f1": float(1 - s2.sum() / s1.sum())}
    r["mae_ci_lo"], r["mae_ci_hi"] = K.boot_cluster(bl, lambda i: a1[i].mean() - a2[i].mean(), f"f4c|mae|{v}|{tag}")
    r["r2_ci_lo"], r["r2_ci_hi"] = K.boot_cluster(bl, lambda i: 1 - s2[i].sum() / s1[i].sum(), f"f4c|r2|{v}|{tag}")
    r["dm_stat"], r["p"] = V2.dm_loss(s1, s2, h)
    e = _con_etichetta(g)
    if len(e) >= 10:
        b1, b2 = (e["F1_p_su"].to_numpy() - e["su"].to_numpy()) ** 2, (e[f"{v}_p_su"].to_numpy() - e["su"].to_numpy()) ** 2
        ble = K.blocks(e["idx_origine"].to_numpy())
        r |= {"n_etichette": len(e), "brier_skill_vs_f1": float(1 - b2.mean() / b1.mean())}
        r["brier_ci_lo"], r["brier_ci_hi"] = K.boot_cluster(ble, lambda i: 1 - b2[i].mean() / b1[i].mean(), f"f4c|br|{v}|{tag}")
        r["dm_stat_brier"], r["p_brier"] = V2.dm_loss(b1, b2, h)
    return r


def tabelle(frames: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    conf, diff = [], []
    for h, f in frames.items():
        for nome, val in GRUPPI.items():
            g = f if val is None else f[f[nome] == val]
            for v in _versioni(f):
                conf.append({"gruppo": nome, "h": h, "versione": v} | riga(g, v))
                if v != "F1":
                    diff.append({"gruppo": nome, "h": h, "versione": v} | differenze(g, h, v, f"{nome}|{h}"))
    return pd.DataFrame(conf), pd.DataFrame(diff)


def pannelli_g3(df: pd.DataFrame, variante: str) -> tuple[list[dict], pd.DataFrame]:
    """Stesse 12 origini della Fase 2b; candele di F1 con la variante scelta, fasce D4 e linea della mediana."""
    n = len(df)
    out, rows = [], []
    for k in range(VB.G3_N):
        t = n - 1 - VB.G3_PRIMA - VB.G3_PASSO * k
        prev, cand = VS.prevedi_f1(df, t, variante)
        vista = prev | {"settimane": [s | {c: x[c] for c in KA.CAMPI} for s, x in zip(prev["settimane"], cand)]}
        reali = df.iloc[t + 1:t + 5]
        s4, y4, c0 = prev["settimane"][-1], float(reali["close"].iloc[-1]), prev["chiusura_origine"]
        d50, d80 = s4["fasce"]["50"][0] <= y4 <= s4["fasce"]["50"][1], s4["fasce"]["80"][0] <= y4 <= s4["fasce"]["80"][1]
        et = [(x["direzione_chiusura"], (r.close > c0) if x["direzione_chiusura"] == "rialzo" else (r.close <= c0)) for x, r in zip(cand, reali.itertuples())]
        out.append({"previsione": vista, "storia": df.iloc[t - VB.G3_STORIA + 1:t + 1], "reali": reali, "dentro_50": d50, "dentro_80": d80})
        rows.append({"data_origine": prev["data_origine"], "chiusura_origine": c0, "chiusura_centrale_4": s4["chiusura"], "candela_chiusura_4": cand[-1]["chiusura"],
                     "chiusura_reale_4": y4, "dentro_50": d50, "dentro_80": d80, "etichette_non_neutre": int(sum(d != "neutra" for d, _ in et)),
                     "etichette_corrette": int(sum(ok for d, ok in et if d != "neutra"))})
    return out[::-1], pd.DataFrame(rows[::-1])


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    variante = json.loads((OUT / "a_info.json").read_text())["scelta"]["scelta"]
    passa = json.loads((OUT / "b_info.json").read_text())["passa"]
    assert set(passa) <= {"N1"}, "F3 e' definita solo attorno a N1: un altro modello della Parte B ha passato la regola"
    w, g = dati3.load_settimanale(), dati3.load_giornaliero()
    z = VS.x1_settimanale(w["date"], g)
    frames = walk(w, z, variante, con_f3="N1" in passa)
    conf, diff = tabelle(frames)
    conf.to_csv(OUT / "c_confronto.csv", index=False, float_format="%.6g")
    diff.to_csv(OUT / "c_differenze.csv", index=False, float_format="%.6g")
    pd.concat([f.drop(columns="idx_origine") for f in frames.values()], ignore_index=True).to_csv(
        OUT / "c_previsioni_dev.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    df = dati2b.load_dev_candele()
    pan, tg3 = pannelli_g3(df, variante)
    tg3.to_csv(OUT / "c_g3_pannelli.csv", index=False, float_format="%.4f")
    n50, n80, nn, okk = int(tg3["dentro_50"].sum()), int(tg3["dentro_80"].sum()), int(tg3["etichette_non_neutre"].sum()), int(tg3["etichette_corrette"].sum())
    riga_g3 = (f"Chiusura reale a 4 settimane dentro la fascia al 50%: {n50} pannelli su {len(tg3)}; dentro la fascia all'80%: {n80} su {len(tg3)}. "
               f"Etichette della chiusura non neutre: {nn} su {4 * len(tg3)}, corrette {okk}.")
    grafici.g3(pan, FIG / "g3_verifica_passato.png", riga_g3, mediana=True)
    f1 = frames[1]
    et = f1[f1["F2_p_su"].notna()]
    info = {"variante_ampiezze": variante, "versioni": [v.lower() for v in _versioni(f1)], "f3_attorno_a": passa, "x1": VS.X1, "settimane_x1_valide": int(np.isfinite(z).sum()), "prima_settimana_x1": str(w["date"][int(np.argmax(np.isfinite(z)))].date()),
            "per_orizzonte": {str(h): {"n": int(len(f)), "prima_data_origine": str(pd.Timestamp(f["data_origine"].iloc[0]).date()),
                                       "prima_data_obiettivo": str(pd.Timestamp(f["data_obiettivo"].iloc[0]).date()),
                                       "ultima_data_obiettivo": str(pd.Timestamp(f["data_obiettivo"].iloc[-1]).date()),
                                       "n_fascia_13_22": int(f["fascia_13_22"].sum()), "n_etichette": int(f["F2_p_su"].notna().sum())} for h, f in frames.items()},
            "prima_origine_con_etichetta_f2": str(pd.Timestamp(et["data_origine"].iloc[0]).date()) if len(et) else None,
            "g3": {"riga": riga_g3, "n": int(len(tg3)), "regola": f"{VB.G3_N} origini, una ogni {VB.G3_PASSO} settimane a ritroso a partire da {VB.G3_PRIMA} settimane prima dell'ultima data di sviluppo"}}
    (OUT / "c_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with K.guardia_dati():
        run()
