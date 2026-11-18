"""VIX9D, Parti B-E: walk-forward di sviluppo, tabelle e regole di composizione delle versioni G1 e GS.

Finestra espandibile, ristima a ogni origine, prima stima su almeno 156 settimane. Ogni confronto e' sulle stesse origini:
il campione principale e' quello delle «origini comuni», in cui esistono tutti i candidati (i modelli con la candela del
VIX9D esistono solo dopo 156 settimane con candela vera). Per R0, R1 e R2 sono riportati anche i risultati su tutte le
origini disponibili per la sola chiusura.

Regola per «batte» (dalla consegna): errore piu' basso con p < 0,05 dopo Holm; Clark-West a una coda se il modello
contiene il riferimento, Diebold-Mariano a due code altrimenti. Precisazioni fissate prima di guardare i risultati:
- la scelta e' fatta per ogni componente e per ogni orizzonte (e, per gli estremi, per massimo e minimo separatamente);
- Holm e' applicato dentro ogni famiglia di confronti: chiusura senza VIX (R3 contro R1: 2 test), chiusura con VIX
  (R2 e R4 contro il modello di G1: 4 test), estremi senza VIX (S1 contro S0: 4 test), estremi con VIX (S2 contro
  l'insieme di G1: 4 test), direzione con VIX (2 test);
- le fasce seguono le variabili del modello di chiusura scelto; le probabilita' delle soglie usano il suo centro;
- errori: quadratico in log per la chiusura, assoluto in log per gli estremi, Brier per la direzione.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..evaluate import diebold_mariano
from ..fase1 import comune as K
from ..fase1.valuta import clark_west, holm
from ..fase2 import valuta as V2
from ..fase2c import valuta as VC
from ..fase4c import valuta as V4C
from . import config as G
from . import dati
from . import modelli as M

ALFA = 0.05
SOGLIE_ETICHETTA = (0.60, 0.65, 0.70)
QN = [f"q{int(round(100 * t)):02d}" for t in G.TAUS]
CAMPI = ("apertura", "massimo", "minimo", "chiusura")
EST = ("massimo", "minimo")
ANNIDATI = {("R1", "R2"), ("R1", "R3"), ("R1", "R4"), ("R3", "R4"), ("R2", "R4"), ("R0", "R1")}      # (piccolo, grande)


# ------------------------------------------------------------------ walk-forward

def walk(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Per ogni orizzonte e ogni origine t >= 155: tutte le previsioni di tutti i candidati, con la sola storia fino a t."""
    VV = M.variabili(df)
    x = df["log_close"].to_numpy()
    o_, h_, l_, c_ = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    cand, date = df["candela"].to_numpy(bool), df["date"].to_numpy()
    Z = {k: VV[list(v)].to_numpy() for k, v in (M.CHIUSURA | M.ESTREMI | M.DIREZIONE).items()}
    frames = {}
    for h in G.ORIZZONTI:
        rows = []
        for t in range(G.MIN_SETTIMANE - 1, len(x) - h):
            xt, ct = x[: t + 1], cand[: t + 1]
            r = {"h": h, "idx_origine": t, "data_origine": date[t], "data_obiettivo": date[t + h], "chiusura_origine": c_[t], "x0": x[t], "y": x[t + h],
                 "apertura_reale": o_[t + h], "massimo_reale": h_[t + h], "minimo_reale": l_[t + h], "chiusura_reale": c_[t + h], "candela_reale": bool(cand[t + h]),
                 "su": float(x[t + h] > x[t]), "strappo": bool(c_[t + h] > (1 + G.STRAPPO) * c_[t]), "quintile": M.quintile(c_[t], c_[: t + 1]), "f_R0": x[t],
                 "p_FREQ": M.frequenza_rialzo(xt, h)}
            for k in M.CHIUSURA:
                r[f"f_{k}"] = M.chiusura(xt, Z[k][: t + 1], h)
                q = M.quantili(xt, Z[k][: t + 1], h)
                r |= {f"{k}_{n}": v for n, v in zip(QN, q)}
            for k in M.DIREZIONE:
                r[f"p_{k}"] = M.direzione(xt, Z[k][: t + 1], h)
            for e, L in (("massimo", h_), ("minimo", l_)):
                for k in M.ESTREMI:
                    r[f"{k}_{e}"] = M.estremo(xt, L[: t + 1], Z[k][: t + 1], h, ct)
                r[f"FISSA_{e}"] = M.regola_fissa(xt, L[: t + 1], h, ct)
            r |= {f"comp_{k}": v for k, v in M.componenti_candela(o_[: t + 1], c_[: t + 1], ct).items()}
            rows.append(r)
        frames[h] = pd.DataFrame(rows)
    return frames


def comuni(f: pd.DataFrame) -> np.ndarray:
    """Origini in cui esistono tutti i candidati e la settimana prevista ha una candela vera."""
    cols = [f"f_{k}" for k in M.CHIUSURA] + [f"p_{k}" for k in M.DIREZIONE] + [f"{k}_{e}" for k in M.ESTREMI for e in EST] + ["FISSA_massimo", "comp_corpo"]
    return f[cols].notna().all(axis=1).to_numpy() & f["candela_reale"].to_numpy(bool)


# ------------------------------------------------------------------ Parte B: chiusura

def riga_chiusura(g: pd.DataFrame, m: str, h: int, tag: str | None = None) -> dict:
    y, f, r0, r1 = (g[c].to_numpy() for c in ("y", f"f_{m}", "f_R0", "f_R1"))
    e, e0, e1 = y - f, y - r0, y - r1
    ev, rel = np.exp(y) - np.exp(f), np.abs(np.exp(f - y) - 1)
    row = {"n": len(g), "rmse_log": np.sqrt((e ** 2).mean()), "mae_log": np.abs(e).mean(), "rmse_punti": np.sqrt((ev ** 2).mean()), "mae_punti": np.abs(ev).mean(),
           "entro_5": (rel <= 0.05).mean(), "entro_10": (rel <= 0.10).mean(), "entro_20": (rel <= 0.20).mean()}
    if m != "R0":
        row["r2_vs_r0"] = 1 - (e ** 2).sum() / (e0 ** 2).sum()
    if m not in ("R0", "R1"):
        row["r2_vs_r1"] = 1 - (e ** 2).sum() / (e1 ** 2).sum()
        row["stat"], row["p"] = clark_west(e1, e, r1, f, h)
        row["test"] = "Clark-West (una coda)"
        if tag:
            row["r2_ci_lo"], row["r2_ci_hi"] = K.boot_cluster(K.blocks(g["idx_origine"].to_numpy()), lambda i: 1 - (e[i] ** 2).sum() / (e1[i] ** 2).sum(), tag)
    elif m == "R1":
        row["stat"], row["p"] = clark_west(e0, e, r0, f, h)
        row["test"] = "Clark-West contro R0 (una coda)"
    return row


def tabella_chiusura(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for h, f in frames.items():
        for campione, g, modelli in (("origini comuni", f[comuni(f)], ("R0",) + tuple(M.CHIUSURA)), ("tutte le origini della chiusura", f[f["f_R2"].notna()], ("R0", "R1", "R2"))):
            for m in modelli:
                rows.append({"campione": campione, "h": h, "modello": m} | riga_chiusura(g, m, h, f"v9|b|{campione}|{m}|{h}"))
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for campione in t["campione"].unique():
        k = (t["campione"] == campione) & t["modello"].isin(["R2", "R3", "R4"])
        t.loc[k, "p_holm"] = holm(t.loc[k, "p"].to_numpy())
    return t


def confronto(g: pd.DataFrame, rif: str, mod: str, h: int) -> tuple[str, float, float]:
    """(test, statistica, p) per `mod` contro `rif` sulla chiusura: Clark-West se `mod` contiene `rif`, altrimenti Diebold-Mariano.
    Statistica positiva = `mod` migliore."""
    y, fr, fm = (g[c].to_numpy() for c in ("y", f"f_{rif}", f"f_{mod}"))
    if (rif, mod) in ANNIDATI:
        s, p = clark_west(y - fr, y - fm, fr, fm, h)
        return "Clark-West", s, p
    s, p = diebold_mariano(y - fr, y - fm, h)
    return "Diebold-Mariano", s, p


def scegli(righe: list[dict]) -> dict:
    """Applica Holm alla famiglia di confronti e, per ogni chiave (orizzonte o estremo-orizzonte), sceglie il candidato che
    batte il riferimento con l'errore piu' basso; se nessuno, resta il riferimento."""
    t = pd.DataFrame(righe)
    t["p_holm"] = holm(t["p"].to_numpy())
    t["batte"] = (t["p_holm"] < ALFA) & (t["stat"] > 0) & (t["errore"] < t["errore_riferimento"])
    scelta = {}
    for chiave, g in t.groupby("chiave", sort=False):
        b = g[g["batte"]].sort_values("errore")
        scelta[chiave] = str(b["candidato"].iloc[0]) if len(b) else str(g["riferimento"].iloc[0])
    return {"tabella": t, "scelta": scelta}


# ------------------------------------------------------------------ Parte C: fasce, probabilita', direzione

def riga_fasce(g: pd.DataFrame, m: str) -> dict:
    y = g["y"].to_numpy()
    q = g[[f"{m}_{n}" for n in QN]].to_numpy()
    pb = V2.pinball(y, q, M.TAUS)
    r = {"n": len(g), "crps7": 2 * pb.mean(), "mae_mediana_punti": float(np.abs(np.exp(y) - np.exp(q[:, 3])).mean())}
    for lev, (a, b) in G.FASCE.items():
        lo, hi = q[:, list(G.TAUS).index(a)], q[:, list(G.TAUS).index(b)]
        r |= {f"copertura_{lev}": float(((y >= lo) & (y <= hi)).mean()), f"larghezza_{lev}": float((np.exp(hi) - np.exp(lo)).mean())}
    return r


def tabella_fasce(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for h, f in frames.items():
        g = f[comuni(f)]
        y = g["y"].to_numpy()
        l1 = V2.pinball(y, g[[f"R1_{n}" for n in QN]].to_numpy(), M.TAUS).mean(axis=1)
        for m in M.CHIUSURA:
            r = {"h": h, "modello": m} | riga_fasce(g, m)
            if m != "R1":
                l = V2.pinball(y, g[[f"{m}_{n}" for n in QN]].to_numpy(), M.TAUS).mean(axis=1)
                r["crps_vs_r1"] = 1 - l.mean() / l1.mean()
                r["dm_stat"], r["p"] = V2.dm_loss(l1, l, h)
            rows.append(r)
    return pd.DataFrame(rows)


def probabilita_soglie(f: pd.DataFrame, m: str, h: int) -> pd.DataFrame:
    """P(VIX9D > soglia) = quota degli errori passati (con esito noto) che, sommati al centro del modello m, superano la soglia.
    Disponibile solo con almeno 52 errori passati."""
    g = f[f[f"f_{m}"].notna()].reset_index(drop=True)
    idx, c, y = g["idx_origine"].to_numpy(), g[f"f_{m}"].to_numpy(), g["y"].to_numpy()
    err = y - c
    out = {f"p_{int(s)}": np.full(len(g), np.nan) for s in G.SOGLIE}
    for k, t in enumerate(idx.tolist()):
        noti = idx + h <= t
        if noti.sum() >= G.MIN_ERRORI:
            for s in G.SOGLIE:
                out[f"p_{int(s)}"][k] = float((c[k] + err[noti] > np.log(s)).mean())
    return pd.DataFrame({"idx_origine": idx} | out)


def tabella_soglie(frames: dict[int, pd.DataFrame], modelli: dict[str, dict[int, str]]) -> tuple[pd.DataFrame, dict]:
    rows, prob = [], {}
    for v, per_h in modelli.items():
        for h, m in per_h.items():
            f = frames[h]
            p = probabilita_soglie(f, m, h).merge(f[["idx_origine", "chiusura_reale", "chiusura_origine"]], on="idx_origine")
            prob[(v, h)] = p
            ok = p[f"p_{int(G.SOGLIE[0])}"].notna() & p["idx_origine"].isin(f.loc[comuni(f), "idx_origine"])
            for s in G.SOGLIE:
                pp, yy = p.loc[ok, f"p_{int(s)}"].to_numpy(), (p.loc[ok, "chiusura_reale"] > s).to_numpy().astype(float)
                pers = (p.loc[ok, "chiusura_origine"] > s).to_numpy().astype(float)
                rows.append({"versione": v, "h": h, "modello": m, "soglia": int(s), "n": int(ok.sum()), "frequenza_evento": yy.mean(), "prob_media": pp.mean(), "brier": ((pp - yy) ** 2).mean(),
                             "brier_frequenza": yy.mean() * (1 - yy.mean()), "brier_persistenza": ((pers - yy) ** 2).mean()})
    return pd.DataFrame(rows), prob


def etichette(p: np.ndarray, y: np.ndarray, soglia: float = 0.60) -> dict:
    su, giu = p >= soglia, p <= 1 - soglia
    nn = su | giu
    ok = np.where(su, y == 1, y == 0)
    k = int(round(100 * soglia))
    return {f"quota_non_neutre_{k}": float(nn.mean()), f"n_non_neutre_{k}": int(nn.sum()), f"accuratezza_{k}": float(ok[nn].mean()) if nn.any() else np.nan,
            f"n_rialzo_{k}": int(su.sum()), f"accuratezza_rialzo_{k}": float((y[su] == 1).mean()) if su.any() else np.nan,
            f"n_ribasso_{k}": int(giu.sum()), f"accuratezza_ribasso_{k}": float((y[giu] == 0).mean()) if giu.any() else np.nan}


def tabella_direzione(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for h, f in frames.items():
        g = f[comuni(f)]
        y, bl = g["su"].to_numpy(), K.blocks(g["idx_origine"].to_numpy())
        lb, lf = (g["p_BASE"].to_numpy() - y) ** 2, (g["p_FREQ"].to_numpy() - y) ** 2
        for m in ("FREQ", "BASE", "RAPP"):
            p = g[f"p_{m}"].to_numpy()
            l = (p - y) ** 2
            r = {"h": h, "modello": m, "n": len(g), "frequenza_rialzo": y.mean(), "accuratezza": float(((p >= 0.5) == (y == 1)).mean())} | V2.brier_parts(p, y)
            for s in SOGLIE_ETICHETTA:
                r |= etichette(p, y, s)
            if m != "FREQ":
                r["skill_vs_freq"] = 1 - l.mean() / lf.mean()
                r["dm_stat_freq"], r["p_freq"] = V2.dm_loss(lf, l, h)
            if m == "RAPP":
                r["skill_vs_base"] = 1 - l.mean() / lb.mean()
                r["dm_stat_base"], r["p_base"] = V2.dm_loss(lb, l, h)
                r["skill_ci_lo"], r["skill_ci_hi"] = K.boot_cluster(bl, lambda i: 1 - l[i].mean() / lb[i].mean(), f"v9|dir|{h}")
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Parte D: estremi

def tabella_estremi(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for h, f in frames.items():
        g = f[comuni(f)]
        gruppi = {"tutte": np.ones(len(g), bool), "senza strappi": ~g["strappo"].to_numpy(bool)} | {f"quintile {k}": (g["quintile"] == k).to_numpy() for k in range(1, 6)}
        for nome, m in gruppi.items():
            gg = g[m]
            if len(gg) == 0:
                continue
            for k in tuple(M.ESTREMI) + ("FISSA",):
                dentro = {e: np.abs(gg[f"{k}_{e}"].to_numpy() / gg[f"{e}_reale"].to_numpy() - 1) <= 0.05 for e in EST}
                for e in EST:
                    rows.append({"h": h, "gruppo": nome, "insieme": k, "estremo": e} | V4C.metriche(gg[f"{k}_{e}"].to_numpy(), gg[f"{e}_reale"].to_numpy())
                                | {"entrambi_entro_5": float((dentro["massimo"] & dentro["minimo"]).mean())})
    return pd.DataFrame(rows)


def errore_estremo(g: pd.DataFrame, k: str, e: str) -> np.ndarray:
    return np.abs(np.log(g[f"{k}_{e}"].to_numpy() / g[f"{e}_reale"].to_numpy()))


# ------------------------------------------------------------------ Parte E: composizione e candele

def componi(frames: dict[int, pd.DataFrame], sc: dict, v: str, prob: dict) -> dict[int, pd.DataFrame]:
    """Colonne della versione v sulle origini comuni: chiusura centrale e fasce, probabilita' di direzione, estremi, candela."""
    out = {}
    for h in G.ORIZZONTI:
        f = frames[h]
        g = f[comuni(f)].reset_index(drop=True).copy()
        mc, md = sc["chiusura"][v][str(h)], sc["direzione"][v][str(h)]
        for n in QN:
            g[f"{v}_{n}"] = np.exp(g[f"{mc}_{n}"])
        g[f"{v}_centro"], g[f"{v}_p_su"] = np.exp(g[f"f_{mc}"]), g[f"p_{md}"]
        for e in EST:
            g[f"{v}_est_{e}"] = g[f"{sc['estremi'][v][e][str(h)]}_{e}"]
        pr = prob[(v, h)].set_index("idx_origine").reindex(g["idx_origine"])
        for s in G.SOGLIE:
            g[f"{v}_p_{int(s)}"] = pr[f"p_{int(s)}"].to_numpy()
        out[h] = g
    c1 = dict(zip(out[1]["idx_origine"].tolist(), out[1][f"{v}_q50"].tolist()))
    for h in G.ORIZZONTI:
        g = out[h]
        cols = {c: [] for c in CAMPI + ("chiusura_tagliata",)}
        for r in g.to_dict("records"):
            base = r["chiusura_origine"] if h == 1 else c1.get(r["idx_origine"], np.nan)
            k = M.candela_prevista(base, {"rapporto_apertura": r["comp_rapporto_apertura"], "p_su": r["comp_p_su"], "corpo": r["comp_corpo"]}, r[f"{v}_est_massimo"], r[f"{v}_est_minimo"])
            for c in cols:
                cols[c].append(k[c])
        for c in cols:
            g[f"{v}_{c}"] = cols[c]
        out[h] = g[g[f"{v}_chiusura"].notna()].reset_index(drop=True)
    return out


def riga_versione(g: pd.DataFrame, v: str, h: int) -> dict:
    y, rc = g["y"].to_numpy(), g["chiusura_reale"].to_numpy()
    cen = g[f"{v}_q50"].to_numpy()
    rel = np.abs(cen / rc - 1)
    r = {"n": len(g), "mae_centrale": float(np.abs(cen - rc).mean()), "rmse_log_centrale": float(np.sqrt(((np.log(cen) - y) ** 2).mean())),
         "entro_5_chiusura": float((rel <= 0.05).mean()), "entro_10_chiusura": float((rel <= 0.10).mean()), "entro_20_chiusura": float((rel <= 0.20).mean())}
    for lev, (a, b) in G.FASCE.items():
        lo, hi = g[f"{v}_q{int(round(100 * a)):02d}"].to_numpy(), g[f"{v}_q{int(round(100 * b)):02d}"].to_numpy()
        r |= {f"copertura_{lev}": float(((rc >= lo) & (rc <= hi)).mean()), f"larghezza_{lev}": float((hi - lo).mean())}
    p, su = g[f"{v}_p_su"].to_numpy(), g["su"].to_numpy()
    et = etichette(p, su, 0.60)
    r |= {"brier_direzione": float(((p - su) ** 2).mean()), "quota_non_neutre": et["quota_non_neutre_60"], "n_non_neutre": et["n_non_neutre_60"], "accuratezza_non_neutre": et["accuratezza_60"]}
    for e in EST:
        m = V4C.metriche(g[f"{v}_{e}"].to_numpy(), g[f"{e}_reale"].to_numpy())
        r |= {f"{e}_entro_5": m["entro_5"], f"{e}_entro_10": m["entro_10"], f"{e}_errore_pct_mediano": m["errore_pct_mediano"], f"{e}_errore_punti_mediano": m["errore_punti_mediano"],
              f"{e}_oltre_5_per_eccesso": m["oltre_5_per_eccesso"], f"{e}_oltre_5_per_difetto": m["oltre_5_per_difetto"]}
    dentro = {e: np.abs(g[f"{v}_{e}"].to_numpy() / g[f"{e}_reale"].to_numpy() - 1) <= 0.05 for e in EST}
    pg = VC.pagella_riga(g, v)
    return r | {"entrambi_entro_5": float((dentro["massimo"] & dentro["minimo"]).mean()), "sovrapposizione": pg["sovrapposizione"], "chiusura_nell_escursione": pg["chiusura_nell_escursione"],
                "chiusura_nel_corpo": pg["chiusura_nel_corpo"], "rapporto_corpo": pg["rapporto_corpo"], "rapporto_escursione": pg["rapporto_escursione"], "direzione_corpo": pg["direzione_corpo"],
                "mae_apertura": pg["mae_apertura"], "mae_massimo": pg["mae_massimo"], "mae_minimo": pg["mae_minimo"], "mae_chiusura": pg["mae_chiusura"],
                "quota_chiusura_tagliata": float(g[f"{v}_chiusura_tagliata"].mean())}


# ------------------------------------------------------------------ regole di composizione

def regole(frames: dict[int, pd.DataFrame]) -> dict:
    C = {h: f[comuni(f)] for h, f in frames.items()}
    rmse = lambda g, m: float(np.sqrt(((g["y"] - g[f"f_{m}"]) ** 2).mean()))
    # chiusura senza VIX: R3 contro R1
    r = []
    for h, g in C.items():
        test, s, p = confronto(g, "R1", "R3", h)
        r.append({"famiglia": "chiusura senza VIX", "chiave": str(h), "riferimento": "R1", "candidato": "R3", "test": test, "stat": s, "p": p, "errore": rmse(g, "R3"), "errore_riferimento": rmse(g, "R1")})
    a = scegli(r)
    # chiusura con VIX: R2 e R4 contro il modello di G1
    r = []
    for h, g in C.items():
        rif = a["scelta"][str(h)]
        for m in M.CON_VIX["chiusura"]:
            test, s, p = confronto(g, rif, m, h)
            r.append({"famiglia": "chiusura con VIX", "chiave": str(h), "riferimento": rif, "candidato": m, "test": test, "stat": s, "p": p, "errore": rmse(g, m), "errore_riferimento": rmse(g, rif)})
    b = scegli(r)
    # estremi senza VIX: S1 contro S0; con VIX: S2 contro l'insieme di G1
    r = []
    for h, g in C.items():
        for e in EST:
            e0, e1 = errore_estremo(g, "S0", e), errore_estremo(g, "S1", e)
            s, p = V2.dm_loss(e0, e1, h)
            r.append({"famiglia": "estremi senza VIX", "chiave": f"{e}|{h}", "riferimento": "S0", "candidato": "S1", "test": "Diebold-Mariano", "stat": s, "p": p, "errore": e1.mean(), "errore_riferimento": e0.mean()})
    c = scegli(r)
    r = []
    for h, g in C.items():
        for e in EST:
            rif = c["scelta"][f"{e}|{h}"]
            e0, e1 = errore_estremo(g, rif, e), errore_estremo(g, "S2", e)
            s, p = V2.dm_loss(e0, e1, h)
            r.append({"famiglia": "estremi con VIX", "chiave": f"{e}|{h}", "riferimento": rif, "candidato": "S2", "test": "Diebold-Mariano", "stat": s, "p": p, "errore": e1.mean(), "errore_riferimento": e0.mean()})
    d = scegli(r)
    # direzione con VIX: logistica con ln(VIX9D/VIX) contro la logistica di base
    r = []
    for h, g in C.items():
        y = g["su"].to_numpy()
        lb, lr = (g["p_BASE"].to_numpy() - y) ** 2, (g["p_RAPP"].to_numpy() - y) ** 2
        s, p = V2.dm_loss(lb, lr, h)
        r.append({"famiglia": "direzione con VIX", "chiave": str(h), "riferimento": "BASE", "candidato": "RAPP", "test": "Diebold-Mariano", "stat": s, "p": p, "errore": lr.mean(), "errore_riferimento": lb.mean()})
    e_ = scegli(r)
    hz = [str(h) for h in G.ORIZZONTI]
    scelte = {"chiusura": {"g1": {h: a["scelta"][h] for h in hz}, "gs": {h: b["scelta"][h] for h in hz}},
              "estremi": {"g1": {e: {h: c["scelta"][f"{e}|{h}"] for h in hz} for e in EST}, "gs": {e: {h: d["scelta"][f"{e}|{h}"] for h in hz} for e in EST}},
              "direzione": {"g1": {h: "BASE" for h in hz}, "gs": {h: e_["scelta"][h] for h in hz}}}
    return {"scelte": scelte, "tabella": pd.concat([k["tabella"] for k in (a, b, c, d, e_)], ignore_index=True)}


# ------------------------------------------------------------------ esecuzione

def run() -> None:
    G.OUT.mkdir(parents=True, exist_ok=True)
    G.FIG.mkdir(parents=True, exist_ok=True)
    df = dati.load_dev()
    frames = walk(df)
    reg = regole(frames)
    sc = reg["scelte"]
    t_sog, prob = tabella_soglie(frames, {v: {h: sc["chiusura"][v][str(h)] for h in G.ORIZZONTI} for v in ("g1", "gs")})
    vers = {v: componi(frames, sc, v, prob) for v in ("g1", "gs")}
    righe = []
    for v, per_h in vers.items():
        for h, g in per_h.items():
            for gruppo, m in (("tutte", np.ones(len(g), bool)), ("senza strappi", ~g["strappo"].to_numpy(bool))):
                righe.append({"versione": v, "h": h, "gruppo": gruppo} | riga_versione(g[m], v, h))
    tabs = {"b_chiusura": tabella_chiusura(frames), "c_fasce": tabella_fasce(frames), "c_soglie": t_sog, "c_direzione": tabella_direzione(frames), "d_estremi": tabella_estremi(frames),
            "regole": reg["tabella"], "e_versioni": pd.DataFrame(righe)}
    for nome, t in tabs.items():
        t.to_csv(G.OUT / f"{nome}.csv", index=False, float_format="%.6g")
    for v, per_h in vers.items():
        tieni = ["h", "data_origine", "data_obiettivo", "chiusura_origine", "strappo", "quintile"] + [f"{c}_reale" for c in CAMPI] + [k for k in per_h[1].columns if k.startswith(f"{v}_")]
        pd.concat([g[tieni] for g in per_h.values()], ignore_index=True).to_csv(G.OUT / f"previsioni_dev_{v}.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    pd.concat([f.drop(columns="idx_origine") for f in frames.values()], ignore_index=True).to_csv(G.OUT / "previsioni_dev_candidati.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    f1 = frames[1]
    com = f1[comuni(f1)]
    info = {"scelte": sc, "n_settimane": int(len(df)), "campione": [str(df["date"].iloc[0].date()), str(df["date"].iloc[-1].date())],
            "prima_settimana_con_candela": str(df.loc[df["candela"], "date"].iloc[0].date()),
            "origini": {str(h): {"tutte_chiusura": int(f["f_R1"].notna().sum()), "prima_origine_chiusura": str(pd.Timestamp(f.loc[f["f_R1"].notna(), "data_origine"].iloc[0]).date()),
                                 "comuni": int(comuni(f).sum()), "prima_origine_comune": str(pd.Timestamp(f.loc[comuni(f), "data_origine"].iloc[0]).date()),
                                 "ultima_data_obiettivo": str(pd.Timestamp(f["data_obiettivo"].iloc[-1]).date()), "strappi_tra_le_comuni": int(f.loc[comuni(f), "strappo"].sum())} for h, f in frames.items()},
            "componenti_ultima_origine": {k: float(com[f"comp_{k}"].iloc[-1]) for k in ("rapporto_apertura", "p_su", "corpo")},
            "min_settimane": G.MIN_SETTIMANE, "min_coppie": G.MIN_COPPIE, "min_errori": G.MIN_ERRORI, "finestra_corpo": G.FINESTRA_CORPO, "strappo": G.STRAPPO, "alfa": ALFA}
    (G.OUT / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with dati.guardia():
        run()
