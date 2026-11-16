"""Fase 2b: walk-forward sul campione di sviluppo per h = 1, 2, 3, 4 (solo misura; nessuna scelta di modello).

Misura errore, larghezza e copertura della chiusura (D4), l'errore di massimo e minimo contro il riferimento
semplice, e produce il grafico G3. Tutto sul file di sviluppo, sotto la guardia.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config as C
from ..fase1 import comune as K
from ..fase1 import modelli as M
from ..fase2 import densita as D
from ..fase2 import valuta as V2
from . import dati, grafici
from . import modello as MO

OUT = C.REPORTS_DIR / "fase2b"
FIG = C.FIG_DIR / "fase2b"
G3_N, G3_PASSO, G3_PRIMA = 12, 52, 4          # 12 origini, una ogni 52 settimane a ritroso da 4 settimane prima della fine dev
G3_STORIA = 8


def walk(df: pd.DataFrame) -> dict:
    close = df["close"].to_numpy(dtype=float)
    high, low = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
    x, lh, ll = np.log(close), np.log(high), np.log(low)
    dates = df["date"].to_numpy()
    res, med = {}, {}
    for h in MO.ORIZZONTI:
        c1 = D.centers(x, M.forecast_har, h)
        o = np.arange(D.first_origin(h), len(x) - h)
        _, d4 = D.density_walk(x, "D4", h, c1)
        _, d1 = D.density_walk(x, "D1", h, c1)
        Q = np.array([d.quantiles() for d in d4])
        P = np.array([[d.prob_above(np.log(s)) for s in MO.SOGLIE] for d in d1])
        hi = np.exp([MO.forecast_hl(x[: t + 1], lh[: t + 1], h) for t in o])
        lo = np.exp([MO.forecast_hl(x[: t + 1], ll[: t + 1], h) for t in o])
        hi_ref = np.array([MO.riferimento_hl(close[: t + 1], high[: t + 1], h) for t in o])
        lo_ref = np.array([MO.riferimento_hl(close[: t + 1], low[: t + 1], h) for t in o])
        m = np.exp(Q[:, V2.QI[3]])
        med[h] = dict(zip(o.tolist(), m.tolist()))
        ap = close[o] if h == 1 else np.array([med[h - 1][t] for t in o.tolist()])     # apertura = chiusura prevista precedente
        top, bot = np.maximum(ap, m), np.minimum(ap, m)
        base = pd.DataFrame({"h": h, "idx_origine": o, "data_origine": dates[o], "data_obiettivo": dates[o + h], "x0": x[o], "vix0": close[o],
                             "y": x[o + h], "chiusura_reale": close[o + h], "massimo_reale": high[o + h], "minimo_reale": low[o + h],
                             "m1": np.exp(c1[o - D.O0]), "mediana_d4": m, "apertura_prevista": ap,
                             "massimo_grezzo": hi, "minimo_grezzo": lo, "massimo_previsto": np.maximum(hi, top), "minimo_previsto": np.minimum(lo, bot),
                             "massimo_corretto": hi < top, "minimo_corretto": lo > bot, "massimo_riferimento": hi_ref, "minimo_riferimento": lo_ref})
        for j, tau in zip(V2.QI, D.Q_REPORT):
            base[f"q{int(100 * tau):02d}"] = np.exp(Q[:, j])
        for k, s in enumerate(MO.SOGLIE):
            base[f"p_{int(s)}"] = P[:, k]
        res[h] = {"base": base, "Q": Q, "incroci": int(sum(d.crossed for d in d4))}
    return res


def tabelle(res: dict) -> dict[str, pd.DataFrame]:
    ch, hl, pr = [], [], []
    for h, r in res.items():
        b = r["base"]
        row = {"h": h} | V2.density_row(r["Q"], b, True, f"f2b|{h}")
        row |= {"mae_m1_vix": float(np.abs(b["chiusura_reale"] - b["m1"]).mean()), "quantili_incrociati": r["incroci"],
                "prima_data_obiettivo": str(pd.Timestamp(b["data_obiettivo"].iloc[0]).date()), "ultima_data_obiettivo": str(pd.Timestamp(b["data_obiettivo"].iloc[-1]).date())}
        ch.append(row)
        for nome in ("massimo", "minimo"):
            e_mod = np.abs(b[f"{nome}_reale"] - b[f"{nome}_previsto"]).to_numpy()
            e_raw = np.abs(b[f"{nome}_reale"] - b[f"{nome}_grezzo"]).to_numpy()
            e_ref = np.abs(b[f"{nome}_reale"] - b[f"{nome}_riferimento"]).to_numpy()
            dm, p = V2.dm_loss(e_ref, e_mod, h)
            bl = K.blocks(b["idx_origine"].to_numpy())
            lo, hi = K.boot_cluster(bl, lambda i: e_ref[i].mean() - e_mod[i].mean(), f"f2bhl|{h}|{nome}")
            hl.append({"h": h, "obiettivo": nome, "n": len(b), "mae_modello": e_mod.mean(), "mae_modello_senza_vincoli": e_raw.mean(),
                       "mae_riferimento": e_ref.mean(), "guadagno_punti": e_ref.mean() - e_mod.mean(), "guadagno_ci_lo": lo, "guadagno_ci_hi": hi,
                       "guadagno_pct": 1 - e_mod.mean() / e_ref.mean(), "dm_stat": dm, "p": p,
                       "n_correzioni": int(b[f"{nome}_corretto"].sum()), "quota_correzioni": float(b[f"{nome}_corretto"].mean()),
                       "errore_medio_con_segno": float((b[f"{nome}_reale"] - b[f"{nome}_previsto"]).mean()),
                       "quota_reale_oltre_previsto": float((b[f"{nome}_reale"] > b[f"{nome}_previsto"]).mean() if nome == "massimo"
                                                           else (b[f"{nome}_reale"] < b[f"{nome}_previsto"]).mean())})
        for s in MO.SOGLIE:                      # misura aggiuntiva: probabilita' di D1 (h = 2 e 3 non erano mai state misurate)
            p_, y_ = b[f"p_{int(s)}"].to_numpy(), (b["chiusura_reale"] > s).to_numpy().astype(float)
            pr.append({"h": h, "evento": f"VIX>{int(s)}", "n": len(b), "frequenza_evento": y_.mean(), "prob_media": p_.mean()} | V2.brier_parts(p_, y_))
    return {"chiusura_metriche": pd.DataFrame(ch), "massimo_minimo_metriche": pd.DataFrame(hl), "probabilita_metriche": pd.DataFrame(pr)}


def pannelli_g3(df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    n = len(df)
    out, rows = [], []
    for k in range(G3_N):
        t = n - 1 - G3_PRIMA - G3_PASSO * k
        prev = MO.prevedi(df, t)
        reali = df.iloc[t + 1:t + 5]
        s4 = prev["settimane"][-1]
        y4 = float(reali["close"].iloc[-1])
        d50 = s4["fasce"]["50"][0] <= y4 <= s4["fasce"]["50"][1]
        d80 = s4["fasce"]["80"][0] <= y4 <= s4["fasce"]["80"][1]
        out.append({"previsione": prev, "storia": df.iloc[t - G3_STORIA + 1:t + 1], "reali": reali, "dentro_50": d50, "dentro_80": d80})
        rows.append({"data_origine": prev["data_origine"], "chiusura_origine": prev["chiusura_origine"], "chiusura_prevista_4": s4["chiusura"],
                     "fascia_50_inf": s4["fasce"]["50"][0], "fascia_50_sup": s4["fasce"]["50"][1], "fascia_80_inf": s4["fasce"]["80"][0],
                     "fascia_80_sup": s4["fasce"]["80"][1], "chiusura_reale_4": y4, "dentro_50": d50, "dentro_80": d80})
    out, rows = out[::-1], rows[::-1]                # in ordine di tempo
    return out, pd.DataFrame(rows)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = dati.load_dev_candele()
    res = walk(df)
    tabs = tabelle(res)
    for name, t in tabs.items():
        t.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")
    pd.concat([r["base"].drop(columns="idx_origine") for r in res.values()], ignore_index=True).to_csv(
        OUT / "previsioni_dev.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")
    pan, tg3 = pannelli_g3(df)
    tg3.to_csv(OUT / "g3_pannelli.csv", index=False, float_format="%.4f")
    n50, n80 = int(tg3["dentro_50"].sum()), int(tg3["dentro_80"].sum())
    riga = (f"Chiusura reale a 4 settimane dentro la fascia al 50%: {n50} pannelli su {len(tg3)}; dentro la fascia all'80%: {n80} su {len(tg3)}.")
    grafici.g3(pan, FIG / "g3_verifica_passato.png", riga)
    info = {"campione": [str(df["date"].min().date()), str(df["date"].max().date())], "n_settimane": int(len(df)),
            "orizzonti": list(MO.ORIZZONTI), "modelli": {"mediana_e_fasce": "D4", "probabilita": "D1", "massimo_minimo": "regressione diretta di M1 su log(high) e log(low)"},
            "min_errori_passati": D.MIN_ERR, "origine_minima_indice": MO.origine_minima(),
            "g3": {"regola": f"{G3_N} origini, una ogni {G3_PASSO} settimane a ritroso a partire da {G3_PRIMA} settimane prima dell'ultima data di sviluppo",
                   "dentro_50": n50, "dentro_80": n80, "n": int(len(tg3)), "riga": riga},
            "per_orizzonte": {str(h): {"n_previsioni": int(len(r["base"])), "incroci": r["incroci"]} for h, r in res.items()}}
    (OUT / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with K.guardia_dati():
        run()
