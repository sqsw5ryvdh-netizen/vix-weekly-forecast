"""Valutazione walk-forward delle baseline (puntuali e probabilistica) sul solo campione di sviluppo."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import baselines as B
from . import config as C

BENCH = "random_walk"


def diebold_mariano(e_bench: np.ndarray, e_model: np.ndarray, h: int) -> tuple[float, float]:
    """Test DM su perdita quadratica, d = e_bench^2 - e_model^2 (statistica > 0: il modello batte il benchmark).

    Varianza di lungo periodo con h-1 autocovarianze (kernel rettangolare; Bartlett se risulta non positiva),
    correzione per piccoli campioni di Harvey-Leybourne-Newbold, p-value bilaterale da t di Student (n-1 g.d.l.).
    """
    d = e_bench ** 2 - e_model ** 2
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return float("nan"), float("nan")
    dc = d - d.mean()
    gam = [dc @ dc / n] + [dc[k:] @ dc[:-k] / n for k in range(1, h)]
    lrv = gam[0] + 2 * sum(gam[1:])
    if lrv <= 0:
        lrv = gam[0] + 2 * sum((1 - k / h) * g for k, g in enumerate(gam[1:], start=1))
    dm = d.mean() / np.sqrt(lrv / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), df=n - 1))


def point_metrics(g: pd.DataFrame, h: int) -> dict:
    """Metriche per un modello su un insieme di previsioni. Colonne richieste: y, f, x0, f_bench."""
    e, eb = (g["y"] - g["f"]).to_numpy(), (g["y"] - g["f_bench"]).to_numpy()
    ev = np.exp(g["y"]) - np.exp(g["f"])
    dy, df_ = np.sign(g["y"] - g["x0"]), np.sign(g["f"] - g["x0"])
    ok = (dy != 0) & (df_ != 0)            # il random walk prevede variazione nulla: direzione non definita
    dm, p = diebold_mariano(eb, e, h)
    return {"n": len(g), "rmse_log": np.sqrt((e ** 2).mean()), "mae_log": np.abs(e).mean(),
            "rmse_vix": np.sqrt((ev ** 2).mean()), "mae_vix": np.abs(ev).mean(),
            "bias_log": e.mean(),
            "acc_direzionale": (dy[ok] == df_[ok]).mean() if ok.any() else np.nan,
            "r2_oos_vs_rw": 1 - (e ** 2).sum() / (eb ** 2).sum(),
            "dm_stat": dm, "dm_p": p}


def build_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Tabella lunga con tutte le previsioni walk-forward (puntuali in log, probabilistiche in [0, 1])."""
    x, c, dates = df["log_close"].to_numpy(), df["close"].to_numpy(), df["date"].to_numpy()
    out = []
    for h in C.HORIZONS:
        for kind, models, series in (("puntuale", B.MODELS, x), ("probabilistica", B.PROB_MODELS, c)):
            for name, fn in models.items():
                o, f = B.walk_forward(series, fn, h)
                out.append(pd.DataFrame({
                    "tipo": kind, "modello": name, "h": h, "data_origine": dates[o], "data_obiettivo": dates[o + h],
                    "x0": x[o], "vix0": c[o], "y": x[o + h] if kind == "puntuale" else (c[o + h] > C.EVENT_THRESHOLD).astype(float),
                    "f": f}))
    fc = pd.concat(out, ignore_index=True)
    fc["fascia_origine"] = np.array(C.BAND_LABELS)[B.band_index(fc["vix0"])]
    yr = pd.to_datetime(fc["data_obiettivo"]).dt.year
    fc["sottoperiodo"] = ""
    for a, b in C.SUBPERIODS:
        fc.loc[(yr >= a) & (yr <= b), "sottoperiodo"] = f"{a}-{b}"
    return fc


def _with_bench(fc: pd.DataFrame, bench: str) -> pd.DataFrame:
    b = fc.loc[fc["modello"] == bench, ["h", "data_origine", "f"]].rename(columns={"f": "f_bench"})
    return fc.merge(b, on=["h", "data_origine"], how="left", validate="many_to_one")


def point_tables(fc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = _with_bench(fc[fc["tipo"] == "puntuale"], BENCH)
    order = {m: i for i, m in enumerate(B.MODELS)}

    def table(keys):
        rows = [dict(zip(keys, k if isinstance(k, tuple) else (k,))) | point_metrics(g, int(g["h"].iloc[0]))
                for k, g in p.groupby(keys, sort=False)]
        t = pd.DataFrame(rows)
        t["_o"] = t["modello"].map(order)
        return t.sort_values([k for k in keys if k != "modello"] + ["_o"], kind="stable").drop(columns="_o").reset_index(drop=True)

    reg = table(["h", "fascia_origine", "modello"])
    reg["_b"] = reg["fascia_origine"].map({b: i for i, b in enumerate(C.BAND_LABELS)})
    reg = reg.sort_values(["h", "_b"], kind="stable").drop(columns="_b").reset_index(drop=True)
    return {"baseline_metriche": table(["h", "modello"]),
            "baseline_metriche_sottoperiodi": table(["h", "sottoperiodo", "modello"]),
            "baseline_metriche_regime": reg}


def prob_metrics(g: pd.DataFrame) -> dict:
    y, p, pc = g["y"].to_numpy(), g["f"].to_numpy(), g["f_bench"].to_numpy()
    bs, bs_c = ((p - y) ** 2).mean(), ((pc - y) ** 2).mean()
    return {"n": len(g), "frequenza_evento": y.mean(), "prob_media": p.mean(), "brier": bs,
            "brier_climatologia": bs_c, "brier_skill_vs_climatologia": 1 - bs / bs_c if bs_c > 0 else np.nan}


def prob_tables(fc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = _with_bench(fc[fc["tipo"] == "probabilistica"], "climatologia")
    order = {m: i for i, m in enumerate(B.PROB_MODELS)}

    def table(keys):
        t = pd.DataFrame([dict(zip(keys, k if isinstance(k, tuple) else (k,))) | prob_metrics(g)
                          for k, g in p.groupby(keys, sort=False)])
        t["_o"] = t["modello"].map(order)
        return t.sort_values([k for k in keys if k != "modello"] + ["_o"], kind="stable").drop(columns="_o").reset_index(drop=True)

    reg = table(["h", "fascia_origine", "modello"])
    reg["_b"] = reg["fascia_origine"].map({b: i for i, b in enumerate(C.BAND_LABELS)})
    reg = reg.sort_values(["h", "_b"], kind="stable").drop(columns="_b").reset_index(drop=True)

    # Curva di calibrazione del modello markoviano: 10 classi di probabilita' di ampiezza 0.1
    m = p[p["modello"] == "markov_fasce"].copy()
    m["classe"] = np.minimum((m["f"] * 10).astype(int), 9)
    cal = (m.groupby(["h", "classe"]).agg(n=("y", "size"), prob_media=("f", "mean"), freq_osservata=("y", "mean"))
           .reset_index())
    cal["classe"] = cal["classe"].map(lambda k: f"{k / 10:.1f}-{(k + 1) / 10:.1f}")
    cal["se_binomiale"] = np.sqrt(cal["freq_osservata"] * (1 - cal["freq_osservata"]) / cal["n"])
    return {"prob_metriche": table(["h", "modello"]), "prob_metriche_sottoperiodi": table(["h", "sottoperiodo", "modello"]),
            "prob_metriche_regime": reg, "prob_calibrazione": cal}


def figures(fc: pd.DataFrame, tabs: dict[str, pd.DataFrame]) -> None:
    cal = tabs["prob_calibrazione"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, h in zip(axes, C.HORIZONS):
        c = cal[(cal["h"] == h) & (cal["n"] >= 5)]      # le classi con meno di 5 previsioni restano solo nel CSV
        ax.plot([0, 1], [0, 1], color="#7f8c8d", ls="--", lw=0.8, label="calibrazione perfetta")
        ax.errorbar(c["prob_media"], c["freq_osservata"], yerr=1.96 * c["se_binomiale"], fmt="o-", color="#1f4e79",
                    capsize=2, ms=4, label="catena di Markov sulle fasce")
        for _, r in c.iterrows():
            ax.annotate(int(r["n"]), (r["prob_media"], r["freq_osservata"]), xytext=(4, -9), textcoords="offset points", fontsize=7)
        ax.set(title=f"Calibrazione P(VIX > {C.EVENT_THRESHOLD:.0f}), h = {h} (etichette = n. previsioni)",
               xlabel="probabilità prevista", ylabel="frequenza osservata", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "07_calibrazione.png")
    plt.close(fig)

    # Errore quadratico cumulato relativo al random walk: dove (nel tempo) i modelli guadagnano o perdono
    p = _with_bench(fc[fc["tipo"] == "puntuale"], BENCH)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, h in zip(axes, C.HORIZONS):
        for name, col in (("media_storica", "#7f8c8d"), ("ar1", "#b03a2e"), ("har", "#1f4e79")):
            g = p[(p["h"] == h) & (p["modello"] == name)].sort_values("data_obiettivo")
            ax.plot(pd.to_datetime(g["data_obiettivo"]), ((g["y"] - g["f_bench"]) ** 2 - (g["y"] - g["f"]) ** 2).cumsum(),
                    label=name, color=col, lw=1)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"Σ (e²_RW − e²_modello), h = {h} (sopra zero: meglio del RW)")
        ax.legend(frameon=False)
    axes[0].set_ylim(-3, None)
    axes[1].set_ylim(-6, None)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "08_errore_cumulato_vs_rw.png")
    plt.close(fig)


def run() -> None:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.DEV_FILE, parse_dates=["date"])
    fc = build_forecasts(df)
    fc.to_csv(C.REPORTS_DIR / "baseline_previsioni.csv", index=False, date_format="%Y-%m-%d", float_format="%.6f")
    tabs = point_tables(fc) | prob_tables(fc)
    for name, t in tabs.items():
        t.to_csv(C.REPORTS_DIR / f"{name}.csv", index=False, float_format="%.6g")
    figures(fc, tabs)
    p = fc[fc["tipo"] == "puntuale"]
    info = {"min_train_settimane": C.MIN_TRAIN_WEEKS, "schema": "finestra espandibile, ristima a ogni passo",
            "benchmark": BENCH,
            "per_orizzonte": {str(h): {"n_previsioni": int(((p["h"] == h) & (p["modello"] == BENCH)).sum()),
                                       "prima_data_obiettivo": str(p.loc[p["h"] == h, "data_obiettivo"].min())[:10],
                                       "ultima_data_obiettivo": str(p.loc[p["h"] == h, "data_obiettivo"].max())[:10]}
                              for h in C.HORIZONS}}
    (C.REPORTS_DIR / "baseline_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    run()
