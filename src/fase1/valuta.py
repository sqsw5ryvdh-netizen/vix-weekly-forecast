"""Fase 1, Parte B: valutazione walk-forward (stesso protocollo della Fase 0) per h = 1, 4, 12.

Confronto con M1 (tipo HAR): Clark-West per i modelli annidati (M0 dentro M1, M1 dentro M3),
Diebold-Mariano per gli altri; correzione di Holm sulla tabella principale (7 modelli x 3 orizzonti).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .. import baselines as B
from .. import config as C
from ..evaluate import diebold_mariano
from . import comune as K
from . import modelli as M
from . import pattern as P

REF = "M1"
AFTER_PEAK_WEEKS = 12


def clark_west(e_small: np.ndarray, e_big: np.ndarray, f_small: np.ndarray, f_big: np.ndarray, h: int) -> tuple[float, float]:
    """Clark-West (2007) per modelli annidati. H0: il modello grande non migliora il piccolo.

    f_t = e_small^2 - [e_big^2 - (f_small - f_big)^2]; statistica t con varianza Newey-West (h-1 ritardi),
    p-value a una coda dalla normale (alternativa: il modello grande e' migliore).
    """
    f = e_small ** 2 - (e_big ** 2 - (f_small - f_big) ** 2)
    n = len(f)
    if n < 10 or np.allclose(f, 0):
        return float("nan"), float("nan")
    fc = f - f.mean()
    lrv = fc @ fc / n + 2 * sum((1 - k / h) * (fc[k:] @ fc[:-k] / n) for k in range(1, h))
    if lrv <= 0:
        return float("nan"), float("nan")
    stat = f.mean() / np.sqrt(lrv / n)
    return float(stat), float(stats.norm.sf(stat))


def holm(p: np.ndarray) -> np.ndarray:
    """p-value corretti con il metodo di Holm (step-down)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, j in enumerate(order):
        run = max(run, (len(p) - rank) * p[j])
        adj[j] = min(1.0, run)
    return adj


def test_vs_ref(model: str, y, f, f_ref, h: int) -> dict:
    """Statistica con segno: positiva = il modello della riga e' migliore di M1."""
    e, e_ref = y - f, y - f_ref
    if model == REF:
        return {"test": "", "stat": np.nan, "p": np.nan}
    if model in M.NESTED_IN_M1:            # M0 parsimonioso, M1 grande: CW > 0 significa che M1 batte la riga
        s, p = clark_west(e, e_ref, f, f_ref, h)
        return {"test": "CW (una coda; H1: M1 migliore)", "stat": -s, "p": p}
    if model in M.NESTS_M1:                # M1 parsimonioso, riga grande
        s, p = clark_west(e_ref, e, f_ref, f, h)
        return {"test": "CW (una coda; H1: modello migliore)", "stat": s, "p": p}
    s, p = diebold_mariano(e_ref, e, h)
    return {"test": "DM (due code)", "stat": s, "p": p}


def metrics(model: str, g: pd.DataFrame, h: int) -> dict:
    y, f, x0 = g["y"].to_numpy(), g["f"].to_numpy(), g["x0"].to_numpy()
    e, e_rw, e_ref = y - f, y - g["f_M0"].to_numpy(), y - g["f_M1"].to_numpy()
    ev = np.exp(y) - np.exp(f)
    dy, df_ = np.sign(y - x0), np.sign(f - x0)
    ok = (dy != 0) & (df_ != 0)
    return {"n": len(g), "rmse_log": np.sqrt((e ** 2).mean()), "mae_log": np.abs(e).mean(),
            "rmse_vix": np.sqrt((ev ** 2).mean()), "mae_vix": np.abs(ev).mean(), "bias_log": e.mean(),
            "acc_direzionale": (dy[ok] == df_[ok]).mean() if ok.any() else np.nan,
            "r2_oos_vs_rw": 1 - (e ** 2).sum() / (e_rw ** 2).sum(),
            "r2_oos_vs_m1": 1 - (e ** 2).sum() / (e_ref ** 2).sum()} | test_vs_ref(model, y, f, g["f_M1"].to_numpy(), h)


def build_forecasts(df: pd.DataFrame, models: dict | None = None, horizons=K.HORIZONS) -> pd.DataFrame:
    """Tabella larga: una riga per (h, origine), una colonna f_<modello> per ogni modello."""
    models = models or (M.MODELS | M.SENSITIVITY)
    x, c, dates = df["log_close"].to_numpy(), df["close"].to_numpy(), df["date"].to_numpy()
    out = []
    for h in horizons:
        cols = {}
        for name, fn in models.items():
            o, f = B.walk_forward(x, fn, h)
            cols[f"f_{name}"] = f
        out.append(pd.DataFrame({"h": h, "idx_origine": o, "data_origine": dates[o], "data_obiettivo": dates[o + h],
                                 "x0": x[o], "vix0": c[o], "y": x[o + h]} | cols))
    return pd.concat(out, ignore_index=True)


def add_labels(fc: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Etichette per suddividere la valutazione. `dopo_picco` e `periodo_calmo` sono assegnate A POSTERIORI
    (il picco e l'appartenenza a un periodo calmo si conoscono solo dopo): non sono usabili per prevedere."""
    fc = fc.copy()
    fc["fascia_origine"] = np.array(C.BAND_LABELS)[B.band_index(fc["vix0"])]
    yr = pd.to_datetime(fc["data_obiettivo"]).dt.year
    fc["sottoperiodo"] = ""
    for a, b in C.SUBPERIODS:
        fc.loc[(yr >= a) & (yr <= b), "sottoperiodo"] = f"{a}-{b}"
    n = len(df)
    after, calm = np.zeros(n, bool), np.zeros(n, bool)
    for p in P.episodes(df, "soglia30")["p"]:
        after[p:p + AFTER_PEAK_WEEKS + 1] = True
    for r in P.calm_periods(df, "sotto15").itertuples():
        calm[r.s:r.e + 1] = True
    fc["dopo_picco"] = after[fc["idx_origine"].to_numpy()]
    fc["periodo_calmo"] = calm[fc["idx_origine"].to_numpy()]
    return fc


def table(fc: pd.DataFrame, models, group_col: str | None = None) -> pd.DataFrame:
    rows = []
    for h, gh in fc.groupby("h", sort=True):
        groups = [(None, gh)] if group_col is None else list(gh.groupby(group_col, sort=False))
        for key, g in groups:
            for m in models:
                r = {"h": h} | ({group_col: key} if group_col else {}) | {"modello": m}
                rows.append(r | metrics(m, g.assign(f=g[f"f_{m}"]), int(h)))
    return pd.DataFrame(rows)


def figures(fc: pd.DataFrame, main: pd.DataFrame) -> None:
    colors = {"M0": "#7f8c8d", "M2": "#b03a2e", "M3": "#1f4e79", "M4": "#d68910", "M5": "#7d3c98", "M6": "#117a65", "M7": "#000000"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, h in zip(axes, K.HORIZONS):
        g = fc[fc["h"] == h].sort_values("data_obiettivo")
        for m, col in colors.items():
            ax.plot(pd.to_datetime(g["data_obiettivo"]), ((g["y"] - g["f_M1"]) ** 2 - (g["y"] - g[f"f_{m}"]) ** 2).cumsum(),
                    label=m, color=col, lw=1.6 if m == "M7" else 1)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"Σ (e²_M1 − e²_modello), h = {h} (sopra zero: meglio di M1)", fontsize=9)
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(K.FIG / "b_errore_cumulato_vs_m1.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ms = [m for m in M.MODELS if m != REF]
    w = 0.27
    for j, h in enumerate(K.HORIZONS):
        v = main[main["h"] == h].set_index("modello").loc[ms, "r2_oos_vs_m1"]
        ax.bar(np.arange(len(ms)) + (j - 1) * w, 100 * v, w, label=f"h = {h}", color=["#1f4e79", "#5d8fc2", "#b9cfe6"][j])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(ms)), ms)
    ax.set_ylabel("R² fuori campione rispetto a M1 (%)")
    ax.set_title("Guadagno (sopra zero) o perdita rispetto al tipo HAR", fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(K.FIG / "b_r2_vs_m1.png")
    plt.close(fig)


def run() -> None:
    K.OUT.mkdir(parents=True, exist_ok=True)
    K.FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    df = K.load_dev()
    fc = add_labels(build_forecasts(df), df)
    save = lambda t, name: t.to_csv(K.OUT / f"{name}.csv", index=False, float_format="%.6g", date_format="%Y-%m-%d")
    fc.drop(columns="idx_origine").to_csv(K.OUT / "b_previsioni.csv", index=False, float_format="%.6f", date_format="%Y-%m-%d")

    main = table(fc, M.MODELS)
    tested = main["p"].notna()
    main["p_holm"] = np.nan
    main.loc[tested, "p_holm"] = holm(main.loc[tested, "p"].to_numpy())
    main["significativo_holm_5pct"] = main["p_holm"] < 0.05
    save(main, "b_metriche")
    save(table(fc, M.MODELS, "sottoperiodo"), "b_metriche_sottoperiodi")
    reg = table(fc, M.MODELS, "fascia_origine")
    reg["_o"] = reg["fascia_origine"].map({b: i for i, b in enumerate(C.BAND_LABELS)})
    save(reg.sort_values(["h", "_o"], kind="stable").drop(columns="_o"), "b_metriche_fascia")
    subs = []
    for col, yes, no in (("dopo_picco", f"entro {AFTER_PEAK_WEEKS} settimane dopo un picco", "altre origini (non dopo un picco)"),
                         ("periodo_calmo", "dentro un periodo calmo", "altre origini (non in periodo calmo)")):
        t = table(fc, M.MODELS, col)
        t.insert(1, "sottoinsieme", t.pop(col).map({True: yes, False: no}))
        subs.append(t)
    save(pd.concat(subs, ignore_index=True).sort_values(["h", "sottoinsieme"], kind="stable"), "b_metriche_sottoinsiemi")
    sens = table(fc, {"M6_k15": 0, "M6": 0, "M6_k60": 0})
    save(sens.drop(columns=["test", "stat", "p"]), "b_sensibilita_k")
    figures(fc, main)

    info = {"schema": "finestra espandibile, ristima a ogni passo (M4: ultime 520 settimane)", "min_train_settimane": C.MIN_TRAIN_WEEKS,
            "riferimento": REF, "n_test_holm": int(tested.sum()),
            "arma": M.ARMA_STATS | {"ristima": "a ogni origine", "metodo": "massima verosimiglianza esatta (spazio degli stati, statsmodels ARIMA)"},
            "etichette_a_posteriori": {"dopo_picco": f"origine tra 0 e {AFTER_PEAK_WEEKS} settimane dopo il picco di un episodio (chiusura >= 30)",
                                       "periodo_calmo": "origine dentro un periodo con chiusura < 15 per almeno 8 settimane"},
            "per_orizzonte": {str(h): {"n_previsioni": int((fc["h"] == h).sum()),
                                       "prima_data_obiettivo": str(fc.loc[fc["h"] == h, "data_obiettivo"].min())[:10],
                                       "ultima_data_obiettivo": str(fc.loc[fc["h"] == h, "data_obiettivo"].max())[:10],
                                       "n_dopo_picco": int(fc.loc[fc["h"] == h, "dopo_picco"].sum()),
                                       "n_periodo_calmo": int(fc.loc[fc["h"] == h, "periodo_calmo"].sum())} for h in K.HORIZONS}}
    (K.OUT / "b_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with K.guardia_dati():
        run()
