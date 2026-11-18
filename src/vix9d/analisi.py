"""VIX9D, Parte A: analisi compatta del campione di sviluppo e confronto con il VIX sullo stesso periodo.

Solo descrizione, nessun modello. Le statistiche sulle candele (corpo, escursione, apertura) usano solo le settimane con
candela vera. Le analisi «cosa succede dopo» quando il rapporto VIX9D / VIX e' molto alto o molto basso usano i decili
calcolati su tutto il campione di sviluppo: sono descrittive e a posteriori, non una regola di previsione.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..fase2b import grafici
from . import config as G
from . import dati

RITARDI = (1, 2, 4, 8, 12, 26)


def _acf(v: np.ndarray, k: int) -> float:
    v = v - v.mean()
    return float((v[k:] @ v[:-k]) / (v @ v))


def _ar1(x: np.ndarray) -> dict:
    a, phi = np.linalg.lstsq(np.column_stack([np.ones(len(x) - 1), x[:-1]]), x[1:], rcond=None)[0]
    return {"phi": float(phi), "mezza_vita_settimane": float(np.log(0.5) / np.log(phi)) if 0 < phi < 1 else None, "livello_di_equilibrio": float(np.exp(a / (1 - phi)))}


def serie(close: np.ndarray) -> dict:
    """Livello, variazioni settimanali, autocorrelazione e ritorno alla media di una serie di chiusure settimanali."""
    x = np.log(close)
    d, s = np.diff(x), close[1:] / close[:-1] - 1
    return {"n_settimane": int(len(close)),
            "livello": {"media": float(close.mean()), "mediana": float(np.median(close)), "dev_std": float(close.std()), "min": float(close.min()), "max": float(close.max()),
                        "p05": float(np.quantile(close, 0.05)), "p25": float(np.quantile(close, 0.25)), "p75": float(np.quantile(close, 0.75)), "p95": float(np.quantile(close, 0.95))},
            "variazione_log_settimanale": {"media": float(d.mean()), "dev_std": float(d.std()), "mediana_assoluta": float(np.median(np.abs(d))), "asimmetria": float(stats.skew(d)),
                                           "curtosi": float(stats.kurtosis(d, fisher=False)), "p05": float(np.quantile(d, 0.05)), "p95": float(np.quantile(d, 0.95)), "min": float(d.min()), "max": float(d.max())},
            "rialzi_oltre_20": {"n": int((s > 0.20).sum()), "quota": float((s > 0.20).mean())}, "rialzi_oltre_30": {"n": int((s > 0.30).sum()), "quota": float((s > 0.30).mean())},
            "ribassi_oltre_20": {"n": int((s < -0.20).sum()), "quota": float((s < -0.20).mean())},
            "acf_livello_log": {str(k): _acf(x, k) for k in RITARDI}, "acf_variazioni": {str(k): _acf(d, k) for k in (1, 2, 4)}, "ar1": _ar1(x)}


def candele(d: pd.DataFrame) -> dict:
    c = d[d["candela"]] if "candela" in d.columns else d
    o, h, l, cl = (c[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    corpo, esc = np.abs(np.log(cl / o)), np.log(h / l)
    return {"n": int(len(c)), "corpo_log": {"media": float(corpo.mean()), "mediana": float(np.median(corpo))}, "escursione_log": {"media": float(esc.mean()), "mediana": float(np.median(esc))},
            "ombra_superiore_log": float(np.log(h / np.maximum(o, cl)).mean()), "ombra_inferiore_log": float(np.log(np.minimum(o, cl) / l).mean()),
            "quota_chiusura_sopra_apertura": float((cl > o).mean()), "massimo_su_chiusura_precedente_mediana": float(np.median(h[1:] / cl[:-1])),
            "minimo_su_chiusura_precedente_mediana": float(np.median(l[1:] / cl[:-1]))}


def dopo_rapporto(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Variazione del VIX9D nelle 1 e 2 settimane successive, per decile del rapporto VIX9D / VIX all'origine."""
    x, r = df["log_close"].to_numpy(), (df["close"] / df["vix"]).to_numpy()
    dec = pd.qcut(r, 10, labels=False) + 1
    rows = []
    for k in range(1, 11):
        m = dec == k
        row = {"decile": k, "rapporto_min": float(r[m].min()), "rapporto_max": float(r[m].max())}
        for h in G.ORIZZONTI:
            i = np.flatnonzero(m)
            i = i[i + h < len(x)]
            v = x[i + h] - x[i]
            row |= {f"n_{h}": int(len(i)), f"variazione_log_media_{h}": float(v.mean()), f"variazione_log_mediana_{h}": float(np.median(v)), f"quota_rialzo_{h}": float((v > 0).mean())}
        rows.append(row)
    info = {"media": float(r.mean()), "mediana": float(np.median(r)), "p05": float(np.quantile(r, 0.05)), "p25": float(np.quantile(r, 0.25)), "p75": float(np.quantile(r, 0.75)),
            "p95": float(np.quantile(r, 0.95)), "min": float(r.min()), "max": float(r.max()), "quota_sopra_1": float((r > 1).mean()),
            "correlazione_livelli_log": float(np.corrcoef(x, np.log(df["vix"].to_numpy()))[0, 1]), "correlazione_variazioni_log": float(np.corrcoef(np.diff(x), np.diff(np.log(df["vix"].to_numpy())))[0, 1])}
    return pd.DataFrame(rows), info


def figure(df: pd.DataFrame, tab: pd.DataFrame) -> None:
    with plt.rc_context(grafici.STILE):
        fig, (a, b) = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        a.plot(df["date"], df["close"], color=grafici.GIU, lw=0.9, label="VIX9D, chiusura settimanale")
        a.plot(df["date"], df["vix"], color="#9aa0a6", lw=0.9, label="VIX, chiusura settimanale")
        a.set_yscale("log")
        a.set_yticks([10, 15, 20, 30, 50, 80], ["10", "15", "20", "30", "50", "80"])
        a.set_title("VIX9D e VIX: chiusure settimanali nel campione di sviluppo (scala logaritmica)", fontsize=10)
        a.legend(frameon=False)
        b.plot(df["date"], df["close"] / df["vix"], color="#7d3c98", lw=0.8)
        b.axhline(1, color="black", lw=0.7)
        b.set_ylabel("VIX9D / VIX")
        fig.tight_layout()
        fig.savefig(G.FIG / "a_livelli_e_rapporto.png")
        plt.close(fig)

        fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4))
        bins = np.linspace(-0.8, 1.2, 61)
        a.hist(np.diff(df["log_close"]), bins=bins, color=grafici.GIU, alpha=0.75, label="VIX9D")
        a.hist(np.diff(np.log(df["vix"])), bins=bins, color="#9aa0a6", alpha=0.75, label="VIX")
        a.set_title("Variazioni log settimanali della chiusura", fontsize=10)
        a.legend(frameon=False)
        w = 0.38
        for j, h in enumerate(G.ORIZZONTI):
            b.bar(tab["decile"] + (j - 0.5) * w, 100 * tab[f"variazione_log_mediana_{h}"], w, color=["#1f4e79", "#9db9d9"][j], label=f"dopo {h} settiman{'a' if h == 1 else 'e'}")
        b.axhline(0, color="black", lw=0.7)
        b.set_xticks(range(1, 11))
        b.set_xlabel("decile del rapporto VIX9D / VIX all'origine (1 = VIX9D molto sotto il VIX, 10 = molto sopra)")
        b.set_title("Variazione mediana del VIX9D dopo (%, log), per decile del rapporto", fontsize=10)
        b.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(G.FIG / "a_variazioni_e_rapporto.png")
        plt.close(fig)


def run() -> None:
    G.OUT.mkdir(parents=True, exist_ok=True)
    G.FIG.mkdir(parents=True, exist_ok=True)
    df = dati.load_dev()
    tab, rap = dopo_rapporto(df)
    yr = df["date"].dt.year
    sotto = []
    for a, b in G.SOTTOPERIODI:
        g = df[(yr >= a) & (yr <= b)]
        s9, sv = serie(g["close"].to_numpy()), serie(g["vix"].to_numpy())
        sotto.append({"sottoperiodo": f"{a}-{b}", "n": len(g), "vix9d_livello_mediano": s9["livello"]["mediana"], "vix_livello_mediano": sv["livello"]["mediana"],
                      "vix9d_dev_std_variazioni": s9["variazione_log_settimanale"]["dev_std"], "vix_dev_std_variazioni": sv["variazione_log_settimanale"]["dev_std"],
                      "vix9d_phi": s9["ar1"]["phi"], "vix_phi": sv["ar1"]["phi"], "vix9d_rialzi_oltre_20": s9["rialzi_oltre_20"]["quota"], "vix_rialzi_oltre_20": sv["rialzi_oltre_20"]["quota"],
                      "rapporto_mediano": float((g["close"] / g["vix"]).median())})
    vix_c = pd.read_csv(dati.C.DEV_FILE, parse_dates=["date", "week_start"])
    vix_c = vix_c[vix_c["week_start"].isin(df.loc[df["candela"], "week_start"])]
    out = {"vix9d": serie(df["close"].to_numpy()), "vix_stesso_periodo": serie(df["vix"].to_numpy()), "candele_vix9d": candele(df), "candele_vix_stesso_periodo": candele(vix_c), "rapporto_vix9d_su_vix": rap}
    (G.OUT / "a_analisi.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    tab.to_csv(G.OUT / "a_dopo_rapporto.csv", index=False, float_format="%.6g")
    pd.DataFrame(sotto).to_csv(G.OUT / "a_sottoperiodi.csv", index=False, float_format="%.6g")
    figure(df, tab)


if __name__ == "__main__":
    with dati.guardia():
        run()
