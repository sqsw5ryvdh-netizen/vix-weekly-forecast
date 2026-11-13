"""Analisi esplorativa della Fase 0. Legge SOLO data/processed/vix_weekly_dev.csv (holdout escluso)."""
from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from . import config as C

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
BLUE, RED, GREY = "#1f4e79", "#b03a2e", "#7f8c8d"

# Etichette descrittive per gli episodi di stress, assegnate in base all'anno-mese del picco.
EPISODE_LABELS = {
    "2001-09": "11 settembre", "2002-07": "Crollo 2002 (WorldCom)", "2002-10": "Crollo 2002",
    "2008-10": "Crisi finanziaria globale", "2008-11": "Crisi finanziaria globale",
    "2010-05": "Flash crash / Grecia", "2011-08": "Debito USA / Eurozona", "2011-09": "Debito USA / Eurozona",
    "2015-08": "Svalutazione yuan", "2018-02": "Volmageddon", "2018-12": "Sell-off dicembre 2018",
    "2020-03": "COVID-19", "2022-03": "Ucraina / inflazione", "2022-06": "Stretta Fed", "2022-09": "Stretta Fed",
}


def load_dev() -> pd.DataFrame:
    return pd.read_csv(C.DEV_FILE, parse_dates=["date", "week_start"])


def band_index(level, thresholds=C.LEVEL_THRESHOLDS) -> np.ndarray:
    """Indice di fascia: 0 = sotto la prima soglia, ..., len(thresholds) = sopra l'ultima (soglia inclusa in alto)."""
    return np.searchsorted(np.asarray(thresholds), np.asarray(level), side="right")


def transition_counts(states: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k))
    np.add.at(m, (states[:-1], states[1:]), 1)
    return m


def ar1_fit(x: np.ndarray) -> dict:
    """OLS di x_t = a + phi x_{t-1} + e. Restituisce anche velocita' di ritorno e half-life."""
    y, z = x[1:], x[:-1]
    X = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = res @ res / (len(y) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    a, phi = beta
    hl = lambda p: float(np.log(0.5) / np.log(p)) if 0 < p < 1 else float("nan")
    return {"a": float(a), "phi": float(phi), "se_phi": float(se[1]), "sigma_res": float(np.sqrt(s2)),
            "mu_log": float(a / (1 - phi)), "mu_vix": float(np.exp(a / (1 - phi))),
            "kappa_sett": float(-np.log(phi)), "half_life_sett": hl(phi),
            "half_life_ic95": [hl(phi - 1.96 * se[1]), hl(min(phi + 1.96 * se[1], 0.99999))], "n": int(len(y))}


def spells(states: np.ndarray) -> list[tuple[int, int, int]]:
    """Periodi consecutivi nello stesso stato: (stato, indice iniziale, durata)."""
    out, start = [], 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            out.append((int(states[start]), start, i - start))
            start = i
    return out


# --------------------------------------------------------------------------- sezioni

def stress_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Episodi = periodi con close >= soglia; periodi separati da <= 4 settimane sono uniti."""
    above = (df["close"] >= C.STRESS_THRESHOLD).to_numpy()
    runs = [(s, s + d - 1) for st, s, d in spells(above.astype(int)) if st == 1]
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] - 1 <= 4:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    rows = []
    close, x = df["close"].to_numpy(), df["log_close"].to_numpy()
    for s, e in merged:
        p = s + int(np.argmax(close[s:e + 1]))
        lo = max(0, p - 26)
        t0 = lo + int(np.argmin(close[lo:p + 1]))           # minimo nelle 26 settimane prima del picco
        half = (x[t0] + x[p]) / 2                           # meta' della salita (in log)
        after = np.where(x[p:] <= half)[0]
        before = np.where(x[:p + 1] <= half)[0]              # ultima settimana sotto la meta' prima del picco
        rows.append({
            "inizio": df["date"].iloc[s].date(), "fine": df["date"].iloc[e].date(),
            "settimane_sopra_soglia": int(above[s:e + 1].sum()),
            "data_picco": df["date"].iloc[p].date(), "picco_close": close[p],
            "etichetta": EPISODE_LABELS.get(df["date"].iloc[p].strftime("%Y-%m"), ""),
            "minimo_precedente": close[t0], "sett_salita_meta_picco": p - int(before[-1]),
            "sett_discesa_picco_meta": int(after[0]) if len(after) else np.nan,
        })
    return pd.DataFrame(rows)


def fig_series(df: pd.DataFrame, ep: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, col, lab in ((axes[0], "close", "VIX (chiusura settimanale)"), (axes[1], "log_close", "log(VIX)")):
        ax.plot(df["date"], df[col], color=BLUE, lw=0.9)
        for _, r in ep.iterrows():
            ax.axvspan(pd.Timestamp(r["inizio"]) - pd.Timedelta(days=4), pd.Timestamp(r["fine"]) + pd.Timedelta(days=3),
                       color=RED, alpha=0.18, lw=0)
        ax.set_ylabel(lab)
    for th in C.LEVEL_THRESHOLDS:
        axes[0].axhline(th, color=GREY, lw=0.5, ls=":")
    for _, r in ep.sort_values("picco_close", ascending=False).head(6).iterrows():
        axes[0].annotate(r["etichetta"] or str(r["data_picco"]), (pd.Timestamp(r["data_picco"]), r["picco_close"]),
                         xytext=(2, 4), textcoords="offset points", ha="left", rotation=30, fontsize=7, color=RED)
    axes[0].set_title(f"VIX settimanale, campione di sviluppo (holdout escluso). In rosso: episodi con close ≥ {C.STRESS_THRESHOLD:.0f}")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "01_serie_storica.png")
    plt.close(fig)


def distribution(df: pd.DataFrame) -> pd.DataFrame:
    series = {"livello": df["close"], "log": df["log_close"], "dlog": df["dlog_close"].dropna()}
    rows = []
    for name, s in series.items():
        z = (s - s.mean()) / s.std()
        jb = stats.jarque_bera(s)
        q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        rows.append({"serie": name, "n": len(s), "media": s.mean(), "dev_std": s.std(), "asimmetria": stats.skew(s),
                     "curtosi_eccesso": stats.kurtosis(s), "min": s.min(), "q01": q[0.01], "q05": q[0.05],
                     "q25": q[0.25], "mediana": q[0.5], "q75": q[0.75], "q95": q[0.95], "q99": q[0.99],
                     "max": s.max(), "quota_z_sopra_3": (z > 3).mean(), "quota_z_sotto_m3": (z < -3).mean(),
                     "quota_attesa_normale_per_coda": stats.norm.sf(3), "jarque_bera_p": jb.pvalue})
    out = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (name, s) in zip(axes, series.items()):
        ax.hist(s, bins=60, density=True, color=BLUE, alpha=0.75)
        g = np.linspace(s.min(), s.max(), 300)
        ax.plot(g, stats.norm.pdf(g, s.mean(), s.std()), color=RED, lw=1, label="normale")
        ax.set_title(name)
        ax.legend(frameon=False)
    stats.probplot(series["dlog"], dist="norm", plot=axes[3])
    axes[3].set_title("QQ-plot dlog vs normale")
    axes[3].get_lines()[0].set(markersize=2.5, color=BLUE)
    axes[3].get_lines()[1].set(color=RED)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "02_distribuzioni.png")
    plt.close(fig)
    return out


def stationarity(df: pd.DataFrame) -> pd.DataFrame:
    series = {"livello": df["close"], "log": df["log_close"], "dlog": df["dlog_close"].dropna()}
    rows = []
    for name, s in series.items():
        adf_stat, adf_p, adf_lag, *_ = adfuller(s, regression="c", autolag="AIC", result_object=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # il p-value KPSS e' tabulato solo in [0.01, 0.10]
            k_stat, k_p, k_lag, _ = kpss(s, regression="c", nlags="auto")
        rows.append({"serie": name, "adf_stat": adf_stat, "adf_p": adf_p, "adf_ritardi": adf_lag,
                     "adf_rifiuta_radice_unitaria_5pct": bool(adf_p < 0.05),
                     "kpss_stat": k_stat, "kpss_p_troncato_0.01_0.10": k_p, "kpss_ritardi": k_lag,
                     "kpss_rifiuta_stazionarieta_5pct": bool(k_p < 0.05)})
    return pd.DataFrame(rows)


def autocorrelation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, d = df["log_close"], df["dlog_close"].dropna()
    series = {"log": x, "dlog": d, "dlog_quadrato": d ** 2, "dlog_assoluto": d.abs()}
    lags = [1, 2, 3, 4, 8, 12, 26, 52]
    rows = []
    for name, s in series.items():
        a, p = acf(s, nlags=52, fft=True), pacf(s, nlags=52, method="ywm")
        rows += [{"serie": name, "lag": k, "acf": a[k], "pacf": p[k], "banda_95": 1.96 / np.sqrt(len(s))} for k in lags]
    lb = []
    for name in ("dlog", "dlog_quadrato", "dlog_assoluto"):
        r = acorr_ljungbox(series[name], lags=[4, 12, 26])
        lb += [{"serie": name, "lag": int(k), "ljung_box_q": v.lb_stat, "p_value": v.lb_pvalue} for k, v in r.iterrows()]

    fig, axes = plt.subplots(2, 3, figsize=(13, 5.5))
    for j, (name, title) in enumerate((("log", "log(VIX)"), ("dlog", "Δlog(VIX)"), ("dlog_quadrato", "Δlog(VIX)²"))):
        plot_acf(series[name], lags=52, ax=axes[0, j], zero=False, color=BLUE, vlines_kwargs={"colors": BLUE})
        plot_pacf(series[name], lags=52, ax=axes[1, j], zero=False, method="ywm", color=BLUE, vlines_kwargs={"colors": BLUE})
        axes[0, j].set_title(f"ACF {title}")
        axes[1, j].set_title(f"PACF {title}")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "03_acf_pacf.png")
    plt.close(fig)
    return pd.DataFrame(rows), pd.DataFrame(lb)


def mean_reversion(df: pd.DataFrame) -> dict:
    x = df["log_close"].to_numpy()
    out = {"ar1_log": ar1_fit(x)}
    # Velocita' di ritorno distinta sopra / sotto la media: dx_t = a + b_sopra*dev+ + b_sotto*dev- + e
    mu = x.mean()
    dev = x[:-1] - mu
    X = np.column_stack([np.ones(len(dev)), np.maximum(dev, 0), np.minimum(dev, 0)])
    y = np.diff(x)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    cov = (res @ res / (len(y) - 3)) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    diff = beta[1] - beta[2]
    se_diff = np.sqrt(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2])
    hl = lambda b: float(np.log(0.5) / np.log(1 + b)) if -1 < b < 0 else float("nan")
    out["asimmetrica"] = {
        "media_log_campionaria": float(mu), "media_vix_equivalente": float(np.exp(mu)),
        "b_sopra_media": float(beta[1]), "se_sopra": float(se[1]), "half_life_sopra_sett": hl(beta[1]),
        "b_sotto_media": float(beta[2]), "se_sotto": float(se[2]), "half_life_sotto_sett": hl(beta[2]),
        "t_differenza": float(diff / se_diff), "p_differenza": float(2 * stats.norm.sf(abs(diff / se_diff))),
    }
    return out


def asymmetry(df: pd.DataFrame, ep: pd.DataFrame) -> dict:
    d = df["dlog_close"].dropna().to_numpy()
    up, dn = d[d > 0], d[d < 0]
    sp = spells(np.sign(d).astype(int))
    run_up = [(n, d[s:s + n].sum()) for st, s, n in sp if st == 1]
    run_dn = [(n, d[s:s + n].sum()) for st, s, n in sp if st == -1]
    return {
        "n": int(len(d)), "quota_settimane_rialzo": float((d > 0).mean()), "quota_settimane_ribasso": float((d < 0).mean()),
        "rialzo_medio": float(up.mean()), "ribasso_medio": float(dn.mean()),
        "rialzo_mediano": float(np.median(up)), "ribasso_mediano": float(np.median(dn)),
        "q95_dlog": float(np.quantile(d, 0.95)), "q05_dlog": float(np.quantile(d, 0.05)),
        "q99_dlog": float(np.quantile(d, 0.99)), "q01_dlog": float(np.quantile(d, 0.01)),
        "max_rialzo": float(d.max()), "max_ribasso": float(d.min()),
        "n_rialzi_oltre_0.30": int((d > 0.30).sum()), "n_ribassi_oltre_0.30": int((d < -0.30).sum()),
        "asimmetria_dlog": float(stats.skew(d)),
        "durata_media_serie_rialzi": float(np.mean([n for n, _ in run_up])),
        "durata_media_serie_ribassi": float(np.mean([n for n, _ in run_dn])),
        "ampiezza_media_serie_rialzi": float(np.mean([c for _, c in run_up])),
        "ampiezza_media_serie_ribassi": float(np.mean([c for _, c in run_dn])),
        "velocita_media_serie_rialzi_per_sett": float(np.sum([c for _, c in run_up]) / np.sum([n for n, _ in run_up])),
        "velocita_media_serie_ribassi_per_sett": float(np.sum([c for _, c in run_dn]) / np.sum([n for n, _ in run_dn])),
        "episodi_n": int(len(ep)),
        "episodi_definizione": "meta' = punto medio in log tra il minimo delle 26 settimane precedenti e il picco; "
                                "salita = settimane dall'ultima chiusura sotto la meta' al picco; discesa = settimane dal picco alla prima chiusura sotto la meta'",
        "episodi_media_sett_salita": float(ep["sett_salita_meta_picco"].mean()),
        "episodi_media_sett_discesa_meta": float(ep["sett_discesa_picco_meta"].mean()),
        "episodi_quota_salita_piu_rapida": float((ep["sett_salita_meta_picco"] < ep["sett_discesa_picco_meta"]).mean()),
        "episodi_quota_discesa_piu_rapida": float((ep["sett_salita_meta_picco"] > ep["sett_discesa_picco_meta"]).mean()),
        "episodi_mediana_sett_salita": float(ep["sett_salita_meta_picco"].median()),
        "episodi_mediana_sett_discesa_meta": float(ep["sett_discesa_picco_meta"].median()),
    }


def regimes(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    close = df["close"].to_numpy()
    out = {}
    quart = np.quantile(close, [0.25, 0.5, 0.75])
    schemes = {
        "soglie": (C.LEVEL_THRESHOLDS, list(C.BAND_LABELS)),
        "quartili": (tuple(quart), [f"Q1 (<{quart[0]:.2f})", f"Q2 ({quart[0]:.2f}-{quart[1]:.2f})",
                                    f"Q3 ({quart[1]:.2f}-{quart[2]:.2f})", f"Q4 (>={quart[2]:.2f})"]),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, (th, labels)) in zip(axes, schemes.items()):
        st = band_index(close, th)
        k = len(labels)
        sp = spells(st)
        freq = []
        for i, lab in enumerate(labels):
            dur = [n for s, _, n in sp if s == i]
            freq.append({"fascia": lab, "settimane": int((st == i).sum()), "frequenza": float((st == i).mean()),
                         "n_periodi": len(dur), "durata_media": float(np.mean(dur)) if dur else np.nan,
                         "durata_mediana": float(np.median(dur)) if dur else np.nan,
                         "durata_massima": int(max(dur)) if dur else 0})
        cnt = transition_counts(st, k)
        P = cnt / cnt.sum(axis=1, keepdims=True)
        out[f"regimi_{name}_frequenze"] = pd.DataFrame(freq)
        out[f"regimi_{name}_transizione"] = pd.DataFrame(P, index=pd.Index(labels, name="da \\ a"), columns=labels).reset_index()
        out[f"regimi_{name}_conteggi"] = pd.DataFrame(cnt.astype(int), index=pd.Index(labels, name="da \\ a"), columns=labels).reset_index()
        ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(k), labels, rotation=30, ha="right")
        ax.set_yticks(range(k), labels)
        ax.grid(False)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if P[i, j] > 0.5 else "black")
        ax.set_title(f"Transizione settimanale tra fasce ({name})")
        ax.set_xlabel("fascia a t+1")
        ax.set_ylabel("fascia a t")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "04_matrici_transizione.png")
    plt.close(fig)
    return out


def seasonality(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df.dropna(subset=["dlog_close"]).copy()
    d["mese"] = d["date"].dt.month
    df = df.assign(mese=df["date"].dt.month)
    g = d.groupby("mese")["dlog_close"]
    tab = pd.DataFrame({"n": g.size(), "dlog_media": g.mean(), "dlog_se": g.std() / np.sqrt(g.size()),
                        "dlog_t": g.mean() / (g.std() / np.sqrt(g.size())),
                        "quota_rialzi": g.apply(lambda s: (s > 0).mean()),
                        "livello_mediano": df.groupby("mese")["close"].median()}).reset_index()
    groups = [s.to_numpy() for _, s in g]
    kw, an = stats.kruskal(*groups), stats.f_oneway(*groups)
    lev = stats.levene(*groups)
    test = {"kruskal_wallis_stat": float(kw.statistic), "kruskal_wallis_p": float(kw.pvalue),
            "anova_f": float(an.statistic), "anova_p": float(an.pvalue),
            "levene_varianza_stat": float(lev.statistic), "levene_varianza_p": float(lev.pvalue),
            "n_mesi_con_t_oltre_1.96": int((tab["dlog_t"].abs() > 1.96).sum()),
            "soglia_t_bonferroni_12_test": float(stats.norm.isf(0.025 / 12)),
            "n_mesi_significativi_bonferroni": int((tab["dlog_t"].abs() > stats.norm.isf(0.025 / 12)).sum())}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].bar(tab["mese"], tab["dlog_media"], yerr=1.96 * tab["dlog_se"], color=BLUE, alpha=0.8, ecolor=GREY, capsize=2)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_title("Variazione log media settimanale per mese (± 1,96 e.s.)")
    axes[1].bar(tab["mese"], tab["livello_mediano"], color=BLUE, alpha=0.8)
    axes[1].set_title("Livello mediano del VIX per mese")
    for ax in axes:
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("mese")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "05_stagionalita.png")
    plt.close(fig)
    return tab, test


def subperiods(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    blocks = [(f"{a}-{b}", (df["date"].dt.year >= a) & (df["date"].dt.year <= b)) for a, b in C.SUBPERIODS]
    blocks.append(("intero campione", pd.Series(True, index=df.index)))
    for name, m in blocks:
        s = df.loc[m]
        d = s["dlog_close"].dropna()
        ar = ar1_fit(s["log_close"].to_numpy())
        rows.append({"periodo": name, "n": len(s), "vix_medio": s["close"].mean(), "vix_mediano": s["close"].median(),
                     "vix_min": s["close"].min(), "vix_max": s["close"].max(), "dev_std_log": s["log_close"].std(),
                     "dev_std_dlog": d.std(), "asimmetria_dlog": stats.skew(d), "curtosi_eccesso_dlog": stats.kurtosis(d),
                     "phi_ar1": ar["phi"], "half_life_sett": ar["half_life_sett"], "mu_vix_ar1": ar["mu_vix"],
                     "quota_sopra_25": (s["close"] > C.EVENT_THRESHOLD).mean(), "acf1_dlog": d.autocorr(1),
                     "acf1_dlog_quadrato": (d ** 2).autocorr(1)})
    out = pd.DataFrame(rows)

    w = 260
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(df["date"], np.exp(df["log_close"].rolling(w).mean()), color=BLUE)
    axes[0].set_title(f"Statistiche mobili su {w} settimane (~5 anni)")
    axes[0].set_ylabel("media geometrica VIX")
    axes[1].plot(df["date"], df["dlog_close"].rolling(w).std(), color=BLUE)
    axes[1].set_ylabel("dev. std. Δlog")
    x = df["log_close"]
    phi = x.rolling(w).corr(x.shift(1))
    axes[2].plot(df["date"], phi, color=BLUE)
    axes[2].set_ylabel("autocorr. lag 1 di log(VIX)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "06_stabilita.png")
    plt.close(fig)
    return out


def run() -> None:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dev()
    R = C.REPORTS_DIR
    save = lambda t, name: t.to_csv(R / f"{name}.csv", index=False, float_format="%.6g")
    dump = lambda o, name: (R / f"{name}.json").write_text(json.dumps(o, indent=2, ensure_ascii=False))

    ep = stress_episodes(df)
    save(ep, "eda_episodi_stress")
    fig_series(df, ep)
    save(distribution(df), "eda_distribuzioni")
    save(stationarity(df), "eda_stazionarieta")
    a, lb = autocorrelation(df)
    save(a, "eda_acf")
    save(lb, "eda_ljung_box")
    dump(mean_reversion(df), "eda_mean_reversion")
    dump(asymmetry(df, ep), "eda_asimmetria")
    for name, t in regimes(df).items():
        save(t, f"eda_{name}")
    tab, test = seasonality(df)
    save(tab, "eda_stagionalita")
    dump(test, "eda_stagionalita_test")
    save(subperiods(df), "eda_sottoperiodi")


if __name__ == "__main__":
    run()
