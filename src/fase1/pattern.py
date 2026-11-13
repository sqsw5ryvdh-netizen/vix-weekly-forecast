"""Fase 1, Parte A: pattern storici (descrittiva, campione di sviluppo).

Ogni stima riporta il numero di casi e un intervallo di confidenza bootstrap al 95%:
- unita' = episodi o periodi  -> bootstrap i.i.d. sulle unita';
- unita' = settimane (dipendenti) -> bootstrap a blocchi di 26 settimane consecutive.
`high` e `low` sono usati solo in A2 (grandezza `escursione`, esplorativa). `open` non e' mai letto.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from .. import config as C
from ..eda import ar1_fit, spells
from . import comune as K

BLUE, RED, GREY = "#1f4e79", "#b03a2e", "#7f8c8d"
KS = (1, 2, 4, 8, 13, 26)
MEASURES = {"livello": "livello (chiusura)", "var4": "variazione log a 4 settimane", "var12": "variazione log a 12 settimane",
            "sd12": "dev. std. delle variazioni (12 sett.)", "escursione": "escursione media ln(high/low), 12 sett. [esplorativo]",
            "sett_da_prec": "settimane dall'episodio precedente"}
TERC = ("basso", "medio", "alto")


# ------------------------------------------------------------------ utilita'

def runs(flag: np.ndarray) -> list[tuple[int, int]]:
    return [(s, s + d - 1) for st, s, d in spells(np.asarray(flag).astype(int)) if st == 1]


def summarize(values, tag: str, stat: str = "mediana") -> dict:
    """n, stima (mediana o media), IC bootstrap i.i.d., quartili. I valori mancanti sono contati a parte."""
    v = np.asarray(values, dtype=float)
    ok = v[np.isfinite(v)]
    f = np.median if stat == "mediana" else np.mean
    out = {"n": int(len(ok)), "n_non_disponibili": int(len(v) - len(ok)), "statistica": stat,
           "stima": float(f(ok)) if len(ok) else np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
           "q25": float(np.quantile(ok, 0.25)) if len(ok) else np.nan, "q75": float(np.quantile(ok, 0.75)) if len(ok) else np.nan}
    if len(ok) >= 2:
        out["ci_lo"], out["ci_hi"] = K.boot_iid(len(ok), lambda i: f(ok[i]), tag)
    return out


def save(t: pd.DataFrame, name: str) -> pd.DataFrame:
    t.to_csv(K.OUT / f"{name}.csv", index=False, float_format="%.6g", date_format="%Y-%m-%d")
    return t


# ------------------------------------------------------------------ A1: dopo i picchi

def episodes(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    close, x, dates = df["close"].to_numpy(), df["log_close"].to_numpy(), df["date"].to_numpy()
    n = len(df)
    if kind == "soglia30":
        flag = close >= C.STRESS_THRESHOLD
    else:
        med = pd.Series(close).rolling(K.PRE_WIN).median().shift(1).to_numpy()   # mediana delle 26 settimane PRECEDENTI
        flag = close >= 1.5 * med                                               # False finche' la mediana non esiste
    merged: list[list[int]] = []
    for s, e in runs(flag):
        if merged and s - merged[-1][1] - 1 <= K.MERGE_GAP:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    ar_full = ar1_fit(x)
    rows = []
    for j, (s, e) in enumerate(merged):
        p = s + int(np.argmax(close[s:e + 1]))
        peak = close[p]
        n_base = min(K.PRE_WIN, s)
        base = float(np.median(close[s - n_base:s])) if n_base >= 4 else np.nan
        rise = peak - base
        ok_rise = bool(np.isfinite(rise) and rise >= K.MIN_RISE)
        after = close[p + 1:]

        def first(cond):
            w = np.flatnonzero(cond)
            return float(w[0] + 1) if len(w) else np.nan

        t50 = first(after <= peak - 0.5 * rise) if ok_rise else np.nan
        t100 = first(after <= base) if ok_rise else np.nan
        end_desc = int(min(p + (t100 if np.isfinite(t100) else K.TRAJ_WEEKS), p + K.TRAJ_WEEKS, n - 1))
        pct = close[p + 1:end_desc + 1] / close[p:end_desc] - 1
        cls = next((lab for lo, hi, lab in K.PEAK_CLASSES if lo <= peak < hi), "<30")
        r = {"definizione": kind, "id": j, "s": s, "e": e, "p": p, "inizio": dates[s], "fine": dates[e], "data_picco": dates[p],
             "sett_sopra_soglia": int(flag[s:e + 1].sum()), "picco": peak, "classe": cls, "base": base, "n_sett_base": n_base,
             "salita": rise, "quota_definita": ok_rise, "sett_recupero_50": t50, "sett_recupero_100": t100,
             "sett_sotto_25": first(after < 25), "sett_sotto_20": first(after < 20),
             "rimbalzi_oltre_20pct": int((pct > K.REBOUND).sum()), "sett_discesa_osservate": int(end_desc - p)}
        rt = ar1_fit(x[:p + 1]) if p + 1 >= 104 else None      # AR(1) stimato solo sulla storia fino al picco
        for k in KS:
            has = p + k < n
            r[f"punti_k{k}"] = close[p + k] - peak if has else np.nan
            r[f"quota_k{k}"] = (peak - close[p + k]) / rise if has and ok_rise else np.nan
            r[f"ar_intero_k{k}"] = x[p + k] - (ar_full["mu_log"] + ar_full["phi"] ** k * (x[p] - ar_full["mu_log"])) if has else np.nan
            r[f"ar_storia_k{k}"] = x[p + k] - (rt["mu_log"] + rt["phi"] ** k * (x[p] - rt["mu_log"])) if has and rt else np.nan
        r["rt_par"] = (rt["mu_log"], rt["phi"]) if rt else (np.nan, np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def trajectories(df: pd.DataFrame, ep: pd.DataFrame) -> pd.DataFrame:
    close, x = df["close"].to_numpy(), df["log_close"].to_numpy()
    rows = []
    for r in ep.itertuples():
        mu, phi = r.rt_par
        for k in range(0, K.TRAJ_WEEKS + 1):
            if r.p + k >= len(close):
                break
            c = close[r.p + k]
            rows.append({"definizione": r.definizione, "id": r.id, "classe": r.classe, "k": k, "close": c,
                         "punti_dal_picco": c - r.picco, "quota_recuperata": (r.picco - c) / r.salita if r.quota_definita else np.nan,
                         "scarto_log_da_ar1_storia": x[r.p + k] - (mu + phi ** k * (x[r.p] - mu))})
    return pd.DataFrame(rows)


def a1(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    eps, trs, summ, curve = {}, [], [], []
    for kind in K.EP_DEFS:
        ep = episodes(df, kind)
        eps[kind] = ep
        tr = trajectories(df, ep)
        trs.append(tr)
        groups = [("tutti", ep)] + [(c, ep[ep["classe"] == c]) for c in ("<30", "30-40", "40-50", ">=50") if (ep["classe"] == c).any()]
        cols = ([(f"quota_k{k}", "mediana") for k in KS] + [(f"punti_k{k}", "mediana") for k in KS]
                + [(c, "mediana") for c in ("sett_recupero_50", "sett_recupero_100", "sett_sotto_25", "sett_sotto_20")]
                + [("rimbalzi_oltre_20pct", "media"), ("almeno_un_rimbalzo", "media")]
                + [(f"ar_intero_k{k}", "mediana") for k in KS] + [(f"ar_storia_k{k}", "mediana") for k in KS])
        for cname, g in groups:
            g = g.assign(almeno_un_rimbalzo=(g["rimbalzi_oltre_20pct"] > 0).astype(float))
            for col, st in cols:
                summ.append({"definizione": kind, "classe": cname, "misura": col} | summarize(g[col], f"a1|{kind}|{cname}|{col}", st))
        for col in ("quota_recuperata", "punti_dal_picco", "scarto_log_da_ar1_storia"):   # curve per la figura
            for k in range(1, K.TRAJ_WEEKS + 1):
                curve.append({"definizione": kind, "misura": col, "k": k}
                             | summarize(tr.loc[tr["k"] == k, col], f"a1c|{kind}|{col}|{k}"))
    ep_all = pd.concat(eps.values(), ignore_index=True).drop(columns="rt_par")
    out = {"episodi": save(ep_all, "a1_episodi"), "traiettorie": save(pd.concat(trs, ignore_index=True), "a1_traiettorie"),
           "sintesi": save(pd.DataFrame(summ), "a1_sintesi"), "curve": save(pd.DataFrame(curve), "a1_curve")}
    cnt = (ep_all.groupby(["definizione", "classe"]).agg(n_episodi=("id", "size"), quota_definita=("quota_definita", "sum"))
           .reset_index())
    out["classi"] = save(cnt, "a1_classi")
    return out


def fig_a1(A1: dict) -> None:
    tr, cv = A1["traiettorie"], A1["curve"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for j, kind in enumerate(K.EP_DEFS):
        for i, (col, lab) in enumerate((("punti_dal_picco", "punti VIX rispetto al picco"), ("quota_recuperata", "quota della salita recuperata"))):
            ax = axes[i, j]
            t = tr[tr["definizione"] == kind]
            for _, g in t.groupby("id"):
                ax.plot(g["k"], g[col], color=GREY, lw=0.6, alpha=0.55)
            c = cv[(cv["definizione"] == kind) & (cv["misura"] == col)]
            ax.fill_between(c["k"], c["q25"], c["q75"], color=BLUE, alpha=0.22, lw=0, label="quartili")
            ax.plot(c["k"], c["stima"], color=BLUE, lw=2, label="mediana")
            if col == "quota_recuperata":
                ax.axhline(1, color=RED, lw=0.7, ls="--")
                ax.axhline(0, color="black", lw=0.5)
                ax.set_ylim(-0.6, 1.6)
            n_ep = t["id"].nunique() if col == "punti_dal_picco" else t.loc[t["quota_recuperata"].notna(), "id"].nunique()
            ax.set_title(f"{K.EP_DEFS[kind]} — n = {n_ep}", fontsize=9)
            ax.set_ylabel(lab)
            if i == 1:
                ax.set_xlabel("settimane dal picco")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(K.FIG / "a1_traiettorie.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, kind in zip(axes, K.EP_DEFS):
        c = cv[(cv["definizione"] == kind) & (cv["misura"] == "scarto_log_da_ar1_storia")]
        ax.fill_between(c["k"], c["ci_lo"], c["ci_hi"], color=BLUE, alpha=0.22, lw=0, label="IC 95% bootstrap")
        ax.plot(c["k"], c["stima"], color=BLUE, lw=1.8, label="mediana")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{K.EP_DEFS[kind]} (n = {int(c['n'].max())})", fontsize=9)
        ax.set_xlabel("settimane dal picco")
    axes[0].set_ylabel("log reale − log previsto dall'AR(1)\n(sotto zero: discesa più rapida)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(K.FIG / "a1_confronto_ar1.png")
    plt.close(fig)


# ------------------------------------------------------------------ A2: prima dei picchi

def week_measures(df: pd.DataFrame, ep: pd.DataFrame, kind: str) -> pd.DataFrame:
    close, x = df["close"].to_numpy(), df["log_close"].to_numpy()
    n = len(df)
    d = np.diff(x, prepend=np.nan)
    rng_hl = np.log(df["high"].to_numpy() / df["low"].to_numpy())          # [esplorativo]
    t = pd.DataFrame({"definizione": kind, "idx": np.arange(n), "date": df["date"].to_numpy(), "livello": close})
    t["var4"] = pd.Series(x).diff(4).to_numpy()
    t["var12"] = pd.Series(x).diff(12).to_numpy()
    t["sd12"] = pd.Series(d).rolling(12).std().to_numpy()
    t["escursione"] = pd.Series(rng_hl).rolling(12).mean().to_numpy()
    in_ep, last_end = np.zeros(n, bool), np.full(n, np.nan)
    starts = ep["s"].to_numpy()
    for r in ep.itertuples():
        in_ep[r.s:r.e + 1] = True
        last_end[r.e + 1:] = r.e
    t["sett_da_prec"] = np.arange(n) - last_end
    t["in_episodio"] = in_ep
    t["pre_inizio"] = np.isin(np.arange(n) + 1, starts)
    t["evento_4sett"] = [bool(((starts > i) & (starts <= i + 4)).any()) for i in range(n)]
    t["ammissibile"] = (~in_ep) & (np.arange(n) >= 13) & (np.arange(n) + 4 <= n - 1)
    return t


def tercile_stats(w: pd.DataFrame, m: str, tag: str) -> tuple[list[dict], dict]:
    """Frequenza di inizio episodio entro 4 settimane per terzile della grandezza m (settimane ammissibili)."""
    g = w[w["ammissibile"] & w[m].notna()]
    v, ev, bl = g[m].to_numpy(), g["evento_4sett"].to_numpy().astype(float), K.blocks(g["idx"].to_numpy())
    if len(v) < 30:
        return [], {"stima": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_settimane": int(len(v))}
    q = np.quantile(v, [1 / 3, 2 / 3])
    terc = np.searchsorted(q, v, side="right")
    freq = lambda i, k: ev[i][terc[i] == k].mean() if (terc[i] == k).any() else np.nan
    rows = []
    for k in range(3):
        lo, hi = K.boot_cluster(bl, lambda i: freq(i, k), f"{tag}|{k}")
        rows.append({"terzile": TERC[k], "soglia_inf": [v.min(), q[0], q[1]][k], "soglia_sup": [q[0], q[1], v.max()][k],
                     "n_settimane": int((terc == k).sum()), "n_settimane_con_evento": int(ev[terc == k].sum()),
                     "frequenza": float(ev[terc == k].mean()), "ci_lo": lo, "ci_hi": hi})
    all_i = np.arange(len(v))
    lo, hi = K.boot_cluster(bl, lambda i: freq(i, 2) - freq(i, 0), f"{tag}|diff")
    return rows, {"stima": float(freq(all_i, 2) - freq(all_i, 0)), "ci_lo": lo, "ci_hi": hi, "n_settimane": int(len(v)),
                  "frequenza_base": float(ev.mean())}


def a2(df: pd.DataFrame, ep_all: pd.DataFrame) -> dict[str, pd.DataFrame]:
    weeks, comp, terc, diffs = [], [], [], []
    for kind in K.EP_DEFS:
        ep = ep_all[ep_all["definizione"] == kind]
        w = week_measures(df, ep, kind)
        weeks.append(w)
        pre = w[w["pre_inizio"] & (w["idx"] >= 13)]
        oth = w[w["ammissibile"] & ~w["pre_inizio"]]
        for m in MEASURES:
            a, b = pre[m].dropna().to_numpy(), oth[m].dropna()
            bv, bl = b.to_numpy(), K.blocks(oth.loc[b.index, "idx"].to_numpy())
            groups = [np.flatnonzero(bl == g) for g in np.unique(bl)]
            rng = K._rng(f"a2cmp|{kind}|{m}")
            sims = []
            for _ in range(K.B_BOOT):          # episodi ricampionati i.i.d., altre settimane a blocchi
                ia = rng.integers(0, len(a), len(a))
                ib = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
                sims.append(np.median(a[ia]) - np.median(bv[ib]))
            lo, hi = K._ci(sims)
            srt = np.sort(bv)
            pr = 100 * np.searchsorted(srt, a, side="right") / len(srt)
            plo, phi = K.boot_iid(len(a), lambda i: pr[i].mean(), f"a2pr|{kind}|{m}")
            comp.append({"definizione": kind, "grandezza": m, "n_episodi": int(len(a)), "n_altre_settimane": int(len(bv)),
                         "mediana_pre_episodio": float(np.median(a)), "mediana_altre": float(np.median(bv)),
                         "differenza": float(np.median(a) - np.median(bv)), "ci_lo": lo, "ci_hi": hi,
                         "percentile_medio_pre": float(pr.mean()), "percentile_ci_lo": plo, "percentile_ci_hi": phi,
                         "mann_whitney_p": float(stats.mannwhitneyu(a, bv).pvalue)})
            rows, dd = tercile_stats(w, m, f"a2t|{kind}|{m}")
            terc += [{"definizione": kind, "grandezza": m} | r for r in rows]
            diffs.append({"definizione": kind, "grandezza": m, "n_episodi": int(len(ep))} | dd)
    return {"settimane": save(pd.concat(weeks, ignore_index=True), "a2_settimane"), "confronto": save(pd.DataFrame(comp), "a2_confronto"),
            "terzili": save(pd.DataFrame(terc), "a2_terzili"), "terzili_differenza": save(pd.DataFrame(diffs), "a2_terzili_differenza")}


def fig_a2(A2: dict) -> None:
    w, tz = A2["settimane"], A2["terzili"]
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.6))
    for i, kind in enumerate(K.EP_DEFS):
        g = w[w["definizione"] == kind]
        for j, m in enumerate(MEASURES):
            ax = axes[i, j]
            a = g.loc[g["pre_inizio"] & (g["idx"] >= 13), m].dropna()
            b = g.loc[g["ammissibile"] & ~g["pre_inizio"], m].dropna()
            ax.boxplot([b, a], widths=0.6, showfliers=False, medianprops={"color": RED})
            ax.scatter(np.full(len(a), 2.0), a, s=8, color=BLUE, alpha=0.6, zorder=3)
            ax.set_xticks([1, 2], [f"altre\n(n={len(b)})", f"prima\n(n={len(a)})"], fontsize=7)
            ax.set_title(m, fontsize=8)
            if j == 0:
                ax.set_ylabel(kind)
    fig.suptitle("Settimana precedente l'inizio di un episodio contro tutte le altre settimane", fontsize=10)
    fig.tight_layout()
    fig.savefig(K.FIG / "a2_prima_dei_picchi.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 6, figsize=(15, 5.2), sharey="row")
    for i, kind in enumerate(K.EP_DEFS):
        for j, m in enumerate(MEASURES):
            ax = axes[i, j]
            t = tz[(tz["definizione"] == kind) & (tz["grandezza"] == m)]
            ax.bar(range(len(t)), t["frequenza"], color=BLUE, alpha=0.8,
                   yerr=[t["frequenza"] - t["ci_lo"], t["ci_hi"] - t["frequenza"]], ecolor=GREY, capsize=2)
            ax.set_xticks(range(len(t)), t["terzile"], fontsize=7)
            ax.set_title(m, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{kind}\nfreq. inizio entro 4 sett.", fontsize=8)
    fig.tight_layout()
    fig.savefig(K.FIG / "a2_terzili.png")
    plt.close(fig)


# ------------------------------------------------------------------ A3: periodi calmi

CALM_DEFS = {"sotto15": "chiusura < 15 per almeno 8 settimane consecutive",
             "escursione_bassa": "escursione del log-VIX su 12 settimane nel 20% più basso della storia"}
HAZ_BUCKETS = {"sotto15": ((8, 12, "8-12"), (13, 25, "13-25"), (26, 10 ** 6, ">=26")),
               "escursione_bassa": ((1, 4, "1-4"), (5, 12, "5-12"), (13, 10 ** 6, ">=13"))}


def calm_periods(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    close, x, dates = df["close"].to_numpy(), df["log_close"].to_numpy(), df["date"].to_numpy()
    n = len(df)
    if kind == "sotto15":
        rr = [(s, e) for s, e in runs(close < K.CALM_LEVEL) if e - s + 1 >= K.CALM_MIN_WEEKS]
    else:
        win = sliding_window_view(x, K.CALM_RANGE_WIN)
        r12 = np.full(n, np.nan)
        r12[K.CALM_RANGE_WIN - 1:] = win.max(axis=1) - win.min(axis=1)
        flag = np.zeros(n, bool)
        for i in range(K.CALM_RANGE_MIN_HIST, n):      # soglia = quantile 20% della storia fino a i (espandibile)
            flag[i] = r12[i] <= np.quantile(r12[K.CALM_RANGE_WIN - 1:i + 1], K.CALM_RANGE_Q)
        rr = runs(flag)
    rows = []
    for j, (s, e) in enumerate(rr):
        cens = e == n - 1
        g = lambda k: x[e + k] - x[e] if e + k < n else np.nan
        rows.append({"definizione": kind, "id": j, "s": s, "e": e, "inizio": dates[s], "fine": dates[e], "durata": e - s + 1,
                     "censurato": cens, "min": close[s:e + 1].min(), "max": close[s:e + 1].max(),
                     "banda_punti": close[s:e + 1].max() - close[s:e + 1].min(), "banda_log": x[s:e + 1].max() - x[s:e + 1].min(),
                     "var_1": g(1), "var_2": g(2), "var_4": g(4),
                     "uscita_con_rialzo_oltre_20pct": float(close[e + 1] / close[e] - 1 > K.REBOUND) if e + 1 < n else np.nan,
                     "uscita_al_rialzo": float(x[e + 1] > x[e]) if e + 1 < n else np.nan,
                     "prosecuzione_1_4": x[e + 4] - x[e + 1] if e + 4 < n else np.nan})
    return pd.DataFrame(rows)


def hazard_weeks(per: pd.DataFrame, kind: str, n: int, dates) -> pd.DataFrame:
    rows = []
    lo_min = HAZ_BUCKETS[kind][0][0]
    for r in per.itertuples():
        for i in range(r.s, r.e + 1):
            d, left = i - r.s + 1, r.e - i
            if d < lo_min or (left <= 3 and r.censurato):
                continue
            rows.append({"definizione": kind, "id": r.id, "idx": i, "date": dates[i], "durata_trascorsa": d,
                         "classe_durata": next(lab for a, b, lab in HAZ_BUCKETS[kind] if a <= d <= b),
                         "fine_entro_4": float(left <= 3)})
    return pd.DataFrame(rows)


def hazard_stats(hw: pd.DataFrame, kind: str, tag: str) -> tuple[list[dict], dict]:
    labs = [b[2] for b in HAZ_BUCKETS[kind]]
    if hw.empty:
        return [], {}
    cl, ev, ids = hw["classe_durata"].to_numpy(), hw["fine_entro_4"].to_numpy(), hw["id"].to_numpy()
    freq = lambda i, lab: ev[i][cl[i] == lab].mean() if (cl[i] == lab).any() else np.nan
    rows = []
    for lab in labs:
        m = cl == lab
        lo, hi = K.boot_cluster(ids, lambda i: freq(i, lab), f"{tag}|{lab}")
        rows.append({"classe_durata": lab, "n_settimane": int(m.sum()), "n_periodi": int(len(np.unique(ids[m]))),
                     "prob_fine_entro_4": float(ev[m].mean()) if m.any() else np.nan, "ci_lo": lo, "ci_hi": hi})
    a = np.arange(len(ev))
    lo, hi = K.boot_cluster(ids, lambda i: freq(i, labs[-1]) - freq(i, labs[0]), f"{tag}|diff")
    # confronto a due classi (tutte le durate oltre la prima classe contro la prima): regge anche con pochi periodi lunghi
    f2 = lambda i: (ev[i][cl[i] != labs[0]].mean() - ev[i][cl[i] == labs[0]].mean()
                    if ((cl[i] != labs[0]).any() and (cl[i] == labs[0]).any()) else np.nan)
    lo2, hi2 = K.boot_cluster(ids, f2, f"{tag}|diff2")
    return rows, {"stima": float(freq(a, labs[-1]) - freq(a, labs[0])), "ci_lo": lo, "ci_hi": hi,
                  "confronto": f"{labs[-1]} meno {labs[0]}", "n_periodi": int(len(np.unique(ids))),
                  "stima_2classi": float(f2(a)), "ci_lo_2classi": lo2, "ci_hi_2classi": hi2,
                  "confronto_2classi": f"oltre {labs[0]} meno {labs[0]}",
                  "n_periodi_oltre_prima_classe": int(len(np.unique(ids[cl != labs[0]])))}


def a3(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pers, hws, summ, haz, hd = [], [], [], [], []
    for kind in CALM_DEFS:
        per = calm_periods(df, kind)
        pers.append(per)
        hw = hazard_weeks(per, kind, len(df), df["date"].to_numpy())
        hws.append(hw)
        for col, st in (("durata", "mediana"), ("banda_punti", "mediana"), ("banda_log", "mediana"), ("var_1", "mediana"),
                        ("var_2", "mediana"), ("var_4", "mediana"), ("prosecuzione_1_4", "mediana"),
                        ("uscita_con_rialzo_oltre_20pct", "media"), ("uscita_al_rialzo", "media")):
            summ.append({"definizione": kind, "misura": col} | summarize(per[col], f"a3|{kind}|{col}", st)
                        | {"min": float(per[col].min()), "max": float(per[col].max())})
        rows, dd = hazard_stats(hw, kind, f"a3h|{kind}")
        haz += [{"definizione": kind} | r for r in rows]
        hd.append({"definizione": kind} | dd)
    return {"periodi": save(pd.concat(pers, ignore_index=True), "a3_periodi"), "settimane": save(pd.concat(hws, ignore_index=True), "a3_settimane"),
            "sintesi": save(pd.DataFrame(summ), "a3_sintesi"), "durata": save(pd.DataFrame(haz), "a3_durata"),
            "durata_differenza": save(pd.DataFrame(hd), "a3_durata_differenza")}


def fig_a3(A3: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for i, kind in enumerate(CALM_DEFS):
        per = A3["periodi"][A3["periodi"]["definizione"] == kind]
        hz = A3["durata"][A3["durata"]["definizione"] == kind]
        axes[i, 0].hist(per["durata"], bins=20, color=BLUE, alpha=0.8)
        axes[i, 0].set_title(f"Durata dei periodi (n = {len(per)})", fontsize=9)
        axes[i, 0].set_xlabel("settimane")
        axes[i, 0].set_ylabel(kind)
        data = [per[c].dropna() for c in ("var_1", "var_2", "var_4")]
        axes[i, 1].boxplot(data, showfliers=True, medianprops={"color": RED}, flierprops={"markersize": 3})
        axes[i, 1].axhline(0, color="black", lw=0.5)
        axes[i, 1].set_xticks([1, 2, 3], ["+1", "+2", "+4"])
        axes[i, 1].set_title("Variazione log dopo l'ultima settimana del periodo", fontsize=9)
        axes[i, 1].set_xlabel("settimane dopo")
        axes[i, 2].bar(range(len(hz)), hz["prob_fine_entro_4"], color=BLUE, alpha=0.8,
                       yerr=[hz["prob_fine_entro_4"] - hz["ci_lo"], hz["ci_hi"] - hz["prob_fine_entro_4"]], ecolor=GREY, capsize=3)
        axes[i, 2].set_xticks(range(len(hz)), [f"{a}\n(n={b} sett.)" for a, b in zip(hz["classe_durata"], hz["n_settimane"])], fontsize=8)
        axes[i, 2].set_title("P(fine entro 4 settimane) per durata già trascorsa", fontsize=9)
    fig.tight_layout()
    fig.savefig(K.FIG / "a3_periodi_calmi.png")
    plt.close(fig)


# ------------------------------------------------------------------ A4: tabelle condizionate

def a4_weeks(df: pd.DataFrame) -> pd.DataFrame:
    x, close = df["log_close"].to_numpy(), df["close"].to_numpy()
    n = len(x)
    t = pd.DataFrame({"idx": np.arange(n), "date": df["date"].to_numpy(), "close": close, "var4": pd.Series(x).diff(4).to_numpy()})
    t = t[t["idx"] >= 4].copy()
    t["fascia"] = K.band7(t["close"])
    t["direzione"] = K.direction(t["var4"])
    for h in K.HORIZONS:
        t[f"var_h{h}"] = (pd.Series(x).shift(-h) - pd.Series(x)).to_numpy()[4:]
    return t


def cell_stats(v: np.ndarray, idx: np.ndarray, tag: str) -> dict:
    ok = np.isfinite(v)
    v, idx = v[ok], idx[ok]
    if len(v) == 0:
        return {"n": 0}
    lo, hi = K.boot_cluster(K.blocks(idx), lambda i: np.median(v[i]), tag) if len(v) >= 5 else (np.nan, np.nan)
    q = np.quantile(v, [0.1, 0.25, 0.75, 0.9])
    return {"n": int(len(v)), "mediana": float(np.median(v)), "ci_lo": lo, "ci_hi": hi, "q10": q[0], "q25": q[1],
            "q75": q[2], "q90": q[3], "quota_rialzi": float((v > 0).mean())}


def a4(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    w = a4_weeks(df)
    rows = []
    for h in K.HORIZONS:
        for b, bl in enumerate(K.BAND7_LABELS):
            for d, dl in list(enumerate(K.DIR_LABELS)) + [(-1, "tutte")]:
                m = (w["fascia"] == b) & ((w["direzione"] == d) | (d == -1))
                rows.append({"h": h, "fascia": bl, "direzione": dl}
                            | cell_stats(w.loc[m, f"var_h{h}"].to_numpy(), w.loc[m, "idx"].to_numpy(), f"a4|{h}|{b}|{d}"))
    return {"settimane": save(w, "a4_settimane"), "celle": save(pd.DataFrame(rows), "a4_celle")}


def fig_a4(A4: dict) -> None:
    c = A4["celle"]
    c = c[c["direzione"] != "tutte"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for j, h in enumerate(K.HORIZONS):
        g = c[c["h"] == h]
        med = g.pivot(index="fascia", columns="direzione", values="mediana").reindex(index=K.BAND7_LABELS, columns=K.DIR_LABELS)
        cnt = g.pivot(index="fascia", columns="direzione", values="n").reindex(index=K.BAND7_LABELS, columns=K.DIR_LABELS)
        lim = np.nanmax(np.abs(med.to_numpy()))
        for i, (tab, cmap, kw, fmt) in enumerate(((med, "RdBu_r", {"vmin": -lim, "vmax": lim}, "{:+.3f}"), (cnt, "Greys", {"vmin": 0}, "{:.0f}"))):
            ax = axes[i, j]
            ax.imshow(tab.to_numpy(dtype=float), cmap=cmap, aspect="auto", **kw)
            ax.set_xticks(range(3), K.DIR_LABELS)
            ax.set_yticks(range(7), K.BAND7_LABELS)
            ax.grid(False)
            for a in range(7):
                for b in range(3):
                    val, nn = tab.iloc[a, b], cnt.iloc[a, b]
                    if np.isfinite(val):
                        ax.text(b, a, fmt.format(val) + ("*" if i == 0 and nn < 20 else ""), ha="center", va="center", fontsize=8,
                                color="white" if ((i == 1 and val > 0.6 * np.nanmax(cnt.to_numpy())) or (i == 0 and abs(val) > 0.6 * lim)) else "black")
            ax.set_title(("Mediana della variazione log" if i == 0 else "Numero di casi") + f", h = {h}", fontsize=9)
            if j == 0:
                ax.set_ylabel("fascia di livello")
            if i == 1:
                ax.set_xlabel("direzione nelle ultime 4 settimane")
    fig.suptitle("Variazione successiva per fascia e direzione recente (* = meno di 20 casi)", fontsize=10)
    fig.tight_layout()
    fig.savefig(K.FIG / "a4_tabelle_condizionate.png")
    plt.close(fig)


# ------------------------------------------------------------------ A5: verifica sulle due meta'

def a5(A1: dict, A2: dict, A3: dict, A4: dict) -> pd.DataFrame:
    split = pd.Timestamp(K.SPLIT_DATE)
    halves = lambda t, col: (t[pd.to_datetime(t[col]) < split], t[pd.to_datetime(t[col]) >= split])
    rows = []

    def add(pid, blocco, pattern, unita, est_fn, nota=""):
        """est_fn(meta') -> (stima, ci_lo, ci_hi, n_casi)."""
        r = {"id": pid, "blocco": blocco, "pattern": pattern, "unita_dei_casi": unita, "nota": nota}
        ok = []
        for k in (1, 2):
            est, lo, hi, n = est_fn(k)
            r |= {f"n_{k}": int(n), f"stima_{k}": est, f"ci_lo_{k}": lo, f"ci_hi_{k}": hi}
            ok.append((n, est, lo, hi))
        if min(o[0] for o in ok) < 10:
            r["verdetto"] = "dati insufficienti"
        else:
            excl = [np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0) for _, _, lo, hi in ok]
            same = np.sign(ok[0][1]) == np.sign(ok[1][1]) and ok[0][1] != 0
            r["verdetto"] = "regge" if (all(excl) and same) else "non regge"
        rows.append(r)

    def med(t, col, tag):
        s = summarize(t[col], tag)
        return s["stima"], s["ci_lo"], s["ci_hi"], s["n"]

    ep = A1["episodi"]
    for kind in K.EP_DEFS:
        h = halves(ep[ep["definizione"] == kind], "data_picco")
        add(f"A1-{kind}-quota4", "A1", f"[{kind}] quota della salita recuperata 4 settimane dopo il picco (mediana) > 0", "episodi",
            lambda k: med(h[k - 1], "quota_k4", f"a5|{kind}|q4|{k}"), "in parte meccanico: il picco è il massimo dell'episodio")
        add(f"A1-{kind}-quota13", "A1", f"[{kind}] quota della salita recuperata 13 settimane dopo il picco (mediana) > 0", "episodi",
            lambda k: med(h[k - 1], "quota_k13", f"a5|{kind}|q13|{k}"), "in parte meccanico: il picco è il massimo dell'episodio")
        add(f"A1-{kind}-ar4", "A1", f"[{kind}] discesa più rapida dell'AR(1): log reale − log previsto a 4 settimane (mediana) < 0", "episodi",
            lambda k: med(h[k - 1], "ar_storia_k4", f"a5|{kind}|ar4|{k}"), "AR(1) stimato sulla storia fino al picco; picco noto solo a posteriori")
    wk = A2["settimane"]
    for kind in K.EP_DEFS:
        w = wk[wk["definizione"] == kind]
        hw = halves(w, "date")
        he = halves(ep[ep["definizione"] == kind], "inizio")
        for m in MEASURES:
            def f(k, m=m):
                _, d = tercile_stats(hw[k - 1], m, f"a5t|{kind}|{m}|{k}")
                return d["stima"], d["ci_lo"], d["ci_hi"], len(he[k - 1])
            add(f"A2-{kind}-{m}", "A2", f"[{kind}] frequenza di inizio episodio entro 4 settimane: terzile alto − terzile basso di «{m}»",
                "episodi iniziati nella metà", f, "terzili ricalcolati dentro ciascuna metà"
                + ("; in parte meccanico: per superare 30 bisogna essere vicini a 30" if (kind == "soglia30" and m == "livello") else ""))
    per, hz = A3["periodi"], A3["settimane"]
    for kind in CALM_DEFS:
        hp = halves(per[per["definizione"] == kind], "fine")
        hh = halves(hz[hz["definizione"] == kind], "date")
        add(f"A3-{kind}-var4", "A3", f"[{kind}] variazione log nelle 4 settimane dopo la fine del periodo (mediana)", "periodi",
            lambda k: med(hp[k - 1], "var_4", f"a5|{kind}|v4|{k}"),
            "in parte meccanico per «sotto15»: il periodo finisce con una chiusura ≥ 15" if kind == "sotto15" else "")
        add(f"A3-{kind}-prosecuzione", "A3", f"[{kind}] prosecuzione dopo l'uscita: variazione log dalla 1ª alla 4ª settimana dopo la fine (mediana)",
            "periodi", lambda k: med(hp[k - 1], "prosecuzione_1_4", f"a5|{kind}|pr|{k}"))

        def fh(k, kind=kind):
            _, d = hazard_stats(hh[k - 1], kind, f"a5h|{kind}|{k}")
            return d.get("stima_2classi", np.nan), d.get("ci_lo_2classi", np.nan), d.get("ci_hi_2classi", np.nan), d.get("n_periodi_oltre_prima_classe", 0)
        b = HAZ_BUCKETS[kind]
        add(f"A3-{kind}-durata", "A3", f"[{kind}] P(fine entro 4 settimane): durata oltre {b[0][2]} settimane meno durata {b[0][2]}",
            "periodi che superano la prima classe di durata", fh)
    w4 = A4["settimane"]
    h4 = halves(w4, "date")

    def cell(k, mask_fn, col, tag):
        g = h4[k - 1]
        g = g[mask_fn(g)]
        s = cell_stats(g[col].to_numpy(), g["idx"].to_numpy(), f"{tag}|{k}")
        return s.get("mediana", np.nan), s.get("ci_lo", np.nan), s.get("ci_hi", np.nan), s["n"]

    def dirdiff(k, col, tag):
        g = h4[k - 1].dropna(subset=[col])
        v, d, bl = g[col].to_numpy(), g["direzione"].to_numpy(), K.blocks(g["idx"].to_numpy())
        f = lambda i: np.median(v[i][d[i] == 2]) - np.median(v[i][d[i] == 0]) if ((d[i] == 2).any() and (d[i] == 0).any()) else np.nan
        lo, hi = K.boot_cluster(bl, f, f"{tag}|{k}")
        return float(f(np.arange(len(v)))), lo, hi, int(min((d == 2).sum(), (d == 0).sum()))

    for h in K.HORIZONS:
        add(f"A4-basso-h{h}", "A4", f"partendo da VIX < 15: variazione log a {h} settimane (mediana) > 0", "settimane",
            lambda k, h=h: cell(k, lambda g: g["close"] < 15, f"var_h{h}", f"a5c|lo|{h}"), "settimane sovrapposte: casi indipendenti molto meno numerosi")
        add(f"A4-alto-h{h}", "A4", f"partendo da VIX ≥ 25: variazione log a {h} settimane (mediana) < 0", "settimane",
            lambda k, h=h: cell(k, lambda g: g["close"] >= 25, f"var_h{h}", f"a5c|hi|{h}"), "settimane sovrapposte: casi indipendenti molto meno numerosi")
        add(f"A4-direzione-h{h}", "A4", f"effetto della direzione recente: mediana a {h} settimane dopo «in rialzo» meno dopo «in calo»", "settimane (gruppo meno numeroso)",
            lambda k, h=h: dirdiff(k, f"var_h{h}", f"a5d|{h}"), "tutte le fasce insieme")
    return save(pd.DataFrame(rows), "a5_verdetti")


def run() -> None:
    K.OUT.mkdir(parents=True, exist_ok=True)
    K.FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    df = K.load_dev()
    A1 = a1(df)
    fig_a1(A1)
    A2 = a2(df, A1["episodi"])
    fig_a2(A2)
    A3 = a3(df)
    fig_a3(A3)
    A4 = a4(df)
    fig_a4(A4)
    a5(A1, A2, A3, A4)
    info = {"repliche_bootstrap": K.B_BOOT, "blocco_settimane": K.BLOCK, "intervallo": "percentile 95%",
            "definizioni_episodio": K.EP_DEFS, "definizioni_periodo_calmo": CALM_DEFS, "salita_minima_punti": K.MIN_RISE,
            "divisione_a5": K.SPLIT_DATE, "campione": [str(df["date"].min().date()), str(df["date"].max().date())], "n_settimane": int(len(df))}
    (K.OUT / "a_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with K.guardia_dati():
        run()
