"""Fase 2b, Parte E: grafici a candele (PNG, testi in italiano).

Candele reali: corpo pieno. Candele previste: corpo vuoto con bordo tratteggiato e colori piu' tenui.
Rosso = chiusura sopra l'apertura (VIX in salita), blu = chiusura sotto l'apertura.
La colonna `open` del file e' usata solo qui, per disegnare le candele reali.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

SU, GIU = "#b03a2e", "#1f4e79"
SU_T, GIU_T = "#d98880", "#7fa8cf"          # previste: toni piu' tenui
FASCIA = "#8e7cc3"
MEDIANA = "#4a2d82"
GRIGIO = "#7f8c8d"
MESI = ["gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic"]
STILE = {"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
         "axes.spines.right": False, "font.size": 9}


def _fmt_mese(v, _pos=None):
    d = mdates.num2date(v)
    return f"{MESI[d.month - 1]} {d.year % 100:02d}"


def _fmt_giorno(v, _pos=None):
    d = mdates.num2date(v)
    return f"{d.day} {MESI[d.month - 1]}"


def n(v: float, d: int = 1) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def candela(ax, x, o, h, l, c, w, prevista: bool = False, z: int = 3) -> None:
    su = c >= o
    if prevista:
        col = SU_T if su else GIU_T
        edge = SU if su else GIU
        ax.vlines(x, l, h, color=edge, lw=1.0, ls=(0, (2, 1.5)), zorder=z)
        ax.add_patch(Rectangle((x - w / 2, min(o, c)), w, max(abs(c - o), 1e-9), facecolor="white", edgecolor=edge, lw=1.3,
                               ls=(0, (2.5, 1.5)), hatch="////", zorder=z + 1, alpha=0.95))
        ax.add_patch(Rectangle((x - w / 2, min(o, c)), w, max(abs(c - o), 1e-9), facecolor=col, edgecolor="none", alpha=0.25, zorder=z + 1))
    else:
        col = SU if su else GIU
        ax.vlines(x, l, h, color=col, lw=0.9, zorder=z)
        ax.add_patch(Rectangle((x - w / 2, min(o, c)), w, max(abs(c - o), 0.02), facecolor=col, edgecolor=col, lw=0.6, zorder=z + 1))


def candele_reali(ax, d: pd.DataFrame, x, w) -> None:
    for xi, r in zip(x, d.itertuples()):
        candela(ax, xi, r.open, r.high, r.low, r.close, w)


def candele_previste(ax, prev: dict, x, w) -> None:
    for xi, s in zip(x, prev["settimane"]):
        candela(ax, xi, s["apertura"], s["massimo"], s["minimo"], s["chiusura"], w, prevista=True)


def fasce(ax, prev: dict, x0, x, livelli=(80, 50)) -> None:
    """Ventaglio delle fasce della chiusura, dietro le candele: parte dall'ultima chiusura reale."""
    xs = [x0] + list(x)
    for lev, a in zip(livelli, (0.13, 0.26)):
        lo = [prev["chiusura_origine"]] + [s["fasce"][str(lev)][0] for s in prev["settimane"]]
        hi = [prev["chiusura_origine"]] + [s["fasce"][str(lev)][1] for s in prev["settimane"]]
        ax.fill_between(xs, lo, hi, color=FASCIA, alpha=a, lw=0, zorder=1)


def linea_mediana(ax, prev: dict, x0, x) -> None:
    """Linea della mediana D4 (valore centrale della chiusura), dall'ultima chiusura reale alla settimana +4."""
    ax.plot([x0] + list(x), [prev["chiusura_origine"]] + [s["quantili"]["q50"] for s in prev["settimane"]], color=MEDIANA, lw=1.4,
            marker="o", ms=3.2, zorder=6)


def legenda(ax, percorsi: bool = False, riferimenti: bool = True, mediana: bool = False, **kw) -> None:
    h = [Patch(facecolor=SU, edgecolor=SU, label="reale: settimana in rialzo"), Patch(facecolor=GIU, edgecolor=GIU, label="reale: settimana in ribasso"),
         Patch(facecolor="white", edgecolor=SU, ls="--", hatch="////", label="PREVISTA (corpo vuoto, bordo tratteggiato)"),
         Patch(facecolor=FASCIA, alpha=0.40, label="fascia della chiusura al 50%"), Patch(facecolor=FASCIA, alpha=0.15, label="fascia della chiusura all'80%")]
    if mediana:
        h.append(Line2D([], [], color=MEDIANA, lw=1.4, marker="o", ms=3.2, label="mediana D4: valore centrale della chiusura"))
    if riferimenti:
        h += [Line2D([], [], color="#d68910", lw=1.2, label="media mobile a 12 settimane"),
              Line2D([], [], color="black", lw=0.9, ls=":", label="mediana storica (fino all'origine)")]
    h.append(Line2D([], [], color="black", lw=1.0, ls="--", label="data di origine"))
    if percorsi:
        h.append(Line2D([], [], color=GRIGIO, lw=0.8, label="casi simili: percorso reale a 4 settimane"))
    ax.legend(handles=h, frameon=False, fontsize=7.5, **kw)


def _date(prev: dict) -> tuple[float, list[float]]:
    x0 = mdates.date2num(pd.Timestamp(prev["data_origine"]))
    return x0, [mdates.date2num(pd.Timestamp(s["data"])) for s in prev["settimane"]]


def _zoom(ax, d: pd.DataFrame, prev: dict, L: dict | None, n_sett: int, percorsi: bool, mediana: bool = False) -> None:
    x0, xp = _date(prev)
    z = d.iloc[-n_sett:]
    fasce(ax, prev, x0, xp)
    if percorsi and L:
        for k in L["casi_simili"]["casi"]:
            ax.plot([x0] + xp, [prev["chiusura_origine"]] + [prev["chiusura_origine"] * (1 + v / 100) for v in k["variazioni_pct"]],
                    color=GRIGIO, lw=0.7, alpha=0.6, zorder=2)
    candele_reali(ax, z, mdates.date2num(z["date"]), 4.2)
    candele_previste(ax, prev, xp, 4.2)
    if mediana:
        linea_mediana(ax, prev, x0, xp)
    ax.axvline(x0, color="black", lw=1.0, ls="--", zorder=2)
    ax.set_xlim(mdates.date2num(z["date"].iloc[0]) - 5, xp[-1] + 9)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.FR, interval=2))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_giorno))
    ax.set_ylabel("VIX")


def g1(d: pd.DataFrame, prev: dict, L: dict, path, settimane: int = 104, mediana: bool = False) -> None:
    """Grafico principale: ultimi 2 anni + 4 candele previste; sotto, ingrandimento con i percorsi dei casi simili."""
    with plt.rc_context(STILE):
        fig, (ax, az) = plt.subplots(2, 1, figsize=(15, 10.5), gridspec_kw={"height_ratios": [3, 2.2]})
        h = d.iloc[-settimane:]
        x0, xp = _date(prev)
        fasce(ax, prev, x0, xp)
        candele_reali(ax, h, mdates.date2num(h["date"]), 4.2)
        candele_previste(ax, prev, xp, 4.2)
        if mediana:
            linea_mediana(ax, prev, x0, xp)
        ax.plot(mdates.date2num(h["date"]), d["close"].rolling(12).mean().iloc[-settimane:], color="#d68910", lw=1.2, zorder=2)
        med = float(d["close"].median())
        ax.axhline(med, color="black", lw=0.9, ls=":", zorder=1)
        ax.axvline(x0, color="black", lw=1.0, ls="--", zorder=2)
        ymax = max(h["high"].max(), max(s["fasce"]["80"][1] for s in prev["settimane"])) * 1.05
        ax.set_ylim(min(h["low"].min(), min(s["fasce"]["80"][0] for s in prev["settimane"])) * 0.93, ymax)
        ax.annotate(f"origine\n{pd.Timestamp(prev['data_origine']).strftime('%d/%m/%Y')}\nchiusura {n(prev['chiusura_origine'], 2)}",
                    (x0, ymax), xytext=(-6, -4), textcoords="offset points", ha="right", va="top", fontsize=8)
        s4 = prev["settimane"][-1]
        ax.annotate((f"settimana +4\ncandela: chiusura {n(s4['chiusura'])}\nmediana D4: {n(s4['quantili']['q50'])}\nfascia 50%: {n(s4['fasce']['50'][0])}–{n(s4['fasce']['50'][1])}"
                     if mediana else f"settimana +4\nchiusura prevista {n(s4['chiusura'])}\nfascia 50%: {n(s4['fasce']['50'][0])}–{n(s4['fasce']['50'][1])}"),
                    (xp[-1], s4["massimo"]), xytext=(10, 14), textcoords="offset points", ha="left", va="bottom", fontsize=8.5,
                    fontweight="bold", arrowprops={"arrowstyle": "-", "color": GRIGIO, "lw": 0.7})
        ax.set_xlim(mdates.date2num(h["date"].iloc[0]) - 6, xp[-1] + 75)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_mese))
        ax.set_ylabel("VIX")
        ax.set_title(f"VIX settimanale: ultimi 2 anni e previsione a 4 settimane dall'origine {pd.Timestamp(prev['data_origine']).strftime('%d/%m/%Y')}", fontsize=11)
        legenda(ax, loc="upper left", ncol=2, mediana=mediana)
        _zoom(az, d, prev, L, 16, percorsi=True, mediana=mediana)
        for xi, s in zip(xp, prev["settimane"]):
            et = n(s["chiusura"]) + (f"\n{s['direzione']} {round(100 * s['prob_direzione'])}%" if "prob_direzione" in s else "")
            az.annotate(et, (xi, s["massimo"]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, color="#222",
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8} if "prob_direzione" in s else None, zorder=8)
        az.set_title(f"Ingrandimento: ultime 16 settimane, previsione e percorsi reali a 4 settimane dei {L['casi_simili']['n']} casi simili "
                     "(riportati al livello attuale come variazione percentuale)", fontsize=9.5)
        legenda(az, percorsi=True, riferimenti=False, mediana=mediana, loc="upper left", ncol=2)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def g2(d: pd.DataFrame, prev: dict, path, mediana: bool = False) -> None:
    """Tutto lo storico: linea delle chiusure (le candele non sarebbero leggibili) e riquadro con le candele previste."""
    with plt.rc_context(STILE):
        fig, ax = plt.subplots(figsize=(15, 6.2))
        x0, xp = _date(prev)
        ax.plot(mdates.date2num(d["date"]), d["close"], color=GIU, lw=0.8, label="chiusura settimanale (storia)")
        ax.plot(mdates.date2num(d["date"]), d["close"].rolling(12).mean(), color="#d68910", lw=0.8, alpha=0.9, label="media mobile a 12 settimane")
        med = float(d["close"].median())
        ax.axhline(med, color="black", lw=0.9, ls=":", label=f"mediana storica ({n(med)})")
        ax.axvline(x0, color="black", lw=1.0, ls="--", label="data di origine")
        fasce(ax, prev, x0, xp)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(mdates.date2num(d["date"].iloc[0]) - 60, xp[-1] + 60)
        ax.set_ylabel("VIX")
        ax.set_title(f"VIX settimanale dal {pd.Timestamp(d['date'].iloc[0]).year}: storia (linea delle chiusure) e previsione dall'origine "
                     f"{pd.Timestamp(prev['data_origine']).strftime('%d/%m/%Y')}", fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ins = ax.inset_axes([0.452, 0.44, 0.275, 0.50])      # zona 2012-2019, dove la linea resta bassa
        _zoom(ins, d, prev, None, 20, percorsi=False, mediana=mediana)
        ins.set_title("ultime 20 settimane (candele reali) e 4 candele previste", fontsize=8)
        ins.tick_params(labelsize=7)
        ins.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.FR, interval=6))
        ins.yaxis.set_major_locator(plt.MultipleLocator(5))
        ins.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: n(v, 0)))
        ins.set_ylabel("")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def g3(pannelli: list[dict], path, riga: str, mediana: bool = False) -> None:
    """Verifica visiva sul passato: per ogni origine le 4 candele previste accanto alle 4 candele reali."""
    with plt.rc_context(STILE):
        fig, axes = plt.subplots(3, 4, figsize=(16, 10.5))
        for ax, p in zip(axes.ravel(), pannelli):
            prev, st, re = p["previsione"], p["storia"], p["reali"]
            xs = np.arange(-len(st) + 1, 1)
            xf = np.arange(1, 5)
            fasce(ax, prev, 0, xf)
            candele_reali(ax, st, xs, 0.55)
            for k, (s, r) in enumerate(zip(prev["settimane"], re.itertuples()), start=1):
                candela(ax, k - 0.2, s["apertura"], s["massimo"], s["minimo"], s["chiusura"], 0.32, prevista=True)
                candela(ax, k + 0.2, r.open, r.high, r.low, r.close, 0.32)
            if mediana:
                linea_mediana(ax, prev, 0, xf)
            ax.axvline(0.5, color="black", lw=0.9, ls="--")
            ax.set_xticks([-6, -3, 0, 1, 2, 3, 4], ["-6", "-3", "0", "+1", "+2", "+3", "+4"])
            ax.set_xlim(-len(st) + 0.3, 4.8)
            esito = ("dentro la fascia al 50%" if p["dentro_50"] else "dentro la fascia all'80%" if p["dentro_80"] else "fuori dalla fascia all'80%")
            ax.set_title(f"origine {pd.Timestamp(prev['data_origine']).strftime('%d/%m/%Y')} — chiusura a +4: {esito}", fontsize=8.5)
        fig.suptitle("Verifica visiva sul passato (campione di sviluppo): candele previste (vuote, a sinistra) accanto alle candele reali (piene, a destra)", fontsize=11)
        legenda(fig, riferimenti=False, mediana=mediana, loc="upper center", ncol=7 if mediana else 6, bbox_to_anchor=(0.5, 0.965))
        fig.text(0.5, 0.012, riga, ha="center", fontsize=10, fontweight="bold")
        fig.tight_layout(rect=(0, 0.03, 1, 0.935))
        fig.savefig(path)
        plt.close(fig)
