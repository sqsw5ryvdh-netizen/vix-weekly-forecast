"""VIX9D: grafici a candele (PNG, testi in italiano). Usa le primitive dei grafici del VIX senza modificarle."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..fase2b import grafici as GR

VIX = "#9aa0a6"


def legenda(ax, vix: bool = True, **kw) -> None:
    h = [Patch(facecolor=GR.SU, edgecolor=GR.SU, label="reale: settimana in rialzo"), Patch(facecolor=GR.GIU, edgecolor=GR.GIU, label="reale: settimana in ribasso")]
    h += [Patch(facecolor=GR.COLORE_ETICHETTA[k], edgecolor="black", ls="--", alpha=0.85, label=f"PREVISTA: chiusura rispetto all'origine «{k}»") for k in ("rialzo", "ribasso", "neutra")]
    h += [Patch(facecolor=GR.FASCIA, alpha=0.40, label="fascia della chiusura al 50%"), Patch(facecolor=GR.FASCIA, alpha=0.15, label="fascia della chiusura all'80%"),
          Line2D([], [], color=GR.MEDIANA, lw=1.4, marker="o", ms=3.2, label="chiusura centrale"), Line2D([], [], color=GR.RECENTE, lw=2.2, label="livello recente: media a 12 settimane")]
    if vix:
        h.append(Line2D([], [], color=VIX, lw=1.1, label="VIX (in secondo piano)"))
    h.append(Line2D([], [], color="black", lw=1.0, ls="--", label="data di origine"))
    ax.legend(handles=h, frameon=False, fontsize=7.5, **kw)


def _previste(ax, prev: dict, xp, w: float) -> None:
    for xi, s in zip(xp, prev["settimane"]):
        GR.candela(ax, xi, s["apertura"], s["massimo"], s["minimo"], s["chiusura_candela"], w, prevista=True, colore=s["colore"])


def _pannello(ax, d: pd.DataFrame, prev: dict, n_sett: int, etichette: bool) -> None:
    z = d.iloc[-n_sett:]
    x0 = mdates.date2num(pd.Timestamp(prev["data_origine"]))
    xp = [mdates.date2num(pd.Timestamp(s["data"])) for s in prev["settimane"]]
    GR.fasce(ax, prev, x0, xp)
    if z["vix"].notna().any():
        ax.plot(mdates.date2num(z["date"]), z["vix"], color=VIX, lw=1.1, zorder=1.5)
    GR.candele_reali(ax, z, mdates.date2num(z["date"]), 4.2)
    _previste(ax, prev, xp, 4.2)
    GR.linea_mediana(ax, prev, x0, xp)
    GR.livello_recente(ax, d, n_sett, x0, xp[-1] + 5, testo=not etichette)
    ax.axvline(x0, color="black", lw=1.0, ls="--", zorder=2)
    if etichette:
        for xi, s in zip(xp, prev["settimane"]):
            ax.annotate("chiusura:\n" + s["etichetta"], (xi, s["massimo"]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, color="#222",
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8}, zorder=8)
    ax.set_ylabel("VIX9D")


def principale(d: pd.DataFrame, prev: dict, path) -> None:
    """Ultimi 2 anni a candele con le 2 candele previste; sotto, ingrandimento sulle ultime 16 settimane."""
    with plt.rc_context(GR.STILE):
        fig, (ax, az) = plt.subplots(2, 1, figsize=(15, 10.5), gridspec_kw={"height_ratios": [3, 2.2]})
        _pannello(ax, d, prev, 104, etichette=False)
        xp = [mdates.date2num(pd.Timestamp(s["data"])) for s in prev["settimane"]]
        ax.set_xlim(mdates.date2num(d["date"].iloc[-104]) - 6, xp[-1] + 70)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(GR._fmt_mese))
        ax.set_title(f"VIX9D settimanale: ultimi 2 anni e previsione a 2 settimane dall'origine {pd.Timestamp(prev['data_origine']).strftime('%d/%m/%Y')} "
                     f"(chiusura {GR.n(prev['chiusura_origine'], 2)})", fontsize=11)
        legenda(ax, loc="upper left", ncol=2)
        _pannello(az, d, prev, 16, etichette=True)
        az.set_xlim(mdates.date2num(d["date"].iloc[-16]) - 5, xp[-1] + 12)
        az.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.FR, interval=2))
        az.xaxis.set_major_formatter(plt.FuncFormatter(GR._fmt_giorno))
        az.set_title("Ingrandimento: ultime 16 settimane e le 2 candele previste (la scritta dice la direzione della chiusura rispetto all'origine)", fontsize=9.5)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def verifica(pannelli: list[dict], path, riga: str) -> None:
    """Verifica visiva sul passato: per ogni origine le 2 candele previste (a sinistra) accanto alle 2 reali (a destra)."""
    with plt.rc_context(GR.STILE):
        fig, axes = plt.subplots(3, 4, figsize=(16, 10.5))
        for ax, p in zip(axes.ravel(), pannelli):
            prev, st, re = p["previsione"], p["storia"], p["reali"]
            xs, xf = np.arange(-len(st) + 1, 1), np.arange(1, 3)
            GR.fasce(ax, prev, 0, xf)
            GR.candele_reali(ax, st, xs, 0.55)
            for k, (s, r) in enumerate(zip(prev["settimane"], re.itertuples()), start=1):
                GR.candela(ax, k - 0.2, s["apertura"], s["massimo"], s["minimo"], s["chiusura_candela"], 0.32, prevista=True, colore=s["colore"])
                GR.candela(ax, k + 0.2, r.open, r.high, r.low, r.close, 0.32)
            GR.linea_mediana(ax, prev, 0, xf)
            ax.axvline(0.5, color="black", lw=0.9, ls="--")
            ax.set_xticks([-6, -3, 0, 1, 2], ["-6", "-3", "0", "+1", "+2"])
            ax.set_xlim(-len(st) + 0.3, 2.8)
            ax.set_title(f"origine {pd.Timestamp(prev['data_origine']).strftime('%d/%m/%Y')}", fontsize=8.5)
        fig.suptitle("VIX9D — verifica visiva sul passato (campione di sviluppo): candele previste (a sinistra, bordo tratteggiato) accanto alle candele reali (a destra)", fontsize=11)
        legenda(fig, vix=False, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.965))
        fig.text(0.5, 0.012, riga, ha="center", fontsize=10, fontweight="bold")
        fig.tight_layout(rect=(0, 0.03, 1, 0.92))
        fig.savefig(path)
        plt.close(fig)


def vista(prev: dict, cand: list[dict]) -> dict:
    """Previsione nel formato dei grafici: le settimane portano la candela, il colore e l'etichetta."""
    return prev | {"settimane": [s | {"apertura": c["apertura"], "massimo": c["massimo"], "minimo": c["minimo"], "chiusura_candela": c["chiusura"], "colore": c["colore"], "etichetta": c["etichetta"]}
                                 for s, c in zip(prev["settimane"], cand)]}
