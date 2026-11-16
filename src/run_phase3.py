"""Rigenera le misure della Fase 3 sul campione di sviluppo:  python -m src.run_phase3

Parte 0: nuove candele (ampiezze medie, direzione L0, etichetta della chiusura), con pagella e grafico G3.
Blocchi: VIX giornaliero (G1) e dati di mercato (X1-X4, X6, X7) per chiusura (T1), direzione (T2) e fasce (T3).
Richiede reports/fase2b e reports/fase2c (prodotti da `run_phase2b` e `run_phase2c`) e i dati esterni di sviluppo
(data/external/dev, prodotti da `python -m src.fetch_external`). Legge solo le parti di sviluppo, sotto `guardia_dati()`.
"""
import json
import platform
from importlib.metadata import version

from . import config as C
from .fase1 import comune as K
from .fase3 import dati, report, valuta, valuta_candele


def main() -> None:
    with K.guardia_dati():
        passi = (("1/3 Parte 0: nuove candele, pagella e grafico G3 (meno di un minuto)", valuta_candele.run),
                 ("2/3 blocchi G1, X1-X4, X6, X7: walk-forward, valutazione, regole di scelta (qualche minuto)", valuta.run), ("3/3 report", report.run))
        for name, fn in passi:
            print(f"[fase 3] {name} ...", flush=True)
            fn()
        manifest = {"python": platform.python_version(), "pacchetti": {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")},
                    "file_dati_letti": [K.NOME_FILE_DATI, dati.ESTERNI_DEV.name], "guardia_holdout_attiva": True}
        (valuta.OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fase 3] completata. Report: {C.REPORTS_DIR / 'fase3_report.md'}")


if __name__ == "__main__":
    main()
