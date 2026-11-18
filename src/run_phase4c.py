"""Rigenera le misure della Fase 4c sul campione di sviluppo:  python -m src.run_phase4c

Massimo e minimo diretti della prima candela con gli insiemi S0-S5 (Parte A), previsione incrociata con l'S&P 500
(Parte B: legame, tetto teorico, versione reale), corpo K5 contro K2 (Parte C), pagella prima e dopo (Parte D); poi il
report e il piano d'esame aggiornato (solo scritto: l'holdout non viene aperto).
Richiede i risultati di `run_phase4b` e i dati esterni di sviluppo (`python -m src.fetch_external`).
Legge solo le parti di sviluppo, sotto `guardia_dati()`. Dopo aver registrato il codice: `python -m src.fase4c.congela`.
"""
import json
import platform
from importlib.metadata import version

from . import config as C
from .fase1 import comune as K
from .fase3 import dati
from .fase4c import piano, report, valuta


def main() -> None:
    with K.guardia_dati():
        passi = (("1/3 Parti A-D: massimo e minimo diretti, previsione incrociata, corpo, pagella (circa 2 minuti)", valuta.run),
                 ("2/3 report", report.run), ("3/3 piano d'esame aggiornato (solo scritto)", piano.run))
        for name, fn in passi:
            print(f"[fase 4c] {name} ...", flush=True)
            fn()
        manifest = {"python": platform.python_version(), "pacchetti": {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")},
                    "file_dati_letti": [K.NOME_FILE_DATI, dati.ESTERNI_DEV.name], "guardia_holdout_attiva": True}
        (valuta.OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fase 4c] completata. Report: {C.REPORTS_DIR / 'fase4c_report.md'}; piano d'esame: {C.REPORTS_DIR / 'piano_esame_holdout.md'}")


if __name__ == "__main__":
    main()
