"""Rigenera le misure della Fase 4b sul campione di sviluppo:  python -m src.run_phase4b

Prima candela (settimana +1): candidati per la direzione (Parte A), scelta della sfidante FS (Parte B), costruzione
K4 contro K2 (Parte C), minimo M1q contro M0 (Parte D); poi il report e il piano d'esame aggiornato (solo scritto:
l'holdout non viene aperto). Richiede i risultati di `run_phase2b`, `run_phase3` e `run_phase4`.
Legge solo il file di sviluppo, sotto `guardia_dati()`. Dopo aver registrato il codice: `python -m src.fase4b.congela`.
"""
import json
import platform
from importlib.metadata import version

from . import config as C
from .fase1 import comune as K
from .fase4b import piano, report, valuta


def main() -> None:
    with K.guardia_dati():
        passi = (("1/3 Parti A-D: direzione, sfidante, corpo e minimo della prima candela (meno di un minuto)", valuta.run),
                 ("2/3 report", report.run), ("3/3 piano d'esame aggiornato (solo scritto)", piano.run))
        for name, fn in passi:
            print(f"[fase 4b] {name} ...", flush=True)
            fn()
        manifest = {"python": platform.python_version(), "pacchetti": {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")},
                    "file_dati_letto": K.NOME_FILE_DATI, "guardia_holdout_attiva": True}
        (valuta.OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fase 4b] completata. Report: {C.REPORTS_DIR / 'fase4b_report.md'}; piano d'esame: {C.REPORTS_DIR / 'piano_esame_holdout.md'}")


if __name__ == "__main__":
    main()
