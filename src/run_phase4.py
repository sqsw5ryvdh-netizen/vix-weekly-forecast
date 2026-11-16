"""Rigenera le misure della Fase 4 sul campione di sviluppo:  python -m src.run_phase4

Parte A: varianti delle ampiezze delle candele (A0, A1, A2), pagella e regola di scelta.
Parte B: modello a regimi N1 e gradient boosting N2, N3 per chiusura e direzione (solo misura).
Parte C: confronto di sviluppo tra le versioni finaliste F1, F2 (e F3, se un modello della Parte B passa la regola).
Poi il report e il piano d'esame sull'holdout (solo scritto: l'holdout non viene aperto).
Richiede i risultati di `run_phase2b`, `run_phase2c`, `run_phase3` e i dati esterni di sviluppo (`python -m src.fetch_external`).
Legge solo le parti di sviluppo, sotto `guardia_dati()`. Dopo aver registrato il codice: `python -m src.fase4.congela`.
"""
import json
import platform
from importlib.metadata import version

from . import config as C
from .fase1 import comune as K
from .fase3 import dati
from .fase4 import piano, report, valuta_a, valuta_b, valuta_c


def main() -> None:
    with K.guardia_dati():
        passi = (("1/5 Parte A: varianti delle ampiezze e pagella delle candele (meno di un minuto)", valuta_a.run),
                 ("2/5 Parte B: modello a regimi e gradient boosting (circa 3 minuti)", valuta_b.run),
                 ("3/5 Parte C: confronto tra le versioni finaliste (circa 3 minuti)", valuta_c.run),
                 ("4/5 report", report.run), ("5/5 piano d'esame (solo scritto)", piano.run))
        for name, fn in passi:
            print(f"[fase 4] {name} ...", flush=True)
            fn()
        manifest = {"python": platform.python_version(), "pacchetti": {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib")},
                    "file_dati_letti": [K.NOME_FILE_DATI, dati.ESTERNI_DEV.name], "guardia_holdout_attiva": True}
        (valuta_a.OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fase 4] completata. Report: {C.REPORTS_DIR / 'fase4_report.md'}; piano d'esame: {C.REPORTS_DIR / 'piano_esame_holdout.md'}")


if __name__ == "__main__":
    main()
