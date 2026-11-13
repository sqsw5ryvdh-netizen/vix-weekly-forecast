"""Rigenera tutti i risultati della Fase 1:  python -m src.run_phase1

Richiede data/processed/vix_weekly_dev.csv (prodotto da `python -m src.run_phase0`).
Tutta l'esecuzione avviene sotto `guardia_dati()`: aprire il file di holdout, il file clean
o il file grezzo solleva un errore.
"""
import json
import platform
import random
from importlib.metadata import version

import numpy as np

from . import config as C
from .fase1 import comune as K
from .fase1 import pattern, report, valuta


def main() -> None:
    random.seed(C.SEED)
    np.random.seed(C.SEED)      # i bootstrap usano generatori propri derivati da C.SEED
    steps = (("1/3 Parte A: pattern storici", pattern.run), ("2/3 Parte B: modelli walk-forward", valuta.run),
             ("3/3 report", report.run))
    with K.guardia_dati():
        for name, fn in steps:
            print(f"[fase 1] {name} ...", flush=True)
            fn()
        manifest = {"seed": C.SEED, "python": platform.python_version(),
                    "pacchetti": {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")},
                    "file_dati_letto": K.NOME_FILE_DATI, "guardia_holdout_attiva": True,
                    "repliche_bootstrap": K.B_BOOT, "orizzonti": list(K.HORIZONS)}
        (K.OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fase 1] completata. Report: {C.REPORTS_DIR / 'fase1_report.md'}")


if __name__ == "__main__":
    main()
