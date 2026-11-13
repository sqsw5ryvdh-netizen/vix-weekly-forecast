"""Rigenera tutti i risultati della Fase 2 (Parti A e B e report):  python -m src.run_phase2

Richiede data/processed/vix_weekly_dev.csv. La Parte C (dati esterni) e' un comando separato,
`python -m src.fetch_external`: il report ne legge solo l'inventario (reports/fase2/c_*.json).
Tutta l'esecuzione avviene sotto `guardia_dati()`.
"""
import json
import platform
import random
from importlib.metadata import version

import numpy as np

from . import config as C
from .fase1 import comune as K
from .fase2 import densita as D
from .fase2 import report, scheda, valuta


def main() -> None:
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    steps = (("1/3 Parte A: modelli di densità ed eventi (alcuni minuti)", valuta.run), ("2/3 Parte B: scheda di previsione", scheda.run),
             ("3/3 report", report.run))
    with K.guardia_dati():
        for name, fn in steps:
            print(f"[fase 2] {name} ...", flush=True)
            fn()
        manifest = {"seed": C.SEED, "python": platform.python_version(),
                    "pacchetti": {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "arch")},
                    "file_dati_letto": K.NOME_FILE_DATI, "guardia_holdout_attiva": True, "min_errori_passati": D.MIN_ERR,
                    "orizzonti": list(K.HORIZONS)}
        (valuta.OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fase 2] completata. Report: {C.REPORTS_DIR / 'fase2_report.md'}")


if __name__ == "__main__":
    main()
