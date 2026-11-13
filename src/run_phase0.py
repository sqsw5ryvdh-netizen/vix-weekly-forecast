"""Rigenera tutti i risultati della Fase 0:  python -m src.run_phase0"""
import json
import platform
import random
from importlib.metadata import version

import numpy as np

from . import config as C
from . import data_audit, eda, evaluate, report


def main() -> None:
    random.seed(C.SEED)
    np.random.seed(C.SEED)      # la Fase 0 e' deterministica; il seed e' fissato per le fasi successive
    steps = (("1/4 audit dei dati", data_audit.run), ("2/4 analisi esplorativa", eda.run),
             ("3/4 baseline walk-forward", evaluate.run), ("4/4 report", report.run))
    for name, fn in steps:
        print(f"[fase 0] {name} ...", flush=True)
        fn()
    manifest = {"seed": C.SEED, "python": platform.python_version(),
                "pacchetti": {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")},
                "holdout_settimane": C.HOLDOUT_WEEKS, "min_train_settimane": C.MIN_TRAIN_WEEKS}
    (C.REPORTS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fase 0] completata. Report: {C.REPORTS_DIR / 'fase0_report.md'}")


if __name__ == "__main__":
    main()
