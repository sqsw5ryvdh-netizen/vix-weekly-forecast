"""Rigenera le misure della Fase 2b sul campione di sviluppo:  python -m src.run_phase2b

Walk-forward per h = 1, 2, 3, 4 (chiusura, massimo, minimo), grafico G3 e report. Legge solo il file di sviluppo,
sotto `guardia_dati()`. Il programma di previsione vero e proprio e' `python -m src.forecast`, che legge queste misure.
"""
import json
import platform
from importlib.metadata import version

from . import config as C
from .fase1 import comune as K
from .fase2b import report, valuta


def main() -> None:
    with K.guardia_dati():
        for name, fn in (("1/2 walk-forward di sviluppo per h = 1-4 e grafico G3 (alcuni minuti)", valuta.run), ("2/2 report", report.run)):
            print(f"[fase 2b] {name} ...", flush=True)
            fn()
        manifest = {"python": platform.python_version(), "pacchetti": {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")},
                    "file_dati_letto": K.NOME_FILE_DATI, "guardia_holdout_attiva": True}
        (valuta.OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fase 2b] completata. Report: {C.REPORTS_DIR / 'fase2b_report.md'}")


if __name__ == "__main__":
    main()
