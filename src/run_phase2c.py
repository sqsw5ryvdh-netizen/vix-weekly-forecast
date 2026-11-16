"""Rigenera le misure della Fase 2c sul campione di sviluppo:  python -m src.run_phase2c

Componenti delle candele, pagella delle quattro costruzioni, regola di scelta, grafico G3 e report.
Richiede reports/fase2b/previsioni_dev.csv (prodotto da `python -m src.run_phase2b`). Legge solo il file di
sviluppo, sotto `guardia_dati()`.
"""
import json
import platform
from importlib.metadata import version

from . import config as C
from .fase1 import comune as K
from .fase2c import report, valuta


def main() -> None:
    with K.guardia_dati():
        for name, fn in (("1/2 componenti, pagella delle candele e grafico G3 (circa un minuto)", valuta.run), ("2/2 report", report.run)):
            print(f"[fase 2c] {name} ...", flush=True)
            fn()
        manifest = {"python": platform.python_version(), "pacchetti": {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")},
                    "file_dati_letto": K.NOME_FILE_DATI, "guardia_holdout_attiva": True}
        (valuta.OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[fase 2c] completata. Report: {C.REPORTS_DIR / 'fase2c_report.md'}")


if __name__ == "__main__":
    main()
