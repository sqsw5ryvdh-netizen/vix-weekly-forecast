"""Rigenera tutte le misure della Fase 9D-1 (VIX9D settimanale):  python -m src.vix9d.run_fase1

1. candele settimanali e audit dai file giornalieri Cboe (se gli originali non ci sono, usa i file gia' presenti in data/vix9d);
2. analisi compatta e confronto con il VIX (Parte A);
3. walk-forward di chiusura, fasce, probabilita', direzione, estremi, candele e composizione delle versioni G1 e GS (Parti B-E);
4. verifica visiva sul passato (12 origini);
5. report e piano d'esame (solo scritto: l'holdout non viene aperto).
I passi 2-5 leggono solo i file di sviluppo, sotto la guardia. Dopo aver registrato il codice: `python -m src.vix9d.congela`.
"""
import json
import platform
from importlib.metadata import version

from .. import config as C
from ..fase1 import comune as K
from . import analisi, dati, piano, prepara, report, valuta, verifica
from . import config as G


def main() -> None:
    print("[vix9d] 1/6 candele settimanali e audit ...", flush=True)
    ricostruiti = prepara.main([])
    with dati.guardia():
        passi = (("2/6 Parte A: analisi e confronto con il VIX", analisi.run), ("3/6 Parti B-E: walk-forward, regole, versioni (circa 2 minuti)", valuta.run),
                 ("4/6 verifica visiva sul passato", verifica.run), ("5/6 report", report.run), ("6/6 piano d'esame (solo scritto)", piano.run))
        for nome, fn in passi:
            print(f"[vix9d] {nome} ...", flush=True)
            fn()
        manifest = {"python": platform.python_version(), "pacchetti": {p: version(p) for p in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")},
                    "file_dati_letti_dalle_misure": [G.FILE_DEV.name, K.NOME_FILE_DATI], "candele_ricostruite_dagli_originali": bool(ricostruiti), "guardia_holdout_attiva": True}
        (G.OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[vix9d] completata. Report: {G.OUT / 'fase1_report.md'}; piano d'esame: {G.OUT / 'piano_esame_holdout.md'}")


if __name__ == "__main__":
    main()
