"""Fase 4: congela le versioni finaliste in versioni/*.json:  python -m src.fase4.congela

Per ogni versione scrive tutti i parametri, le finestre e le regole (da `versioni.configurazione`), le scelte fatte
dalle regole della Fase 4 (variante delle ampiezze, modello della Parte B che ha passato la regola), l'impronta
SHA-256 dei file sorgente che definiscono i modelli e l'hash del commit su cui il congelamento e' stato fatto.
E' cio' che verra' esaminato sull'holdout, senza modifiche: un test verifica che i file congelati coincidano con la
configurazione ricostruita dal codice e che i sorgenti non siano cambiati dopo il congelamento.
Va eseguito DOPO `python -m src.run_phase4` e dopo aver registrato il codice nel repository (il commit indicato e'
quello del codice; il file di configurazione viene registrato nel commit successivo).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date

from .. import config as C
from . import versioni as VS

R = C.REPORTS_DIR / "fase4"
VARIABILI_ESECUZIONE = ("congelata_il", "commit", "sorgenti_identici_al_commit")


def impronte() -> dict[str, str]:
    return {f: hashlib.sha256((C.ROOT / f).read_bytes()).hexdigest() for f in VS.SORGENTI_CONGELATI}


def costruisci() -> dict[str, dict]:
    """Configurazione di ogni versione finalista, senza i campi che dipendono dal momento del congelamento."""
    a, b = json.loads((R / "a_info.json").read_text()), json.loads((R / "b_info.json").read_text())
    variante = a["scelta"]["scelta"]
    assert set(b["passa"]) <= {"N1"}, "F3 e' definita solo attorno a N1"
    out = {}
    for v in ("f1", "f2") + (("f3",) if "N1" in b["passa"] else ()):
        cfg = VS.configurazione(v, variante)
        cfg["scelte_della_fase_4"] = {"variante_ampiezze": {"regola": "rapporto corpo previsto / reale più vicino a 1 a h = 1, purché l'errore quadratico sulle ampiezze non sia significativamente peggiore di A0 (DM, p < 0,05); a parità vince la prima dell'elenco A0, A1, A2",
                                                            "scelta": variante, "rapporto_corpo_h1": a["scelta"]["rapporto_corpo_h1"]},
                                      "modelli_della_parte_b_che_passano": b["passa"]}
        cfg["piano_d_esame"] = "reports/piano_esame_holdout.md"
        cfg["sha256_sorgenti"] = impronte()
        out[v] = cfg
    return out


def main() -> None:
    git = lambda *args: subprocess.run(["git", *args], cwd=C.ROOT, capture_output=True, text=True, check=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    puliti = git("status", "--porcelain", "--", *VS.SORGENTI_CONGELATI) == ""
    VS.CARTELLA.mkdir(exist_ok=True)
    for vecchio in VS.CARTELLA.glob("f*.json"):
        vecchio.unlink()
    for v, cfg in costruisci().items():
        testa = {"versione": cfg.pop("versione"), "nome": cfg.pop("nome"), "congelata_il": date.today().isoformat(), "commit": commit, "sorgenti_identici_al_commit": puliti}
        (VS.CARTELLA / f"{v}.json").write_text(json.dumps(testa | cfg, indent=2, ensure_ascii=False) + "\n")
        print(f"versioni/{v}.json scritto (commit {commit[:10]}, sorgenti identici al commit: {'sì' if puliti else 'NO'})")
    if not puliti:
        print("ATTENZIONE: alcuni sorgenti congelati hanno modifiche non registrate: registrare il codice e rieseguire.")


if __name__ == "__main__":
    main()
