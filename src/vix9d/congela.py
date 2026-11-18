"""VIX9D: congela le versioni G1 e GS in versioni/vix9d/*.json:  python -m src.vix9d.congela

Per ogni versione: parametri, finestre e regole (da `versioni.configurazione`), le scelte fatte dalle regole della Fase
9D-1, l'impronta SHA-256 dei file sorgente che definiscono i modelli e l'hash del commit su cui il congelamento e' fatto.
Va eseguito DOPO `python -m src.vix9d.run_fase1` e dopo aver registrato il codice nel repository.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date

from .. import config as C
from . import config as G
from . import versioni as VS

VARIABILI_ESECUZIONE = ("congelata_il", "commit", "sorgenti_identici_al_commit")


def impronte() -> dict[str, str]:
    return {f: hashlib.sha256((C.ROOT / f).read_bytes()).hexdigest() for f in VS.SORGENTI_CONGELATI}


def costruisci() -> dict[str, dict]:
    info = json.loads((G.OUT / "info.json").read_text())
    out = {}
    for v in VS.VERSIONI:
        cfg = VS.configurazione(v, info["scelte"])
        cfg |= {"scelte": info["scelte"], "regola": "un candidato entra se batte il riferimento: errore più basso con p < 0,05 dopo Holm (Clark-West se annidato, Diebold-Mariano altrimenti), "
                                                    "per componente e per orizzonte, sulle origini comuni a tutti i candidati",
                "piano_d_esame": "reports/vix9d/piano_esame_holdout.md", "sha256_sorgenti": impronte()}
        out[v] = cfg
    return out


def main() -> None:
    git = lambda *args: subprocess.run(["git", *args], cwd=C.ROOT, capture_output=True, text=True, check=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    puliti = git("status", "--porcelain", "--", *VS.SORGENTI_CONGELATI) == ""
    G.VERSIONI.mkdir(parents=True, exist_ok=True)
    for v, cfg in costruisci().items():
        testa = {"versione": cfg.pop("versione"), "nome": cfg.pop("nome"), "congelata_il": date.today().isoformat(), "commit": commit, "sorgenti_identici_al_commit": puliti}
        (G.VERSIONI / f"{v}.json").write_text(json.dumps(testa | cfg, indent=2, ensure_ascii=False) + "\n")
        print(f"versioni/vix9d/{v}.json scritto (commit {commit[:10]}, sorgenti identici al commit: {'sì' if puliti else 'NO'})")
    if not puliti:
        print("ATTENZIONE: alcuni sorgenti congelati hanno modifiche non registrate: registrare il codice e rieseguire.")


if __name__ == "__main__":
    main()
