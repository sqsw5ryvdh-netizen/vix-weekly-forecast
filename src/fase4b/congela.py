"""Fase 4b: ricongela le due versioni che andranno all'esame, F1 e FS:  python -m src.fase4b.congela

Scrive versioni/f1.json e versioni/fs.json (e toglie i file delle versioni che non vanno piu' all'esame): parametri,
finestre e regole (da `versioni.configurazione`), le scelte fatte dalle regole delle Fasi 4 e 4b, l'impronta SHA-256
dei file sorgente che definiscono i modelli e l'hash del commit su cui il congelamento e' stato fatto.
Va eseguito DOPO `python -m src.run_phase4b` e dopo aver registrato il codice nel repository.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date

from .. import config as C
from . import versioni as VS

R = C.REPORTS_DIR / "fase4b"
VARIABILI_ESECUZIONE = ("congelata_il", "commit", "sorgenti_identici_al_commit")


def impronte() -> dict[str, str]:
    return {f: hashlib.sha256((C.ROOT / f).read_bytes()).hexdigest() for f in VS.SORGENTI_CONGELATI}


def costruisci() -> dict[str, dict]:
    """Configurazione di F1 e FS, senza i campi che dipendono dal momento del congelamento."""
    info = json.loads((R / "info.json").read_text())
    scelte, b, c, d = info["scelte"], info["parte_b"], info["parte_c"], info["parte_d"]
    regole = {"variante_ampiezze (Fase 4, Parte A)": scelte["variante"],
              "candidato della sfidante (Parte B)": {"regola": "Brier più basso a h = 1 tra P2, P3, P4, P5; a parità il primo dell'elenco", "scelta": b["candidato"], "brier": b["brier"]},
              "prima candela (Parte C)": {"regola": "K4 se la sovrapposizione media non è più bassa di quella di K2 e l'errore sulla chiusura della candela non è significativamente peggiore (DM, p < 0,05); "
                                                    "altrimenti K2 con il colore legato all'etichetta. Applicata a F1 su tutte le sue origini", "scelta": c["scelta"], "sovrapposizione": c["sovrapposizione"]},
              "minimo della prima candela (Parte D)": {"regola": "M1q se l'errore assoluto medio è più basso con DM p < 0,05; altrimenti M0. Applicata a F1 su tutte le sue origini", "scelta": d["scelta"], "mae": d["mae"]}}
    out = {}
    for v in VS.VERSIONI:
        cfg = VS.configurazione(v, scelte)
        cfg |= {"scelte": scelte, "scelte_delle_regole": regole, "piano_d_esame": "reports/piano_esame_holdout.md", "sha256_sorgenti": impronte()}
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
