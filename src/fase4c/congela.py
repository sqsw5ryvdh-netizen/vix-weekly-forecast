"""Fase 4c: ricongela F1 e FS con la prima candela comandata da massimo e minimo:  python -m src.fase4c.congela

Scrive versioni/f1.json e versioni/fs.json: parametri, finestre e regole (da `versioni.configurazione`), le scelte
fatte dalle regole delle Fasi 4, 4b e 4c, l'impronta SHA-256 dei file sorgente che definiscono i modelli e l'hash del
commit su cui il congelamento e' stato fatto. Va eseguito DOPO `python -m src.run_phase4c` e dopo aver registrato il
codice nel repository.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date

from .. import config as C
from . import versioni as VS

R = C.REPORTS_DIR / "fase4c"
VARIABILI_ESECUZIONE = ("congelata_il", "commit", "sorgenti_identici_al_commit")


def impronte() -> dict[str, str]:
    return {f: hashlib.sha256((C.ROOT / f).read_bytes()).hexdigest() for f in VS.SORGENTI_CONGELATI}


def costruisci() -> dict[str, dict]:
    """Configurazione di F1 e FS, senza i campi che dipendono dal momento del congelamento."""
    info = json.loads((R / "info.json").read_text())
    b = json.loads((C.REPORTS_DIR / "fase4b" / "info.json").read_text())
    scelte, c = info["scelte"], info["parte_c"]
    regole = {"variante_ampiezze (Fase 4, Parte A)": scelte["variante"],
              "candidato della sfidante (Fase 4b, Parte B)": {"regola": "Brier più basso a h = 1 tra P2, P3, P4, P5; a parità il primo dell'elenco", "scelta": b["parte_b"]["candidato"]},
              "massimo e minimo della prima candela (Fase 4c, Parte A)": {
                  "regola": "per ciascun estremo, tra gli insiemi significativamente migliori di S0 (DM sull'errore assoluto in log, p < 0,05 dopo Holm) e con quota entro il 5% non più bassa di S0, "
                            "quello con l'errore più basso; se nessuno, S0; a parità meno variabili. F1: S1 contro S0 su tutte le origini; FS: S1-S5 e B3 contro S0 sulle origini comuni",
                  "scelte": info["insiemi"]},
              "corpo della prima candela (Fase 4c, Parte C)": {"regola": "K5 se l'errore sulla chiusura della candela è significativamente più basso (DM, p < 0,05) e la sovrapposizione non è "
                                                                         "significativamente più bassa; altrimenti K2. Applicata a F1 su tutte le origini", "scelta": c["scelta"], "mae_chiusura": c["mae_chiusura"]}}
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
