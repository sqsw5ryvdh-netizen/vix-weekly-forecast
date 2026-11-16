"""Fase 2c: riconoscimento della settimana incompleta in coda al file, anche senza la colonna `note`.

Regola: l'ultima riga e' incompleta se ha la nota, oppure se ha meno sedute di quelle attese E l'ultima seduta
non e' l'ultima seduta attesa della settimana. Le sedute attese sono i giorni da lunedi' a venerdi' meno le
festivita' del mercato USA (calendario a regole, senza chiusure straordinarie).

Nota sui dati: nel file, da maggio 2022 il VIX ha un valore anche in molte festivita' (Martin Luther King, Presidents'
Day, Memorial Day, Juneteenth, 4 luglio, Labor Day, Thanksgiving); restano senza valore Venerdi' Santo, Natale e
Capodanno. Per questo da quella data si tolgono dalle sedute attese solo queste tre festivita'.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def _pasqua(y: int) -> date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l) // 433
    month = (h + l - 7 * m + 90) // 25
    return date(y, month, (h + l - 7 * m + 33 * month + 19) % 32)


def _n_esimo(y: int, m: int, weekday: int, n: int) -> date:
    d = date(y, m, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _osservata(d: date) -> date | None:
    """Festivita' a data fissa: di sabato si osserva il venerdi' (tranne Capodanno), di domenica il lunedi'."""
    if d.weekday() == 5:
        return None if (d.month, d.day) == (1, 1) else d - timedelta(days=1)
    return d + timedelta(days=1) if d.weekday() == 6 else d


VIX_ANCHE_NEI_FESTIVI_DA = date(2022, 5, 1)


def festivo_senza_vix(d) -> bool:
    """Festivita' in cui il VIX non ha un valore nemmeno dopo il 2022: Venerdi' Santo, Natale, Capodanno."""
    d = pd.Timestamp(d).date()
    y = d.year
    return d in {_osservata(date(y, 1, 1)), _osservata(date(y, 12, 25)), _osservata(date(y + 1, 1, 1)), _pasqua(y) - timedelta(days=2)}


def festivo(d) -> bool:
    """Festivita' del mercato azionario USA (regole ordinarie)."""
    d = pd.Timestamp(d).date()
    y = d.year
    fisse = [date(y, 1, 1), date(y, 7, 4), date(y, 12, 25), date(y + 1, 1, 1)] + ([date(y, 6, 19)] if y >= 2022 else [])
    giorni = {_osservata(f) for f in fisse}
    ultimo_lun_maggio = _n_esimo(y, 6, 0, 1) - timedelta(weeks=1)
    giorni |= {_n_esimo(y, 1, 0, 3), _n_esimo(y, 2, 0, 3), _pasqua(y) - timedelta(days=2), ultimo_lun_maggio,
               _n_esimo(y, 9, 0, 1), _n_esimo(y, 11, 3, 4)}
    return d in giorni


def sedute_attese(week_start) -> list[date]:
    lun = pd.Timestamp(week_start).date()
    chiuso = festivo_senza_vix if lun >= VIX_ANCHE_NEI_FESTIVI_DA else festivo
    return [lun + timedelta(days=k) for k in range(5) if not chiuso(lun + timedelta(days=k))]


def ultima_riga_incompleta(week_start, last_trading_day, trading_days: int, note) -> tuple[bool, str | None]:
    """(incompleta, messaggio da mostrare a schermo quando la decisione non viene dalla nota)."""
    if isinstance(note, str) and note.strip():
        return True, None
    attese = sedute_attese(week_start)
    ultima = pd.Timestamp(last_trading_day).date()
    if not attese:
        return False, None
    if int(trading_days) < len(attese) and ultima < attese[-1]:
        return True, (f"ATTENZIONE: l'ultima riga (settimana del {pd.Timestamp(week_start).date()}) ha {int(trading_days)} sedute su {len(attese)} attese e l'ultima seduta "
                      f"è il {ultima}, prima dell'ultima seduta attesa ({attese[-1]}): la tratto come settimana INCOMPLETA e non la uso. "
                      "Se invece è completa (chiusura straordinaria del mercato), aggiungi la riga successiva o segnalalo.")
    if int(trading_days) < len(attese) or ultima < attese[-1]:
        return False, (f"Nota: l'ultima riga (settimana del {pd.Timestamp(week_start).date()}) ha {int(trading_days)} sedute su {len(attese)} attese, ultima seduta {ultima}: "
                       "la tratto come completa, ma il caso è dubbio.")
    return False, None
