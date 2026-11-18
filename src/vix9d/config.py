"""VIX9D: percorsi e costanti (fissati prima di guardare i risultati)."""
from pathlib import Path

from .. import config as C

DATI = C.ROOT / "data" / "vix9d"
OUT = C.REPORTS_DIR / "vix9d"
FIG = OUT / "figures"
VERSIONI = C.ROOT / "versioni" / "vix9d"

FILE_COMPLETO = DATI / "VIX9D_settimanale.csv"          # tutta la storia, formato del file grezzo del VIX: letto SOLO dal programma di previsione
FILE_VIX_COMPLETO = DATI / "VIX_settimanale_cboe.csv"    # VIX settimanale dagli stessi file Cboe: letto SOLO dal programma di previsione
FILE_DEV = DATI / "vix9d_weekly_dev.csv"                 # unico file del VIX9D letto dalle misure
FILE_HOLDOUT = DATI / "vix9d_weekly_holdout.csv"         # BLOCCATO
ORIGINALI = {"vix9d": C.ROOT / "data" / "external" / "raw" / "cboe" / "VIX9D_History.csv", "vix": C.ROOT / "data" / "external" / "raw" / "cboe" / "VIX_History.csv"}
INDIRIZZI = {"vix9d": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv", "vix": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"}

CONFINE = "2024-11-08"            # stesso confine del VIX: sviluppo fino a questa data, il resto e' holdout
SETTIMANE_HOLDOUT = C.HOLDOUT_WEEKS   # stesso periodo d'esame del VIX: le prime 104 settimane complete dopo il confine
MIN_SETTIMANE = 156               # prima stima su almeno 156 settimane
MIN_COPPIE = MIN_SETTIMANE - 12   # coppie (origine, esito) valide necessarie: 156 settimane meno le 11 iniziali delle medie e 1 di esito
ORIZZONTI = (1, 2)
SOGLIE = (20.0, 25.0, 30.0)
TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
FASCE = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95)}
MIN_ERRORI = 52                   # errori passati (con esito noto) necessari per le probabilita' delle soglie
FINESTRA_CORPO = 260              # ampiezza media del corpo: ultime 260 settimane con candela vera (o quelle disponibili)
NEUTRA = (0.40, 0.60)
STRAPPO = 0.20                    # settimana di strappo: chiusura reale oltre +20% sulla chiusura all'origine (etichetta a posteriori)
SOTTOPERIODI = ((2011, 2014), (2015, 2019), (2020, 2024))
SEED = C.SEED
