"""Costanti e percorsi condivisi della Fase 0."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "VIX_settimanale_2000-2026.csv"
CLEAN_FILE = PROCESSED_DIR / "vix_weekly_clean.csv"
DEV_FILE = PROCESSED_DIR / "vix_weekly_dev.csv"          # unico file usato da EDA e baseline
HOLDOUT_FILE = PROCESSED_DIR / "vix_weekly_holdout.csv"  # BLOCCATO: non usare in Fase 0

SEED = 12345
HOLDOUT_WEEKS = 104
MIN_TRAIN_WEEKS = 260            # ~5 anni prima della prima previsione
HORIZONS = (1, 4)
LEVEL_THRESHOLDS = (15.0, 20.0, 25.0, 30.0)
BAND_LABELS = ("<15", "15-20", "20-25", "25-30", ">=30")
EVENT_THRESHOLD = 25.0           # evento della baseline probabilistica: VIX > 25
STRESS_THRESHOLD = 30.0          # soglia per individuare gli episodi di stress
# Sottoperiodi (anno iniziale, anno finale inclusi) per stabilita' e valutazione
SUBPERIODS = ((2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2024))
