"""Genera reports/fase0_report.md. Tutti i numeri sono letti dai file reports/*.csv e *.json."""
from __future__ import annotations

import json
import re

import pandas as pd

from . import config as C

R = C.REPORTS_DIR
MODEL_NAMES = {"random_walk": "Random walk", "media_storica": "Media storica", "ar1": "AR(1)", "har": "Tipo HAR",
               "markov_fasce": "Catena di Markov sulle fasce", "climatologia": "Climatologia (frequenza storica)",
               "persistenza": "Persistenza (0/1)"}


def J(name: str) -> dict:
    return json.loads((R / f"{name}.json").read_text())


def T(name: str) -> pd.DataFrame:
    return pd.read_csv(R / f"{name}.csv")


def md(df: pd.DataFrame, fmt=".3f") -> str:
    return re.sub(r"\bnan\b", "  –", df.to_markdown(index=False, floatfmt=fmt, missingval="–"))


def pct(v: float, d: int = 1) -> str:
    return f"{100 * v:.{d}f}%".replace(".", ",")


def n(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}".replace(".", ",")


def pv(p: float) -> str:
    return "< 0,001" if p < 0.001 else n(p, 3)


def run() -> None:
    A = J("data_audit")
    dist = T("eda_distribuzioni").set_index("serie")
    stz = T("eda_stazionarieta").set_index("serie")
    acf = T("eda_acf").set_index(["serie", "lag"])
    lb = T("eda_ljung_box").set_index(["serie", "lag"])
    MR, AS = J("eda_mean_reversion"), J("eda_asimmetria")
    ar, asy = MR["ar1_log"], MR["asimmetrica"]
    ep = T("eda_episodi_stress")
    fq, tr = T("eda_regimi_soglie_frequenze"), T("eda_regimi_soglie_transizione")
    fq_q, tr_q = T("eda_regimi_quartili_frequenze"), T("eda_regimi_quartili_transizione")
    sea, sea_t = T("eda_stagionalita"), J("eda_stagionalita_test")
    sub = T("eda_sottoperiodi")
    bm, bm_s, bm_r = T("baseline_metriche"), T("baseline_metriche_sottoperiodi"), T("baseline_metriche_regime")
    pm, pm_s, pm_r, cal = T("prob_metriche"), T("prob_metriche_sottoperiodi"), T("prob_metriche_regime"), T("prob_calibrazione")
    BI = J("baseline_info")

    ds = A["dataset"]
    bad = A["righe_ohlc_incoerenti"]
    part = A["righe_settimane_parziali"]
    short = A["righe_settimane_corte"]
    gap = A["gap_apertura"]
    subp = sub[sub["periodo"] != "intero campione"]
    B = bm.set_index(["h", "modello"])
    P = pm.set_index(["h", "modello"])
    diag = {r["da \\ a"]: r[r["da \\ a"]] for _, r in tr.iterrows()}
    phi_implied_acf1 = -(1 - ar["phi"]) / 2          # autocorrelazione lag 1 di dlog implicata da un AR(1) puro
    best = {h: bm[(bm["h"] == h)].sort_values("rmse_log").iloc[0]["modello"] for h in C.HORIZONS}
    ep_top = ep.sort_values("picco_close", ascending=False).head(10)
    MIN_N = 30

    def point_table(df, extra_cols):
        t = df[df["n"] >= MIN_N].copy()
        t["modello"] = t["modello"].map(MODEL_NAMES)
        cols = extra_cols + ["modello", "n", "rmse_log", "mae_log", "rmse_vix", "mae_vix", "acc_direzionale",
                             "r2_oos_vs_rw", "dm_stat", "dm_p"]
        return md(t[cols].rename(columns={"rmse_log": "RMSE log", "mae_log": "MAE log", "rmse_vix": "RMSE VIX",
                                          "mae_vix": "MAE VIX", "acc_direzionale": "acc. dir.",
                                          "r2_oos_vs_rw": "R² OOS vs RW", "dm_stat": "DM", "dm_p": "p DM"}))

    def prob_table(df, extra_cols):
        t = df[df["n"] >= MIN_N].copy()
        t["modello"] = t["modello"].map(MODEL_NAMES)
        cols = extra_cols + ["modello", "n", "frequenza_evento", "prob_media", "brier", "brier_skill_vs_climatologia"]
        return md(t[cols].rename(columns={"frequenza_evento": "freq. evento", "prob_media": "prob. media",
                                          "brier": "Brier", "brier_skill_vs_climatologia": "Brier skill vs clim."}))

    ext = pd.DataFrame(A["righe_variazioni_estreme"])[["week_start", "last_trading_day", "close", "dlog_close", "z_robusto"]]
    wd, tdays = A["giorno_last_trading_day"], A["trading_days"]
    sub_hl = subp.set_index("periodo")["half_life_sett"]
    r_h1 = bm_r[(bm_r["h"] == 1) & (bm_r["modello"] == "har")].set_index("fascia_origine")
    r_h4 = bm_r[(bm_r["h"] == 4) & (bm_r["modello"] == "ar1")].set_index("fascia_origine")
    s_har = bm_s[(bm_s["modello"] == "har") & (bm_s["n"] >= MIN_N)].set_index(["h", "sottoperiodo"])["r2_oos_vs_rw"]
    pr = pm_r[pm_r["modello"] == "markov_fasce"].set_index(["h", "fascia_origine"])
    mese_max, mese_min = sea.loc[sea["dlog_t"].idxmax()], sea.loc[sea["dlog_t"].idxmin()]

    out = f"""# Fase 0 — Fondamenta, audit dei dati, analisi esplorativa, baseline

*Report generato da `python -m src.run_phase0`. Ogni numero è letto dai file `reports/*.csv` e `reports/*.json`
prodotti dal codice; nessun valore è scritto a mano. Tutte le analisi usano solo il campione di sviluppo
({ds['dev']['da']} → {ds['dev']['a']}, {ds['dev']['n']} settimane): le ultime {ds['holdout']['n']} settimane sono bloccate.*

## 1. Sintesi

1. **Dati.** Il file contiene barre settimanali OHLC del VIX: {A['n_righe']} righe dal {A['prima_week_start']} al {A['ultimo_last_trading_day']}, senza settimane mancanti ({A['settimane_mancanti']}) né duplicati ({A['duplicati_week_start']}). Il target è la **chiusura** dell'ultima seduta della settimana.
2. **Anomalie segnalate, non corrette.** {A['n_settimane_parziali']} settimana incompleta in coda (esclusa), {A['n_ohlc_incoerenti']} valore di `open` incoerente ({n(bad[0]['open'])} il {bad[0]['week_start']}), un salto sistematico tra chiusura e apertura successiva (mediana {pct(gap['mediana'])}).
3. **Campioni.** Sviluppo: {ds['dev']['n']} settimane; holdout bloccato: {ds['holdout']['n']} settimane ({ds['holdout']['da']} → {ds['holdout']['a']}), salvato a parte e mai letto dal codice di analisi.
4. **Distribuzione.** Il livello è molto asimmetrico (asimmetria {n(dist.loc['livello', 'asimmetria'])}, curtosi in eccesso {n(dist.loc['livello', 'curtosi_eccesso'])}); il logaritmo lo è molto meno ({n(dist.loc['log', 'asimmetria'])}; {n(dist.loc['log', 'curtosi_eccesso'])}). Le variazioni log settimanali hanno code pesanti (curtosi in eccesso {n(dist.loc['dlog', 'curtosi_eccesso'])}) e asimmetria positiva ({n(dist.loc['dlog', 'asimmetria'])}).
5. **Stazionarietà e persistenza.** ADF rifiuta la radice unitaria su livello, log e variazioni; KPSS non rifiuta la stazionarietà al 5% (per il log è al limite: p = {n(stz.loc['log', 'kpss_p_troncato_0.01_0.10'], 3)}). Il log-VIX è stazionario ma molto persistente: φ = {n(ar['phi'], 3)}, half-life ≈ {n(ar['half_life_sett'], 1)} settimane, media di lungo periodo ≈ {n(ar['mu_vix'], 1)} punti.
6. **La persistenza non è stabile.** La half-life stimata per quinquenni va da {n(sub_hl.min(), 1)} a {n(sub_hl.max(), 1)} settimane.
7. **Struttura oltre l'AR(1).** Le variazioni log hanno autocorrelazione a lag 1 pari a {n(acf.loc[('dlog', 1), 'acf'], 3)}, molto più negativa di quella implicata da un AR(1) puro ({n(phi_implied_acf1, 3)}): c'è una componente transitoria che rientra in fretta. Il clustering della volatilità delle variazioni è presente ma moderato (ACF lag 1 dei quadrati {n(acf.loc[('dlog_quadrato', 1), 'acf'], 3)}).
8. **Asimmetria della dinamica.** I rialzi sono meno frequenti ({pct(AS['quota_settimane_rialzo'])} delle settimane) ma più ampi: {AS['n_rialzi_oltre_0.30']} rialzi oltre +0,30 log contro {AS['n_ribassi_oltre_0.30']} ribassi oltre −0,30. Nessuna stagionalità mensile rilevabile (Kruskal-Wallis p = {n(sea_t['kruskal_wallis_p'], 2)}).
9. **Baseline puntuali.** Il random walk è difficile da battere a 1 settimana: il modello tipo HAR ottiene R² fuori campione {pct(B.loc[(1, 'har'), 'r2_oos_vs_rw'])} (AR(1): {pct(B.loc[(1, 'ar1'), 'r2_oos_vs_rw'])}); a 4 settimane {pct(B.loc[(4, 'har'), 'r2_oos_vs_rw'])} e {pct(B.loc[(4, 'ar1'), 'r2_oos_vs_rw'])}. La media storica è nettamente peggiore del random walk.
10. **Baseline probabilistica.** Per P(VIX > {C.EVENT_THRESHOLD:.0f}) la catena di Markov ha Brier {n(P.loc[(1, 'markov_fasce'), 'brier'], 3)} a h = 1 e {n(P.loc[(4, 'markov_fasce'), 'brier'], 3)} a h = 4 (climatologia: {n(P.loc[(1, 'climatologia'), 'brier'], 3)} e {n(P.loc[(4, 'climatologia'), 'brier'], 3)}), ma è poco informativa quando si parte dalle fasce intermedie 20–30.

## 2. Audit dei dati

### 2.1 Il file

| voce | valore |
|---|---|
| file | `data/raw/{A['file']}` |
| sha256 | `{A['sha256']}` |
| formato | {A['formato']} |
| colonne | {', '.join(f'`{k}` ({v})' for k, v in A['colonne'].items())} |
| tipo di valore | {A['tipo_valore']} |
| righe | {A['n_righe']} |
| intervallo `week_start` | {A['prima_week_start']} → {A['ultima_week_start']} |
| intervallo `last_trading_day` | {A['primo_last_trading_day']} → {A['ultimo_last_trading_day']} |
| giorno di `week_start` | {', '.join(f'{k}: {v}' for k, v in A['giorno_week_start'].items())} |
| giorno di `last_trading_day` | {', '.join(f'{k}: {v}' for k, v in wd.items())} |
| sedute per settimana (`trading_days`) | {', '.join(f'{k} sedute: {v}' for k, v in tdays.items())} |
| ordinato per data | {'sì' if A['ordinato_per_data'] else 'no'} |
| settimane mancanti | {A['settimane_mancanti']} |
| duplicati (`week_start` / righe intere) | {A['duplicati_week_start']} / {A['duplicati_righe']} |
| valori mancanti nelle colonne numeriche | {sum(v for k, v in A['valori_mancanti'].items() if k != 'note')} |
| valori non positivi | {A['valori_non_positivi']} |
| decimali (massimo) | {A['decimali_max']} |

Ogni riga è una settimana di calendario che inizia di lunedì (`week_start`). `last_trading_day` è l'ultima seduta
della settimana: quasi sempre venerdì, giovedì quando il venerdì è festivo. **Giorno di riferimento adottato:**
la chiusura settimanale è datata a `last_trading_day` (colonna `date` dei file puliti).

### 2.2 Anomalie e casi particolari

- **Settimana incompleta.** {A['n_settimane_parziali']} riga ha la colonna `note` valorizzata: settimana del {part[0]['week_start']}, nota «{part[0]['note']}», {part[0]['trading_days']} sedute. Non è una chiusura settimanale definitiva: è **esclusa** dai file puliti (trasformazione T2).
- **OHLC incoerente.** {A['n_ohlc_incoerenti']} riga viola high ≥ max(open, close) e low ≤ min(open, close): settimana del {bad[0]['week_start']}, open = {n(bad[0]['open'])} con low = {n(bad[0]['low'])}, high = {n(bad[0]['high'])}, close = {n(bad[0]['close'])}. Ha l'aspetto di un errore di battitura nell'`open`. **Non è stato corretto**: la riga è marcata con `flag_ohlc_incoerente = 1`. La chiusura, unica colonna usata in questa fase, è coerente con le settimane vicine.
- **Salto chiusura → apertura.** ln(open_t / close_t−1) ha mediana {pct(gap['mediana'])} ed è positivo nel {pct(gap['quota_positivi'])} delle settimane (escluso holdout). Un salto così sistematico suggerisce che `open` non sia confrontabile con la chiusura precedente (o che sia calcolato in modo diverso). In questa fase `open`, `high` e `low` non sono usati; prima di usarli serve un chiarimento sulla fonte.
- **Settimane corte.** {A['n_settimane_corte']} settimane hanno 3 sedute o meno: {'; '.join(f"{r['week_start']} ({r['trading_days']} sedute, ultima il {r['last_trading_day']})" for r in short)}. Sono conservate e marcate con `flag_settimana_corta = 1`. La settimana del 2001-09-10 ha una sola seduta (lunedì) per la chiusura dei mercati dopo l'11 settembre.
- **Variazioni estreme.** {A['n_variazioni_estreme']} variazioni log settimanali della chiusura superano la soglia robusta ({A['variazioni_estreme_regola']}). Coincidono tutte con episodi di mercato noti e sono **conservate**:

{md(ext, ".3f")}

- **Intervalli dei valori (escluso holdout).** {'; '.join(f"`{k}`: {n(v[0])}–{n(v[1])}" for k, v in A['min_max_escluso_holdout'].items())}.

### 2.3 Dataset puliti e trasformazioni

{chr(10).join(f'- {t}' for t in A['trasformazioni'])}

| file | righe | da | a | uso |
|---|---|---|---|---|
| `data/processed/{ds['clean']['file']}` | {ds['clean']['n']} | {ds['clean']['da']} | {ds['clean']['a']} | unione dei due seguenti |
| `data/processed/{ds['dev']['file']}` | {ds['dev']['n']} | {ds['dev']['da']} | {ds['dev']['a']} | unico file letto da analisi e baseline |
| `data/processed/{ds['holdout']['file']}` | {ds['holdout']['n']} | {ds['holdout']['da']} | {ds['holdout']['a']} | **bloccato** |

## 3. Analisi esplorativa (campione di sviluppo)

### 3.1 Serie storica ed episodi di stress

![Serie storica](figures/01_serie_storica.png)

Un episodio di stress è definito come un periodo con chiusura ≥ {C.STRESS_THRESHOLD:.0f} (periodi separati da al più 4 settimane sono uniti).
Ne risultano {len(ep)}; i dieci con il picco più alto (le etichette sono descrittive, assegnate a mano in base alla data del picco):

{md(ep_top[['inizio', 'fine', 'settimane_sopra_soglia', 'data_picco', 'picco_close', 'etichetta']].fillna(''), ".2f")}

### 3.2 Distribuzioni

![Distribuzioni](figures/02_distribuzioni.png)

{md(dist.reset_index()[['serie', 'n', 'media', 'dev_std', 'asimmetria', 'curtosi_eccesso', 'min', 'q01', 'q05', 'mediana', 'q95', 'q99', 'max']])}

- Il passaggio al logaritmo riduce l'asimmetria da {n(dist.loc['livello', 'asimmetria'])} a {n(dist.loc['log', 'asimmetria'])} e la curtosi in eccesso da {n(dist.loc['livello', 'curtosi_eccesso'])} a {n(dist.loc['log', 'curtosi_eccesso'])}: giustifica il log come scala di lavoro. La normalità resta comunque rifiutata (Jarque-Bera) per tutte e tre le serie.
- Code delle variazioni log: il {pct(dist.loc['dlog', 'quota_z_sopra_3'], 2)} delle osservazioni è oltre +3 deviazioni standard e il {pct(dist.loc['dlog', 'quota_z_sotto_m3'], 2)} oltre −3, contro lo {pct(dist.loc['dlog', 'quota_attesa_normale_per_coda'], 2)} atteso per coda sotto normalità. La coda destra (rialzi) è più pesante della sinistra.
- Una variazione settimanale tipica è ampia: deviazione standard {n(dist.loc['dlog', 'dev_std'], 3)} in log, cioè circa il {pct(dist.loc['dlog', 'dev_std'], 0)} del livello.

### 3.3 Stazionarietà

{md(stz.reset_index()[['serie', 'adf_stat', 'adf_p', 'adf_ritardi', 'kpss_stat', 'kpss_p_troncato_0.01_0.10', 'kpss_ritardi']].rename(columns={'kpss_p_troncato_0.01_0.10': 'kpss_p'}), ".4f")}

ADF (costante, ritardi scelti con AIC) rifiuta la radice unitaria per tutte le serie. KPSS (ipotesi nulla: stazionarietà attorno a una costante;
il p-value è tabulato solo tra 0,01 e 0,10) non rifiuta al 5%, ma per il log è al limite (p = {n(stz.loc['log', 'kpss_p_troncato_0.01_0.10'], 3)}).
Lettura: livello e log sono stazionari con ritorno alla media lento; non serve differenziare, ma la media di lungo periodo è stimata con molta incertezza.

### 3.4 Autocorrelazione e clustering

![ACF e PACF](figures/03_acf_pacf.png)

{md(acf.reset_index().pivot(index='lag', columns='serie', values='acf').reset_index()[['lag', 'log', 'dlog', 'dlog_quadrato', 'dlog_assoluto']])}

(Banda al 95% sotto l'ipotesi di rumore bianco: ±{n(acf.loc[('dlog', 1), 'banda_95'], 3)}.)

- **log-VIX**: l'ACF decade lentamente ({n(acf.loc[('log', 1), 'acf'], 2)} a lag 1, {n(acf.loc[('log', 12), 'acf'], 2)} a lag 12, {n(acf.loc[('log', 52), 'acf'], 2)} a un anno). Il decadimento è più lento di quello geometrico di un AR(1) con φ = {n(ar['phi'], 3)} (che a lag 12 darebbe {n(ar['phi'] ** 12, 2)}, a lag 52 {n(ar['phi'] ** 52, 2)}): la memoria è più lunga di quella di un AR(1).
- **Variazioni**: autocorrelazione a lag 1 negativa e significativa ({n(acf.loc[('dlog', 1), 'acf'], 3)}); Ljung-Box a 4 ritardi Q = {n(lb.loc[('dlog', 4), 'ljung_box_q'], 1)} (p {pv(lb.loc[('dlog', 4), 'p_value'])}). Un AR(1) puro sul log implicherebbe solo {n(phi_implied_acf1, 3)}. Insieme al punto precedente è la firma di due componenti: una persistente e una transitoria che rientra in 1–2 settimane.
- **Clustering**: i quadrati delle variazioni sono autocorrelati (lag 1: {n(acf.loc[('dlog_quadrato', 1), 'acf'], 3)}; Ljung-Box a 12 ritardi p = {pv(lb.loc[('dlog_quadrato', 12), 'p_value'])}) e i valori assoluti di più (lag 1: {n(acf.loc[('dlog_assoluto', 1), 'acf'], 3)}). L'effetto esiste ma è contenuto: la «volatilità della volatilità» varia nel tempo, senza la persistenza tipica dei rendimenti azionari.

### 3.5 Mean reversion

Regressione AR(1) su log-VIX, x_t = a + φ·x_t−1 + ε (OLS, {ar['n']} osservazioni):

| parametro | stima |
|---|---|
| φ | {n(ar['phi'], 4)} (e.s. {n(ar['se_phi'], 4)}) |
| velocità di ritorno κ = −ln φ | {n(ar['kappa_sett'], 4)} per settimana |
| half-life = ln 0,5 / ln φ | {n(ar['half_life_sett'], 1)} settimane (IC 95%: {n(ar['half_life_ic95'][0], 1)}–{n(ar['half_life_ic95'][1], 1)}) |
| media di lungo periodo | {n(ar['mu_log'], 3)} in log, cioè {n(ar['mu_vix'], 1)} punti VIX |
| dev. std. dei residui | {n(ar['sigma_res'], 4)} |

Ogni settimana si chiude circa il {pct(1 - ar['phi'])} della distanza dalla media. La stima OLS di φ è distorta verso il basso
in campioni finiti, quindi la half-life vera è, semmai, un po' più lunga.

**Ritorno diverso sopra e sotto la media?** Stimando Δx_t = a + b⁺·(x_t−1 − μ)⁺ + b⁻·(x_t−1 − μ)⁻ si ottiene
b⁺ = {n(asy['b_sopra_media'], 3)} (half-life {n(asy['half_life_sopra_sett'], 1)} settimane sopra la media) e
b⁻ = {n(asy['b_sotto_media'], 3)} (half-life {n(asy['half_life_sotto_sett'], 1)} settimane sotto la media).
La differenza **non è statisticamente significativa** (t = {n(asy['t_differenza'], 2)}, p = {n(asy['p_differenza'], 2)}): con questi dati non si può affermare che la velocità di rientro dipenda dal lato.

### 3.6 Asimmetria della dinamica: i rialzi sono più rapidi dei ribassi?

| misura | rialzi | ribassi |
|---|---|---|
| quota di settimane | {pct(AS['quota_settimane_rialzo'])} | {pct(AS['quota_settimane_ribasso'])} |
| variazione log media | {n(AS['rialzo_medio'], 3)} | {n(AS['ribasso_medio'], 3)} |
| variazione log mediana | {n(AS['rialzo_mediano'], 3)} | {n(AS['ribasso_mediano'], 3)} |
| quantile estremo (95° / 5°) | {n(AS['q95_dlog'], 3)} | {n(AS['q05_dlog'], 3)} |
| quantile estremo (99° / 1°) | {n(AS['q99_dlog'], 3)} | {n(AS['q01_dlog'], 3)} |
| variazione massima | {n(AS['max_rialzo'], 3)} | {n(AS['max_ribasso'], 3)} |
| settimane oltre ±0,30 log | {AS['n_rialzi_oltre_0.30']} | {AS['n_ribassi_oltre_0.30']} |
| durata media di una serie consecutiva (settimane) | {n(AS['durata_media_serie_rialzi'])} | {n(AS['durata_media_serie_ribassi'])} |
| ampiezza media di una serie consecutiva (log) | {n(AS['ampiezza_media_serie_rialzi'], 3)} | {n(AS['ampiezza_media_serie_ribassi'], 3)} |

Sì, con una precisazione. Le settimane di ribasso sono più numerose e le serie di ribassi durano di più; i rialzi sono più
rari, più concentrati e con una coda molto più pesante. Le serie consecutive di rialzi e di ribassi hanno la stessa ampiezza
media: il VIX sale in meno tempo di quanto impieghi a scendere della stessa quantità. La differenza sta soprattutto nelle code;
le variazioni mediane sono quasi uguali.

Sugli {AS['episodi_n']} episodi di stress, definendo la «metà» come punto medio (in log) tra il minimo delle 26 settimane
precedenti e il picco: dall'ultima chiusura sotto la metà al picco passano in mediana {n(AS['episodi_mediana_sett_salita'], 1)} settimane
(media {n(AS['episodi_media_sett_salita'], 1)}); dal picco al ritorno sotto la metà {n(AS['episodi_mediana_sett_discesa_meta'], 1)}
(media {n(AS['episodi_media_sett_discesa_meta'], 1)}). La salita è più rapida nel {pct(AS['episodi_quota_salita_piu_rapida'], 0)} degli episodi,
la discesa nel {pct(AS['episodi_quota_discesa_piu_rapida'], 0)}. Sono pochi episodi: è un'indicazione, non una stima precisa.

### 3.7 Regimi

Fasce a soglie fisse (15 / 20 / 25 / 30):

{md(fq, ".2f")}

Matrice di transizione settimanale (righe: fascia a t; colonne: fascia a t+1):

{md(tr)}

![Matrici di transizione](figures/04_matrici_transizione.png)

- Le fasce estreme sono le più persistenti: probabilità di restare {n(diag['<15'], 2)} per «<15» e {n(diag['>=30'], 2)} per «>=30». Le fasce intermedie sono di passaggio ({n(diag['20-25'], 2)} per «20-25», {n(diag['25-30'], 2)} per «25-30»).
- La matrice è quasi tridiagonale: i salti di più di una fascia in una settimana sono rari.
- Le durate sono molto asimmetriche: la mediana è di 1–3 settimane in ogni fascia, ma il periodo più lungo sopra 30 è durato {int(fq.set_index('fascia').loc['>=30', 'durata_massima'])} settimane e quello sotto 15 {int(fq.set_index('fascia').loc['<15', 'durata_massima'])}.

Fasce per quartili (quartili calcolati sull'intero campione di sviluppo: uso solo descrittivo, non previsivo):

{md(fq_q, ".2f")}

{md(tr_q)}

### 3.8 Stagionalità

![Stagionalità](figures/05_stagionalita.png)

{md(sea, ".4f")}

Nessuna evidenza di stagionalità mensile nelle variazioni: Kruskal-Wallis p = {n(sea_t['kruskal_wallis_p'], 2)}, ANOVA p = {n(sea_t['anova_p'], 2)};
nemmeno la varianza cambia tra mesi (Levene p = {n(sea_t['levene_varianza_p'], 2)}). I mesi con statistica t più estrema sono
il mese {int(mese_max['mese'])} (t = {n(mese_max['dlog_t'], 2)}) e il mese {int(mese_min['mese'])} (t = {n(mese_min['dlog_t'], 2)}):
{sea_t['n_mesi_con_t_oltre_1.96']} mesi su 12 superano |t| = 1,96. Le differenze di livello mediano tra mesi riflettono in buona parte
il mese in cui sono caduti i pochi grandi episodi di crisi, non un effetto stagionale.

### 3.9 Stabilità nel tempo

{md(subp[['periodo', 'n', 'vix_medio', 'vix_mediano', 'vix_min', 'vix_max', 'dev_std_log', 'dev_std_dlog']], ".2f")}

{md(subp[['periodo', 'asimmetria_dlog', 'curtosi_eccesso_dlog', 'phi_ar1', 'half_life_sett', 'mu_vix_ar1', 'quota_sopra_25', 'acf1_dlog', 'acf1_dlog_quadrato']], ".3f")}

![Stabilità](figures/06_stabilita.png)

- **Il livello medio cambia regime per anni**: VIX mediano da {n(subp['vix_mediano'].min(), 1)} a {n(subp['vix_mediano'].max(), 1)} tra quinquenni; la quota di settimane sopra 25 va dal {pct(subp['quota_sopra_25'].min())} al {pct(subp['quota_sopra_25'].max())}.
- **La persistenza non è costante**: φ tra {n(subp['phi_ar1'].min(), 3)} e {n(subp['phi_ar1'].max(), 3)}, half-life tra {n(sub_hl.min(), 1)} e {n(sub_hl.max(), 1)} settimane. Parte dell'elevata persistenza stimata sull'intero campione riflette spostamenti lenti del livello medio.
- **Le variazioni sono diventate più ampie e più asimmetriche**: la deviazione standard di Δlog passa da {n(subp['dev_std_dlog'].iloc[0], 3)} ({subp['periodo'].iloc[0]}) a {n(subp['dev_std_dlog'].iloc[-1], 3)} ({subp['periodo'].iloc[-1]}), l'asimmetria da {n(subp['asimmetria_dlog'].iloc[0], 2)} a {n(subp['asimmetria_dlog'].iloc[-1], 2)}.
- **L'autocorrelazione negativa delle variazioni è stabile nel segno** in tutti i sottoperiodi (tra {n(subp['acf1_dlog'].min(), 2)} e {n(subp['acf1_dlog'].max(), 2)}): è la regolarità più robusta emersa.

## 4. Baseline con valutazione walk-forward

**Protocollo.** Target: log(VIX) a h = 1 e h = 4 settimane. {BI['schema'].capitalize()}; prima stima su {BI['min_train_settimane']} settimane (5 anni).
La previsione fatta all'origine t usa solo le chiusure fino a t e riguarda t + h. Previsioni valutate:
{BI['per_orizzonte']['1']['n_previsioni']} per h = 1 (obiettivi dal {BI['per_orizzonte']['1']['prima_data_obiettivo']} al {BI['per_orizzonte']['1']['ultima_data_obiettivo']}) e
{BI['per_orizzonte']['4']['n_previsioni']} per h = 4 (dal {BI['per_orizzonte']['4']['prima_data_obiettivo']} al {BI['per_orizzonte']['4']['ultima_data_obiettivo']}).
L'holdout non è toccato.

**Modelli.** (a) random walk; (b) media storica espandibile; (c) AR(1) su log-VIX, previsione iterata; (d) tipo HAR, regressione
diretta di x_t+h su ultimo valore, media a 4 e media a 12 settimane.

**Metriche.** RMSE e MAE in log e in punti VIX (previsione in punti = exp della previsione in log); accuratezza direzionale
(segno della variazione prevista contro segno di quella realizzata, rispetto al valore all'origine: non definita per il random walk, che prevede
variazione nulla); R² fuori campione = 1 − SSE_modello / SSE_RW; Diebold-Mariano su perdita quadratica contro il random walk
(positivo = il modello è migliore; varianza HAC con h − 1 ritardi, correzione di Harvey-Leybourne-Newbold).

### 4.1 Tabella comparativa

{point_table(bm, ['h'])}

- **h = 1.** Il random walk ha RMSE {n(B.loc[(1, 'random_walk'), 'rmse_log'], 4)} in log ({n(B.loc[(1, 'random_walk'), 'rmse_vix'])} punti VIX). AR(1) e HAR lo migliorano di poco ma in modo statisticamente significativo: R² OOS {pct(B.loc[(1, 'ar1'), 'r2_oos_vs_rw'])} e {pct(B.loc[(1, 'har'), 'r2_oos_vs_rw'])}. In punti VIX il vantaggio dell'HAR quasi scompare (RMSE {n(B.loc[(1, 'har'), 'rmse_vix'])} contro {n(B.loc[(1, 'random_walk'), 'rmse_vix'])}): gli errori in punti sono dominati dalle poche settimane di crisi.
- **h = 4.** Il ritorno alla media diventa sfruttabile: R² OOS {pct(B.loc[(4, 'ar1'), 'r2_oos_vs_rw'])} per l'AR(1) e {pct(B.loc[(4, 'har'), 'r2_oos_vs_rw'])} per l'HAR, accuratezza direzionale {pct(B.loc[(4, 'ar1'), 'acc_direzionale'])} e {pct(B.loc[(4, 'har'), 'acc_direzionale'])}.
- **La media storica** è molto peggiore del random walk a entrambi gli orizzonti (R² OOS {pct(B.loc[(1, 'media_storica'), 'r2_oos_vs_rw'], 0)} a h = 1), ma ha un'accuratezza direzionale simile agli altri modelli: «il VIX tornerà verso la media» indovina il segno, non l'ampiezza.
- **Modello migliore per RMSE in log:** {MODEL_NAMES[best[1]]} a h = 1, {MODEL_NAMES[best[4]]} a h = 4. Il riferimento da battere in Fase 1 è questo, non il random walk.

![Errore cumulato rispetto al random walk](figures/08_errore_cumulato_vs_rw.png)

### 4.2 Per sottoperiodo (data dell'obiettivo)

{point_table(bm_s, ['h', 'sottoperiodo'])}

(Omessi i gruppi con meno di {MIN_N} previsioni; il file CSV li contiene tutti.)

Il vantaggio sul random walk non è uniforme. Per l'HAR l'R² OOS a h = 1 è {pct(s_har.loc[(1, '2005-2009')])} nel 2005-2009,
{pct(s_har.loc[(1, '2010-2014')])} nel 2010-2014, {pct(s_har.loc[(1, '2015-2019')])} nel 2015-2019 e {pct(s_har.loc[(1, '2020-2024')])} nel 2020-2024;
a h = 4 rispettivamente {pct(s_har.loc[(4, '2005-2009')])}, {pct(s_har.loc[(4, '2010-2014')])}, {pct(s_har.loc[(4, '2015-2019')])} e {pct(s_har.loc[(4, '2020-2024')])}.
Nel quinquennio della crisi finanziaria nessun modello batte il random walk in modo statisticamente significativo: un lungo spostamento di livello premia chi non scommette sul rientro.

### 4.3 Per regime di partenza (fascia del VIX all'origine)

{point_table(bm_r, ['h', 'fascia_origine'])}

- L'errore cresce con il livello di partenza: per il random walk a h = 1 l'RMSE in log va da {n(bm_r[(bm_r['h'] == 1) & (bm_r['modello'] == 'random_walk')]['rmse_log'].min(), 3)} a {n(bm_r[(bm_r['h'] == 1) & (bm_r['modello'] == 'random_walk')]['rmse_log'].max(), 3)} tra le fasce. Gli errori sono eteroschedastici rispetto al livello.
- Il guadagno dei modelli con ritorno alla media si concentra ai margini. A h = 4 l'AR(1) ha R² OOS {pct(r_h4.loc['>=30', 'r2_oos_vs_rw'])} partendo da «>=30» e {pct(r_h4.loc['<15', 'r2_oos_vs_rw'])} da «<15», ma {pct(r_h4.loc['15-20', 'r2_oos_vs_rw'])} da «15-20», dove il VIX è già vicino alla sua media.
- Il random walk ha una distorsione sistematica condizionata al regime: sottostima partendo da livelli bassi e sovrastima partendo da livelli alti (colonna `bias_log` in `reports/baseline_metriche_regime.csv`).
- I gruppi «25-30» e «>=30» hanno {int(r_h1.loc['>=30', 'n'])} previsioni ciascuno: i test lì hanno poca potenza.

### 4.4 Baseline probabilistica: P(VIX > {C.EVENT_THRESHOLD:.0f})

A ogni origine t si stima la matrice di transizione empirica a un passo tra le cinque fasce usando la storia fino a t; la probabilità a h
passi è la riga della fascia corrente della matrice elevata a h, sommata sulle fasce sopra {C.EVENT_THRESHOLD:.0f}. Riferimenti:
climatologia (frequenza storica espandibile dell'evento) e persistenza (1 se oggi VIX > {C.EVENT_THRESHOLD:.0f}, altrimenti 0).

{prob_table(pm, ['h'])}

![Calibrazione](figures/07_calibrazione.png)

{md(cal, ".3f")}

- La catena di Markov riduce il Brier score del {pct(P.loc[(1, 'markov_fasce'), 'brier_skill_vs_climatologia'], 0)} rispetto alla climatologia a h = 1 e del {pct(P.loc[(4, 'markov_fasce'), 'brier_skill_vs_climatologia'], 0)} a h = 4, e batte la persistenza a entrambi gli orizzonti.
- La climatologia è distorta: prevede in media {pct(P.loc[(1, 'climatologia'), 'prob_media'])} contro una frequenza realizzata del {pct(P.loc[(1, 'climatologia'), 'frequenza_evento'])}, perché i primi cinque anni del campione avevano un VIX più alto.
- **Calibrazione.** Buona agli estremi, dove sta la maggior parte delle previsioni. Nelle classi intermedie la catena tende a sovrastimare, soprattutto a h = 4 (per esempio nella classe 0,3–0,4 la frequenza osservata è {n(cal[(cal['h'] == 4) & (cal['classe'] == '0.3-0.4')]['freq_osservata'].iloc[0], 2)}). Il modello emette solo cinque valori distinti per data (uno per fascia): ignora dove si trova il VIX all'interno della fascia.

Per sottoperiodo:

{prob_table(pm_s, ['h', 'sottoperiodo'])}

Per fascia di partenza:

{prob_table(pm_r, ['h', 'fascia_origine'])}

Partendo da «20-25» la catena non fa meglio della climatologia (Brier skill {n(pr.loc[(1, '20-25'), 'brier_skill_vs_climatologia'], 3)} a h = 1 e {n(pr.loc[(4, '20-25'), 'brier_skill_vs_climatologia'], 3)} a h = 4),
e da «25-30» a h = 4 il guadagno è minimo ({n(pr.loc[(4, '25-30'), 'brier_skill_vs_climatologia'], 3)}). È proprio vicino alla soglia che una previsione probabilistica
serve di più: qui sta il margine di miglioramento più chiaro.

## 5. Problemi e ambiguità riscontrati

1. **Fonte dei dati non documentata.** Il file non indica la provenienza né il metodo di aggregazione settimanale. I valori non sono stati confrontati con una fonte esterna.
2. **Settimana incompleta in coda** ({part[0]['week_start']}): esclusa. Di conseguenza l'holdout è formato dalle ultime {ds['holdout']['n']} settimane *complete* ({ds['holdout']['da']} → {ds['holdout']['a']}). Se si preferisce contare la settimana parziale, il confine si sposta di una settimana.
3. **`open` anomalo** del {bad[0]['week_start']} ({n(bad[0]['open'])}): lasciato com'è e marcato. Da decidere come trattarlo se in Fase 1 si usano le colonne OHLC.
4. **Salto sistematico tra chiusura e apertura successiva** (mediana {pct(gap['mediana'])}, positivo nel {pct(gap['quota_positivi'])} dei casi): la colonna `open` va chiarita prima dell'uso.
5. **Giorno di riferimento non uniforme.** La chiusura è di venerdì in {wd.get('ven', 0)} casi, di giovedì in {wd.get('gio', 0)}, di lunedì in {wd.get('lun', 0)} (settimana dell'11 settembre 2001). Gli orizzonti sono espressi in settimane, non in un numero fisso di giorni di calendario.
6. **Settimane corte** ({A['n_settimane_corte']} con 3 sedute o meno): conservate, perché toglierle creerebbe buchi nella serie.
7. **Contatto con l'holdout durante l'ispezione iniziale.** Prima di separare i dati ho ispezionato l'intero file grezzo (prime e ultime righe, conteggi, statistiche descrittive e le maggiori variazioni sull'intero file): ho quindi visto alcune righe e alcuni valori estremi che ricadono nell'holdout. Nessuna scelta di modello, soglia o parametro è stata fatta su quei valori; da quel momento ogni controllo sui valori è limitato al campione di sviluppo e i file di analisi leggono solo `{ds['dev']['file']}`. Lo segnalo per trasparenza.
8. **Test di Diebold-Mariano con modelli annidati.** Il random walk è un caso particolare dell'AR(1) e dell'HAR; in questo caso la distribuzione del DM non è quella standard e il test tende a essere conservativo. I p-value sono indicativi. Inoltre i test sono molti (modelli × orizzonti × sottogruppi) e non sono corretti per confronti multipli.
9. **Metriche in punti VIX.** exp(previsione in log) è una previsione della mediana, non della media: le metriche in punti sono coerenti tra i modelli ma non ottimali per l'RMSE in punti.
10. **Accuratezza direzionale** non definita per il random walk. Il confronto naturale è il 50%, ma dato che i ribassi sono più frequenti ({pct(AS['quota_settimane_ribasso'])} delle settimane), anche «prevedi sempre ribasso» supera il 50%.
11. **Sottoperiodo 2000-2004.** Serve da finestra iniziale: contiene {int(bm_s[(bm_s['sottoperiodo'] == '2000-2004') & (bm_s['h'] == 1)]['n'].iloc[0])} sola previsione a h = 1 ed è omesso dalle tabelle del report.
12. **Catena di Markov.** Nessun lisciamento delle frequenze; l'orizzonte 4 usa la matrice a un passo elevata alla quarta (ipotesi markoviana), non una matrice a 4 passi stimata direttamente.
13. **KPSS.** p-value disponibile solo nell'intervallo 0,01–0,10; per il log-VIX il risultato è al limite e dipende dalla scelta dei ritardi.
14. **Etichette degli episodi di stress**: assegnate a mano in base alla data del picco, solo a scopo descrittivo.

## 6. Proposte per la Fase 1

Ordinate per rapporto atteso tra beneficio e rischio; ognuna nasce da un risultato di questa fase.

1. **Modelli a due componenti (ARMA(1,1) o forma a spazio degli stati: livello persistente + disturbo transitorio).** Motivo: l'autocorrelazione a lag 1 delle variazioni ({n(acf.loc[('dlog', 1), 'acf'], 3)}) è molto più negativa di quanto consenta un AR(1) ({n(phi_implied_acf1, 3)}), e l'ACF del log decade più lentamente di φ^k. L'HAR, che media su 4 e 12 settimane, già ne cattura una parte (R² OOS {pct(B.loc[(1, 'har'), 'r2_oos_vs_rw'])} contro {pct(B.loc[(1, 'ar1'), 'r2_oos_vs_rw'])} a h = 1). È l'estensione più economica e meglio motivata.
2. **Previsioni di densità con varianza dipendente dallo stato.** Motivo: gli errori crescono con il livello di partenza, le variazioni hanno code pesanti e asimmetria positiva, e c'è clustering (moderato). Proposta: modellare la varianza condizionata dei residui (in funzione del livello e/o di tipo GARCH) con una distribuzione asimmetrica, e valutare con CRPS e log score oltre a RMSE.
3. **P(VIX > 25) da un modello continuo anziché dalle fasce.** Motivo: la catena di Markov non aggiunge nulla proprio partendo da 20–30, perché ignora la posizione dentro la fascia. Proposta: regressione logistica su log-VIX e medie mobili, oppure probabilità ricavata dalla densità del punto 2; confronto con Brier, scomposizione affidabilità/risoluzione e curva di calibrazione, con la catena di Markov come riferimento.
4. **Gestione dell'instabilità dei parametri.** Motivo: half-life tra {n(sub_hl.min(), 1)} e {n(sub_hl.max(), 1)} settimane e livello medio che si sposta per anni; nel 2005-2009 nessun modello batte il random walk in modo significativo. Proposta: confrontare finestra espandibile, finestra mobile e pesi esponenziali; valutare la combinazione di previsioni (per esempio media tra random walk e HAR).
5. **Non linearità mirate.** Motivo: i guadagni si concentrano agli estremi (sotto 15 e sopra 30). Proposta: modelli a soglia o con ritorno alla media dipendente dal livello. Avvertenza: l'asimmetria del ritorno sopra/sotto la media non è risultata significativa (p = {n(asy['p_differenza'], 2)}), quindi il rischio di sovradattamento è concreto; da tenere solo se il guadagno regge in ogni sottoperiodo.
6. **Informazione infrasettimanale già presente nel file** (massimo, minimo, escursione della settimana). Motivo: è disponibile senza nuove fonti ed è plausibile che un'escursione ampia segnali una componente transitoria. Subordinata al chiarimento sulle colonne OHLC (punti 3 e 4 della sezione 5).
7. **Variabili esterne** (rendimenti dell'S&P 500, struttura a termine dei future sul VIX, volatilità realizzata). Potenzialmente la fonte di guadagno maggiore, ma richiede nuovi dati con le relative date di disponibilità: serve una decisione esplicita.
8. **Protocollo di valutazione.** Mantenere lo stesso walk-forward e le stesse tabelle; aggiungere il test di Clark-West per i modelli annidati e una correzione per confronti multipli; fissare in anticipo l'elenco dei modelli; aprire l'holdout una sola volta, alla fine.

**Decisioni richieste prima della Fase 1:** (i) qual è la fonte del file e come è definito `open`; (ii) conferma dell'esclusione della settimana parziale e della chiusura dell'ultima seduta come valore di riferimento; (iii) se sono ammessi dati esterni; (iv) se l'obiettivo principale è la previsione puntuale, quella probabilistica o entrambe.

## 7. Riproducibilità

```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m src.run_phase0     # rigenera dataset, tabelle, figure e questo report
python -m pytest             # test, incluso il controllo anti look-ahead
```

Versioni e seed sono registrati in `reports/run_manifest.json`. Le previsioni walk-forward complete sono in `reports/baseline_previsioni.csv`.
"""
    (R / "fase0_report.md").write_text(out)


if __name__ == "__main__":
    run()
