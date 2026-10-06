import itertools
from pathlib import Path

import pandas as pd

from src.data.downloader import download_ohlcv, save_data, load_data, add_leveraged_series
from src.data.config import (
    TICKER,
    TRAIN_START, TRAIN_END,
    VAL_START,   VAL_END,
    TEST_START,  TEST_END,
)
from src.training import train
from src.evaluation import evaluate
from config import LEVERAGE_FACTOR, LOOKBACK_WINDOW

ROOT_DIR     = Path(__file__).resolve().parent
RESULTS_DIR  = ROOT_DIR / "results"


def _prepend_lookback(df_main: pd.DataFrame, df_prev: pd.DataFrame) -> pd.DataFrame:
    """
    Stellt LOOKBACK_WINDOW Tage aus df_prev voran, damit der Agent ab dem ersten
    Handelstag einer Phase ein volles Beobachtungsfenster hat.

    Die Leveraged-Serie wird für den kombinierten Datensatz neu berechnet,
    damit keine falschen Log-Renditen an der Naht entstehen.
    """
    combined  = pd.concat([df_prev.iloc[-LOOKBACK_WINDOW:], df_main])
    daily_ret = combined["Close"].pct_change().fillna(0)
    combined["Leveraged_Close"] = (
        combined["Close"].iloc[0] * (1 + LEVERAGE_FACTOR * daily_ret).cumprod()
    )
    return combined

# ──────────────────────────────────────────────────────────────────────────────
# Steuerung: Was soll ausgeführt werden?
# ──────────────────────────────────────────────────────────────────────────────
DOWNLOAD        = False  # Daten neu herunterladen (bereits gecacht in data/)
LAMBDA_TEST     = False  # Lambda-Sensitivitätsanalyse (Modellauswahl: Val, Evaluation: Test)
OPTIMIZE        = False  # Hyperparameteroptimierung (Grid Search auf Validierungsdaten)
TRAIN           = True   # Finales Training mit gewählten Hyperparametern
EVALUATE        = False  # Evaluation auf Validierungs- und Testdaten

# Finale Hyperparameter (nach Optimierung hier eintragen)
FINAL_RUN_NAME    = "final_model"
FINAL_LR          = 0.0001  # Grid Search Rang 1 (Sharpe_Mean=1.122, 5 Runs)
FINAL_GAMMA       = 0.95    # Grid Search Rang 1
FINAL_CLIP        = 0.1     # Grid Search Rang 1

# Anzahl Wiederholungen der Grid Search (Stabilitätsanalyse)
N_RUNS            = 5

# ──────────────────────────────────────────────────────────────────────────────
# 1. Daten herunterladen
# ──────────────────────────────────────────────────────────────────────────────
if DOWNLOAD:
    perioden = [
        ("dax_train.csv",      TRAIN_START, TRAIN_END),
        ("dax_validation.csv", VAL_START,   VAL_END),
        ("dax_test.csv",       TEST_START,  TEST_END),
    ]
    for dateiname, start, ende in perioden:
        df = download_ohlcv(TICKER, start, ende)
        save_data(df, dateiname)
        print(f"{dateiname}: {df.shape[0]} Handelstage ({start} bis {ende})\n")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Lambda-Sensitivitätsanalyse (nur wenn LAMBDA_TEST = True)
# ──────────────────────────────────────────────────────────────────────────────
if LAMBDA_TEST:
    print("\n" + "="*60)
    print("Lambda-Sensitivitaetsanalyse (Risikogewicht in Rewardfunktion)")
    print("="*60)

    df_train_lam    = add_leveraged_series(load_data("dax_train.csv"),      LEVERAGE_FACTOR)
    df_val_lam      = add_leveraged_series(load_data("dax_validation.csv"), LEVERAGE_FACTOR)
    df_test_lam     = add_leveraged_series(load_data("dax_test.csv"),       LEVERAGE_FACTOR)
    df_val_lam_ext  = _prepend_lookback(df_val_lam,  df_train_lam)
    df_test_lam_ext = _prepend_lookback(df_test_lam, df_val_lam)

    lambda_values = [1.0, 5.0, 10.0]
    lambda_results = []

    for lam in lambda_values:
        run_name = f"lambda_{str(lam).replace('.', '_')}"
        print(f"\n[Lambda = {lam}]  run: {run_name}")

        # Modellauswahl auf Validierungsdaten (Early Stopping + Best-Model-Tracking)
        model_path = train(
            df_train_lam, df_val=df_val_lam_ext, run_name=run_name,
            lambda_risk=lam,
        )
        # Evaluation auf Testdaten (2021–2025) – vom Modell nie gesehen
        metrics = evaluate(model_path, df_test_lam_ext, label=run_name, plot=True)

        lambda_results.append({
            "lambda_risk": lam,
            "run_name":    run_name,
            **metrics,
        })

    df_lambda = pd.DataFrame(lambda_results).sort_values("Sharpe Ratio", ascending=False)
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "lambda_sensitivity.csv"
    df_lambda.to_csv(csv_path, index=False)

    print("\n" + "="*60)
    print("Lambda-Ergebnisse (sortiert nach Sharpe Ratio):")
    print("="*60)
    cols = ["lambda_risk", "Gesamtrendite (%)", "Sharpe Ratio", "Max Drawdown (%)"]
    print(df_lambda[cols].to_string(index=False))
    print(f"\nTabelle gespeichert: {csv_path}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Hyperparameteroptimierung – Grid Search (nur wenn OPTIMIZE = True)
#    N_RUNS × 27 Kombinationen auf Trainingsdaten, Auswahl via Validierungs-Sharpe
# ──────────────────────────────────────────────────────────────────────────────
if OPTIMIZE:
    print("\n" + "="*60)
    print(f"Grid Search: {N_RUNS} Durchläufe × 27 Kombinationen")
    print("="*60)

    df_train_opt   = add_leveraged_series(load_data("dax_train.csv"),      LEVERAGE_FACTOR)
    df_val_opt     = add_leveraged_series(load_data("dax_validation.csv"), LEVERAGE_FACTOR)
    df_val_opt_ext = _prepend_lookback(df_val_opt, df_train_opt)

    learning_rates = [0.0001, 0.0003, 0.001]
    gammas         = [0.95, 0.99, 0.999]
    clip_ranges    = [0.1, 0.2, 0.3]
    kombinationen  = list(itertools.product(learning_rates, gammas, clip_ranges))

    RESULTS_DIR.mkdir(exist_ok=True)
    alle_dfs = []

    for run in range(1, N_RUNS + 1):
        print(f"\n{'-'*60}")
        print(f"Durchlauf {run}/{N_RUNS}")
        print(f"{'-'*60}")

        ergebnisse    = []
        trained_paths = []

        for i, (lr, gamma, eps) in enumerate(kombinationen, 1):
            hp_name = f"gs_r{run:02d}_k{i:02d}"
            print(f"  [{i:2d}/{len(kombinationen)}] lr={lr}  gamma={gamma}  clip={eps}")

            hp_path = train(
                df_train_opt, df_val=df_val_opt_ext, run_name=hp_name,
                learning_rate=lr, gamma=gamma, clip_range=eps,
            )
            metrics = evaluate(hp_path, df_val_opt_ext, label=hp_name, plot=False)

            ergebnisse.append({
                "Durchlauf":         run,
                "learning_rate":     lr,
                "gamma":             gamma,
                "clip_range":        eps,
                "Sharpe Ratio":      metrics["Sharpe Ratio"],
                "Gesamtrendite (%)": metrics["Gesamtrendite (%)"],
                "Max Drawdown (%)":  metrics["Max Drawdown (%)"],
            })
            trained_paths.append(hp_path)

        df_run = (pd.DataFrame(ergebnisse)
                  .sort_values("Sharpe Ratio", ascending=False)
                  .reset_index(drop=True))
        df_run.index += 1
        df_run.index.name = "Rang"
        df_run.to_csv(RESULTS_DIR / f"gridsearch_run{run:02d}.csv")
        alle_dfs.append(df_run)

        best = df_run.iloc[0]
        print(f"\n  Sieger Run {run}: lr={best['learning_rate']}  "
              f"gamma={best['gamma']}  clip={best['clip_range']}  "
              f"Sharpe={best['Sharpe Ratio']:.4f}")

        # Alle Modelle dieses Laufs löschen (Metriken sind in CSV gesichert)
        for hp_path in trained_paths:
            for suffix in [".zip", "_meta.json", "_episodes.csv"]:
                p = Path(str(hp_path) + suffix)
                if p.exists():
                    p.unlink()

    # ── Gesamtauswertung ──────────────────────────────────────────────────────
    df_concat = pd.concat(alle_dfs, ignore_index=True)

    df_summary = (
        df_concat
        .groupby(["learning_rate", "gamma", "clip_range"])
        .agg(
            Sharpe_Mean    = ("Sharpe Ratio",      "mean"),
            Sharpe_Std     = ("Sharpe Ratio",      "std"),
            Rendite_Mean   = ("Gesamtrendite (%)", "mean"),
            Rendite_Std    = ("Gesamtrendite (%)", "std"),
            Drawdown_Mean  = ("Max Drawdown (%)",  "mean"),
            Drawdown_Std   = ("Max Drawdown (%)",  "std"),
        )
        .reset_index()
        .sort_values("Sharpe_Mean", ascending=False)
        .reset_index(drop=True)
    )
    df_summary.index += 1
    df_summary.index.name = "Rang"
    df_summary.to_csv(RESULTS_DIR / "gridsearch_summary.csv")

    # Sieger-Häufigkeit: welche Kombination hat wie oft Rang 1 belegt
    df_siege = (
        df_concat
        .loc[df_concat.groupby("Durchlauf")["Sharpe Ratio"].idxmax()]
        .groupby(["learning_rate", "gamma", "clip_range"])
        .size()
        .reset_index(name="Siege")
        .sort_values("Siege", ascending=False)
    )
    df_siege.to_csv(RESULTS_DIR / "gridsearch_siege.csv", index=False)

    best_overall = df_summary.iloc[0]
    print("\n" + "="*60)
    print("Zusammenfassung – Mittelwert über alle Durchläufe (Top 10):")
    print("="*60)
    print(df_summary.head(10).to_string())
    print("\nSiege pro Konfiguration:")
    print(df_siege.to_string(index=False))
    print(f"\nEmpfehlung: lr={best_overall['learning_rate']}  "
          f"gamma={best_overall['gamma']}  clip={best_overall['clip_range']}")
    print("Trage diese Werte als FINAL_LR / FINAL_GAMMA / FINAL_CLIP oben ein.")
    print(f"\nGespeichert in: {RESULTS_DIR}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Finales Training – 5 unabhängige Durchläufe
#    Je Durchlauf: Training mit Early Stopping (Val 2020) + Evaluation (Val + Test)
#    Alle Agentenaktionen werden als Action-Log-CSV gespeichert.
# ──────────────────────────────────────────────────────────────────────────────
if TRAIN:
    print("\n" + "="*60)
    print(f"Finales Training: 5 Durchläufe")
    print(f"  lr={FINAL_LR}  gamma={FINAL_GAMMA}  clip={FINAL_CLIP}")
    print("="*60)

    df_train    = add_leveraged_series(load_data("dax_train.csv"),      LEVERAGE_FACTOR)
    df_val      = add_leveraged_series(load_data("dax_validation.csv"), LEVERAGE_FACTOR)
    df_test     = add_leveraged_series(load_data("dax_test.csv"),       LEVERAGE_FACTOR)
    df_val_ext  = _prepend_lookback(df_val,  df_train)
    df_test_ext = _prepend_lookback(df_test, df_val)

    RESULTS_DIR.mkdir(exist_ok=True)

    # Bereits vorhandene Ergebnisse laden (für kumulative Auswertung)
    existing_csv = RESULTS_DIR / "final_results.csv"
    if existing_csv.exists():
        existing_df  = pd.read_csv(existing_csv)
        start_run    = int(existing_df["Durchlauf"].max()) + 1
    else:
        existing_df  = None
        start_run    = 1
    end_run = start_run + 40  # immer 40 neue Durchläufe

    print(f"  Starte Durchläufe {start_run} bis {end_run - 1}")
    final_results = []

    for run in range(start_run, end_run):
        run_name = f"{FINAL_RUN_NAME}_run{run:02d}"
        print(f"\n{'-'*60}")
        print(f"Durchlauf {run}/{end_run - 1}: {run_name}")
        print(f"{'-'*60}")

        model_path = train(
            df_train, df_val=df_val_ext, run_name=run_name,
            learning_rate=FINAL_LR, gamma=FINAL_GAMMA, clip_range=FINAL_CLIP,
        )

        metrics_val  = evaluate(model_path, df_val_ext,  label=f"{run_name}_val",  plot=True)
        metrics_test = evaluate(model_path, df_test_ext, label=f"{run_name}_test", plot=True)

        final_results.append({
            "Durchlauf":            run,
            "run_name":             run_name,
            "Val Rendite (%)":      metrics_val["Gesamtrendite (%)"],
            "Val Sharpe":           metrics_val["Sharpe Ratio"],
            "Val Max DD (%)":       metrics_val["Max Drawdown (%)"],
            "Test Rendite (%)":     metrics_test["Gesamtrendite (%)"],
            "Test Sharpe":          metrics_test["Sharpe Ratio"],
            "Test Max DD (%)":      metrics_test["Max Drawdown (%)"],
            "BaH Val Rendite (%)":  metrics_val["BaH Gesamtrendite (%)"],
            "BaH Val Sharpe":       metrics_val["BaH Sharpe Ratio"],
            "BaH Test Rendite (%)": metrics_test["BaH Gesamtrendite (%)"],
            "BaH Test Sharpe":      metrics_test["BaH Sharpe Ratio"],
        })

    # Neue Ergebnisse mit bestehenden zusammenführen
    df_new = pd.DataFrame(final_results)
    if existing_df is not None:
        df_final = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        df_final = df_new
    df_final.to_csv(RESULTS_DIR / "final_results.csv", index=False)

    # Mittelwert und Standardabweichung
    numeric_cols = [c for c in df_final.columns if c not in ["Durchlauf", "run_name"]]
    df_summary_final = df_final[numeric_cols].agg(["mean", "std"]).round(4)

    print("\n" + "="*60)
    print(f"Zusammenfassung – {len(df_final)} finale Trainingsdurchläufe gesamt:")
    print("="*60)
    print(df_final[["Durchlauf", "Val Sharpe", "Val Rendite (%)",
                     "Test Sharpe", "Test Rendite (%)"]].to_string(index=False))
    print("\nMittelwert / Std:")
    print(df_summary_final.to_string())
    print(f"\nGespeichert: {RESULTS_DIR / 'final_results.csv'}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Evaluation
# ──────────────────────────────────────────────────────────────────────────────
if EVALUATE:
    model_path = Path("models") / FINAL_RUN_NAME

    df_train    = add_leveraged_series(load_data("dax_train.csv"),      LEVERAGE_FACTOR)
    df_val      = add_leveraged_series(load_data("dax_validation.csv"), LEVERAGE_FACTOR)
    df_test     = add_leveraged_series(load_data("dax_test.csv"),       LEVERAGE_FACTOR)
    df_val_ext  = _prepend_lookback(df_val,  df_train)
    df_test_ext = _prepend_lookback(df_test, df_val)

    print("\n" + "="*60)
    evaluate(model_path, df_val_ext,  label="validation")
    evaluate(model_path, df_test_ext, label="test")
