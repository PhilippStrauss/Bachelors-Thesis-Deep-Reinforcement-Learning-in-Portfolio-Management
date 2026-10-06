# ──────────────────────────────────────────────
# Zentrale Konfiguration – hier alle Parameter
# ──────────────────────────────────────────────

# Asset
TICKER = "^GDAXI"

# Zeiträume (Abschnitt 3.3.2)
TRAIN_START = "2000-01-01"
TRAIN_END   = "2019-12-31"

VAL_START = "2020-01-01"
VAL_END   = "2020-12-31"

TEST_START = "2021-01-01"
TEST_END   = "2025-12-31"

# Environment
INITIAL_CAPITAL  = 100_000.0  # Startkapital in EUR
LOOKBACK_WINDOW  = 60         # Rollendes Zeitfenster (Handelstage) – Abschnitt 3.2.1
LEVERAGE_FACTOR  = 2.0        # Hebelfaktor synthetischer Leveraged-DAX

# Kosten (Abschnitt 3.2.3)
TRANSACTION_COST_RATE  = 0.001  # kappa: Transaktionskostenrate (0,1 % auf Gewichtsaenderungen)
HOLDING_COST_DAX       = 0.001  # ETF-Verwaltungsgebuehr DAX-ETF (0,1 % p.a.)
HOLDING_COST_LEVERAGED = 0.004  # ETF-Verwaltungsgebuehr Leveraged-ETF (0,4 % p.a.)

# Reward (Abschnitt 3.2.4)
LAMBDA_RISK   = 5.0  # Risikoaversionsparameter lambda
REWARD_WINDOW = 60   # Fenster fuer rollierende Varianz (Handelstage)

# Training / PPO-Hyperparameter (Abschnitte 3.2.5 / 3.4.3 / 3.4.4)
TOTAL_TIMESTEPS = 200_000  # Gesamtanzahl simulierter Handelstage
EPISODE_LENGTH  = 252      # Episodenlänge: 1 Handelsjahr (zufaelliger Startzeitpunkt)
EVAL_FREQ       = 5_000    # Evaluierungsintervall in Timesteps (Early Stopping)
PATIENCE        = 5        # Evaluierungen ohne Verbesserung vor Early Stopping
N_ENVS          = 1        # Anzahl paralleler Environments (N)
LEARNING_RATE   = 0.0003   # Lernrate (alpha)
N_STEPS         = 2048     # Rollout-Laenge: Schritte zwischen zwei Policy-Updates (T)
BATCH_SIZE      = 64       # Mini-Batch-Groesse (M)
N_EPOCHS        = 10       # Updates pro Rollout (K)
GAMMA           = 0.99     # Diskontfaktor (gamma)
GAE_LAMBDA      = 0.95     # GAE-Lambda fuer Advantage-Schaetzung (lambda)
CLIP_RANGE      = 0.2      # PPO-Clipping-Parameter (epsilon)
ENT_COEF        = 0.0      # Entropy-Koeffizient (c_2)
VF_COEF         = 0.5      # Value-Loss-Koeffizient (c_1)
