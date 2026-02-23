# predicting-ball

A pipeline for fitting a two-stage Dixon-Coles Poisson model that predicts match outcomes and simulates season standings in the Bundesliga. The pipeline consumes match data from a DuckDB database, trains a Dixon-Coles model, calibrates probabilities, generates predictions, and displays results via the [Predicting Ball](https://predicting-ball.co.uk) Streamlit app.

## Pipeline Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCE                                    │
│                                                                             │
│   DuckDB Database                                                           │
│   Match results, xG, squad values, Elo ratings, betting odds                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. TRAIN MODEL                                     │
│                                                                             │
│   Fit two-stage Dixon-Coles Poisson model:                                  │
│     Stage 1: Team strengths & home advantage                                │
│     Stage 2: Form residual coefficient + odds blend weight                  │
│                                                                             │
│   Model components:                                                         │
│     - Informed priors (Elo + squad value + previous season rating)          │
│     - Optuna hyperparameter optimisation (monthly)                          │
│                                                                             │
│   Output: buli_model.pkl                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          2. CALIBRATE                                       │
│                                                                             │
│   Apply probability calibration:                                            │
│     - Global temperature scaling                                            │
│     - Holdout validation                                                    │
│                                                                             │
│   Output: calibrators.pkl                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          3. CALIBRATION CHECK                               │
│                                                                             │
│   Verify calibrators are still effective on recent data:                    │
│     - RPS before and after applying calibrators                             │
│     - Exits non-zero if calibration has degraded                            │
│     - Skippable with --skip-check flag                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          4. GENERATE PREDICTIONS                            │
│                                                                             │
│   Produce predictions and simulations:                                      │
│     - MLE posterior draws for parameter uncertainty                         │
│     - Hot Monte Carlo season simulation (10,000 iterations)                 │
│     - Match-level predictions with calibrated probabilities                 │
│                                                                             │
│   Outputs uploaded to DO Spaces:                                            │
│     - serving/latest_buli_matches.parquet                                   │
│     - serving/latest_buli_projections.parquet                               │
│     - serving/buli_model.pkl, buli_calibrators.pkl                          │
│     - incoming/buli_run_{timestamp}.parquet (for monitoring).               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT APP                                      │
│                                                                             │
│   Reads from DO Spaces:                                                     │
│     - Season projections with title/UCL/relegation probabilities            │
│     - Team strength ratings (overall/attack/defense)                        │
│     - Match predictions for upcoming fixtures                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Running the Pipeline

### Full Pipeline

Run the complete pipeline with a single command:

```bash
uv run scripts/modeling/run_model_pipeline.py
```

Pull fresh data from DO Spaces before running:

```bash
uv run scripts/modeling/run_model_pipeline.py --refresh
```

Run with hyperparameter tuning (monthly):

```bash
uv run scripts/modeling/run_model_pipeline.py --tune --n-trials 50
```

### Individual Pipeline Stages

Run each stage independently:

```bash
# download database
uv run scripts/automation/download_db.py

# train model (using existing hyperparameters)
uv run scripts/modeling/train_model.py

# train with tuning
uv run scripts/modeling/train_model.py --tune --n-trials 50 --metric rps

# calibrate model
uv run scripts/modeling/run_calibration.py \
    --model-path outputs/models/buli_model.pkl

# calibration check (quick in-pipeline check)
uv run scripts/evaluation/check_calibration.py \
    --model-path outputs/models/buli_model.pkl \
    --calibrator-path outputs/models/calibrators.pkl

# generate (and upload) predictions
uv run scripts/modeling/generate_predictions.py \
    --model-path outputs/models/buli_model.pkl \
    --calibrator-path outputs/models/calibrators.pkl \
    --n-simulations 10000 \
    --hot-k-att 0.05 \
    --hot-k-def 0.025
```

### Evaluation Scripts

```bash
# walk-forward backtest (genuine out-of-sample evaluation)
uv run scripts/evaluation/run_backtest.py --n-seasons 3

# in-sample diagnostics and odds blend ablation
uv run scripts/evaluation/inspect_model.py \
    --model-path outputs/models/buli_model.pkl

# weekly live monitoring (run from CI or manually)
uv run scripts/evaluation/run_monitoring.py --lookback-days 60
```

### Running the App Locally

```bash
streamlit run run.py
```

## Project Structure

```text
predicting-ball/
├── app/
│   ├── main.py                     # App framework, loads data from DO Spaces
│   ├── pages/                      # Individual app pages
│   │   ├── projections.py          # Season projections table
│   │   ├── team_strengths.py       # Attack/defense ratings visualisation
│   │   ├── fixtures.py             # Match predictions
│   │   └── about.py                # Model methodology
│   ├── components/                 # Reusable UI components
│   └── styles/                     # Custom CSS
├── src/
│   ├── models/                     # Core modeling code
│   │   ├── poisson.py              # Team attack/defense ratings
│   │   ├── dixon_coles.py          # Correction for low-scoring matches
│   │   ├── fisher_information.py   # Analytical MLE uncertainty
│   │   ├── priors.py               # Informed priors
│   │   ├── hyperparameters.py      # Optuna-based tuning
│   │   └── calibration.py          # Temperature scaling
│   ├── simulation/                 # Prediction and simulation
│   │   ├── hot_simulation.py       # Season sim with MLE draws + rating updates
│   │   ├── monte_carlo.py          # Legacy season simulation
│   │   └── predictions.py          # Match-level predictions
│   ├── evaluation/                 # Model evaluation
│   │   ├── metrics.py              # RPS, log loss, Brier score
│   │   ├── significance.py         # Diebold-Mariano significance testing
│   │   ├── calibration_plots.py    # Reliability diagrams and calibration plots
│   │   └── baselines.py            # Bookmaker baseline comparisons
│   ├── validation/                 # Cross-validation and diagnostics
│   ├── features/                   # Feature engineering
│   ├── processing/                 # Data preparation
│   ├── io/                         # Model I/O and DO Spaces integration
│   └── visualisation/              # Plotting utilities
├── scripts/
│   ├── modeling/                   # Pipeline scripts
│   │   ├── run_model_pipeline.py   # Full pipeline orchestration
│   │   ├── train_model.py          # Model training
│   │   ├── run_calibration.py      # Probability calibration
│   │   └── generate_predictions.py # Prediction generation + upload
│   ├── evaluation/                 # Evaluation and diagnostics
│   │   ├── check_calibration.py    # Quick in-pipeline calibration check
│   │   ├── inspect_model.py        # In-sample diagnostics and odds ablation
│   │   ├── run_backtest.py         # Walk-forward backtesting
│   │   └── run_monitoring.py       # Weekly live monitoring
│   └── automation/                 # Database sync
│       └── download_db.py          # Download DuckDB from DO Spaces
└── run.py                          # Streamlit entry point
```

## Automated Workflows

GitHub Actions handle daily updates:

1. Downloads DuckDB from DigitalOcean Spaces
2. Checks if hyperparameter tuning is needed (every 30 days)
3. Trains model (with tuning if due, downloading previous model otherwise)
4. Runs calibration and calibration check
5. Generates predictions and uploads to DO Spaces
6. Triggers Streamlit app deployment

A separate weekly workflow runs every Monday and scores recent production predictions against actual results, reporting rolling RPS trends and alerting on drift or calibration issues.
