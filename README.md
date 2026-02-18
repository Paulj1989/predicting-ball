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
│   Output: production_model.pkl                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          2. CALIBRATE                                       │
│                                                                             │
│   Apply probability calibration:                                            │
│     - Outcome-specific temperature scaling (home/draw/away)                 │
│     - Holdout validation                                                    │
│                                                                             │
│   Output: calibrators.pkl                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          3. VALIDATE                                        │
│                                                                             │
│   Evaluate model performance:                                               │
│     - RPS, log loss, Brier score                                            │
│     - Comparison against bookmaker baselines                                │
│     - Calibration diagnostics                                               │
│                                                                             │
│   Output: validation/metrics.json                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          4. GENERATE PREDICTIONS                            │
│                                                                             │
│   Produce predictions and simulations:                                      │
│     - Fisher information matrix for analytical MLE uncertainty              │
│     - Monte Carlo season simulation (10,000 iterations)                     │
│       with MLE posterior draws and natural gradient rating updates          │
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
    --model-path outputs/models/production_model.pkl \
    --comprehensive \
    --outcome-specific

# validate model
uv run scripts/modeling/validate_model.py \
    --model-path outputs/models/production_model.pkl \
    --calibrator-path outputs/models/calibrators.pkl

# generate (and upload) predictions
uv run scripts/modeling/generate_predictions.py \
    --model-path outputs/models/production_model.pkl \
    --calibrator-path outputs/models/calibrators.pkl \
    --n-simulations 10000 \
    --hot-k-att 0.02 \
    --hot-k-def 0.01
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
│   │   ├── validate_model.py       # Model validation
│   │   └── generate_predictions.py # Prediction generation + upload
│   └── automation/                 # Database sync
│       └── download_db.py          # Download DuckDB from DO Spaces
└── run.py                          # Streamlit entry point
```

## Automated Workflows

GitHub Actions handle daily updates:

1. Downloads DuckDB from DigitalOcean Spaces
2. Checks if hyperparameter tuning is needed (every 30 days)
3. Trains model (with tuning if due, downloading previous model otherwise)
4. Runs calibration and validation
5. Generates predictions and uploads to DO Spaces
6. Triggers Streamlit app deployment
