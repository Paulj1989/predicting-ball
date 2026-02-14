#!/usr/bin/env python3
"""Check if calibration is still working on latest matches"""

from src.evaluation.metrics import evaluate_model_comprehensive
from src.io.model_io import load_calibrators, load_model
from src.models.calibration import validate_calibration_on_holdout
from src.processing.model_preparation import prepare_bundesliga_data

# load
model = load_model("outputs/models/production_model.pkl")
calibrators = load_calibrators("outputs/models/calibrators.pkl")

# get latest matches
_, current_season = prepare_bundesliga_data(verbose=False)
latest_matches = current_season[current_season["is_played"]].tail(50)

# get predictions
_, predictions, actuals = evaluate_model_comprehensive(
    model["params"], latest_matches, use_dixon_coles=True
)

# validate
metrics = validate_calibration_on_holdout(
    calibrators["temperature"], predictions, actuals, verbose=True
)

# alert if calibration degraded
if metrics["rps_improvement"] < -0.005:
    print("\n WARNING: Calibration has degraded!")
    print("  Consider recalibrating with recent data")

else:
    print("\n✓ Calibration still effective")
