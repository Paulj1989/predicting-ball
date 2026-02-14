# src/io/model_io.py

import pickle
from pathlib import Path
from typing import Any


def save_model(model: dict[str, Any], filepath: Path) -> None:
    """Save model to disk"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: str | Path) -> dict[str, Any]:
    """Load model from disk"""
    with open(filepath, "rb") as f:
        model = pickle.load(f)

    return model


def save_calibrators(calibrators: dict[str, Any], filepath: Path) -> None:
    """Save calibrators to disk"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(calibrators, f)


def load_calibrators(filepath: str | Path) -> dict[str, Any]:
    """Load calibrators from disk"""
    with open(filepath, "rb") as f:
        calibrators = pickle.load(f)

    return calibrators
