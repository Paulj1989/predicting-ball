# src/io/model_io.py

import pickle
from pathlib import Path
from typing import Any, Dict, List


def save_model(model: Dict[str, Any], filepath: Path) -> None:
    """Save model to disk"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: Path) -> Dict[str, Any]:
    """Load model from disk"""
    with open(filepath, "rb") as f:
        model = pickle.load(f)

    return model


def save_calibrators(calibrators: Dict[str, Any], filepath: Path) -> None:
    """Save calibrators to disk"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(calibrators, f)


def load_calibrators(filepath: Path) -> Dict[str, Any]:
    """Load calibrators from disk"""
    with open(filepath, "rb") as f:
        calibrators = pickle.load(f)

    return calibrators
