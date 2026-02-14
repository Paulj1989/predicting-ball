# src/validation/splits.py

from collections.abc import Iterator

import numpy as np
import pandas as pd


class TimeSeriesSplit:
    """
    Time-series cross-validation splitter.

    Ensures that test data always comes after training data,
    preventing data leakage.
    """

    def __init__(self, n_splits: int = 5, test_size: int | None = None, gap: int = 0):
        """Initialise time-series splitter"""
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, X: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate indices for train/test splits"""
        n_samples = len(X)

        if self.test_size is None:
            # expanding window: test size grows
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            # calculate split point
            if self.test_size is None:
                # expanding window
                split_point = (i + 1) * test_size
                test_end = split_point + test_size
            else:
                # rolling window with fixed test size
                split_point = n_samples - (self.n_splits - i) * test_size
                test_end = split_point + test_size

            if test_end > n_samples:
                break

            # apply gap
            train_end = split_point - self.gap

            if train_end < 50:  # minimum training size
                continue

            train_indices = indices[:train_end]
            test_indices = indices[split_point:test_end]

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits


def create_train_test_split(
    data: pd.DataFrame,
    test_seasons: list[int],
    date_column: str = "date",
    season_column: str = "season_end_year",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets by season"""
    # ensure data is sorted by date
    data = data.sort_values(date_column).reset_index(drop=True)

    # split by season
    train_data = data[~data[season_column].isin(test_seasons)].copy()
    test_data = data[data[season_column].isin(test_seasons)].copy()

    # verify no temporal leakage
    if len(train_data) > 0 and len(test_data) > 0:
        last_train_date = train_data[date_column].max()
        first_test_date = test_data[date_column].min()

        if last_train_date >= first_test_date:
            print("  Warning: Temporal overlap detected")
            print(f"  Last train date: {last_train_date}")
            print(f"  First test date: {first_test_date}")

    return train_data, test_data


def create_calibration_split(
    train_data: pd.DataFrame,
    calibration_fraction: float = 0.15,
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training data into fitting set and calibration set.

    Uses the most recent portion of training data for calibration
    to ensure it's representative of current conditions.
    """
    # sort by date
    train_sorted = train_data.sort_values(date_column).reset_index(drop=True)

    # calculate split point
    n_total = len(train_sorted)
    n_calibration = int(n_total * calibration_fraction)
    n_calibration = max(n_calibration, 50)  # minimum 50 matches

    # split
    fit_data = train_sorted.iloc[:-n_calibration].copy()
    calibration_data = train_sorted.iloc[-n_calibration:].copy()

    return fit_data, calibration_data


def validate_split_quality(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    date_column: str = "date",
    verbose: bool = True,
) -> dict:
    """
    Validate quality of train/test split.

    Checks for:
    - Temporal leakage
    - Sufficient data in both sets
    - Team overlap
    """
    results = {
        "temporal_leakage": False,
        "sufficient_train_data": False,
        "team_overlap": False,
        "issues": [],
    }

    # check temporal ordering
    last_train_date = train_data[date_column].max()
    first_test_date = test_data[date_column].min()

    if last_train_date >= first_test_date:
        results["temporal_leakage"] = True
        results["issues"].append(
            f"Temporal leakage: train extends to {last_train_date}, "
            f"test starts at {first_test_date}"
        )

    # check training data size
    if len(train_data) >= 300:
        results["sufficient_train_data"] = True
    else:
        results["issues"].append(
            f"Insufficient training data: {len(train_data)} matches (recommend 300+)"
        )

    # check team overlap
    train_teams = set(train_data["home_team"].unique()) | set(train_data["away_team"].unique())
    test_teams = set(test_data["home_team"].unique()) | set(test_data["away_team"].unique())

    teams_only_in_test = test_teams - train_teams

    if len(teams_only_in_test) == 0:
        results["team_overlap"] = True
    else:
        results["issues"].append(f"Teams in test but not train: {teams_only_in_test}")

    if verbose:
        print("\n" + "=" * 60)
        print("SPLIT QUALITY VALIDATION")
        print("=" * 60)

        if not results["temporal_leakage"]:
            print("✓ No temporal leakage")
        else:
            print("✗ Temporal leakage detected")

        if results["sufficient_train_data"]:
            print(f"✓ Sufficient training data ({len(train_data)} matches)")
        else:
            print(f"⚠ Limited training data ({len(train_data)} matches)")

        if results["team_overlap"]:
            print("✓ All test teams appear in training")
        else:
            print(f"⚠ {len(teams_only_in_test)} teams only in test set")

        if results["issues"]:
            print("\nIssues found:")
            for issue in results["issues"]:
                print(f"  - {issue}")

    return results
