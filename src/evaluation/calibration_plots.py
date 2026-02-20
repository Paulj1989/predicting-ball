# src/evaluation/calibration_plots.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# color palette
COLORS = {
    "primary": "#026E99",
    "secondary": "#D93649",
    "accent": "#FFA600",
}

# set seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)


def _convert_inputs(
    predictions: pd.DataFrame | np.ndarray, actuals: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to numpy arrays with proper format"""
    # convert predictions
    if isinstance(predictions, pd.DataFrame):
        required_cols = ["home_win", "draw", "away_win"]
        if not all(col in predictions.columns for col in required_cols):
            raise ValueError(f"DataFrame must have columns {required_cols}")
        predictions = predictions[required_cols].values
    elif not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    # convert actuals
    if isinstance(actuals, pd.Series):
        actuals = actuals.values
    elif not isinstance(actuals, np.ndarray):
        actuals = np.array(actuals)

    # handle string vs integer outcomes
    if actuals.dtype.kind in ("U", "O"):
        outcome_map = {"H": 0, "D": 1, "A": 2}
        try:
            actuals = np.array([outcome_map[a] for a in actuals])
        except KeyError as e:
            raise ValueError(f"Unknown outcome: {e}") from e
    else:
        actuals = actuals.astype(int)

    return predictions, actuals


def calculate_expected_calibration_error(
    predictions: pd.DataFrame | np.ndarray,
    actuals: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures the average difference between predicted probabilities
    and empirical frequencies across binned predictions.
    """
    # convert to numpy arrays
    predictions, actuals = _convert_inputs(predictions, actuals)

    # for multi-class, use maximum predicted probability
    if predictions.ndim > 1:
        max_probs = predictions.max(axis=1)
        pred_classes = predictions.argmax(axis=1)
        correct = (pred_classes == actuals).astype(float)
    else:
        max_probs = predictions
        correct = actuals

    # create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        # find predictions in this bin
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # average confidence in bin
            avg_confidence = max_probs[in_bin].mean()
            # average accuracy in bin
            avg_accuracy = correct[in_bin].mean()
            # add weighted difference to ECE
            ece += prop_in_bin * abs(avg_confidence - avg_accuracy)

    return ece


def plot_calibration_curve(
    predictions: pd.DataFrame | np.ndarray,
    actuals: pd.Series | np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (18, 5),
) -> None:
    """Plot calibration curve (reliability diagram)"""
    # convert to numpy arrays
    predictions, actuals = _convert_inputs(predictions, actuals)

    # for multi-class, analyse each class separately
    if predictions.ndim > 1:
        _fig, axes = plt.subplots(1, 3, figsize=figsize)
        class_names = ["Home Win", "Draw", "Away Win"]
        colors = [COLORS["primary"], COLORS["accent"], COLORS["secondary"]]

        for class_idx, (ax, class_name, color) in enumerate(
            zip(axes, class_names, colors, strict=False)
        ):
            pred_probs = predictions[:, class_idx]
            actual_binary = (actuals == class_idx).astype(float)

            _plot_single_calibration(
                pred_probs,
                actual_binary,
                n_bins,
                ax,
                f"{class_name} Calibration",
                color=color,
            )

        plt.tight_layout()
    else:
        # binary case
        _fig, ax = plt.subplots(figsize=(8, 6))
        _plot_single_calibration(
            predictions, actuals, n_bins, ax, title, color=COLORS["primary"]
        )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Calibration plot saved: {save_path}")
    else:
        default_path = "outputs/evaluation/figures/calibration_curve.png"
        plt.savefig(default_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Calibration plot saved: {default_path}")

    plt.close()


def _plot_single_calibration(
    pred_probs: np.ndarray,
    actual_binary: np.ndarray,
    n_bins: int,
    ax: plt.Axes,
    title: str,
    color: str = "#026E99",
) -> None:
    """Helper function to plot single calibration curve"""
    # ensure inputs are numpy arrays
    pred_probs = np.asarray(pred_probs)
    actual_binary = np.asarray(actual_binary)

    # create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    # calculate empirical frequencies in each bin
    bin_pred_means = []
    bin_true_freqs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (pred_probs >= bin_boundaries[i]) & (pred_probs < bin_boundaries[i + 1])

        if i == n_bins - 1:  # last bin includes upper boundary
            in_bin = (pred_probs >= bin_boundaries[i]) & (pred_probs <= bin_boundaries[i + 1])

        if in_bin.sum() > 0:
            bin_pred_means.append(pred_probs[in_bin].mean())
            bin_true_freqs.append(actual_binary[in_bin].mean())
            bin_counts.append(int(in_bin.sum()))
        else:
            bin_pred_means.append(np.nan)
            bin_true_freqs.append(np.nan)
            bin_counts.append(0)

    # plot perfect calibration line
    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=2,
        label="Perfect Calibration",
        alpha=0.7,
        zorder=1,
    )

    # plot bars with enhanced visibility
    bin_width = 1.0 / n_bins
    for i, (pred_mean, true_freq, count) in enumerate(
        zip(bin_pred_means, bin_true_freqs, bin_counts, strict=False)
    ):
        if not np.isnan(pred_mean) and count > 0:
            # varying alpha based on calibration quality
            calibration_gap = abs(pred_mean - true_freq)
            alpha = 0.8 if calibration_gap < 0.1 else 0.6

            # plot bar from 0 to true_freq
            ax.bar(
                bin_centers[i],
                true_freq,
                width=bin_width * 0.9,
                alpha=alpha,
                color=color,
                edgecolor="black",
                linewidth=1.5,
                zorder=2,
            )

            # add count labels
            ax.text(
                bin_centers[i],
                true_freq + 0.02,
                str(count),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # add gap visualisation (only for bins with data)
    for i, (pred_mean, true_freq, count) in enumerate(
        zip(bin_pred_means, bin_true_freqs, bin_counts, strict=False)
    ):
        if not np.isnan(pred_mean) and count > 0:
            # draw line from predicted to actual
            ax.plot(
                [bin_centers[i], bin_centers[i]],
                [pred_mean, true_freq],
                color=COLORS["secondary"],
                alpha=0.6,
                linewidth=2.5,
                zorder=3,
            )

    ax.set_xlabel("Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Observed Frequency", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1.05))  # slightly higher to accommodate labels
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)

    # calculate and display ECE
    ece = calculate_expected_calibration_error(pred_probs, actual_binary, n_bins)
    ax.text(
        0.95,
        0.05,
        f"ECE: {ece:.4f}",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor=color,
            linewidth=2,
            alpha=0.9,
        ),
    )


def plot_calibration_comparison(
    predictions_dict: dict[str, pd.DataFrame | np.ndarray],
    actuals: pd.Series | np.ndarray,
    n_bins: int = 10,
    save_path: str | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """Compare calibration of multiple models"""
    n_models = len(predictions_dict)

    # determine figure size
    if figsize is None:
        figsize = (6 * n_models, 5)

    color_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
    if n_models > 3:
        # extend with seaborn palette if needed
        extended_palette = sns.color_palette("husl", n_models)
        color_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]] + [
            "#{:02x}{:02x}{:02x}".format(*tuple(int(c * 255) for c in rgb))
            for rgb in extended_palette[3:]
        ]

    # convert actuals once
    _, actuals_array = _convert_inputs(np.zeros((len(actuals), 3)), actuals)

    # for multi-class, show home win probability only (for simplicity)
    _fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, preds), color in zip(
        axes, predictions_dict.items(), color_cycle, strict=False
    ):
        # convert predictions
        preds_array, _ = _convert_inputs(preds, actuals)

        if preds_array.ndim > 1:
            # use home win probability
            pred_probs = preds_array[:, 0]
            actual_binary = (actuals_array == 0).astype(float)
        else:
            pred_probs = preds_array
            actual_binary = actuals_array

        _plot_single_calibration(
            pred_probs, actual_binary, n_bins, ax, f"{model_name}", color=color
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Calibration comparison saved: {save_path}")
    else:
        default_path = "outputs/evaluation/figures/calibration_comparison.png"
        plt.savefig(default_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Calibration comparison saved: {default_path}")

    plt.close()


def create_calibration_report(
    predictions: pd.DataFrame | np.ndarray,
    actuals: pd.Series | np.ndarray,
    model_name: str = "Model",
    save_path: str | Path | None = None,
) -> dict[str, float]:
    """Create comprehensive calibration report"""
    # convert to numpy arrays
    predictions, actuals = _convert_inputs(predictions, actuals)

    # calculate metrics
    from .metrics import calculate_brier_score, calculate_rps

    ece = calculate_expected_calibration_error(predictions, actuals)
    brier = calculate_brier_score(predictions, actuals)
    rps = calculate_rps(predictions, actuals)

    # create plot
    plot_calibration_curve(
        predictions, actuals, title=f"{model_name} Calibration", save_path=save_path
    )

    # print report
    print("\n" + "=" * 60)
    print(f"CALIBRATION REPORT: {model_name}")
    print("=" * 60)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Ranked Probability Score: {rps:.4f}")

    if ece < 0.05:
        print("✓ Excellent calibration")
    elif ece < 0.10:
        print("✓ Good calibration")
    elif ece < 0.15:
        print("⚠ Moderate calibration")
    else:
        print("✗ Poor calibration - consider post-hoc calibration")

    return {
        "ece": ece,
        "brier_score": brier,
        "rps": rps,
    }


def plot_confidence_histogram(
    predictions: pd.DataFrame | np.ndarray,
    actuals: pd.Series | np.ndarray,
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot histogram of prediction confidence levels.

    Shows distribution of predicted probabilities and whether
    predictions were correct or incorrect.
    """
    # convert to numpy arrays
    predictions, actuals = _convert_inputs(predictions, actuals)

    _fig, ax = plt.subplots(figsize=figsize)

    # get maximum predicted probabilities
    if predictions.ndim > 1:
        max_probs = predictions.max(axis=1)
        pred_classes = predictions.argmax(axis=1)
        correct = pred_classes == actuals
    else:
        max_probs = predictions
        correct = predictions > 0.5

    # separate into correct and incorrect
    correct_probs = max_probs[correct]
    incorrect_probs = max_probs[~correct]

    # plot histograms
    bins = np.linspace(0, 1, 21).tolist()

    ax.hist(
        correct_probs,
        bins=bins,
        alpha=0.7,
        color=COLORS["primary"],
        label=f"Correct ({len(correct_probs)})",
        edgecolor="black",
        linewidth=1,
    )
    ax.hist(
        incorrect_probs,
        bins=bins,
        alpha=0.7,
        color=COLORS["secondary"],
        label=f"Incorrect ({len(incorrect_probs)})",
        edgecolor="black",
        linewidth=1,
    )

    ax.set_xlabel("Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax.set_title(
        "Distribution of Prediction Confidence", fontsize=14, fontweight="bold", pad=15
    )
    ax.legend(loc="upper center", frameon=True, fancybox=True, shadow=True)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5, axis="y")

    # add accuracy line
    accuracy = correct.mean()
    ax.axhline(
        y=len(max_probs) * accuracy / 20,
        color=COLORS["accent"],
        linestyle="--",
        linewidth=2,
        label=f"Accuracy: {accuracy:.1%}",
        alpha=0.7,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Confidence histogram saved: {save_path}")
    else:
        default_path = "outputs/evaluation/figures/confidence_histogram.png"
        plt.savefig(default_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Confidence histogram saved: {default_path}")

    plt.close()


def plot_reliability_diagram(
    predictions: np.ndarray,
    actuals: np.ndarray,
    outcome_idx: int,
    outcome_label: str,
    ax: plt.Axes,
    n_bins: int = 10,
) -> None:
    """
    Plot a reliability diagram for one outcome on a given axis.

    predictions: shape (N, 3) — columns are [home_win, draw, away_win]
    actuals: integer array 0=H, 1=D, 2=A, length N
    outcome_idx: which column of predictions and actuals value to plot (0, 1, or 2)
    """
    pred_probs = predictions[:, outcome_idx]
    actual_binary = (actuals == outcome_idx).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = []
    observed_freqs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # include upper edge in the last bin
        mask = (pred_probs >= lo) & (pred_probs < hi if i < n_bins - 1 else pred_probs <= hi)
        if mask.sum() == 0:
            continue
        bin_centres.append(pred_probs[mask].mean())
        observed_freqs.append(actual_binary[mask].mean())
        bin_counts.append(mask.sum())

    bin_centres = np.array(bin_centres)
    observed_freqs = np.array(observed_freqs)
    bin_counts = np.array(bin_counts)

    # perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Perfect")

    # calibration curve
    ax.plot(
        bin_centres,
        observed_freqs,
        "o-",
        color=COLORS["primary"],
        linewidth=2,
        markersize=5,
        label="Model",
    )

    # histogram of prediction density (secondary y-axis)
    ax2 = ax.twinx()
    ax2.bar(
        bin_centres,
        bin_counts,
        width=0.08,
        alpha=0.2,
        color="steelblue",
        label="Count",
    )
    ax2.set_ylabel("Predictions per bin", fontsize=9)
    ax2.tick_params(labelsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability", fontsize=10)
    ax.set_ylabel("Observed frequency", fontsize=10)
    ax.set_title(outcome_label, fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(labelsize=9)


def save_reliability_diagrams(
    predictions: np.ndarray,
    actuals: np.ndarray,
    output_path: str | Path,
    title: str = "Reliability Diagrams",
    n_bins: int = 10,
) -> None:
    """
    Save reliability diagrams for all three outcomes as a 3-panel figure.

    predictions: shape (N, 3) — [home_win, draw, away_win]
    actuals: integer array 0=H, 1=D, 2=A
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    outcomes = [
        (0, "Home Win"),
        (1, "Draw"),
        (2, "Away Win"),
    ]

    for ax, (idx, label) in zip(axes, outcomes, strict=False):
        plot_reliability_diagram(predictions, actuals, idx, label, ax, n_bins=n_bins)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
