# src/evaluation/calibration_plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List

from .metrics import calculate_accuracy


# define colour palette
COLORS = {
    "primary": "#026E99",  # blue
    "secondary": "#D93649",  # red
    "accent": "#FFA600",  # yellow
}

# set seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)


def calculate_expected_calibration_error(
    predictions: np.ndarray, actuals: np.ndarray, n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures the average difference between predicted probabilities
    and empirical frequencies across binned predictions.
    """
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

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
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
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 5),
) -> None:
    """
    Plot calibration curve (reliability diagram).

    Shows predicted probability vs. observed frequency.
    Perfect calibration follows the diagonal line.
    """
    # for multi-class, analyse each class separately
    if predictions.ndim > 1:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        class_names = ["Home Win", "Draw", "Away Win"]
        colors = [COLORS["primary"], COLORS["accent"], COLORS["secondary"]]

        for class_idx, (ax, class_name, color) in enumerate(
            zip(axes, class_names, colors)
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
        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_single_calibration(
            predictions, actuals, n_bins, ax, title, color=COLORS["primary"]
        )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Calibration plot saved: {save_path}")
    else:
        default_path = "outputs/figures/calibration_curve.png"
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
    # create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    # calculate empirical frequencies in each bin
    bin_pred_means = []
    bin_true_freqs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (pred_probs >= bin_boundaries[i]) & (
            pred_probs < bin_boundaries[i + 1]
        )

        if i == n_bins - 1:  # last bin includes upper boundary
            in_bin = (pred_probs >= bin_boundaries[i]) & (
                pred_probs <= bin_boundaries[i + 1]
            )

        if in_bin.sum() > 0:
            bin_pred_means.append(pred_probs[in_bin].mean())
            bin_true_freqs.append(actual_binary[in_bin].mean())
            bin_counts.append(in_bin.sum())
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

    # plot bars
    bin_width = 1.0 / n_bins
    for i, (pred_mean, true_freq, count) in enumerate(
        zip(bin_pred_means, bin_true_freqs, bin_counts)
    ):
        if not np.isnan(pred_mean):
            # use primary color with varying alpha based on calibration quality
            calibration_gap = abs(pred_mean - true_freq)
            alpha = 0.9 if calibration_gap < 0.1 else 0.6

            ax.bar(
                bin_centers[i],
                true_freq,
                width=bin_width * 0.8,
                alpha=alpha,
                color=color,
                edgecolor="black",
                linewidth=1,
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

    # add gap visualisation
    for pred_mean, true_freq in zip(bin_pred_means, bin_true_freqs):
        if not np.isnan(pred_mean):
            ax.plot(
                [pred_mean, pred_mean],
                [pred_mean, true_freq],
                color=COLORS["secondary"],
                alpha=0.5,
                linewidth=2,
                zorder=3,
            )

    ax.set_xlabel("Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Observed Frequency", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
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
    predictions_dict: Dict[str, np.ndarray],
    actuals: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """Compare calibration of multiple models"""
    n_models = len(predictions_dict)

    # determine figure size
    if figsize is None:
        figsize = (6 * n_models, 5)

    # create colour cycle
    color_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
    if n_models > 3:
        # extend with seaborn palette if needed
        extended_palette = sns.color_palette("husl", n_models)
        color_cycle = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]] + [
            "#%02x%02x%02x" % tuple(int(c * 255) for c in rgb)
            for rgb in extended_palette[3:]
        ]

    # for multi-class, show home win probability only (for simplicity)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, preds), color in zip(
        axes, predictions_dict.items(), color_cycle
    ):
        if preds.ndim > 1:
            # use home win probability
            pred_probs = preds[:, 0]
            actual_binary = (actuals == 0).astype(float)
        else:
            pred_probs = preds
            actual_binary = actuals

        _plot_single_calibration(
            pred_probs, actual_binary, n_bins, ax, f"{model_name}", color=color
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Calibration comparison saved: {save_path}")
    else:
        default_path = "outputs/figures/calibration_comparison.png"
        plt.savefig(default_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Calibration comparison saved: {default_path}")

    plt.close()


def create_calibration_report(
    predictions: np.ndarray,
    actuals: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """Create comprehensive calibration report"""
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
    predictions: np.ndarray,
    actuals: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot histogram of prediction confidence levels.

    Shows distribution of predicted probabilities and whether
    predictions were correct or incorrect.
    """
    fig, ax = plt.subplots(figsize=figsize)

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
    bins = np.linspace(0, 1, 21)

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
        default_path = "outputs/figures/confidence_histogram.png"
        plt.savefig(default_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Confidence histogram saved: {default_path}")

    plt.close()
