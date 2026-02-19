# src/visualisation/diagnostics.py

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# define colour palette
COLORS = {
    "primary": "#026E99",  # blue
    "secondary": "#D93649",  # red
    "accent": "#FFA600",  # yellow
}

# set seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)


def plot_residual_analysis(
    predictions: np.ndarray,
    actuals: np.ndarray,
    save_path: str | None = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Plot residual analysis for model predictions"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # get predicted outcome and confidence
    predicted_class = np.argmax(predictions, axis=1)
    predicted_prob = np.max(predictions, axis=1)

    # correctness
    correct = (predicted_class == actuals).astype(int)

    # plot 1: confidence vs correctness
    ax1 = axes[0]

    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    accuracy_by_confidence = []
    counts_by_bin = []

    for i in range(len(bins) - 1):
        mask = (predicted_prob >= bins[i]) & (predicted_prob < bins[i + 1])
        if i == len(bins) - 2:  # last bin includes upper bound
            mask = (predicted_prob >= bins[i]) & (predicted_prob <= bins[i + 1])

        if mask.sum() > 0:
            accuracy_by_confidence.append(correct[mask].mean())
            counts_by_bin.append(mask.sum())
        else:
            accuracy_by_confidence.append(0)
            counts_by_bin.append(0)

    ax1.plot(
        bin_centers,
        accuracy_by_confidence,
        "o-",
        linewidth=3,
        markersize=10,
        color=COLORS["primary"],
        markeredgecolor="black",
        markeredgewidth=1.5,
        label="Empirical Accuracy",
    )
    ax1.plot(
        [0, 1],
        [0, 1],
        "--",
        alpha=0.7,
        linewidth=2.5,
        color="black",
        label="Perfect Calibration",
    )

    ax1.set_xlabel("Predicted Probability", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Empirical Accuracy", fontsize=12, fontweight="bold")
    ax1.set_title("Calibration: Confidence vs Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    # plot 2: histogram of prediction confidence
    ax2 = axes[1]

    # use seaborn for better styling
    bins_hist = 20
    ax2.hist(
        predicted_prob[correct == 1],
        bins=bins_hist,
        alpha=0.7,
        label=f"Correct ({(correct == 1).sum()})",
        color=COLORS["primary"],
        edgecolor="black",
        linewidth=1,
    )
    ax2.hist(
        predicted_prob[correct == 0],
        bins=bins_hist,
        alpha=0.7,
        label=f"Incorrect ({(correct == 0).sum()})",
        color=COLORS["secondary"],
        edgecolor="black",
        linewidth=1,
    )

    ax2.set_xlabel("Predicted Probability", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax2.set_title("Prediction Confidence Distribution", fontsize=13, fontweight="bold")
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(alpha=0.3, axis="y", linestyle="--", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Residual analysis saved: {save_path}")

    return fig


def plot_team_ratings(
    params: dict[str, Any],
    top_n: int = 18,
    save_path: str | None = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plot team attack and defense ratings"""
    # create dataframe
    teams = params["teams"]
    ratings_data = []

    for team in teams:
        ratings_data.append(
            {
                "team": team,
                "attack_rating": params["attack_rating"].get(team, 0),
                "defense_rating": params["defense_rating"].get(team, 0),
                "overall_rating": params["overall_rating"].get(team, 0),
            }
        )

    df = pd.DataFrame(ratings_data).sort_values("overall_rating", ascending=False)

    # select top teams
    plot_data = df.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(plot_data))
    width = 0.35

    # create bars
    ax.bar(
        x - width / 2,
        plot_data["attack"],
        width,
        label="Attack",
        color=COLORS["primary"],
        edgecolor="black",
        linewidth=1,
    )
    ax.bar(
        x + width / 2,
        plot_data["defense"],
        width,
        label="Defence",
        color=COLORS["accent"],
        edgecolor="black",
        linewidth=1,
    )

    # labels
    ax.set_xlabel("Team", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rating", fontsize=12, fontweight="bold")
    ax.set_title("Team Attack and Defense Ratings", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data["team"], rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Team ratings plot saved: {save_path}")

    return fig


def plot_prediction_intervals(
    test_data: pd.DataFrame,
    predictions_lower: np.ndarray,
    predictions_mean: np.ndarray,
    predictions_upper: np.ndarray,
    n_matches: int = 20,
    save_path: str | None = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Plot prediction intervals for matches"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # select matches
    n_matches = min(n_matches, len(test_data))
    indices = range(n_matches)

    # actual goals
    actual_home = test_data["home_goals"].iloc[:n_matches].values
    actual_away = test_data["away_goals"].iloc[:n_matches].values

    # match labels
    match_labels = [
        f"{row['home_team'][:10]} vs {row['away_team'][:10]}"
        for _, row in test_data.iloc[:n_matches].iterrows()
    ]

    # plot home goals
    ax1.plot(
        indices,
        actual_home,
        "o",
        markersize=10,
        color=COLORS["secondary"],
        label="Actual",
        zorder=3,
        markeredgecolor="black",
        markeredgewidth=1.5,
    )
    ax1.plot(
        indices,
        predictions_mean[:n_matches, 0],
        "s",
        markersize=8,
        color=COLORS["primary"],
        label="Predicted",
        zorder=2,
        markeredgecolor="black",
        markeredgewidth=1.5,
    )
    ax1.fill_between(
        indices,
        predictions_lower[:n_matches, 0],
        predictions_upper[:n_matches, 0],
        alpha=0.3,
        color=COLORS["primary"],
        label="80% Interval",
        zorder=1,
    )

    ax1.set_ylabel("Home Goals", fontsize=12, fontweight="bold")
    ax1.set_title("Prediction Intervals - Home Goals", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    # plot away goals
    ax2.plot(
        indices,
        actual_away,
        "o",
        markersize=10,
        color=COLORS["secondary"],
        label="Actual",
        zorder=3,
        markeredgecolor="black",
        markeredgewidth=1.5,
    )
    ax2.plot(
        indices,
        predictions_mean[:n_matches, 1],
        "s",
        markersize=8,
        color=COLORS["primary"],
        label="Predicted",
        zorder=2,
        markeredgecolor="black",
        markeredgewidth=1.5,
    )
    ax2.fill_between(
        indices,
        predictions_lower[:n_matches, 1],
        predictions_upper[:n_matches, 1],
        alpha=0.3,
        color=COLORS["primary"],
        label="80% Interval",
        zorder=1,
    )

    ax2.set_ylabel("Away Goals", fontsize=12, fontweight="bold")
    ax2.set_title("Prediction Intervals - Away Goals", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Match", fontsize=12, fontweight="bold")
    ax2.set_xticks(indices)
    ax2.set_xticklabels(match_labels, rotation=45, ha="right", fontsize=9)
    ax2.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax2.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Prediction intervals plot saved: {save_path}")

    return fig
