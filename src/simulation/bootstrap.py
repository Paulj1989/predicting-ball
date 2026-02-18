# src/simulation/bootstrap.py

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parametric_bootstrap_with_residuals(
    df_train: pd.DataFrame,
    base_params: dict[str, Any],
    hyperparams: dict[str, float],
    promoted_priors: dict[str, dict[str, float]] | None = None,
    n_bootstrap: int = 500,
    use_two_stage: bool = True,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Parametric bootstrap with residual resampling.

    Quantifies parameter uncertainty by:
    1. Calculating residuals from fitted model
    2. Resampling residuals with replacement
    3. Creating synthetic data by adding resampled residuals to fitted values
    4. Refitting model on synthetic data
    5. Repeating steps 2-4 many times

    This captures uncertainty in estimated team strengths and other parameters.
    """
    # import here to avoid circular dependency
    from ..models.poisson import calculate_lambdas, fit_poisson_model_two_stage

    if verbose:
        print("\n" + "=" * 60)
        print(f"PARAMETRIC BOOTSTRAP WITH CALIBRATION ({n_bootstrap} iterations)")
        print("=" * 60)
        print(f"Fitting method: {'two-stage' if use_two_stage else 'joint'}")

    bootstrap_params = [base_params]

    # calculate base model predictions
    lambda_home_base, lambda_away_base = calculate_lambdas(df_train, base_params)
    observed_home = df_train["home_goals_weighted"].values
    observed_away = df_train["away_goals_weighted"].values

    # calculate residuals
    residuals_home = observed_home - lambda_home_base
    residuals_away = observed_away - lambda_away_base

    # bootstrap loop
    for b in range(1, n_bootstrap):
        if verbose and b % 50 == 0:
            print(f"Bootstrap iteration {b}/{n_bootstrap}")

        # resample residuals
        resampled_idx = np.random.choice(len(df_train), size=len(df_train), replace=True)

        # create synthetic data
        synthetic_home = np.maximum(0, lambda_home_base + residuals_home[resampled_idx])
        synthetic_away = np.maximum(0, lambda_away_base + residuals_away[resampled_idx])

        df_synthetic = df_train.copy()
        df_synthetic["home_goals_weighted"] = synthetic_home
        df_synthetic["away_goals_weighted"] = synthetic_away

        # use selected fitting function
        try:
            bootstrap_fit = fit_poisson_model_two_stage(
                df_synthetic,
                hyperparams,
                promoted_priors=promoted_priors,
                n_random_starts=1,
                verbose=False,
            )

            if bootstrap_fit and bootstrap_fit["success"]:
                bootstrap_params.append(bootstrap_fit)

        except Exception as e:
            if b < 10 and verbose:
                print(f"  Bootstrap iteration {b} failed: {e}")
            continue

    if verbose:
        print(f"\nCompleted {len(bootstrap_params)}/{n_bootstrap} bootstrap iterations")

    if len(bootstrap_params) > 1:
        # calculate parameter uncertainty
        home_advs = [p["home_adv"] for p in bootstrap_params]
        blend_weights = [p.get("odds_blend_weight", 1.0) for p in bootstrap_params]
        beta_form = [p.get("beta_form", 0.0) for p in bootstrap_params]

        if verbose:
            print("\nParameter uncertainty (mean ± std):")
            print(f"  Home advantage: {np.mean(home_advs):.3f} ± {np.std(home_advs):.3f}")
            print(f"  Form weight: {np.mean(beta_form):.3f} ± {np.std(beta_form):.3f}")
            print(f"  Odds blend: {np.mean(blend_weights):.3f} ± {np.std(blend_weights):.3f}")
            disp = base_params.get("dispersion_factor", 1.0)
            disp_flag = " ⚠ (> 1.2)" if disp > 1.2 else ""
            print(f"  Dispersion (diagnostic): {disp:.3f}{disp_flag}")

    return bootstrap_params


def plot_parameter_diagnostics(
    bootstrap_params: list[dict[str, Any]],
    base_params: dict[str, Any],
    save_path: str | None = None,
) -> None:
    """Plot parameter distributions from bootstrap"""
    print("\n" + "=" * 60)
    print("PARAMETER DIAGNOSTICS")
    print("=" * 60)

    # extract parameters
    param_dict = {
        "home_adv": [p["home_adv"] for p in bootstrap_params],
        "odds_blend_weight": [p.get("odds_blend_weight", 1.0) for p in bootstrap_params],
        "beta_form": [p.get("beta_form", 0.0) for p in bootstrap_params],
    }

    params_df = pd.DataFrame(param_dict)

    print("\nParameter statistics:")
    print(params_df.describe())

    # check for boundary issues
    print("\nBoundary checks:")
    bounds_dict = {
        "home_adv": (0.05, 0.5),
        "odds_blend_weight": (0.0, 1.0),
        "beta_form": (-0.5, 0.5),
    }

    for param, (lower, upper) in bounds_dict.items():
        at_lower = (params_df[param] <= lower + 0.01).sum()
        at_upper = (params_df[param] >= upper - 0.01).sum()

        if at_lower > len(bootstrap_params) * 0.05:
            print(f"  ✗ {param}: {at_lower} samples at lower bound")
        if at_upper > len(bootstrap_params) * 0.05:
            print(f"  ✗ {param}: {at_upper} samples at upper bound")

    # create plots
    n_params = len(param_dict)
    _fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5))
    if n_params == 1:
        axes = [axes]

    for ax, (param, values) in zip(axes, param_dict.items(), strict=False):
        ax.hist(values, bins=30, alpha=0.6, edgecolor="black", color="steelblue")

        # mark base model value
        base_value = base_params.get(param, 0)
        ax.axvline(
            base_value,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Base: {base_value:.3f}",
        )

        # mark bounds
        if param in bounds_dict:
            lower, upper = bounds_dict[param]
            ax.axvline(lower, color="gray", linestyle=":", alpha=0.5, label="Bounds")
            ax.axvline(upper, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel(param.replace("_", " ").title())
        ax.set_ylabel("Frequency")
        ax.set_title(f"{param} Distribution")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Parameter diagnostic plot saved: {save_path}")
    else:
        plt.savefig("outputs/figures/parameter_diagnostics.png", dpi=150, bbox_inches="tight")
        print("\n✓ Parameter diagnostic plot saved: outputs/figures/parameter_diagnostics.png")

    plt.close()
