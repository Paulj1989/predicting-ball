"""Diebold-Mariano test for equal predictive accuracy between two forecasters."""

import numpy as np
from scipy import stats


def diebold_mariano_test(
    losses_model: np.ndarray,
    losses_baseline: np.ndarray,
    alternative: str = "less",
    max_lag: int | None = None,
) -> dict:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Tests whether the model has significantly lower losses than the baseline.
    Uses Newey-West HAC standard errors to account for serial correlation in
    match predictions (form, fixtures and team strength all have persistence).

    alternative="less" tests H1: model loss < baseline loss (model is better).
    Returns p_value < 0.05 with dm_statistic < 0 if model is significantly better.
    """
    losses_model = np.asarray(losses_model, dtype=float)
    losses_baseline = np.asarray(losses_baseline, dtype=float)

    if len(losses_model) != len(losses_baseline):
        raise ValueError(
            f"Length mismatch: {len(losses_model)} model losses vs {len(losses_baseline)} baseline"
        )

    n = len(losses_model)
    if n < 10:
        raise ValueError(f"Need at least 10 observations for DM test, got {n}")

    # loss differential: negative = model is better
    d = losses_model - losses_baseline
    d_bar = np.mean(d)
    d_centered = d - d_bar

    if max_lag is None:
        max_lag = int(np.floor(np.sqrt(n)))

    # newey-west HAC variance with Bartlett kernel
    gamma_0 = np.dot(d_centered, d_centered) / n
    nw_var = gamma_0
    for k in range(1, max_lag + 1):
        # bartlett weight downweights higher lags
        w = 1.0 - k / (max_lag + 1)
        gamma_k = np.dot(d_centered[k:], d_centered[:-k]) / n
        nw_var += 2.0 * w * gamma_k

    if nw_var <= 0:
        nw_var = gamma_0  # fallback to iid variance if HAC is degenerate

    se = np.sqrt(nw_var / n)

    if se == 0:
        # all loss differentials are identical — dm statistic is 0 by definition
        dm_stat = 0.0
    else:
        dm_stat = d_bar / se

    # t-distribution is more conservative than normal for small samples
    if alternative == "less":
        p_value = float(stats.t.cdf(dm_stat, df=n - 1))
    elif alternative == "greater":
        p_value = float(stats.t.sf(dm_stat, df=n - 1))
    elif alternative == "two-sided":
        p_value = float(2.0 * stats.t.sf(abs(dm_stat), df=n - 1))
    else:
        raise ValueError(
            f"alternative must be 'less', 'greater', or 'two-sided', got {alternative!r}"
        )

    return {
        "dm_statistic": float(dm_stat),
        "p_value": p_value,
        "mean_loss_difference": float(d_bar),  # negative = model is better
        "se": float(se),
        "n": n,
        "max_lag": max_lag,
        # significant at 5% AND in the right direction
        "significant": bool(
            p_value < 0.05
            and (
                (alternative == "less" and d_bar < 0)
                or (alternative == "greater" and d_bar > 0)
                or alternative == "two-sided"
            )
        ),
    }
