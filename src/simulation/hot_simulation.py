# src/simulation/hot_simulation.py
"""Hot season simulation with MLE posterior draws.

Replaces bootstrap + cold simulation. Each simulation:
1. Draws initial ratings from the MLE posterior (multivariate normal)
2. Simulates matches in matchweek order with Poisson goal draws
3. Applies natural gradient rating updates after each match (when K_att or K_def > 0)
"""

import numpy as np
import pandas as pd

from .sampling import sample_scoreline_dixon_coles


def simulate_season_hot(
    future_fixtures: pd.DataFrame,
    state_mean: np.ndarray,
    state_cov: np.ndarray,
    current_standings: dict[str, dict[str, int]],
    state_teams: list[str],
    n_simulations: int = 10000,
    K_att: float = 0.05,
    K_def: float = 0.025,
    rho: float = -0.13,
    seed: int | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Simulate remaining season with MLE posterior draws and hot rating updates.

    Each simulation draws a fresh parameter set from the multivariate normal
    posterior, then simulates matches chronologically with optional natural gradient
    rating updates after each match. Separate learning rates for attack (K_att) and
    defence (K_def) reflect that the defence signal is noisier. Set both to 0 for
    a cold simulation.
    """
    if seed is not None:
        np.random.seed(seed)

    # setup team indices
    teams_in_standings = set(current_standings.keys())
    teams_in_fixtures = set(future_fixtures["home_team"].unique()) | set(
        future_fixtures["away_team"].unique()
    )
    all_teams = sorted(teams_in_standings | teams_in_fixtures)
    nt = len(all_teams)
    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    # state vector ordering: [att_0..att_n, def_0..def_n, home_adv]
    n_state_teams = len(state_teams)
    # map state vector team names to simulation team indices
    state_to_sim = {}
    for si, team in enumerate(state_teams):
        if team in team_to_idx:
            state_to_sim[si] = team_to_idx[team]

    # initialise with current standings
    base_points = np.zeros(nt)
    base_gf = np.zeros(nt)
    base_ga = np.zeros(nt)

    for team, stats in current_standings.items():
        idx = team_to_idx[team]
        base_points[idx] = stats["points"]
        base_gf[idx] = stats["goals_for"]
        base_ga[idx] = stats["goals_against"]

    # ensure covariance is positive definite
    eigvals = np.linalg.eigvalsh(state_cov)
    if eigvals.min() < 0:
        state_cov = state_cov + (-eigvals.min() + 1e-8) * np.eye(len(state_cov))

    # draw all parameter samples up front
    draws = np.random.multivariate_normal(state_mean, state_cov, size=n_simulations)

    # group fixtures by matchweek for chronological simulation
    fixtures = future_fixtures.sort_values("date").copy()

    if "matchweek" in fixtures.columns:
        mw_groups = list(fixtures.groupby("matchweek", sort=True))
    else:
        mw_groups = list(fixtures.groupby("date", sort=True))

    # extract match indices in sorted order
    home_idx = np.array([team_to_idx[t] for t in fixtures["home_team"]], dtype=int)
    away_idx = np.array([team_to_idx[t] for t in fixtures["away_team"]], dtype=int)

    # results storage
    results = {
        "points": np.zeros((n_simulations, nt)),
        "goals_for": np.zeros((n_simulations, nt)),
        "goals_against": np.zeros((n_simulations, nt)),
        "positions": np.zeros((n_simulations, nt), dtype=int),
    }

    for s in range(n_simulations):
        if s % 1000 == 0 and s > 0:
            print(f"\rSimulation {s}/{n_simulations}", end="")

        # extract this simulation's initial ratings
        draw = draws[s]
        att = draw[:n_state_teams].copy()
        dfc = draw[n_state_teams : 2 * n_state_teams].copy()
        ha = np.clip(draw[-1], 0.05, 0.5)

        # re-centre ratings (draws may shift mean slightly)
        att -= att.mean()
        dfc -= dfc.mean()

        # map state vector ratings to simulation teams by name
        # teams not in the state vector get mean (zero) ratings
        att_lookup = np.zeros(nt)
        dfc_lookup = np.zeros(nt)
        for si, sim_idx in state_to_sim.items():
            att_lookup[sim_idx] = att[si]
            dfc_lookup[sim_idx] = dfc[si]

        sim_points = base_points.copy()
        sim_gf = base_gf.copy()
        sim_ga = base_ga.copy()

        # process fixtures matchweek by matchweek
        match_offset = 0
        for _mw, mw_df in mw_groups:
            n_matches = len(mw_df)
            mw_home = home_idx[match_offset : match_offset + n_matches]
            mw_away = away_idx[match_offset : match_offset + n_matches]

            for m in range(n_matches):
                hi, ai = mw_home[m], mw_away[m]

                # expected goals from current ratings
                mu_h = np.exp(np.clip(ha + att_lookup[hi] + dfc_lookup[ai], -3, 3))
                mu_a = np.exp(np.clip(att_lookup[ai] + dfc_lookup[hi], -3, 3))

                # sample scoreline from dixon-coles joint pmf
                hg, ag = sample_scoreline_dixon_coles(mu_h, mu_a, rho=rho)

                # update standings
                sim_gf[hi] += hg
                sim_ga[hi] += ag
                sim_gf[ai] += ag
                sim_ga[ai] += hg

                if hg > ag:
                    sim_points[hi] += 3
                elif hg == ag:
                    sim_points[hi] += 1
                    sim_points[ai] += 1
                else:
                    sim_points[ai] += 3

                # hot updates: natural gradient step — (goals - lambda) / lambda is the
                # poisson score function (pearson residual), which weights updates by
                # the inverse fisher information so high-scoring matches don't dominate.
                # separate rates for attack and defence reflect noisier defence signal.
                if K_att > 0 or K_def > 0:
                    att_lookup[hi] += K_att * (hg - mu_h) / max(mu_h, 1e-6)
                    dfc_lookup[ai] += K_def * (hg - mu_h) / max(mu_h, 1e-6)
                    att_lookup[ai] += K_att * (ag - mu_a) / max(mu_a, 1e-6)
                    dfc_lookup[hi] += K_def * (ag - mu_a) / max(mu_a, 1e-6)
                    # re-centre to maintain identifiability
                    att_lookup -= att_lookup.mean()
                    dfc_lookup -= dfc_lookup.mean()

            match_offset += n_matches

        # store results
        results["points"][s] = sim_points
        results["goals_for"][s] = sim_gf
        results["goals_against"][s] = sim_ga

        # calculate positions (sort by points, then goal diff)
        goal_diff = sim_gf - sim_ga
        order = np.lexsort((-goal_diff, -sim_points))
        for pos, idx in enumerate(order):
            results["positions"][s, idx] = pos + 1

    if n_simulations >= 1000:
        print(f"\rSimulation {n_simulations}/{n_simulations}")

    return results, all_teams
