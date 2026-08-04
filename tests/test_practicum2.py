import numpy as np
import pandas as pd

from dse_practicum.practicum2 import (
    P2FitConfig,
    fit_practicum2,
    log_likelihood_ratio,
    solve_eta,
    truncated_normal_logpdf,
)


def test_likelihood_ratio_matches_density_difference():
    x = np.linspace(-1.0, 2.0, 9)
    psi, mu_w, mu_s, sigma = -1.0, 0.2, -0.3, 0.8
    actual = log_likelihood_ratio(
        x, psi=psi, mu_w=mu_w, mu_s=mu_s, sigma=sigma
    )
    expected = truncated_normal_logpdf(
        x, mu_s, sigma, psi
    ) - truncated_normal_logpdf(x, mu_w, sigma, psi)
    np.testing.assert_allclose(actual, expected)


def test_eta_solves_sample_moment():
    g = np.array([2.0, 1.3, 0.5, 0.4])
    r = 1.2
    eta = solve_eta(g, r)
    q = r - g
    assert eta > 0
    assert abs(np.mean(q / (1.0 + eta * q))) < 1e-10


def test_real_practicum2_smoke():
    result = fit_practicum2(
        "practicum/practicum2/data/simulated_moral_hazard_data.dta",
        config=P2FitConfig(n_starts=4, bootstrap_reps=0),
    )
    assert len(result.estimates) == 19
    assert np.all(np.isfinite(list(result.estimates.values())))
    assert result.stage1.converged
    assert result.stage2.converged
    assert np.isclose(result.diagnostics["cf_delta_1_ratio"], 0.5)
    assert np.isclose(result.diagnostics["cf_delta_2_ratio"], 0.5)
    assert np.isclose(result.diagnostics["cf_delta_3_difference"], 0.0)

    template = pd.read_csv(
        "practicum/practicum2/data/sample_submission.csv"
    )
    assert set(template["ID"]) == set(result.estimates)
