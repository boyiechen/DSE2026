import numpy as np
import pandas as pd

from dse_practicum.practicum1 import (
    P1_DATASET_SPECS,
    P1FitConfig,
    _following_quarter,
    build_theta_matrix,
    fit_practicum1,
)


def test_theta_matrix_uses_submission_term_syntax():
    frame = pd.DataFrame({"x": [2.0, 3.0], "z": [4.0, 5.0]})
    matrix = build_theta_matrix(frame, ("const", "x", "x^2", "x*z"))
    expected = np.array([[1.0, 2.0, 4.0, 8.0], [1.0, 3.0, 9.0, 15.0]])
    np.testing.assert_allclose(matrix, expected)


def test_following_quarter_handles_year_boundary():
    result = _following_quarter(np.array([19854, 19861, 19924]))
    np.testing.assert_array_equal(result, np.array([19861, 19862, 19931]))


def test_dataset_calibrations():
    assert P1_DATASET_SPECS["A"].discount_factor == 0.96
    assert P1_DATASET_SPECS["B"].discount_factor == 0.96
    assert P1_DATASET_SPECS["C"].discount_factor == 0.70


def test_real_dataset_a_smoke():
    result = fit_practicum1(
        "practicum/practicum1/data/set_A.parquet",
        spec="A",
        config=P1FitConfig(
            n_ccp_folds=2,
            ccp_max_iter=30,
            transition_polynomial_degree=1,
            gmm_max_nfev=500,
        ),
    )
    assert result.structural.converged
    assert result.structural.sigma > 0
    assert result.structural.beta == 0.96
    assert set(result.structural.theta) == set(
        P1_DATASET_SPECS["A"].theta_terms
    )

