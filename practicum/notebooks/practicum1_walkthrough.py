# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Practicum 1 — dynamic bank closure
#
# This notebook estimates monetary closure cost, cross-fitted closure CCPs,
# conditional continuation values, and the structural parameters requested by
# the submission template.
#
# A/B use calibrated beta=0.96. C uses beta=0.70. Beta is displayed for
# transparency but is not a submission row.

# %%
from pathlib import Path

import pandas as pd

from dse_practicum import (
    P1CounterfactualConfig,
    P1FitConfig,
    fit_practicum1,
    fit_practicum1_all,
)

ROOT = next(
    candidate
    for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "practicum" / "practicum1" / "data").exists()
)

DATA = ROOT / "practicum" / "practicum1" / "data"
DATA

# %% [markdown]
# ## 1. Fit one dataset first
#
# This is the recommended debugging workflow. The full settings use three CCP
# folds and 200 boosting iterations.

# %%
config = P1FitConfig()

result_a = fit_practicum1(
    DATA / "set_A.parquet",
    spec="A",
    config=config,
)
result_a.summary()

# %%
pd.Series(result_a.diagnostics, name="value").to_frame()

# %% [markdown]
# Useful diagnostics:
#
# - monetary-cost R-squared should be reasonably high;
# - CCP probabilities must not be concentrated at the clipping bounds;
# - continuation R-squared values show whether the one-period transition
#   approximation is working;
# - the structural fit must converge.

# %%
{
    "structural_converged": result_a.structural.converged,
    "sigma": result_a.structural.sigma,
    "calibrated_beta": result_a.structural.beta,
    "ccp_auc": result_a.ccp.diagnostics["roc_auc"],
    "gmm_j": result_a.structural.j_statistic,
}

# %% [markdown]
# ## 2. Sensitivity to beta
#
# This does not change the official default. It is useful for checking the
# calibration. `estimate_beta=True` estimates beta as a nuisance parameter,
# but it is weakly identified and may approach its upper bound.

# %%
# sensitivity_a = fit_practicum1(
#     DATA / "set_A.parquet",
#     spec="A",
#     config=P1FitConfig(fixed_beta=0.95),
# )
# sensitivity_a.summary()

# %%
# weak_id_check = fit_practicum1(
#     DATA / "set_A.parquet",
#     spec="A",
#     config=P1FitConfig(estimate_beta=True),
# )
# weak_id_check.summary()

# %% [markdown]
# ## 3. Fit A, B, and C

# %%
all_results = fit_practicum1_all(DATA, config=config)
all_results.summary()

# %% [markdown]
# ## 4. Provisional counterfactuals
#
# The local instruction does not define the shock magnitude, horizon, or
# reported statistic. The following cell must therefore be consciously enabled.
# Its assumptions are:
#
# - `no_political`: set `house` and `senate` to zero;
# - `myopic`: set beta to zero;
# - `npl_stress`: add one sample standard deviation to `npf_a`;
# - report the one-period mean closure-probability level.
#
# Replace this block when the organizer supplies the exact definitions.

# %%
provisional_cf = P1CounterfactualConfig(
    acknowledge_provisional=True,
    npl_shift_in_standard_deviations=1.0,
    report="level",
)

all_results_with_cf = fit_practicum1_all(
    DATA,
    config=config,
    counterfactual_config=provisional_cf,
)
all_results_with_cf.summary()

# %% [markdown]
# ## 5. Create the submission file
#
# This uses the official template as the schema and refuses missing, extra, or
# non-finite rows.

# %%
submission = all_results_with_cf.to_submission(
    DATA / "sample_submission.csv",
    ROOT / "practicum" / "practicum1_submission.csv",
)
submission
