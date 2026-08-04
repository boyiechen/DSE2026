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
# # Practicum 2 — executive compensation
#
# This notebook estimates the lower-truncated output distributions, solves the
# incentive-compatibility equation, fits the optimal wage schedule, decomposes
# moral-hazard costs, and evaluates the double-gamma counterfactual.

# %%
from pathlib import Path

import pandas as pd

from dse_practicum import P2FitConfig, fit_practicum2

ROOT = next(
    candidate
    for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "practicum" / "practicum2" / "data").exists()
)

DATA = ROOT / "practicum" / "practicum2" / "data"
DATA

# %% [markdown]
# ## 1. Fast point estimates
#
# `bootstrap_reps=0` returns fast analytic standard-error approximations.

# %%
quick_result = fit_practicum2(
    DATA / "simulated_moral_hazard_data.dta",
    config=P2FitConfig(n_starts=12, bootstrap_reps=0),
)
quick_result.summary()

# %%
pd.Series(quick_result.diagnostics, name="value").to_frame()

# %% [markdown]
# The double-gamma invariants provide a strong internal check:
#
# - counterfactual delta 1 / baseline delta 1 = 0.5;
# - counterfactual delta 2 / baseline delta 2 = 0.5;
# - counterfactual delta 3 − baseline delta 3 = 0.

# %%
{
    "delta_1_ratio": quick_result.diagnostics["cf_delta_1_ratio"],
    "delta_2_ratio": quick_result.diagnostics["cf_delta_2_ratio"],
    "delta_3_difference": quick_result.diagnostics[
        "cf_delta_3_difference"
    ],
}

# %% [markdown]
# ## 2. Final inference with a two-stage bootstrap
#
# Start with 25 repetitions to check runtime. Use 100–500 for the final
# submission. Each resample re-estimates both stages.

# %%
# final_result = fit_practicum2(
#     DATA / "simulated_moral_hazard_data.dta",
#     config=P2FitConfig(
#         n_starts=12,
#         bootstrap_reps=200,
#         bootstrap_starts=2,
#     ),
# )
# final_result.summary()

# %% [markdown]
# Until the bootstrap cell is run, use the quick result.

# %%
result = quick_result

# %% [markdown]
# ## 3. Create the submission file

# %%
submission = result.to_submission(
    DATA / "sample_submission.csv",
    ROOT / "practicum" / "practicum2_submission.csv",
)
submission
