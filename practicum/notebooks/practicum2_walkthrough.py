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
# # Practicum 2 - executive compensation
#
# This notebook estimates the lower-truncated output distributions, solves the
# incentive-compatibility equation, fits the optimal wage schedule, decomposes
# moral-hazard costs, and evaluates the double-gamma counterfactual.
#
# The economic foundation is **Margiotta and Miller (2000), "Managerial
# Compensation and the Cost of Moral Hazard."** The original paper has multiple
# periods, industries, and executives. The practicum is a one-period,
# one-manager adaptation, so this notebook preserves the paper's core
# likelihood-ratio contract and two-stage estimation logic while removing the
# original panel heterogeneity.
#
# Three notation cautions:
#
# - `net_excess_ret` is the practicum's output \(x\);
# - `totalComp` is the observed wage \(w\);
# - the practicum's `gamma` is the CARA/risk-aversion scale in the static wage
#   formula. Its `beta = alpha/r` is a preference parameter, not the subjective
#   time-discount factor denoted by beta elsewhere in the dynamic paper.
#
# Local paper:
# [Margiotta and Miller (2000)](../practicum2/ref/managerCompCostMoralHaz2000.pdf).

# %%
from pathlib import Path

import pandas as pd

from dse_practicum import P2FitConfig, fit_practicum2

# Find the repository regardless of whether Jupyter starts at the repository
# root or inside practicum/notebooks.
ROOT = next(
    candidate
    for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "practicum" / "practicum2" / "data").exists()
)

DATA = ROOT / "practicum" / "practicum2" / "data"
DATA

# %% [markdown]
# ## 1. How the code maps to the paper
#
# ### Stage 1: output distribution under diligent work
#
# Margiotta and Miller Section 9.1, pp. 699-702, assumes abnormal returns have
# a normal parent distribution truncated below at \(\psi\). Equation (22)
# gives the density, Equation (23) gives the likelihood ratio
# \(g(x)=f_s(x)/f_w(x)\), Equation (27) estimates the lower support point by the
# sample minimum, and Equation (28) estimates the remaining distribution
# parameters by maximum likelihood.
#
# The practicum has one pooled sector and treats observed output as coming from
# diligent work. The code therefore sets
#
# \[
# \hat{\psi}=\min_i x_i
# \]
#
# and estimates \((\mu_w,\sigma)\) by lower-truncated-normal MLE. The shirking
# location \(\mu_s\) is estimated in Stage 2 because no shirked output sample is
# observed.
#
# ### Stage 2: incentive-compatible contract
#
# Proposition 4 in Section 5, p. 682, defines a unique positive multiplier
# \(\eta\) in Equation (20) and inserts it into the optimal contract in
# Equation (21). In the static practicum normalization this becomes
#
# \[
# w^*(x)=\frac{1}{\gamma}
# \left\{\log\alpha+\log[1+\eta(r-g(x))]\right\},
# \qquad r=\alpha/\beta>1.
# \]
#
# For every trial \((\mu_s,\gamma,\alpha,r)\), the code recomputes \(g(x)\),
# solves the sample incentive-compatibility equation for positive \(\eta\), and
# only then evaluates the wage residuals. This nesting is directly analogous to
# Section 9.2, especially Equation (30), p. 704, where the paper's GMM
# criterion must be evaluated subject to Equation (20).
#
# The outer criterion differs: the paper uses GMM moments from participation,
# incentive compatibility, and compensation equations, while the practicum
# supplies only output and compensation and asks for a static fit. The code
# therefore uses nonlinear least squares for the wage equation. It preserves
# the paper's inner economic restriction but adapts the outer estimator to the
# available data.

# %% [markdown]
# ## 2. Fast point estimates
#
# `bootstrap_reps=0` returns fast analytic standard-error approximations. Use
# this pass to check parameter magnitudes and optimizer diagnostics before
# starting a much more expensive bootstrap.

# %%
quick_result = fit_practicum2(
    DATA / "simulated_moral_hazard_data.dta",
    config=P2FitConfig(
        # The nested root makes the outer objective nonconvex. Multiple starts
        # reduce dependence on one arbitrary initial parameter vector.
        n_starts=12,
        # Zero keeps this first pass fast. Section 4 below explains final
        # inference that propagates both estimation stages.
        bootstrap_reps=0,
    ),
)
quick_result.summary()

# %%
# These diagnostics expose both optimization status and economic identities.
# A low wage RMSE alone is not sufficient if either estimation stage failed.
pd.Series(quick_result.diagnostics, name="value").to_frame()

# %% [markdown]
# ## 3. Moral-hazard cost decomposition
#
# Section 7, pp. 685-687, defines three conceptually different costs:
#
# 1. \(\Delta_1\): the expected compensation premium caused by unobservable
#    effort - the shareholders' value of perfect monitoring for the same task;
# 2. \(\Delta_2\): the compensating differential associated with diligent work
#    rather than shirking;
# 3. \(\Delta_3\): the expected gross output loss from replacing the diligent
#    output distribution with the shirking distribution.
#
# The practicum's static normalization evaluates these as
#
# \[
# \Delta_1=E_f[w^*(X)]-\frac{\log\alpha}{\gamma},\qquad
# \Delta_2=\frac{\log(\alpha/\beta)}{\gamma},\qquad
# \Delta_3=E_w[X]-E_s[X].
# \]
#
# The two expectations in \(\Delta_3\) use truncated-normal means, not the
# parent-normal difference \(\mu_w-\mu_s\). This distinction follows from the
# truncation correction in Section 9.1, Equation (25), p. 701.
#
# Doubling gamma is a practicum-specific counterfactual rather than an exercise
# reported in the original paper. Holding technology and preference ratios
# fixed makes the wage-based terms homogeneous of degree -1 in gamma, while
# the output-distribution term does not depend on gamma. The following
# invariants are therefore a strong implementation check:
#
# - counterfactual delta 1 / baseline delta 1 = 0.5;
# - counterfactual delta 2 / baseline delta 2 = 0.5;
# - counterfactual delta 3 - baseline delta 3 = 0.

# %%
{
    "delta_1_ratio": quick_result.diagnostics["cf_delta_1_ratio"],
    "delta_2_ratio": quick_result.diagnostics["cf_delta_2_ratio"],
    "delta_3_difference": quick_result.diagnostics[
        "cf_delta_3_difference"
    ],
}

# %% [markdown]
# ## 4. Final inference with a two-stage bootstrap
#
# Section 9.2, pp. 704-705, derives an asymptotic covariance matrix that accounts
# for the pre-estimated output distribution when estimating the remaining
# parameters. The quick analytic standard errors in this code are conditional
# approximations and do not fully propagate every Stage-1 channel.
#
# A pairs bootstrap provides a transparent practicum analogue: resample
# \((x_i,w_i)\) pairs and re-estimate \(\psi,\mu_w,\sigma,\mu_s,\gamma,\alpha\),
# and beta in every draw. Start with 25 repetitions to check runtime. Use
# 100-500 for final inference.

# %%
# final_result = fit_practicum2(
#     DATA / "simulated_moral_hazard_data.dta",
#     config=P2FitConfig(
#         # Use the well-searched point estimate as the main fit.
#         n_starts=12,
#         # Each bootstrap draw repeats both the truncated-normal MLE and
#         # nested contract fit.
#         bootstrap_reps=200,
#         # Bootstrap draws start near the main estimate, so fewer additional
#         # random starts are normally sufficient.
#         bootstrap_starts=2,
#     ),
# )
# final_result.summary()

# %% [markdown]
# Until the bootstrap cell is run, use the quick result. After running it,
# replace this assignment with `result = final_result`.

# %%
result = quick_result

# %% [markdown]
# ## 5. Create the submission file
#
# This final step is a competition schema check rather than a procedure from
# Margiotta and Miller. It preserves the official row order and rejects missing,
# extra, or non-finite estimates.

# %%
# The template validator prevents, for example, confusing the practicum's beta
# preference parameter with a discount factor or swapping baseline and
# counterfactual cost rows.
submission = result.to_submission(
    DATA / "sample_submission.csv",
    ROOT / "practicum" / "practicum2_submission.csv",
)
submission

# %% [markdown]
# ## Paper cross-reference
#
# | Notebook component | Margiotta and Miller (2000) |
# |---|---|
# | Positive eta and the optimal contract | Section 5, Proposition 4, Equations (20)-(21), pp. 682-683 |
# | Three moral-hazard cost concepts | Section 7, pp. 685-687 |
# | Truncated-normal density and likelihood ratio | Section 9.1, Equations (22)-(23), pp. 699-700 |
# | Truncated mean and support estimator | Section 9.1, Equations (25)-(27), p. 701 |
# | First-stage maximum likelihood | Section 9.1, Equation (28), pp. 701-702 |
# | Remaining parameters and nested estimation | Section 9.2, Equations (29)-(30), pp. 702-704 |
# | Generated-regressor inference | Section 9.2, pp. 704-705 |
#
# **Reference:** Margiotta, M. M. and Miller, R. A. (2000), "Managerial
# Compensation and the Cost of Moral Hazard," *International Economic Review*
# 41(3), 669-719.
