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
# # Practicum 1 - dynamic bank closure
#
# This notebook estimates monetary closure cost, cross-fitted closure CCPs,
# conditional continuation values, and the structural parameters requested by
# the submission template.
#
# The economic foundation is **Kang, Lowery, and Wardlaw (2015), "The Costs of
# Closing Failed Banks."** The most important connection is their Section 1.2,
# especially Equations (4)-(9), pp. 1067-1069. A terminal closure action and
# Type-I extreme-value choice shocks imply
#
# \[
# \sigma\log\frac{p_0(x_t)}{p_1(x_t)}
# =
# c(x_t)+\beta E\left[
# -\sigma\log p_1(x_{t+1})-c(x_{t+1})
# \mid x_t,\text{continue}
# \right].
# \]
#
# This equation avoids solving the complete dynamic program. Estimated choice
# probabilities stand in for differences in continuation values, which is the
# central Hotz-Miller/CCP insight used by the paper.
#
# Important implementation conventions:
#
# - the practicum reports \(c(x)=MC(x)-\tilde{x}'\theta\), whereas Equation (7)
#   in the paper writes the nonmonetary component with the opposite sign;
# - A and B use calibrated \(\beta=0.96\), close to the paper's early-period
#   estimate in Section 3.4; C uses \(\beta=0.70\), selected from the simulated
#   data's GMM profile;
# - beta is displayed for transparency but is not a submission row.
#
# Local paper:
# [Kang et al. (2015)](../practicum1/ref/Kang-CostsClosingFailed-2015.pdf).

# %%
from pathlib import Path

import pandas as pd

from dse_practicum import (
    P1CounterfactualConfig,
    P1FitConfig,
    fit_practicum1,
    fit_practicum1_all,
)

# Locate the repository from either the repository root or the notebook folder.
# This makes the same notebook work in JupyterLab, VS Code, and an executed
# notebook without hard-coding an absolute path.
ROOT = next(
    candidate
    for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "practicum" / "practicum1" / "data").exists()
)

DATA = ROOT / "practicum" / "practicum1" / "data"
DATA

# %% [markdown]
# ## 1. How the code maps to the paper
#
# `fit_practicum1` follows the five-stage estimator listed in Kang et al.
# Section 1.2, p. 1068, with two deliberate adaptations to the simulated data:
#
# 1. **Pre-estimate monetary closure cost.** Section 2.2, pp. 1075-1078,
#    estimates monetary cost before the structural parameters. The paper uses a
#    censored regression because it augments failures with zero-cost unassisted
#    mergers. The practicum files contain observed cost only for failed banks
#    and contain no merger observations, so the code uses OLS on observed
#    `estimated_cost` rows. This preserves the paper's pre-estimation logic
#    without inventing the missing censoring observations.
# 2. **Estimate closure CCPs.** Section 2.4, pp. 1081-1083, uses a flexible
#    logit with cubic B-splines. The code uses a flexible histogram gradient
#    boosting classifier for the same reduced-form object \(p_1(x)\). It
#    cross-fits by `rssd_id`, so no observations from a held-out bank are used
#    to predict that bank's probabilities.
# 3. **Estimate state transitions and continuation objects.** Section 2.3,
#    pp. 1079-1081, estimates autoregressive transitions and simulates future
#    states. Because Equation (8) is linear in
#    \(E[\log p_1(x')]\), \(E[MC(x')]\), and \(E[\tilde{x}']\), the code
#    projects these objects directly on a second-degree polynomial of the
#    current state using ridge regression. This is a conditional-mean
#    approximation to the paper's transition-plus-simulation step.
# 4. **Estimate structural parameters by GMM.** Equation (8) supplies the
#    residual and Equation (9) supplies the moments. Appendix A.1.1, p. 1097,
#    recommends instruments that enter payoffs, monetary costs, or transitions.
#    The code builds those instruments from the current state and applies
#    two-step GMM. The paper uses continuously updated GMM, so the weighting
#    update is an approximation, not an exact replication.
#
# These distinctions matter: "paper-inspired" does not mean that the simulated
# practicum contains every variable required for a literal replication.

# %% [markdown]
# ## 2. Fit one dataset first
#
# Fitting A first is the recommended debugging workflow. Explicit numerical
# settings below make the computation visible instead of hiding it behind
# defaults.

# %%
config = P1FitConfig(
    # Three folds balance computation against genuine out-of-bank CCP checks.
    n_ccp_folds=3,
    # A flexible first-stage CCP is important because log(p0/p1) enters the
    # structural equation nonlinearly.
    ccp_max_iter=200,
    # Degree two allows nonlinear conditional means while keeping the
    # continuation projection much smaller than a full transition simulator.
    transition_polynomial_degree=2,
)

# This one function executes the monetary-cost, CCP, continuation, and GMM
# stages described above. Start with one dataset so diagnostics can be checked
# before paying the cost of estimating A, B, and C.
result_a = fit_practicum1(
    DATA / "set_A.parquet",
    spec="A",
    config=config,
)
result_a.summary()

# %%
pd.Series(result_a.diagnostics, name="value").to_frame()

# %% [markdown]
# ### Why inspect these diagnostics?
#
# The structural step treats monetary cost, CCPs, and continuation expectations
# as generated regressors. A numerical optimizer can converge even when those
# inputs are poor, so optimizer convergence alone is not enough.
#
# - Monetary-cost \(R^2\) checks the first pre-estimation stage in Section 2.2.
# - CCP AUC/log loss and the min/max probability check the flexible choice
#   model in Section 2.4. Extremely small probabilities are especially
#   dangerous because the structural equation contains their logarithms.
# - Continuation \(R^2\) checks the conditional expectations replacing the
#   simulations in Section 2.3.
# - The GMM convergence flag and \(J\)-statistic assess the final moment system
#   discussed in Equation (9) and Appendix A.1.1.

# %%
{
    "structural_converged": result_a.structural.converged,
    # sigma is the scale of the Type-I extreme-value choice shock.
    "sigma": result_a.structural.sigma,
    # beta is calibrated for this practicum dataset and is not submitted.
    "calibrated_beta": result_a.structural.beta,
    "ccp_auc": result_a.ccp.diagnostics["roc_auc"],
    "gmm_j": result_a.structural.j_statistic,
}

# %% [markdown]
# ## 3. Sensitivity to beta
#
# Kang et al. Section 1.2.2 explains that beta is identified only when some
# variables shift transitions without directly shifting payoffs. Section 3.4
# reports an early-period quarterly estimate around 0.96. Because this
# practicum asks for sigma and theta but not beta, the main specification
# calibrates beta and treats this cell as a sensitivity exercise.
#
# `estimate_beta=True` is available, but the simulated panel provides weak
# standalone identification and the estimate may approach its upper bound.
# That boundary behavior is diagnostic evidence, not a value to submit.

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
# ## 4. Fit A, B, and C
#
# The sample-submission file supplies a different \(\tilde{x}\) specification
# for each dataset. Passing `spec="A"`, `"B"`, or `"C"` makes the code construct
# exactly those nonmonetary-cost terms, including the squared and interaction
# terms. This corresponds to changing the covariates in Equation (7), not
# searching over specifications.

# %%
# Reuse the same first-stage and GMM settings across datasets. Dataset-specific
# theta terms and beta calibrations are selected internally from the label.
all_results = fit_practicum1_all(DATA, config=config)
all_results.summary()

# %% [markdown]
# ## 5. Provisional counterfactuals
#
# Kang et al. Section 4, pp. 1093-1096, evaluates **temporary** policy changes.
# Appendix A.2, pp. 1098-1100, shows how a one-period counterfactual can retain
# the estimated future CCP as the post-policy continuation value and then map a
# changed current cost or discount factor into a counterfactual closure
# probability. The provisional code below uses that one-period CCP logic.
#
# It is not a literal replication of the paper's policy experiments. The local
# practicum instruction names three scenarios but does not define their shock
# magnitude, horizon, or reported statistic. The cell must therefore be
# consciously enabled and currently assumes:
#
# - `no_political`: set `house` and `senate` to zero;
# - `myopic`: set beta to zero;
# - `npl_stress`: add one sample standard deviation to `npf_a`;
# - report the one-period mean closure-probability level.
#
# Replace these assumptions when the organizer supplies exact definitions.

# %%
provisional_cf = P1CounterfactualConfig(
    # The API refuses to compute undocumented counterfactuals unless this
    # acknowledgement is explicit.
    acknowledge_provisional=True,
    npl_shift_in_standard_deviations=1.0,
    report="level",
)

# Re-estimate all three datasets so every result object carries the same
# counterfactual assumptions and can fill the complete submission template.
all_results_with_cf = fit_practicum1_all(
    DATA,
    config=config,
    counterfactual_config=provisional_cf,
)
all_results_with_cf.summary()

# %% [markdown]
# ## 6. Create the submission file
#
# This uses the official template as the schema and refuses missing, extra, or
# non-finite rows. This is a competition-data validation step rather than a
# method taken from Kang et al.

# %%
# Filling the template by row name prevents squared/interaction terms from
# silently moving to the wrong dataset or sign convention.
submission = all_results_with_cf.to_submission(
    DATA / "sample_submission.csv",
    ROOT / "practicum" / "practicum1_submission.csv",
)
submission

# %% [markdown]
# ## Paper cross-reference
#
# | Notebook component | Kang, Lowery, and Wardlaw (2015) |
# |---|---|
# | Terminal closure and CCP inversion | Section 1.2, Equations (4)-(6), pp. 1067-1068 |
# | Monetary/nonmonetary cost split | Section 1.2, Equation (7), p. 1068 |
# | Five-stage estimator and structural residual | Section 1.2, Equations (8)-(9), pp. 1068-1069 |
# | Monetary-cost pre-estimation | Section 2.2, pp. 1075-1078 |
# | Transition process and future expectations | Section 2.3, pp. 1079-1081 |
# | Flexible closure CCP | Section 2.4, pp. 1081-1083 |
# | Instrument choice and GMM | Appendix A.1.1, p. 1097 |
# | Temporary-policy counterfactuals | Section 4 and Appendix A.2, pp. 1093-1100 |
#
# **Reference:** Kang, K., Lowery, R., and Wardlaw, M. (2015), "The Costs of
# Closing Failed Banks: A Structural Estimation of Regulatory Incentives,"
# *Review of Financial Studies* 28(4), 1060-1102.
