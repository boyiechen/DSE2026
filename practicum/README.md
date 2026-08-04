# DSE 2026 practica: Python implementation

The estimators live in `src/dse_practicum` and are designed to be called one
stage at a time from Jupyter.

## Start Jupyter

From the repository root:

```bash
uv sync --group dev
uv run jupyter lab
```

Choose the repository's `.venv` kernel and open either notebook in
`practicum/notebooks/`.

## Practicum 1 assumptions

- Monetary closure cost is estimated by OLS on rows where `estimated_cost` is
  observed. The simulated files do not contain the zero-cost merger
  observations needed for the paper's Tobit specification.
- Closure CCPs use bank-grouped cross-fitting.
- Conditional expectations are estimated with a polynomial ridge transition
  regression before structural GMM.
- The discount factor is calibrated rather than submitted. A and B use 0.96,
  the paper's baseline quarterly value. C uses 0.70; this is the value selected
  by its GMM profile and the near-integer parameter design in the simulated
  data. Both values can be overridden with `P1FitConfig(fixed_beta=...)`.
- The three named counterfactuals are not defined in the local competition
  instructions. The provided routine is therefore opt-in and explicitly
  provisional. It reports one-period mean closure probabilities under:
  political variables set to zero, beta set to zero, and NPL increased by one
  sample standard deviation.

Do not treat the provisional counterfactual rows as final competition answers
until the organizer confirms the shock size, horizon, and reported statistic.

## Practicum 2 assumptions

- `net_excess_ret` is output `x`; `totalComp` is compensation `w`.
- `psi` is the sample minimum.
- `mu_w` and `sigma` are estimated by lower-truncated-normal MLE.
- The remaining contract parameters are fitted by nested nonlinear least
  squares, solving the sample incentive-compatibility equation for `eta` at
  every trial value.
- Analytic standard errors are returned by default. Set `bootstrap_reps` to at
  least 100 for final results that propagate both estimation stages.

