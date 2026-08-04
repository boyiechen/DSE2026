"""Static Margiotta--Miller practicum estimator.

The module is deliberately written as small, callable steps so that every part
can be inspected from a notebook.  The top-level :func:`fit_practicum2`
function runs the full two-stage estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .common import fill_submission_template, require_columns


@dataclass(frozen=True)
class P2FitConfig:
    """Numerical settings for the static principal--agent estimator."""

    x_column: str = "net_excess_ret"
    wage_column: str = "totalComp"
    n_starts: int = 12
    bootstrap_reps: int = 0
    bootstrap_starts: int = 2
    random_seed: int = 20260804
    max_nfev: int = 3_000
    root_tolerance: float = 1e-11


@dataclass(frozen=True)
class TruncatedNormalFit:
    psi: float
    mu_w: float
    sigma: float
    covariance: np.ndarray
    negative_log_likelihood: float
    converged: bool


@dataclass(frozen=True)
class ContractFit:
    mu_s: float
    gamma: float
    alpha: float
    r: float
    beta: float
    eta: float
    predicted_wage: np.ndarray
    rmse: float
    transformed_parameters: np.ndarray
    transformed_covariance: np.ndarray
    converged: bool
    nfev: int


@dataclass
class Practicum2Result:
    """All estimated quantities requested by the Practicum 2 template."""

    stage1: TruncatedNormalFit
    stage2: ContractFit
    costs: dict[str, float]
    counterfactual_costs: dict[str, float]
    standard_errors: dict[str, float]
    bootstrap_estimates: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def estimates(self) -> dict[str, float]:
        """Return estimates keyed exactly like the Kaggle template."""

        values = {
            "psi": self.stage1.psi,
            "mu_w": self.stage1.mu_w,
            "sigma": self.stage1.sigma,
            "mu_s": self.stage2.mu_s,
            "gamma": self.stage2.gamma,
            "alpha": self.stage2.alpha,
            "beta": self.stage2.beta,
            **self.standard_errors,
            **self.costs,
            **self.counterfactual_costs,
        }
        return {key: float(value) for key, value in values.items()}

    def summary(self) -> pd.DataFrame:
        """A compact table suitable for display in a notebook."""

        return pd.DataFrame(
            {"estimate": pd.Series(self.estimates, dtype=float)}
        ).rename_axis("quantity")

    def to_submission(
        self,
        template_path: str | Path,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Fill and optionally save the official submission template."""

        return fill_submission_template(
            template_path,
            self.estimates,
            id_column="ID",
            value_column="ESTIMATE",
            output_path=output_path,
        )


def load_practicum2_data(
    data: str | Path | pd.DataFrame,
    *,
    x_column: str = "net_excess_ret",
    wage_column: str = "totalComp",
) -> pd.DataFrame:
    """Load the Stata file (or validate a supplied DataFrame)."""

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        path = Path(data)
        if path.suffix.lower() == ".dta":
            frame = pd.read_stata(path)
        elif path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported Practicum 2 data format: {path.suffix}")

    require_columns(frame, [x_column, wage_column], label="Practicum 2 data")
    frame = frame[[x_column, wage_column]].astype(float)
    if frame.isna().any().any():
        raise ValueError("Practicum 2 data contain missing output or compensation")
    if len(frame) < 20:
        raise ValueError("Practicum 2 estimation requires at least 20 observations")
    return frame.reset_index(drop=True)


def truncated_normal_logpdf(
    x: np.ndarray,
    mu: float,
    sigma: float,
    psi: float,
) -> np.ndarray:
    """Log density of N(mu, sigma²) conditional on X >= psi."""

    if sigma <= 0:
        return np.full_like(np.asarray(x, dtype=float), -np.inf)
    z = (np.asarray(x, dtype=float) - mu) / sigma
    log_normalizer = special.log_ndtr((mu - psi) / sigma)
    return stats.norm.logpdf(z) - np.log(sigma) - log_normalizer


def fit_truncated_work_distribution(x: np.ndarray) -> TruncatedNormalFit:
    """Estimate psi, mu_w, and sigma by lower-truncated-normal MLE."""

    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("x must be a finite one-dimensional array")

    psi = float(np.min(values))

    def objective(parameters: np.ndarray) -> float:
        mu, log_sigma = parameters
        sigma = float(np.exp(log_sigma))
        log_density = truncated_normal_logpdf(values, mu, sigma, psi)
        if not np.all(np.isfinite(log_density)):
            return 1e100
        return float(-np.sum(log_density))

    def gradient(parameters: np.ndarray) -> np.ndarray:
        mu, log_sigma = parameters
        sigma = float(np.exp(log_sigma))
        z = (values - mu) / sigma
        k = (mu - psi) / sigma
        inverse_mills = float(
            np.exp(stats.norm.logpdf(k) - special.log_ndtr(k))
        )
        score_mu = z / sigma - inverse_mills / sigma
        score_log_sigma = z**2 - 1.0 + inverse_mills * k
        return -np.array([np.sum(score_mu), np.sum(score_log_sigma)])

    initial = np.array([np.mean(values), np.log(np.std(values, ddof=1))])
    result = optimize.minimize(
        objective,
        initial,
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-5, "maxiter": 5_000},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Truncated-normal MLE failed: {result.message}")

    mu_w = float(result.x[0])
    sigma = float(np.exp(result.x[1]))
    inverse_hessian = np.asarray(result.hess_inv, dtype=float)
    jacobian = np.diag([1.0, sigma])
    covariance = jacobian @ inverse_hessian @ jacobian.T
    return TruncatedNormalFit(
        psi=psi,
        mu_w=mu_w,
        sigma=sigma,
        covariance=covariance,
        negative_log_likelihood=float(result.fun),
        converged=bool(result.success),
    )


def log_likelihood_ratio(
    x: np.ndarray,
    *,
    psi: float,
    mu_w: float,
    mu_s: float,
    sigma: float,
) -> np.ndarray:
    """Compute log(f_shirk(x) / f_work(x)) in a stable closed form."""

    values = np.asarray(x, dtype=float)
    log_normalizer_w = special.log_ndtr((mu_w - psi) / sigma)
    log_normalizer_s = special.log_ndtr((mu_s - psi) / sigma)
    quadratic_difference = (
        values * (mu_s - mu_w) + 0.5 * (mu_w**2 - mu_s**2)
    ) / sigma**2
    return log_normalizer_w - log_normalizer_s + quadratic_difference


def solve_eta(
    likelihood_ratio: np.ndarray,
    r: float,
    *,
    tolerance: float = 1e-11,
) -> float:
    """Solve the sample incentive-compatibility equation for positive eta."""

    g = np.asarray(likelihood_ratio, dtype=float)
    if r <= 1 or not np.all(np.isfinite(g)) or np.any(g <= 0):
        raise ValueError("The eta equation requires finite g > 0 and r > 1")

    q = r - g
    negative = q < 0
    if not np.any(negative):
        raise ValueError("No positive eta root: g(x) never exceeds r")

    at_zero = float(np.mean(q))
    if at_zero <= 0:
        raise ValueError("No positive eta root: sample moment is nonpositive at eta=0")

    upper_boundary = float(np.min(-1.0 / q[negative]))
    upper = upper_boundary * (1.0 - 1e-10)

    def moment(eta: float) -> float:
        denominator = 1.0 + eta * q
        return float(np.mean(q / denominator))

    if moment(upper) >= 0:
        upper = upper_boundary * (1.0 - 1e-13)
    return float(optimize.brentq(moment, 0.0, upper, xtol=tolerance, rtol=tolerance))


def optimal_wage(
    *,
    gamma: float,
    alpha: float,
    eta: float,
    r: float,
    likelihood_ratio: np.ndarray,
) -> np.ndarray:
    """Evaluate the optimal static compensation schedule."""

    q = r - np.asarray(likelihood_ratio, dtype=float)
    inside = 1.0 + eta * q
    if gamma <= 0 or alpha <= 0 or np.any(inside <= 0):
        raise ValueError("Invalid contract parameters")
    return (np.log(alpha) + np.log(inside)) / gamma


def truncated_normal_mean(*, mu: float, sigma: float, psi: float) -> float:
    """Mean of N(mu, sigma²) conditional on X >= psi."""

    standardized_cutoff = (psi - mu) / sigma
    inverse_mills = np.exp(
        stats.norm.logpdf(standardized_cutoff)
        - special.log_ndtr((mu - psi) / sigma)
    )
    return float(mu + sigma * inverse_mills)


def _decode_contract_parameters(
    transformed: np.ndarray,
    *,
    mu_w: float,
) -> tuple[float, float, float, float, float]:
    gap = float(np.exp(transformed[0]))
    gamma = float(np.exp(transformed[1]))
    alpha = float(np.exp(transformed[2]))
    r = float(1.0 + np.exp(transformed[3]))
    mu_s = float(mu_w - gap)
    beta = float(alpha / r)
    return mu_s, gamma, alpha, r, beta


def _initial_contract_parameters(
    x: np.ndarray,
    wage: np.ndarray,
    stage1: TruncatedNormalFit,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    gap = stage1.sigma * np.exp(rng.uniform(np.log(0.08), np.log(1.8)))
    gamma = np.exp(
        np.log(1.0 / max(abs(np.mean(wage)), np.std(wage), 1.0))
        + rng.uniform(-1.5, 1.5)
    )
    r = 1.0 + np.exp(rng.uniform(np.log(0.02), np.log(3.0)))
    mu_s = stage1.mu_w - gap
    log_g = log_likelihood_ratio(
        x,
        psi=stage1.psi,
        mu_w=stage1.mu_w,
        mu_s=mu_s,
        sigma=stage1.sigma,
    )
    g = np.exp(np.clip(log_g, -700, 700))
    try:
        eta = solve_eta(g, r)
        contract_component = np.mean(np.log1p(eta * (r - g)))
    except ValueError:
        contract_component = 0.0
    log_alpha = gamma * np.mean(wage) - contract_component
    return np.array(
        [np.log(gap), np.log(gamma), np.clip(log_alpha, -15, 15), np.log(r - 1)]
    )


def fit_optimal_contract(
    x: np.ndarray,
    wage: np.ndarray,
    stage1: TruncatedNormalFit,
    *,
    config: P2FitConfig,
    initial_transformed: np.ndarray | None = None,
    n_starts: int | None = None,
) -> ContractFit:
    """Fit mu_s, gamma, alpha and r by nested nonlinear least squares."""

    output = np.asarray(x, dtype=float)
    compensation = np.asarray(wage, dtype=float)
    wage_scale = float(max(np.std(compensation, ddof=1), 1.0))
    rng = np.random.default_rng(config.random_seed)
    starts = config.n_starts if n_starts is None else n_starts

    lower = np.array(
        [
            np.log(stage1.sigma * 0.003),
            np.log(1e-9),
            -20.0,
            np.log(1e-4),
        ]
    )
    upper = np.array(
        [
            np.log(stage1.sigma * 8.0),
            np.log(0.1),
            20.0,
            np.log(30.0),
        ]
    )

    def residuals(transformed: np.ndarray) -> np.ndarray:
        mu_s, gamma, alpha, r, _ = _decode_contract_parameters(
            transformed, mu_w=stage1.mu_w
        )
        log_g = log_likelihood_ratio(
            output,
            psi=stage1.psi,
            mu_w=stage1.mu_w,
            mu_s=mu_s,
            sigma=stage1.sigma,
        )
        if np.max(log_g) > 700:
            return np.full_like(compensation, 1e6)
        g = np.exp(log_g)
        try:
            eta = solve_eta(g, r, tolerance=config.root_tolerance)
            fitted = optimal_wage(
                gamma=gamma,
                alpha=alpha,
                eta=eta,
                r=r,
                likelihood_ratio=g,
            )
        except (ValueError, FloatingPointError):
            return np.full_like(compensation, 1e6)
        if not np.all(np.isfinite(fitted)):
            return np.full_like(compensation, 1e6)
        return (fitted - compensation) / wage_scale

    candidates: list[np.ndarray] = []
    if initial_transformed is not None:
        candidates.append(np.clip(np.asarray(initial_transformed), lower, upper))
    candidates.extend(
        _initial_contract_parameters(output, compensation, stage1, rng=rng)
        for _ in range(max(starts - len(candidates), 0))
    )

    fitted_results: list[optimize.OptimizeResult] = []
    for initial in candidates:
        initial = np.clip(initial, lower + 1e-8, upper - 1e-8)
        result = optimize.least_squares(
            residuals,
            initial,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=config.max_nfev,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        if np.isfinite(result.cost) and np.max(np.abs(result.fun)) < 1e5:
            fitted_results.append(result)

    if not fitted_results:
        raise RuntimeError("All optimal-contract starting values failed")
    best = min(fitted_results, key=lambda item: item.cost)

    mu_s, gamma, alpha, r, beta = _decode_contract_parameters(
        best.x, mu_w=stage1.mu_w
    )
    log_g = log_likelihood_ratio(
        output,
        psi=stage1.psi,
        mu_w=stage1.mu_w,
        mu_s=mu_s,
        sigma=stage1.sigma,
    )
    g = np.exp(log_g)
    eta = solve_eta(g, r, tolerance=config.root_tolerance)
    predicted = optimal_wage(
        gamma=gamma,
        alpha=alpha,
        eta=eta,
        r=r,
        likelihood_ratio=g,
    )

    degrees_of_freedom = max(len(output) - len(best.x), 1)
    residual_variance = float(np.sum(best.fun**2) / degrees_of_freedom)
    information = best.jac.T @ best.jac
    transformed_covariance = residual_variance * np.linalg.pinv(information)
    return ContractFit(
        mu_s=mu_s,
        gamma=gamma,
        alpha=alpha,
        r=r,
        beta=beta,
        eta=eta,
        predicted_wage=predicted,
        rmse=float(np.sqrt(np.mean((predicted - compensation) ** 2))),
        transformed_parameters=best.x.copy(),
        transformed_covariance=transformed_covariance,
        converged=bool(best.success),
        nfev=int(best.nfev),
    )


def moral_hazard_costs(
    x: np.ndarray,
    stage1: TruncatedNormalFit,
    stage2: ContractFit,
    *,
    gamma_multiplier: float = 1.0,
) -> dict[str, float]:
    """Compute the three cost measures at a counterfactual gamma."""

    gamma = stage2.gamma * gamma_multiplier
    log_g = log_likelihood_ratio(
        x,
        psi=stage1.psi,
        mu_w=stage1.mu_w,
        mu_s=stage2.mu_s,
        sigma=stage1.sigma,
    )
    g = np.exp(log_g)
    wages = optimal_wage(
        gamma=gamma,
        alpha=stage2.alpha,
        eta=stage2.eta,
        r=stage2.r,
        likelihood_ratio=g,
    )
    delta_1 = float(np.mean(wages) - np.log(stage2.alpha) / gamma)
    delta_2 = float(np.log(stage2.r) / gamma)
    delta_3 = truncated_normal_mean(
        mu=stage1.mu_w, sigma=stage1.sigma, psi=stage1.psi
    ) - truncated_normal_mean(
        mu=stage2.mu_s, sigma=stage1.sigma, psi=stage1.psi
    )
    return {"delta_1": delta_1, "delta_2": delta_2, "delta_3": float(delta_3)}


def _analytic_standard_errors(
    stage1: TruncatedNormalFit,
    stage2: ContractFit,
) -> dict[str, float]:
    stage1_variances = np.maximum(np.diag(stage1.covariance), 0)

    transformed = stage2.transformed_parameters
    gap = np.exp(transformed[0])
    r = stage2.r
    derivative = np.zeros((4, 4))
    derivative[0, 0] = -gap
    derivative[1, 1] = stage2.gamma
    derivative[2, 2] = stage2.alpha
    derivative[3, 2] = stage2.beta
    derivative[3, 3] = -stage2.beta * (r - 1.0) / r
    covariance = derivative @ stage2.transformed_covariance @ derivative.T

    # mu_s = mu_w - gap.  This adds the first-stage mu_w uncertainty but still
    # understates covariance through g(x); use the bootstrap for final results.
    covariance[0, 0] += stage1_variances[0]
    variances = np.maximum(np.diag(covariance), 0)
    return {
        "se_mu_w": float(np.sqrt(stage1_variances[0])),
        "se_sigma": float(np.sqrt(stage1_variances[1])),
        "se_mu_s": float(np.sqrt(variances[0])),
        "se_gamma": float(np.sqrt(variances[1])),
        "se_alpha": float(np.sqrt(variances[2])),
        "se_beta": float(np.sqrt(variances[3])),
    }


def _bootstrap_standard_errors(
    x: np.ndarray,
    wage: np.ndarray,
    main_stage2: ContractFit,
    *,
    config: P2FitConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    rng = np.random.default_rng(config.random_seed + 1)
    rows: list[dict[str, float]] = []
    bootstrap_config = P2FitConfig(
        **{
            **config.__dict__,
            "bootstrap_reps": 0,
            "n_starts": config.bootstrap_starts,
        }
    )
    for _ in range(config.bootstrap_reps):
        index = rng.integers(0, len(x), size=len(x))
        x_boot = x[index]
        wage_boot = wage[index]
        try:
            stage1 = fit_truncated_work_distribution(x_boot)
            stage2 = fit_optimal_contract(
                x_boot,
                wage_boot,
                stage1,
                config=bootstrap_config,
                initial_transformed=main_stage2.transformed_parameters,
                n_starts=config.bootstrap_starts,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        rows.append(
            {
                "mu_w": stage1.mu_w,
                "sigma": stage1.sigma,
                "mu_s": stage2.mu_s,
                "gamma": stage2.gamma,
                "alpha": stage2.alpha,
                "beta": stage2.beta,
            }
        )

    estimates = pd.DataFrame(rows)
    minimum_successes = max(10, int(np.ceil(config.bootstrap_reps * 0.7)))
    if len(estimates) < minimum_successes:
        warnings.warn(
            f"Only {len(estimates)} of {config.bootstrap_reps} bootstrap fits "
            "succeeded; returning analytic standard errors.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}, estimates
    standard_errors = {
        f"se_{column}": float(estimates[column].std(ddof=1))
        for column in estimates.columns
    }
    return standard_errors, estimates


def fit_practicum2(
    data: str | Path | pd.DataFrame,
    *,
    config: P2FitConfig | None = None,
) -> Practicum2Result:
    """Run the full two-stage estimator and moral-hazard decomposition."""

    settings = config or P2FitConfig()
    frame = load_practicum2_data(
        data,
        x_column=settings.x_column,
        wage_column=settings.wage_column,
    )
    x = frame[settings.x_column].to_numpy(dtype=float)
    wage = frame[settings.wage_column].to_numpy(dtype=float)

    stage1 = fit_truncated_work_distribution(x)
    stage2 = fit_optimal_contract(x, wage, stage1, config=settings)
    costs = moral_hazard_costs(x, stage1, stage2)
    counterfactual = {
        f"cf_gamma_{key}": value
        for key, value in moral_hazard_costs(
            x, stage1, stage2, gamma_multiplier=2.0
        ).items()
    }

    standard_errors = _analytic_standard_errors(stage1, stage2)
    bootstrap_estimates = pd.DataFrame()
    if settings.bootstrap_reps > 0:
        bootstrap_se, bootstrap_estimates = _bootstrap_standard_errors(
            x, wage, stage2, config=settings
        )
        if bootstrap_se:
            standard_errors = bootstrap_se

    diagnostics = {
        "n_observations": len(frame),
        "stage1_converged": stage1.converged,
        "stage2_converged": stage2.converged,
        "stage2_nfev": stage2.nfev,
        "wage_rmse": stage2.rmse,
        "bootstrap_successes": len(bootstrap_estimates),
        "cf_delta_1_ratio": counterfactual["cf_gamma_delta_1"]
        / costs["delta_1"],
        "cf_delta_2_ratio": counterfactual["cf_gamma_delta_2"]
        / costs["delta_2"],
        "cf_delta_3_difference": counterfactual["cf_gamma_delta_3"]
        - costs["delta_3"],
    }
    return Practicum2Result(
        stage1=stage1,
        stage2=stage2,
        costs=costs,
        counterfactual_costs=counterfactual,
        standard_errors=standard_errors,
        bootstrap_estimates=bootstrap_estimates,
        diagnostics=diagnostics,
    )
