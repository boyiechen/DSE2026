"""Dynamic bank-closure estimator for Practicum 1.

The first implementation uses observed next-quarter states as unbiased draws
from the transition distribution.  This avoids imposing a large transition
system during initial estimation while preserving the conditional moment
restriction.  A separate, explicitly provisional one-step counterfactual
routine is included because the local competition instructions do not define
the requested counterfactual shocks or reported statistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, special
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .common import (
    fill_submission_template,
    require_columns,
    stable_logit,
    symmetric_matrix_sqrt,
)


@dataclass(frozen=True)
class P1DatasetSpec:
    """Nonmonetary-cost specification supplied by the competition."""

    label: str
    filename: str
    theta_terms: tuple[str, ...]
    discount_factor: float


P1_DATASET_SPECS: dict[str, P1DatasetSpec] = {
    "A": P1DatasetSpec(
        label="A",
        filename="set_A.parquet",
        theta_terms=(
            "const",
            "log_assets",
            "log_assets^2",
            "npf_a",
            "roa",
            "realest_a",
            "house",
            "house*npf_a",
        ),
        discount_factor=0.96,
    ),
    "B": P1DatasetSpec(
        label="B",
        filename="set_B.parquet",
        theta_terms=(
            "const",
            "log_assets",
            "log_assets^2",
            "npf_a",
            "roa",
            "realest_a",
            "house",
            "house*log_assets",
        ),
        discount_factor=0.96,
    ),
    "C": P1DatasetSpec(
        label="C",
        filename="set_C.parquet",
        theta_terms=(
            "const",
            "log_assets",
            "log_assets^2",
            "npf_a",
            "roa",
            "realest_a",
            "senate",
            "trend",
        ),
        discount_factor=0.70,
    ),
}


@dataclass(frozen=True)
class P1FitConfig:
    """Settings for the first-stage estimators and structural GMM."""

    monetary_features: tuple[str, ...] = (
        "log_assets",
        "equity_a",
        "npf_a",
        "realest_a",
        "roa",
        "unemp",
    )
    ccp_candidate_features: tuple[str, ...] = (
        "equity_a",
        "npf_a",
        "roa",
        "log_assets",
        "realest_a",
        "unemp",
        "house",
        "senate",
        "trend",
        "bank_age",
        "n_branches",
        "yearq",
    )
    n_ccp_folds: int = 3
    ccp_max_iter: int = 200
    ccp_learning_rate: float = 0.05
    ccp_max_leaf_nodes: int = 31
    ccp_min_samples_leaf: int = 75
    transition_polynomial_degree: int = 2
    transition_ridge_alpha: float = 1.0
    probability_clip: float = 1e-5
    # The practicum asks for sigma and theta, not beta.  None uses the
    # dataset-specific calibration in P1DatasetSpec (A/B=.96, C=.70).
    fixed_beta: float | None = None
    estimate_beta: bool = False
    beta_min: float = 0.01
    beta_max: float = 0.999
    gmm_max_nfev: int = 3_000
    gmm_weight_ridge: float = 1e-7
    random_seed: int = 20260804


@dataclass(frozen=True)
class P1CounterfactualConfig:
    """Explicit assumptions for the otherwise undefined competition shocks.

    These defaults produce one-period mean closure-probability levels.  They
    are engineering placeholders, not claims about the hidden competition
    definition.
    """

    acknowledge_provisional: bool = False
    npl_shift_in_standard_deviations: float = 1.0
    report: str = "level"  # "level" or "change"


@dataclass
class MonetaryCostFit:
    model: LinearRegression
    features: tuple[str, ...]
    n_observed_costs: int
    rmse: float
    r_squared: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        require_columns(frame, list(self.features), label="monetary-cost data")
        return self.model.predict(frame.loc[:, list(self.features)]).astype(float)


@dataclass
class CCPFit:
    model: HistGradientBoostingClassifier
    features: tuple[str, ...]
    out_of_fold_probability: np.ndarray
    diagnostics: dict[str, float]
    clip: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        require_columns(frame, list(self.features), label="CCP data")
        probability = self.model.predict_proba(
            frame.loc[:, list(self.features)]
        )[:, 1]
        return np.clip(probability, self.clip, 1.0 - self.clip)


@dataclass(frozen=True)
class StructuralFit:
    sigma: float
    beta: float
    theta: dict[str, float]
    n_observations: int
    n_moments: int
    j_statistic: float
    residual_rmse: float
    converged: bool
    moment_vector: np.ndarray


@dataclass
class ContinuationFit:
    """Conditional expectations entering the one-period CCP equation."""

    model: Any
    features: tuple[str, ...]
    target_r_squared: dict[str, float]


@dataclass
class Practicum1Result:
    spec: P1DatasetSpec
    monetary_cost: MonetaryCostFit
    ccp: CCPFit
    continuation: ContinuationFit
    structural: StructuralFit
    counterfactuals: dict[str, float] | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def estimates(self) -> dict[str, float]:
        prefix = self.spec.label
        values: dict[str, float] = {f"{prefix}_sigma": self.structural.sigma}
        values.update(
            {
                f"{prefix}_theta_{term}": self.structural.theta[term]
                for term in self.spec.theta_terms
            }
        )
        if self.counterfactuals is not None:
            values.update(
                {
                    f"{prefix}_cf_{name}": value
                    for name, value in self.counterfactuals.items()
                }
            )
        return values

    def summary(self) -> pd.DataFrame:
        values = {
            "sigma": self.structural.sigma,
            "beta (calibration; not submitted)": self.structural.beta,
            **{
                f"theta_{term}": value
                for term, value in self.structural.theta.items()
            },
        }
        if self.counterfactuals:
            values.update(
                {f"cf_{key}": value for key, value in self.counterfactuals.items()}
            )
        return pd.DataFrame({"estimate": pd.Series(values)}).rename_axis("quantity")


@dataclass
class Practicum1AllResults:
    results: dict[str, Practicum1Result]

    @property
    def estimates(self) -> dict[str, float]:
        combined: dict[str, float] = {}
        for label in ("A", "B", "C"):
            combined.update(self.results[label].estimates)
        return combined

    def summary(self) -> pd.DataFrame:
        frames = []
        for label, result in self.results.items():
            table = result.summary().reset_index()
            table.insert(0, "dataset", label)
            frames.append(table)
        return pd.concat(frames, ignore_index=True)

    def to_submission(
        self,
        template_path: str | Path,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        return fill_submission_template(
            template_path,
            self.estimates,
            id_column="Id",
            value_column="Prediction",
            output_path=output_path,
        )


@dataclass(frozen=True)
class _StructuralSample:
    mc_current: np.ndarray
    mc_next: np.ndarray
    x_current_raw: np.ndarray
    x_next_raw: np.ndarray
    x_current_standardized: np.ndarray
    x_next_standardized: np.ndarray
    x_means: np.ndarray
    x_scales: np.ndarray
    log_continue_close_ratio: np.ndarray
    log_close_next: np.ndarray
    instruments: np.ndarray
    cost_scale: float
    n_observations: int
    continuation_fit: ContinuationFit


def get_p1_spec(label: str) -> P1DatasetSpec:
    try:
        return P1_DATASET_SPECS[label.upper()]
    except KeyError as error:
        raise ValueError("Practicum 1 dataset label must be A, B, or C") from error


def load_practicum1_data(data: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load and validate a bank-quarter panel."""

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        path = Path(data)
        if path.suffix.lower() != ".parquet":
            raise ValueError("Practicum 1 data must be a parquet file")
        frame = pd.read_parquet(path)

    required = ["rssd_id", "yearq", "failed", "estimated_cost"]
    require_columns(frame, required, label="Practicum 1 data")
    if frame.duplicated(["rssd_id", "yearq"]).any():
        raise ValueError("Duplicate bank-quarter observations found")
    if not set(frame["failed"].dropna().unique()).issubset({0, 1}):
        raise ValueError("failed must be coded as zero or one")
    return frame.sort_values(["rssd_id", "yearq"]).reset_index(drop=True)


def build_theta_matrix(
    frame: pd.DataFrame,
    terms: tuple[str, ...],
) -> np.ndarray:
    """Evaluate the simple term syntax used by the submission template."""

    columns: list[np.ndarray] = []
    for term in terms:
        if term == "const":
            values = np.ones(len(frame), dtype=float)
        elif term.endswith("^2"):
            source = term[:-2]
            require_columns(frame, [source], label="theta specification")
            values = frame[source].to_numpy(dtype=float) ** 2
        elif "*" in term:
            left, right = term.split("*", maxsplit=1)
            require_columns(frame, [left, right], label="theta specification")
            values = (
                frame[left].to_numpy(dtype=float)
                * frame[right].to_numpy(dtype=float)
            )
        else:
            require_columns(frame, [term], label="theta specification")
            values = frame[term].to_numpy(dtype=float)
        columns.append(values)
    matrix = np.column_stack(columns)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Non-finite values in nonmonetary-cost regressors")
    return matrix


def fit_monetary_cost(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
) -> MonetaryCostFit:
    """Fit the simulated monetary closure-cost equation on observed costs."""

    require_columns(frame, [*features, "estimated_cost"], label="monetary-cost data")
    observed = frame["estimated_cost"].notna()
    if observed.sum() <= len(features):
        raise ValueError("Too few observed closure costs to fit the monetary model")

    model = LinearRegression()
    model.fit(
        frame.loc[observed, list(features)],
        frame.loc[observed, "estimated_cost"],
    )
    fitted = model.predict(frame.loc[observed, list(features)])
    actual = frame.loc[observed, "estimated_cost"].to_numpy(dtype=float)
    return MonetaryCostFit(
        model=model,
        features=features,
        n_observed_costs=int(observed.sum()),
        rmse=float(np.sqrt(mean_squared_error(actual, fitted))),
        r_squared=float(r2_score(actual, fitted)),
    )


def _new_ccp_model(config: P1FitConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=config.ccp_learning_rate,
        max_iter=config.ccp_max_iter,
        max_leaf_nodes=config.ccp_max_leaf_nodes,
        min_samples_leaf=config.ccp_min_samples_leaf,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=config.random_seed,
    )


def fit_ccp(frame: pd.DataFrame, *, config: P1FitConfig) -> CCPFit:
    """Estimate cross-fitted closure probabilities, grouping by bank."""

    features = tuple(
        column for column in config.ccp_candidate_features if column in frame.columns
    )
    if not features:
        raise ValueError("No candidate CCP features are present")
    require_columns(frame, ["rssd_id", "failed", *features], label="CCP data")
    x = frame.loc[:, list(features)]
    y = frame["failed"].to_numpy(dtype=int)
    groups = frame["rssd_id"].to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=config.n_ccp_folds,
        shuffle=True,
        random_state=config.random_seed,
    )
    probability = np.empty(len(frame), dtype=float)
    for train_index, test_index in splitter.split(x, y, groups):
        model = _new_ccp_model(config)
        model.fit(x.iloc[train_index], y[train_index])
        probability[test_index] = model.predict_proba(x.iloc[test_index])[:, 1]

    probability = np.clip(
        probability, config.probability_clip, 1.0 - config.probability_clip
    )
    final_model = _new_ccp_model(config)
    final_model.fit(x, y)
    diagnostics = {
        "event_rate": float(np.mean(y)),
        "log_loss": float(log_loss(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "probability_min": float(np.min(probability)),
        "probability_max": float(np.max(probability)),
    }
    return CCPFit(
        model=final_model,
        features=features,
        out_of_fold_probability=probability,
        diagnostics=diagnostics,
        clip=config.probability_clip,
    )


def _following_quarter(yearq: np.ndarray) -> np.ndarray:
    values = np.asarray(yearq, dtype=int)
    year = values // 10
    quarter = values % 10
    if np.any((quarter < 1) | (quarter > 4)):
        raise ValueError("yearq must use the YYYYQ integer convention")
    return np.where(quarter < 4, values + 1, (year + 1) * 10 + 1)


def _standardized_instruments(
    frame: pd.DataFrame,
    row_index: np.ndarray,
    x_standardized: np.ndarray,
    terms: tuple[str, ...],
    *,
    ccp_features: tuple[str, ...],
    log_ratio: np.ndarray,
    mc_current: np.ndarray,
) -> np.ndarray:
    named_columns: dict[str, np.ndarray] = {}

    for position, term in enumerate(terms):
        if term != "const":
            named_columns[f"x:{term}"] = x_standardized[:, position]
    for column in ccp_features:
        if column not in named_columns:
            named_columns[f"state:{column}"] = frame.loc[row_index, column].to_numpy(
                dtype=float
            )
    named_columns["log_continue_close_ratio"] = log_ratio
    named_columns["predicted_monetary_cost"] = mc_current

    standardized: list[np.ndarray] = []
    for values in named_columns.values():
        values = np.asarray(values, dtype=float)
        scale = float(np.std(values))
        if not np.isfinite(scale) or scale < 1e-10:
            continue
        z = (values - np.mean(values)) / scale
        standardized.extend([z, z**2 - np.mean(z**2)])

    instruments = np.column_stack([np.ones(len(row_index)), *standardized])
    if not np.all(np.isfinite(instruments)):
        raise ValueError("Non-finite GMM instruments")
    return instruments


def _build_structural_sample(
    frame: pd.DataFrame,
    spec: P1DatasetSpec,
    monetary_cost: MonetaryCostFit,
    ccp: CCPFit,
    *,
    config: P1FitConfig,
) -> _StructuralSample:
    work = frame.copy()
    work["_mc_hat"] = monetary_cost.predict(work)
    work["_p_close"] = ccp.out_of_fold_probability

    next_bank = work.groupby("rssd_id", sort=False)["rssd_id"].shift(-1)
    next_yearq = work.groupby("rssd_id", sort=False)["yearq"].shift(-1)
    expected_yearq = _following_quarter(work["yearq"].to_numpy())
    valid = (
        (work["failed"].to_numpy() == 0)
        & next_bank.notna().to_numpy()
        & (next_yearq.to_numpy() == expected_yearq)
    )
    row_index = np.flatnonzero(valid)
    next_index = row_index + 1
    if len(row_index) < 100:
        raise ValueError("Too few continued bank-quarters with an observed next state")

    x_raw_all = build_theta_matrix(work, spec.theta_terms)
    x_current_raw = x_raw_all[row_index]
    x_next_raw = x_raw_all[next_index]
    x_means = np.zeros(x_current_raw.shape[1])
    x_scales = np.ones(x_current_raw.shape[1])
    for column, term in enumerate(spec.theta_terms):
        if term == "const":
            continue
        x_means[column] = np.mean(x_current_raw[:, column])
        x_scales[column] = max(np.std(x_current_raw[:, column]), 1e-10)
    x_current_standardized = (x_current_raw - x_means) / x_scales
    x_next_standardized = (x_next_raw - x_means) / x_scales

    p_current = work["_p_close"].to_numpy(dtype=float)[row_index]
    p_next = work["_p_close"].to_numpy(dtype=float)[next_index]
    log_ratio = -stable_logit(p_current, clip=config.probability_clip)
    realized_log_close_next = np.log(
        np.clip(p_next, config.probability_clip, 1.0 - config.probability_clip)
    )
    mc = work["_mc_hat"].to_numpy(dtype=float)
    mc_current = mc[row_index]
    realized_mc_next = mc[next_index]

    transition_features = ccp.features
    transition_x = work.loc[row_index, list(transition_features)]
    transition_targets = np.column_stack(
        [
            realized_log_close_next,
            realized_mc_next,
            x_next_standardized,
        ]
    )
    transition_model = make_pipeline(
        PolynomialFeatures(
            degree=config.transition_polynomial_degree,
            include_bias=False,
        ),
        StandardScaler(),
        Ridge(alpha=config.transition_ridge_alpha),
    )
    transition_model.fit(transition_x, transition_targets)
    expected_targets = transition_model.predict(transition_x)
    log_close_next = expected_targets[:, 0]
    mc_next = expected_targets[:, 1]
    expected_x_next = expected_targets[:, 2:]
    if spec.theta_terms[0] == "const":
        expected_x_next[:, 0] = 1.0

    target_names = (
        "log_close_probability",
        "monetary_cost",
        *[f"x:{term}" for term in spec.theta_terms],
    )
    target_r_squared = {
        name: float(r2_score(transition_targets[:, index], expected_targets[:, index]))
        if np.std(transition_targets[:, index]) > 1e-12
        else 1.0
        for index, name in enumerate(target_names)
    }
    continuation_fit = ContinuationFit(
        model=transition_model,
        features=transition_features,
        target_r_squared=target_r_squared,
    )
    cost_scale = float(
        max(np.median(np.abs(mc_current)), np.std(mc_current), 1.0)
    )
    instruments = _standardized_instruments(
        work,
        row_index,
        x_current_standardized,
        spec.theta_terms,
        ccp_features=ccp.features,
        log_ratio=log_ratio,
        mc_current=mc_current,
    )
    return _StructuralSample(
        mc_current=mc_current,
        mc_next=mc_next,
        x_current_raw=x_current_raw,
        x_next_raw=x_next_raw,
        x_current_standardized=x_current_standardized,
        x_next_standardized=expected_x_next,
        x_means=x_means,
        x_scales=x_scales,
        log_continue_close_ratio=log_ratio,
        log_close_next=log_close_next,
        instruments=instruments,
        cost_scale=cost_scale,
        n_observations=len(row_index),
        continuation_fit=continuation_fit,
    )


def _encode_beta(beta: float, config: P1FitConfig) -> float:
    fraction = (beta - config.beta_min) / (config.beta_max - config.beta_min)
    fraction = float(np.clip(fraction, 1e-8, 1.0 - 1e-8))
    return float(np.log(fraction) - np.log1p(-fraction))


def _decode_structural_parameters(
    parameters: np.ndarray,
    sample: _StructuralSample,
    config: P1FitConfig,
) -> tuple[float, float, np.ndarray]:
    sigma = float(sample.cost_scale * np.exp(parameters[0]))
    if config.fixed_beta is None:
        beta_fraction = float(special.expit(parameters[1]))
        beta = config.beta_min + (
            config.beta_max - config.beta_min
        ) * beta_fraction
        phi = sample.cost_scale * parameters[2:]
    else:
        beta = float(config.fixed_beta)
        phi = sample.cost_scale * parameters[1:]
    return sigma, beta, phi


def _structural_residuals(
    parameters: np.ndarray,
    sample: _StructuralSample,
    config: P1FitConfig,
) -> np.ndarray:
    sigma, beta, phi = _decode_structural_parameters(parameters, sample, config)
    residual = (
        sigma * sample.log_continue_close_ratio
        - sample.mc_current
        + sample.x_current_standardized @ phi
        + beta
        * (
            sigma * sample.log_close_next
            + sample.mc_next
            - sample.x_next_standardized @ phi
        )
    )
    return residual / sample.cost_scale


def _initial_gmm_parameters(
    sample: _StructuralSample,
    config: P1FitConfig,
) -> list[np.ndarray]:
    starts: list[np.ndarray] = []
    if config.fixed_beta is None:
        beta_grid = [0.25, 0.5, 0.75, 0.9, 0.96, 0.99, 0.995]
        beta_grid = [
            beta
            for beta in beta_grid
            if config.beta_min < beta < config.beta_max
        ]
    else:
        beta_grid = [config.fixed_beta]
    for beta in beta_grid:
        design = np.column_stack(
            [
                sample.log_continue_close_ratio
                + beta * sample.log_close_next,
                sample.x_current_standardized
                - beta * sample.x_next_standardized,
            ]
        )
        target = sample.mc_current - beta * sample.mc_next
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        sigma = float(coefficients[0])
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = sample.cost_scale * 0.25
        phi = coefficients[1:]
        starts.append(
            np.concatenate(
                (
                    [np.log(sigma / sample.cost_scale)],
                    (
                        [_encode_beta(beta, config)]
                        if config.fixed_beta is None
                        else []
                    ),
                    phi / sample.cost_scale,
                )
            )
        )
    return starts


def fit_structural_gmm(
    sample: _StructuralSample,
    spec: P1DatasetSpec,
    *,
    config: P1FitConfig,
) -> StructuralFit:
    """Estimate sigma, beta and theta using two-step GMM."""

    z = sample.instruments

    def moment(parameters: np.ndarray) -> np.ndarray:
        residual = _structural_residuals(parameters, sample, config)
        return np.mean(z * residual[:, None], axis=0)

    if config.fixed_beta is not None and not 0 <= config.fixed_beta < 1:
        raise ValueError("fixed_beta must lie in [0, 1)")
    n_leading_parameters = 2 if config.fixed_beta is None else 1
    n_parameters = n_leading_parameters + len(spec.theta_terms)
    if config.fixed_beta is None:
        lower_leading = [-10.0, -14.0]
        upper_leading = [10.0, 14.0]
    else:
        lower_leading = [-10.0]
        upper_leading = [10.0]
    lower = np.concatenate(
        [lower_leading, np.full(n_parameters - n_leading_parameters, -1e4)]
    )
    upper = np.concatenate(
        [upper_leading, np.full(n_parameters - n_leading_parameters, 1e4)]
    )

    first_stage_fits: list[optimize.OptimizeResult] = []
    for initial in _initial_gmm_parameters(sample, config):
        result = optimize.least_squares(
            moment,
            np.clip(initial, lower + 1e-9, upper - 1e-9),
            bounds=(lower, upper),
            x_scale="jac",
            max_nfev=config.gmm_max_nfev,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        if np.isfinite(result.cost):
            first_stage_fits.append(result)
    if not first_stage_fits:
        raise RuntimeError("All first-step GMM starting values failed")
    first = min(first_stage_fits, key=lambda item: item.cost)

    first_residual = _structural_residuals(first.x, sample, config)
    observation_moments = z * first_residual[:, None]
    centered = observation_moments - observation_moments.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    ridge = config.gmm_weight_ridge * max(
        float(np.trace(covariance) / len(covariance)), 1.0
    )
    weighting = np.linalg.pinv(covariance + ridge * np.eye(len(covariance)))
    weighting_sqrt = symmetric_matrix_sqrt(weighting)

    def weighted_moment(parameters: np.ndarray) -> np.ndarray:
        return weighting_sqrt @ moment(parameters)

    second = optimize.least_squares(
        weighted_moment,
        first.x,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=config.gmm_max_nfev,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    sigma, beta, phi = _decode_structural_parameters(second.x, sample, config)

    theta_array = phi / sample.x_scales
    theta_array[0] = phi[0] - np.sum(
        sample.x_means[1:] * theta_array[1:]
    )
    theta = {
        term: float(theta_array[position])
        for position, term in enumerate(spec.theta_terms)
    }
    final_moment = moment(second.x)
    residual = _structural_residuals(second.x, sample, config) * sample.cost_scale
    j_statistic = float(
        sample.n_observations * final_moment @ weighting @ final_moment
    )
    return StructuralFit(
        sigma=sigma,
        beta=beta,
        theta=theta,
        n_observations=sample.n_observations,
        n_moments=z.shape[1],
        j_statistic=j_statistic,
        residual_rmse=float(np.sqrt(np.mean(residual**2))),
        converged=bool(second.success),
        moment_vector=final_moment,
    )


def _theta_cost(
    frame: pd.DataFrame,
    spec: P1DatasetSpec,
    theta: Mapping[str, float],
    monetary_cost: MonetaryCostFit,
) -> np.ndarray:
    coefficients = np.asarray([theta[term] for term in spec.theta_terms])
    return monetary_cost.predict(frame) - build_theta_matrix(
        frame, spec.theta_terms
    ) @ coefficients


def _counterfactual_probability(
    current_cost: np.ndarray,
    next_cost: np.ndarray,
    p_close_next: np.ndarray,
    *,
    sigma: float,
    beta: float,
    probability_clip: float,
) -> np.ndarray:
    continuation_value = (
        -sigma
        * np.log(
            np.clip(
                p_close_next, probability_clip, 1.0 - probability_clip
            )
        )
        - next_cost
    )
    log_continue_close = (current_cost + beta * continuation_value) / sigma
    return special.expit(-log_continue_close)


def compute_provisional_counterfactuals(
    frame: pd.DataFrame,
    result: Practicum1Result,
    *,
    config: P1CounterfactualConfig,
    fit_config: P1FitConfig,
) -> dict[str, float]:
    """Compute transparent one-period placeholders for the three named shocks."""

    if not config.acknowledge_provisional:
        raise ValueError(
            "The local instructions do not define the Practicum 1 "
            "counterfactuals. Set acknowledge_provisional=True only after "
            "reviewing the assumptions in P1CounterfactualConfig."
        )
    if config.report not in {"level", "change"}:
        raise ValueError("counterfactual report must be 'level' or 'change'")
    warnings.warn(
        "Practicum 1 counterfactual definitions are provisional: reporting "
        "one-period mean closure probabilities.",
        RuntimeWarning,
        stacklevel=2,
    )

    work = frame.sort_values(["rssd_id", "yearq"]).reset_index(drop=True).copy()
    work["_p_close"] = result.ccp.out_of_fold_probability
    next_bank = work.groupby("rssd_id", sort=False)["rssd_id"].shift(-1)
    next_yearq = work.groupby("rssd_id", sort=False)["yearq"].shift(-1)
    valid = (
        (work["failed"].to_numpy() == 0)
        & next_bank.notna().to_numpy()
        & (
            next_yearq.to_numpy()
            == _following_quarter(work["yearq"].to_numpy())
        )
    )
    current_index = np.flatnonzero(valid)
    next_index = current_index + 1
    p_next = work["_p_close"].to_numpy()[next_index]

    def probabilities(
        altered: pd.DataFrame,
        *,
        beta: float,
    ) -> np.ndarray:
        costs = _theta_cost(
            altered,
            result.spec,
            result.structural.theta,
            result.monetary_cost,
        )
        return _counterfactual_probability(
            costs[current_index],
            costs[next_index],
            p_next,
            sigma=result.structural.sigma,
            beta=beta,
            probability_clip=fit_config.probability_clip,
        )

    baseline = probabilities(work, beta=result.structural.beta)

    no_political_frame = work.copy()
    for column in ("house", "senate"):
        if column in no_political_frame:
            no_political_frame[column] = 0.0
    no_political = probabilities(
        no_political_frame, beta=result.structural.beta
    )
    myopic = probabilities(work, beta=0.0)

    stressed = work.copy()
    npl_shift = (
        config.npl_shift_in_standard_deviations
        * float(work["npf_a"].std(ddof=0))
    )
    stressed["npf_a"] = np.maximum(stressed["npf_a"] + npl_shift, 0.0)
    npl_stress = probabilities(stressed, beta=result.structural.beta)

    def aggregate(values: np.ndarray) -> float:
        if config.report == "level":
            return float(np.mean(values))
        return float(np.mean(values) - np.mean(baseline))

    return {
        "no_political": aggregate(no_political),
        "myopic": aggregate(myopic),
        "npl_stress": aggregate(npl_stress),
    }


def fit_practicum1(
    data: str | Path | pd.DataFrame,
    *,
    spec: str | P1DatasetSpec,
    config: P1FitConfig | None = None,
    counterfactual_config: P1CounterfactualConfig | None = None,
) -> Practicum1Result:
    """Run all estimable stages for one of datasets A, B, or C."""

    requested_settings = config or P1FitConfig()
    dataset_spec = get_p1_spec(spec) if isinstance(spec, str) else spec
    if requested_settings.estimate_beta:
        settings = P1FitConfig(
            **{**requested_settings.__dict__, "fixed_beta": None}
        )
    else:
        calibrated_beta = (
            dataset_spec.discount_factor
            if requested_settings.fixed_beta is None
            else requested_settings.fixed_beta
        )
        settings = P1FitConfig(
            **{**requested_settings.__dict__, "fixed_beta": calibrated_beta}
        )
    frame = load_practicum1_data(data)
    require_columns(
        frame,
        [
            *settings.monetary_features,
            *[
                term.replace("^2", "")
                for term in dataset_spec.theta_terms
                if term != "const" and "*" not in term
            ],
        ],
        label=f"Practicum 1 dataset {dataset_spec.label}",
    )

    monetary = fit_monetary_cost(
        frame, features=settings.monetary_features
    )
    ccp = fit_ccp(frame, config=settings)
    sample = _build_structural_sample(
        frame, dataset_spec, monetary, ccp, config=settings
    )
    structural = fit_structural_gmm(sample, dataset_spec, config=settings)
    result = Practicum1Result(
        spec=dataset_spec,
        monetary_cost=monetary,
        ccp=ccp,
        continuation=sample.continuation_fit,
        structural=structural,
        counterfactuals=None,
        diagnostics={
            "n_rows": len(frame),
            "n_banks": int(frame["rssd_id"].nunique()),
            "n_failures": int(frame["failed"].sum()),
            "monetary_cost_rmse": monetary.rmse,
            "monetary_cost_r_squared": monetary.r_squared,
            **{f"ccp_{key}": value for key, value in ccp.diagnostics.items()},
            **{
                f"continuation_r2_{key}": value
                for key, value in sample.continuation_fit.target_r_squared.items()
            },
            "gmm_j_statistic": structural.j_statistic,
            "gmm_n_moments": structural.n_moments,
            "gmm_n_observations": structural.n_observations,
        },
    )
    if counterfactual_config is not None:
        result.counterfactuals = compute_provisional_counterfactuals(
            frame,
            result,
            config=counterfactual_config,
            fit_config=settings,
        )
    return result


def fit_practicum1_all(
    data_directory: str | Path,
    *,
    config: P1FitConfig | None = None,
    counterfactual_config: P1CounterfactualConfig | None = None,
) -> Practicum1AllResults:
    """Estimate A, B, and C using the official specifications."""

    directory = Path(data_directory)
    results = {
        label: fit_practicum1(
            directory / spec.filename,
            spec=spec,
            config=config,
            counterfactual_config=counterfactual_config,
        )
        for label, spec in P1_DATASET_SPECS.items()
    }
    return Practicum1AllResults(results=results)
