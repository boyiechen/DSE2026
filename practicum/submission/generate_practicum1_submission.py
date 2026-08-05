#!/usr/bin/env python3
"""Estimate Practicum 1 on S1/S2/S3 and write the official Kaggle submission.

Run this file from anywhere inside the repository.  It uses the official
S1/S2/S3 parquet panels, the specifications described in the student
notebook, and the structural-moment/counterfactual utilities in ``klw.py``.

Following the TA's correction, ``submission_template.csv`` is the authoritative
51-row template.  Its IDs and row order drive the output.  The obsolete
36-row A/B/C ``sample_submission.csv`` is deliberately never used.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, ndtri
from scipy.stats import qmc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PRACTICUM1_DIR = REPO_ROOT / "practicum" / "practicum1"
DEFAULT_DATA_DIR = PRACTICUM1_DIR / "data"
DEFAULT_OUTPUT = SCRIPT_DIR / "practicum1_submission.csv"
EXPECTED_ROWS = 51
SIZE_CENTER = 12.0
DEFAULT_COUNTERFACTUAL_SEEDS = (0, 1, 7, 42)

# Import the official toolkit by absolute repository path so the script does
# not depend on the caller's current working directory.
sys.path.insert(0, str(PRACTICUM1_DIR / "code"))
from klw import (  # noqa: E402
    CCP_BASIS,
    FIN_VARS,
    MC_SPEC_PAPER,
    PAPER_SPEC,
    Result,
    SCENARIOS,
    build_design,
    counterfactual,
    expected_design,
    gmm_fit,
    monetary_cost,
    next_state_arr,
    state_bounds,
)


BETA = 0.9582
NMC_SPECS = {
    "S1": list(PAPER_SPEC),
    "S2": list(PAPER_SPEC),
    "S3": [*PAPER_SPEC, "trend"],
}
MC_SPECS = {
    "S1": [*MC_SPEC_PAPER, "unemp"],
    "S2": [*MC_SPEC_PAPER, "unemp", "npf_a^2"],
    "S3": [*MC_SPEC_PAPER, "unemp", "npf_a^2"],
}
CCP_SPECS = {
    "S1": list(CCP_BASIS),
    # The fixed-point documentation in klw.py notes that the richer cost
    # function requires npf_a^2 in the CCP projection basis.
    "S2": [*CCP_BASIS, "npf_a^2"],
    "S3": [*CCP_BASIS, "npf_a^2"],
}


@dataclass(frozen=True)
class PanelDiagnostics:
    label: str
    n_rows: int
    n_failures: int
    ccp_mean: float
    ccp_log_likelihood: float
    expectation_draws: int
    gmm_j: float
    gmm_pvalue: float
    gmm_dof: int
    gmm_condition: float
    sigma_equity_only: float
    sigma_unemp_only: float
    political_change_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the official Practicum 1 S1/S2/S3 Kaggle CSV."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing set_S1/S2/S3.parquet (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Official 51-row template (default: <data-dir>/submission_template.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--expectation-draws",
        type=int,
        default=512,
        help="Scrambled Sobol draws for E[log p(close next)|x] (default: 512)",
    )
    parser.add_argument(
        "--counterfactual-draws",
        type=int,
        default=512,
        help="Monte Carlo draws per S3 counterfactual fixed point (default: 512)",
    )
    parser.add_argument(
        "--counterfactual-anchor",
        type=int,
        default=8_000,
        help="S3 states used in each fixed-point projection (default: 8000)",
    )
    parser.add_argument(
        "--counterfactual-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_COUNTERFACTUAL_SEEDS),
        help="Seeds averaged for final S3 counterfactuals (default: 0 1 7 42)",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _require_files(data_dir: Path, template_path: Path) -> None:
    required = [
        data_dir / "set_S1.parquet",
        data_dir / "set_S2.parquet",
        data_dir / "set_S3.parquet",
        template_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Practicum 1 file(s):\n" + "\n".join(missing))


def _canonical_ids() -> set[str]:
    identifiers: set[str] = set()
    for label in ("S1", "S2", "S3"):
        identifiers.add(f"{label}_sigma")
        identifiers.update(f"{label}_theta_{term}" for term in NMC_SPECS[label])
        identifiers.update(f"{label}_mc_{term}" for term in MC_SPECS[label])
    identifiers.update(f"S3_cf_{scenario}" for scenario in SCENARIOS)
    return identifiers


def _read_template(path: Path) -> pd.DataFrame:
    template = pd.read_csv(path)
    if list(template.columns) != ["Id", "Prediction"]:
        raise ValueError(
            f"{path} has columns {list(template.columns)}; expected ['Id', 'Prediction']"
        )
    if len(template) != EXPECTED_ROWS:
        raise ValueError(
            f"{path} has {len(template)} rows; the official S1/S2/S3 template "
            f"must have {EXPECTED_ROWS}."
        )
    if template["Id"].duplicated().any():
        duplicates = template.loc[template["Id"].duplicated(), "Id"].tolist()
        raise ValueError(f"{path} contains duplicate submission IDs: {duplicates}")
    template_ids = set(template["Id"])
    canonical_ids = _canonical_ids()
    if template_ids != canonical_ids:
        missing = sorted(canonical_ids.difference(template_ids))
        extra = sorted(template_ids.difference(canonical_ids))
        raise ValueError(
            "Template IDs do not match the official S1/S2/S3 specification; "
            f"missing={missing}, extra={extra}"
        )
    return template


def _load_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(path).sort_values(["rssd_id", "yearq"]).reset_index(drop=True)
    required = {
        "rssd_id",
        "yearq",
        "failed",
        "estimated_cost",
        *FIN_VARS,
        "unemp",
        "house",
        "senate",
        "trend",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if panel.duplicated(["rssd_id", "yearq"]).any():
        raise ValueError(f"{path.name} contains duplicate bank-quarter rows")
    if not set(panel["failed"].unique()).issubset({0, 1}):
        raise ValueError(f"{path.name}: failed must be coded 0/1")
    return panel


def _fit_monetary_cost(panel: pd.DataFrame, spec: list[str]) -> dict[str, float]:
    resolved = panel.loc[panel["failed"].eq(1) & panel["estimated_cost"].notna()]
    if len(resolved) <= len(spec):
        raise ValueError("Too few observed resolution costs for monetary-cost OLS")
    design = build_design(resolved, spec)
    coefficients, *_ = np.linalg.lstsq(
        design, resolved["estimated_cost"].to_numpy(float), rcond=None
    )
    result = dict(zip(spec, coefficients, strict=True))
    if not np.all(np.isfinite(list(result.values()))):
        raise RuntimeError("Monetary-cost OLS returned a non-finite coefficient")
    return {key: float(value) for key, value in result.items()}


def _fit_pooled_monetary_cost(
    panels: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Fit the cumulative S1/S2/S3 monetary-cost design jointly."""

    resolved_frames: list[pd.DataFrame] = []
    for label, panel in panels.items():
        resolved = panel.loc[
            panel["failed"].eq(1) & panel["estimated_cost"].notna()
        ].copy()
        resolved["_rich_npf2"] = (
            resolved["npf_a"].pow(2) if label in {"S2", "S3"} else 0.0
        )
        resolved_frames.append(resolved)
    resolved = pd.concat(resolved_frames, ignore_index=True)
    pooled_spec = [*MC_SPECS["S1"], "_rich_npf2"]
    design = np.column_stack(
        [
            np.ones(len(resolved)),
            *[
                resolved[term].to_numpy(float)
                for term in pooled_spec
                if term != "const"
            ],
        ]
    )
    outcome = resolved["estimated_cost"].to_numpy(float)
    coefficient, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = outcome - design @ coefficient
    r2 = 1.0 - float(
        residual @ residual
        / ((outcome - outcome.mean()) @ (outcome - outcome.mean()))
    )
    rmse = float(np.sqrt(np.mean(residual**2)))
    shared = dict(zip(pooled_spec, coefficient, strict=True))

    result: dict[str, dict[str, float]] = {}
    for label in ("S1", "S2", "S3"):
        values = {
            term: float(shared[term])
            for term in MC_SPECS[label]
            if term != "npf_a^2"
        }
        if "npf_a^2" in MC_SPECS[label]:
            values["npf_a^2"] = float(shared["_rich_npf2"])
        result[label] = values
    return result, r2, rmse


def _fit_transitions(panel: pd.DataFrame) -> dict[str, object]:
    """Fit continuous AR(1)s and a joint House/Senate Markov chain."""

    transitions: dict[str, object] = {}
    for variable in FIN_VARS:
        lag = panel.groupby("rssd_id", sort=False)[variable].shift(1)
        valid = lag.notna().to_numpy()
        design = np.column_stack([np.ones(valid.sum()), lag.to_numpy()[valid]])
        target = panel[variable].to_numpy(float)[valid]
        coefficient, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficient
        transitions[variable] = (
            float(coefficient[0]),
            float(coefficient[1]),
            float(np.var(residual)),
        )

    unemployment = panel.groupby("yearq")["unemp"].mean().sort_index().to_numpy(float)
    design = np.column_stack([np.ones(len(unemployment) - 1), unemployment[:-1]])
    coefficient, *_ = np.linalg.lstsq(design, unemployment[1:], rcond=None)
    residual = unemployment[1:] - design @ coefficient
    transitions["unemp"] = (
        float(coefficient[0]),
        float(coefficient[1]),
        float(np.var(residual)),
    )

    house_lag = panel.groupby("rssd_id", sort=False)["house"].shift(1)
    senate_lag = panel.groupby("rssd_id", sort=False)["senate"].shift(1)
    valid = house_lag.notna() & senate_lag.notna()
    origin = (
        house_lag[valid].to_numpy(int) * 3
        + senate_lag[valid].to_numpy(int)
    )
    destination = (
        panel.loc[valid, "house"].to_numpy(int) * 3
        + panel.loc[valid, "senate"].to_numpy(int)
    )
    counts = np.full((12, 12), 0.5)
    np.add.at(counts, (origin, destination), 1.0)
    transitions["political_joint"] = counts / counts.sum(axis=1, keepdims=True)
    return transitions


def _fit_ccp(
    panel: pd.DataFrame, basis: list[str]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit the flexible reduced-form closure CCP and return raw-basis coefficients."""

    # Standardizing nonconstant columns makes the unpenalized logit stable
    # despite including log_assets, log_assets^2, and log_assets^3 together.
    raw_nonconstant = build_design(panel, basis[1:])
    scaler = StandardScaler()
    standardized = scaler.fit_transform(raw_nonconstant)
    model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=5_000,
        tol=1e-10,
    )
    outcome = panel["failed"].to_numpy(int)
    model.fit(standardized, outcome)
    if model.n_iter_[0] >= model.max_iter:
        raise RuntimeError("The closure-CCP logit reached max_iter without converging")

    scaled_coefficient = model.coef_[0]
    raw_coefficient = scaled_coefficient / scaler.scale_
    raw_intercept = float(
        model.intercept_[0] - np.sum(scaled_coefficient * scaler.mean_ / scaler.scale_)
    )
    gamma = np.concatenate([[raw_intercept], raw_coefficient])
    index = build_design(panel, basis) @ gamma
    probability = expit(index)
    probability = np.clip(probability, 1e-9, 1.0 - 1e-9)
    log_likelihood = float(
        np.sum(outcome * np.log(probability) + (1 - outcome) * np.log1p(-probability))
    )
    return gamma, probability, log_likelihood


def _conditional_mean_state(
    panel: pd.DataFrame,
    transitions: dict[str, object],
) -> dict[str, np.ndarray]:
    arr = {column: panel[column].to_numpy(float) for column in panel.columns}
    result = dict(arr)
    for variable in [*FIN_VARS, "unemp"]:
        const, rho, _ = transitions[variable]
        result[variable] = const + rho * arr[variable]
    result["trend"] = np.clip(arr["trend"] + 1.0 / 27.0, 0.0, 1.0)
    return result


def _expected_log_ccp(
    panel: pd.DataFrame,
    transitions: dict[str, object],
    basis: list[str],
    gamma: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Quasi-Monte Carlo estimate of E[log p_1(x') | x].

    Scrambled Sobol points reduce integration noise substantially relative to
    independent pseudo-random draws at the same computational cost.
    """

    if draws < 2:
        raise ValueError("--expectation-draws must be at least 2")
    variables = [*FIN_VARS, "unemp"]
    has_political_transition = "political_joint" in transitions
    dimension = len(variables) + int(has_political_transition)
    unit_draws = qmc.Sobol(d=dimension, scramble=True, seed=seed).random(draws)
    normal_draws = ndtri(
        np.clip(unit_draws[:, : len(variables)], 1e-12, 1.0 - 1e-12)
    )
    standard_deviations = np.array(
        [np.sqrt(max(transitions[variable][2], 0.0)) for variable in variables]
    )
    innovations = normal_draws * standard_deviations
    arr = {column: panel[column].to_numpy(float) for column in panel.columns}
    bounds = state_bounds(panel)
    total = np.zeros(len(panel))
    for index in range(draws):
        shock = {
            variable: innovations[index, position]
            for position, variable in enumerate(variables)
        }
        if has_political_transition:
            shock["__political_u"] = unit_draws[index, -1]
        next_state = next_state_arr(arr, shock, transitions, bounds)
        next_index = build_design(next_state, basis) @ gamma
        total += -np.logaddexp(0.0, -next_index)
    return total / draws


def _size_center_transform(spec: list[str]) -> np.ndarray:
    """Map raw [1, logA, logA^2] columns to a centered polynomial basis."""

    if spec[:3] != ["const", "log_assets", "log_assets^2"]:
        raise ValueError("NMC specification must start with const/log_assets/log_assets^2")
    transform = np.eye(len(spec))
    transform[:3, :3] = np.array(
        [
            [1.0, -SIZE_CENTER, SIZE_CENTER**2],
            [0.0, 1.0, -2.0 * SIZE_CENTER],
            [0.0, 0.0, 1.0],
        ]
    )
    return transform


def _fit_structural_panel(
    label: str,
    panel: pd.DataFrame,
    *,
    cost_coef: dict[str, float] | None = None,
    expectation_draws: int,
    seed: int,
) -> tuple[Result, PanelDiagnostics, list[str]]:
    mc_spec = MC_SPECS[label]
    nmc_spec = NMC_SPECS[label]
    ccp_spec = CCP_SPECS[label]

    if cost_coef is None:
        cost_coef = _fit_monetary_cost(panel, mc_spec)
    if list(cost_coef) != mc_spec:
        raise ValueError(
            f"{label}: monetary-cost coefficient order {list(cost_coef)} "
            f"does not match {mc_spec}"
        )
    transitions = _fit_transitions(panel)
    gamma, p_close, ccp_log_likelihood = _fit_ccp(panel, ccp_spec)

    expectation = _expected_log_ccp(
        panel,
        transitions,
        ccp_spec,
        gamma,
        draws=expectation_draws,
        seed=seed,
    )
    arr = {column: panel[column].to_numpy(float) for column in panel.columns}
    mean_next = _conditional_mean_state(panel, transitions)
    next_variance = {
        variable: transitions[variable][2] for variable in [*FIN_VARS, "unemp"]
    }

    size_transform = _size_center_transform(nmc_spec)
    x_now = build_design(arr, nmc_spec) @ size_transform
    x_next = expected_design(arr, nmc_spec, transitions) @ size_transform
    mc_now = monetary_cost(arr, cost_coef)
    mc_next = monetary_cost(mean_next, cost_coef, var=next_variance)
    log_continue_close = np.log1p(-p_close) - np.log(p_close)

    c_matrix = np.column_stack(
        [
            log_continue_close + BETA * expectation,
            x_now - BETA * x_next,
        ]
    )
    d_vector = BETA * mc_next - mc_now

    # The paper's instruments are the NMC index plus excluded state variables.
    # These simulated panels omit asset_growth_1y, so equity and unemployment
    # provide the two exclusion terms in addition to the complete NMC design.
    nmc_instruments = build_design(arr, nmc_spec) @ size_transform
    excluded_instruments = build_design(arr, ["equity_a", "unemp"])
    instruments = np.column_stack([nmc_instruments, excluded_instruments])
    gmm = gmm_fit(c_matrix, d_vector, instruments, maxiter=2)
    if not np.all(np.isfinite(gmm.params)):
        raise RuntimeError(f"{label}: structural GMM returned non-finite estimates")

    sigma = float(gmm.params[0])
    if sigma <= 0:
        raise RuntimeError(f"{label}: estimated sigma must be positive, got {sigma}")
    raw_theta = size_transform @ np.asarray(gmm.params[1:], float)
    theta = {
        term: float(value)
        for term, value in zip(nmc_spec, raw_theta, strict=True)
    }
    j_stat, j_pvalue, j_dof = gmm.jtest()
    column_scale = np.sqrt(np.mean(c_matrix**2, axis=0))
    instrument_scale = np.sqrt(np.mean(instruments**2, axis=0))
    scaled_cross_moment = (
        (instruments / instrument_scale).T
        @ (c_matrix / column_scale)
        / len(panel)
    )
    gmm_condition = float(np.linalg.cond(scaled_cross_moment))
    sigma_equity_only = float(
        gmm_fit(
            c_matrix,
            d_vector,
            np.column_stack([nmc_instruments, excluded_instruments[:, 0]]),
            maxiter=2,
        ).params[0]
    )
    sigma_unemp_only = float(
        gmm_fit(
            c_matrix,
            d_vector,
            np.column_stack([nmc_instruments, excluded_instruments[:, 1]]),
            maxiter=2,
        ).params[0]
    )
    political_lag = panel.groupby("rssd_id", sort=False)[["house", "senate"]].shift(1)
    valid_political = political_lag.notna().all(axis=1)
    political_change_rate = float(
        np.mean(
            np.any(
                panel.loc[valid_political, ["house", "senate"]].to_numpy()
                != political_lag.loc[valid_political].to_numpy(),
                axis=1,
            )
        )
    )
    result = Result(
        beta=BETA,
        sigma=sigma,
        theta=theta,
        cost_coef=cost_coef,
        trans=transitions,
        spec=nmc_spec,
        data=panel,
    )
    diagnostics = PanelDiagnostics(
        label=label,
        n_rows=len(panel),
        n_failures=int(panel["failed"].sum()),
        ccp_mean=float(p_close.mean()),
        ccp_log_likelihood=ccp_log_likelihood,
        expectation_draws=expectation_draws,
        gmm_j=float(j_stat),
        gmm_pvalue=float(j_pvalue),
        gmm_dof=int(j_dof),
        gmm_condition=gmm_condition,
        sigma_equity_only=sigma_equity_only,
        sigma_unemp_only=sigma_unemp_only,
        political_change_rate=political_change_rate,
    )
    return result, diagnostics, ccp_spec


def _collect_estimates(
    results: dict[str, Result],
    grid: dict[tuple[str, str], float],
) -> dict[str, float]:
    """Return the complete 51-estimate S1/S2/S3 submission mapping."""

    values: dict[str, float] = {}
    for label, result in results.items():
        values[f"{label}_sigma"] = float(result.sigma)
        values.update(
            {
                f"{label}_theta_{term}": float(result.theta[term])
                for term in result.spec
            }
        )
        values.update(
            {
                f"{label}_mc_{term}": float(coefficient)
                for term, coefficient in result.cost_coef.items()
            }
        )
    values.update(
        {
            f"{label}_cf_{scenario}": float(value)
            for (label, scenario), value in grid.items()
        }
    )
    if set(values) != _canonical_ids():
        missing = sorted(_canonical_ids().difference(values))
        extra = sorted(set(values).difference(_canonical_ids()))
        raise RuntimeError(
            f"Internal estimate-ID mismatch; missing={missing}, extra={extra}"
        )
    return values


def _write_and_validate_submission(
    output: Path,
    template: pd.DataFrame,
    estimates: dict[str, float],
) -> pd.DataFrame:
    missing = [identifier for identifier in template["Id"] if identifier not in estimates]
    if missing:
        raise ValueError(f"No estimate is available for template IDs: {missing}")
    submission = template.copy()
    submission["Prediction"] = [
        float(estimates[identifier]) for identifier in submission["Id"]
    ]
    values = submission["Prediction"].to_numpy(float)
    if len(submission) != EXPECTED_ROWS or not np.all(np.isfinite(values)):
        raise ValueError(f"Submission must contain {EXPECTED_ROWS} finite predictions")
    if np.allclose(values, 0.0):
        raise ValueError("Submission still contains only template placeholder zeros")

    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    check = pd.read_csv(output)
    if list(check.columns) != list(template.columns):
        raise RuntimeError("Written submission columns do not match the template")
    if check["Id"].tolist() != template["Id"].tolist():
        raise RuntimeError("Written submission IDs/order do not match the template")
    if not np.all(
        np.isfinite(pd.to_numeric(check["Prediction"], errors="coerce").to_numpy(float))
    ):
        raise RuntimeError("Written submission contains non-finite predictions")
    return check


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    template_path = (
        data_dir / "submission_template.csv"
        if args.template is None
        else args.template.expanduser().resolve()
    )
    output = args.output.expanduser().resolve()
    _require_files(data_dir, template_path)
    template = _read_template(template_path)
    print(f"Using submission template: {template_path} ({len(template)} rows)")

    started = time.perf_counter()
    panels = {
        label: _load_panel(data_dir / f"set_{label}.parquet")
        for label in ("S1", "S2", "S3")
    }
    pooled_cost, pooled_cost_r2, pooled_cost_rmse = _fit_pooled_monetary_cost(
        panels
    )
    print(
        "Pooled cumulative monetary-cost model: "
        f"R2={pooled_cost_r2:.6f}, RMSE={pooled_cost_rmse:.3f}"
    )
    print(f"Shared S1 coefficients: {pooled_cost['S1']}")
    print(f"Shared S2/S3 quadratic coefficient: {pooled_cost['S2']['npf_a^2']}")

    results: dict[str, Result] = {}
    ccp_specs: dict[str, list[str]] = {}
    for offset, label in enumerate(("S1", "S2", "S3")):
        panel_path = data_dir / f"set_{label}.parquet"
        print(f"\n[{label}] Estimating {panel_path}")
        panel = panels[label]
        result, diagnostics, ccp_spec = _fit_structural_panel(
            label,
            panel,
            cost_coef=pooled_cost[label],
            expectation_draws=args.expectation_draws,
            seed=args.seed + offset,
        )
        results[label] = result
        ccp_specs[label] = ccp_spec
        print(
            f"[{label}] n={diagnostics.n_rows:,}, failures={diagnostics.n_failures:,}, "
            f"mean CCP={diagnostics.ccp_mean:.6f}, sigma={result.sigma:.6f}, "
            f"J/N={diagnostics.gmm_j / diagnostics.n_rows:.6f}, "
            f"condition={diagnostics.gmm_condition:.1f}"
        )
        print(
            f"[{label}] exact-ID sigma sensitivity: "
            f"equity={diagnostics.sigma_equity_only:.3f}, "
            f"unemp={diagnostics.sigma_unemp_only:.3f}; "
            f"political transition rate={diagnostics.political_change_rate:.3%}"
        )
        print(f"[{label}] monetary cost: {result.cost_coef}")
        print(f"[{label}] nonmonetary theta: {result.theta}")

    if not args.counterfactual_seeds:
        raise ValueError("--counterfactual-seeds requires at least one seed")
    print(
        "\n[S3] Solving converged fixed-point counterfactuals for seeds "
        f"{args.counterfactual_seeds}..."
    )
    grid: dict[tuple[str, str], float] = {}
    counterfactual_draws: dict[str, list[float]] = {
        scenario: [] for scenario in SCENARIOS
    }
    for counterfactual_seed in args.counterfactual_seeds:
        seed_values: dict[str, float] = {}
        for scenario in SCENARIOS:
            value = counterfactual(
                results["S3"],
                scenario,
                seed=counterfactual_seed,
                R=args.counterfactual_draws,
                n_anchor=args.counterfactual_anchor,
                basis=ccp_specs["S3"],
            )
            counterfactual_draws[scenario].append(float(value))
            seed_values[scenario] = float(value)
        print(f"[S3] seed={counterfactual_seed}: {seed_values}")
    for scenario, values in counterfactual_draws.items():
        average = float(np.mean(values))
        grid[("S3", scenario)] = average
        print(
            f"[S3] {scenario}: mean={average:.6f}%, "
            f"range=({min(values):.6f}, {max(values):.6f})"
        )

    estimates = _collect_estimates(results, grid)
    submission = _write_and_validate_submission(
        output,
        template,
        estimates,
    )
    elapsed = time.perf_counter() - started
    print(
        f"\nSUCCESS: {output}\n"
        f"Validated {len(submission)} rows against {template_path.name}; "
        f"all predictions are finite. Total runtime: {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
