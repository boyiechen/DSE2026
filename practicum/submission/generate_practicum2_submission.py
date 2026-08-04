#!/usr/bin/env python3
"""Estimate Practicum 2 and write the validated Kaggle submission.

This is a standalone version of the completed student notebook.  It estimates
the truncated working-output distribution, the optimal compensation contract,
the moral-hazard cost decomposition, bootstrap standard errors, and the
double-risk-aversion counterfactual.  The final 19-row CSV is checked against
the official Kaggle sample submission.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PRACTICUM2_DIR = REPO_ROOT / "practicum" / "practicum2"
DEFAULT_DATA = PRACTICUM2_DIR / "data" / "simulated_moral_hazard_data.dta"
DEFAULT_TEMPLATE = PRACTICUM2_DIR / "data" / "sample_submission.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "practicum2_submission.csv"

X_COLUMN = "net_excess_ret"
WAGE_COLUMN = "totalComp"
EXPECTED_IDS = [
    "psi",
    "mu_w",
    "sigma",
    "mu_s",
    "gamma",
    "alpha",
    "beta",
    "se_mu_w",
    "se_sigma",
    "se_mu_s",
    "se_gamma",
    "se_alpha",
    "se_beta",
    "delta_1",
    "delta_2",
    "delta_3",
    "cf_gamma_delta_1",
    "cf_gamma_delta_2",
    "cf_gamma_delta_3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the official 19-row Practicum 2 Kaggle CSV."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"Input Stata data (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"Official sample submission (default: {DEFAULT_TEMPLATE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=500,
        help="Nonparametric bootstrap replications (default: 500)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20_260_726,
        help="Bootstrap random seed (default: 20260726)",
    )
    return parser.parse_args()


def truncated_normal_mean(mu: float, sigma: float, psi: float) -> float:
    """Mean of N(mu, sigma^2) conditional on being at least psi."""

    standardized_cutoff = (psi - mu) / sigma
    return float(
        mu
        + sigma
        * np.exp(
            stats.norm.logpdf(standardized_cutoff)
            - stats.norm.logsf(standardized_cutoff)
        )
    )


class MoralHazardModel:
    """Sequential estimator for the Margiotta--Miller static model."""

    def __init__(self) -> None:
        self.x: np.ndarray | None = None
        self.wage: np.ndarray | None = None
        self.n: int | None = None
        self.psi: float | None = None
        self.mu_w_hat: float | None = None
        self.sigma_hat: float | None = None
        self.gamma_hat: float | None = None
        self.alpha_hat: float | None = None
        self.beta_hat: float | None = None
        self.mu_s_hat: float | None = None
        self.eta_hat: float | None = None
        self.ssr_: float | None = None

    def set_data(self, x: np.ndarray, wage: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=float).reshape(-1)
        self.wage = np.asarray(wage, dtype=float).reshape(-1)
        if len(self.x) != len(self.wage):
            raise ValueError("Output and compensation must have equal lengths")
        if len(self.x) < 20:
            raise ValueError("At least 20 observations are required")
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.wage)):
            raise ValueError("Output and compensation must contain only finite values")
        self.n = len(self.x)

    def step1_estimate_threshold(self) -> float:
        if self.x is None:
            raise RuntimeError("Call set_data before Step 1")
        self.psi = float(np.min(self.x))
        return self.psi

    def step2_estimate_work_distribution(self) -> tuple[float, float]:
        if self.x is None or self.psi is None:
            raise RuntimeError("Run Steps 0-1 before Step 2")

        def negative_log_likelihood(parameters: np.ndarray) -> float:
            mu, log_sigma = parameters
            sigma = float(np.exp(log_sigma))
            standardized = (self.x - mu) / sigma
            cutoff = (self.psi - mu) / sigma
            log_density = (
                stats.norm.logpdf(standardized)
                - log_sigma
                - stats.norm.logsf(cutoff)
            )
            if not np.all(np.isfinite(log_density)):
                return np.inf
            return float(-np.sum(log_density))

        initial = np.array([self.x.mean(), np.log(self.x.std())])
        fit = optimize.minimize(
            negative_log_likelihood,
            initial,
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 5_000},
        )
        if not fit.success or not np.isfinite(fit.fun):
            raise RuntimeError(f"Truncated-normal MLE failed: {fit.message}")
        self.mu_w_hat = float(fit.x[0])
        self.sigma_hat = float(np.exp(fit.x[1]))
        return self.mu_w_hat, self.sigma_hat

    def likelihood_ratio(
        self,
        x: np.ndarray,
        mu_s: float,
        *,
        mu_w: float | None = None,
        sigma: float | None = None,
        psi: float | None = None,
    ) -> np.ndarray:
        mu_w = self.mu_w_hat if mu_w is None else mu_w
        sigma = self.sigma_hat if sigma is None else sigma
        psi = self.psi if psi is None else psi
        if mu_w is None or sigma is None or psi is None or sigma <= 0:
            raise ValueError("The working distribution must be fitted first")
        values = np.asarray(x, dtype=float)
        log_g = (
            stats.norm.logcdf((mu_w - psi) / sigma)
            - stats.norm.logcdf((mu_s - psi) / sigma)
            + (mu_w**2 - mu_s**2) / (2.0 * sigma**2)
            + ((mu_s - mu_w) / sigma**2) * values
        )
        return np.exp(np.clip(log_g, -50.0, 50.0))

    @staticmethod
    def solve_eta(r: float, likelihood_ratio: np.ndarray) -> tuple[float, bool]:
        difference = r - likelihood_ratio
        minimum = float(difference.min())
        if minimum >= 0:
            return 0.0, False
        upper = -1.0 / minimum
        lower = 1e-14
        upper *= 1.0 - 1e-9

        def equation(eta: float) -> float:
            return float(np.mean(difference / (1.0 + eta * difference)))

        if equation(lower) * equation(upper) > 0:
            return 0.0, False
        eta = optimize.brentq(
            equation, lower, upper, xtol=1e-12, rtol=1e-12, maxiter=300
        )
        return float(eta), True

    def optimal_wage(
        self,
        x: np.ndarray,
        gamma: float,
        alpha: float,
        beta: float,
        mu_s: float,
        *,
        mu_w: float | None = None,
        sigma: float | None = None,
        psi: float | None = None,
    ) -> tuple[np.ndarray, float]:
        if gamma <= 0 or alpha <= 0 or beta <= 0:
            raise ValueError("gamma, alpha, and beta must be positive")
        r = alpha / beta
        g = self.likelihood_ratio(
            x, mu_s, mu_w=mu_w, sigma=sigma, psi=psi
        )
        eta, _ = self.solve_eta(r, g)
        inside = alpha * (1.0 + eta * (r - g))
        if np.any(inside <= 0) or not np.all(np.isfinite(inside)):
            raise ValueError("The trial contract has an invalid logarithm")
        return np.log(inside) / gamma, eta

    def _residuals(self, parameters: np.ndarray) -> np.ndarray:
        if self.x is None or self.wage is None:
            raise RuntimeError("Data are not set")
        gamma, alpha, r, mu_s = parameters
        beta = alpha / r
        fitted_wage, _ = self.optimal_wage(
            self.x, gamma, alpha, beta, mu_s
        )
        return self.wage - fitted_wage

    def _search_space(
        self, n_starts: int
    ) -> tuple[tuple[list[float], list[float]], list[list[float]]]:
        if self.psi is None or self.mu_w_hat is None:
            raise RuntimeError("Run Steps 1-2 first")
        lower = [1e-9, 1e-3, 1.0001, self.psi + 1e-3]
        upper = [1e-2, 1_000.0, 100.0, self.mu_w_hat - 1e-3]
        starts = [
            [1e-4, 2.0, 1.5, self.mu_w_hat - 0.2],
            [1e-4, 5.0, 1.1, self.mu_w_hat - 0.4],
            [1e-5, 1.5, 2.0, self.mu_w_hat - 0.6],
            [1e-3, 10.0, 1.3, self.mu_w_hat - 0.1],
            [1e-4, 3.0, 1.8, self.mu_w_hat - 0.3],
        ][:n_starts]
        return (lower, upper), starts

    def step3_estimate_contract(self, n_starts: int = 5) -> dict[str, float]:
        if self.x is None:
            raise RuntimeError("Data are not set")
        bounds, starts = self._search_space(n_starts)
        best: Optional[dict[str, float]] = None
        for initial in starts:
            fit = optimize.least_squares(
                self._residuals,
                x0=initial,
                method="trf",
                bounds=bounds,
                x_scale="jac",
                max_nfev=3_000,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            ssr = float(np.sum(fit.fun**2))
            if np.isfinite(ssr) and (best is None or ssr < best["ssr"]):
                gamma, alpha, r, mu_s = fit.x
                beta = alpha / r
                _, eta = self.optimal_wage(
                    self.x, gamma, alpha, beta, mu_s
                )
                best = {
                    "gamma": float(gamma),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "r": float(r),
                    "mu_s": float(mu_s),
                    "eta": float(eta),
                    "ssr": ssr,
                }
        if best is None:
            raise RuntimeError("Contract NLS failed from every starting value")
        self.gamma_hat = best["gamma"]
        self.alpha_hat = best["alpha"]
        self.beta_hat = best["beta"]
        self.mu_s_hat = best["mu_s"]
        self.eta_hat = best["eta"]
        self.ssr_ = best["ssr"]
        return best

    def cost_decomposition(self) -> dict[str, float]:
        if self.x is None:
            raise RuntimeError("Data are not set")
        required = (
            self.gamma_hat,
            self.alpha_hat,
            self.beta_hat,
            self.eta_hat,
            self.mu_w_hat,
            self.mu_s_hat,
            self.sigma_hat,
            self.psi,
        )
        if any(value is None for value in required):
            raise RuntimeError("The full model must be fitted first")
        gamma = float(self.gamma_hat)
        alpha = float(self.alpha_hat)
        beta = float(self.beta_hat)
        eta = float(self.eta_hat)
        mu_w = float(self.mu_w_hat)
        mu_s = float(self.mu_s_hat)
        sigma = float(self.sigma_hat)
        psi = float(self.psi)
        r = alpha / beta
        g = self.likelihood_ratio(self.x, mu_s)
        contract_term = 1.0 + eta * (r - g)
        if np.any(contract_term <= 0):
            raise ValueError("The fitted contract has an invalid logarithm")
        return {
            "delta_1": float(np.mean(np.log(contract_term)) / gamma),
            "delta_2": float(np.log(r) / gamma),
            "delta_3": float(
                truncated_normal_mean(mu_w, sigma, psi)
                - truncated_normal_mean(mu_s, sigma, psi)
            ),
        }

    def _bootstrap_replicate(
        self,
        x: np.ndarray,
        wage: np.ndarray,
        *,
        psi_fixed: float,
        n_starts: int,
    ) -> dict[str, float] | None:
        model = MoralHazardModel()
        model.x = x
        model.wage = wage
        model.n = len(x)
        model.psi = psi_fixed
        try:
            model.step2_estimate_work_distribution()
            fit = model.step3_estimate_contract(n_starts=n_starts)
        except (RuntimeError, ValueError, FloatingPointError):
            return None
        values = {
            "mu_w": float(model.mu_w_hat),
            "sigma": float(model.sigma_hat),
            "mu_s": float(model.mu_s_hat),
            "gamma": float(model.gamma_hat),
            "alpha": float(model.alpha_hat),
            "beta": float(model.beta_hat),
            "r": float(fit["r"]),
        }
        if not np.all(np.isfinite(list(values.values()))):
            return None
        return values

    def bootstrap_standard_errors(
        self,
        *,
        repetitions: int = 500,
        seed: int = 20_260_726,
        n_starts: int = 2,
    ) -> dict[str, float]:
        if self.x is None or self.wage is None or self.n is None or self.psi is None:
            raise RuntimeError("Fit the main model before bootstrapping")
        if repetitions < 2:
            raise ValueError("--bootstrap-reps must be at least 2")
        rng = np.random.default_rng(seed)
        records: list[dict[str, float]] = []
        for replicate in range(repetitions):
            index = rng.integers(0, self.n, size=self.n)
            result = self._bootstrap_replicate(
                self.x[index],
                self.wage[index],
                psi_fixed=self.psi,
                n_starts=n_starts,
            )
            if result is not None:
                records.append(result)
            if (replicate + 1) % 50 == 0 or replicate + 1 == repetitions:
                print(
                    f"Bootstrap progress: {replicate + 1}/{repetitions}; "
                    f"{len(records)} usable fits"
                )
        minimum = max(10, int(np.ceil(0.7 * repetitions)))
        if len(records) < minimum:
            raise RuntimeError(
                f"Only {len(records)}/{repetitions} bootstrap fits succeeded; "
                f"at least {minimum} are required"
            )
        return {
            f"se_{name}": float(
                np.std([record[name] for record in records], ddof=1)
            )
            for name in ("mu_w", "sigma", "mu_s", "gamma", "alpha", "beta")
        }

    def counterfactual(self, *, gamma_multiplier: float) -> dict[str, float]:
        if self.x is None:
            raise RuntimeError("Data are not set")
        gamma = float(self.gamma_hat) * gamma_multiplier
        alpha = float(self.alpha_hat)
        beta = float(self.beta_hat)
        mu_w = float(self.mu_w_hat)
        mu_s = float(self.mu_s_hat)
        sigma = float(self.sigma_hat)
        psi = float(self.psi)
        r = alpha / beta
        g = self.likelihood_ratio(
            self.x, mu_s, mu_w=mu_w, sigma=sigma, psi=psi
        )
        eta, _ = self.solve_eta(r, g)
        contract_term = 1.0 + eta * (r - g)
        if np.any(contract_term <= 0):
            raise ValueError("The counterfactual contract has an invalid logarithm")
        return {
            "delta_1": float(np.mean(np.log(contract_term)) / gamma),
            "delta_2": float(np.log(r) / gamma),
            "delta_3": float(
                truncated_normal_mean(mu_w, sigma, psi)
                - truncated_normal_mean(mu_s, sigma, psi)
            ),
        }


def _load_and_validate_template(path: Path) -> pd.DataFrame:
    template = pd.read_csv(path)
    if list(template.columns) != ["ID", "ESTIMATE"]:
        raise ValueError(f"Unexpected Practicum 2 template columns: {list(template.columns)}")
    if template["ID"].tolist() != EXPECTED_IDS:
        raise ValueError("The official Practicum 2 template has an unexpected ID order")
    return template


def _write_submission(
    model: MoralHazardModel,
    standard_errors: dict[str, float],
    costs: dict[str, float],
    counterfactual: dict[str, float],
    *,
    template_path: Path,
    output: Path,
) -> pd.DataFrame:
    values = {
        "psi": float(model.psi),
        "mu_w": float(model.mu_w_hat),
        "sigma": float(model.sigma_hat),
        "mu_s": float(model.mu_s_hat),
        "gamma": float(model.gamma_hat),
        "alpha": float(model.alpha_hat),
        "beta": float(model.beta_hat),
        **standard_errors,
        **costs,
        **{
            f"cf_gamma_{name}": value
            for name, value in counterfactual.items()
        },
    }
    missing = [identifier for identifier in EXPECTED_IDS if identifier not in values]
    if missing:
        raise ValueError(f"Missing submission estimates: {missing}")
    numeric = np.asarray([values[identifier] for identifier in EXPECTED_IDS], dtype=float)
    if not np.all(np.isfinite(numeric)):
        bad = [
            EXPECTED_IDS[index]
            for index in np.flatnonzero(~np.isfinite(numeric))
        ]
        raise ValueError(f"Non-finite submission estimates: {bad}")

    submission = _load_and_validate_template(template_path).copy()
    submission["ESTIMATE"] = numeric
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)

    check = pd.read_csv(output)
    if list(check.columns) != ["ID", "ESTIMATE"]:
        raise RuntimeError("Written submission has the wrong columns")
    if check["ID"].tolist() != EXPECTED_IDS:
        raise RuntimeError("Written submission has the wrong ID order")
    if not np.all(np.isfinite(pd.to_numeric(check["ESTIMATE"], errors="coerce"))):
        raise RuntimeError("Written submission contains non-finite estimates")
    return check


def main() -> None:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    template_path = args.template.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Practicum 2 data not found: {data_path}")
    if not template_path.is_file():
        raise FileNotFoundError(f"Practicum 2 template not found: {template_path}")

    started = time.perf_counter()
    frame = pd.read_stata(data_path)
    missing = sorted({X_COLUMN, WAGE_COLUMN}.difference(frame.columns))
    if missing:
        raise ValueError(f"Practicum 2 data are missing columns: {missing}")

    model = MoralHazardModel()
    model.set_data(
        frame[X_COLUMN].to_numpy(float),
        frame[WAGE_COLUMN].to_numpy(float),
    )
    model.step1_estimate_threshold()
    model.step2_estimate_work_distribution()
    fit = model.step3_estimate_contract(n_starts=5)
    costs = model.cost_decomposition()
    print(
        f"Main fit: n={model.n:,}, psi={model.psi:.8f}, "
        f"mu_w={model.mu_w_hat:.8f}, sigma={model.sigma_hat:.8f}, "
        f"gamma={fit['gamma']:.10g}, alpha={fit['alpha']:.8f}, "
        f"beta={fit['beta']:.8f}, mu_s={fit['mu_s']:.8f}"
    )

    standard_errors = model.bootstrap_standard_errors(
        repetitions=args.bootstrap_reps,
        seed=args.bootstrap_seed,
        n_starts=2,
    )
    counterfactual = model.counterfactual(gamma_multiplier=2.0)
    submission = _write_submission(
        model,
        standard_errors,
        costs,
        counterfactual,
        template_path=template_path,
        output=output,
    )
    elapsed = time.perf_counter() - started
    print(
        f"\nSUCCESS: {output}\n"
        f"Validated {len(submission)} rows against {template_path.name}; "
        f"all estimates are finite. Total runtime: {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
