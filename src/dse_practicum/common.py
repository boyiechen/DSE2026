"""Shared validation and numerical helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def require_columns(frame: pd.DataFrame, columns: list[str], *, label: str) -> None:
    """Raise a useful error when a data file has an unexpected schema."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def fill_submission_template(
    template_path: str | Path,
    estimates: Mapping[str, float],
    *,
    id_column: str,
    value_column: str,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fill a Kaggle sample-submission file without changing its row order."""

    template = pd.read_csv(template_path)
    require_columns(template, [id_column, value_column], label="submission template")

    expected = template[id_column].astype(str).tolist()
    missing = [key for key in expected if key not in estimates]
    extra = sorted(set(estimates).difference(expected))
    if missing:
        raise ValueError(f"No estimates supplied for submission rows: {missing}")
    if extra:
        raise ValueError(f"Estimates contain rows not present in the template: {extra}")

    values = np.asarray([estimates[key] for key in expected], dtype=float)
    if not np.all(np.isfinite(values)):
        bad = [expected[i] for i in np.flatnonzero(~np.isfinite(values))]
        raise ValueError(f"Submission contains non-finite estimates for: {bad}")

    result = template.copy()
    result[value_column] = values
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
    return result


def symmetric_matrix_sqrt(matrix: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    """Return a stable symmetric square root for a PSD matrix."""

    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    values = np.maximum(values, floor)
    return (vectors * np.sqrt(values)) @ vectors.T


def stable_logit(probability: np.ndarray, *, clip: float) -> np.ndarray:
    """Compute log(p / (1-p)) after explicit probability clipping."""

    p = np.clip(np.asarray(probability, dtype=float), clip, 1.0 - clip)
    return np.log(p) - np.log1p(-p)
