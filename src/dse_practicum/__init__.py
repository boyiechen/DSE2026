"""Notebook-friendly estimators for the two DSE 2026 practica."""

from .practicum1 import (
    P1CounterfactualConfig,
    P1DatasetSpec,
    P1FitConfig,
    Practicum1AllResults,
    Practicum1Result,
    fit_practicum1,
    fit_practicum1_all,
)
from .practicum2 import P2FitConfig, Practicum2Result, fit_practicum2

__all__ = [
    "P1CounterfactualConfig",
    "P1DatasetSpec",
    "P1FitConfig",
    "P2FitConfig",
    "Practicum1AllResults",
    "Practicum1Result",
    "Practicum2Result",
    "fit_practicum1",
    "fit_practicum1_all",
    "fit_practicum2",
]
