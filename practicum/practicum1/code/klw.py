"""klw.py — the numerical toolkit for DSE2026 Practicum 1.

Kang, Lowery & Wardlaw (2015), *Review of Financial Studies* 28(4): 1060-1102.

This module holds the machinery the practicum notebook is GIVEN: three general estimators,
the calendar-lag and basis construction the panel needs, and the simulated-panel solver that
makes a counterfactual a counterfactual.  **Nothing here is specific to the paper.**  Every
specification, every moment condition and every modelling choice stays in the notebook, which
is where the economics belongs.

    estimators      Tobit          censored regression, any censoring regime
                    PanelVAR       panel VAR(p) on CALENDAR lags
                    gmm_fit        GMM for a moment linear in the parameters

    panel plumbing  find_data_dir / load_real_data / load_sim_data
                    calendar_seq / add_calendar_lags
                    bspline_basis

    simulated panel build_design / monetary_cost / make_cost_fn / expected_design
                    solve_ccp / solve_ccp_gamma / counterfactual
                    Result / as_series / make_submission

Import by name, not as a namespace -- the notebook reads `Tobit.from_frame(...)`, never
`klw.Tobit.from_frame(...)`:

    from klw import Tobit, PanelVAR, gmm_fit, counterfactual, ...

Run `python klw.py` to execute the self-tests.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.interpolate import BSpline
from scipy.special import expit
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.sandbox.regression.gmm import LinearIVGMM


# ======================================================================== locating the data
#
# One marker file identifies the bundle.  On Kaggle the folder is attached read-only somewhere
# under /kaggle/input/<dataset>/; locally it is `data/` at or above the notebook.

_MARKER = "replication.parquet"

# Where the panels are published.  klw.py itself is fetched from `{GITHUB_RAW}/src/` by the
# notebook's setup cell; the data files live under `{GITHUB_RAW}/data/`.
GITHUB_RAW = ("https://raw.githubusercontent.com/mhnam/dse2026-practicum/main/"
              "practicum1-bank-closure")

# What the notebook actually opens.  The S-panels are the Kaggle assignment, not this notebook,
# so they are not downloaded by default -- pass them to `download_data` if you want them.
_BUNDLE = ("replication.parquet", "replication_real.parquet", "refcheck.json",
           "klw_first_stage.npz", "dataforparam_KLW.csv")

DATA_DIR = None          # set by find_data_dir(); the loaders below default to it


def _kaggle_find(marker):
    """The directory under /kaggle/input holding `marker`, or None.

    Kaggle mounts attached data through symlinked directories, which `os.walk` does NOT
    descend into by default -- a competition can sit at /kaggle/input/competitions/<slug>/
    and be invisible to a plain walk.  So the shallow paths are globbed explicitly first
    (glob resolves symlinks), and only then is a followlinks walk used as a catch-all.
    """
    import glob
    for pat in ("/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/"):
        for d in sorted(glob.glob(pat)):
            if os.path.isfile(os.path.join(d, marker)):
                return d.rstrip("/")
    for r, _, files in os.walk("/kaggle/input", followlinks=True):
        if marker in files:
            return r
    return None


def _kaggle_tree(limit=40):
    """What is actually mounted under /kaggle/input -- for the not-found message."""
    import glob
    if not os.path.isdir("/kaggle/input"):
        return "  (no /kaggle/input on this machine)"
    seen = sorted(glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")
                  + glob.glob("/kaggle/input/*/*/*"))
    if not seen:
        return "  (nothing is attached)"
    out = "\n".join("  " + s for s in seen[:limit])
    return out + ("\n  ..." if len(seen) > limit else "")


def download_data(dest=None, files=_BUNDLE):
    """Fetch the practicum's panels from GitHub into `dest`, and remember the directory.

    The fallback when no bundle is attached.  On Kaggle that means Internet must be ON
    (Notebook options -> Internet).  Files already downloaded are not fetched again, so
    re-running the setup cell is cheap.  About 20 MB in total.
    """
    global DATA_DIR
    import urllib.request
    if dest is None:
        base = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
        dest = os.path.join(base, "data")
    os.makedirs(dest, exist_ok=True)
    for f in files:
        path = os.path.join(dest, f)
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{GITHUB_RAW}/data/{f}", path)
    DATA_DIR = dest
    return dest


def find_data_dir(root=None, marker=_MARKER, levels=4, download=False):
    """Locate the practicum's `data/` folder and remember it for the loaders.

    Searches /kaggle/input first when running on Kaggle, then `data/` at `root` and up to
    `levels` parents (the answer key sits two levels below the practicum root).  `root`
    defaults to $DSE2026_ROOT or the working directory.  With `download=True`, a bundle that
    is nowhere to be found is fetched from GitHub instead of raising.
    """
    global DATA_DIR
    if os.path.isdir("/kaggle/input"):
        hit = _kaggle_find(marker)
        if hit:
            DATA_DIR = hit
            return hit
    here = root or os.environ.get("DSE2026_ROOT") or os.getcwd()
    for _ in range(levels):
        cand = os.path.join(here, "data")
        if os.path.isfile(os.path.join(cand, marker)):
            DATA_DIR = cand
            return cand
        here = os.path.dirname(here)
    if download:
        return download_data()
    raise FileNotFoundError(
        f"could not find the practicum's data/ folder (looked for {marker} under "
        "/kaggle/input, then in data/ at or above $DSE2026_ROOT or the working directory).\n"
        "Pass download=True to fetch the panels from GitHub instead.\n"
        f"What is attached under /kaggle/input:\n{_kaggle_tree()}")


def _data_dir(data_dir=None):
    d = data_dir or DATA_DIR
    if d is None:
        raise RuntimeError("call find_data_dir() before loading a panel")
    return d


_REAL_PANEL = "replication_real.parquet"     # NOT _MARKER: that names the SIMULATED panel


def load_real_data(data_dir=None):
    """The resolved-bank Call-Report panel (48,245 bank-quarters, 26 columns).

    A different file from `load_sim_data("replication")`, which is the simulated Stage-0
    panel: only this one carries the LEVELS (`assets`, `asset_growth_1y`) that the §2 Tobit,
    VAR and instrument set are built on.
    """
    path = os.path.join(_data_dir(data_dir), _REAL_PANEL)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{_REAL_PANEL} is not in this bundle (looked in "
                                f"{_data_dir(data_dir)})")
    return pd.read_parquet(path)


_DATASETS = {"replication": "replication.parquet",
             "S1": "set_S1.parquet", "S2": "set_S2.parquet", "S3": "set_S3.parquet"}


def load_sim_data(name, data_dir=None):
    """Load a simulated panel by name: 'replication', 'S1', 'S2' or 'S3'."""
    if name not in _DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {list(_DATASETS)}")
    d = _data_dir(data_dir)
    path = os.path.join(d, _DATASETS[name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{_DATASETS[name]} is not in this bundle (looked in {d})")
    return pd.read_parquet(path)


# ================================================================ calendar lags and the basis

def calendar_seq(yearq) -> np.ndarray:
    """Contiguous quarter index from this panel's YYYYQ encoding (19844 -> 1984Q4).

    A property of the data, not of any estimator: both the VAR and the CCP sample need a
    period that advances by 1, and this is how to get one here.
    """
    yq = np.asarray(yearq).astype("int64")
    return (yq // 10) * 4 + (yq % 10)


def add_calendar_lags(df, lag_vars, lags, unit="rssd_id", time="yearq", period=calendar_seq):
    """Return a copy of `df` with `{v}_lag{k}` COLUMNS, for each k in `lags` greater than 0.

    Lag k is the same unit k periods earlier, by MERGE -- a unit with a gap in its history
    yields NaN rather than borrowing the previous ROW's value.  `PanelVAR` builds its own
    design instead of columns; this is for code that needs the lags on the frame.
    """
    df = df.copy()
    df["_seq"] = period(df[time].to_numpy())
    for k in sorted(x for x in lags if x > 0):
        src = df[[unit, "_seq"] + list(lag_vars)].copy()
        src["_seq"] = src["_seq"] + k
        src = src.rename(columns={v: f"{v}_lag{k}" for v in lag_vars})
        df = df.merge(src, on=[unit, "_seq"], how="left")
    return df.drop(columns="_seq")


def bspline_basis(x, lo, hi, degree=3, interior=(), drop_last=True):
    """Clamped B-spline basis on [lo, hi], EXTRAPOLATING (not clipping) outside it.

    With no interior knots this is the degree-`degree` Bernstein basis -- for degree 3 exactly
    what MATLAB CompEcon's `fundefn('spli', 4, lo, hi)` produces, so each fitted term is one
    GLOBAL cubic and the gain over a linear index is curvature, not local adaptivity.  Adding
    interior knots makes it a genuine piecewise spline; the column count becomes
    `len(interior) + degree + 1`.

    Three properties callers rely on:

      1. partition of unity -- the columns sum to 1 at every x, so dropping one against an
         intercept loses nothing (`drop_last`);
      2. frozen knots -- [lo, hi] is fixed by the caller, so a SIMULATED x is measured on the
         same scale the model was fitted on;
      3. extrapolation -- outside [lo, hi] the polynomials continue and columns may go
         negative, so a simulated state just past the observed range still moves smoothly.
    """
    knots = np.r_[[lo] * (degree + 1), np.asarray(interior, float), [hi] * (degree + 1)]
    ncol = len(knots) - degree - 1
    x = np.asarray(x, float)
    cols = []
    for j in range(ncol - 1 if drop_last else ncol):
        c = np.zeros(ncol)
        c[j] = 1.0
        cols.append(BSpline(knots, c, degree, extrapolate=True)(x))
    return np.column_stack(cols)


# ==================================================================== a fitted model's output

@dataclass
class Estimates:
    """One fitted object: its coefficients, standard errors, and the stats its table needs."""
    model: str
    label: str
    terms: list
    coef: dict
    se: dict
    stats: dict = field(default_factory=dict)
    params: object = None


# ============================================================= 3.1  censored regression

class Tobit(GenericLikelihoodModel):
    """Censored (Tobit) regression by maximum likelihood -- a general estimator.

        y* = x'b + e,   e ~ N(0, s^2),   y = clip(y*, left, right)

    Nothing here is specific to any dataset: give it a frame, an outcome, a design, and
    where the outcome is censored.  `left=None` / `right=None` switch off that side, so
    left-censored, right-censored, two-sided and uncensored fits all come from one class.

        m   = Tobit.from_frame(data, y="wage", x=["educ", "exper"], left=0.0)
        res = m.fit()
        m.summary_frame(res)      # coef, se, t, dE/dx
        m.predict(res.params)     # E[y|x], the censored mean

    Parameters are [b_0, ..., b_{k-1}, log sigma]; searching LOG sigma keeps the scale
    positive without imposing a constraint.
    """

    def __init__(self, endog, exog, left=0.0, right=None, **kw):
        self.left, self.right = left, right
        # naming the extra parameter stops statsmodels counting log sigma as a regressor
        super().__init__(endog, exog, extra_params_names=["log_sigma"], **kw)

    @classmethod
    def from_frame(cls, data, y, x, left=0.0, right=None, add_const=True):
        """Build from a DataFrame: `y` one column name, `x` a list of them.

        Rows missing any of those columns are dropped.  The column names are kept on the
        model so `.summary()` and the frames below can label their output.
        """
        x = list(x)
        d = data.dropna(subset=[y] + x)
        exog = [np.ones(len(d))] if add_const else []
        exog += [d[v].to_numpy(float) for v in x]
        model = cls(d[y].to_numpy(float), np.column_stack(exog), left=left, right=right)
        model.term_names = (["const"] if add_const else []) + x
        model.data.xnames = model.term_names + ["log_sigma"]
        return model

    def censored(self):
        """Boolean mask of the rows sitting on a censoring point."""
        n = len(self.endog)
        lo = self.endog <= self.left if self.left is not None else np.zeros(n, bool)
        hi = self.endog >= self.right if self.right is not None else np.zeros(n, bool)
        return lo | hi

    def nloglikeobs(self, params):
        """MINUS the log-likelihood of each row -- censored rows contribute a PROBABILITY,
        uncensored rows a DENSITY, and that difference is the whole of "Tobit"."""
        b, s = params[:-1], np.exp(params[-1])
        xb = self.exog @ b
        ll = norm.logpdf((self.endog - xb) / s) - np.log(s)                  # uncensored
        if self.left is not None:
            ll = np.where(self.endog <= self.left,
                          norm.logcdf((self.left - xb) / s), ll)             # Pr(y* <= left)
        if self.right is not None:
            ll = np.where(self.endog >= self.right,
                          norm.logcdf((xb - self.right) / s), ll)            # Pr(y* >= right)
        return -ll

    # ---- fitting ----------------------------------------------------------------------
    def start_params_ols(self):
        """OLS on the observed y, with log sigma from its residual spread."""
        k = self.exog.shape[1]
        b, *_ = np.linalg.lstsq(self.exog, self.endog, rcond=None)
        return np.append(b, np.log(max((self.endog - self.exog @ b).std(ddof=k), 1e-3)))

    def fit(self, start_params=None, maxiter=5000, disp=0, **kw):
        """Maximum likelihood, in two stages.

        newton alone lands on the optimum from an OLS start but returns NaN from a poorer
        one; bfgs alone stops ~1e-4 short.  Together they converge from a wide range of
        starts, to ~1e-8, without a convergence warning.
        """
        if start_params is None:
            start_params = self.start_params_ols()
        res = super().fit(start_params=start_params, method="bfgs",
                          maxiter=maxiter, disp=disp, **kw)
        res = super().fit(start_params=res.params, method="newton",
                          maxiter=maxiter, disp=disp, **kw)
        if not np.isfinite(res.params).all():
            raise ValueError("Tobit did not converge -- the optimiser returned NaN. Check "
                             "that the design is not collinear and that `left`/`right` "
                             "actually bracket the observed outcome.")
        return res

    # ---- reading the fit --------------------------------------------------------------
    def sigma(self, params):
        """The residual scale s (the fit searches log s)."""
        return float(np.exp(params[-1]))

    def coef(self, res):
        """Coefficients as a labelled Series, log sigma excluded."""
        return pd.Series(res.params[:self.exog.shape[1]], index=self.term_names)

    def predict(self, params, exog=None):
        """E[y | x] for the CENSORED outcome -- the censored mean, NOT x'b.

        Evaluated at `exog` if supplied, else at the fitted design.  This is what lets a
        Tobit predict where the outcome was never observed uncensored; OLS on the observed
        y cannot.
        """
        b, s = params[:-1], np.exp(params[-1])
        X = self.exog if exog is None else np.asarray(exog, float)
        return expected_y(X @ b, s, left=self.left, right=self.right)

    def margeff(self, params, scale=1.0, at=None):
        """dE[y]/dx, the UNCONDITIONAL marginal effect: Pr(uncensored) * b_k, times `scale`.

        `at` is the design row to evaluate at (default: the column means).  `scale` converts
        the result into the caller's reporting units -- 1 for a plain regression.

        Hand-written because `get_margeff` does not exist for a GenericLikelihoodModel, and
        the version shipped for discrete models returns the latent-index effect (just b_k).
        Wooldridge (2010), eq. 17.19.
        """
        b, s = params[:-1], np.exp(params[-1])
        xb = float((self.exog.mean(0) if at is None else np.asarray(at, float)) @ b)
        lo = norm.cdf((self.left - xb) / s) if self.left is not None else 0.0
        hi = norm.cdf((self.right - xb) / s) if self.right is not None else 1.0
        return pd.Series((hi - lo) * b * scale, index=self.term_names)

    def summary_frame(self, res, labels=None, scale=1.0):
        """The estimates as a table: coefficient, standard error, t, and dE/dx.

        `labels` renames rows for display; `scale` is handed to `margeff`.  The constant
        gets no marginal-effect entry and is moved to the bottom.
        """
        k = self.exog.shape[1]
        out = pd.DataFrame({"coef": res.params[:k], "se": res.bse[:k],
                            "t": res.params[:k] / res.bse[:k],
                            "dE/dx": self.margeff(res.params, scale=scale).to_numpy()},
                           index=self.term_names)
        if "const" in out.index:
            out.loc["const", "dE/dx"] = np.nan
            out = out.reindex([t for t in self.term_names if t != "const"] + ["const"])
        if labels:
            out.index = [labels.get(t, t) for t in out.index]
        return out


def expected_y(xb, s, left=0.0, right=None):
    """E[y] for y = clip(x'b + e, left, right), e ~ N(0, s^2) -- Wooldridge (2010) eq. 17.14.

    NOT x'b: it mixes the probability mass piled on each censoring point with the density
    between them.  `Tobit.predict` is this evaluated at a fitted model; it is a free
    function as well because a caller may hold a coefficient vector with no model attached
    -- e.g. a PUBLISHED estimate.
    """
    xb = np.asarray(xb, float)
    a = (left - xb) / s if left is not None else np.full(xb.shape, -np.inf)
    b = (right - xb) / s if right is not None else np.full(xb.shape, np.inf)
    Fa, Fb = norm.cdf(a), norm.cdf(b)
    fa = norm.pdf(a) if left is not None else 0.0
    fb = norm.pdf(b) if right is not None else 0.0
    ey = xb * (Fb - Fa) + s * (fa - fb)
    if left is not None:
        ey = ey + left * Fa
    if right is not None:
        ey = ey + right * (1.0 - Fb)
    return ey


# ================================================================== 3.2  panel VAR(p)

class PanelVAR:
    """Panel VAR(p) on CALENDAR lags -- a general estimator.

        y_{k,t+1} = a_k + sum_j sum_{l=0}^{p-1} rho_{kjl} x_{j,t-l} + u_{k,t+1}

    One shared right-hand side, one OLS equation per target; only the left-hand side
    changes.  Nothing here is specific to any dataset -- give it a frame, the column
    identifying the unit, the column giving the period, the targets and the regressors.

        m   = PanelVAR(df, unit="id", time="qtr",
                       targets=["a", "b"], regressors=["a", "b", "z"], nlags=4)
        res = m.fit()
        m.design_names()

    A lag is a MERGE on (unit, period - k), never a positional shift, so a unit with a gap
    in its history yields NaN rather than borrowing the previous row's value.  `time` must
    therefore be an integer period that advances by 1; pass `period=` to convert a column
    that is not already in that form.
    """

    def __init__(self, data, unit, time, targets, regressors=None, nlags=4, period=None):
        self.data, self.unit, self.nlags = data, unit, nlags
        self.targets = list(targets)
        self.regressors = list(regressors) if regressors is not None else list(self.targets)
        dupes = {v for v in self.regressors if self.regressors.count(v) > 1}
        if dupes:                      # a repeated regressor would silently give a singular design
            raise ValueError(f"repeated regressor(s): {sorted(dupes)}")
        seq = data[time].to_numpy()
        self.seq = np.asarray(period(seq) if period is not None else seq).astype("int64")
        self._pos = {(u, s): i for i, (u, s) in
                     enumerate(zip(data[unit].to_numpy(), self.seq))}

    def gather(self, values, offset):
        """`values` at (same unit, period + offset); NaN where that row does not exist."""
        units, out = self.data[self.unit].to_numpy(), np.full(len(self.seq), np.nan)
        for i in range(len(out)):
            j = self._pos.get((units[i], self.seq[i] + offset))
            if j is not None:
                out[i] = values[j]
        return out

    def design_names(self):
        """The shared right-hand side's column names -- regressor-major, lag-minor, const last."""
        return [f"{v}_L{k}" for v in self.regressors for k in range(self.nlags)] + ["const"]

    def design(self) -> pd.DataFrame:
        """The shared right-hand side: `nlags` calendar lags of every regressor, plus a constant.

        L0 is the current period, L1..L{p-1} the calendar lags.
        """
        cols = {f"{v}_L{k}": self.gather(self.data[v].to_numpy(float), -k)
                for v in self.regressors for k in range(self.nlags)}
        cols["const"] = np.ones(len(self.data))
        return pd.DataFrame(cols)

    def lead(self, var) -> np.ndarray:
        """Next period's `var` for the same unit -- the left-hand side."""
        return self.gather(self.data[var].to_numpy(float), +1)

    @staticmethod
    def ols(y, X):
        """Coefficients, classical standard errors, R^2 and n -- statsmodels does the algebra.

        The lag construction above is what no package provides; the regression itself is
        ordinary OLS, so there is no reason to hand-roll it.
        """
        r = sm.OLS(y, X).fit()
        return r.params, r.bse, r.rsquared, int(r.nobs)

    def fit_equation(self, target, X=None) -> Estimates:
        """One equation: OLS of lead(target) on the shared design, over complete rows.

        `terms` reports the OWN-lag block as ar1..ar{p} (ar1 is the CURRENT period, since
        the left-hand side is one period ahead) plus the constant; the full coefficient
        vector over every design column is kept in `stats["coef_full"]`.
        """
        X = self.design() if X is None else X
        Xa, y = X.to_numpy(dtype=float), np.asarray(self.lead(target), dtype=float)
        # listwise-complete rows only, and the SAME rows in every equation: the lead is NaN
        # at a unit's last period and the lags are NaN wherever its history has a gap.
        ok = np.isfinite(y) & np.isfinite(Xa).all(axis=1)
        beta, se, r2, n = self.ols(y[ok], Xa[ok])

        pos = [X.columns.get_loc(f"{target}_L{k}") for k in range(self.nlags)]
        terms = [f"ar{k}" for k in range(1, self.nlags + 1)] + ["const"]
        coef = {f"ar{k}": float(beta[pos[k - 1]]) for k in range(1, self.nlags + 1)}
        se_d = {f"ar{k}": float(se[pos[k - 1]]) for k in range(1, self.nlags + 1)}
        coef["const"], se_d["const"] = float(beta[-1]), float(se[-1])
        return Estimates(model=f"var{self.nlags}_transition", label=target, terms=terms,
                         coef=coef, se=se_d,
                         stats=dict(r2=float(r2), n=int(n),
                                    coef_full=pd.Series(beta, index=X.columns)))

    def fit(self) -> dict:
        """Every equation, sharing one design matrix.  Returns {target: Estimates}."""
        X = self.design()
        return {v: self.fit_equation(v, X=X) for v in self.targets}

    def summary_frame(self, results, labels=None):
        """The estimates as a table: one row per equation, AR1..AR{p} plus const, R2 and n.

        AR1 is the CURRENT period, since the left-hand side is one period ahead.  `labels`
        renames rows for display.
        """
        rows = {}
        for v, r in results.items():
            rows[(labels or {}).get(v, v)] = (
                [r.coef[f"ar{k}"] for k in range(1, self.nlags + 1)]
                + [r.coef["const"], r.stats["r2"], r.stats["n"]])
        return pd.DataFrame.from_dict(
            rows, orient="index",
            columns=[f"AR{k}" for k in range(1, self.nlags + 1)] + ["const", "R2", "n"])


# ============================================================= 3.5  GMM, moment linear in c

def gmm_fit(C, d, Z, maxiter=2):
    """Two-step GMM for a moment that is LINEAR in the parameters.

        g_i(c) = C_i'c + d_i,        E[Z_i g_i] = 0

    A linear-in-parameters moment IS a linear IV problem: put y = -d and X = C, and then
    E[Z(y - Xc)] = -E[Zg], so the two moment conditions are the same and statsmodels'
    `LinearIVGMM` estimates it directly.  There is no optimiser and no start value: because
    g is linear in c, the criterion is quadratic and each step is one weighted-IV solve

        c_hat = -[(Z'C)' W^-1 (Z'C)]^-1 (Z'C)' W^-1 (Z'd)

    evaluated first at W = Z'Z/N and then at the efficient W = Z'gg'Z/N.  Note this holds
    for EVERY element of c -- in the KLW moment that includes the T1EV scale sigma, which a
    first reading of sigma*log(p0/p1) suggests has to be searched over.  It does not.

    Returns the statsmodels results object, so `.params`, `.bse`, `.jtest()`, `.summary()`
    and `.cov_params()` are all available.  `.jval` IS the J statistic (do not scale by N).

    Two facts that are easy to get wrong here:

    * **`centered=False` is required, not cosmetic.**  The efficient weight is the
      UNCENTERED second moment Z'gg'Z/N, which is the right one under the null E[Zg] = 0.
      statsmodels centers by default; on the KLW moment that leaves the point estimates
      1.1e-04 off and inflates the standard errors by up to 19%.
    * **`maxiter='cue'` does not work.**  It terminates after one iteration and returns the
      one-step estimate.  Use two-step (the default here) or an explicit iteration count.
    """
    C = np.asarray(C, float)
    return LinearIVGMM(-np.asarray(d, float), C, np.asarray(Z, float)).fit(
        maxiter=maxiter, wargs=dict(centered=False))


# ================================================== 5  the simulated-panel toolkit
#
# Kang-Lowery-Wardlaw (2015) eq. 6.  Closure d=1 is terminal/absorbing and p1 is the closure CCP:
#     sigma * log(p0/p1) = c(x) + beta * E[ -sigma*ln p1' - c'(x') | x, continue ]
#     c(x) = MC(x) - x_tilde' theta_nmc                                 (paper convention)
# On the SIMULATED panels MC is a LEVEL in $thousands -- `monetary_cost` sums b_k * term_k(x)
# and there is no `assets` column.  (On the real panel of §2 the Tobit fits the RATIO
# cost/assets, so MC there is expected_y(x'b, s) * assets; that rescale lives in the notebook.)

FIN_VARS = ["log_assets", "equity_a", "npf_a", "roa", "realest_a"]
TREND_STEP = 1.0 / 27.0            # one-quarter advance of the normalized time trend

# Rich, fixed first-stage CCP basis (a flexible reduced form; correctly specified for the DGP).
CCP_BASIS = ["const", "log_assets", "log_assets^2", "log_assets^3", "equity_a", "npf_a", "roa",
             "realest_a", "unemp", "house", "senate", "trend", "house*log_assets",
             "npf_a*log_assets", "equity_a*log_assets", "house*npf_a"]

# The paper's two specifications: the non-monetary-cost index x_tilde, and the monetary cost.
PAPER_SPEC    = ["const", "log_assets", "log_assets^2", "npf_a", "roa", "realest_a",
                 "house", "senate"]
MC_SPEC_PAPER = ["const", "log_assets", "equity_a", "npf_a", "realest_a"]

# The three fixed counterfactual scenarios (identical across datasets).
SCENARIOS = ["no_political", "myopic", "npl_stress"]
_POLITICAL_TOKENS = ("house", "senate")     # terms treated as "political" in `no_political`
NPL_STRESS_SHIFT = 0.02                      # +2pp to npf_a in `npl_stress`


# ---- the design-matrix term grammar: "const", raw columns, "a^2" (integer powers), "a*b" ----
def _as_arrays(data):
    if isinstance(data, pd.DataFrame):
        return {c: np.asarray(data[c].values, dtype=float) for c in data.columns}
    return data


def _term(arr, name):
    name = name.strip()
    if name == "const":
        return np.ones(len(next(iter(arr.values()))))
    if "*" in name:
        a, b = name.split("*", 1)
        return _term(arr, a) * _term(arr, b)
    if "^" in name:
        base, p = name.split("^", 1)
        return _term(arr, base) ** int(p)
    return arr[name]


def build_design(data, spec):
    """Return an (N, len(spec)) design matrix for a DataFrame or {col: array} mapping."""
    arr = _as_arrays(data)
    return np.column_stack([_term(arr, t) for t in spec])


# ---- transitions: continuous AR(1)s plus an optional joint political Markov chain ----
def cond_mean(arr, v, trans):
    """E[v' | x] under the AR(1) transition for state variable v."""
    c, rho, _ = trans[v]
    return c + rho * arr[v]


def make_innovation_draws(trans, R, seed):
    """R shared innovation draws for every stochastic transition."""
    rng = default_rng(seed)
    d = {v: rng.normal(0.0, np.sqrt(max(trans[v][2], 0.0)), R) for v in FIN_VARS}
    d["unemp"] = rng.normal(0.0, np.sqrt(max(trans["unemp"][2], 0.0)), R)
    if "political_joint" in trans:
        d["__political_u"] = rng.random(R)
    return d


def next_state_arr(arr, eps_r, trans, bounds):
    """One draw of next-period states: AR(1) mean + innovation (clipped), trend advances,
    an optional joint Markov transition advances House/Senate, and decoys are carried."""
    out = {}
    for v in FIN_VARS:
        lo, hi = bounds[v]
        out[v] = np.clip(cond_mean(arr, v, trans) + eps_r[v], lo, hi)
    lo, hi = bounds["unemp"]
    out["unemp"] = np.clip(cond_mean(arr, "unemp", trans) + eps_r["unemp"], lo, hi)
    out["trend"] = np.clip(arr["trend"] + TREND_STEP, 0.0, 1.0)
    if "political_joint" in trans:
        political = np.asarray(trans["political_joint"], float)
        if political.shape != (12, 12):
            raise ValueError("political_joint must be a 12-by-12 transition matrix")
        current = arr["house"].astype(int) * 3 + arr["senate"].astype(int)
        destination = np.empty(len(current), dtype=int)
        draw = float(eps_r["__political_u"])
        cdf = np.cumsum(political, axis=1)
        for state in range(12):
            destination[current == state] = np.searchsorted(
                cdf[state], draw, side="right"
            )
        destination = np.minimum(destination, 11)
        out["house"] = (destination // 3).astype(float)
        out["senate"] = (destination % 3).astype(float)
    for c in arr:
        if c not in out:
            out[c] = arr[c]
    return out


def expected_design(arr, spec, trans):
    """E[design(x') | x], closed form. Linear terms -> E[x']; squares -> (E[x'])^2 + Var;
    interactions -> product of expectations; trend advances; political/decoys carried."""
    Em, Vv = {}, {}
    for v in FIN_VARS + ["unemp"]:
        Em[v] = cond_mean(arr, v, trans)
        Vv[v] = trans[v][2]
    Em["trend"] = np.clip(arr["trend"] + TREND_STEP, 0.0, 1.0)
    Vv["trend"] = 0.0
    if "political_joint" in trans:
        political = np.asarray(trans["political_joint"], float)
        if political.shape != (12, 12):
            raise ValueError("political_joint must be a 12-by-12 transition matrix")
        destination = np.arange(12)
        expected_house = political @ (destination // 3)
        expected_senate = political @ (destination % 3)
        current = arr["house"].astype(int) * 3 + arr["senate"].astype(int)
        Em["house"] = expected_house[current]
        Em["senate"] = expected_senate[current]
        Vv["house"] = 0.0
        Vv["senate"] = 0.0
    for c in arr:
        if c not in Em:
            Em[c] = arr[c]
            Vv[c] = 0.0
    cols = []
    for t in spec:
        t = t.strip()
        if t == "const":
            cols.append(np.ones(len(arr["log_assets"])))
        elif "^" in t:
            base, p = t.split("^", 1)
            p = int(p)
            # why: E[(v')^2] = (E v')^2 + Var(v') -- evaluating a square at the conditional
            # MEAN alone understates the second moment.
            cols.append(Em[base] ** 2 + Vv[base] if p == 2 else Em[base] ** p)
        elif "*" in t:
            a, b = t.split("*", 1)
            cols.append(Em[a.strip()] * Em[b.strip()])
        else:
            cols.append(Em[t])
    return np.column_stack(cols)


# ---- clip bounds for the forward draws ----
def _bounds_from(arr, margin=0.05):
    """Per-variable clip bounds derived from the data (no hard-coded DGP knowledge)."""
    b = {}
    for v in FIN_VARS + ["unemp"]:
        lo, hi = float(np.min(arr[v])), float(np.max(arr[v]))
        pad = margin * (hi - lo + 1e-9)
        b[v] = (lo - pad, hi + pad)
    return b


def state_bounds(data, margin=0.05):
    """Per-variable clip bounds from the panel (used by forward simulation)."""
    return _bounds_from(_as_arrays(data), margin)


# ---- the cost function: c(x) = MC(x) - x_tilde' theta_nmc  (paper convention) ----
def monetary_cost(arr, cost_coef, var=None):
    """Monetary cost MC(x) = sum_k b_k * term_k(x), over whatever keys `cost_coef` carries.

    Any key is evaluated through `build_design`'s term grammar, so a richer cost function
    (e.g. `unemp`, `npf_a^2`) needs no change here -- just a longer coefficient dict.

    `var` maps a base variable to Var(v'|x) and applies ONLY to squared terms.  Pass it when
    evaluating E[MC'|x] at conditional MEANS, because E[(v')^2] = (E[v'])^2 + Var(v'); omitting
    it understates the second moment.  For MC at a realised or simulated state pass var=None,
    or the variance is double-counted.
    """
    arr = _as_arrays(arr)
    out = float(cost_coef["const"]) * np.ones(len(next(iter(arr.values()))))
    for k, b in cost_coef.items():
        if k == "const":
            continue
        out = out + b * _term(arr, k)
        if var and "^2" in k:
            base = k.split("^")[0]
            if base in var:
                out = out + b * var[base]
    return out


def make_cost_fn(cost_coef, theta_nmc, nmc_spec):
    """Return c(arr) = MC(arr) - x_tilde' theta_nmc  (paper convention)."""
    tv = np.array([theta_nmc[t] for t in nmc_spec], dtype=float)

    def cost_fn(arr):
        arr = _as_arrays(arr)
        return monetary_cost(arr, cost_coef) - build_design(arr, nmc_spec) @ tv

    return cost_fn


@dataclass
class Result:
    """One fitted structural model: it remembers its own panel and fitted objects, so the
    counterfactual engine never has to guess which data or which spec produced it."""
    beta: float
    sigma: float
    theta: dict            # NMC coefficients keyed by spec term
    cost_coef: dict        # monetary-cost coefficients (const, log_assets, equity_a, ...)
    trans: dict            # AR(1) transition params
    spec: list             # the NMC specification used
    data: pd.DataFrame     # the panel this Result was fit on
    diagnostics: dict = field(default_factory=dict)

    def cost_fn(self):
        return make_cost_fn(self.cost_coef, self.theta, self.spec)


def reference_model(name="replication", data_dir=None, beta=0.9582):
    """A fitted `Result` for one simulated panel, ready to hand to `counterfactual`.

    Re-solving the model needs all four of its primitives at once, so this assembles them:

        cost_coef   OLS of `estimated_cost` on MC_SPEC_PAPER over the resolved banks
        trans       one AR(1) per financial state variable plus one for aggregate
                    unemployment, each stored as (const, rho, innovation variance).
                    The variance is not optional -- the solver draws next states from it.
        sigma,theta the published values for this panel, read from refcheck.json

    This is plumbing, not estimation: it exists so a counterfactual can be demonstrated in
    one line.  Estimating sigma and theta from scratch is what the notebook's second stage does.
    """
    d = _data_dir(data_dir)
    panel = load_sim_data(name, d)

    resolved = panel[panel.failed == 1].dropna(subset=["estimated_cost"])
    b, *_ = np.linalg.lstsq(build_design(resolved, MC_SPEC_PAPER),
                            resolved.estimated_cost.values, rcond=None)
    cost_coef = dict(zip(MC_SPEC_PAPER, b))

    srt, trans = panel.sort_values(["rssd_id", "yearq"]), {}
    for v in FIN_VARS:
        lag = srt.groupby("rssd_id")[v].shift(1)
        m = lag.notna().values
        X = np.column_stack([np.ones(m.sum()), lag.values[m]])
        c, *_ = np.linalg.lstsq(X, srt[v].values[m], rcond=None)
        trans[v] = (c[0], c[1], float((srt[v].values[m] - X @ c).var()))
    u = panel.groupby("yearq")["unemp"].mean().sort_index().values
    Xu = np.column_stack([np.ones(len(u) - 1), u[:-1]])
    bu, *_ = np.linalg.lstsq(Xu, u[1:], rcond=None)
    trans["unemp"] = (bu[0], bu[1], float((u[1:] - Xu @ bu).var()))

    with open(os.path.join(d, "refcheck.json")) as fh:
        ref = json.load(fh)["structural"]

    return Result(beta=beta, sigma=ref["sigma"], theta=ref["theta"], cost_coef=cost_coef,
                  trans=trans, spec=list(PAPER_SPEC), data=panel)


# ---- the CCP fixed point ----
def _ccp_precompute(anchor, cost_fn, trans, bounds, basis, R, seed):
    """The pieces of the fixed point that do not move between iterations: the anchor design,
    the R next-state designs, and the Monte-Carlo E[c'|x]."""
    eps = make_innovation_draws(trans, R, seed + 1)
    B = build_design(anchor, basis)
    Na, p = B.shape
    c_now = cost_fn(anchor)
    # why: these are fixed across iterations -- only gamma moves -- so paying for them once
    # turns a 150-iteration solve into 150 tensordots.
    Bnext = np.empty((R, Na, p))
    Ec_next = np.zeros(Na)
    for k in range(R):
        nxt = next_state_arr(anchor, {v: eps[v][k] for v in eps}, trans, bounds)
        Bnext[k] = build_design(nxt, basis)
        Ec_next += cost_fn(nxt)
    Ec_next /= R
    return B, Bnext, c_now, Ec_next


def solve_ccp_gamma(states, cost_fn, beta, sigma, trans, *, R=400, n_anchor=8000, seed=0,
                    tol=1e-7, max_iter=500, basis=None):
    """Solve the KLW eq.-6 fixed point: the closure CCP consistent with the continuation value
    it itself generates.  Iterate log-odds(p1) = -(c + beta*E[V'])/sigma, projecting each
    Bellman target onto `basis`, until the index stops moving.  Returns (gamma, p1_on_states)
    with p1(x) = expit(basis(x) @ gamma).  This re-solve is what makes a counterfactual a
    counterfactual: change a primitive and closure decisions feed back into continuation
    values, which change closure decisions again.

    `basis` defaults to CCP_BASIS and must SPAN the true CCP: the routine PROJECTS onto it, so
    any feature of c(x) the basis cannot represent is silently dropped.  A cost function
    carrying `npf_a^2` needs it in the basis (regressing npf_a^2 on the default CCP_BASIS
    gives R^2 = 0.878 -- a real hole)."""
    basis = CCP_BASIS if basis is None else list(basis)
    arr = _as_arrays(states)
    N = len(arr["log_assets"])
    bounds = _bounds_from(arr)
    rng = default_rng(seed)
    sub = rng.choice(N, min(n_anchor, N), replace=False) if N > n_anchor else np.arange(N)
    anchor = {k: v[sub] for k, v in arr.items()}
    B, Bnext, c_now, Ec_next = _ccp_precompute(anchor, cost_fn, trans, bounds, basis, R, seed)

    gamma = np.zeros(B.shape[1])
    idx_prev = B @ gamma
    converged = False
    final_difference = np.inf
    for _ in range(max_iter):
        idx = np.tensordot(Bnext, gamma, axes=([2], [0]))         # (R, Na)
        # why -logaddexp(0,-idx): ln expit(idx) computed without ever forming expit, so a
        # deeply negative index gives -|idx| instead of log(0.0).
        Eln = (-np.logaddexp(0.0, -idx)).mean(0)                  # E[ln p1'|x]
        F = -sigma * Eln - Ec_next                                # E[-sigma ln p1' - c' | x]
        target = -(c_now + beta * F) / sigma                      # log-odds(p1)
        g_new, *_ = np.linalg.lstsq(B, target, rcond=None)
        idx_new = B @ g_new
        # why the INDEX and not the coefficients: the index drives p1 and reaches tolerance
        # well within max_iter, whereas a near-collinear coefficient mode converges slowly.
        final_difference = float(np.max(np.abs(idx_new - idx_prev)))
        if final_difference < tol:
            gamma = g_new
            converged = True
            break
        gamma, idx_prev = g_new, idx_new
    if not converged:
        raise RuntimeError(
            "CCP fixed point did not converge: "
            f"max index change={final_difference:.3e} after {max_iter} iterations"
        )
    return gamma, expit(build_design(arr, basis) @ gamma)


def solve_ccp(states, cost_fn, beta, sigma, trans, **kw):
    """Return p1 (closure CCP) on `states` for (cost_fn, beta, sigma, trans)."""
    return solve_ccp_gamma(states, cost_fn, beta, sigma, trans, **kw)[1]


def _zero_political(theta, spec):
    """theta with every political term (house, senate) set to zero -- the `no_political` primitive."""
    out = dict(theta)
    for t in spec:
        if any(tok in t for tok in _POLITICAL_TOKENS):
            out[t] = 0.0
    return out


def counterfactual(result: Result, scenario: str, *, seed=0, R=300, n_anchor=6000, basis=None):
    """Change one primitive of a fitted model, re-solve the CCP fixed point, and report
    Delta% = 100*(sum p1_cf - sum p1_base)/sum p1_base over the panel.

    The three primitives: `no_political` zeroes the political terms of the non-monetary cost,
    `myopic` sets beta -> 0 (the regulator stops valuing continuation), `npl_stress` adds 2pp
    to every bank's NPL ratio.  Each is a change to the environment, NOT a re-evaluation of
    the old policy under new numbers -- the whole CCP is re-solved.

    `basis` is forwarded to the solver and must match the one the panel was generated on.
    The unmodified base solve is shared across the 3 scenarios; it is memoized ON the Result
    object (lifetime-tied, so it can never alias a different Result via a recycled id())."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    arr = _as_arrays(result.data)
    base_cost = result.cost_fn()
    ck = (seed, R, n_anchor, tuple(basis) if basis else None)
    cache = result.diagnostics.setdefault("_p1_base", {})
    if ck in cache:
        p1_base = cache[ck]
    else:
        p1_base = solve_ccp(arr, base_cost, result.beta, result.sigma, result.trans,
                            R=R, n_anchor=n_anchor, seed=seed, basis=basis)
        cache[ck] = p1_base

    if scenario == "no_political":
        cf_cost = make_cost_fn(result.cost_coef, _zero_political(result.theta, result.spec),
                               result.spec)
        p1_cf = solve_ccp(arr, cf_cost, result.beta, result.sigma, result.trans,
                          R=R, n_anchor=n_anchor, seed=seed, basis=basis)
    elif scenario == "myopic":
        p1_cf = solve_ccp(arr, base_cost, 0.0, result.sigma, result.trans,
                          R=R, n_anchor=n_anchor, seed=seed, basis=basis)
    else:  # npl_stress
        shifted = dict(arr)
        shifted["npf_a"] = arr["npf_a"] + NPL_STRESS_SHIFT
        p1_cf = solve_ccp(shifted, base_cost, result.beta, result.sigma, result.trans,
                          R=R, n_anchor=n_anchor, seed=seed, basis=basis)

    return 100.0 * (p1_cf.sum() - p1_base.sum()) / p1_base.sum()


# ---- reporting and the Kaggle submission ----
def as_series(result: Result) -> pd.Series:
    """sigma and every theta term of a fitted Result, as one labelled Series.

    Useful for putting two fits side by side:
        pd.DataFrame({"unrevised": as_series(a), "revised": as_series(b)})
    """
    d = {"sigma": float(result.sigma)}
    d.update({t: float(result.theta[t]) for t in result.spec})
    return pd.Series(d)


def submission_rows(dataset, sigma, theta, spec, cfs, cost_coef=None):
    """The structured (Id, Prediction) rows for one dataset: the estimated scale sigma, each NMC
    coefficient, each MONETARY-cost coefficient, and the three counterfactual outcomes. The same
    Id scheme is used by the private solution generator, so Kaggle can compare every parameter
    and every counterfactual by itself.

    `cost_coef` is optional so older callers keep working; pass it and the `{dataset}_mc_*` rows
    appear.  They are worth grading: the S1/S2 revisions ARE the cost specification, and the MC
    block is the best-identified part of the model (reference error <5% on 17 of 20 cells,
    against 21-49% on the near-collinear size terms of the NMC index)."""
    rows = [(f"{dataset}_sigma", float(sigma))]
    for t in spec:
        rows.append((f"{dataset}_theta_{t}", float(theta[t])))
    for k in (cost_coef or {}):
        rows.append((f"{dataset}_mc_{k}", float(cost_coef[k])))
    for s in SCENARIOS:                       # absent for panels we do not grade CFs on
        if s in cfs:
            rows.append((f"{dataset}_cf_{s}", float(cfs[s])))
    return rows


def make_submission(res_dict, grid, path="submission.csv"):
    """Write a KAGGLE-format submission (columns: Id, Prediction) and save it automatically.

    One row per estimated parameter (sigma + each NMC coefficient + each monetary-cost
    coefficient) for every panel in `res_dict`, plus one row per counterfactual present in
    `grid`, so Kaggle scores each item on its own.  The three revision panels carry DIFFERENT
    numbers of parameters (S2/S3's cost function has the extra `npf_a^2` term, S3's index the
    extra `trend`) and the counterfactuals are run on S3 only, so the row count is not a
    multiple of the panel count.  beta is calibrated, so it is never submitted.
    """
    rows = []
    for d in sorted(res_dict):
        r = res_dict[d]
        cfs = {s: float(grid[(d, s)]) for s in SCENARIOS if (d, s) in grid}
        rows += submission_rows(d, r.sigma, r.theta, r.spec, cfs, r.cost_coef)
    out = pd.DataFrame(rows, columns=["Id", "Prediction"])
    out.to_csv(path, index=False)
    n_cf = sum(1 for i in out["Id"] if "_cf_" in i)
    _p = os.path.join(*(["..."] + os.path.abspath(path).split(os.sep)[-2:]))
    print(f"✅ Kaggle submission saved to '{_p}' ({len(out)} rows: "
          f"{len(out)-n_cf} parameters + {n_cf} counterfactuals). Upload this file to Kaggle.")
    return out


# ================================================================================ self-tests

def _selftest():
    """Every check runs on synthetic data — no bundle, no network, ~3 s."""
    # ---- constants the notebook and the private scorer both depend on --------------------
    assert SCENARIOS == ["no_political", "myopic", "npl_stress"]      # submission row order
    assert NPL_STRESS_SHIFT == 0.02 and TREND_STEP == 1.0 / 27.0
    assert len(CCP_BASIS) == 16 and CCP_BASIS[0] == "const"

    # ---- the term grammar and the variance correction ------------------------------------
    assert np.allclose(build_design({"x": np.array([2., 3.])}, ["const", "x", "x^2", "x*x"]),
                       [[1., 2., 4., 4.], [1., 3., 9., 9.]])
    assert monetary_cost({"npf_a": np.array([2.])}, {"const": 0., "npf_a^2": 1.},
                         var={"npf_a": 4.})[0] == 8.   # E[(npf')^2] = (E npf')^2 + Var(npf')
    assert [i for i, _ in submission_rows("S3", 1., {"const": 2.}, ["const"],
                                          {"myopic": 3.})] \
           == ["S3_sigma", "S3_theta_const", "S3_cf_myopic"]
    assert [i for i, _ in submission_rows("S3", 1., {"const": 2.}, ["const"], {"myopic": 3.},
                                          {"const": 4., "npf_a^2": 5.})] \
           == ["S3_sigma", "S3_theta_const", "S3_mc_const", "S3_mc_npf_a^2", "S3_cf_myopic"]

    # ---- the optional political chain advances House/Senate jointly --------------------
    political = np.eye(12)
    political[0] = 0.0
    political[0, 11] = 1.0                 # state (House=0, Senate=0) -> (3, 2)
    trans = {v: (0.0, 1.0, 0.0) for v in FIN_VARS + ["unemp"]}
    trans["political_joint"] = political
    state = {v: np.array([0.1]) for v in FIN_VARS + ["unemp"]}
    state.update({"house": np.array([0.0]), "senate": np.array([0.0]),
                  "trend": np.array([0.0])})
    bounds = {v: (-1.0, 1.0) for v in FIN_VARS + ["unemp"]}
    shocks = {v: 0.0 for v in FIN_VARS + ["unemp"]}
    shocks["__political_u"] = 0.5
    advanced = next_state_arr(state, shocks, trans, bounds)
    assert advanced["house"][0] == 3.0 and advanced["senate"][0] == 2.0
    assert np.allclose(expected_design(state, ["house", "senate"], trans), [[3.0, 2.0]])

    # ---- the basis partitions unity, inside AND outside the frozen range -----------------
    for deg, inter in ((3, ()), (3, (0.4,)), (2, ())):
        B = bspline_basis(np.array([0., .3, 1., 1.4]), 0., 1., deg, inter, drop_last=False)
        assert B.shape[1] == len(inter) + deg + 1
        assert np.allclose(B.sum(1), 1.0), (deg, inter)

    # ---- Tobit recovers a known DGP in all four censoring regimes ------------------------
    rng = default_rng(0)
    n, b_true, s_true = 4000, np.array([1.0, 2.0, -1.5]), 1.2
    X = rng.normal(size=(n, 2))
    ystar = b_true[0] + X @ b_true[1:] + rng.normal(0, s_true, n)
    d = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1]})
    for tag, lo, hi in (("left", 0.0, None), ("right", None, 2.0),
                        ("two-sided", 0.0, 2.0), ("uncensored", None, None)):
        d["y"] = np.clip(ystar, lo, hi)
        m = Tobit.from_frame(d, y="y", x=["x1", "x2"], left=lo, right=hi)
        res = m.fit()
        assert np.abs(m.coef(res).to_numpy() - b_true).max() < 0.10, (tag, m.coef(res))
        assert abs(m.sigma(res.params) - s_true) < 0.10, (tag, m.sigma(res.params))
    # the censored mean is not the index, and integrates to the observed mean
    d["y"] = np.clip(ystar, 0.0, None)
    m = Tobit.from_frame(d, y="y", x=["x1", "x2"], left=0.0)
    res = m.fit()
    assert abs(m.predict(res.params).mean() - d["y"].mean()) < 0.05

    # ---- PanelVAR: a calendar GAP must yield NaN, not the previous row's value -----------
    rho = np.array([0.6, 0.2])
    rows = []
    for firm in range(60):
        y = [0.0, 0.0]
        for t in range(40):
            y.append(rho[0] * y[-1] + rho[1] * y[-2] + rng.normal(0, 0.3))
            if not (firm == 0 and t == 20):            # firm 0 skips one period
                rows.append((firm, t, y[-1]))
    pan = pd.DataFrame(rows, columns=["firm", "period", "y"])
    pv = PanelVAR(pan, unit="firm", time="period", targets=["y"], nlags=2)
    got = pv.fit()["y"]
    assert abs(got.coef["ar1"] - rho[0]) < 0.05 and abs(got.coef["ar2"] - rho[1]) < 0.05, got.coef
    des = pv.design()
    hole = (pan.firm == 0) & (pan.period == 21)        # its L1 is the period it never reported
    assert des.loc[hole.to_numpy(), "y_L1"].isna().all()
    assert pv.design_names() == ["y_L0", "y_L1", "const"]

    # ---- gmm_fit == the closed-form weighted-IV solve, and exact ID recovers the truth ---
    K, L, N = 3, 5, 2000
    c_true = np.array([1.5, -2.0, 0.7])
    Zs = rng.normal(size=(N, L))
    Cs = Zs[:, :K] * 1.3 + rng.normal(0, 0.4, size=(N, K))
    ds = -(Cs @ c_true) + rng.normal(0, 0.5, N)

    def _solve(W):
        """c_hat = -[(Z'C)'W^-1(Z'C)]^-1 (Z'C)'W^-1(Z'd) — the FOC in closed form."""
        ZC, Zd = Zs.T @ Cs / N, Zs.T @ ds / N
        WiZC = np.linalg.solve(W, ZC)
        return -np.linalg.solve(ZC.T @ WiZC, WiZC.T @ Zd)

    c1 = _solve(Zs.T @ Zs / N)                                   # step 1: W = Z'Z/N
    gZ = Zs * (Cs @ c1 + ds)[:, None]
    c2 = _solve(gZ.T @ gZ / N)                                   # step 2: efficient W
    got = np.asarray(gmm_fit(Cs, ds, Zs).params)
    assert np.abs(got / c2 - 1).max() < 1e-8, (got, c2)
    # exactly identified (Z = C): GMM must reproduce the IV/OLS solution whatever W is
    exact = np.asarray(gmm_fit(Cs, ds, Cs).params)
    ols, *_ = np.linalg.lstsq(Cs, -ds, rcond=None)
    assert np.abs(exact - ols).max() < 1e-8, (exact, ols)

    print("✅ klw.py self-tests passed "
          "(constants, term grammar, B-spline basis, Tobit x4, PanelVAR, gmm_fit)")


if __name__ == "__main__":
    _selftest()
