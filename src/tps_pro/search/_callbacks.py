"""Optuna callback implementations and statistical helpers for early stopping.

Module-level state
------------------
This module is **stateless** at module level.  ``GPStoppingCallback`` and
``ProgressBarUpdateCallback`` are instantiated per-study by the search orchestrator
and hold their state as instance attributes.  The only cross-module
reference is ``ProgressBarUpdateCallback`` reading the progress-bar handle from
``search_display.pbar_state`` (a tiny ``SimpleNamespace``).

Public API:
- GPStoppingCallback — GP-based early-stopping with L-BFGS-B EI maximisation.
- ProgressBarUpdateCallback — progress-bar tick on every trial completion.
- trial_scalar_value — scalar accessor safe for multi-objective studies.

Error strategy (see errors.py for full documentation):
    - GPStoppingCallback: catches LinAlgError/ValueError from GP fitting
      and scipy optimization (logged at debug).  These are expected when
      the GP cannot fit well to noisy data -- the callback simply skips
      the stopping check and lets the study continue.
    - trial_scalar_value / safe_best_value: catch RuntimeError from
      multi-objective studies where .value is undefined.
    - sklearn/scipy ImportError: logged at warning, falls back to
      patience-only stopping.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import optuna

logger = logging.getLogger(__name__)

_GP_EPSILON = 1e-8

__all__ = [
    "GPStoppingCallback",
    "ProgressBarUpdateCallback",
    "_expected_improvement",
    "_encode_param",
    "safe_best_value",
    "trial_scalar_value",
]


def trial_scalar_value(t: optuna.trial.FrozenTrial) -> float | None:
    """Get a trial's scalar value, safe for both single- and multi-objective studies.

    In multi-objective studies, t.value raises RuntimeError — fall back to t.values[0].
    """
    try:
        return t.value
    except RuntimeError:
        return t.values[0] if t.values else None


def _encode_param(
    value: float | int | str,
    distribution: optuna.distributions.BaseDistribution,
) -> float:
    """Encode a parameter value to [0, 1] range."""
    if isinstance(distribution, optuna.distributions.CategoricalDistribution):
        choices = distribution.choices
        idx = choices.index(value) if value in choices else 0
        return idx / max(1, len(choices) - 1)
    elif isinstance(distribution, optuna.distributions.IntDistribution):
        low, high = distribution.low, distribution.high
        return (value - low) / max(1, high - low)
    elif isinstance(distribution, optuna.distributions.FloatDistribution):
        low, high = distribution.low, distribution.high
        return (value - low) / max(1e-8, high - low)
    return 0.5


def _expected_improvement(
    mu: npt.NDArray[np.floating[Any]],
    sigma: npt.NDArray[np.floating[Any]],
    best_y: float,
    xi: float = 0.01,
) -> npt.NDArray[np.floating[Any]]:
    """Compute Expected Improvement. xi is exploration-exploitation tradeoff.

    Requires scipy.stats.norm — imported lazily inside GPStoppingCallback.__call__
    so this helper is only invoked when scipy is already available.
    """
    from scipy.stats import norm

    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        imp = mu - best_y - xi
        Z = np.where(sigma > _GP_EPSILON, imp / sigma, 0.0)  # noqa: N806
        ei = np.where(sigma > _GP_EPSILON, imp * norm.cdf(Z) + sigma * norm.pdf(Z), 0.0)
    return ei


def safe_best_value(study: optuna.Study) -> float | None:
    """Get study.best_value without throwing if no valid trials exist."""
    try:
        return study.best_value
    except ValueError:
        return None


class ProgressBarUpdateCallback:
    """Ensures the progress bar updates even if a trial is pruned or fails."""

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        from ._display import pbar_state

        if pbar_state.current is not None and hasattr(pbar_state.current, "count"):
            pbar_state.current.count += 1


class GPStoppingCallback:
    """Stops optimization when the GP's max EI drops below threshold.

    Uses L-BFGS-B gradient optimization to find the true EI maximum rather than
    random sampling, which is unreliable in high-dimensional spaces (16+ params).
    This gives a trustworthy stopping decision: if even directed search can't find
    a promising region, there genuinely isn't one.
    """

    def __init__(  # noqa: PLR0913
        self,
        ei_threshold: float = 0.5,
        patience_fallback: int = 30,
        min_trials: int = 10,
        min_trials_before_stop: int = 50,
        n_restarts: int = 10,
        seed: int = 42,
        baseline_score: float | None = None,
        check_every: int = 3,
    ):
        self._ei_threshold = ei_threshold
        self._patience_fallback = patience_fallback
        self._min_trials = min_trials
        self._min_trials_before_stop = min_trials_before_stop
        self._n_restarts = n_restarts
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._trials_without_improvement = 0
        self._best_value = None
        self._baseline_score = baseline_score
        self._check_every = check_every
        self._call_count = 0

        # Cache heavy imports — None means "not yet attempted", False means unavailable
        self._scipy_minimize = None
        self._GaussianProcessRegressor = None
        self._gp_kernels = None
        self._imports_available: bool | None = None

    def _ensure_imports(self) -> bool:
        """Import sklearn/scipy once and cache. Returns True if available."""
        if self._imports_available is not None:
            return self._imports_available
        try:
            from scipy.optimize import minimize as scipy_minimize
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import (
                ConstantKernel,
                Matern,
                WhiteKernel,
            )

            self._scipy_minimize = scipy_minimize
            self._GaussianProcessRegressor = GaussianProcessRegressor
            self._gp_kernels = (ConstantKernel, Matern, WhiteKernel)
            self._imports_available = True
        except ImportError:
            logger.warning(
                "scikit-learn or scipy not installed — GP early-stopping disabled, "
                "falling back to patience-only stopping."
            )
            self._imports_available = False
        return self._imports_available

    def _fit_gaussian_process(self, trials, search_space, param_names):
        """Fit a GP to completed trial data. Returns (gp, X, y) or None on failure."""
        X, y = [], []  # noqa: N806
        for t in trials:
            row = []
            for name in param_names:
                if name not in t.params:
                    break
                row.append(_encode_param(t.params[name], search_space[name]))
            else:
                X.append(row)
                y.append(trial_scalar_value(t))

        if len(X) < self._min_trials:
            return None

        X = np.array(X)  # noqa: N806
        y = np.array(y)

        if X.ndim < 2 or X.shape[1] == 0:  # noqa: PLR2004
            return None

        if not self._ensure_imports():
            return None

        ConstantKernel, Matern, WhiteKernel = self._gp_kernels  # noqa: N806
        kernel = ConstantKernel(1.0) * Matern(
            nu=2.5, length_scale=np.ones(X.shape[1])
        ) + WhiteKernel(noise_level=0.1)
        gp = self._GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=self._seed,
            normalize_y=True,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X, y)
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.debug("GP stopping callback fit failed: %s", e)
            return None

        return gp, X, y

    def _compute_expected_improvement(self, gp, best_y, param_names):
        """Find maximum EI via L-BFGS-B restarts. Returns max_ei float."""
        n_dims = len(param_names)
        bounds = [(0.0, 1.0)] * n_dims
        max_ei = 0.0

        def neg_ei(x: npt.NDArray[np.floating[Any]]) -> float:
            x_2d = x.reshape(1, -1)
            mu, sigma = gp.predict(x_2d, return_std=True)
            return -float(_expected_improvement(mu, sigma, best_y)[0])

        for _ in range(self._n_restarts):
            x0 = self._rng.uniform(0, 1, size=n_dims)
            try:
                result = self._scipy_minimize(
                    neg_ei, x0, method="L-BFGS-B", bounds=bounds
                )
                ei_val = -result.fun
                if ei_val > max_ei:
                    max_ei = ei_val
            except (ValueError, np.linalg.LinAlgError):
                continue

        return max_ei

    def _check_stopping_criteria(
        self, ei_value, best_value, below_baseline, too_early, study
    ):
        """Check if EI is below threshold and stop the study if appropriate."""
        scaled_threshold = self._ei_threshold * best_value / 100

        if ei_value < scaled_threshold:
            if not (below_baseline or too_early):
                logger.debug(
                    "GP stopping: max EI=%.2f"
                    " < threshold=%.2f"
                    " (confident no untested config"
                    " beats %.1f)",
                    ei_value,
                    scaled_threshold,
                    best_value,
                )
                study.stop()

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        self._call_count += 1

        # Track patience as fallback
        _bv = safe_best_value(study)
        if _bv is None or _bv == self._best_value:
            self._trials_without_improvement += 1
        else:
            self._best_value = _bv
            self._trials_without_improvement = 0

        below_baseline = (
            self._baseline_score is not None
            and _bv is not None
            and _bv < self._baseline_score
        )
        from ._study import get_positive_completed_trials

        n_completed = len(get_positive_completed_trials(study))
        too_early = n_completed < self._min_trials_before_stop

        # Hard fallback: stop after patience_fallback trials without improvement
        if self._trials_without_improvement >= self._patience_fallback:
            if below_baseline:
                logger.debug(
                    "Patience exhausted but best is below baseline — continuing"
                )
            elif too_early:
                logger.debug(
                    "Patience exhausted at %d trials but min is %d — continuing",
                    n_completed,
                    self._min_trials_before_stop,
                )
            else:
                logger.debug(
                    "GP stopping (fallback):"
                    " no improvement in %d trials"
                    " (%d completed).",
                    self._patience_fallback,
                    n_completed,
                )
                study.stop()
                return

        # Only fit the GP every N trials
        if self._call_count % self._check_every != 0:
            return

        completed = get_positive_completed_trials(study)

        if len(completed) < self._min_trials:
            return

        search_space = optuna.search_space.intersection_search_space(study.trials)
        if not search_space:
            return

        param_names = sorted(search_space.keys())

        gp_result = self._fit_gaussian_process(completed, search_space, param_names)
        if gp_result is None:
            return

        gp, X, y = gp_result  # noqa: N806
        best_y = np.max(y)

        max_ei = self._compute_expected_improvement(gp, best_y, param_names)
        self._check_stopping_criteria(max_ei, best_y, below_baseline, too_early, study)
