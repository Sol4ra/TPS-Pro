"""Optuna study management, callbacks, trial scoring, and phase result I/O."""
import json
import logging
import sys
import time
from pathlib import Path

import warnings

logger = logging.getLogger(__name__)

import numpy as np
import optuna
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from .state import ctx, _config
from .constants import SCORE_VERSION

try:
    from tqdm import tqdm as _tqdm_cls
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# Module-level progress bar
_active_pbar = None


def _trial_scalar_value(t):
    """Get a trial's scalar value, safe for both single- and multi-objective studies.

    In multi-objective studies, t.value raises RuntimeError — fall back to t.values[0].
    """
    try:
        return t.value
    except RuntimeError:
        return t.values[0] if t.values else None


class GPSampler(optuna.samplers.BaseSampler):
    """Gaussian Process sampler for Optuna using Expected Improvement.

    Fits a GP to all completed trials, then proposes the next trial by maximizing
    Expected Improvement (EI). Handles mixed parameter types by encoding everything
    to [0,1]. Falls back to random sampling for the first n_startup_trials.
    """

    def __init__(self, seed=42, n_startup_trials=10, n_candidates=2000):
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._n_startup = n_startup_trials
        self._n_candidates = n_candidates
        self._random_sampler = optuna.samplers.RandomSampler(seed=seed)

    def infer_relative_search_space(self, study, trial):
        return optuna.search_space.intersection_search_space(study.trials)

    def sample_relative(self, study, trial, search_space):
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE and _trial_scalar_value(t) is not None and _trial_scalar_value(t) > 0]

        if len(completed) < self._n_startup or not search_space:
            return {}  # fall back to sample_independent (random)

        # Encode completed trials into X matrix and y vector
        param_names = sorted(search_space.keys())
        X, y, noise_vars = [], [], []
        for t in completed:
            row = []
            for name in param_names:
                if name not in t.params:
                    break
                row.append(self._encode_param(t.params[name], search_space[name]))
            else:
                X.append(row)
                y.append(_trial_scalar_value(t))
                # Noise-aware GP: use per-trial measurement variance if available
                tps_std = t.user_attrs.get("tps_std", 0)
                noise_vars.append(max(tps_std ** 2, 0.01))  # floor to prevent zero variance

        if len(X) < self._n_startup:
            return {}

        X = np.array(X)
        y = np.array(y)
        alpha = np.array(noise_vars)  # per-sample noise variance

        # Fit GP with noise-aware kernel
        # alpha passes per-observation noise to the GP, so the model knows which
        # measurements are reliable vs noisy (e.g., a trial with CV=10% vs CV=1%)
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(X.shape[1])) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=2, random_state=self._seed, normalize_y=True)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X, y)
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.debug("GP fit failed, falling back to random sampling: %s", e)
            return {}  # GP fit failed, fall back to random

        # Generate random candidates and pick the one with highest EI
        candidates = self._rng.uniform(0, 1, size=(self._n_candidates, len(param_names)))
        mu, sigma = gp.predict(candidates, return_std=True)

        best_y = np.max(y)
        ei = self._expected_improvement(mu, sigma, best_y)

        best_idx = np.argmax(ei)
        best_candidate = candidates[best_idx]

        # Decode back to parameter values
        params = {}
        for i, name in enumerate(param_names):
            params[name] = self._decode_param(best_candidate[i], search_space[name])
        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._random_sampler.sample_independent(study, trial, param_name, param_distribution)

    @staticmethod
    def _expected_improvement(mu, sigma, best_y, xi=0.01):
        """Compute Expected Improvement. xi is exploration-exploitation tradeoff."""
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = mu - best_y - xi
            Z = np.where(sigma > 1e-8, imp / sigma, 0.0)
            ei = np.where(sigma > 1e-8, imp * norm.cdf(Z) + sigma * norm.pdf(Z), 0.0)
        return ei

    @staticmethod
    def _encode_param(value, distribution):
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

    @staticmethod
    def _decode_param(encoded, distribution):
        """Decode a [0, 1] value back to the parameter's original type/range."""
        if isinstance(distribution, optuna.distributions.CategoricalDistribution):
            choices = distribution.choices
            idx = int(round(encoded * (len(choices) - 1)))
            idx = max(0, min(idx, len(choices) - 1))
            return choices[idx]
        elif isinstance(distribution, optuna.distributions.IntDistribution):
            low, high = distribution.low, distribution.high
            step = distribution.step
            raw = low + encoded * (high - low)
            # Snap to step grid
            return int(round((raw - low) / step) * step + low)
        elif isinstance(distribution, optuna.distributions.FloatDistribution):
            low, high = distribution.low, distribution.high
            return low + encoded * (high - low)
        return encoded


class TqdmUpdateCallback:
    """Ensures the progress bar updates even if a trial is pruned or fails."""
    def __call__(self, study, trial):
        global _active_pbar
        if _active_pbar is not None:
            _active_pbar.update(1)


class GPStoppingCallback:
    """Stops optimization when the GP's maximum Expected Improvement drops below a threshold.

    This means the GP is confident that no untested configuration is likely to beat
    the current best — a mathematically principled replacement for patience-based stopping.
    """

    def __init__(self, ei_threshold=0.5, n_candidates=2000, patience_fallback=15, min_trials=10, min_trials_before_stop=20, seed=42, baseline_score=None, check_every=3):
        self._ei_threshold = ei_threshold
        self._n_candidates = n_candidates
        self._patience_fallback = patience_fallback
        self._min_trials = min_trials
        self._min_trials_before_stop = min_trials_before_stop  # must run at least this many before GP can stop
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._trials_without_improvement = 0
        self._best_value = None
        self._baseline_score = baseline_score  # don't stop early if nothing has beaten baseline
        self._check_every = check_every  # only fit GP every N trials (expensive)
        self._call_count = 0

    def __call__(self, study, trial):
        self._call_count += 1

        # Track patience as fallback
        _bv = _safe_best_value(study)
        if _bv is None or _bv == self._best_value:
            self._trials_without_improvement += 1
        else:
            self._best_value = _bv
            self._trials_without_improvement = 0

        # Hard fallback: stop after patience_fallback trials without improvement
        # But never stop if we haven't beaten baseline — keep searching
        best_so_far = _safe_best_value(study)
        below_baseline = (self._baseline_score is not None and best_so_far is not None
                          and best_so_far < self._baseline_score)
        n_completed = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE and _trial_scalar_value(t) is not None and _trial_scalar_value(t) > 0])
        too_early = n_completed < self._min_trials_before_stop

        if self._trials_without_improvement >= self._patience_fallback:
            if below_baseline or too_early:
                pass  # keep going
            else:
                print(f"\n  [!] GP stopping (fallback): no improvement in {self._patience_fallback} trials.")
                study.stop()
                return

        # Only fit the GP every N trials — it's expensive (~0.5-2s per fit)
        if self._call_count % self._check_every != 0:
            return

        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE and _trial_scalar_value(t) is not None and _trial_scalar_value(t) > 0]

        if len(completed) < self._min_trials:
            return  # not enough data to fit GP

        search_space = optuna.search_space.intersection_search_space(study.trials)
        if not search_space:
            return

        param_names = sorted(search_space.keys())
        X, y = [], []
        for t in completed:
            row = []
            for name in param_names:
                if name not in t.params:
                    break
                row.append(GPSampler._encode_param(t.params[name], search_space[name]))
            else:
                X.append(row)
                y.append(_trial_scalar_value(t))

        if len(X) < self._min_trials:
            return

        X = np.array(X)
        y = np.array(y)

        if X.ndim < 2 or X.shape[1] == 0:
            return  # empty search space after filtering — can't fit GP

        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(X.shape[1])) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=self._seed, normalize_y=True)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X, y)
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.debug("GP stopping callback fit failed: %s", e)
            return  # GP fit failed, continue with trials

        candidates = self._rng.uniform(0, 1, size=(self._n_candidates, len(param_names)))
        mu, sigma = gp.predict(candidates, return_std=True)

        best_y = np.max(y)
        ei = GPSampler._expected_improvement(mu, sigma, best_y)
        max_ei = np.max(ei)

        # Scale threshold relative to best score (percentage-based)
        scaled_threshold = self._ei_threshold * best_y / 100

        if max_ei < scaled_threshold:
            if below_baseline or too_early:
                # Don't stop — either below baseline or haven't run enough trials
                pass
            else:
                print(f"\n  [!] GP stopping: max EI={max_ei:.2f} < threshold={scaled_threshold:.2f} "
                      f"(confident no untested config beats {best_y:.1f})")
                study.stop()


def ensure_results_dir():
    ctx.results_dir.mkdir(parents=True, exist_ok=True)


def check_duplicate_trial(trial):
    """Check if this exact param combo was already tested. Returns cached score or None.

    In multi-objective (Pareto) mode, returns the tuple of values instead of a scalar.
    Uses a frozen-params dict for O(1) lookup instead of scanning all past trials.
    """
    # Build lookup cache on first call (lazy, attached to study object)
    study = trial.study
    if not hasattr(study, "_param_cache"):
        study._param_cache = {}
        for past in study.trials:
            if past.state == optuna.trial.TrialState.COMPLETE:
                key = tuple(sorted(past.params.items()))
                study._param_cache[key] = past

    key = tuple(sorted(trial.params.items()))
    past = study._param_cache.get(key)
    if past is not None:
        for k, v in past.user_attrs.items():
            trial.set_user_attr(k, v)
        is_multi = len(study.directions) > 1
        if is_multi and past.values is not None:
            return past.values
        return past.value
    return None


def setup_study(study_name, n_trials, seed=42, sampler_override=None, pruner=None, is_pareto=False):
    """Create/resume an Optuna study. Returns (study, remaining_trials, completed).

    Study names are versioned with SCORE_VERSION so that formula changes
    automatically start fresh studies instead of resuming stale data.

    Args:
        is_pareto: If True, creates a multi-objective study with NSGA-II
                   (3 directions: TPS, -VRAM, quality). Phases must opt in
                   explicitly — single-objective phases must leave this False.
        pruner: Optional Optuna pruner for multi-fidelity trials (e.g., MedianPruner).
    """
    ensure_results_dir()
    versioned_name = f"{study_name}_{SCORE_VERSION}" + ("_pareto" if is_pareto else "")

    # If fresh run requested, delete any existing study with this name
    if ctx.fresh_run:
        try:
            optuna.delete_study(study_name=versioned_name, storage=ctx.optuna_db)
        except KeyError:
            pass  # study doesn't exist yet

    # Default pruner: WilcoxonPruner uses statistical testing on paired measurements,
    # which is better than MedianPruner for noisy server benchmarks where measurements
    # don't have a natural ordering (prompt 3 isn't "later" than prompt 1).
    if pruner is None:
        pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.1)

    if is_pareto:
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            study_name=versioned_name,
            storage=ctx.optuna_db,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )
    else:
        sampler = sampler_override if sampler_override is not None else optuna.samplers.TPESampler(
            multivariate=True, seed=seed, warn_independent_sampling=False,
        )
        study = optuna.create_study(
            direction="maximize",
            study_name=versioned_name,
            storage=ctx.optuna_db,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )

    completed = len(study.trials)
    remaining = n_trials
    if completed > 0:
        print(f"\n[*] Resuming from trial {completed}/{n_trials} ({completed} completed)")
        remaining = max(0, n_trials - completed)
        if remaining == 0:
            print("    All trials already completed. Use more trials or reset DB.")

    return study, remaining, completed


def save_phase_results(phase_name, results):
    """Save phase results to JSON atomically (write tmp, then rename).

    Prevents data corruption if user hits Ctrl+C mid-write.
    Tags results with SCORE_VERSION so load_phase_results can reject stale data.
    """
    ensure_results_dir()
    results["score_version"] = SCORE_VERSION
    final_path = ctx.results_dir / f"{phase_name}_results.json"
    temp_path = ctx.results_dir / f"{phase_name}_results.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    temp_path.replace(final_path)
    print(f"\n  Results saved to {final_path}")


def load_phase_results(phase_name):
    """Load saved results from a completed phase.

    Returns None if the results were saved under a different SCORE_VERSION,
    forcing a re-run so stale configs from an old formula aren't reused.
    """
    path = ctx.results_dir / f"{phase_name}_results.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            print(f"  [!] {phase_name} results corrupted — will re-run")
            return None
        if data.get("score_version") != SCORE_VERSION:
            print(f"  [!] {phase_name} results from score {data.get('score_version', 'v1')} "
                  f"(current: {SCORE_VERSION}) — will re-run")
            return None
        return data
    return None


def _safe_best_value(study):
    """Get study.best_value without throwing if no valid trials exist."""
    try:
        return study.best_value
    except ValueError:
        return None


def create_phase_pbar(total, desc=""):
    """Create a tqdm progress bar that writes to raw stdout (bypasses LogTee).

    Returns the bar object, or None if tqdm isn't available.
    Caller must call .close() when done.
    """
    global _active_pbar
    if _HAS_TQDM:
        bar = _tqdm_cls(
            total=total,
            desc=f"  {desc}",
            file=sys.__stdout__,  # bypass LogTee — only show on screen
            bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ncols=80,
            leave=True,
        )
        _active_pbar = bar
        return bar
    _active_pbar = None
    return None


def close_phase_pbar():
    """Close the active progress bar if one exists."""
    global _active_pbar
    if _active_pbar is not None:
        _active_pbar.close()
        _active_pbar = None


def print_trial_result(trial_num, total_trials, tps, perf, params_short, best_score, final_score=None):
    """Print a formatted trial result line. Returns new best_score.

    When tqdm is active: updates the progress bar and only prints full lines
    for new bests (via tqdm.write, which prints above the bar).
    When tqdm is not available: prints every trial as a full line with a text bar.
    """
    from .measurement import compute_score

    score = final_score if final_score is not None else compute_score(perf)
    is_new_best = score > best_score
    if is_new_best:
        best_score = score

    vram_str = ""
    if perf and perf.get("vram_used_mb"):
        vram_str = f" | VRAM:{perf['vram_used_mb']/1024:.1f}GB"

    line = (f"  Trial {trial_num:3d}/{total_trials}: {tps:6.1f} t/s | "
            f"pp:{perf['prompt_tps']:5.0f} t/s | TTFT:{perf['ttft']:4.0f}ms{vram_str} | "
            f"score:{score:5.1f} | {params_short}")

    if _active_pbar is not None:
        marker = " *** NEW BEST ***" if is_new_best else ""
        _active_pbar.write(f"{line}{marker}")
    else:
        # Fallback: text-based progress bar
        done = trial_num + 1
        pct = done / total_trials * 100
        bar_len = 20
        filled = int(bar_len * done / total_trials)
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
        marker = " *** NEW BEST ***" if is_new_best else ""
        print(f"  [{bar}] {pct:5.1f}%  {line}{marker}")

    return best_score


def print_param_importance(study):
    """Print a ranked table of parameter importances using fANOVA."""
    # Filter out failed trials (score=0) so fANOVA has clean data
    completed = [t for t in study.trials if _trial_scalar_value(t) is not None and _trial_scalar_value(t) > 0]
    if len(completed) < 10:
        print(f"\n  (Only {len(completed)} successful trials — need 10+ for importance analysis)")
        return {}

    try:
        importances = optuna.importance.get_param_importances(study)
    except (RuntimeError, ValueError) as e:
        # Fallback to mean decrease impurity if fANOVA fails
        try:
            from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
            importances = optuna.importance.get_param_importances(
                study, evaluator=MeanDecreaseImpurityImportanceEvaluator()
            )
        except (RuntimeError, ValueError, ImportError):
            print(f"\n  (Could not compute parameter importance: {e})")
            return {}

    if not importances or len(importances) <= 1:
        return importances

    print(f"\n  Parameter Importance:")
    print(f"  {'Param':<28} {'Impact':>7}  {'':}")
    print(f"  {'\u2500' * 28} {'\u2500' * 7}  {'\u2500' * 20}")

    max_bar = 20
    for param, importance in importances.items():
        pct = importance * 100
        bar_len = int(importance / max(importances.values()) * max_bar) if max(importances.values()) > 0 else 0
        bar = "\u2588" * bar_len
        print(f"  {param:<28} {pct:6.1f}%  {bar}")

    return importances


def print_param_histogram(study, param_name, best_value=None):
    """Print a text histogram showing best score per parameter value.

    For single-parameter sweeps — shows how score varies across the parameter range.
    """
    # Collect best score per parameter value
    scores_by_val = {}
    for t in study.trials:
        tv = _trial_scalar_value(t)
        if tv is None or tv <= 0:
            continue
        val = t.params.get(param_name)
        if val is None:
            continue
        if val not in scores_by_val or tv > scores_by_val[val]:
            scores_by_val[val] = tv

    if not scores_by_val:
        return

    # Sort by parameter value (left = lowest, right = highest)
    sorted_vals = sorted(scores_by_val.keys())
    max_score = max(scores_by_val.values())
    bar_max = 30  # max bar width

    print(f"\n  Score by {param_name}:")
    print(f"  {'Value':>6}  {'Score':>7}  {'':}")
    print(f"  {'\u2500' * 6}  {'\u2500' * 7}  {'\u2500' * bar_max}")

    for val in sorted_vals:
        score = scores_by_val[val]
        bar_len = int(score / max_score * bar_max) if max_score > 0 else 0
        bar = "\u2588" * bar_len
        marker = " \u25c4 best" if best_value is not None and val == best_value else ""
        print(f"  {val:>6}  {score:>7.1f}  {bar}{marker}")
