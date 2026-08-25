"""Significance testing - Phase 3 design item #5 (approved).

Not naive binomial/iid tests: daily predictions are serially correlated (squared-
1-day-return autocorrelation of 0.184, established in the target-design pass), so a
block bootstrap (block length 40 trading days, inside the 20-60 day vol-clustering
persistence range already measured) is used throughout instead of iid resampling.

Three distinct tools, each aimed at a different question:
- `block_bootstrap_metric`: CI + approximate two-sided p-value (vs. 0) for a scalar
  metric from a per-date series (e.g. overall or per-regime mean P&L). The p-value is
  the proportion-beyond-zero of the bootstrap replicate distribution, doubled for
  two-sidedness - a standard, if approximate, bootstrap hypothesis-testing convention,
  not a pivotal/studentized test.
- `paired_block_bootstrap`: same mechanism, applied to a per-day (model - baseline)
  difference series, for the model-vs-rule-based-baseline comparison specifically -
  paired because both are evaluated on identical dates.
- `regime_permutation_test`: the instrument aimed at the project's actual thesis, not
  a generic add-on. Null: regime membership carries no information about per-date
  performance. Block-*permutes* (not resamples) the regime-label sequence - preserves
  each regime's real overall frequency exactly, only scrambles which stretch of dates
  each regime's block-run lands on - and asks how much apparent regime-to-regime
  performance spread would arise from partitioning ~3,900 days into persistent,
  regime-sized buckets by chance alone.

`benjamini_hochberg` applies FDR correction across the resulting family of tests,
per proposal §2/§4.4's explicit requirement to check whether apparent results survive
correction for how many things were tried.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BLOCK_LEN = 40
N_ITER = 2000
RNG_SEED = 0


def _block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Resample WITH replacement, in contiguous blocks - standard moving-block
    bootstrap, for CI/p-value construction."""
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, max(n - block_len, 0) + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
    return idx[:n]


def _block_permute(arr: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Permute (WITHOUT replacement) the ORDER of contiguous, non-overlapping blocks
    - preserves each block's internal run structure and the overall value frequency
    exactly; only scrambles which stretch of dates each block lands on. Used for the
    regime-permutation null, not for bootstrap CIs."""
    n = len(arr)
    n_blocks = int(np.ceil(n / block_len))
    blocks = [arr[i * block_len:(i + 1) * block_len] for i in range(n_blocks)]
    order = rng.permutation(n_blocks)
    return np.concatenate([blocks[i] for i in order])[:n]


def block_bootstrap_metric(
    values: pd.Series, metric_fn=np.mean, n_iter: int = N_ITER, block_len: int = BLOCK_LEN, seed: int = RNG_SEED
) -> dict:
    rng = np.random.default_rng(seed)
    arr = values.dropna().to_numpy()
    n = len(arr)
    point_estimate = float(metric_fn(arr))

    boot_stats = np.empty(n_iter)
    for b in range(n_iter):
        idx = _block_bootstrap_indices(n, block_len, rng)
        boot_stats[b] = metric_fn(arr[idx])

    ci_lo, ci_hi = np.percentile(boot_stats, [2.5, 97.5])
    p_value = 2 * min((boot_stats <= 0).mean(), (boot_stats >= 0).mean())
    return {
        "point_estimate": point_estimate,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": float(min(p_value, 1.0)),
        "n": n,
    }


def paired_block_bootstrap(
    diff_series: pd.Series, n_iter: int = N_ITER, block_len: int = BLOCK_LEN, seed: int = RNG_SEED
) -> dict:
    return block_bootstrap_metric(diff_series, np.mean, n_iter=n_iter, block_len=block_len, seed=seed)


def regime_permutation_test(
    regime: pd.Series, per_date_metric: pd.Series, n_iter: int = N_ITER, block_len: int = BLOCK_LEN, seed: int = RNG_SEED
) -> dict:
    rng = np.random.default_rng(seed)
    common_idx = regime.dropna().index.intersection(per_date_metric.dropna().index)
    reg = regime.reindex(common_idx).to_numpy()
    val = per_date_metric.reindex(common_idx).to_numpy()
    n = len(val)

    def regime_spread(labels_arr: np.ndarray) -> float:
        means = [val[labels_arr == r].mean() for r in np.unique(labels_arr) if (labels_arr == r).sum() >= 2]
        return float(np.max(means) - np.min(means)) if len(means) >= 2 else 0.0

    observed_spread = regime_spread(reg)
    perm_spreads = np.empty(n_iter)
    for p in range(n_iter):
        perm_spreads[p] = regime_spread(_block_permute(reg, block_len, rng))

    p_value = float((perm_spreads >= observed_spread).mean())
    return {
        "observed_spread": observed_spread,
        "null_mean": float(perm_spreads.mean()),
        "null_p95": float(np.percentile(perm_spreads, 95)),
        "p_value": p_value,
        "n": n,
    }


def benjamini_hochberg(p_values: dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    names = list(p_values.keys())
    pvals = np.array([p_values[nm] for nm in names])
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = len(pvals)
    thresh = (np.arange(1, m + 1) / m) * alpha

    passed = ranked <= thresh
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        k_max = int(np.max(np.where(passed)[0]))
        reject[: k_max + 1] = True

    df = pd.DataFrame({
        "test": [names[i] for i in order],
        "p_value": ranked,
        "bh_threshold": thresh,
        "reject_null": reject,
    })
    return df.sort_values("p_value").reset_index(drop=True)
