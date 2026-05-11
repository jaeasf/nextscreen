"""Bootstrap confidence intervals on feature selection rankings."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result extractors — one per method type
# ---------------------------------------------------------------------------

def extract_tabular(result: pd.DataFrame) -> pd.DataFrame:
    """Extract (feature, rank) from LASSO / Random Forest output."""
    return result[["feature", "rank"]]


def extract_shap(result: dict[str, Any]) -> pd.DataFrame:
    """Extract (feature, rank) from run_shap output."""
    return result["feature_importance"][["feature", "rank"]]


def extract_pca(result: dict[str, Any]) -> pd.DataFrame:
    """Extract (feature, rank) from run_pca output."""
    return result["feature_rank"][["feature", "rank"]]


def extract_correlations(result: pd.DataFrame) -> pd.DataFrame:
    """Extract (feature, rank) from run_correlations output."""
    return (
        result[["rank"]]
        .reset_index()
        .rename(columns={"index": "feature"})[["feature", "rank"]]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bootstrap_ranks(
    method: Callable,
    X: pd.DataFrame,
    y: pd.Series | None,
    result_extractor: Callable[[Any], pd.DataFrame],
    n_bootstrap: int = 100,
    random_state: int = 42,
    method_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Estimate rank stability for a feature selection method via bootstrap.

    Resamples the training data *n_bootstrap* times with replacement,
    reruns *method* on each sample, and aggregates the rank distribution
    per feature.

    Parameters
    ----------
    method : callable
        A feature selection function.  Called as ``method(X, y,
        **method_kwargs)`` (or ``method(X, **method_kwargs)`` when
        *y* is ``None``).
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or None
        Target variable.  Pass ``None`` for unsupervised methods (PCA).
    result_extractor : callable
        Function that takes the raw method output and returns a
        ``pd.DataFrame`` with columns ``['feature', 'rank']``.
        Use the module-level helpers: :func:`extract_tabular`,
        :func:`extract_shap`, :func:`extract_pca`,
        :func:`extract_correlations`.
    n_bootstrap : int, optional
        Number of bootstrap resamples.  Default 100.
    random_state : int, optional
        Seed for the resampling RNG.  Default 42.
    method_kwargs : dict, optional
        Extra keyword arguments forwarded to *method* on every call.

    Returns
    -------
    pd.DataFrame
        One row per feature, sorted by ascending ``median_rank``.
        Columns: ``['feature', 'median_rank', 'rank_q25', 'rank_q75',
        'rank_std', 'n_success']``.
        ``n_success`` is the number of bootstrap iterations that
        completed without error.

    Raises
    ------
    ValueError
        If *X* is empty or *n_bootstrap* < 1.
    """
    if X.empty:
        raise ValueError("X must contain at least one row.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1.")

    kwargs = method_kwargs or {}
    rng = np.random.default_rng(random_state)
    ranks_by_feature: dict[str, list[int]] = defaultdict(list)
    n_success = 0

    for b in range(n_bootstrap):
        idx = rng.integers(0, len(X), size=len(X))
        X_boot = X.iloc[idx].reset_index(drop=True)
        y_boot = (
            y.iloc[idx].reset_index(drop=True)
            if y is not None else None
        )
        try:
            if y_boot is not None:
                raw = method(X_boot, y_boot, **kwargs)
            else:
                raw = method(X_boot, **kwargs)
            rank_df = result_extractor(raw)
            for _, row in rank_df.iterrows():
                ranks_by_feature[str(row["feature"])].append(
                    int(row["rank"])
                )
            n_success += 1
        except Exception:  # noqa: BLE001
            logger.debug(
                "Bootstrap iteration %d/%d failed; skipping.", b, n_bootstrap
            )

    if n_success == 0:
        logger.warning(
            "All %d bootstrap iterations failed for method '%s'.",
            n_bootstrap,
            getattr(method, "__name__", str(method)),
        )
        return pd.DataFrame(
            columns=[
                "feature", "median_rank", "rank_q25",
                "rank_q75", "rank_std", "n_success",
            ]
        )

    records = []
    for feat, rank_list in ranks_by_feature.items():
        arr = np.array(rank_list, dtype=float)
        records.append(
            {
                "feature": feat,
                "median_rank": float(np.median(arr)),
                "rank_q25": float(np.percentile(arr, 25)),
                "rank_q75": float(np.percentile(arr, 75)),
                "rank_std": float(np.std(arr)),
                "n_success": n_success,
            }
        )

    result = (
        pd.DataFrame(records)
        .sort_values("median_rank")
        .reset_index(drop=True)
    )

    logger.info(
        "Bootstrap (%d/%d OK): top feature = '%s' "
        "(median rank %.1f, IQR [%.1f, %.1f]).",
        n_success,
        n_bootstrap,
        result.iloc[0]["feature"],
        result.iloc[0]["median_rank"],
        result.iloc[0]["rank_q25"],
        result.iloc[0]["rank_q75"],
    )
    return result
