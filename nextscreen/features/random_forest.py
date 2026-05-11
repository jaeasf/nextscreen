"""Random Forest feature importance via mean decrease in impurity."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


def run_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit a Random Forest and return per-feature importances.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with shape (n_samples, n_features).
    y : pd.Series
        Target variable with length n_samples.
    n_estimators : int, optional
        Number of trees in the forest. Default is 100.
    max_depth : int or None, optional
        Maximum depth of each tree. ``None`` means nodes are expanded
        until all leaves are pure or contain fewer than
        ``min_samples_split`` samples. Default is ``None``.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['feature', 'importance', 'rank']``,
        sorted by descending ``importance``. Rank 1 is most important.
        Importances are non-negative and sum to 1.0.

    Raises
    ------
    ValueError
        If *X* is empty, if *X* and *y* have different lengths, or if
        *n_estimators* is less than 1.
    """
    if X.empty:
        raise ValueError("X must contain at least one row.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length "
            f"({len(X)} vs {len(y)})."
        )
    if n_estimators < 1:
        raise ValueError(
            f"n_estimators must be at least 1; got {n_estimators}."
        )

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    rf.fit(X, y)

    result = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance": rf.feature_importances_,
        }
    )
    result = result.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    result["rank"] = (
        result["importance"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    logger.info(
        "Random Forest: top feature = '%s' (importance=%.4f).",
        result.iloc[0]["feature"],
        result.iloc[0]["importance"],
    )
    return result
