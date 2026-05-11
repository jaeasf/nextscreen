"""SHAP-based feature importance using TreeExplainer on a Random Forest."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


def run_shap(
    X: pd.DataFrame,
    y: pd.Series,
    background_samples: int = 100,
    random_state: int = 42,
) -> dict[str, object]:
    """Compute SHAP values using a Random Forest and TreeExplainer.

    A Random Forest is trained internally; the caller does not need to
    supply a fitted model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with shape (n_samples, n_features).
    y : pd.Series
        Target variable with length n_samples.
    background_samples : int, optional
        Number of background samples selected from *X* and stored in the
        returned ``'X_background'`` key (used for visualisation).
        Capped at ``len(X)`` automatically. Default is 100.
    random_state : int, optional
        Random seed for the underlying Random Forest and background
        sample selection. Default is 42.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``'shap_values'`` : np.ndarray
            SHAP value matrix of shape (n_samples, n_features).
        ``'feature_importance'`` : pd.DataFrame
            Mean absolute SHAP values per feature, with columns
            ``['feature', 'mean_abs_shap', 'rank']``, sorted descending.
        ``'X_background'`` : pd.DataFrame
            The background sample subset (used for beeswarm plots).

    Raises
    ------
    ValueError
        If *X* is empty or if *X* and *y* have different lengths.
    """
    if X.empty:
        raise ValueError("X must contain at least one row.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length "
            f"({len(X)} vs {len(y)})."
        )

    # Select background sample subset (for visualisation only).
    n_bg = min(background_samples, len(X))
    rng = np.random.default_rng(random_state)
    bg_idx = sorted(rng.choice(len(X), size=n_bg, replace=False))
    X_background: pd.DataFrame = X.iloc[bg_idx].reset_index(drop=True)

    # Train a Random Forest surrogate model.
    rf = RandomForestRegressor(
        n_estimators=100, random_state=random_state
    )
    rf.fit(X, y)

    # Compute SHAP values via TreeExplainer (exact for tree ensembles).
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X)

    # Guard against list output from multi-output / classifier models.
    if isinstance(sv, list):
        sv = sv[0]
    shap_values: np.ndarray = np.asarray(sv)

    mean_abs = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame(
        {
            "feature": list(X.columns),
            "mean_abs_shap": mean_abs,
        }
    )
    feature_importance = feature_importance.sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)
    feature_importance["rank"] = (
        feature_importance["mean_abs_shap"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    logger.info(
        "SHAP: top feature = '%s' (mean |SHAP|=%.4f).",
        feature_importance.iloc[0]["feature"],
        feature_importance.iloc[0]["mean_abs_shap"],
    )

    return {
        "shap_values": shap_values,
        "feature_importance": feature_importance,
        "X_background": X_background,
    }
