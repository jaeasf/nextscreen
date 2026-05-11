"""Consensus feature ranking aggregated across multiple selection methods."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_consensus(
    rankings: dict[str, pd.DataFrame],
    feature_col: str = "feature",
    rank_col: str = "rank",
) -> pd.DataFrame:
    """Aggregate per-method feature ranks into a single consensus ranking.

    Consensus score is computed as the average *normalized* rank across all
    provided methods, where normalized rank = rank / n_features (lower is
    better). The final ``consensus_rank`` column sorts by ascending average
    normalized rank (rank 1 = most important).

    Parameters
    ----------
    rankings : dict of str → pd.DataFrame
        Mapping of method name to its ranking DataFrame. Each DataFrame must
        contain at least the columns identified by *feature_col* and
        *rank_col*.
    feature_col : str, optional
        Name of the column containing feature names in each input DataFrame.
        Default is ``'feature'``.
    rank_col : str, optional
        Name of the column containing integer ranks in each input DataFrame.
        Default is ``'rank'``.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per feature and columns:
        ``['feature', '<method>_rank', ..., 'avg_normalized_rank',
           'consensus_rank', 'n_methods_top_k']``.
        ``n_methods_top_k`` counts how many methods placed the feature in the
        top-K (K = max(3, n_features // 3)).

    Notes
    -----
    Features absent from a method's ranking (e.g., dropped by LASSO) are
    assigned the worst normalized rank (1.0) for that method.
    """
    if not rankings:
        return pd.DataFrame(
            columns=[
                "feature",
                "avg_normalized_rank",
                "consensus_rank",
                "n_methods_top_k",
            ]
        )

    # Collect all unique features (preserve first-seen order).
    seen: set[str] = set()
    all_features: list[str] = []
    for df in rankings.values():
        for f in df[feature_col].tolist():
            if f not in seen:
                all_features.append(str(f))
                seen.add(str(f))

    n_features = len(all_features)
    top_k = max(3, n_features // 3)

    data: dict[str, list] = {"feature": all_features}
    norm_cols: list[str] = []
    top_k_counts: list[int] = [0] * n_features

    for method_name, df in rankings.items():
        rank_map: dict[str, int] = {
            str(f): int(r)
            for f, r in zip(df[feature_col], df[rank_col])
        }
        raw: list[int] = [
            rank_map.get(f, n_features) for f in all_features
        ]
        col_rank = f"{method_name}_{rank_col}"
        norm_col = f"_norm_{method_name}"
        data[col_rank] = raw
        data[norm_col] = [r / n_features for r in raw]
        norm_cols.append(norm_col)

        for i, r in enumerate(raw):
            if r <= top_k:
                top_k_counts[i] += 1

    result = pd.DataFrame(data)
    result["avg_normalized_rank"] = result[norm_cols].mean(axis=1)
    result = result.drop(columns=norm_cols)
    result["n_methods_top_k"] = top_k_counts

    result = (
        result.sort_values("avg_normalized_rank")
        .reset_index(drop=True)
    )
    result["consensus_rank"] = (
        range(1, len(result) + 1)
    )
    result["consensus_rank"] = result["consensus_rank"].astype(int)

    # Reorder: feature → method rank cols → aggregates.
    _skip = frozenset(
        ["feature", "avg_normalized_rank",
         "consensus_rank", "n_methods_top_k"]
    )
    method_rank_cols = [
        c for c in result.columns if c not in _skip
    ]
    col_order = (
        ["feature"]
        + method_rank_cols
        + ["avg_normalized_rank", "consensus_rank", "n_methods_top_k"]
    )
    result = result[col_order]

    logger.info(
        "Consensus: %d features ranked across %d method(s).",
        n_features,
        len(rankings),
    )
    return result


def label_importance(
    consensus_df: pd.DataFrame,
    n_methods: int,
) -> pd.DataFrame:
    """Attach a human-readable importance label to each feature.

    Labels are assigned based on the fraction of methods that ranked the
    feature in the top-K:

    - >= 75% of methods → ``'strongly important'``
    - 50–74% of methods → ``'moderately important'``
    - < 50% of methods → ``'weakly important or inconsistent'``

    Parameters
    ----------
    consensus_df : pd.DataFrame
        Output of :func:`compute_consensus`, must contain
        ``'n_methods_top_k'``.
    n_methods : int
        Total number of methods used to build the consensus.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional ``'importance_label'`` column.
    """
    if n_methods <= 0:
        raise ValueError(
            f"n_methods must be >= 1; got {n_methods}."
        )

    def _label(n_top_k: int) -> str:
        frac = n_top_k / n_methods
        if frac >= 0.75:
            return "strongly important"
        if frac >= 0.50:
            return "moderately important"
        return "weakly important or inconsistent"

    result = consensus_df.copy()
    result["importance_label"] = (
        result["n_methods_top_k"].apply(_label)
    )
    return result
