"""Tests for nextscreen.features.consensus and nextscreen.interpretation.narrator."""

from __future__ import annotations

import pandas as pd
import pytest

from nextscreen.features.consensus import compute_consensus, label_importance
from nextscreen.interpretation.narrator import interpret_consensus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_rankings() -> dict[str, pd.DataFrame]:
    """Three-method ranking over four features.

    Derived ranks (n_features=4, top_k=max(3, 4//3)=3):
        A: lasso=1, rf=1, shap=2  → n_methods_top_k=3  (all ≤ 3)
        B: lasso=2, rf=3, shap=1  → n_methods_top_k=3  (all ≤ 3)
        C: lasso=3, rf=2, shap=4  → n_methods_top_k=2  (lasso, rf ≤ 3)
        D: lasso=4, rf=4, shap=3  → n_methods_top_k=1  (only shap ≤ 3)

    avg_normalized_rank:
        A = (1 + 1 + 2) / (4 * 3) = 0.333  → consensus_rank 1
        B = (2 + 3 + 1) / (4 * 3) = 0.500  → consensus_rank 2
        C = (3 + 2 + 4) / (4 * 3) = 0.750  → consensus_rank 3
        D = (4 + 4 + 3) / (4 * 3) = 0.917  → consensus_rank 4
    """
    return {
        "lasso": pd.DataFrame(
            {"feature": ["A", "B", "C", "D"], "rank": [1, 2, 3, 4]}
        ),
        "random_forest": pd.DataFrame(
            {"feature": ["A", "C", "B", "D"], "rank": [1, 2, 3, 4]}
        ),
        "shap": pd.DataFrame(
            {"feature": ["B", "A", "D", "C"], "rank": [1, 2, 3, 4]}
        ),
    }


# ---------------------------------------------------------------------------
# compute_consensus
# ---------------------------------------------------------------------------

class TestComputeConsensus:
    def test_returns_dataframe(self, sample_rankings: dict) -> None:
        result = compute_consensus(sample_rankings)
        assert isinstance(result, pd.DataFrame)

    def test_all_features_present(self, sample_rankings: dict) -> None:
        result = compute_consensus(sample_rankings)
        assert set(result["feature"]) == {"A", "B", "C", "D"}

    def test_consensus_rank_column_present(
        self, sample_rankings: dict
    ) -> None:
        result = compute_consensus(sample_rankings)
        assert "consensus_rank" in result.columns

    def test_consensus_rank_starts_at_one(
        self, sample_rankings: dict
    ) -> None:
        result = compute_consensus(sample_rankings)
        assert result["consensus_rank"].min() == 1

    def test_consensus_rank_contiguous(
        self, sample_rankings: dict
    ) -> None:
        result = compute_consensus(sample_rankings)
        ranks = sorted(result["consensus_rank"].tolist())
        assert ranks == list(range(1, len(ranks) + 1))

    def test_avg_normalized_rank_present(
        self, sample_rankings: dict
    ) -> None:
        result = compute_consensus(sample_rankings)
        assert "avg_normalized_rank" in result.columns

    def test_method_rank_columns_present(
        self, sample_rankings: dict
    ) -> None:
        result = compute_consensus(sample_rankings)
        for method in ("lasso", "random_forest", "shap"):
            assert f"{method}_rank" in result.columns

    def test_n_methods_top_k_present(
        self, sample_rankings: dict
    ) -> None:
        result = compute_consensus(sample_rankings)
        assert "n_methods_top_k" in result.columns

    def test_top_feature_is_a(self, sample_rankings: dict) -> None:
        result = compute_consensus(sample_rankings)
        top = result[result["consensus_rank"] == 1].iloc[0]
        assert top["feature"] == "A"

    def test_bottom_feature_is_d(self, sample_rankings: dict) -> None:
        result = compute_consensus(sample_rankings)
        bottom = result[result["consensus_rank"] == 4].iloc[0]
        assert bottom["feature"] == "D"

    def test_missing_feature_gets_worst_rank(self) -> None:
        """A feature absent from one method receives rank = n_features."""
        rankings = {
            "m1": pd.DataFrame(
                {"feature": ["A", "B", "C"], "rank": [1, 2, 3]}
            ),
            "m2": pd.DataFrame(
                {"feature": ["A", "B"], "rank": [1, 2]}
            ),  # C absent
        }
        result = compute_consensus(rankings)
        c_row = result[result["feature"] == "C"]
        # n_features=3, so C gets rank 3 from m2.
        assert c_row["m2_rank"].iloc[0] == 3

    def test_perfect_agreement_gives_rank_one(self) -> None:
        rankings = {
            "m1": pd.DataFrame(
                {"feature": ["X", "Y"], "rank": [1, 2]}
            ),
            "m2": pd.DataFrame(
                {"feature": ["X", "Y"], "rank": [1, 2]}
            ),
        }
        result = compute_consensus(rankings)
        x_rank = result[result["feature"] == "X"][
            "consensus_rank"
        ].iloc[0]
        assert x_rank == 1

    def test_empty_rankings_returns_empty_dataframe(self) -> None:
        result = compute_consensus({})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# label_importance
# ---------------------------------------------------------------------------

class TestLabelImportance:
    def test_strongly_important_label(
        self, sample_rankings: dict
    ) -> None:
        """A ranked top-k by all 3 methods → 'strongly important'."""
        con = compute_consensus(sample_rankings)
        labelled = label_importance(con, n_methods=3)
        a_label = labelled[labelled["feature"] == "A"][
            "importance_label"
        ].iloc[0]
        assert a_label == "strongly important"

    def test_weakly_important_label(
        self, sample_rankings: dict
    ) -> None:
        """D ranked top-k by only 1 of 3 methods → 'weakly important'."""
        con = compute_consensus(sample_rankings)
        labelled = label_importance(con, n_methods=3)
        d_label = labelled[labelled["feature"] == "D"][
            "importance_label"
        ].iloc[0]
        assert d_label == "weakly important or inconsistent"

    def test_moderately_important_label(
        self, sample_rankings: dict
    ) -> None:
        """C ranked top-k by 2 of 3 methods (66 %) → 'moderately'."""
        con = compute_consensus(sample_rankings)
        labelled = label_importance(con, n_methods=3)
        c_label = labelled[labelled["feature"] == "C"][
            "importance_label"
        ].iloc[0]
        assert c_label == "moderately important"

    def test_all_features_labelled(
        self, sample_rankings: dict
    ) -> None:
        con = compute_consensus(sample_rankings)
        labelled = label_importance(con, n_methods=3)
        assert "importance_label" in labelled.columns
        assert len(labelled) == 4
        assert not labelled["importance_label"].isna().any()

    def test_invalid_n_methods_raises(
        self, sample_rankings: dict
    ) -> None:
        con = compute_consensus(sample_rankings)
        with pytest.raises(ValueError, match="n_methods"):
            label_importance(con, n_methods=0)


# ---------------------------------------------------------------------------
# interpret_consensus (narrator)
# ---------------------------------------------------------------------------

@pytest.fixture()
def consensus_df(sample_rankings: dict) -> pd.DataFrame:
    """Labelled consensus DataFrame built from sample_rankings."""
    con = compute_consensus(sample_rankings)
    return label_importance(con, n_methods=3)


_TARGET = "yield"
_CAVEAT = "Domain expertise should guide final decisions"


class TestInterpretConsensus:
    def test_returns_string(self, consensus_df: pd.DataFrame) -> None:
        out = interpret_consensus(consensus_df, _TARGET, 3)
        assert isinstance(out, str)

    def test_contains_caveat(self, consensus_df: pd.DataFrame) -> None:
        out = interpret_consensus(consensus_df, _TARGET, 3)
        assert _CAVEAT in out

    def test_mentions_top_feature(
        self, consensus_df: pd.DataFrame
    ) -> None:
        out = interpret_consensus(consensus_df, _TARGET, 3)
        assert "A" in out

    def test_mentions_target_name(
        self, consensus_df: pd.DataFrame
    ) -> None:
        out = interpret_consensus(consensus_df, _TARGET, 3)
        assert _TARGET in out
