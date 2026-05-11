"""Tests for nextscreen.data.loader."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from nextscreen.data.loader import (
    detect_replicates,
    drop_missing_rows,
    handle_replicates,
    load_file,
)

# ---------------------------------------------------------------------------
# Shared column names
# ---------------------------------------------------------------------------
FEAT_COLS = ["temperature", "pressure", "catalyst"]
TARGET_COLS = ["yield"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_csv(tmp_path: Path) -> Path:
    """CSV with 4 rows: one replicate pair (rows 0-1) and two unique rows."""
    content = (
        "temperature,pressure,catalyst,yield\n"
        "100,1.0,A,0.85\n"
        "100,1.0,A,0.87\n"  # replicate of row 0
        "200,2.0,B,0.60\n"
        "300,1.5,C,0.72\n"
    )
    p = tmp_path / "test_data.csv"
    p.write_text(content)
    return p


@pytest.fixture()
def simple_df(simple_csv: Path) -> pd.DataFrame:
    """DataFrame loaded from simple_csv."""
    return pd.read_csv(simple_csv)


@pytest.fixture()
def no_replicate_df() -> pd.DataFrame:
    """DataFrame with no duplicate conditions."""
    return pd.DataFrame(
        {
            "temperature": [100, 200, 300],
            "pressure": [1.0, 2.0, 3.0],
            "catalyst": ["A", "B", "C"],
            "yield": [0.80, 0.70, 0.60],
        }
    )


@pytest.fixture()
def multi_replicate_df() -> pd.DataFrame:
    """DataFrame with two distinct replicate groups (3 + 2 rows)."""
    return pd.DataFrame(
        {
            "temperature": [100, 100, 100, 200, 200, 300],
            "pressure": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
            "catalyst": ["A", "A", "A", "B", "B", "C"],
            "yield": [0.80, 0.82, 0.81, 0.65, 0.67, 0.55],
        }
    )


@pytest.fixture()
def excel_file(tmp_path: Path, simple_df: pd.DataFrame) -> Path:
    """Save simple_df to an Excel file and return its path."""
    p = tmp_path / "test_data.xlsx"
    simple_df.to_excel(p, index=False)
    return p


@pytest.fixture()
def two_target_df() -> pd.DataFrame:
    """DataFrame with two target columns (yield + selectivity)."""
    return pd.DataFrame(
        {
            "temperature": [100, 100, 200],
            "pressure": [1.0, 1.0, 2.0],
            "yield": [0.80, 0.84, 0.70],
            "selectivity": [0.90, 0.92, 0.85],
        }
    )


# ---------------------------------------------------------------------------
# load_file
# ---------------------------------------------------------------------------


class TestLoadFile:
    def test_returns_dataframe(self, simple_csv: Path) -> None:
        df = load_file(simple_csv)
        assert isinstance(df, pd.DataFrame)

    def test_loads_csv_shape(self, simple_csv: Path) -> None:
        df = load_file(simple_csv)
        assert df.shape == (4, 4)

    def test_loads_csv_columns(self, simple_csv: Path) -> None:
        df = load_file(simple_csv)
        assert list(df.columns) == [
            "temperature", "pressure", "catalyst", "yield"
        ]

    def test_loads_csv_accepts_str_path(self, simple_csv: Path) -> None:
        df = load_file(str(simple_csv))
        assert not df.empty

    def test_loads_excel_shape(
        self, excel_file: Path, simple_df: pd.DataFrame
    ) -> None:
        df = load_file(excel_file)
        assert df.shape == simple_df.shape

    def test_loads_excel_columns(
        self, excel_file: Path, simple_df: pd.DataFrame
    ) -> None:
        df = load_file(excel_file)
        assert list(df.columns) == list(simple_df.columns)

    def test_raises_on_unsupported_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "data.txt"
        p.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_file(p)

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_file(tmp_path / "does_not_exist.csv")

    def test_raises_on_json_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        p.write_text('{"a": 1}')
        with pytest.raises(ValueError):
            load_file(p)

    def test_csv_with_header_only_returns_empty_df(
        self, tmp_path: Path
    ) -> None:
        """A CSV with only a header row returns a 0-row DataFrame."""
        p = tmp_path / "header_only.csv"
        p.write_text("temperature,pressure,yield\n")
        df = load_file(p)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["temperature", "pressure", "yield"]

    def test_xls_extension_accepted(self, tmp_path: Path) -> None:
        """Extension check must accept .xls, not raise ValueError."""
        # Write intentionally corrupt bytes — only the extension guard
        # is tested here, not the parse step.
        p = tmp_path / "data.xls"
        p.write_bytes(b"")
        with pytest.raises(Exception) as exc_info:
            load_file(p)
        assert "Unsupported file extension" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# detect_replicates
# ---------------------------------------------------------------------------


class TestDetectReplicates:
    def test_returns_dataframe(self, simple_df: pd.DataFrame) -> None:
        result = detect_replicates(simple_df, FEAT_COLS)
        assert isinstance(result, pd.DataFrame)

    def test_detects_correct_row_count(
        self, simple_df: pd.DataFrame
    ) -> None:
        result = detect_replicates(simple_df, FEAT_COLS)
        assert len(result) == 2  # rows 0 and 1 are replicates

    def test_replicate_group_column_present(
        self, simple_df: pd.DataFrame
    ) -> None:
        result = detect_replicates(simple_df, FEAT_COLS)
        assert "replicate_group" in result.columns

    def test_replicate_group_is_integer_dtype(
        self, simple_df: pd.DataFrame
    ) -> None:
        result = detect_replicates(simple_df, FEAT_COLS)
        assert pd.api.types.is_integer_dtype(result["replicate_group"])

    def test_replicate_group_starts_at_zero(
        self, simple_df: pd.DataFrame
    ) -> None:
        result = detect_replicates(simple_df, FEAT_COLS)
        assert result["replicate_group"].min() == 0

    def test_non_replicate_rows_excluded(
        self, simple_df: pd.DataFrame
    ) -> None:
        result = detect_replicates(simple_df, FEAT_COLS)
        returned_temps = set(result["temperature"].tolist())
        # Rows with temperature 200 and 300 are unique — must not appear.
        assert 200 not in returned_temps
        assert 300 not in returned_temps

    def test_returns_empty_when_no_replicates(
        self, no_replicate_df: pd.DataFrame
    ) -> None:
        result = detect_replicates(no_replicate_df, FEAT_COLS)
        assert result.empty

    def test_empty_result_has_replicate_group_column(
        self, no_replicate_df: pd.DataFrame
    ) -> None:
        result = detect_replicates(no_replicate_df, FEAT_COLS)
        assert "replicate_group" in result.columns

    def test_multiple_groups_numbered_correctly(
        self, multi_replicate_df: pd.DataFrame
    ) -> None:
        result = detect_replicates(multi_replicate_df, FEAT_COLS)
        assert result["replicate_group"].nunique() == 2
        assert set(result["replicate_group"].unique()) == {0, 1}

    def test_multiple_groups_row_counts(
        self, multi_replicate_df: pd.DataFrame
    ) -> None:
        result = detect_replicates(multi_replicate_df, FEAT_COLS)
        counts = (
            result.groupby("replicate_group").size().sort_index().tolist()
        )
        assert sorted(counts) == [2, 3]

    def test_raises_on_empty_feature_cols(
        self, simple_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="feature_cols"):
            detect_replicates(simple_df, [])

    def test_raises_on_missing_feature_col(
        self, simple_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            detect_replicates(simple_df, ["nonexistent_column"])

    def test_original_index_preserved(self, simple_df: pd.DataFrame) -> None:
        """Replicate rows should retain their original DataFrame index."""
        result = detect_replicates(simple_df, FEAT_COLS)
        assert set(result.index).issubset(set(simple_df.index))

    def test_all_rows_replicate(self) -> None:
        """All rows returned when every row shares the same condition."""
        df = pd.DataFrame({"x": [1, 1, 1], "y": [0.5, 0.6, 0.7]})
        result = detect_replicates(df, ["x"])
        assert len(result) == 3
        assert result["replicate_group"].nunique() == 1


# ---------------------------------------------------------------------------
# handle_replicates
# ---------------------------------------------------------------------------


class TestHandleReplicates:
    # --- average strategy ---

    def test_average_reduces_row_count(
        self, simple_df: pd.DataFrame
    ) -> None:
        processed, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "average"
        )
        # 4 rows with 1 replicate pair → 3 unique conditions.
        assert len(processed) == 3

    def test_average_values_correct(self, simple_df: pd.DataFrame) -> None:
        processed, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "average"
        )
        row = processed[processed["temperature"] == 100].iloc[0]
        expected_yield = (0.85 + 0.87) / 2
        assert math.isclose(row["yield"], expected_yield, rel_tol=1e-9)

    def test_average_unique_rows_unchanged(
        self, simple_df: pd.DataFrame
    ) -> None:
        processed, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "average"
        )
        row = processed[processed["temperature"] == 200].iloc[0]
        assert math.isclose(row["yield"], 0.60, rel_tol=1e-9)

    def test_average_returns_dataframe(
        self, simple_df: pd.DataFrame
    ) -> None:
        processed, summary = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "average"
        )
        assert isinstance(processed, pd.DataFrame)
        assert isinstance(summary, pd.DataFrame)

    def test_average_column_order(self, simple_df: pd.DataFrame) -> None:
        processed, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "average"
        )
        assert list(processed.columns) == FEAT_COLS + TARGET_COLS

    def test_average_index_reset(self, simple_df: pd.DataFrame) -> None:
        processed, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "average"
        )
        assert list(processed.index) == list(range(len(processed)))

    def test_average_no_replicates_unchanged(
        self, no_replicate_df: pd.DataFrame
    ) -> None:
        processed, summary = handle_replicates(
            no_replicate_df, FEAT_COLS, TARGET_COLS, "average"
        )
        assert len(processed) == len(no_replicate_df)
        assert summary.empty

    # --- keep_all strategy ---

    def test_keep_all_preserves_row_count(
        self, simple_df: pd.DataFrame
    ) -> None:
        processed, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "keep_all"
        )
        assert len(processed) == len(simple_df)

    def test_keep_all_preserves_values(
        self, simple_df: pd.DataFrame
    ) -> None:
        processed, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "keep_all"
        )
        pd.testing.assert_frame_equal(
            processed.reset_index(drop=True),
            simple_df.reset_index(drop=True),
        )

    def test_keep_all_returns_copy(self, simple_df: pd.DataFrame) -> None:
        processed, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "keep_all"
        )
        processed.iloc[0, 0] = -999
        assert simple_df.iloc[0, 0] != -999

    def test_keep_all_still_builds_summary(
        self, simple_df: pd.DataFrame
    ) -> None:
        _, summary = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "keep_all"
        )
        assert not summary.empty
        assert "n_replicates" in summary.columns

    # --- std_uncertainty strategy ---

    def test_std_uncertainty_reduces_rows(
        self, simple_df: pd.DataFrame
    ) -> None:
        processed, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "std_uncertainty"
        )
        assert len(processed) == 3

    def test_std_uncertainty_adds_std_columns(
        self, simple_df: pd.DataFrame
    ) -> None:
        processed, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "std_uncertainty"
        )
        assert "yield_std" in processed.columns

    def test_std_uncertainty_mean_matches_average(
        self, simple_df: pd.DataFrame
    ) -> None:
        avg, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "average"
        )
        std, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "std_uncertainty"
        )
        pd.testing.assert_series_equal(
            avg["yield"].reset_index(drop=True),
            std["yield"].reset_index(drop=True),
            check_names=False,
        )

    def test_std_uncertainty_correct_std_value(
        self, simple_df: pd.DataFrame
    ) -> None:
        processed, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "std_uncertainty"
        )
        row = processed[processed["temperature"] == 100].iloc[0]
        expected_std = pd.Series([0.85, 0.87]).std()
        assert math.isclose(row["yield_std"], expected_std, rel_tol=1e-9)

    def test_std_for_non_replicate_is_nan(
        self, simple_df: pd.DataFrame
    ) -> None:
        """Single-occurrence rows should have NaN standard deviation."""
        processed, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "std_uncertainty"
        )
        row = processed[processed["temperature"] == 200].iloc[0]
        assert math.isnan(row["yield_std"])

    def test_std_column_order(self, simple_df: pd.DataFrame) -> None:
        processed, _ = handle_replicates(
            simple_df, FEAT_COLS, TARGET_COLS, "std_uncertainty"
        )
        expected = FEAT_COLS + TARGET_COLS + ["yield_std"]
        assert list(processed.columns) == expected

    # --- replicate_summary ---

    def test_summary_not_empty_when_replicates_present(
        self, simple_df: pd.DataFrame
    ) -> None:
        _, summary = handle_replicates(simple_df, FEAT_COLS, TARGET_COLS)
        assert not summary.empty

    def test_summary_empty_when_no_replicates(
        self, no_replicate_df: pd.DataFrame
    ) -> None:
        _, summary = handle_replicates(
            no_replicate_df, FEAT_COLS, TARGET_COLS
        )
        assert summary.empty

    def test_summary_has_n_replicates_column(
        self, simple_df: pd.DataFrame
    ) -> None:
        _, summary = handle_replicates(simple_df, FEAT_COLS, TARGET_COLS)
        assert "n_replicates" in summary.columns

    def test_summary_n_replicates_value(
        self, simple_df: pd.DataFrame
    ) -> None:
        _, summary = handle_replicates(simple_df, FEAT_COLS, TARGET_COLS)
        assert summary["n_replicates"].iloc[0] == 2

    def test_summary_has_target_mean_column(
        self, simple_df: pd.DataFrame
    ) -> None:
        _, summary = handle_replicates(simple_df, FEAT_COLS, TARGET_COLS)
        assert "yield_mean" in summary.columns

    def test_summary_has_target_std_column(
        self, simple_df: pd.DataFrame
    ) -> None:
        _, summary = handle_replicates(simple_df, FEAT_COLS, TARGET_COLS)
        assert "yield_std" in summary.columns

    def test_summary_one_row_per_group(
        self, multi_replicate_df: pd.DataFrame
    ) -> None:
        _, summary = handle_replicates(
            multi_replicate_df, FEAT_COLS, TARGET_COLS
        )
        assert len(summary) == 2

    # --- multi-target ---

    def test_two_targets_average(self, two_target_df: pd.DataFrame) -> None:
        feat = ["temperature", "pressure"]
        tgt = ["yield", "selectivity"]
        processed, _ = handle_replicates(
            two_target_df, feat, tgt, "average"
        )
        assert len(processed) == 2
        assert "yield" in processed.columns
        assert "selectivity" in processed.columns

    def test_two_targets_std_columns_added(
        self, two_target_df: pd.DataFrame
    ) -> None:
        feat = ["temperature", "pressure"]
        tgt = ["yield", "selectivity"]
        processed, _ = handle_replicates(
            two_target_df, feat, tgt, "std_uncertainty"
        )
        assert "yield_std" in processed.columns
        assert "selectivity_std" in processed.columns

    # --- validation / error cases ---

    def test_raises_on_invalid_strategy(
        self, simple_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            handle_replicates(  # type: ignore[arg-type]
                simple_df, FEAT_COLS, TARGET_COLS, "invalid"
            )

    def test_raises_on_missing_feature_col(
        self, simple_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            handle_replicates(simple_df, ["nonexistent"], TARGET_COLS)

    def test_raises_on_missing_target_col(
        self, simple_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            handle_replicates(simple_df, FEAT_COLS, ["nonexistent"])

    def test_raises_on_empty_feature_cols(
        self, simple_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="feature_cols"):
            handle_replicates(simple_df, [], TARGET_COLS)

    def test_raises_on_empty_target_cols(
        self, simple_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="target_cols"):
            handle_replicates(simple_df, FEAT_COLS, [])

    def test_raises_on_overlapping_feature_target_cols(
        self, simple_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="appear in both"):
            handle_replicates(
                simple_df, FEAT_COLS + ["yield"], ["yield"]
            )

    # --- edge cases ---

    def test_single_row_df_average(self) -> None:
        df = pd.DataFrame({"x": [1], "y": [0.5]})
        processed, summary = handle_replicates(
            df, ["x"], ["y"], "average"
        )
        assert len(processed) == 1
        assert summary.empty

    def test_all_replicates_average(self) -> None:
        """Every row is a replicate of the same condition."""
        df = pd.DataFrame({"x": [1, 1, 1], "y": [0.3, 0.5, 0.7]})
        processed, summary = handle_replicates(df, ["x"], ["y"], "average")
        assert len(processed) == 1
        assert math.isclose(processed["y"].iloc[0], 0.5, rel_tol=1e-9)
        assert summary["n_replicates"].iloc[0] == 3

    def test_empty_df_returns_empty(self) -> None:
        """A 0-row DataFrame should round-trip cleanly through average."""
        df = pd.DataFrame(columns=["temperature", "pressure", "yield"])
        processed, summary = handle_replicates(
            df, ["temperature", "pressure"], ["yield"]
        )
        assert isinstance(processed, pd.DataFrame)
        assert len(processed) == 0
        assert summary.empty

    def test_extra_metadata_column_kept(self) -> None:
        """Non-feature, non-target columns should survive aggregation."""
        df = pd.DataFrame(
            {
                "temperature": [100, 100, 200],
                "pressure": [1.0, 1.0, 2.0],
                "yield": [0.80, 0.84, 0.70],
                "experimenter": ["Alice", "Alice", "Bob"],
            }
        )
        processed, _ = handle_replicates(
            df, ["temperature", "pressure"], ["yield"], "average"
        )
        assert "experimenter" in processed.columns


# ---------------------------------------------------------------------------
# drop_missing_rows
# ---------------------------------------------------------------------------


class TestDropMissingRows:
    def test_no_missing_returns_unchanged(self) -> None:
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        cleaned, dropped = drop_missing_rows(df)
        assert len(cleaned) == 3
        assert dropped == []

    def test_drops_row_with_nan(self) -> None:
        df = pd.DataFrame({"A": [1.0, None, 3.0], "B": ["x", "y", "z"]})
        cleaned, dropped = drop_missing_rows(df)
        assert len(cleaned) == 2
        assert dropped == [1]

    def test_drops_multiple_rows(self) -> None:
        df = pd.DataFrame({"A": [1.0, None, 3.0, None], "B": ["x", "y", "z", "w"]})
        cleaned, dropped = drop_missing_rows(df)
        assert len(cleaned) == 2
        assert dropped == [1, 3]

    def test_returned_index_is_reset(self) -> None:
        df = pd.DataFrame({"A": [1.0, None, 3.0]})
        cleaned, _ = drop_missing_rows(df)
        assert list(cleaned.index) == [0, 1]

    def test_cols_limits_check_to_subset(self) -> None:
        """NaN in a non-checked column should not cause a row to be dropped."""
        df = pd.DataFrame({"A": [1.0, None, 3.0], "B": [None, "y", "z"]})
        cleaned, dropped = drop_missing_rows(df, cols=["A"])
        # Only row 1 (NaN in A) should be dropped; row 0 (NaN in B) is kept.
        assert dropped == [1]
        assert len(cleaned) == 2

    def test_all_rows_missing_returns_empty(self) -> None:
        df = pd.DataFrame({"A": [None, None], "B": [None, None]})
        cleaned, dropped = drop_missing_rows(df)
        assert len(cleaned) == 0
        assert dropped == [0, 1]

    def test_original_df_not_mutated(self) -> None:
        df = pd.DataFrame({"A": [1.0, None, 3.0]})
        original_len = len(df)
        drop_missing_rows(df)
        assert len(df) == original_len
