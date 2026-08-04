import pandas as pd
import pytest

from dse_practicum.common import fill_submission_template


def test_fill_submission_preserves_template_order(tmp_path):
    template = tmp_path / "sample.csv"
    pd.DataFrame({"ID": ["b", "a"], "ESTIMATE": [0.0, 0.0]}).to_csv(
        template, index=False
    )
    result = fill_submission_template(
        template,
        {"a": 1.0, "b": 2.0},
        id_column="ID",
        value_column="ESTIMATE",
    )
    assert result["ID"].tolist() == ["b", "a"]
    assert result["ESTIMATE"].tolist() == [2.0, 1.0]


def test_fill_submission_rejects_missing_rows(tmp_path):
    template = tmp_path / "sample.csv"
    pd.DataFrame({"ID": ["a"], "ESTIMATE": [0.0]}).to_csv(
        template, index=False
    )
    with pytest.raises(ValueError, match="No estimates"):
        fill_submission_template(
            template,
            {},
            id_column="ID",
            value_column="ESTIMATE",
        )

