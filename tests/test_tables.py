import numpy as np
import pandas as pd

from aqlab.tables import format_cell, markdown_table


def test_format_cell_basics():
    assert format_cell(None) == ""
    assert format_cell(float("nan")) == "n/a"
    assert format_cell(3.0) == "3"
    assert format_cell(128.39) == "128.39"
    assert format_cell(0.7784000000000001).startswith("0.7784")
    assert format_cell(7) == "7"
    assert format_cell(True) == "true"
    assert format_cell("SYN001") == "SYN001"


def test_format_cell_escapes_pipes_and_newlines():
    assert format_cell("a|b") == "a\\|b"
    assert format_cell("line1\nline2") == "line1 line2"


def test_markdown_table_structure_and_alignment():
    df = pd.DataFrame(
        {
            "策略": ["ma_cross", "momentum"],
            "总收益%": [55.8, 9.83],
            "交易数": [12, 367],
            "备注": ["a|b", "ok"],
        }
    )
    text = markdown_table(df)
    lines = text.splitlines()
    assert lines[0] == "| 策略 | 总收益% | 交易数 | 备注 |"
    assert lines[1] == "| --- | ---: | ---: | --- |"
    assert "| ma_cross | 55.8 | 12 | a\\|b |" in text
    assert text.count("\n") == 3


def test_markdown_table_handles_nan_and_index():
    df = pd.DataFrame({"score": [1.0, np.nan]}, index=["A", "B"])
    text = markdown_table(df, index=True)
    assert "| index | score |" in text
    assert "| A | 1 |" in text
    assert "| B | n/a |" in text


def test_markdown_table_empty():
    assert markdown_table(pd.DataFrame()) == ""


def test_reports_never_require_tabulate(monkeypatch):
    """The report path must work even if tabulate is absent (this broke CI once)."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tabulate":
            raise ImportError("tabulate is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert "| a | b |" in markdown_table(df)
