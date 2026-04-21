"""
Unit tests for backend/tools/code_eval.py — PrintCollector capture.

All tests run the real RestrictedPython + PrintCollector stack; zero mocks.
"""

import pytest

from tools.code_eval import execute


@pytest.mark.unit
def test_single_print_captured():
    result = execute("test", {"code": "print('hello')"})
    assert result["error"] == ""
    assert result["text"].strip() == "hello"


@pytest.mark.unit
def test_multiple_prints_captured():
    code = "print('line1')\nprint('line2')\nprint('line3')"
    result = execute("test", {"code": code})
    assert result["error"] == ""
    text = result["text"]
    assert "line1" in text
    assert "line2" in text
    assert "line3" in text


@pytest.mark.unit
def test_print_with_fstring():
    """Scenario 052 regression: f-string formatted print must return the value."""
    code = "monthly_payment = 1234.56\nprint(f'{monthly_payment:.2f}')"
    result = execute("test", {"code": code})
    assert result["error"] == ""
    assert result["text"].strip() == "1234.56"


@pytest.mark.unit
def test_exception_path_returns_captured_text_and_error():
    code = "print('before error')\nraise ValueError('boom')"
    result = execute("test", {"code": code})
    assert "before error" in result["text"]
    assert "ValueError" in result["error"]
    assert "boom" in result["error"]


@pytest.mark.unit
def test_no_print_returns_empty_text():
    result = execute("test", {"code": "x = 1 + 1"})
    assert result["error"] == ""
    assert result["text"] == ""
