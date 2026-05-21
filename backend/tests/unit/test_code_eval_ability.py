import pytest

from abilities.code_eval import CodeEvalAbility


def _run(code: str) -> dict:
    return CodeEvalAbility().execute(channel="user", params={"code": code}, telemetry=None)


@pytest.mark.unit
class TestCodeEvalAbility:
    def test_print_output_returned(self):
        result = _run("print(2+2)")
        assert "4" in result["text"]

    def test_no_print_returns_hint(self):
        result = _run("x = 42")
        assert result["text"] == "(no output — wrap your result in print() to emit it)"

    def test_syntax_error_message(self):
        result = _run("def broken(")
        assert "Syntax error" in result["text"]

    def test_runtime_error_message(self):
        result = _run("print('partial')\n1/0")
        assert "ZeroDivisionError" in result["text"]
        assert "partial" in result["text"]

    def test_empty_code_returns_no_code_message(self):
        result = _run("")
        assert result["text"] == "No code provided"
