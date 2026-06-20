
import json
import re
from typing import cast


def extract_body(tag_name: str, output: str) -> str:
    opener_pattern = re.compile(
        r"^\[" + re.escape(tag_name) + r"\([^)]*\)\]",
        re.MULTILINE,
    )
    terminator = f"[end:{tag_name}]"

    assert opener_pattern.search(output), (
        f"Expected opener '[{tag_name}(...)]' not found in: {output!r}"
    )
    assert terminator in output, (
        f"Expected '{terminator}' not found in: {output!r}"
    )

    # Body is everything between the first newline after the opener and
    # the last newline before the terminator.
    after_opener = output.split("\n", 1)
    if len(after_opener) < 2:
        return ""
    remainder = after_opener[1]
    before_terminator = remainder.rsplit(f"\n{terminator}", 1)
    return before_terminator[0] if len(before_terminator) > 1 else ""


def extract_json(tag_name: str, output: str) -> "dict[str, object] | list[object]":
    body = extract_body(tag_name, output)
    return cast("dict[str, object] | list[object]", json.loads(body))


def has_opener(tag_name: str, output: str) -> bool:
    """Return True when '[<tag_name>(...)]' opener exists in output."""
    return bool(re.search(r"\[" + re.escape(tag_name) + r"\([^)]*\)\]", output))


def has_terminator(tag_name: str, output: str) -> bool:
    """Return True when '[end:<tag_name>]' terminator exists in output."""
    return f"[end:{tag_name}]" in output


def assert_both_markers(tag_name: str, output: str) -> None:
    """Assert both opener and terminator are present."""
    assert has_opener(tag_name, output), (
        f"Missing opener '[{tag_name}(...)]' in: {output!r}"
    )
    assert has_terminator(tag_name, output), (
        f"Missing '[end:{tag_name}]' in: {output!r}"
    )


_OPENER_RE = re.compile(r"\[[^\]()]+\(([^)]*)\)\]")


def has_error_arg(output: str) -> bool:
    """Return True when any opener contains an 'error=...' arg.

    Two-stage match avoids the polynomial-backtracking risk Sonar flags
    on a single nested-quantifier regex (S5852). The outer pattern uses
    disjoint character classes (`[^\\]()]+` then `[^)]*`), and the inner
    `error=` lookup is run only on the captured arg list.
    """
    for opener in _OPENER_RE.finditer(output):
        if "error=" in opener.group(1):
            return True
    return False
