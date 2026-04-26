"""Helpers for asserting the canonical skill-output tag format.

Format:
  [<name>(k1=v1, k2=v2)]
  <optional body>
  [end:<name>]

Use these helpers instead of repeating regex inline in every test file.
"""

import json
import re


def extract_body(tag_name: str, output: str) -> str:
    """Return the body text between the opener and [end:<tag_name>].

    Returns an empty string when there is no body (error-only tags).
    Raises AssertionError when the opener or terminator is missing.
    """
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


def extract_json(tag_name: str, output: str) -> dict | list:
    """Return the JSON-parsed body between the tag markers.

    Raises AssertionError on missing markers.
    Raises json.JSONDecodeError if body is not valid JSON.
    """
    body = extract_body(tag_name, output)
    return json.loads(body)


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


def has_error_arg(output: str, keyword: str | None = None) -> bool:
    """Return True when the opener contains 'error=...' (optionally matching keyword)."""
    opener_match = re.search(r"\[[^\]]+\([^)]*error=([^,)]+)[^)]*\)\]", output)
    if not opener_match:
        return False
    if keyword is None:
        return True
    return keyword in opener_match.group(1)
