"""Unit tests for code_review tool handler."""

import sys
import os
import pytest

# Add tool directory to path so we can import handler directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'code_review'))

import handler

pytestmark = pytest.mark.unit

SIMPLE_DIFF = "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n+new line\n-old line\n"


class TestWebhookParsing:
    def test_empty_diff_returns_silent(self):
        result = handler.execute("", {"_webhook": {"diff": ""}}, {}, {})
        assert result.get("output") is None

    def test_no_diff_key_returns_silent(self):
        result = handler.execute("", {"_webhook": {}}, {}, {})
        assert result.get("output") is None

    def test_no_webhook_key_returns_silent(self):
        result = handler.execute("", {}, {}, {})
        assert result.get("output") is None

    def test_valid_diff_returns_tool_output(self):
        result = handler.execute("", {"_webhook": {"diff": SIMPLE_DIFF}}, {}, {})
        assert result.get("output") == "tool"
        assert "text" in result
        assert len(result["text"]) > 0

    def test_result_includes_repo_name(self):
        result = handler.execute(
            "", {"_webhook": {"repo": "myorg/myrepo", "diff": SIMPLE_DIFF}}, {}, {}
        )
        assert "myorg/myrepo" in result["text"]

    def test_result_includes_author(self):
        result = handler.execute(
            "", {"_webhook": {"author": "alice", "diff": SIMPLE_DIFF}}, {}, {}
        )
        assert "alice" in result["text"]

    def test_result_includes_branch(self):
        result = handler.execute(
            "", {"_webhook": {"branch": "feat/new-thing", "diff": SIMPLE_DIFF}}, {}, {}
        )
        assert "feat/new-thing" in result["text"]

    def test_commits_included(self):
        commits = [{"sha": "abc1234ef", "message": "Add feature X"}]
        result = handler.execute(
            "", {"_webhook": {"diff": SIMPLE_DIFF, "commits": commits}}, {}, {}
        )
        assert "abc1234" in result["text"] or "Add feature X" in result["text"]

    def test_output_text_within_size_limit(self):
        big_diff = SIMPLE_DIFF * 200
        result = handler.execute("", {"_webhook": {"diff": big_diff}}, {}, {})
        # Should not exceed MAX_OUTPUT_CHARS significantly
        assert len(result.get("text", "")) <= handler.MAX_OUTPUT_CHARS + 500


class TestDiffFiltering:
    def test_package_lock_skipped(self):
        assert handler._should_skip_file("package-lock.json")

    def test_yarn_lock_skipped(self):
        assert handler._should_skip_file("yarn.lock")

    def test_poetry_lock_skipped(self):
        assert handler._should_skip_file("poetry.lock")

    def test_minified_js_skipped(self):
        assert handler._should_skip_file("dist/bundle.min.js")

    def test_png_skipped(self):
        assert handler._should_skip_file("logo.png")

    def test_source_file_not_skipped(self):
        assert not handler._should_skip_file("src/main.py")

    def test_regular_js_not_skipped(self):
        assert not handler._should_skip_file("src/app.js")

    def test_security_sensitive_auth(self):
        assert handler._is_security_sensitive("auth_service.py")

    def test_security_sensitive_crypto(self):
        assert handler._is_security_sensitive("crypto_utils.py")

    def test_security_sensitive_password(self):
        assert handler._is_security_sensitive("password_handler.py")

    def test_regular_file_not_security_sensitive(self):
        assert not handler._is_security_sensitive("dashboard.py")
        assert not handler._is_security_sensitive("utils/formatting.py")


class TestDiffParsing:
    def test_parses_single_file(self):
        diff = "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n+new\n-old\n"
        files = handler._parse_diff_files(diff)
        assert len(files) == 1
        assert files[0]["filename"] == "foo.py"
        assert files[0]["additions"] == 1
        assert files[0]["deletions"] == 1

    def test_parses_multiple_files(self):
        diff = (
            "diff --git a/a.py b/a.py\n+added\n"
            "diff --git a/b.py b/b.py\n-removed\n"
        )
        files = handler._parse_diff_files(diff)
        assert len(files) == 2
        assert files[0]["filename"] == "a.py"
        assert files[1]["filename"] == "b.py"

    def test_counts_additions_and_deletions(self):
        diff = "diff --git a/x.py b/x.py\n+line1\n+line2\n-old\n"
        files = handler._parse_diff_files(diff)
        assert files[0]["additions"] == 2
        assert files[0]["deletions"] == 1

    def test_empty_diff_returns_empty_list(self):
        assert handler._parse_diff_files("") == []

    def test_header_lines_not_counted(self):
        diff = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n+real addition\n"
        files = handler._parse_diff_files(diff)
        # --- and +++ lines should not be counted
        assert files[0]["additions"] == 1
        assert files[0]["deletions"] == 0


class TestIntelligentChunking:
    def test_output_within_limit(self):
        files = [
            {"filename": "main.py", "lines": ["+new"] * 100, "additions": 100, "deletions": 0}
        ]
        result = handler._chunk_diff(files, 500)
        assert len(result) <= 500 + 50  # small slack for headers

    def test_security_files_prioritized(self):
        files = [
            {"filename": "readme.md", "lines": ["+doc"] * 5, "additions": 5, "deletions": 0},
            {"filename": "auth.py", "lines": ["+auth"] * 5, "additions": 5, "deletions": 0},
        ]
        result = handler._chunk_diff(files, 2000)
        auth_pos = result.find("auth.py")
        readme_pos = result.find("readme.md")
        assert auth_pos < readme_pos  # Security file appears first in output

    def test_summary_always_present(self):
        files = [
            {"filename": "foo.py", "lines": ["+x"], "additions": 1, "deletions": 0}
        ]
        result = handler._chunk_diff(files, 2000)
        assert "Changed Files" in result
        assert "foo.py" in result

    def test_large_file_shows_omission_marker(self):
        lines = [f"+line {i}" for i in range(100)]
        files = [{"filename": "big.py", "lines": lines, "additions": 100, "deletions": 0}]
        result = handler._chunk_diff(files, 400)
        assert "omitted" in result

    def test_skipped_lockfiles_noted_in_summary(self):
        lock_files = [
            {"filename": "package-lock.json", "lines": ["+lock"] * 5, "additions": 5, "deletions": 0}
        ]
        # Only the lockfile; it should be skipped and noted
        result = handler._chunk_diff(lock_files, 1000)
        assert "Skipped" in result or "package-lock.json" not in result.split("##")[0:1][0]

    def test_empty_files_list_returns_summary_only(self):
        result = handler._chunk_diff([], 500)
        assert isinstance(result, str)
        assert "Changed Files" in result


class TestContinuation:
    def test_chalie_response_surfaces_as_prompt(self):
        result = handler.execute(
            "",
            {"_chalie": {"text": "This looks clean overall."}},
            {},
            {}
        )
        assert result.get("output") == "prompt"
        assert result.get("text") == "This looks clean overall."

    def test_chalie_empty_response(self):
        result = handler.execute(
            "",
            {"_chalie": {"text": ""}},
            {},
            {}
        )
        assert result.get("output") == "prompt"
        assert result.get("text") == ""

    def test_chalie_takes_priority_over_webhook(self):
        # If both _chalie and _webhook are present, _chalie takes priority
        result = handler.execute(
            "",
            {"_chalie": {"text": "looks good"}, "_webhook": {"diff": SIMPLE_DIFF}},
            {},
            {}
        )
        assert result.get("output") == "prompt"
        assert result.get("text") == "looks good"
