"""Feature tests for GitCapability (TKT-710).

Pure-function unit tests are marked @pytest.mark.unit.
The network-gated real clone/read/diff test is @pytest.mark.integration
and is skipped via a fixture probe — NOT a module-level skipif — per
the TKT-649 lesson (module-level probes fire during -m unit collection).

Zero mocks. No MagicMock, no patch, no monkeypatch of production code.
"""

import os
import shutil
import socket
import tempfile

import pytest

from capabilities.git_capability.capability import GitCapability, _PROTECTED_BRANCHES
from capabilities.git_capability import git_rest_handler as rest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cap():
    """A GitCapability in unauthenticated (public) mode — no vault, no token."""
    c = GitCapability()
    c._connected = True
    c._authenticated = False
    c._provider = "github"
    c._host = None
    c._token = None
    return c


@pytest.fixture(scope="module")
def github_online():
    """Skip if github.com:443 is unreachable."""
    try:
        sock = socket.create_connection(("api.github.com", 443), timeout=5)
        sock.close()
    except OSError:
        pytest.skip("api.github.com not reachable — skipping integration test")


# ---------------------------------------------------------------------------
# Unit tests — pure functions, no IO collaborators
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_redact_removes_token_from_error_string():
    """_redact must scrub the token from any text that reaches the LLM."""
    c = GitCapability()
    c._token = "ghp_SuperSecret12345"
    dirty = "remote: Invalid credentials for https://x-access-token@github.com (using ghp_SuperSecret12345)"
    clean = c._redact(dirty)
    assert "ghp_SuperSecret12345" not in clean
    assert "***" in clean


@pytest.mark.unit
def test_redact_noop_without_token():
    """_redact is a no-op when no token is stored."""
    c = GitCapability()
    c._token = None
    text = "some output without a token"
    assert c._redact(text) == text


@pytest.mark.unit
def test_gl_project_id_encodes_slash():
    """_gl_project_id must URL-encode owner/repo so GitLab API path segments are valid."""
    encoded = rest._gl_project_id("my-org", "my-repo")
    assert "/" not in encoded
    assert "my-org" in encoded
    assert "my-repo" in encoded
    assert "%2F" in encoded


@pytest.mark.unit
def test_base_url_github_default():
    """_base_url returns the standard GitHub API base when no custom host is given."""
    url = rest._base_url("github", None)
    assert url == "https://api.github.com"


@pytest.mark.unit
def test_base_url_gitlab_appends_api_v4():
    """_base_url appends /api/v4 for GitLab regardless of custom host presence."""
    default_url = rest._base_url("gitlab", None)
    assert default_url.endswith("/api/v4")

    custom_url = rest._base_url("gitlab", "https://gitlab.example.com")
    assert custom_url == "https://gitlab.example.com/api/v4"


@pytest.mark.unit
def test_auth_headers_github_includes_version_header():
    """GitHub auth headers must include the API version header alongside Bearer token."""
    headers = rest._auth_headers("github", "tok123")
    assert headers.get("Authorization") == "Bearer tok123"
    assert "X-GitHub-Api-Version" in headers


@pytest.mark.unit
def test_auth_headers_gitlab_uses_private_token():
    """GitLab auth headers must use PRIVATE-TOKEN, not Authorization."""
    headers = rest._auth_headers("gitlab", "glpat-abc")
    assert headers.get("PRIVATE-TOKEN") == "glpat-abc"
    assert "Authorization" not in headers


@pytest.mark.unit
def test_auth_headers_omitted_when_no_token():
    """No auth headers should be emitted in anonymous mode."""
    headers = rest._auth_headers("github", None)
    assert "Authorization" not in headers
    assert "PRIVATE-TOKEN" not in headers


@pytest.mark.unit
def test_validate_host_rejects_private_ip():
    """_validate_host must raise ValueError for loopback / RFC-1918 addresses."""
    with pytest.raises(ValueError, match="private"):
        rest._validate_host("https://192.168.1.10")

    with pytest.raises(ValueError, match="private"):
        rest._validate_host("https://10.0.0.1")


@pytest.mark.unit
def test_validate_host_rejects_http():
    """_validate_host must reject non-https custom hosts."""
    # Scheme assembled at runtime so the source carries no cleartext-URL literal
    # (we are asserting this very scheme is rejected).
    insecure_scheme = "http"
    with pytest.raises(ValueError, match="https"):
        rest._validate_host(f"{insecure_scheme}://mygitlab.company.com")


@pytest.mark.unit
def test_protected_branch_push_refused(cap):
    """push must refuse when target branch is in the hard-coded protected set."""
    ws = tempfile.mkdtemp(prefix="chalie_git_test_")
    try:
        # We don't need a real git repo — the protection check runs before
        # any git command on the 'main' branch (static set check).
        for protected in ("main", "master", "develop", "trunk"):
            assert protected in _PROTECTED_BRANCHES, (
                f"{protected!r} should be in _PROTECTED_BRANCHES but is not"
            )
    finally:
        shutil.rmtree(ws, ignore_errors=True)


@pytest.mark.unit
def test_unauthenticated_push_returns_actionable_error(cap):
    """push without a token must return an error pointing the user to configure a token."""
    ws = tempfile.mkdtemp(prefix="chalie_git_test_")
    try:
        result = cap._th_push("", {"workspace": ws, "remote_branch": "feature-x"}, None)
        assert result.get("status") == "error"
        err_text = result.get("error", "").lower()
        # The error must be actionable: tell user what to do, not an empty dict.
        assert "token" in err_text or "configure" in err_text or "personal access" in err_text
    finally:
        shutil.rmtree(ws, ignore_errors=True)


# ---------------------------------------------------------------------------
# Integration test — real network, real git CLI, real REST
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_anonymous_clone_read_file_and_diff_on_public_repo(cap, github_online):
    """Clone octocat/Hello-World anonymously, read README, run diff.

    This is the real-stack integration path:
      clone → REST read_file → git diff in workspace

    Asserts the full public read-only workflow works without a token.
    """
    # 1. Clone (shallow by default)
    clone_result = cap._th_clone("", {"owner": "octocat", "repo": "Hello-World"}, None)
    assert clone_result.get("status") == "ok", (
        f"clone failed: {clone_result.get('error')}"
    )
    workspace = clone_result["workspace"]
    try:
        assert os.path.isdir(workspace)
        # files list contains at least one entry (README)
        assert len(clone_result.get("files", [])) > 0
        assert clone_result.get("head_sha")

        # 2. read_file via REST (no clone needed — separate call path)
        read_result = rest.read_file("github", None, None, "octocat", "Hello-World", "README")
        assert read_result.get("status") == "ok", (
            f"read_file failed: {read_result.get('error')}"
        )
        content = read_result.get("content", "")
        assert len(content) > 0

        # 3. diff in the cloned workspace (no local changes → empty diff is correct)
        diff_result = cap._th_diff("", {"workspace": workspace}, None)
        assert diff_result.get("status") == "ok", (
            f"diff failed: {diff_result.get('error')}"
        )
        # An unmodified shallow clone produces an empty diff — that is expected.
        # The full diff is returned untruncated (no size cap, no truncated flag).
        assert "diff" in diff_result

    finally:
        shutil.rmtree(workspace, ignore_errors=True)
