"""Stateless REST client for the GitHub v3 and GitLab v4 APIs.

Every public function accepts explicit connection parameters rather than
storing them — GitCapability owns credential lifecycle.

Provider selection:
  - GitHub: base ``https://api.github.com`` (or custom GHE host)
  - GitLab: base ``https://gitlab.com/api/v4`` (or custom host + /api/v4)

Auth:
  - GitHub:  ``Authorization: Bearer <token>`` + ``X-GitHub-Api-Version: 2022-11-28``
  - GitLab:  ``PRIVATE-TOKEN: <token>``
  - Omit auth headers entirely when no token (public read-only mode).

SSRF: custom hosts are validated via is_private_url() before any request.
Rate-limit surfacing: GitHub 403/429 → read X-RateLimit-* headers; GitLab
  403/429 → read RateLimit-* headers. Never return empty-success on throttle.
SSL: certificate validation is always on — this client carries a PAT, so a
  silent verify=False fallback would risk leaking the token to a MITM.
Output bounding: list endpoints honour per_page/max caps; file/diff capped.
"""

from __future__ import annotations

from urllib.parse import quote

import requests

# is_private_url reused from the shared SSRF guard (TKT-619).
from abilities._ssrf import is_private_url

_TIMEOUT = 20
_MAX_PER_PAGE = 30
_MAX_FILE_BYTES = 512_000   # 500 KB cap on raw file content surfaced to LLM

_GH_API_BASE = "https://api.github.com"
_GH_VERSION_HEADER = "2022-11-28"

# Public hosts that bypass SSRF validation — validated externally by TLS cert.
_PUBLIC_HOSTS = frozenset({"api.github.com", "gitlab.com"})


# ── Internal helpers ──────────────────────────────────────────────────────────

def _base_url(provider: str, host: str | None) -> str:
    """Return the API base URL for the given provider and optional custom host."""
    if provider == "github":
        return (host.rstrip("/") if host else _GH_API_BASE)
    # GitLab — always append /api/v4
    gl_root = host.rstrip("/") if host else "https://gitlab.com"
    return f"{gl_root}/api/v4"


def _auth_headers(provider: str, token: str | None) -> dict:
    """Build provider-specific auth headers; omit auth when no token."""
    headers: dict = {"Accept": "application/json"}
    if token:
        if provider == "github":
            headers["Authorization"] = f"Bearer {token}"
            headers["X-GitHub-Api-Version"] = _GH_VERSION_HEADER
        else:
            headers["PRIVATE-TOKEN"] = token
    return headers


def _gl_project_id(owner: str, repo: str) -> str:
    """URL-encode ``owner/repo`` for GitLab project-id path segments."""
    return quote(f"{owner}/{repo}", safe="")


def _validate_host(host: str) -> None:
    """Raise ValueError if host resolves to a private/loopback/link-local address.

    Skips validation for the known public API hosts.
    Rejects schemeless hosts early — urlparse mis-parses them (hostname=None,
    scheme becomes the first path component), so custom hosts without https://
    would silently bypass the scheme and SSRF checks.
    """
    from urllib.parse import urlparse
    # Enforce https:// before any parsing — schemeless strings make urlparse
    # mis-identify the hostname (None) and would bypass the SSRF check below.
    if not host.startswith("https://"):
        raise ValueError(f"Custom host must use https://, got: {host!r}")
    parsed = urlparse(host)
    # Known public hosts are trusted without an SSRF lookup.
    if parsed.hostname in _PUBLIC_HOSTS:
        return
    if is_private_url(host):
        raise ValueError(
            f"Custom host {host!r} resolves to a private network address. "
            "Only public hosts are allowed."
        )


def _get(url: str, headers: dict, params: dict | None = None) -> requests.Response:
    """GET with certificate validation always on; returns raw Response."""
    return requests.get(url, headers=headers, params=params, timeout=_TIMEOUT)


def _post(url: str, headers: dict, json_body: dict) -> requests.Response:
    """POST with certificate validation always on; returns raw Response."""
    return requests.post(url, headers=headers, json=json_body, timeout=_TIMEOUT)


def _put(url: str, headers: dict, json_body: dict) -> requests.Response:
    """PUT with certificate validation always on; returns raw Response."""
    return requests.put(url, headers=headers, json=json_body, timeout=_TIMEOUT)


def _check_rate_limit(resp: requests.Response, provider: str) -> dict | None:
    """Return a structured error dict if the response indicates rate limiting.

    Returns None when not rate-limited so callers can continue processing.
    """
    if resp.status_code not in (403, 429):
        return None
    if provider == "github":
        remaining = resp.headers.get("X-RateLimit-Remaining")
        reset = resp.headers.get("X-RateLimit-Reset")
        if remaining is not None:
            return {
                "status": "error",
                "error": (
                    f"GitHub rate limit hit. Remaining: {remaining}. "
                    f"Reset at unix timestamp: {reset}. "
                    "Provide a Personal Access Token to increase the limit."
                ),
            }
    else:
        remaining = resp.headers.get("RateLimit-Remaining")
        reset = resp.headers.get("RateLimit-Reset")
        if remaining is not None:
            return {
                "status": "error",
                "error": (
                    f"GitLab rate limit hit. Remaining: {remaining}. "
                    f"Reset at unix timestamp: {reset}. "
                    "Provide a Personal Access Token to increase the limit."
                ),
            }
    return None


def _raise_for_status(resp: requests.Response, provider: str, context: str) -> dict | None:
    """Return a structured error dict if the response is an HTTP error.

    Returns None when the response is successful.
    """
    if resp.status_code < 400:
        return None
    rate_err = _check_rate_limit(resp, provider)
    if rate_err:
        return rate_err
    try:
        body = resp.json()
        message = body.get("message") or body.get("error") or str(body)
    except Exception:
        message = resp.text[:500]
    return {
        "status": "error",
        "error": f"[git:{context}] HTTP {resp.status_code}: {message}",
    }


# ── Public API functions ──────────────────────────────────────────────────────

def validate_token(provider: str, host: str | None, token: str) -> None:
    """Validate a token via a cheap authenticated GET /user call.

    Raises:
        ValueError: If the token is rejected (HTTP 401) or the host is invalid.
    """
    if host:
        _validate_host(host)
    base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    url = f"{base}/user"
    resp = _get(url, headers)
    if resp.status_code == 401:
        raise ValueError(
            f"Git token rejected by {provider.title()} (HTTP 401). "
            "Check your Personal Access Token."
        )


def get_repo(provider: str, host: str | None, token: str | None,
             owner: str, repo: str) -> dict:
    """Fetch repository metadata."""
    base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    if provider == "github":
        url = f"{base}/repos/{owner}/{repo}"
        resp = _get(url, headers)
        err = _raise_for_status(resp, provider, "get_repo")
        if err:
            return err
        data = resp.json()
        return {
            "status": "ok",
            "repo": {
                "full_name": data.get("full_name"),
                "description": data.get("description"),
                "default_branch": data.get("default_branch"),
                "private": data.get("private"),
                "stars": data.get("stargazers_count"),
                "forks": data.get("forks_count"),
                "open_issues": data.get("open_issues_count"),
                "language": data.get("language"),
                "clone_url": data.get("clone_url"),
                "html_url": data.get("html_url"),
            },
        }
    # GitLab
    pid = _gl_project_id(owner, repo)
    url = f"{base}/projects/{pid}"
    resp = _get(url, headers)
    err = _raise_for_status(resp, provider, "get_repo")
    if err:
        return err
    data = resp.json()
    return {
        "status": "ok",
        "repo": {
            "full_name": data.get("path_with_namespace"),
            "description": data.get("description"),
            "default_branch": data.get("default_branch"),
            "private": data.get("visibility") != "public",
            "stars": data.get("star_count"),
            "forks": data.get("forks_count"),
            "open_issues": data.get("open_issues_count"),
            "language": None,
            "clone_url": data.get("http_url_to_repo"),
            "html_url": data.get("web_url"),
        },
    }


def list_branches(provider: str, host: str | None, token: str | None,
                  owner: str, repo: str) -> dict:
    """List branches for a repository."""
    base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    if provider == "github":
        url = f"{base}/repos/{owner}/{repo}/branches"
        resp = _get(url, headers, params={"per_page": _MAX_PER_PAGE})
    else:
        pid = _gl_project_id(owner, repo)
        url = f"{base}/projects/{pid}/repository/branches"
        resp = _get(url, headers, params={"per_page": _MAX_PER_PAGE})
    err = _raise_for_status(resp, provider, "list_branches")
    if err:
        return err
    items = resp.json()
    branches = [b.get("name") for b in items]
    return {"status": "ok", "branches": branches}


def list_commits(provider: str, host: str | None, token: str | None,
                 owner: str, repo: str, branch: str | None = None) -> dict:
    """List recent commits for a repository."""
    base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    if provider == "github":
        url = f"{base}/repos/{owner}/{repo}/commits"
        params: dict = {"per_page": _MAX_PER_PAGE}
        if branch:
            params["sha"] = branch
        resp = _get(url, headers, params=params)
    else:
        pid = _gl_project_id(owner, repo)
        url = f"{base}/projects/{pid}/repository/commits"
        params = {"per_page": _MAX_PER_PAGE}
        if branch:
            params["ref_name"] = branch
        resp = _get(url, headers, params=params)
    err = _raise_for_status(resp, provider, "list_commits")
    if err:
        return err
    raw = resp.json()
    commits = []
    for c in raw:
        if provider == "github":
            commits.append({
                "sha": c.get("sha", "")[:10],
                "message": (c.get("commit", {}).get("message") or "").split("\n")[0],
                "author": c.get("commit", {}).get("author", {}).get("name"),
                "date": c.get("commit", {}).get("author", {}).get("date"),
            })
        else:
            commits.append({
                "sha": (c.get("id") or "")[:10],
                "message": (c.get("title") or ""),
                "author": c.get("author_name"),
                "date": c.get("committed_date"),
            })
    return {"status": "ok", "commits": commits}


def get_commit(provider: str, host: str | None, token: str | None,
               owner: str, repo: str, sha: str) -> dict:
    """Fetch details for a single commit."""
    base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    if provider == "github":
        url = f"{base}/repos/{owner}/{repo}/commits/{sha}"
        resp = _get(url, headers)
        err = _raise_for_status(resp, provider, "get_commit")
        if err:
            return err
        data = resp.json()
        files = [
            {"filename": f.get("filename"), "status": f.get("status"),
             "additions": f.get("additions"), "deletions": f.get("deletions")}
            for f in data.get("files", [])[:20]
        ]
        return {
            "status": "ok",
            "sha": data.get("sha", "")[:10],
            "message": data.get("commit", {}).get("message"),
            "author": data.get("commit", {}).get("author", {}).get("name"),
            "date": data.get("commit", {}).get("author", {}).get("date"),
            "files": files,
        }
    pid = _gl_project_id(owner, repo)
    url = f"{base}/projects/{pid}/repository/commits/{sha}/diff"
    resp = _get(url, headers)
    err = _raise_for_status(resp, provider, "get_commit")
    if err:
        return err
    diffs = resp.json()
    files = [
        {"filename": d.get("new_path"), "status": "modified",
         "additions": None, "deletions": None}
        for d in diffs[:20]
    ]
    return {"status": "ok", "sha": sha[:10], "files": files}


def list_prs(provider: str, host: str | None, token: str | None,
             owner: str, repo: str, state: str = "open") -> dict:
    """List pull/merge requests."""
    base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    if provider == "github":
        url = f"{base}/repos/{owner}/{repo}/pulls"
        resp = _get(url, headers, params={"state": state, "per_page": _MAX_PER_PAGE})
        err = _raise_for_status(resp, provider, "list_prs")
        if err:
            return err
        items = [
            {"number": p.get("number"), "title": p.get("title"),
             "state": p.get("state"), "author": p.get("user", {}).get("login"),
             "branch": p.get("head", {}).get("ref"), "url": p.get("html_url")}
            for p in resp.json()
        ]
    else:
        pid = _gl_project_id(owner, repo)
        url = f"{base}/projects/{pid}/merge_requests"
        resp = _get(url, headers, params={"state": state, "per_page": _MAX_PER_PAGE})
        err = _raise_for_status(resp, provider, "list_prs")
        if err:
            return err
        items = [
            {"number": p.get("iid"), "title": p.get("title"),
             "state": p.get("state"), "author": p.get("author", {}).get("username"),
             "branch": p.get("source_branch"), "url": p.get("web_url")}
            for p in resp.json()
        ]
    return {"status": "ok", "prs": items}


def get_pr(provider: str, host: str | None, token: str | None,
           owner: str, repo: str, pr_number: int) -> dict:
    """Fetch details for a single pull/merge request."""
    base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    if provider == "github":
        url = f"{base}/repos/{owner}/{repo}/pulls/{pr_number}"
        resp = _get(url, headers)
        err = _raise_for_status(resp, provider, "get_pr")
        if err:
            return err
        data = resp.json()
        raw_body = data.get("body") or ""
        return {
            "status": "ok",
            "number": data.get("number"),
            "title": data.get("title"),
            "body": raw_body[:2000],
            "body_truncated": len(raw_body) > 2000,
            "state": data.get("state"),
            "author": data.get("user", {}).get("login"),
            "branch": data.get("head", {}).get("ref"),
            "base_branch": data.get("base", {}).get("ref"),
            "mergeable": data.get("mergeable"),
            "url": data.get("html_url"),
        }
    pid = _gl_project_id(owner, repo)
    url = f"{base}/projects/{pid}/merge_requests/{pr_number}"
    resp = _get(url, headers)
    err = _raise_for_status(resp, provider, "get_pr")
    if err:
        return err
    data = resp.json()
    raw_body = data.get("description") or ""
    return {
        "status": "ok",
        "number": data.get("iid"),
        "title": data.get("title"),
        "body": raw_body[:2000],
        "body_truncated": len(raw_body) > 2000,
        "state": data.get("state"),
        "author": data.get("author", {}).get("username"),
        "branch": data.get("source_branch"),
        "base_branch": data.get("target_branch"),
        "mergeable": data.get("merge_status") == "can_be_merged",
        "url": data.get("web_url"),
    }


def list_issues(provider: str, host: str | None, token: str | None,
                owner: str, repo: str, state: str = "open") -> dict:
    """List issues for a repository."""
    base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    if provider == "github":
        url = f"{base}/repos/{owner}/{repo}/issues"
        resp = _get(url, headers, params={
            "state": state, "per_page": _MAX_PER_PAGE, "pulls": "false"
        })
        err = _raise_for_status(resp, provider, "list_issues")
        if err:
            return err
        items = [
            {"number": i.get("number"), "title": i.get("title"),
             "state": i.get("state"), "author": i.get("user", {}).get("login"),
             "url": i.get("html_url")}
            for i in resp.json() if "pull_request" not in i
        ]
    else:
        pid = _gl_project_id(owner, repo)
        url = f"{base}/projects/{pid}/issues"
        resp = _get(url, headers, params={"state": state, "per_page": _MAX_PER_PAGE})
        err = _raise_for_status(resp, provider, "list_issues")
        if err:
            return err
        items = [
            {"number": i.get("iid"), "title": i.get("title"),
             "state": i.get("state"), "author": i.get("author", {}).get("username"),
             "url": i.get("web_url")}
            for i in resp.json()
        ]
    return {"status": "ok", "issues": items}


def get_issue(provider: str, host: str | None, token: str | None,
              owner: str, repo: str, issue_number: int) -> dict:
    """Fetch details for a single issue."""
    base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    if provider == "github":
        url = f"{base}/repos/{owner}/{repo}/issues/{issue_number}"
        resp = _get(url, headers)
        err = _raise_for_status(resp, provider, "get_issue")
        if err:
            return err
        data = resp.json()
        raw_body = data.get("body") or ""
        return {
            "status": "ok",
            "number": data.get("number"),
            "title": data.get("title"),
            "body": raw_body[:2000],
            "body_truncated": len(raw_body) > 2000,
            "state": data.get("state"),
            "author": data.get("user", {}).get("login"),
            "labels": [lb.get("name") for lb in data.get("labels", [])],
            "url": data.get("html_url"),
        }
    pid = _gl_project_id(owner, repo)
    url = f"{base}/projects/{pid}/issues/{issue_number}"
    resp = _get(url, headers)
    err = _raise_for_status(resp, provider, "get_issue")
    if err:
        return err
    data = resp.json()
    raw_body = data.get("description") or ""
    return {
        "status": "ok",
        "number": data.get("iid"),
        "title": data.get("title"),
        "body": raw_body[:2000],
        "body_truncated": len(raw_body) > 2000,
        "state": data.get("state"),
        "author": data.get("author", {}).get("username"),
        "labels": data.get("labels", []),
        "url": data.get("web_url"),
    }


def list_releases(provider: str, host: str | None, token: str | None,
                  owner: str, repo: str) -> dict:
    """List releases for a repository."""
    base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    if provider == "github":
        url = f"{base}/repos/{owner}/{repo}/releases"
        resp = _get(url, headers, params={"per_page": _MAX_PER_PAGE})
        err = _raise_for_status(resp, provider, "list_releases")
        if err:
            return err
        items = [
            {"tag": r.get("tag_name"), "name": r.get("name"),
             "draft": r.get("draft"), "prerelease": r.get("prerelease"),
             "published_at": r.get("published_at"), "url": r.get("html_url")}
            for r in resp.json()
        ]
    else:
        pid = _gl_project_id(owner, repo)
        url = f"{base}/projects/{pid}/releases"
        resp = _get(url, headers, params={"per_page": _MAX_PER_PAGE})
        err = _raise_for_status(resp, provider, "list_releases")
        if err:
            return err
        items = [
            {"tag": r.get("tag_name"), "name": r.get("name"),
             "draft": False, "prerelease": r.get("upcoming_release", False),
             "published_at": r.get("released_at"), "url": r.get("_links", {}).get("self")}
            for r in resp.json()
        ]
    return {"status": "ok", "releases": items}


def search_repos(provider: str, host: str | None, token: str | None,
                 query: str) -> dict:
    """Search public repositories by keyword."""
    base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    if provider == "github":
        url = f"{base}/search/repositories"
        resp = _get(url, headers, params={"q": query, "per_page": _MAX_PER_PAGE})
        err = _raise_for_status(resp, provider, "search_repos")
        if err:
            return err
        data = resp.json()
        items = [
            {"full_name": r.get("full_name"), "description": r.get("description"),
             "stars": r.get("stargazers_count"), "language": r.get("language"),
             "url": r.get("html_url")}
            for r in data.get("items", [])
        ]
        return {"status": "ok", "total": data.get("total_count"), "repos": items}
    url = f"{base}/projects"
    resp = _get(url, headers, params={
        "search": query, "per_page": _MAX_PER_PAGE, "order_by": "last_activity_at"
    })
    err = _raise_for_status(resp, provider, "search_repos")
    if err:
        return err
    items = [
        {"full_name": r.get("path_with_namespace"), "description": r.get("description"),
         "stars": r.get("star_count"), "language": None, "url": r.get("web_url")}
        for r in resp.json()
    ]
    return {"status": "ok", "total": len(items), "repos": items}


def read_file(provider: str, host: str | None, token: str | None,
              owner: str, repo: str, path: str, ref: str | None = None) -> dict:
    """Read a single file from a repository without cloning.

    Uses the REST contents/blob API; caps output at _MAX_FILE_BYTES.
    """
    import base64
    base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    if provider == "github":
        url = f"{base}/repos/{owner}/{repo}/contents/{path}"
        params = {}
        if ref:
            params["ref"] = ref
        resp = _get(url, headers, params=params or None)
        err = _raise_for_status(resp, provider, "read_file")
        if err:
            return err
        data = resp.json()
        if data.get("encoding") == "base64":
            raw = base64.b64decode(data["content"].replace("\n", ""))
            content = raw[:_MAX_FILE_BYTES].decode("utf-8", errors="replace")
            truncated = len(raw) > _MAX_FILE_BYTES
        else:
            content = data.get("content", "")[:_MAX_FILE_BYTES]
            truncated = False
        return {
            "status": "ok",
            "path": data.get("path"),
            "size": data.get("size"),
            "content": content,
            "truncated": truncated,
        }
    pid = _gl_project_id(owner, repo)
    encoded_path = quote(path, safe="")
    url = f"{base}/projects/{pid}/repository/files/{encoded_path}"
    params = {"ref": ref or "HEAD"}
    resp = _get(url, headers, params=params)
    err = _raise_for_status(resp, provider, "read_file")
    if err:
        return err
    data = resp.json()
    if data.get("encoding") == "base64":
        raw = base64.b64decode(data.get("content", ""))
        content = raw[:_MAX_FILE_BYTES].decode("utf-8", errors="replace")
        truncated = len(raw) > _MAX_FILE_BYTES
    else:
        content = data.get("content", "")[:_MAX_FILE_BYTES]
        truncated = False
    return {
        "status": "ok",
        "path": data.get("file_path"),
        "size": data.get("size"),
        "content": content,
        "truncated": truncated,
    }


def create_pr(provider: str, host: str | None, token: str | None,
              owner: str, repo: str, title: str, head: str,
              base_branch: str, body: str = "") -> dict:
    """Open a new pull/merge request.

    Raises ValueError when called without a token (authentication required).
    """
    if not token:
        return {"status": "error", "error": "A Personal Access Token is required to create PRs/MRs."}
    api_base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    if provider == "github":
        url = f"{api_base}/repos/{owner}/{repo}/pulls"
        resp = _post(url, headers, {
            "title": title, "head": head, "base": base_branch, "body": body
        })
        err = _raise_for_status(resp, provider, "create_pr")
        if err:
            return err
        data = resp.json()
        return {"status": "ok", "number": data.get("number"), "url": data.get("html_url")}
    pid = _gl_project_id(owner, repo)
    url = f"{api_base}/projects/{pid}/merge_requests"
    resp = _post(url, headers, {
        "title": title, "source_branch": head,
        "target_branch": base_branch, "description": body
    })
    err = _raise_for_status(resp, provider, "create_pr")
    if err:
        return err
    data = resp.json()
    return {"status": "ok", "number": data.get("iid"), "url": data.get("web_url")}


def merge_pr(provider: str, host: str | None, token: str | None,
             owner: str, repo: str, pr_number: int,
             merge_message: str = "") -> dict:
    """Merge an open pull/merge request.

    Raises ValueError when called without a token (authentication required).
    """
    if not token:
        return {"status": "error", "error": "A Personal Access Token is required to merge PRs/MRs."}
    api_base = _base_url(provider, host)
    headers = _auth_headers(provider, token)
    if provider == "github":
        url = f"{api_base}/repos/{owner}/{repo}/pulls/{pr_number}/merge"
        body: dict = {"merge_method": "merge"}
        if merge_message:
            body["commit_message"] = merge_message
        resp = _put(url, headers, body)
        err = _raise_for_status(resp, provider, "merge_pr")
        if err:
            return err
        data = resp.json()
        return {"status": "ok", "merged": data.get("merged"), "sha": (data.get("sha") or "")[:10]}
    pid = _gl_project_id(owner, repo)
    url = f"{api_base}/projects/{pid}/merge_requests/{pr_number}/merge"
    body = {}
    if merge_message:
        body["merge_commit_message"] = merge_message
    resp = _put(url, headers, body)
    err = _raise_for_status(resp, provider, "merge_pr")
    if err:
        return err
    data = resp.json()
    return {"status": "ok", "merged": data.get("state") == "merged",
            "sha": (data.get("merge_commit_sha") or "")[:10]}
