"""GitCapability — GitHub and GitLab integration via hybrid git CLI + REST.

Uses the git CLI (subprocess arg-lists, no shell) for clone/diff/commit/push;
raw requests for PR/MR/issue/release/search metadata (git_rest_handler).

Connect modes:
  - No token → connected=True, authenticated=False (public read-only via REST
    + anonymous clone for public repos).
  - Valid token → connected=True, authenticated=True (full read/write).

Credential keys stored via store_credential:
  git:provider, git:host, git:token.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time

import yaml

from capabilities.base import AbstractCapability
from capabilities.git_capability import git_rest_handler as rest
from services.file_mapper_service import FileMapperService

logger = logging.getLogger(__name__)

_MANIFEST_PATH = FileMapperService.get_capabilities_path("git_capability", "manifest.yaml")

# Credential keys
_K_PROVIDER = "git:provider"
_K_HOST = "git:host"
_K_TOKEN = "git:token"

# Workspace TTL — temp dirs older than this are removed at the start of each clone.
_WORKSPACE_TTL_SECS = 6 * 3600

# Protected branches that push is always refused on.
_PROTECTED_BRANCHES = frozenset({"main", "master", "develop", "trunk"})

# Characters rejected in branch names, remote_branch, or paths to prevent
# option injection even when using subprocess arg-lists.
_UNSAFE_ARG_RE = re.compile(r"[;&|`$<>!\\\"'{}()\[\]*?\s]")

# Strict validator for owner/repo names: GitHub/GitLab do not allow slashes.
# Rejects path traversal (e.g. "org/../../admin") and double-dot segments.
_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Safety-net timeout for every git subprocess and REST call (30 minutes).
# Git clone/push on large repos legitimately runs for several minutes; this only
# guards against a genuinely hung process, never against slow-but-valid work.
_TIMEOUT_SECONDS = 1800


class GitCapability(AbstractCapability):
    """Hybrid git CLI + REST capability for GitHub and GitLab.

    Architecture: git CLI (subprocess) handles clone/diff/commit/push for
    workspace operations; raw requests handles metadata (PRs, issues, releases,
    search, read_file).
    """

    def __init__(self) -> None:
        super().__init__()
        self._manifest_cache: dict | None = None
        self._provider: str = "github"
        self._host: str | None = None
        self._token: str | None = None
        self._authenticated: bool = False

    # ── Identity ──────────────────────────────────────────────────────────────

    def get_id(self) -> str:
        return "git"

    def get_manifest(self) -> dict:
        if self._manifest_cache is None:
            with open(_MANIFEST_PATH, encoding="utf-8") as fh:
                self._manifest_cache = yaml.safe_load(fh)
        return self._manifest_cache

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def configure(self, credentials: dict) -> None:
        """Validate and persist provider/host/token.

        Token is optional — omitting it enables public read-only mode.
        When a token is supplied it is validated via a cheap GET /user call.

        Raises:
            ValueError: When the token is rejected (HTTP 401) or the custom
                host resolves to a private network address.
        """
        provider = (credentials.get("provider") or "github").strip().lower()
        if provider not in ("github", "gitlab"):
            raise ValueError(f"Unknown provider: {provider!r}. Choose 'github' or 'gitlab'.")
        host = (credentials.get("host") or "").strip().rstrip("/") or None
        token = (credentials.get("token") or "").strip() or None

        if host:
            rest._validate_host(host)
        if token:
            rest.validate_token(provider, host, token)

        self.store_credential(_K_PROVIDER, provider)
        self.store_credential(_K_HOST, host or "")
        self.store_credential(_K_TOKEN, token or "")

        self._provider = provider
        self._host = host
        self._token = token
        self._authenticated = bool(token)
        self._connected = True

    def connect(self) -> bool:
        """Load stored credentials; always marks connected (public mode is valid)."""
        self._provider = self.load_credential(_K_PROVIDER) or "github"
        self._host = self.load_credential(_K_HOST) or None
        self._token = self.load_credential(_K_TOKEN) or None
        self._authenticated = bool(self._token)
        # Git is on-demand — always connected (public reads work without a token).
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False
        self._authenticated = False
        self._token = None
        self.delete_credentials()
        logger.info("[git] Disconnected and credentials removed.")

    # ── Cognitive pipeline ────────────────────────────────────────────────────
    # Git is on-demand — no passive ingestion or monitoring.

    def ingest(self) -> list:
        return []

    def understand(self, items: list) -> list:
        return items

    def _do_monitor(self) -> None:
        pass

    def act(self, action: str, params: dict) -> dict:
        """Route to the handler map shared with get_tools() — single dispatch source."""
        tool_map = {t["name"]: t["handler"] for t in self.get_tools()}
        handler = tool_map.get(action)
        if handler is None:
            return {"status": "error", "error": f"Unknown git action: {action}"}
        return handler(topic="", params=params)

    # ── Safety helpers ────────────────────────────────────────────────────────

    def _require_auth(self) -> dict | None:
        """Return an error dict when no token is stored (write actions require auth)."""
        if not self._authenticated:
            return {
                "status": "error",
                "error": (
                    "This action requires a Personal Access Token. "
                    "Configure one in Brain → Capabilities → Git."
                ),
            }
        return None

    @staticmethod
    def _validate_git_arg(value: str, label: str) -> None:
        """Reject values containing shell metacharacters or a leading dash.

        Even with subprocess arg-lists these characters hint at option injection
        or accidental flag interpretation.

        Raises:
            ValueError: On unsafe input.
        """
        if value.startswith("-"):
            raise ValueError(f"{label} must not start with '-': {value!r}")
        if _UNSAFE_ARG_RE.search(value):
            raise ValueError(
                f"{label} contains unsafe characters: {value!r}. "
                "Only alphanumeric characters, dots, slashes, underscores, and hyphens are allowed."
            )

    @staticmethod
    def _validate_owner_repo(value: str, label: str) -> None:
        """Reject owner/repo values that contain slashes or path-traversal segments.

        GitHub and GitLab owner and repo names cannot contain slashes; permitting
        them would allow values like 'org/../../api/admin' to be interpolated into
        REST URLs and clone URLs, bypassing SSRF and path-traversal guards.

        Raises:
            ValueError: When the value contains characters outside [A-Za-z0-9._-]
                or is (or contains) the '..' double-dot traversal sequence.
        """
        if ".." in value:
            raise ValueError(
                f"{label} must not contain '..': {value!r}. "
                "Path traversal is not permitted in owner or repo names."
            )
        if not _OWNER_REPO_RE.match(value):
            raise ValueError(
                f"{label} contains invalid characters: {value!r}. "
                "Owner and repo names may only contain alphanumeric characters, "
                "dots, underscores, and hyphens."
            )

    def _redact(self, text: str) -> str:
        """Remove the stored token from text before it reaches the LLM."""
        if self._token:
            return text.replace(self._token, "***")
        return text

    # ── Workspace helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _sweep_old_workspaces() -> None:
        """Remove chalie_git_* temp dirs older than _WORKSPACE_TTL_SECS.

        Called at the start of every clone; bounded-growth heuristic.
        Workspaces are ephemeral scratch — no data loss risk.
        """
        tmp = tempfile.gettempdir()
        now = time.time()
        for name in os.listdir(tmp):
            if not name.startswith("chalie_git_"):
                continue
            full = os.path.join(tmp, name)
            try:
                if os.path.isdir(full) and (now - os.path.getmtime(full)) > _WORKSPACE_TTL_SECS:
                    shutil.rmtree(full, ignore_errors=True)
            except OSError:
                pass

    def _build_clone_env(self, askpass_script: str) -> dict:
        """Build a safe subprocess environment for git operations.

        The token is passed only via GIT_TOKEN in the environment;
        GIT_ASKPASS points to a script that echoes it. The token never
        appears in process argv or .git/config.
        """
        env = {k: v for k, v in os.environ.items()
               if k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TERM")}
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_ASKPASS"] = askpass_script
        env["GIT_TOKEN"] = self._token or ""
        return env

    @staticmethod
    def _write_askpass_script() -> str:
        """Create a GIT_ASKPASS script in its own temp file and return its path.

        The script MUST NOT live inside a workspace: for clone the workspace is
        the destination directory and git refuses to clone into a non-empty dir;
        for commit a stray script in the tree risks being staged. Mode is 0o700.
        The caller deletes it in a finally block.
        """
        fd, path = tempfile.mkstemp(prefix="chalie_git_askpass_", suffix=".sh")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("#!/bin/sh\nprintf '%s' \"$GIT_TOKEN\"\n")
        os.chmod(path, stat.S_IRWXU)
        return path

    def _git(self, args: list[str], cwd: str | None = None,
             env: dict | None = None, timeout: int = _TIMEOUT_SECONDS) -> tuple[int, str, str]:
        """Run a git command and return (returncode, stdout, stderr).

        Always injects -c core.hooksPath=/dev/null and -c credential.helper=
        to prevent hook execution and credential persistence.
        Token is redacted from both stdout and stderr before returning.
        """
        cmd = [
            "git",
            "-c", "core.hooksPath=/dev/null",
            "-c", "credential.helper=",
            *args,
        ]
        result = subprocess.run(
            cmd, shell=False, capture_output=True, text=True,
            timeout=timeout, cwd=cwd, env=env,
        )
        stdout = self._redact(result.stdout)
        stderr = self._redact(result.stderr)
        return result.returncode, stdout, stderr

    def _clone_url(self, owner: str, repo: str) -> str:
        """Build a clone URL that uses username-only (no secret in the URL).

        The actual password is supplied by GIT_ASKPASS at clone time.
        GitHub: https://x-access-token@<host>/<owner>/<repo>.git
        GitLab: https://oauth2@<host>/<owner>/<repo>.git
        No-token: plain HTTPS (public repos only).
        """
        if self._provider == "github":
            host = self._host or "github.com"
            if self._authenticated:
                return f"https://x-access-token@{host}/{owner}/{repo}.git"
            return f"https://github.com/{owner}/{repo}.git"
        # GitLab
        host_part = self._host or "gitlab.com"
        # _host is validated https:// at configure time; strip the scheme for URL assembly.
        host_part = host_part.removeprefix("https://").rstrip("/")
        if self._authenticated:
            return f"https://oauth2@{host_part}/{owner}/{repo}.git"
        return f"https://gitlab.com/{owner}/{repo}.git"

    def _get_default_branch(self, workspace: str, env: dict) -> str | None:
        """Fetch the remote default branch from the cloned workspace."""
        rc, out, _ = self._git(
            ["rev-parse", "--abbrev-ref", "origin/HEAD"], cwd=workspace, env=env
        )
        if rc == 0 and out.strip():
            return out.strip().replace("origin/", "")
        # Fallback: try symbolic-ref
        rc, out, _ = self._git(
            ["symbolic-ref", "refs/remotes/origin/HEAD"], cwd=workspace, env=env
        )
        if rc == 0:
            return out.strip().split("/")[-1]
        return None

    def _is_protected_push_target(self, branch: str, workspace: str, env: dict) -> bool:
        """Return True if branch is in _PROTECTED_BRANCHES or equals the default branch."""
        if branch in _PROTECTED_BRANCHES:
            return True
        default = self._get_default_branch(workspace, env)
        return default is not None and branch == default

    # ── Tool handlers ─────────────────────────────────────────────────────────

    def _require_connection(self) -> dict | None:
        if not self.is_connected():
            return {
                "status": "error",
                "error": "Git capability not connected. Configure it in Brain → Capabilities.",
            }
        return None

    def _th_get_repo(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        owner = params.get("owner", "")
        repo = params.get("repo", "")
        if not owner or not repo:
            return {"status": "error", "error": "'owner' and 'repo' are required for get_repo."}
        try:
            self._validate_owner_repo(owner, "owner")
            self._validate_owner_repo(repo, "repo")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return rest.get_repo(self._provider, self._host, self._token, owner, repo)

    def _th_list_branches(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        owner = params.get("owner", "")
        repo = params.get("repo", "")
        if not owner or not repo:
            return {"status": "error", "error": "'owner' and 'repo' are required for list_branches."}
        try:
            self._validate_owner_repo(owner, "owner")
            self._validate_owner_repo(repo, "repo")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return rest.list_branches(self._provider, self._host, self._token, owner, repo)

    def _th_list_commits(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        owner = params.get("owner", "")
        repo = params.get("repo", "")
        if not owner or not repo:
            return {"status": "error", "error": "'owner' and 'repo' are required for list_commits."}
        try:
            self._validate_owner_repo(owner, "owner")
            self._validate_owner_repo(repo, "repo")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return rest.list_commits(
            self._provider, self._host, self._token,
            owner, repo, params.get("branch"),
        )

    def _th_get_commit(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        owner, repo, sha = params.get("owner", ""), params.get("repo", ""), params.get("sha", "")
        if not owner or not repo or not sha:
            return {"status": "error", "error": "'owner', 'repo', and 'sha' are required for get_commit."}
        try:
            self._validate_owner_repo(owner, "owner")
            self._validate_owner_repo(repo, "repo")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return rest.get_commit(self._provider, self._host, self._token, owner, repo, sha)

    def _th_list_prs(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        owner, repo = params.get("owner", ""), params.get("repo", "")
        if not owner or not repo:
            return {"status": "error", "error": "'owner' and 'repo' are required for list_prs."}
        try:
            self._validate_owner_repo(owner, "owner")
            self._validate_owner_repo(repo, "repo")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return rest.list_prs(
            self._provider, self._host, self._token,
            owner, repo, params.get("state", "open"),
        )

    def _th_get_pr(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        owner, repo = params.get("owner", ""), params.get("repo", "")
        pr_number = params.get("pr_number")
        if not owner or not repo or pr_number is None:
            return {"status": "error", "error": "'owner', 'repo', and 'pr_number' are required for get_pr."}
        try:
            self._validate_owner_repo(owner, "owner")
            self._validate_owner_repo(repo, "repo")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return rest.get_pr(self._provider, self._host, self._token, owner, repo, int(pr_number))

    def _th_list_issues(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        owner, repo = params.get("owner", ""), params.get("repo", "")
        if not owner or not repo:
            return {"status": "error", "error": "'owner' and 'repo' are required for list_issues."}
        try:
            self._validate_owner_repo(owner, "owner")
            self._validate_owner_repo(repo, "repo")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return rest.list_issues(
            self._provider, self._host, self._token,
            owner, repo, params.get("state", "open"),
        )

    def _th_get_issue(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        owner, repo = params.get("owner", ""), params.get("repo", "")
        issue_number = params.get("issue_number")
        if not owner or not repo or issue_number is None:
            return {"status": "error", "error": "'owner', 'repo', and 'issue_number' are required."}
        try:
            self._validate_owner_repo(owner, "owner")
            self._validate_owner_repo(repo, "repo")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return rest.get_issue(
            self._provider, self._host, self._token, owner, repo, int(issue_number)
        )

    def _th_list_releases(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        owner, repo = params.get("owner", ""), params.get("repo", "")
        if not owner or not repo:
            return {"status": "error", "error": "'owner' and 'repo' are required for list_releases."}
        try:
            self._validate_owner_repo(owner, "owner")
            self._validate_owner_repo(repo, "repo")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return rest.list_releases(self._provider, self._host, self._token, owner, repo)

    def _th_search_repos(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        query = params.get("query", "")
        if not query:
            return {"status": "error", "error": "'query' is required for search_repos."}
        return rest.search_repos(self._provider, self._host, self._token, query)

    def _th_read_file(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection()
        if err:
            return err
        owner, repo = params.get("owner", ""), params.get("repo", "")
        path = params.get("path", "")
        if not owner or not repo or not path:
            return {"status": "error", "error": "'owner', 'repo', and 'path' are required for read_file."}
        try:
            self._validate_owner_repo(owner, "owner")
            self._validate_owner_repo(repo, "repo")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return rest.read_file(
            self._provider, self._host, self._token,
            owner, repo, path, params.get("ref"),
        )

    def _th_clone(self, topic, params, config=None, telemetry=None) -> dict:
        """Clone a repository into a temp workspace; sweep old workspaces first."""
        err = self._require_connection()
        if err:
            return err
        owner, repo = params.get("owner", ""), params.get("repo", "")
        if not owner or not repo:
            return {"status": "error", "error": "'owner' and 'repo' are required for clone."}
        branch = params.get("branch")
        full_clone = bool(params.get("full", False))

        try:
            self._validate_owner_repo(owner, "owner")
            self._validate_owner_repo(repo, "repo")
            if branch:
                self._validate_git_arg(branch, "branch")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

        self._sweep_old_workspaces()
        workspace = tempfile.mkdtemp(prefix="chalie_git_")
        askpass = ""

        try:
            askpass = self._write_askpass_script()
            env = self._build_clone_env(askpass)
            clone_url = self._clone_url(owner, repo)

            clone_args = ["clone"]
            if not full_clone:
                clone_args += ["--depth", "1"]
            if branch:
                clone_args += ["--branch", branch, "--"]
            else:
                clone_args.append("--")
            clone_args += [clone_url, workspace]

            rc, stdout, stderr = self._git(clone_args, env=env)
            if rc != 0:
                shutil.rmtree(workspace, ignore_errors=True)
                return {"status": "error", "error": f"git clone failed: {stderr.strip()}"}

            # Collect top-level file listing.
            try:
                entries = sorted(os.listdir(workspace))
                files = [e for e in entries if e != ".git"]
            except OSError:
                files = []

            # Read HEAD sha.
            rc2, head_sha, _ = self._git(["rev-parse", "HEAD"], cwd=workspace, env=env)
            head = head_sha.strip() if rc2 == 0 else ""

            # Detect checked-out branch.
            rc3, branch_out, _ = self._git(
                ["rev-parse", "--abbrev-ref", "HEAD"], cwd=workspace, env=env
            )
            cloned_branch = branch_out.strip() if rc3 == 0 else (branch or "")

            return {
                "status": "ok",
                "workspace": workspace,
                "branch": cloned_branch,
                "head_sha": head,
                "files": files,
            }
        except Exception as exc:
            shutil.rmtree(workspace, ignore_errors=True)
            logger.warning("[git] clone error: %s", exc)
            return {"status": "error", "error": self._redact(str(exc))}
        finally:
            try:
                os.remove(askpass)
            except OSError:
                pass

    def _th_diff(self, topic, params, config=None, telemetry=None) -> dict:
        """Run git diff in an existing workspace."""
        err = self._require_connection()
        if err:
            return err
        workspace = params.get("workspace", "")
        if not workspace or not os.path.isdir(workspace):
            return {"status": "error", "error": "'workspace' must be a valid directory path."}

        staged = bool(params.get("staged", False))
        ref = params.get("ref")

        diff_args = ["diff"]
        if staged:
            diff_args.append("--cached")
        if ref:
            try:
                self._validate_git_arg(ref, "ref")
            except ValueError as exc:
                return {"status": "error", "error": str(exc)}
            diff_args += [ref]

        askpass = ""
        try:
            askpass = self._write_askpass_script()
            env = self._build_clone_env(askpass)
            rc, stdout, stderr = self._git(diff_args, cwd=workspace, env=env)
            if rc != 0:
                return {"status": "error", "error": f"git diff failed: {stderr.strip()}"}
            return {"status": "ok", "diff": stdout}
        finally:
            try:
                os.remove(askpass)
            except OSError:
                pass

    def _th_commit(self, topic, params, config=None, telemetry=None) -> dict:
        """Stage explicit paths and create a commit in an existing workspace.

        Never uses 'git add -A'; always stages the paths listed in 'paths'.
        If 'new_branch' is supplied, creates and switches to it first.
        """
        err = self._require_connection()
        if err:
            return err
        workspace = params.get("workspace", "")
        message = params.get("message", "")
        paths: list[str] = params.get("paths", [])

        if not workspace or not os.path.isdir(workspace):
            return {"status": "error", "error": "'workspace' must be a valid directory path."}
        if not message:
            return {"status": "error", "error": "'message' is required for commit."}
        if not paths:
            return {"status": "error", "error": "'paths' is required — provide explicit file paths to stage."}

        new_branch = params.get("new_branch")
        if new_branch:
            try:
                self._validate_git_arg(new_branch, "new_branch")
            except ValueError as exc:
                return {"status": "error", "error": str(exc)}

        try:
            for p in paths:
                self._validate_git_arg(p, "path")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

        askpass = ""
        try:
            askpass = self._write_askpass_script()
            env = self._build_clone_env(askpass)

            if new_branch:
                rc, _, stderr = self._git(
                    ["checkout", "-b", new_branch], cwd=workspace, env=env
                )
                if rc != 0:
                    return {"status": "error", "error": f"git checkout -b failed: {stderr.strip()}"}

            # Stage explicit paths only — never git add -A.
            add_cmd = ["add", "--"] + paths
            rc, _, stderr = self._git(add_cmd, cwd=workspace, env=env)
            if rc != 0:
                return {"status": "error", "error": f"git add failed: {stderr.strip()}"}

            rc, _, stderr = self._git(
                ["commit", "-m", message], cwd=workspace, env=env
            )
            if rc != 0:
                return {"status": "error", "error": f"git commit failed: {stderr.strip()}"}

            rc2, sha_out, _ = self._git(["rev-parse", "HEAD"], cwd=workspace, env=env)
            sha = sha_out.strip() if rc2 == 0 else ""
            return {"status": "ok", "sha": sha, "paths_committed": paths}
        finally:
            try:
                os.remove(askpass)
            except OSError:
                pass

    def _th_push(self, topic, params, config=None, telemetry=None) -> dict:
        """Push the current workspace branch to the remote.

        Refuses pushes to protected/default branches and never uses --force.
        Token is required; returns actionable error when unauthenticated.
        """
        err = self._require_connection() or self._require_auth()
        if err:
            return err

        workspace = params.get("workspace", "")
        if not workspace or not os.path.isdir(workspace):
            return {"status": "error", "error": "'workspace' must be a valid directory path."}

        remote_branch = params.get("remote_branch", "")
        if remote_branch:
            try:
                self._validate_git_arg(remote_branch, "remote_branch")
            except ValueError as exc:
                return {"status": "error", "error": str(exc)}

        askpass = ""
        try:
            askpass = self._write_askpass_script()
            env = self._build_clone_env(askpass)

            # Determine local branch.
            rc, branch_out, _ = self._git(
                ["rev-parse", "--abbrev-ref", "HEAD"], cwd=workspace, env=env
            )
            if rc != 0:
                return {"status": "error", "error": "Could not determine current branch."}
            local_branch = branch_out.strip()

            target = remote_branch or local_branch

            if self._is_protected_push_target(target, workspace, env):
                return {
                    "status": "error",
                    "error": (
                        f"Push to '{target}' is blocked. "
                        "Pushes to protected/default branches (main, master, develop, trunk) "
                        "are not permitted. Create a feature branch first."
                    ),
                }

            push_args = ["push", "origin"]
            if remote_branch:
                push_args += [f"HEAD:{remote_branch}"]
            else:
                push_args += [local_branch]

            rc, stdout, stderr = self._git(push_args, cwd=workspace, env=env)
            if rc != 0:
                return {"status": "error", "error": f"git push failed: {stderr.strip()}"}
            full = (stdout + stderr).strip()
            return {"status": "ok", "branch": target, "output": full}
        finally:
            try:
                os.remove(askpass)
            except OSError:
                pass

    def _th_create_pr(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection() or self._require_auth()
        if err:
            return err
        owner, repo = params.get("owner", ""), params.get("repo", "")
        title, head = params.get("title", ""), params.get("head", "")
        base_branch = params.get("base_branch", "")
        if not owner or not repo or not title or not head or not base_branch:
            return {
                "status": "error",
                "error": "'owner', 'repo', 'title', 'head', and 'base_branch' are required for create_pr.",
            }
        try:
            self._validate_owner_repo(owner, "owner")
            self._validate_owner_repo(repo, "repo")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return rest.create_pr(
            self._provider, self._host, self._token,
            owner, repo, title, head, base_branch,
            params.get("body", ""),
        )

    def _th_merge_pr(self, topic, params, config=None, telemetry=None) -> dict:
        err = self._require_connection() or self._require_auth()
        if err:
            return err
        owner, repo = params.get("owner", ""), params.get("repo", "")
        pr_number = params.get("pr_number")
        if not owner or not repo or pr_number is None:
            return {"status": "error", "error": "'owner', 'repo', and 'pr_number' are required for merge_pr."}
        try:
            self._validate_owner_repo(owner, "owner")
            self._validate_owner_repo(repo, "repo")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        return rest.merge_pr(
            self._provider, self._host, self._token,
            owner, repo, int(pr_number), params.get("merge_message", ""),
        )

    # ── Tool definitions ──────────────────────────────────────────────────────

    def get_tools(self) -> list:
        """Return all 17 tool definitions; handlers shared with act()."""
        _repo_params = {
            "owner": {"type": "string", "description": "Repository owner (user or org name)."},
            "repo": {"type": "string", "description": "Repository name."},
        }
        return [
            # READ — REST-backed
            {
                "name": "get_repo",
                "handler": self._th_get_repo,
                "timeout": _TIMEOUT_SECONDS,
                "description": "Fetch metadata for a GitHub or GitLab repository.",
                "parameters": _repo_params,
            },
            {
                "name": "list_branches",
                "handler": self._th_list_branches,
                "timeout": _TIMEOUT_SECONDS,
                "description": "List branches for a repository.",
                "parameters": _repo_params,
            },
            {
                "name": "list_commits",
                "handler": self._th_list_commits,
                "timeout": _TIMEOUT_SECONDS,
                "description": "List recent commits for a repository.",
                "parameters": {
                    **_repo_params,
                    "branch": {"type": "string", "description": "Branch or ref to list commits from."},
                },
            },
            {
                "name": "get_commit",
                "handler": self._th_get_commit,
                "timeout": _TIMEOUT_SECONDS,
                "description": "Fetch details and changed files for a single commit.",
                "parameters": {
                    **_repo_params,
                    "sha": {"type": "string", "description": "Commit SHA (full or short)."},
                },
            },
            {
                "name": "list_prs",
                "handler": self._th_list_prs,
                "timeout": _TIMEOUT_SECONDS,
                "description": "List pull/merge requests for a repository.",
                "parameters": {
                    **_repo_params,
                    "state": {"type": "string", "enum": ["open", "closed", "all"],
                               "description": "PR/MR state filter.", "default": "open"},
                },
            },
            {
                "name": "get_pr",
                "handler": self._th_get_pr,
                "timeout": _TIMEOUT_SECONDS,
                "description": "Fetch details for a single pull/merge request.",
                "parameters": {
                    **_repo_params,
                    "pr_number": {"type": "integer", "description": "PR/MR number."},
                },
            },
            {
                "name": "list_issues",
                "handler": self._th_list_issues,
                "timeout": _TIMEOUT_SECONDS,
                "description": "List issues for a repository.",
                "parameters": {
                    **_repo_params,
                    "state": {"type": "string", "enum": ["open", "closed", "all"],
                               "description": "Issue state filter.", "default": "open"},
                },
            },
            {
                "name": "get_issue",
                "handler": self._th_get_issue,
                "timeout": _TIMEOUT_SECONDS,
                "description": "Fetch details for a single issue.",
                "parameters": {
                    **_repo_params,
                    "issue_number": {"type": "integer", "description": "Issue number."},
                },
            },
            {
                "name": "list_releases",
                "handler": self._th_list_releases,
                "timeout": _TIMEOUT_SECONDS,
                "description": "List releases for a repository.",
                "parameters": _repo_params,
            },
            {
                "name": "search_repos",
                "handler": self._th_search_repos,
                "timeout": _TIMEOUT_SECONDS,
                "description": "Search public repositories by keyword.",
                "parameters": {
                    "query": {"type": "string", "description": "Search keywords."},
                },
            },
            {
                "name": "read_file",
                "handler": self._th_read_file,
                "timeout": _TIMEOUT_SECONDS,
                "description": "Read a single file from a repository without cloning.",
                "parameters": {
                    **_repo_params,
                    "path": {"type": "string", "description": "File path within the repository."},
                    "ref": {"type": "string", "description": "Branch, tag, or commit SHA. Defaults to HEAD."},
                },
            },
            # READ — git CLI-backed
            {
                "name": "clone",
                "handler": self._th_clone,
                "timeout": _TIMEOUT_SECONDS,
                "description": (
                    "Clone a repository to a local workspace for editing. "
                    "Returns workspace path, branch, and top-level file listing. "
                    "Use file_write / search_files / bash to edit files, then call commit and push."
                ),
                "parameters": {
                    **_repo_params,
                    "branch": {"type": "string", "description": "Branch to clone. Defaults to the default branch."},
                    "full": {"type": "boolean", "description": "When true, performs a full clone instead of shallow (--depth 1).", "default": False},
                },
            },
            {
                "name": "diff",
                "handler": self._th_diff,
                "timeout": _TIMEOUT_SECONDS,
                "description": "Show git diff for a workspace (unstaged, staged, or vs a ref).",
                "parameters": {
                    "workspace": {"type": "string", "description": "Path to an existing cloned workspace."},
                    "staged": {"type": "boolean", "description": "When true, show staged changes (--cached).", "default": False},
                    "ref": {"type": "string", "description": "Diff against this ref instead of working tree."},
                },
            },
            # WRITE — git CLI-backed
            {
                "name": "commit",
                "handler": self._th_commit,
                "timeout": _TIMEOUT_SECONDS,
                "description": (
                    "Stage explicit file paths and create a commit in a cloned workspace. "
                    "Never stages all files; 'paths' must be provided explicitly. "
                    "Optionally creates a new branch first via 'new_branch'."
                ),
                "parameters": {
                    "workspace": {"type": "string", "description": "Path to an existing cloned workspace."},
                    "message": {"type": "string", "description": "Commit message."},
                    "paths": {"type": "array", "items": {"type": "string"},
                              "description": "Explicit file paths to stage (relative to workspace root)."},
                    "new_branch": {"type": "string", "description": "Create and switch to this branch before committing."},
                },
            },
            {
                "name": "push",
                "handler": self._th_push,
                "timeout": _TIMEOUT_SECONDS,
                "description": (
                    "Push the current workspace branch to the remote. "
                    "Blocked on protected/default branches (main, master, develop, trunk). "
                    "Requires a Personal Access Token."
                ),
                "parameters": {
                    "workspace": {"type": "string", "description": "Path to an existing cloned workspace."},
                    "remote_branch": {"type": "string", "description": "Optional remote branch name. Defaults to the local branch name."},
                },
            },
            # WRITE — REST-backed
            {
                "name": "create_pr",
                "handler": self._th_create_pr,
                "timeout": _TIMEOUT_SECONDS,
                "description": "Open a new pull request (GitHub) or merge request (GitLab).",
                "parameters": {
                    **_repo_params,
                    "title": {"type": "string", "description": "PR/MR title."},
                    "head": {"type": "string", "description": "Source branch containing the changes."},
                    "base_branch": {"type": "string", "description": "Target branch to merge into."},
                    "body": {"type": "string", "description": "PR/MR description (optional)."},
                },
            },
            {
                "name": "merge_pr",
                "handler": self._th_merge_pr,
                "timeout": _TIMEOUT_SECONDS,
                "description": "Merge an open pull/merge request. Requires a Personal Access Token.",
                "parameters": {
                    **_repo_params,
                    "pr_number": {"type": "integer", "description": "PR/MR number to merge."},
                    "merge_message": {"type": "string", "description": "Optional merge commit message."},
                },
            },
        ]
