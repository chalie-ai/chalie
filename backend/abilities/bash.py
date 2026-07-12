"""BashAbility — safe shell command execution with command-derived risk classification."""

import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from typing import TypedDict

    class _TruncMeta(TypedDict, total=False):
        truncated: bool

from abilities._ability import Ability
from configs.enums.param_key import Keys
from abilities._result import ToolResult, truncate

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 100 * 1024
_TRUNCATION_NOTICE = "... (output truncated at 100KB)"
_DEFAULT_TIMEOUT_S = 30
_MAX_TIMEOUT_S = 300
_NONPRINTABLE_THRESHOLD = 0.10
_EMPTY_SUCCESS_MSG = "(no output — command succeeded silently)"
# Single source of truth for where bash runs — the same value the model is told
# in get_summary() and the cwd _run_command actually uses, so the two can never
# drift.
_WORKING_DIR = Path.home()

# The risk class the policy gate keys on. Derived from the COMMAND (classify_action)
# — never self-declared by the model. ``read`` is the benign inspection floor a
# command falls back to when no heavier pattern matches.
_DEFAULT_ACTION = "read"

_REMOTE_WORDS: frozenset[str] = frozenset({
    "ssh", "scp", "rsync", "kubectl", "docker",
})

_WEB_FETCH_WORDS: frozenset[str] = frozenset({
    "curl", "wget", "nc", "ncat",
})

_INSTALL_WORDS: frozenset[str] = frozenset({
    "apt", "apt-get", "pip", "pip3", "npm", "yarn", "brew", "cargo", "gem",
})

_FILE_MOD_WORDS: frozenset[str] = frozenset({
    "rm", "mv", "cp", "mkdir", "rmdir", "touch", "chmod", "chown", "tee",
})

_COMPOUND_OPERATORS: tuple[str, ...] = ("&&", "||", ";", "|", "$(", "`")
_COMPOUND_KEYWORDS: frozenset[str] = frozenset({"eval", "exec", "source", "."})

_DESTRUCTIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"rm\s+(-\w*r\w*\s+.*)?(-\w*f\w*\s+.*)?/(\s|$)"),
    re.compile(r"rm\s+-\w*rf\w*\s+/(\s|$)"),
    re.compile(r":\(\)\s*\{.*\|.*&\s*\}\s*;?\s*:"),
    re.compile(r"mkfs\b"),
    re.compile(r"dd\s+if=/dev/zero\s+of=/dev/"),
    re.compile(r"dd\s+if=/dev/random\s+of=/dev/"),
    re.compile(r">\s*/dev/sd[a-z]"),
    re.compile(r"rm\s+-\w*rf\w*\s+/\*"),
)


def _has_recursive_flag(tokens: list[str]) -> bool:
    for t in tokens[1:]:
        if t == "--recursive":
            return True
        if t.startswith("-") and not t.startswith("--") and ("r" in t or "R" in t):
            return True
        if not t.startswith("-"):
            break
    return False


def _is_rm_rf(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens or tokens[0] != "rm":
        return False
    has_r = False
    has_f = False
    for t in tokens[1:]:
        if t == "--recursive":
            has_r = True
        elif t == "--force":
            has_f = True
        elif t.startswith("-") and not t.startswith("--"):
            if "r" in t or "R" in t:
                has_r = True
            if "f" in t:
                has_f = True
        elif not t.startswith("-"):
            break
    return has_r and has_f


_SECRET_SUFFIXES: tuple[str, ...] = (
    "KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "PRIVATE",
)
_SECRET_PREFIXES: tuple[str, ...] = ("ANTHROPIC_", "OPENAI_")


class BashAbility(Ability):
    """Execute shell commands via ``bash -c`` with policy-gated classification."""

    _SUMMARY: ClassVar[str] = (
        "Run a shell command via bash. "
        "DO USE bash when you need to perform CLI actions which are NOT "
        "possible via any other available tool. "
        "DO NOT USE bash for: downloading files (use web_download), "
        "file operations within your workspace (use document), "
        "file operations outside your workspace (use read, file_write, "
        "file_permissions, search_files). "
        "Prefer document over direct file tools when the file is within your workspace."
    )

    def get_name(self) -> str:
        return "bash"

    def get_search_tooltip(self) -> str:
        return "Shell command execution"

    def get_summary(self) -> str:
        # At build/introspection time (mp is None) return the bare base text so
        # the search index + SHA map stay machine-independent. On a live request
        # append the working directory so the model knows where commands run.
        if self.mp is None:
            return self._SUMMARY
        return (
            self._SUMMARY
            + f" Working directory: {_WORKING_DIR}. Use absolute paths or cd to operate elsewhere."
        )

    def get_examples(self) -> list[str]:
        return [
            "run this bash script for me",
            "execute ffmpeg -i input.mp4 output.gif to convert this video",
            "check what process is listening on port 8080",
            "run git status in my project directory",
            "what does uname -a return on this machine",
            "compress the /tmp/logs folder into a tarball",
            "count the total lines of Python code under /app",
            "ssh into server/machine",
        ]

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.command: {
                "type": "string",
                "description": (
                    "Shell command to run via bash -c. "
                    "MUST NOT be used when other tools apply: "
                    "use read for file/URL reading, search_files for finding files, "
                    "file_write for writing files, file_permissions for chmod, "
                    "web_download for downloads, document for saving notes."
                ),
            },
            Keys.timeout_s: {
                "type": "integer",
                "description": (
                    "Timeout in seconds (default 30, max 300). "
                    "The command is killed if it exceeds this limit."
                ),
            },
        },
        "required": [Keys.command],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def classify_action(self, params: dict[str, object]) -> str:
        """Derive the policy risk class from the COMMAND itself.

        The dispatcher calls this before the policy gate and keys the
        ``bash.<action>`` permission on the result, so the model can NEVER demote
        a destructive command to a benign class by self-declaring one — there is
        no model-supplied action to trust. Returns ``read`` (the benign
        inspection floor) when no heavier pattern matches.
        """
        command = (cast(str, params.get(Keys.command)) or "").strip()
        if not command:
            return _DEFAULT_ACTION
        return _classify_heuristic(command) or _DEFAULT_ACTION

    def run(self, params: dict[str, object]) -> ToolResult:
        command = (cast(str, params.get(Keys.command)) or "").strip()
        if not command:
            return ToolResult.err(
                "command is required.",
                code="missing-command",
                hint="pass the shell command to run in 'command'.",
            )

        destructive_error = _check_destructive(command)
        if destructive_error:
            return ToolResult.err(
                destructive_error,
                code="destructive-blocked",
                hint="this command is blocked outright; it cannot be run.",
            )

        timeout_s = _resolve_timeout(cast("int | None", params.get(Keys.timeout_s)))
        safe_env = _build_safe_env()

        return _run_command(command, timeout_s, safe_env)


# ── Classification ─────────────────────────────────────────────────────────


def _classify_heuristic(command: str) -> str | None:
    for op in _COMPOUND_OPERATORS:
        if _has_unquoted(command, op):
            return "compound"

    try:
        tokens = shlex.split(command)
    except ValueError:
        return "compound"

    if not tokens:
        return None

    if tokens[0] in _COMPOUND_KEYWORDS:
        return "compound"

    first_word = tokens[0]
    if first_word in _REMOTE_WORDS:
        return "remote_execution"
    if first_word in _WEB_FETCH_WORDS:
        return "web_fetch"
    if first_word in _INSTALL_WORDS:
        return "installation"
    if first_word == "rm" and _has_recursive_flag(tokens):
        return "compound"
    if first_word in _FILE_MOD_WORDS:
        return "modify_file"

    if _has_redirect(command):
        return "modify_file"

    return None


def _has_unquoted(command: str, needle: str) -> bool:
    in_single = False
    in_double = False
    i = 0
    while i < len(command) - len(needle) + 1:
        ch = command[i]
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "\\" and in_double and i + 1 < len(command):
            i += 1
        elif not in_single and not in_double:
            if command[i:i + len(needle)] == needle:
                return True
        i += 1
    return False


def _has_redirect(command: str) -> bool:
    return _has_unquoted(command, ">")


# ── Destructive detection ──────────────────────────────────────────────────


def _check_destructive(command: str) -> str | None:
    if _is_rm_rf(command):
        return (
            "BLOCKED: Destructive command pattern detected. "
            "This command cannot be executed."
        )
    for pattern in _DESTRUCTIVE_PATTERNS:
        if pattern.search(command):
            return (
                "BLOCKED: Destructive command pattern detected. "
                "This command cannot be executed."
            )
    return None


# ── Parameter resolution ──────────────────────────────────────────────────


def _resolve_timeout(timeout_param: int | None) -> int:
    if timeout_param is None:
        return _DEFAULT_TIMEOUT_S
    try:
        value = int(timeout_param)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S
    return max(1, min(value, _MAX_TIMEOUT_S))


# ── Subprocess execution ──────────────────────────────────────────────────


def _run_command(command: str, timeout_s: int, env: dict[str, str]) -> ToolResult:
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            timeout=timeout_s,
            cwd=str(_WORKING_DIR),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.err(
            f"Command timed out after {timeout_s}s and was killed.",
            code="command-timeout",
            hint="raise timeout_s (max 300) or run a shorter command.",
        )
    except OSError as exc:
        return ToolResult.err(
            f"Failed to execute command: {exc}",
            code="exec-failed",
        )

    stdout_raw = proc.stdout or b""
    stderr_raw = proc.stderr or b""

    if _is_binary(stdout_raw) or _is_binary(stderr_raw):
        return ToolResult.ok(
            {
                "exit_code": proc.returncode,
                "stdout": f"(binary output, {len(stdout_raw)} bytes)",
                "stderr": f"(binary output, {len(stderr_raw)} bytes)" if stderr_raw else "",
                "binary": True,
            },
            total_bytes=len(stdout_raw) + len(stderr_raw),
        )

    stdout_str = stdout_raw.decode("utf-8", errors="replace")
    stderr_str = stderr_raw.decode("utf-8", errors="replace")

    # Clip combined output to the budget via the shared truncate primitive, giving
    # stdout priority and reporting clipping uniformly through meta truncated=true.
    truncated = False
    stdout_str, clipped_out = truncate(stdout_str, _MAX_OUTPUT_CHARS)
    if clipped_out:
        truncated = True
        stdout_str += _TRUNCATION_NOTICE
        stderr_str = ""
    else:
        stderr_str, clipped_err = truncate(stderr_str, _MAX_OUTPUT_CHARS - len(stdout_str))
        if clipped_err:
            truncated = True
            stderr_str += _TRUNCATION_NOTICE

    if not stdout_str and not stderr_str and proc.returncode == 0:
        stdout_str = _EMPTY_SUCCESS_MSG

    meta: "_TruncMeta" = {"truncated": True} if truncated else {}
    return ToolResult.ok(
        {
            "exit_code": proc.returncode,
            "stdout": stdout_str,
            "stderr": stderr_str,
        },
        **meta,
    )


# ── Helpers ────────────────────────────────────────────────────────────────


def _is_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return True
    sample = data[:8192]
    non_printable = sum(
        1 for b in sample
        if b < 0x20 and b not in (0x09, 0x0A, 0x0D)
    )
    return (non_printable / len(sample)) > _NONPRINTABLE_THRESHOLD


def _build_safe_env() -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(upper.startswith(p) for p in _SECRET_PREFIXES):
            continue
        if any(upper.endswith(s) for s in _SECRET_SUFFIXES):
            continue
        safe[key] = value
    return safe
