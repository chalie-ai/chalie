"""BashAbility — safe shell command execution with LLM classification and heuristic overrides."""

import json
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import ClassVar

from abilities._base import Ability

logger = logging.getLogger(__name__)

_MAX_OUTPUT_BYTES = 100 * 1024
_TRUNCATION_NOTICE = "... (output truncated at 100KB)"
_DEFAULT_TIMEOUT_S = 30
_MAX_TIMEOUT_S = 300
_NONPRINTABLE_THRESHOLD = 0.10
_EMPTY_SUCCESS_MSG = "(no output — command succeeded silently)"

_ACTION_SEVERITY: dict[str, int] = {
    "read": 0,
    "execute": 1,
    "modify_file": 2,
    "web_fetch": 3,
    "installation": 4,
    "remote_execution": 5,
    "compound": 6,
}

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

    NAME = "bash"
    POLICY_CATEGORY = "System"
    POLICY_LABELS: ClassVar[dict[str, str]] = {
        "read": "Read-only commands",
        "execute": "Run scripts/binaries",
        "modify_file": "File modifications",
        "web_fetch": "Network commands",
        "installation": "Package management",
        "remote_execution": "Remote access",
        "compound": "Multi-command chains",
    }
    SEARCH_TOOLTIP = "Shell command execution"
    SUMMARY = (
        "Run a shell command via bash. "
        "DO USE bash when you need to perform CLI actions which are NOT "
        "possible via any other available tool. "
        "DO NOT USE bash for: downloading files (use web_download), "
        "file operations within your workspace (use document), "
        "file operations outside your workspace (use read, file_write, "
        "file_permissions, search_files). "
        "Prefer document over direct file tools when the file is within your workspace."
    )
    EXAMPLES: ClassVar[list[str]] = [
        "run this bash script for me",
        "execute ffmpeg -i input.mp4 output.gif to convert this video",
        "check what process is listening on port 8080",
        "run git status in my project directory",
        "what does uname -a return on this machine",
        "run the deploy script at /home/user/scripts/deploy.sh",
        "compress the /tmp/logs folder into a tarball",
        "count the total lines of Python code under /app",
    ]
    INPUT_SCHEMA: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Shell command to run via bash -c. "
                    "MUST NOT be used when other tools apply: "
                    "use read for file/URL reading, search_files for finding files, "
                    "file_write for writing files, file_permissions for chmod, "
                    "web_download for downloads, document for saving notes."
                ),
            },
            "action": {
                "type": "string",
                "enum": [
                    "read",
                    "execute",
                    "modify_file",
                    "web_fetch",
                    "installation",
                    "remote_execution",
                    "compound",
                ],
                "description": (
                    "Classify the command: "
                    "read = inspection only (ls, cat, ps, df), "
                    "execute = run a script or binary, "
                    "modify_file = file changes (rm, mv, cp, mkdir, tee, redirects), "
                    "web_fetch = network commands (curl, wget), "
                    "installation = package management (pip, npm, brew, apt), "
                    "remote_execution = remote access (ssh, docker, kubectl), "
                    "compound = piped or chained commands."
                ),
            },
            "timeout_s": {
                "type": "integer",
                "description": (
                    "Timeout in seconds (default 30, max 300). "
                    "The command is killed if it exceeds this limit."
                ),
            },
        },
        "required": ["command", "action"],
    }

    def get_description(self) -> str:
        cwd = Path.home()
        return self.SUMMARY + f"Working directory: {cwd}. Use absolute paths or cd to operate elsewhere."

    def run(self, params: dict) -> dict:
        command = (params.get("command") or "").strip()
        if not command:
            return {"text": "Error: command is required."}

        # Escalate the LLM's self-classification via heuristic inspection so a
        # destructive command can never be demoted to a benign action class.
        llm_action = params.get("action", "execute")
        params = {**params, "action": _resolve_action(command, llm_action)}

        destructive_error = _check_destructive(command)
        if destructive_error:
            return {"text": destructive_error}

        timeout_s = _resolve_timeout(params.get("timeout_s"))
        safe_env = _build_safe_env()

        return _run_command(command, timeout_s, safe_env)


# ── Classification ─────────────────────────────────────────────────────────


def _resolve_action(command: str, llm_action: str) -> str:
    """Apply heuristic overrides — can only ESCALATE, never demote."""
    heuristic = _classify_heuristic(command)
    if heuristic is None:
        return llm_action

    llm_severity = _ACTION_SEVERITY.get(llm_action, 1)
    heuristic_severity = _ACTION_SEVERITY.get(heuristic, 1)

    if heuristic_severity > llm_severity:
        return heuristic
    return llm_action


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
    """Check if *needle* appears in *command* outside of quoted strings."""
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


def _run_command(command: str, timeout_s: int, env: dict) -> dict:
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            timeout=timeout_s,
            cwd=str(Path.home()),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"text": json.dumps({
            "error": f"Command timed out after {timeout_s}s and was killed.",
            "returncode": -1,
            "truncated": False,
        })}
    except OSError as exc:
        return {"text": f"Error: failed to execute command: {exc}"}

    stdout_raw = proc.stdout or b""
    stderr_raw = proc.stderr or b""

    if _is_binary(stdout_raw) or _is_binary(stderr_raw):
        total = len(stdout_raw) + len(stderr_raw)
        return {"text": json.dumps({
            "stdout": f"(binary output, {len(stdout_raw)} bytes)",
            "stderr": f"(binary output, {len(stderr_raw)} bytes)" if stderr_raw else "",
            "returncode": proc.returncode,
            "truncated": False,
            "binary": True,
            "total_bytes": total,
        })}

    truncated = False
    combined_len = len(stdout_raw) + len(stderr_raw)
    if combined_len > _MAX_OUTPUT_BYTES:
        truncated = True
        budget = _MAX_OUTPUT_BYTES
        stdout_budget = min(len(stdout_raw), budget)
        stderr_budget = min(len(stderr_raw), budget - stdout_budget)
        stdout_raw = stdout_raw[:stdout_budget]
        stderr_raw = stderr_raw[:stderr_budget]

    stdout_str = stdout_raw.decode("utf-8", errors="replace")
    stderr_str = stderr_raw.decode("utf-8", errors="replace")

    if truncated:
        if stdout_str:
            stdout_str += _TRUNCATION_NOTICE
        elif stderr_str:
            stderr_str += _TRUNCATION_NOTICE

    if not stdout_str and not stderr_str and proc.returncode == 0:
        stdout_str = _EMPTY_SUCCESS_MSG

    return {"text": json.dumps({
        "stdout": stdout_str,
        "stderr": stderr_str,
        "returncode": proc.returncode,
        "truncated": truncated,
    })}


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


def _build_safe_env() -> dict:
    safe = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(upper.startswith(p) for p in _SECRET_PREFIXES):
            continue
        if any(upper.endswith(s) for s in _SECRET_SUFFIXES):
            continue
        safe[key] = value
    return safe
