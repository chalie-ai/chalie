"""Unit tests for bash.py classification functions."""

import pytest

from abilities.bash import (
    _ACTION_SEVERITY,
    _check_destructive,
    _classify_heuristic,
    _has_recursive_flag,
    _is_rm_rf,
    _resolve_action,
)

pytestmark = pytest.mark.unit

_BLOCKED = (
    "BLOCKED: Destructive command pattern detected. This command cannot be executed."
)


# ---------------------------------------------------------------------------
# _has_recursive_flag — pure token list inspection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tokens",
    [
        ["rm", "-r"],
        ["rm", "-R"],
        ["rm", "-rf"],
        ["rm", "-fr"],
        ["rm", "-rR"],
        ["rm", "--recursive"],
        ["rm", "-v", "-r"],
        ["rm", "-r", "somefile"],
        # function is command-agnostic; only inspects tokens[1:]
        ["ls", "-r"],
        ["find", "--recursive"],
    ],
)
def test_has_recursive_flag_returns_true(tokens):
    assert _has_recursive_flag(tokens) is True


@pytest.mark.parametrize(
    "tokens",
    [
        ["rm"],
        [],
        ["rm", "-f"],
        ["rm", "-v"],
        # flags after a positional arg are never seen (early stop on first non-flag)
        ["rm", "somefile", "-r"],
        ["rm", "path", "-R"],
    ],
)
def test_has_recursive_flag_returns_false(tokens):
    assert _has_recursive_flag(tokens) is False


# ---------------------------------------------------------------------------
# _is_rm_rf — shlex-split + flag detection with early stop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /tmp/foo",
        "rm -fr /tmp/foo",
        "rm -r -f /tmp/foo",
        "rm -f -r /tmp/foo",
        "rm --recursive --force /tmp",
        "rm --force --recursive /tmp",
        "rm -rRf /tmp",
        "rm -rf",  # flags only, no path — still True
        "rm -Rf /path",  # capital R
    ],
)
def test_is_rm_rf_returns_true(command):
    assert _is_rm_rf(command) is True


@pytest.mark.parametrize(
    "command",
    [
        "rm -r /tmp",  # no force flag
        "rm -f /tmp",  # no recursive flag
        "rm /tmp",  # no flags
        "rm",  # no tokens after rm
        "",  # empty string
        "ls -rf",  # not rm
        "find . -rf",  # not rm
        "rm /path -rf",  # flags appear after positional — early stop → False
        "rm 'unclosed",  # shlex ValueError → False
        "rm --recursive /path",  # --force missing
        "rm --force /path",  # --recursive missing
    ],
)
def test_is_rm_rf_returns_false(command):
    assert _is_rm_rf(command) is False


# ---------------------------------------------------------------------------
# _classify_heuristic — ordered-chain classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "ls && echo done",
        "ls || echo fail",
        "ls; echo done",
        "ls | grep foo",
        "echo $(date)",
        "echo `date`",
        "eval ls",
        "exec ls",
        "source ~/.bashrc",
        ". ~/.bashrc",  # dot is a compound keyword
        "rm -r /path",  # rm + recursive flag → compound (step 8 beats step 9)
        "rm --recursive /path",
        "echo 'unclosed",  # shlex ValueError → compound
    ],
)
def test_classify_heuristic_returns_compound(command):
    assert _classify_heuristic(command) == "compound"


@pytest.mark.parametrize(
    "command",
    [
        'echo "hello && world"',
        # backslash-escaped quote inside double-quoted string — pipe stays quoted
        'echo "he\\"llo | world"',
    ],
)
def test_classify_heuristic_quoted_operators_not_compound(command):
    assert _classify_heuristic(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "ssh user@host",
        "docker run ubuntu",
        "kubectl get pods",
        "scp file user@host:/path",
        "rsync -a src/ dst/",
    ],
)
def test_classify_heuristic_returns_remote_execution(command):
    assert _classify_heuristic(command) == "remote_execution"


@pytest.mark.parametrize(
    "command",
    [
        "curl https://example.com",
        "wget https://example.com",
        "nc -zv host 80",
        "ncat host 80",
    ],
)
def test_classify_heuristic_returns_web_fetch(command):
    assert _classify_heuristic(command) == "web_fetch"


@pytest.mark.parametrize(
    "command",
    [
        "pip install requests",
        "pip3 install flask",
        "npm install",
        "yarn add lodash",
        "brew install git",
        "apt-get update",
        "cargo build",
        "gem install bundler",
        "apt install vim",
    ],
)
def test_classify_heuristic_returns_installation(command):
    assert _classify_heuristic(command) == "installation"


@pytest.mark.parametrize(
    "command",
    [
        "rm /tmp/foo",  # rm without recursive (step 8 skipped → step 9)
        "rm /path -rf",  # flags after positional → early stop → step 9
        "mv src dst",
        "cp src dst",
        "mkdir /tmp/new",
        "rmdir /tmp/dir",
        "touch /tmp/file",
        "chmod 755 file",
        "chown user file",
        "tee /tmp/out",
        "echo hello > /tmp/out",  # unquoted redirect
        "cat file > other",
    ],
)
def test_classify_heuristic_returns_modify_file(command):
    assert _classify_heuristic(command) == "modify_file"


@pytest.mark.parametrize(
    "command",
    [
        "ls /tmp",
        "ps aux",
        "echo hello",
        "df -h",
        "",  # empty → empty tokens → None
        "cat file.txt",
        "grep foo bar",
        "echo 'hello > world'",  # redirect inside single quotes
        "echo 'ls | grep'",  # pipe inside single quotes
        ".gitignore",  # dot-prefixed non-keyword token
    ],
)
def test_classify_heuristic_returns_none(command):
    assert _classify_heuristic(command) is None


# ---------------------------------------------------------------------------
# _check_destructive — _is_rm_rf first, then regex patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # _is_rm_rf triggers (flag-based, no path filter)
        "rm -rf /tmp/safe",
        "rm -fr /etc",
        "rm --recursive --force /home",
        # both _is_rm_rf and pattern 1/2 match
        "rm -rf /",
        # fork bomb
        ":(){:|:&};:",
        # disk format — mkfs\b matches mkfs.ext4 (dot is a word boundary)
        "mkfs.ext4 /dev/sdb",
        # mkfs standalone also blocked
        "echo mkfs is a command",
        # dd zero-wipe
        "dd if=/dev/zero of=/dev/sda",
        # dd random-wipe
        "dd if=/dev/random of=/dev/sdb",
        # redirect to block device
        "> /dev/sda",
        # rm -rf /*
        "rm -rf /*",
    ],
)
def test_check_destructive_returns_blocked(command):
    assert _check_destructive(command) == _BLOCKED


@pytest.mark.parametrize(
    "command",
    [
        "rm -r /tmp/foo",  # no force flag → _is_rm_rf False; no regex match
        "rm -f /tmp/foo",  # no recursive → _is_rm_rf False; no regex match
        "ls /",
        "df -h",
        "echo hello",
        "cat /etc/hosts",
        # mkfs\b requires next char to be non-word; 't' in "mkfstools" is a word char
        "mkfstools",
    ],
)
def test_check_destructive_returns_none(command):
    assert _check_destructive(command) is None


# ---------------------------------------------------------------------------
# _resolve_action — escalation only, never demote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command,llm_action,expected",
    [
        # heuristic=None → return llm_action unchanged
        ("ls /tmp", "read", "read"),
        ("echo hello", "execute", "execute"),
        ("", "read", "read"),
        # unknown llm_action → default severity 1; heuristic=None → pass through
        ("echo hello", "unknown_action", "unknown_action"),
        # escalation: heuristic severity > llm severity
        ("curl https://x.com", "read", "web_fetch"),  # 3 > 0
        ("ssh user@host", "read", "remote_execution"),  # 5 > 0
        ("pip install flask", "execute", "installation"),  # 4 > 1
        ("docker run ubuntu", "execute", "remote_execution"),  # 5 > 1
        ("ls && pwd", "read", "compound"),  # 6 > 0
        ("rm /tmp/foo", "read", "modify_file"),  # 2 > 0
        ("rm -r /path", "modify_file", "compound"),  # 6 > 2
        # equal severity → keep llm_action (not heuristic)
        ("curl https://x.com", "web_fetch", "web_fetch"),
        ("pip install flask", "installation", "installation"),
        # no demote: llm_action already higher than heuristic
        ("curl https://x.com", "compound", "compound"),  # 6 > 3, keep llm
        ("ssh user@host", "compound", "compound"),  # 6 > 5, keep llm
        # unknown llm_action gets default severity 1; heuristic=web_fetch(3) → escalate
        ("curl x.com", "unknown_action", "web_fetch"),
    ],
)
def test_resolve_action(command, llm_action, expected):
    assert _resolve_action(command, llm_action) == expected


def test_action_severity_keys_cover_all_classifiable_actions():
    """Every value _classify_heuristic can return must have a severity entry."""
    classifiable = {
        "compound",
        "remote_execution",
        "web_fetch",
        "installation",
        "modify_file",
    }
    assert classifiable.issubset(_ACTION_SEVERITY.keys())
