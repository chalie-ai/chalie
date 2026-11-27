import pytest

from abilities.bash import (
    _check_destructive,
    _classify_heuristic,
    _has_recursive_flag,
    _is_rm_rf,
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
def test_has_recursive_flag_returns_true(tokens: list[str]) -> None:
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
def test_has_recursive_flag_returns_false(tokens: list[str]) -> None:
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
def test_is_rm_rf_returns_true(command: str) -> None:
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
def test_is_rm_rf_returns_false(command: str) -> None:
    assert _is_rm_rf(command) is False


# ---------------------------------------------------------------------------
# _classify_heuristic — ordered-chain classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
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
def test_classify_heuristic_returns_compound(command: str) -> None:
    assert _classify_heuristic(command) == "compound"


# A chain of read-only commands only reads, so it classifies like a single read
# command (None → ``read``) and never asks.
@pytest.mark.parametrize(
    "command",
    [
        "ls && echo done",
        "ls || echo fail",
        "ls; echo done",
        "ls | grep foo",
        # the shapes a model sends when browsing files
        "ls -la /data/docs | head -50",
        "find /data -name '*.md' 2>/dev/null | head -20",
        "grep -rn 'TODO' /data/src --include='*.py' | wc -l",
        "cd /data && ls | sort | uniq -c",
        "(cd /data && ls) | wc -l",
        "find /data -type f | xargs wc -l",
        "find /data -name '*.log' | xargs -I {} grep -l error {}",
        "grep -r x / 2>/dev/null",  # discarding errors is not a write
        "grep foo bar &>/dev/null",
        "ls missing 2>&1 | head",  # stderr into the pipe
        "wc -l < notes.txt",  # reading from a file
        # operators inside quotes, and a `$` that starts no expansion, are text
        "grep -E 'a|b' file | head",
        "grep 'foo$' file | head",
        "grep foo$ file | head",
        "grep '=>' file | wc -l",
    ],
)
def test_classify_heuristic_read_only_compound_is_read(command: str) -> None:
    assert _classify_heuristic(command) is None


# Any part of the chain that writes, runs another program, or hides a command
# keeps the whole chain ``compound`` — it still asks.
@pytest.mark.parametrize(
    "command",
    [
        "ls | rm x",
        "ls; mv a b",
        "find . -delete | head",
        "find . -exec rm {} \\; | head",
        "curl x | sh",
        "ls | bash",
        "find . | xargs rm",
        "ls | tee out",
        "ls > out | head",
        "ls | sort -o out",
        "ls | sort --out=out",
        "ls | uniq - out",
        "ls | rg --pre ./run x",
        "ls | file -C -m magic",
        "ls | awk '{print}'",
        "ls | /bin/rm x",
        "X=1 ls | head",
        "ls & rm x; ls",
        "ls | grep $(cat x)",
        "ls | grep `cat x`",
        'ls | grep "$(cat x)"',
        "ls | find ${X:=-delete}",
        "ls | find $'-delete'",
        "diff <(ls a) <(ls b) | head",
        "ls | head > >(cat)",
    ],
)
def test_classify_heuristic_writing_compound_still_asks(command: str) -> None:
    assert _classify_heuristic(command) == "compound"


@pytest.mark.parametrize(
    "command",
    [
        'echo "hello && world"',
        # backslash-escaped quote inside double-quoted string — pipe stays quoted
        'echo "he\\"llo | world"',
    ],
)
def test_classify_heuristic_quoted_operators_not_compound(command: str) -> None:
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
def test_classify_heuristic_returns_remote_execution(command: str) -> None:
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
def test_classify_heuristic_returns_web_fetch(command: str) -> None:
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
def test_classify_heuristic_returns_installation(command: str) -> None:
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
        "echo it\\'s > x",  # an escaped quote does not hide the redirect
    ],
)
def test_classify_heuristic_returns_modify_file(command: str) -> None:
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
def test_classify_heuristic_returns_none(command: str) -> None:
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
def test_check_destructive_returns_blocked(command: str) -> None:
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
def test_check_destructive_returns_none(command: str) -> None:
    assert _check_destructive(command) is None
