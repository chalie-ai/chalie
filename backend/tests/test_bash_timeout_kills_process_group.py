"""Feature test: a bash command that outlives its timeout dies with everything it started.

Real BashAbility run end to end, no mocks. The command backgrounds a child,
records its pid and waits on it; once the timeout fires the child must be gone
too, not orphaned to run on with no limit.
"""

import subprocess
import time
from pathlib import Path

import pytest

from abilities.bash import BashAbility
from contracts.params.bash_params_bag import BashParamsBag

pytestmark = pytest.mark.unit

# The kill is sent before the call returns; the child's state follows a moment later.
_KILL_GRACE_S = 3.0


def _dead_within(pid: int, seconds: float) -> bool:
    """True once ``pid`` no longer runs: gone, or a zombie nobody has reaped yet.

    A killed orphan stays a zombie for good under an init that never reaps (a
    container whose first process is a plain ``sleep``), and a zombie still
    answers signal 0 — so read its state instead.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        ps = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False)
        state = ps.stdout.strip()
        if not state or state.startswith("Z"):
            return True
        time.sleep(0.05)
    return False


def test_a_timeout_kills_the_children_the_command_started(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"

    result = BashAbility().run(BashParamsBag(command=f"sleep 60 & echo $! > {pid_file}; wait", timeout_s=1))

    assert result.status == "error"
    assert result.code == "command-timeout"
    assert _dead_within(int(pid_file.read_text()), _KILL_GRACE_S), "the backgrounded child outlived the timeout"


def test_a_command_that_finishes_in_time_still_returns_its_output() -> None:
    result = BashAbility().run(BashParamsBag(command="echo out; echo err >&2; exit 3", timeout_s=10))

    assert result.code is None
    body = result.body
    assert isinstance(body, dict)
    assert body["exit_code"] == 3
    assert str(body["stdout"]).strip() == "out"
    assert str(body["stderr"]).strip() == "err"
