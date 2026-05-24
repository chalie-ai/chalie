# Copyright 2026 Dylan Grech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LLM request logger — writes every LLM call to its own file, verbatim.

Filename: logs/{caller}-{utc_now().strftime('%Y%m%dT%H%M%S_%f')}.log
Contents:
  # caller: <ClassName or string>
  # job: <job name>
  # provider: <provider name>
  # model: <model id>
  # timestamp: <iso8601 utc>

  user-message: <verbatim final user-message string>

  system-message: <verbatim final system-message string>

  tools: <JSON-serialised tools list, indent=2>

One file per LLM request. Never append. Exceptions are swallowed with
logger.debug() — logging must NEVER block the LLM call.
"""

import json
import logging
import threading

from services.file_mapper_service import FileMapperService
from services.time_utils import utc_now

logger = logging.getLogger(__name__)

_LOGS_DIR = FileMapperService.get_logs_path()

# First-failure-warning flag. Disk-full / permission errors are shouted once
# per process so Dylan notices in the logs; subsequent failures drop to debug
# to avoid spamming an already broken disk.
_warned_on_failure = False
_warn_lock = threading.Lock()


def log_llm_request(
    caller: str,
    job: str,
    provider: str,
    model: str,
    system_message: str,
    user_message: str,
    tools: list | None,
) -> None:
    """Write one LLM request to its own log file.

    Best-effort — any failure is swallowed so the caller's LLM flow is never
    blocked. The first failure per process is surfaced at WARNING level so
    disk-full / permission problems don't stay silent; all subsequent
    failures fall to DEBUG to avoid spamming.

    Filename: ``logs/{caller}-{YYYYMMDDTHHMMSS_ffffff}-{tid}.log``. The thread
    id suffix makes concurrent calls from the same microsecond collision-free.
    """
    try:
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        timestamp = now.strftime('%Y%m%dT%H%M%S_%f')
        safe_caller = caller or 'unknown'
        tid = threading.get_ident()
        filepath = _LOGS_DIR / f"{safe_caller}-{timestamp}-{tid}.log"

        tools_json = json.dumps(tools or [], indent=2, ensure_ascii=False)

        body = (
            f"# caller: {safe_caller}\n"
            f"# job: {job}\n"
            f"# provider: {provider}\n"
            f"# model: {model}\n"
            f"# timestamp: {now.isoformat()}\n"
            f"\n"
            f"user-message: {user_message}\n"
            f"\n"
            f"system-message: {system_message}\n"
            f"\n"
            f"tools: {tools_json}\n"
        )

        filepath.write_text(body, encoding='utf-8')
    except Exception as e:
        global _warned_on_failure
        with _warn_lock:
            first = not _warned_on_failure
            _warned_on_failure = True
        if first:
            logger.warning(
                f"[LLM LOG] Failed to write request log (first failure — "
                f"subsequent failures will be debug-only): {e}",
                exc_info=True,
            )
        else:
            logger.debug(f"[LLM LOG] Failed to write request log: {e}", exc_info=True)
