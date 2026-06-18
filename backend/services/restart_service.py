# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Process restart helper.

A single seam for requesting an in-process restart: a daemon thread waits
briefly so the in-flight HTTP response can flush, then exits with code 42,
which ``run.sh`` interprets as a restart signal. Used by the snapshot-import
flow so a staged restore is applied at the next boot.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)


def request_restart() -> None:
    """Daemon thread waits 2 s (HTTP response to flush) then ``os._exit(42)``.

    Exit code 42 signals ``run.sh`` to restart the process.
    """
    def _deferred_exit() -> None:
        import time
        time.sleep(2)
        logger.info("Restarting Chalie (exit code 42)")
        os._exit(42)

    thread = threading.Thread(target=_deferred_exit, daemon=True)
    thread.start()
    logger.info("Restart requested — exiting in 2 seconds")
