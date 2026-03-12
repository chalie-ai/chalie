"""
Thin static wrapper around Python's standard ``logging`` module.

Centralises log configuration so every module in the backend writes
to the same file and format without needing to call
``logging.basicConfig`` individually.  Import and use ``Logger``
instead of calling ``logging`` directly.
"""

import logging


class Logger:
    """Static façade over the standard ``logging`` module.

    All methods are static so callers never need to instantiate this
    class.  Call ``Logger.start()`` once at process startup (e.g. in
    ``run.py`` or ``consumer.py``) before issuing any log messages.
    """

    @staticmethod
    def start():
        """Initialise the root logger with file output and a standard format.

        Configures ``logging.basicConfig`` to write ``INFO``-level (and above)
        messages to ``/tmp/chalie.log``.  This method is idempotent — calling
        it multiple times has no additional effect because ``basicConfig`` is a
        no-op when handlers are already attached.
        """
        logging.basicConfig(
            level=logging.INFO,
            filename='/tmp/chalie.log',
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    @staticmethod
    def info(message):
        """Log an informational message at the INFO level.

        Args:
            message: The message string to record.
        """
        logging.info(message)

    @staticmethod
    def debug(message):
        """Log a diagnostic message at the DEBUG level.

        Args:
            message: The message string to record.
        """
        logging.debug(message)

    @staticmethod
    def warning(message):
        """Log a cautionary message at the WARNING level.

        Args:
            message: The message string to record.
        """
        logging.warning(message)

    @staticmethod
    def error(message):
        """Log an error message at the ERROR level.

        Args:
            message: The message string to record.
        """
        logging.error(message)

    @staticmethod
    def critical(message):
        """Log a severe error message at the CRITICAL level.

        Args:
            message: The message string to record.
        """
        logging.critical(message)