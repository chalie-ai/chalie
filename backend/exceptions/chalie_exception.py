"""Base class for every Chalie custom exception.

``ChalieException`` logs itself at ERROR level the instant it is raised, so a
failure can never vanish silently: the constructor runs on every
``raise <ChalieException>(...)``. The module logger propagates to the root
handler that :func:`utils.logger.Logger.start` attaches, which writes
the JSON entry to the application log file.

Subclasses keep their own ``__init__`` signatures and call ``super().__init__``
(which resolves here). Because each concrete class constructs exactly once per
raise, the log fires exactly once — the inheritance chain never double-logs.
"""

import logging

logger = logging.getLogger(__name__)


class ChalieException(Exception):
    """Base for every custom exception in Chalie.

    Logging on construct means each ``raise`` of a subclass emits one ERROR
    line carrying the exception class name and its message. Subclasses override
    ``__init__`` freely; the single ``super().__init__`` call they already make
    lands here and triggers the log.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        message = args[0] if args else ""
        logger.error("%s raised: %s", type(self).__name__, message)
