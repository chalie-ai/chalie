"""Single import surface for every Chalie custom exception.

Import from the package directly::

    from exceptions import ChalieException, NotFoundError, RateLimitError

Every name lives canonically in :mod:`exceptions.exception`; this module
only re-exports so callers never depend on the internal file layout.
"""

from exceptions.chalie_exception import ChalieException
from exceptions.exception import (
    DownloadTooLarge,
    EmptyCompletionLoop,
    EndpointError,
    FetchBlocked,
    ForbiddenError,
    McpServerUnreachable,
    McpToolUnknown,
    NewsFetchError,
    NoReadableContent,
    NoTextContent,
    NotAFile,
    NotFoundError,
    ProviderError,
    ProviderResponseError,
    ProviderRetriesExhaustedError,
    ProviderTimeoutError,
    RateLimitError,
    RateLimitException,
    RequestOverCapError,
    ResponseOverLimitError,
    RunAwayLoop,
    SnapshotError,
    SourceIsImage,
    SystemPathBlocked,
    ToolParamError,
    VaultLockedError,
)

__all__ = [
    "ChalieException",
    "DownloadTooLarge",
    "EmptyCompletionLoop",
    "EndpointError",
    "FetchBlocked",
    "ForbiddenError",
    "McpServerUnreachable",
    "McpToolUnknown",
    "NewsFetchError",
    "NoReadableContent",
    "NoTextContent",
    "NotAFile",
    "NotFoundError",
    "ProviderError",
    "ProviderResponseError",
    "ProviderRetriesExhaustedError",
    "ProviderTimeoutError",
    "RateLimitError",
    "RateLimitException",
    "RequestOverCapError",
    "ResponseOverLimitError",
    "RunAwayLoop",
    "SnapshotError",
    "SourceIsImage",
    "SystemPathBlocked",
    "ToolParamError",
    "VaultLockedError",
]
