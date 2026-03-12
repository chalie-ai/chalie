"""
Shared utility helpers for the backend.

Exports:
    Logger: Thin static wrapper around Python's ``logging`` module that
        centralises log configuration (file path, format, level) for all
        backend processes.  Additional helpers such as ``text_utils`` are
        available as sub-modules but are imported directly where needed.
"""

from .logger import Logger


__all__ = [
'Logger'
]
