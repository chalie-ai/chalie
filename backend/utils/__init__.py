"""
Shared utility helpers for the backend.

Exports:
    Logger: Thin static wrapper around Python's ``logging`` module that
        centralises log configuration (file path, format, level) for all
        backend processes.
"""

from .logger import Logger


__all__ = [
'Logger'
]
