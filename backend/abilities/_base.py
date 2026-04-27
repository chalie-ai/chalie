from abc import ABC, abstractmethod
from typing import ClassVar


class Ability(ABC):
    NAME: ClassVar[str]
    SUMMARY: ClassVar[str]
    EXAMPLES: ClassVar[list[str]]
    INPUT_SCHEMA: ClassVar[dict]
    ALWAYS_AVAILABLE: ClassVar[bool] = False
    INTERNAL: ClassVar[bool] = False
    TIMEOUT: ClassVar[int] = 10

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Abstract subclasses (marked with abstractmethod) are exempt.
        if ABC in cls.__bases__:
            return
        # Skip intermediate abstract classes that still have abstract methods.
        if getattr(cls, "__abstractmethods__", None):
            return
        for attr in ("NAME", "SUMMARY", "EXAMPLES", "INPUT_SCHEMA"):
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define class attribute '{attr}'")
        if not isinstance(cls.EXAMPLES, list) or not all(
            isinstance(e, str) for e in cls.EXAMPLES
        ):
            raise TypeError(f"{cls.__name__}.EXAMPLES must be list[str]")
        if not (6 <= len(cls.EXAMPLES) <= 8):
            raise TypeError(
                f"{cls.__name__}.EXAMPLES must have 6–8 entries, got {len(cls.EXAMPLES)}"
            )

    @abstractmethod
    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict: ...

    def pre_dispatch(self, params: dict) -> None: ...

    def post_dispatch(self, result: dict) -> None: ...
