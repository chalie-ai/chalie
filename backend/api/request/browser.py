"""Request DTOs for the browser's live tabs: the REST bodies and the stream socket's input messages."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from .request import Request

_URL_MAX = 8192
_TEXT_MAX = 4096
_KEY_MAX = 64

_Coordinate = Annotated[float, Field(ge=-100_000, le=100_000, allow_inf_nan=False)]
_Modifiers = Annotated[int, Field(ge=0, le=15)]   # Alt=1, Ctrl=2, Meta=4, Shift=8


class BrowserOpenTabRequest(Request):
    """Inbound body opening a user tab; without ``url`` the tab opens blank."""

    url: str | None = Field(default=None, max_length=_URL_MAX)


class BrowserNavigateRequest(Request):
    """Inbound body pointing a tab at a url."""

    url: str = Field(..., max_length=_URL_MAX)


class BrowserHistoryRequest(Request):
    """Inbound body moving a tab through its history."""

    action: Literal["back", "forward", "reload"]


class BrowserMouseInput(Request):
    """A mouse button or move at page CSS px."""

    type: Literal["mouse"]
    event: Literal["mousePressed", "mouseReleased", "mouseMoved"]
    x: _Coordinate
    y: _Coordinate
    button: Literal["left", "middle", "right", "none"]
    click_count: int = Field(..., alias="clickCount", ge=0, le=100)
    modifiers: _Modifiers


class BrowserWheelInput(Request):
    """A wheel scroll at page CSS px."""

    type: Literal["wheel"]
    x: _Coordinate
    y: _Coordinate
    delta_x: _Coordinate = Field(..., alias="deltaX")
    delta_y: _Coordinate = Field(..., alias="deltaY")
    modifiers: _Modifiers


class BrowserKeyInput(Request):
    """A key going down or up; ``text`` is what a key-down types, if anything."""

    type: Literal["key"]
    event: Literal["keyDown", "keyUp"]
    key: str = Field(..., max_length=_KEY_MAX)
    code: str = Field(..., max_length=_KEY_MAX)
    text: str | None = Field(..., max_length=_KEY_MAX)
    key_code: int = Field(..., alias="keyCode", ge=0, le=1023)
    modifiers: _Modifiers


class BrowserTextInput(Request):
    """Text inserted at the caret, as a paste does."""

    type: Literal["text"]
    text: str = Field(..., min_length=1, max_length=_TEXT_MAX)


class BrowserSizeInput(Request):
    """The viewer's size in CSS px and its device pixel ratio — the window the tab's page lays out for."""

    type: Literal["size"]
    width: float = Field(..., ge=1, le=16_384, allow_inf_nan=False)
    height: float = Field(..., ge=1, le=16_384, allow_inf_nan=False)
    scale: float = Field(..., gt=0, le=64, allow_inf_nan=False)


BrowserInput = Annotated[
    BrowserMouseInput | BrowserWheelInput | BrowserKeyInput | BrowserTextInput | BrowserSizeInput,
    Field(discriminator="type"),
]
browser_input: TypeAdapter[BrowserInput] = TypeAdapter(BrowserInput)
"""Validates one stream-socket message (a JSON string) into its input DTO."""
