"""Canonical skill-output block formatter.

Single source of truth for the [<name>(...)] ... [end:<name>] wire format.
Every innate skill must use tag() — no skill builds its own format strings.
"""


def tag(name: str, content: str = "", **args: object) -> str:
    """Return canonical skill-output block."""
    arg_str = ", ".join(f"{k}={str(v)}" for k, v in args.items())
    opener = f"[{name}({arg_str})]"
    if content:
        return f"{opener}\n{content}\n[end:{name}]"
    return f"{opener}\n[end:{name}]"
