"""Canonical skill-output block formatter.

Single source of truth for the [<name>(...)] ... [end:<name>] wire format.
Every innate skill must use tag() — no skill builds its own format strings.
"""


def tag(name: str, content: str = "", **args) -> str:
    """Return canonical skill-output block.

    Format:
      [<name>(k1=v1, k2=v2)]
      <content>
      [end:<name>]

    If content is empty/None, the body line is omitted. Errors live as args:
    `tag("read", error="fetch-failed")` → `[read(error=fetch-failed)]\\n[end:read]`.

    Arg values are str()-coerced; no quoting, no escaping.
    Iteration order is the caller's keyword-argument insertion order (Python 3.7+).
    """
    arg_str = ", ".join(f"{k}={str(v)}" for k, v in args.items())
    opener = f"[{name}({arg_str})]"
    if content:
        return f"{opener}\n{content}\n[end:{name}]"
    return f"{opener}\n[end:{name}]"
