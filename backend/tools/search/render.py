"""
Pure formatter — renders a list of search result dicts to an XML-like
string for model consumption.

Result contract:
  Each dict is ``{title, url, summary}``.  Results are in caller-supplied
  order (provider-merge order).  ``render_records`` assigns 1-based ``index``
  attributes matching the supplied order and emits:

      <result index="1">
      title: <title>
      url: <url>
      summary: <summary>
      </result>

  An empty/absent field renders as an empty string after its label (e.g.
  ``summary: ``) — never omitted, never a placeholder.  Non-string/absent
  values are defensively coerced to ``""``.  ``title`` is capped at 200 chars
  and ``summary`` at 300 chars UPSTREAM by
  ``tools/search/transformers.cap_result_fields`` — this layer adds no
  truncation of its own.  There are no ``score=`` or ``date=`` attributes.
  Blocks are joined by "\\n".
  An empty ``results`` list returns a short sentinel that does NOT contain
  ``<result index=``.
"""


def _neutralize(text: str) -> str:
    """Defang record-boundary tokens in a free-text field.

    A search result whose own ``title`` or ``summary`` contains the literal
    ``<result …>`` or ``</result>`` token could otherwise be mistaken by the
    model for a record delimiter, corrupting its parse of the block. We escape
    only those two tokens — every other ``<``/``>`` (code snippets, math,
    ``List<int>``) is left intact so the summary stays readable.
    """
    return text.replace("</result>", "<\\/result>").replace("<result", "<\\result")


def render_records(results: list[dict[str, object]]) -> str:
    """Render *results* to a string of XML-like ``<result>`` blocks.

    Args:
        results: List of result dicts in caller-supplied order
            (provider-merge order).  Each dict carries ``title``, ``url``,
            and ``summary``; blank/absent fields render as an empty string
            after their label.

    Returns:
        A multi-block string with one ``<result …>…</result>`` block per item,
        joined by ``"\\n"``; or a short sentinel when *results* is empty.
    """
    if not results:
        return "No results found."

    blocks: list[str] = []
    for index, r in enumerate(results, start=1):
        title = _neutralize(str(r.get("title") or ""))
        url = str(r.get("url") or "")
        summary = _neutralize(str(r.get("summary") or ""))
        block = (
            f'<result index="{index}">\n'
            f"title: {title}\n"
            f"url: {url}\n"
            f"summary: {summary}\n"
            f"</result>"
        )
        blocks.append(block)

    return "\n".join(blocks)
