"""
Pure formatter — renders a ranked list of search result dicts to an XML-like
string for model consumption.

Result contract:
  Each dict is ``{title, url, summary, score: float|None, date: str|None}``.
  Results are pre-sorted best-first by the caller.  ``render_records`` assigns
  1-based ``index`` attributes matching the supplied order and emits:

      <result index="1" score="0.87" date="2026-06-15">
      title: <title>
      url: <url>
      summary: <full summary — NEVER truncated>
      </result>

  ``score`` is rendered to 2 decimal places; omitted entirely when ``None``.
  ``date`` is omitted entirely when ``None`` or empty.
  Blocks are joined by "\\n".
  An empty ``results`` list returns a short sentinel that does NOT contain
  ``<result index=``.
"""

from typing import cast


class SearchRenderer:
    """Pure formatter for search result dicts to XML-like strings."""

    @staticmethod
    def _neutralize(text: str) -> str:
        """Defang record-boundary tokens in a free-text field.

        A search result whose own ``title`` or ``summary`` contains the literal
        ``<result …>`` or ``</result>`` token could otherwise be mistaken by the
        model for a record delimiter, corrupting its parse of the block. We escape
        only those two tokens — every other ``<``/``>`` (code snippets, math,
        ``List<int>``) is left intact so the summary stays readable.
        """
        return text.replace("</result>", "<\\/result>").replace("<result", "<\\result")

    @staticmethod
    def render_records(results: list[dict[str, object]]) -> str:
        """Render *results* to a string of XML-like ``<result>`` blocks.

        Args:
            results: Pre-sorted (best-first) list of result dicts.  Each dict must
                have ``title``, ``url``, ``summary``, ``score`` (float or None),
                and ``date`` (str or None).

        Returns:
            A multi-block string with one ``<result …>…</result>`` block per item,
            joined by ``"\\n"``; or a short sentinel when *results* is empty.
        """
        if not results:
            return "No results found."

        blocks: list[str] = []
        for index, r in enumerate(results, start=1):
            score: float | None = cast("float | None", r.get("score"))
            date: str | None = cast("str | None", r.get("date")) or None

            # Build open-tag attributes
            attrs = f'index="{index}"'
            if score is not None:
                attrs += f' score="{score:.2f}"'
            if date:
                attrs += f' date="{date}"'

            title = SearchRenderer._neutralize(str(r.get("title", "")))
            summary = SearchRenderer._neutralize(str(r.get("summary", "")))
            block = (
                f"<result {attrs}>\n"
                f"title: {title}\n"
                f"url: {r.get('url', '')}\n"
                f"summary: {summary}\n"
                f"</result>"
            )
            blocks.append(block)

        return "\n".join(blocks)
