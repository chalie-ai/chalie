"""
Web Search Tool Runner — Generates an inline HTML carousel card.

One result per slide. JS wiring via [data-carousel] convention in tool_result.js.
Outputs IPC contract: {"text": str, "html": str, "results": [...], "_meta": {...}}
"""

import sys
import json
import base64
from html import escape
from handler import execute


# ── Radiant palette ───────────────────────────────────────────────────────────

_ACCENT = "#1a8fff"
_ACCENT_BG = "rgba(26,143,255,0.15)"
_TEXT_PRIMARY = "#eae6f2"
_TEXT_SECONDARY = "rgba(234,230,242,0.58)"
_TEXT_TERTIARY = "rgba(234,230,242,0.38)"
_SURFACE = "rgba(255,255,255,0.04)"
_BORDER = "rgba(255,255,255,0.07)"
_DOT_ACTIVE = "#8A5CFF"
_DOT_INACTIVE = "rgba(255,255,255,0.25)"


# ── SVG icons ─────────────────────────────────────────────────────────────────

_LINK_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2.5" '
    'stroke-linecap="round" stroke-linejoin="round" '
    'style="vertical-align:middle;flex-shrink:0;">'
    '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>'
    '<polyline points="15 3 21 3 21 9"/>'
    '<line x1="10" y1="14" x2="21" y2="3"/>'
    '</svg>'
)

_CHEVRON_LEFT = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2.5" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<polyline points="15 18 9 12 15 6"/>'
    '</svg>'
)

_CHEVRON_RIGHT = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2.5" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<polyline points="9 18 15 12 9 6"/>'
    '</svg>'
)


# ── Slide rendering ───────────────────────────────────────────────────────────

def _render_slide(result: dict, visible: bool) -> str:
    title = result.get("title") or ""
    snippet = result.get("snippet") or ""
    url = result.get("url") or ""
    domain = result.get("domain") or ""
    display = "flex" if visible else "none"

    domain_html = ""
    if domain:
        domain_html = (
            f'<div style="font-size:11px;color:{_ACCENT};font-weight:600;margin-bottom:4px;">'
            f'{escape(domain)}</div>'
        )

    snippet_html = ""
    if snippet:
        snippet_html = (
            f'<p style="font-size:13px;color:{_TEXT_SECONDARY};'
            f'line-height:1.55;margin:0 0 8px 0;">{escape(snippet)}</p>'
        )

    return (
        f'<div data-slide '
        f'style="display:{display};flex-direction:column;'
        f'padding:13px 15px;background:{_SURFACE};'
        f'border-radius:9px;border:1px solid {_BORDER};">'
        + domain_html
        + f'<div style="font-weight:600;font-size:14px;color:{_TEXT_PRIMARY};'
          f'line-height:1.3;margin-bottom:6px;">{escape(title)}</div>'
        + snippet_html
        + f'<a href="{escape(url)}" target="_blank" rel="noopener noreferrer" '
          f'style="display:inline-flex;align-items:center;gap:5px;'
          f'color:{_ACCENT};font-size:12px;text-decoration:none;opacity:0.85;">'
        + _LINK_ICON
        + '<span>Open page</span>'
        + '</a>'
        + '</div>'
    )


# ── Navigation ────────────────────────────────────────────────────────────────

def _render_navigation(count: int) -> str:
    btn_style = (
        f"background:{_SURFACE};border:1px solid rgba(255,255,255,0.12);"
        "border-radius:50%;width:28px;height:28px;display:inline-flex;align-items:center;"
        "justify-content:center;cursor:pointer;color:rgba(234,230,242,0.7);padding:0;"
        "flex-shrink:0;outline:none;"
        "transition:background 220ms ease,border-color 220ms ease,color 220ms ease;"
    )
    dots = "".join(
        f'<span data-dot style="'
        + (
            f"width:7px;height:7px;border-radius:50%;background:{_DOT_ACTIVE};"
            "transform:scale(1.2);flex-shrink:0;cursor:pointer;transition:all 220ms ease;"
            if i == 0 else
            f"width:7px;height:7px;border-radius:50%;background:{_DOT_INACTIVE};"
            "flex-shrink:0;cursor:pointer;transition:all 220ms ease;"
        )
        + '"></span>'
        for i in range(count)
    )
    return (
        '<div style="display:flex;align-items:center;justify-content:center;'
        'gap:8px;margin-top:10px;">'
        + f'<button type="button" data-prev style="{btn_style}">{_CHEVRON_LEFT}</button>'
        + f'<div style="display:flex;align-items:center;gap:5px;">{dots}</div>'
        + f'<button type="button" data-next style="{btn_style}">{_CHEVRON_RIGHT}</button>'
        + '</div>'
    )


# ── Card assembly ─────────────────────────────────────────────────────────────

def _render_html(results: list) -> str:
    results = results[:8]
    if not results:
        return (
            f'<p style="color:{_TEXT_TERTIARY};font-size:13px;'
            f'font-family:system-ui,-apple-system,sans-serif;padding:12px 14px;margin:0;">'
            f'No web results found.</p>'
        )
    slides = "".join(_render_slide(r, i == 0) for i, r in enumerate(results))
    nav = _render_navigation(len(results)) if len(results) > 1 else ""
    return (
        '<div data-carousel '
        'style="font-family:system-ui,-apple-system,sans-serif;">'
        + slides + nav + '</div>'
    )


# ── Text for LLM synthesis ────────────────────────────────────────────────────

def _format_text(results: list, query: str) -> str:
    if not results:
        return (
            f'No web results found for "{query}". '
            f'Try a different query or a domain-specific tool (stackexchange, hackernews, arxiv, news_tool).'
        )
    lines = [f'Web search results for "{query}":']
    for i, r in enumerate(results, 1):
        lines.append(f"\n{i}. {r.get('title', '')}")
        if domain := r.get("domain", ""):
            lines.append(f"   Source: {domain}")
        if snippet := r.get("snippet", ""):
            lines.append(f"   {snippet}")
        if url := r.get("url", ""):
            lines.append(f"   {url}")
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

try:
    payload = json.loads(base64.b64decode(sys.argv[1]))
    params = payload.get("params", {})
    settings = payload.get("settings", {})
    telemetry = payload.get("telemetry", {})

    result = execute(topic="", params=params, config=settings, telemetry=telemetry)
    results = result.get("results", [])

    output = {
        "results": results,
        "count": result.get("count", 0),
        "text": _format_text(results, params.get("query", "")),
        "html": _render_html(results) if results else None,
        "_meta": result.get("_meta", {}),
    }
    if "error" in result:
        output["error"] = result["error"]

    print(json.dumps(output))
except Exception as _e:
    print(json.dumps({
        "results": [], "count": 0,
        "error": f"Runner error: {str(_e)[:200]}",
        "text": f"Web search failed: {str(_e)[:200]}",
        "html": None, "_meta": {},
    }), file=sys.stdout)
    print(f"[web_search runner] Unhandled exception: {_e}", file=sys.stderr)
