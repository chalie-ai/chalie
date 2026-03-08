"""
Reddit Tool Runner — Generates an inline HTML carousel card for Reddit posts.

One post is shown at a time; users click arrows or swipe to navigate.
JS wiring lives in frontend/interface/cards/tool_result.js (data-carousel convention).
Outputs formalized IPC contract: {"text": str, "html": str}
"""

import sys
import json
import base64
from html import escape
from handler import execute


# ── SVG icons (inline, no external resources) ─────────────────────────────────

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

_COMMENT_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2.5" '
    'stroke-linecap="round" stroke-linejoin="round" '
    'style="vertical-align:middle;flex-shrink:0;">'
    '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
    '</svg>'
)

_UPVOTE_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2.5" '
    'stroke-linecap="round" stroke-linejoin="round" '
    'style="vertical-align:middle;flex-shrink:0;">'
    '<polyline points="18 15 12 9 6 15"/>'
    '</svg>'
)

# Radiant design palette constants
_ACCENT = "#FF4500"          # Reddit orange
_ACCENT_BG = "rgba(255,69,0,0.15)"
_TEXT_PRIMARY = "#eae6f2"
_TEXT_SECONDARY = "rgba(234,230,242,0.58)"
_TEXT_TERTIARY = "rgba(234,230,242,0.38)"
_SURFACE = "rgba(255,255,255,0.04)"
_BORDER = "rgba(255,255,255,0.07)"
_DOT_ACTIVE = "#8A5CFF"      # Violet — shared carousel convention
_DOT_INACTIVE = "rgba(255,255,255,0.25)"


# ── Utilities ─────────────────────────────────────────────────────────────────

def _time_ago(created_utc) -> str:
    """Convert Unix UTC timestamp to human-readable relative time."""
    if not created_utc:
        return ""
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        dt = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
        s = int((now - dt).total_seconds())
        if s < 3600:
            return f"{max(0, s // 60)}m ago"
        if s < 86400:
            return f"{s // 3600}h ago"
        d = s // 86400
        if d < 30:
            return f"{d}d ago"
        return f"{d // 30}mo ago"
    except Exception:
        return ""


def _format_score(score: int) -> str:
    """Format vote score compactly (e.g. 12300 → 12.3k)."""
    if score >= 1000:
        return f"{score / 1000:.1f}k"
    return str(score)


# ── Slide rendering ───────────────────────────────────────────────────────────

def _render_slide(post: dict, visible: bool) -> str:
    """Render a single Reddit post as a carousel slide."""
    title = post.get("title") or ""
    subreddit = post.get("subreddit") or ""
    author = post.get("author") or ""
    score = post.get("score", 0)
    upvote_ratio = post.get("upvote_ratio", 0.0)
    num_comments = post.get("num_comments", 0)
    external_url = post.get("url") or ""
    permalink = post.get("permalink") or ""
    selftext = post.get("selftext") or ""
    created_utc = post.get("created_utc", 0)
    top_comments = post.get("top_comments") or []

    display = "flex" if visible else "none"
    time_str = _time_ago(created_utc)

    # Meta line: upvote score, upvote %, comment count, subreddit, author, age
    meta_parts = [
        f'<span style="color:{_ACCENT};font-weight:700;">{_UPVOTE_ICON} {_format_score(score)}</span>'
    ]
    if upvote_ratio:
        pct = int(upvote_ratio * 100)
        meta_parts.append(f'<span style="color:{_TEXT_TERTIARY};">{pct}%</span>')
    meta_parts.append(
        f'<span style="color:{_TEXT_TERTIARY};">{_COMMENT_ICON} {num_comments}</span>'
    )
    if subreddit:
        meta_parts.append(
            f'<span style="color:{_ACCENT};opacity:0.8;">r/{escape(subreddit)}</span>'
        )
    if author and author not in ("[deleted]", "AutoModerator"):
        meta_parts.append(f'<span style="color:{_TEXT_TERTIARY};">u/{escape(author)}</span>')
    if time_str:
        meta_parts.append(f'<span style="color:{_TEXT_TERTIARY};">{time_str}</span>')

    sep = f' <span style="color:rgba(234,230,242,0.2);">\u00b7</span> '
    meta_html = (
        f'<div style="display:flex;align-items:center;gap:4px;'
        f'font-size:11px;margin-bottom:6px;flex-wrap:wrap;">'
        + sep.join(meta_parts)
        + '</div>'
    )

    # Self-text preview
    text_html = ""
    if selftext:
        text_html = (
            f'<p style="font-size:13px;color:{_TEXT_SECONDARY};'
            f'line-height:1.55;margin:0 0 8px 0;">{escape(selftext)}</p>'
        )

    # Top comments
    comments_html = ""
    if top_comments:
        comment_items = ""
        for c in top_comments[:3]:
            body = c.get("body", "")
            c_author = c.get("author", "")
            c_score = c.get("score", 0)
            comment_items += (
                f'<div style="padding:7px 9px;background:rgba(255,255,255,0.03);'
                f'border-radius:6px;border-left:2px solid rgba(255,69,0,0.35);'
                f'margin-bottom:5px;">'
                f'<div style="font-size:10px;color:{_TEXT_TERTIARY};margin-bottom:3px;">'
                f'{escape(c_author)} \u00b7 \u25b2 {c_score}'
                f'</div>'
                f'<div style="font-size:12px;color:{_TEXT_SECONDARY};line-height:1.5;">'
                f'{escape(body)}'
                f'</div>'
                f'</div>'
            )
        comments_html = (
            f'<div style="margin-top:4px;margin-bottom:8px;">'
            + comment_items
            + '</div>'
        )

    # Links: external article (if link post) + Reddit thread
    links_html = '<div style="display:flex;gap:10px;flex-wrap:wrap;">'
    if external_url:
        links_html += (
            f'<a href="{escape(external_url)}" target="_blank" rel="noopener noreferrer" '
            f'style="display:inline-flex;align-items:center;gap:5px;'
            f'color:{_ACCENT};font-size:12px;text-decoration:none;opacity:0.85;">'
            + _LINK_ICON + '<span>Article</span></a>'
        )
    if permalink:
        link_color = _TEXT_SECONDARY if external_url else _ACCENT
        links_html += (
            f'<a href="{escape(permalink)}" target="_blank" rel="noopener noreferrer" '
            f'style="display:inline-flex;align-items:center;gap:5px;'
            f'color:{link_color};font-size:12px;text-decoration:none;opacity:0.85;">'
            + _LINK_ICON + '<span>Reddit thread</span></a>'
        )
    links_html += '</div>'

    return (
        f'<div data-slide '
        f'style="display:{display};flex-direction:column;'
        f'padding:13px 15px;background:{_SURFACE};'
        f'border-radius:9px;border:1px solid {_BORDER};">'
        + meta_html
        + f'<div style="font-weight:600;font-size:14px;color:{_TEXT_PRIMARY};'
          f'line-height:1.3;margin-bottom:6px;">{escape(title)}</div>'
        + text_html
        + comments_html
        + links_html
        + '</div>'
    )


# ── Navigation ────────────────────────────────────────────────────────────────

def _render_navigation(count: int) -> str:
    """Carousel nav buttons + dot indicators. Matches shared convention exactly."""
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
    """Assemble the full carousel card. Hard-capped at 8 slides."""
    results = results[:8]
    if not results:
        return (
            f'<p style="color:{_TEXT_TERTIARY};font-size:13px;'
            f'font-family:system-ui,-apple-system,sans-serif;padding:12px 14px;margin:0;">'
            f'No Reddit posts found.</p>'
        )

    slides = "".join(_render_slide(r, i == 0) for i, r in enumerate(results))
    nav = _render_navigation(len(results)) if len(results) > 1 else ""

    return (
        '<div data-carousel '
        'style="font-family:system-ui,-apple-system,sans-serif;">'
        + slides
        + nav
        + '</div>'
    )


# ── Text for LLM synthesis ────────────────────────────────────────────────────

def _format_text(results: list, query: str) -> str:
    """
    Structured text output — this is what the LLM receives for synthesis.
    Includes post content, scores, and top comments so the LLM can produce
    a balanced summary of community sentiment and advice.
    """
    if not results:
        return (
            f'No Reddit posts found for "{query}". '
            f'Try a different query, a different subreddit, or adjust the time filter.'
        )

    lines = [f'Reddit results for "{query}":']
    for i, r in enumerate(results, 1):
        lines.append(f"\n{i}. {r.get('title', '')}")
        subreddit = r.get("subreddit", "")
        author = r.get("author", "")
        score = r.get("score", 0)
        upvote_ratio = r.get("upvote_ratio", 0.0)
        num_comments = r.get("num_comments", 0)

        meta = f"   r/{subreddit}" if subreddit else ""
        if author and author != "[deleted]":
            meta += f" | u/{author}"
        meta += f" | {_format_score(score)} points"
        if upvote_ratio:
            meta += f" ({int(upvote_ratio * 100)}% upvoted)"
        meta += f" | {num_comments} comments"
        lines.append(meta)

        if selftext := r.get("selftext", ""):
            lines.append(f"   {selftext[:200]}")
        if url := r.get("url", ""):
            lines.append(f"   Link: {url}")
        if permalink := r.get("permalink", ""):
            lines.append(f"   Thread: {permalink}")

        top_comments = r.get("top_comments") or []
        if top_comments:
            lines.append("   Top comments:")
            for c in top_comments[:3]:
                body = c.get("body", "")
                c_author = c.get("author", "")
                c_score = c.get("score", 0)
                lines.append(f"     - {c_author} (+{c_score}): {body[:200]}")

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
        "query": result.get("query", ""),
        "text": _format_text(results, result.get("query", "")),
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
        "text": f"Reddit search failed: {str(_e)[:200]}",
        "html": None, "_meta": {},
    }), file=sys.stdout)
    print(f"[reddit runner] Unhandled exception: {_e}", file=sys.stderr)
