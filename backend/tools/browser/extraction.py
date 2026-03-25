"""
Browser Content Extraction — Clean text from rendered DOM.

Extracts text, HTML, or navigable links from a Playwright page after
JavaScript has fully rendered.  Noise stripping mirrors the read skill's
BS4 pipeline but operates on live DOM via Playwright selectors.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Elements to remove before extraction (noise)
_NOISE_SELECTORS = [
    "script", "style", "noscript", "iframe",
    "nav", "footer", "header",
    "[role='navigation']", "[role='banner']", "[role='contentinfo']",
    "[class*='cookie']", "[class*='Cookie']",
    "[class*='consent']", "[class*='Consent']",
    "[class*='popup']", "[class*='Popup']",
    "[class*='modal']", "[class*='Modal']",
    "[class*='overlay']", "[class*='Overlay']",
    "[class*='ad-']", "[class*='advertisement']",
    "[id*='cookie']", "[id*='consent']",
    "[aria-hidden='true']",
]

# Combined selector for noise removal (joined once)
_NOISE_SELECTOR = ", ".join(_NOISE_SELECTORS)

# Link domains to skip (same as read skill)
_SKIP_DOMAINS = frozenset((
    'facebook.com', 'twitter.com', 'x.com', 'instagram.com', 'linkedin.com',
    'pinterest.com', 'tiktok.com', 'youtube.com', 'reddit.com',
))

_SKIP_PATH_RE = re.compile(
    r'/(login|signin|signup|register|logout|privacy|terms|cookie|legal|contact'
    r'|about-us|careers|advertise|press|help/?)(\b|$)',
    re.IGNORECASE,
)

_WHITESPACE_RE = re.compile(r'\n{3,}')


def extract_text(page, selector: str | None = None, max_chars: int = 8000) -> str:
    """Extract clean text from a rendered page.

    Args:
        page: Playwright Page (already navigated).
        selector: Optional CSS selector to scope extraction.
        max_chars: Maximum characters to return.

    Returns:
        Clean text string.
    """
    try:
        # Remove noise elements from the DOM (in-page mutation — does not affect
        # the actual site, only our ephemeral context)
        page.evaluate("""(sel) => {
            document.querySelectorAll(sel).forEach(el => el.remove());
        }""", _NOISE_SELECTOR)
    except Exception:
        pass  # Page might block evaluate — proceed with noisy DOM

    try:
        if selector:
            # Scope to specific element
            el = page.query_selector(selector)
            if el:
                text = el.inner_text()
            else:
                text = page.inner_text("body")
        else:
            text = page.inner_text("body")
    except Exception as e:
        logger.warning("[BROWSER EXTRACT] inner_text failed: %s", e)
        try:
            text = page.text_content("body") or ""
        except Exception:
            text = ""

    # Collapse excessive whitespace
    text = _WHITESPACE_RE.sub("\n\n", text.strip())

    if max_chars and len(text) > max_chars:
        text = text[:max_chars]

    return text


def extract_html(page, selector: str | None = None, max_chars: int = 8000) -> str:
    """Extract inner HTML from a rendered page.

    Args:
        page: Playwright Page (already navigated).
        selector: Optional CSS selector to scope extraction.
        max_chars: Maximum characters to return.

    Returns:
        HTML string.
    """
    try:
        if selector:
            el = page.query_selector(selector)
            if el:
                html = el.inner_html()
            else:
                html = page.inner_html("body")
        else:
            html = page.inner_html("body")
    except Exception:
        html = ""

    if max_chars and len(html) > max_chars:
        html = html[:max_chars]

    return html


def extract_links(page, base_url: str, max_links: int = 15) -> list[dict]:
    """Extract navigable page links from rendered DOM.

    Returns list of {"text": str, "url": str} dicts.
    Filters noise (social media, login/privacy pages, fragments).
    """
    try:
        raw_links = page.evaluate("""() => {
            const links = [];
            const seen = new Set();
            for (const a of document.querySelectorAll('a[href]')) {
                const href = a.href;
                const text = (a.innerText || a.textContent || '').trim();
                if (!href || !text || href.startsWith('javascript:') || href.startsWith('#')) continue;
                if (seen.has(href)) continue;
                seen.add(href);
                links.push({url: href, text: text.slice(0, 120)});
            }
            return links;
        }""")
    except Exception:
        return []

    result = []
    seen = set()
    for link in raw_links:
        url = link.get("url", "")
        text = link.get("text", "").strip()
        if not url or not text or url in seen:
            continue

        # Filter schemes
        if not url.startswith(("http://", "https://")):
            continue

        # Filter social media domains
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if any(domain == d or domain.endswith('.' + d) for d in _SKIP_DOMAINS):
                continue
            if _SKIP_PATH_RE.search(parsed.path):
                continue
            # Skip same-page links
            base_parsed = urlparse(base_url)
            if parsed.netloc == base_parsed.netloc and parsed.path == base_parsed.path:
                continue
        except Exception:
            continue

        seen.add(url)
        result.append({"text": text, "url": url})
        if len(result) >= max_links:
            break

    return result
