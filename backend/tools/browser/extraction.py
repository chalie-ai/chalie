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
    # Capture raw text BEFORE noise removal — noise removal mutates the DOM
    # and can accidentally nuke real content on JS-heavy / e-commerce sites
    raw_text = _inner_text(page, selector)

    # Remove noise elements from the DOM (destructive — elements are deleted)
    try:
        page.evaluate("""(sel) => {
            document.querySelectorAll(sel).forEach(el => el.remove());
        }""", _NOISE_SELECTOR)
    except Exception:
        pass  # Page might block evaluate — proceed with noisy DOM

    clean_text = _inner_text(page, selector)

    # If noise removal stripped too much, fall back to the raw extraction.
    # Threshold: cleaned text < 200 chars AND raw text had at least 2× more.
    if len(clean_text.strip()) < 200 and len(raw_text.strip()) > len(clean_text.strip()) * 2:
        logger.debug("[BROWSER EXTRACT] Noise removal too aggressive (%d→%d chars), using raw text",
                     len(raw_text.strip()), len(clean_text.strip()))
        text = raw_text
    else:
        text = clean_text

    # Collapse excessive whitespace
    text = _WHITESPACE_RE.sub("\n\n", text.strip())

    if max_chars and len(text) > max_chars:
        text = text[:max_chars]

    return text


def _inner_text(page, selector: str | None) -> str:
    """Get inner_text from page or scoped selector, with fallback."""
    try:
        if selector:
            el = page.query_selector(selector)
            if el:
                return el.inner_text()
            return page.inner_text("body")
        return page.inner_text("body")
    except Exception as e:
        logger.warning("[BROWSER EXTRACT] inner_text failed: %s", e)
        try:
            return page.text_content("body") or ""
        except Exception:
            return ""


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


def _query_dom_links(page) -> list:
    """Query all anchor hrefs and text from the live DOM via Playwright."""
    try:
        return page.evaluate("""() => {
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


def _is_valid_link(url: str, text: str, base_url: str, seen: set) -> bool:
    """Return True if the link should be included in the result set."""
    if not url or not text or url in seen:
        return False
    if not url.startswith(("http://", "https://")):
        return False
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if any(domain == d or domain.endswith('.' + d) for d in _SKIP_DOMAINS):
            return False
        if _SKIP_PATH_RE.search(parsed.path):
            return False
        base_parsed = urlparse(base_url)
        if parsed.netloc == base_parsed.netloc and parsed.path == base_parsed.path:
            return False
    except Exception:
        return False
    return True


def extract_links(page, base_url: str, max_links: int = 15) -> list[dict]:
    """Extract navigable page links from rendered DOM.

    Returns list of {"text": str, "url": str} dicts.
    Filters noise (social media, login/privacy pages, fragments).
    """
    raw_links = _query_dom_links(page)

    result = []
    seen: set = set()
    for link in raw_links:
        url = link.get("url", "")
        text = link.get("text", "").strip()
        if not _is_valid_link(url, text, base_url, seen):
            continue
        seen.add(url)
        result.append({"text": text, "url": url})
        if len(result) >= max_links:
            break

    return result
