"""
Transformers — normalize provider responses into standard result format.

Each provider's raw response (JSON, Atom XML, RSS XML, gzip JSON) is
transformed into a list of:
    {"title": str, "snippet": str, "url": str, "provider": str, "date": str|None}

Provider-specific field mappings are declared as simple dicts. No classes,
no inheritance — just data.
"""

import gzip
import html
import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import cast
from urllib.parse import quote

logger = logging.getLogger(__name__)


# ── Field mappings per provider ──────────────────────────────────────────────

FIELD_MAPS = {
    'wikipedia': {
        'results_path': 'query.search',
        'title': 'title',
        'snippet': 'snippet',
        'url_template': 'https://en.wikipedia.org/wiki/{title}',
    },
    'wikidata': {
        'results_path': 'search',
        'title': 'label',
        'snippet': 'description',
        'url': 'concepturi',
    },
    'github': {
        'results_path': 'items',
        'title': 'full_name',
        'snippet': 'description',
        'url': 'html_url',
        'date': 'updated_at',
    },
    'hn_algolia': {
        'results_path': 'hits',
        'title': 'title',
        'snippet': 'story_text|comment_text',
        'url': 'url|story_url',
        'date': 'created_at',
        'url_fallback_template': 'https://news.ycombinator.com/item?id={objectID}',
    },
    'reddit': {
        'results_path': 'data.children',
        'unwrap_key': 'data',
        'title': 'title',
        'snippet': 'selftext',
        'url': 'url',
        'date': 'created_utc',
        'url_fallback_template': 'https://www.reddit.com{permalink}',
    },
    'stack_exchange': {
        'results_path': 'items',
        'title': 'title',
        'snippet': 'tags',  # No body in search results, tags give context
        'url': 'link',
        'date': 'creation_date',
    },
    'open_library': {
        'results_path': 'docs',
        'title': 'title',
        'snippet': 'author_name',
        'url_template': 'https://openlibrary.org{key}',
        'date': 'first_publish_year',
    },
    'musicbrainz': {
        'results_path': 'recordings',
        'title': 'title',
        'snippet': '_artist_credit',  # special handling
        'url_template': 'https://musicbrainz.org/recording/{id}',
    },
    'itunes': {
        'results_path': 'results',
        'title': 'trackName|collectionName',
        'snippet': 'shortDescription|longDescription|artistName',
        'url': 'trackViewUrl|collectionViewUrl',
        'date': 'releaseDate',
    },
    'nominatim': {
        'results_path': '',  # top-level array
        'title': 'display_name',
        'snippet': 'type',
        'url_template': 'https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=15/{lat}/{lon}',
    },
}


# ── Public API ───────────────────────────────────────────────────────────────

def transform(provider_name: str, response_format: str, raw_data: object, limit: int = 5) -> list[dict[str, object]]:
    """
    Transform a raw provider response into standardized results.

    Args:
        provider_name: Provider identifier (e.g. 'wikipedia')
        response_format: 'json', 'atom_xml', 'rss_xml', 'json_gzip'
        raw_data: Raw response — bytes for gzip/xml, parsed dict for json,
                  or str for xml
        limit: Max results to return

    Returns:
        List of {"title", "snippet", "url", "provider", "date"} dicts.
    """
    try:
        if response_format == 'atom_xml':
            return _transform_atom(provider_name, raw_data, limit)
        elif response_format == 'rss_xml':
            return _transform_rss(provider_name, raw_data, limit)
        elif response_format == 'json_gzip':
            # requests auto-decompresses gzip, so raw_data is already a dict
            if isinstance(raw_data, bytes):
                return _transform_json_gzip(provider_name, raw_data, limit)
            return _transform_json(provider_name, cast("dict[str, object]", raw_data), limit)
        else:  # json
            return _transform_json(provider_name, cast("dict[str, object]", raw_data), limit)
    except Exception as e:
        logger.warning(f'[SEARCH] transform failed for {provider_name}: {e}')
        return []


# ── JSON transform ───────────────────────────────────────────────────────────

def _transform_json(provider_name: str, data: dict[str, object], limit: int) -> list[dict[str, object]]:
    """Transform a JSON response using field mappings."""
    field_map = FIELD_MAPS.get(provider_name)
    if not field_map:
        logger.warning(f'[SEARCH] No field map for provider: {provider_name}')
        return []

    # Navigate to results array
    items = _resolve_path(data, field_map.get('results_path', ''))
    if not isinstance(items, list):
        return []

    results: list[dict[str, object]] = []
    for item in items[:limit]:
        # Some providers wrap items (e.g. Reddit: each child has a 'data' key)
        unwrap = field_map.get('unwrap_key')
        if unwrap and isinstance(item, dict):
            item = item.get(unwrap, item)
        item = cast("dict[str, object]", item)

        fm: dict[str, object] = cast("dict[str, object]", field_map)
        title = _extract_field(item, field_map.get('title', ''))
        snippet = _extract_snippet(item, field_map.get('snippet', ''))
        url = _extract_url(item, fm)
        date = _extract_date(item, field_map.get('date', ''))

        if not title and not snippet:
            continue

        results.append({
            'title': _clean_html(title or ''),
            'snippet': _clean_html(snippet or ''),
            'url': url or '',
            'provider': provider_name,
            'date': date,
        })

    return results


# ── Atom XML transform (ArXiv) ───────────────────────────────────────────────

_ATOM_NS = {'atom': 'http://www.w3.org/2005/Atom'}


def _transform_atom(provider_name: str, raw_data: object, limit: int) -> list[dict[str, object]]:
    """Transform Atom XML (ArXiv) into standard results."""
    if isinstance(raw_data, bytes):
        raw_data = raw_data.decode('utf-8', errors='replace')

    root = ET.fromstring(cast("str", raw_data))
    entries = root.findall('atom:entry', _ATOM_NS)

    results: list[dict[str, object]] = []
    for entry in entries[:limit]:
        title = _xml_text(entry, 'atom:title', _ATOM_NS)
        summary = _xml_text(entry, 'atom:summary', _ATOM_NS)
        # ArXiv: id element contains the URL
        url = _xml_text(entry, 'atom:id', _ATOM_NS)
        published = _xml_text(entry, 'atom:published', _ATOM_NS)

        if not title:
            continue

        results.append({
            'title': _clean_whitespace(title),
            'snippet': _clean_whitespace(summary or ''),
            'url': url or '',
            'provider': provider_name,
            'date': published[:10] if published else None,
        })

    return results


# ── RSS XML transform (Google News) ──────────────────────────────────────────

def _transform_rss(provider_name: str, raw_data: object, limit: int) -> list[dict[str, object]]:
    """Transform RSS XML (Google News) into standard results."""
    if isinstance(raw_data, bytes):
        raw_data = raw_data.decode('utf-8', errors='replace')

    root = ET.fromstring(cast("str", raw_data))
    items = root.findall('.//item')

    results: list[dict[str, object]] = []
    for item in items[:limit]:
        title = _xml_text_plain(item, 'title')
        link = _xml_text_plain(item, 'link')
        pub_date = _xml_text_plain(item, 'pubDate')
        source_el = item.find('source')
        source = source_el.text if source_el is not None else None

        if not title:
            continue

        # Google News titles often include " - Source Name"
        snippet = f'Source: {source}' if source else ''

        results.append({
            'title': title.strip(),
            'snippet': snippet,
            'url': link or '',
            'provider': provider_name,
            'date': _parse_rss_date(pub_date) if pub_date else None,
        })

    return results


# ── Gzip JSON transform (Stack Exchange) ─────────────────────────────────────

def _transform_json_gzip(provider_name: str, raw_data: bytes, limit: int) -> list[dict[str, object]]:
    """Decompress gzip, parse JSON, delegate to JSON transform."""
    try:
        decompressed = gzip.decompress(raw_data)
        data = cast("dict[str, object]", json.loads(decompressed))
    except Exception as e:
        logger.warning(f'[SEARCH] gzip decompress failed for {provider_name}: {e}')
        return []

    return _transform_json(provider_name, data, limit)


# ── Field extraction helpers ─────────────────────────────────────────────────

def _resolve_path(data: object, path: str) -> object:
    """Navigate a dotted path like 'query.search' into a nested dict."""
    if not path:
        return data
    for key in path.split('.'):
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return None
    return data


def _extract_field(item: dict[str, object], field_spec: str) -> str:
    """
    Extract a field value using a field spec.

    Supports:
    - Simple: 'title' → item['title']
    - Fallback chain: 'url|story_url' → item['url'] or item['story_url']
    """
    if not field_spec:
        return ''
    for field in field_spec.split('|'):
        field = field.strip()
        value = item.get(field)
        if value is not None:
            if isinstance(value, list):
                # e.g. Stack Exchange tags
                return ', '.join(str(v) for v in value[:5])
            return str(value)
    return ''


def _extract_snippet(item: dict[str, object], snippet_spec: str) -> str:
    """Extract snippet with provider-specific special handling."""

    # Special: MusicBrainz artist credit
    if snippet_spec == '_artist_credit':
        credits = item.get('artist-credit', [])
        if credits:
            artists = [cast("dict[str, object]", c).get('name', '') for c in cast("list[object]", credits) if isinstance(c, dict)]
            return ', '.join(str(a) for a in artists if a)
        return ''

    return _extract_field(item, snippet_spec)


def _extract_url(item: dict[str, object], field_map: dict[str, object]) -> str:
    """Extract URL, falling back to template if needed."""
    # Try direct field first
    url_spec = field_map.get('url', '')
    if url_spec:
        url = _extract_field(item, cast("str", url_spec))
        if url:
            return url

    # Try fallback template (e.g. HN: construct URL from objectID)
    fallback = cast("str", field_map.get('url_fallback_template', ''))
    if fallback:
        try:
            return fallback.format(**item)
        except (KeyError, ValueError):
            pass

    # Try url_template
    template = cast("str", field_map.get('url_template', ''))
    if template:
        try:
            # URL-encode title for Wikipedia-style URLs
            formatted = template.format(**{
                k: quote(str(v), safe='') if k == 'title' else str(v)
                for k, v in item.items()
                if v is not None
            })
            return formatted
        except (KeyError, ValueError):
            pass

    return ''



def _extract_date(item: dict[str, object], date_spec: str) -> str | None:
    """Extract and normalize date from item."""
    if not date_spec:
        return None

    raw = _extract_field(item, date_spec)
    if not raw:
        return None

    # Unix timestamp (Reddit)
    try:
        ts = float(raw)
        if ts > 1_000_000_000:
            from datetime import datetime, timezone
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
    except (ValueError, TypeError, OverflowError):
        pass

    # Year only (Open Library)
    if re.match(r'^\d{4}$', raw):
        return raw

    # ISO date — take first 10 chars
    if len(raw) >= 10 and raw[4] == '-':
        return raw[:10]

    return None


# ── Text cleaning ────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r'<[^>]+>')
_WHITESPACE_RE = re.compile(r'\s{2,}')


def _clean_html(text: str) -> str:
    """Strip HTML tags and decode entities."""
    text = _HTML_TAG_RE.sub('', text)
    text = html.unescape(text)
    return _WHITESPACE_RE.sub(' ', text).strip()


def _clean_whitespace(text: str) -> str:
    """Collapse whitespace runs."""
    return _WHITESPACE_RE.sub(' ', text).strip()



def _xml_text(element: object, tag: str, namespaces: dict[str, str]) -> str | None:
    """Get text content of an XML sub-element with namespace."""
    el = cast("ET.Element", element).find(tag, namespaces)
    return el.text.strip() if el is not None and el.text else None


def _xml_text_plain(element: object, tag: str) -> str | None:
    """Get text content of an XML sub-element without namespace."""
    el = cast("ET.Element", element).find(tag)
    return el.text.strip() if el is not None and el.text else None


def _parse_rss_date(date_str: str) -> str | None:
    """Parse RSS date format (RFC 822) to ISO date."""
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(date_str)
        return dt.strftime('%Y-%m-%d')
    except Exception:
        return None
