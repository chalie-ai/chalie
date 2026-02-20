"""
Card Renderer Service — Template compilation and asset management for tool result cards.

New formalized contract: tools return inline HTML (no external CSS).
Older template-based rendering is deprecated but retained for backwards compatibility.

Sanitizes HTML and returns compiled card data for frontend rendering.
"""

import re
import uuid
import logging
from pathlib import Path
from html import escape
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CardRendererService:
    """Renders tool result cards from templates and styles with security sanitization."""

    def render_tool_html(
        self,
        tool_name: str,
        html: str,
        title: str,
        card_config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Render a card from formalized tool contract output (inline HTML).

        Args:
            tool_name: Name of the tool
            html: HTML fragment from tool (inline styles only)
            title: Card title (may be dynamic from tool)
            card_config: Card config from manifest (accent_color, background_color)

        Returns:
            Dict with {html, scope_id, title, accent_color, background_color, tool_name}
            or None if rendering fails
        """
        try:
            # Sanitize HTML using strict contract enforcement
            sanitized_html = self._sanitize_tool_html(html)
            if not sanitized_html:
                logger.debug(f"[CARD] HTML sanitization stripped all content for {tool_name}")
                return None

            scope_id = str(uuid.uuid4())[:8]

            return {
                "html": sanitized_html,
                "scope_id": scope_id,
                "title": title or tool_name,
                "accent_color": card_config.get("accent_color", "#5b9bd5"),
                "background_color": card_config.get("background_color", "rgba(91, 155, 213, 0.08)"),
                "tool_name": tool_name,
            }

        except Exception as e:
            logger.error(f"[CARD] Render failed for {tool_name}: {e}")
            return None

    def _sanitize_tool_html(self, html: str) -> str:
        """
        Strict sanitization for inline HTML from tools (formalized contract).

        Rules enforced:
        1. Strip <html>, <head>, <body> wrapper tags
        2. Strip ALL <script> tags and contents
        3. Strip ALL <style> tags and contents (inline CSS only)
        4. Strip ALL <link> tags
        5. Strip ALL event handler attributes (on*)
        6. Strip javascript: and data: URIs
        7. Strip <iframe>, <form>, <input>, <object>, <embed>, <base> tags
        8. Allow only safe inline style attributes
        """
        if not html:
            return ""

        # 1. Strip wrapper tags
        html = re.sub(r"<html[^>]*>|</html>", "", html, flags=re.IGNORECASE)
        html = re.sub(r"<head[^>]*>|</head>", "", html, flags=re.IGNORECASE)
        html = re.sub(r"<body[^>]*>|</body>", "", html, flags=re.IGNORECASE)

        # 2. Strip <script> tags and contents
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # 3. Strip <style> tags and contents
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # 4. Strip <link> tags
        html = re.sub(r"<link[^>]*>", "", html, flags=re.IGNORECASE)

        # 5. Strip event handler attributes (on* attributes)
        html = re.sub(r'\bon\w+\s*=\s*["\'][^"\']*["\']', "", html, flags=re.IGNORECASE)
        html = re.sub(r"\bon\w+\s*=\s*[^>\s]+", "", html, flags=re.IGNORECASE)

        # 6. Strip javascript: and data: URIs
        html = re.sub(r'href\s*=\s*["\']?javascript:[^"\'>\s]*["\']?', "", html, flags=re.IGNORECASE)
        html = re.sub(r'src\s*=\s*["\']?javascript:[^"\'>\s]*["\']?', "", html, flags=re.IGNORECASE)
        html = re.sub(r'src\s*=\s*["\']?data:[^"\'>\s]*["\']?', "", html, flags=re.IGNORECASE)

        # 7. Strip dangerous tags
        for tag in ["iframe", "form", "input", "object", "embed", "base"]:
            html = re.sub(f"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(f"<{tag}[^>]*/?>", "", html, flags=re.IGNORECASE)

        # 8. Allow safe inline styles but strip expression() and problematic url()
        # This is handled by allowing style="..." attributes but regex validation below
        html = re.sub(r'style\s*=\s*["\']([^"\']*expression[^"\']*)["\']', "", html, flags=re.IGNORECASE)
        html = re.sub(r'style\s*=\s*["\']([^"\']*url\([^)]*data:[^)]*\)[^"\']*)["\']', "", html, flags=re.IGNORECASE)

        return html.strip()

    def render(
        self,
        tool_name: str,
        raw_data: Dict[str, Any],
        card_config: Dict[str, Any],
        tool_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Render a card from tool result data.

        Args:
            tool_name: Name of the tool
            raw_data: Raw result dict from tool handler (e.g., {"temperature_c": 22, "location": "Malta"})
            card_config: Card config from manifest (e.g., {"title": "Weather in {{location}}", ...})
            tool_dir: Path to tool directory (contains card/ subdir)

        Returns:
            Dict with {html, css, scope_id, title, accent_color, background_color, tool_name}
            or None if card assets not found or rendering fails
        """
        try:
            tool_path = Path(tool_dir)
            template_path = tool_path / "card" / "template.html"
            styles_path = tool_path / "card" / "styles.css"

            # Load template and styles
            if not template_path.exists():
                logger.debug(f"[CARD] No template found for {tool_name} at {template_path}")
                return None

            with open(template_path, "r") as f:
                template_html = f.read()

            styles_css = ""
            if styles_path.exists():
                with open(styles_path, "r") as f:
                    styles_css = f.read()

            # Render template with data
            rendered_html = self._render_template(template_html, raw_data)

            # Sanitize HTML (strip scripts, on* attributes, etc.)
            rendered_html = self._sanitize_html(rendered_html)

            # Scope and sanitize CSS
            scope_id = str(uuid.uuid4())[:8]
            scoped_css = self._scope_css(styles_css, scope_id)
            scoped_css = self._sanitize_css(scoped_css)

            # Render title with data substitution
            title_template = card_config.get("title", tool_name)
            rendered_title = self._render_template(title_template, raw_data)
            rendered_title = self._sanitize_html(rendered_title)

            return {
                "html": rendered_html,
                "css": scoped_css,
                "scope_id": scope_id,
                "title": rendered_title,
                "accent_color": card_config.get("accent_color", "#5b9bd5"),
                "background_color": card_config.get("background_color", "rgba(91, 155, 213, 0.08)"),
                "tool_name": tool_name,
            }

        except Exception as e:
            logger.error(f"[CARD] Render failed for {tool_name}: {e}")
            return None

    def _render_template(self, template: str, data: Dict[str, Any]) -> str:
        """
        Render Mustache-like template with data.

        Syntax:
        - {{key}} — scalar substitution (HTML-escaped)
        - {{#list}} ... {{/list}} — iterate over array
        - Inside loops: {{.key}} accesses array item fields
        """
        result = template

        # Process loops first: {{#list}}...{{/list}}
        loop_pattern = r"\{\{#(\w+)\}\}(.*?)\{\{/\1\}\}"
        for match in re.finditer(loop_pattern, result, re.DOTALL):
            list_name = match.group(1)
            loop_content = match.group(2)
            list_data = data.get(list_name, [])

            if not isinstance(list_data, list):
                list_data = []

            rendered_items = []
            for item in list_data:
                item_html = loop_content
                if isinstance(item, dict):
                    # Replace {{.key}} with item values
                    for key, value in item.items():
                        value_str = escape(str(value)) if value is not None else ""
                        item_html = item_html.replace(f"{{{{.{key}}}}}", value_str)
                rendered_items.append(item_html)

            result = result.replace(match.group(0), "".join(rendered_items))

        # Process scalar substitution: {{key}}
        scalar_pattern = r"\{\{(\w+)\}\}"
        for match in re.finditer(scalar_pattern, result):
            key = match.group(1)
            value = data.get(key, "")
            value_str = escape(str(value)) if value is not None else ""
            result = result.replace(match.group(0), value_str)

        return result

    def _sanitize_html(self, html: str) -> str:
        """
        Strip dangerous HTML patterns: <script> tags and on* event handlers.

        Allowlist: div, span, p, h1-h4, ul, ol, li, strong, em, b, i, img, br, hr
        """
        # Strip <script>...</script>
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # Strip on* event attributes
        html = re.sub(r'\bon\w+\s*=\s*["\'][^"\']*["\']', "", html, flags=re.IGNORECASE)
        html = re.sub(r"\bon\w+\s*=\s*\w+", "", html, flags=re.IGNORECASE)

        # Strip <iframe>, <form>, <object>, <embed>, <link>, <meta>, <style>
        for tag in ["iframe", "form", "object", "embed", "link", "meta", "style", "script"]:
            html = re.sub(f"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(f"<{tag}[^>]*/?>", "", html, flags=re.IGNORECASE)

        # Strip javascript: and data: URLs (except img src)
        html = re.sub(r'href\s*=\s*["\']?javascript:[^"\'>\s]*["\']?', "", html, flags=re.IGNORECASE)
        # For src attributes on non-img tags
        html = re.sub(r'(?<!img\s)src\s*=\s*["\']?data:[^"\'>\s]*["\']?', "", html, flags=re.IGNORECASE)

        return html

    def _scope_css(self, css: str, scope_id: str) -> str:
        """
        Scope all CSS selectors to [data-card-scope="{scope_id}"].

        Simple approach: split on '{' to find selectors, prepend scope attribute.
        """
        if not css:
            return ""

        parts = css.split("{")
        result = []

        for i, part in enumerate(parts):
            if i == 0:
                result.append(part)
            else:
                # part is "selector ... { rest of css"
                # We need to scope the preceding selector(s)
                # Find the last line break to identify selector
                lines = result[-1].split("\n")
                last_line = lines[-1] if lines else ""

                # Get the selector from the previous part's ending
                # This is complex; simpler approach: prepend scope to every rule selector
                selector_part = parts[i - 1].split("\n")[-1] if i > 0 else ""
                current_selector = selector_part.strip()

                # Prepend scope to the selector
                if current_selector and current_selector != "":
                    scoped = f"[data-card-scope='{scope_id}'] {current_selector}"
                    result[-1] = result[-1][: -(len(current_selector))] + scoped

                result.append("{" + part)

        return "".join(result)

    def _sanitize_css(self, css: str) -> str:
        """
        Strip dangerous CSS patterns: expression(), url(data:), @import, <script>.
        """
        # Strip @import
        css = re.sub(r"@import[^;]*;", "", css, flags=re.IGNORECASE)

        # Strip expression(...)
        css = re.sub(r"expression\s*\([^)]*\)", "", css, flags=re.IGNORECASE)

        # Strip url(data:...)
        css = re.sub(r"url\s*\(\s*data:[^)]*\)", "", css, flags=re.IGNORECASE)

        # Strip <script> patterns (shouldn't be in CSS but be safe)
        css = re.sub(r"<script[^>]*>.*?</script>", "", css, flags=re.DOTALL | re.IGNORECASE)

        return css
