"""
ProgrammingDocsSearchAbility — Official documentation lookup.

Searches and reads official documentation for 12 languages and 11 major
frameworks. Each language source defines search(query) and a base_url.
The lookup() function resolves the language, searches, fetches the top
result, extracts clean text, and returns a tool result dict.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from abilities._base import Ability

# ---------------------------------------------------------------------------
# Content extraction constants
# ---------------------------------------------------------------------------

_STRIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "svg"}
_CONTENT_SELECTORS = ["main", "article"]
_CONTENT_IDS = ["layout-content", "content", "main-content", "bodyContent", "mw-content-text"]
_CONTENT_CLASSES = ["refentry", "mw-content-ltr", "content", "documentation", "doc-content", "markdown-body"]
_MAX_CHARS = 12_000
_TIMEOUT = 8
_RE_HTML_TAGS = r"<[^>]+>"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ChalieBot/1.0; +https://chalie.ai)",
    "Accept": "text/html,application/xhtml+xml,application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip_depth = 0
        self._tag_stack = []

    def handle_starttag(self, tag, _attrs):
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag in _STRIP_TAGS:
            self._skip_depth += 1
        if tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                    "li", "tr", "dt", "dd", "pre", "blockquote", "section"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _STRIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                    "li", "pre", "blockquote", "section", "table"):
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)


def _fetch(url):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def _head_ok(url, timeout=5):
    try:
        req = urllib.request.Request(url, method="HEAD", headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 400
    except Exception:
        return False


def _extract_text(html):
    body = html
    for cid in _CONTENT_IDS:
        m = re.search(
            rf'<(?:div|main|article|section)[^>]*\bid=["\']?{cid}["\']?[^>]*>(.*)',
            html, re.DOTALL | re.IGNORECASE,
        )
        if m:
            body = m.group(1)
            break
    else:
        for tag in _CONTENT_SELECTORS:
            m = re.search(rf"<{tag}[^>]*>(.*)</{tag}>", html, re.DOTALL | re.IGNORECASE)
            if m:
                body = m.group(1)
                break
        else:
            for cls in _CONTENT_CLASSES:
                m = re.search(
                    rf'<(?:div|main|article|section)[^>]*\bclass=["\'][^"\']*\b{cls}\b[^"\']*["\'][^>]*>(.*)',
                    html, re.DOTALL | re.IGNORECASE,
                )
                if m:
                    body = m.group(1)
                    break

    extractor = _TextExtractor()
    extractor.feed(body)
    text = "".join(extractor.parts)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if len(text) > _MAX_CHARS:
        cut = text.rfind("\n\n", 0, _MAX_CHARS)
        if cut > _MAX_CHARS // 2:
            text = text[:cut]
        else:
            text = text[:_MAX_CHARS]

    return text


def _fetch_and_extract(url):
    try:
        _, html = _fetch(url)
        text = _extract_text(html)
        return text if text else f"Page fetched but no content extracted: {url}"
    except Exception as e:
        return f"Failed to fetch {url}: {e}"


def _parse_links(html, base_url):
    results = []
    for m in re.finditer(r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                         html, re.DOTALL | re.IGNORECASE):
        href, title = m.group(1), re.sub(_RE_HTML_TAGS, "", m.group(2)).strip()
        if not title:
            continue
        if href.startswith("/"):
            href = urllib.parse.urljoin(base_url, href)
        elif not href.startswith("http"):
            continue
        results.append({"url": href, "title": title})
    return results


def _sphinx_search(base_url, query, max_results=5):
    try:
        url = f"{base_url.rstrip('/')}/search/?q={urllib.parse.quote(query)}"
        _, html = _fetch(url)
        results = []
        for m in re.finditer(
            r'<li[^>]*>\s*<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
            html, re.DOTALL | re.IGNORECASE,
        ):
            href, title = m.group(1), re.sub(_RE_HTML_TAGS, "", m.group(2)).strip()
            if not title or "search" in href.lower():
                continue
            if href.startswith("http"):
                full_url = href
            elif href.startswith("/"):
                full_url = urllib.parse.urljoin(base_url, href)
            else:
                full_url = urllib.parse.urljoin(url, href)
            results.append({"url": full_url, "title": title})
            if len(results) >= max_results:
                break
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Language sources
# ---------------------------------------------------------------------------

class _Source:
    name = ""
    id = ""
    aliases = []
    base_url = ""

    def search(self, _query, _max_results=5):
        return []


class PHPSource(_Source):
    name = "PHP"
    id = "php"
    aliases = ["php", "php8"]
    base_url = "https://www.php.net/manual/en/"

    def search(self, query, max_results=5):
        q = query.strip().lower().replace(" ", "_")
        candidates = []
        patterns = [
            f"https://www.php.net/manual/en/function.{q.replace('_', '-')}.php",
            f"https://www.php.net/manual/en/class.{q}.php",
        ]
        if "::" in query:
            cls, method = query.split("::", 1)
            patterns.insert(0, f"https://www.php.net/manual/en/{cls.strip().lower()}.{method.strip().lower()}.php")
        for url in patterns:
            if _head_ok(url):
                return [{"url": url, "title": query}]
        try:
            search_url = f"https://www.php.net/manual-lookup.php?pattern={urllib.parse.quote(query)}&scope=quickref"
            _, html = _fetch(search_url)
            links = _parse_links(html, "https://www.php.net")
            for link in links[:max_results]:
                if "/manual/en/" in link["url"]:
                    candidates.append(link)
            return candidates[:max_results]
        except Exception:
            return candidates


class LaravelSource(_Source):
    name = "Laravel"
    id = "laravel"
    aliases = ["laravel"]
    base_url = "https://laravel.com/docs/11.x"

    def search(self, query, max_results=5):
        q = query.strip().lower().replace(" ", "-")
        results = []
        for slug in [q, q.replace("_", "-"), q.split(".")[0], q.split("::")[0]]:
            url = f"https://laravel.com/docs/11.x/{slug}"
            if _head_ok(url):
                results.append({"url": url, "title": f"Laravel: {query}"})
                return results
        for section in ["eloquent", "blade", "routing", "middleware",
                        "controllers", "requests", "responses", "views",
                        "validation", "collections", "queues", "events",
                        "authentication", "authorization", "migrations"]:
            if q in section or section in q:
                url = f"https://laravel.com/docs/11.x/{section}"
                results.append({"url": url, "title": f"Laravel: {section}"})
                return results
        class_name = query.strip().split("\\")[-1].split("::")[-1]
        if class_name[0:1].isupper():
            results.append({"url": "https://laravel.com/api/11.x/Illuminate.html", "title": f"Laravel API: {class_name}"})
        return results[:max_results]


class PythonSource(_Source):
    name = "Python"
    id = "python"
    aliases = ["python", "py", "python3"]
    base_url = "https://docs.python.org/3/"

    def search(self, query, max_results=5):
        q = query.strip().lower()
        candidates = []
        module = q.split(".")[0].replace(" ", "").replace("-", "_")
        if _head_ok(f"https://docs.python.org/3/library/{module}.html"):
            candidates.append({"url": f"https://docs.python.org/3/library/{module}.html", "title": f"{module} — Python docs"})
        if not candidates:
            try:
                idx_url = f"https://docs.python.org/3/genindex-{q[0].upper()}.html" if q[0].isalpha() else None
                if idx_url:
                    _, html = _fetch(idx_url)
                    links = _parse_links(html, "https://docs.python.org/3/")
                    for link in links:
                        if q in link["title"].lower() or q in link["url"].lower():
                            candidates.append(link)
                            if len(candidates) >= max_results:
                                break
            except Exception:
                pass
        if not candidates:
            candidates.append({"url": "https://docs.python.org/3/library/functions.html", "title": "Built-in Functions — Python docs"})
        return candidates[:max_results]


class DjangoSource(_Source):
    name = "Django"
    id = "django"
    aliases = ["django"]
    base_url = "https://docs.djangoproject.com/en/5.0/"

    def search(self, query, max_results=5):
        try:
            url = f"https://docs.djangoproject.com/en/5.0/search/?q={urllib.parse.quote(query)}"
            _, html = _fetch(url)
            results = []
            for m in re.finditer(
                r'<dt>\s*<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                html, re.DOTALL | re.IGNORECASE,
            ):
                href, title = m.group(1), re.sub(_RE_HTML_TAGS, "", m.group(2)).strip()
                if title:
                    full_url = urllib.parse.urljoin("https://docs.djangoproject.com", href)
                    results.append({"url": full_url, "title": title})
                    if len(results) >= max_results:
                        break
            return results[:max_results]
        except Exception:
            q = query.strip().lower().replace(" ", "-")
            return [{"url": f"https://docs.djangoproject.com/en/5.0/ref/{q}/", "title": f"Django: {query}"}]


class FlaskSource(_Source):
    name = "Flask"
    id = "flask"
    aliases = ["flask"]
    base_url = "https://flask.palletsprojects.com/en/3.0.x/"

    def search(self, query, max_results=5):
        results = _sphinx_search(self.base_url, query, max_results)
        if results:
            return results
        return [{"url": f"{self.base_url}api/", "title": "Flask API Reference"}]


class NumPySource(_Source):
    name = "NumPy"
    id = "numpy"
    aliases = ["numpy", "np"]
    base_url = "https://numpy.org/doc/stable/"

    def search(self, query, max_results=5):
        q = query.strip().lower().replace(" ", ".")
        url = f"https://numpy.org/doc/stable/reference/generated/numpy.{q}.html"
        if _head_ok(url):
            return [{"url": url, "title": f"numpy.{q}"}]
        url2 = f"https://numpy.org/doc/stable/reference/generated/numpy.{q.replace('numpy.', '')}.html"
        if url2 != url and _head_ok(url2):
            return [{"url": url2, "title": f"numpy.{q.replace('numpy.', '')}"}]
        topic = q.replace(".", "-").replace("_", "-")
        for section in ["reference/routines", "reference/arrays", "user"]:
            url = f"https://numpy.org/doc/stable/{section}.{topic}.html"
            if _head_ok(url):
                return [{"url": url, "title": f"NumPy: {query}"}]
        return [{"url": "https://numpy.org/doc/stable/reference/index.html", "title": "NumPy Reference"}]


class PandasSource(_Source):
    name = "Pandas"
    id = "pandas"
    aliases = ["pandas", "pd"]
    base_url = "https://pandas.pydata.org/docs/"

    def search(self, query, max_results=5):
        q = query.strip()
        for prefix in ["pandas.", ""]:
            name = q if q.startswith("pandas.") else f"{prefix}{q}"
            url = f"https://pandas.pydata.org/docs/reference/api/{name}.html"
            if _head_ok(url):
                return [{"url": url, "title": name}]
        for cls in ["DataFrame", "Series", "Index"]:
            url = f"https://pandas.pydata.org/docs/reference/api/pandas.{cls}.{q}.html"
            if _head_ok(url):
                return [{"url": url, "title": f"pandas.{cls}.{q}"}]
        return [{"url": "https://pandas.pydata.org/docs/reference/index.html", "title": "Pandas API Reference"}]


class JavaScriptSource(_Source):
    name = "JavaScript (MDN)"
    id = "javascript"
    aliases = ["javascript", "js", "typescript", "ts", "mdn"]
    base_url = "https://developer.mozilla.org/en-US/docs/Web/JavaScript"

    def search(self, query, max_results=5):
        try:
            url = (
                f"https://developer.mozilla.org/api/v1/search"
                f"?q={urllib.parse.quote(query)}&locale=en-US&size={max_results}"
            )
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
            results = []
            for doc in data.get("documents", []):
                results.append({"url": f"https://developer.mozilla.org{doc['mdn_url']}", "title": doc.get("title", "")})
            return results[:max_results]
        except Exception:
            return []


class NodeSource(_Source):
    name = "Node.js"
    id = "node"
    aliases = ["node", "nodejs"]
    base_url = "https://nodejs.org/api/"

    def search(self, query, max_results=5):
        q = query.strip().lower().replace(" ", "_")
        results = []
        module = q.split(".")[0].replace("-", "_")
        url = f"https://nodejs.org/api/{module}.html"
        if _head_ok(url):
            results.append({"url": url, "title": f"Node.js: {module}"})
            return results
        for mod in ["fs", "http", "https", "path", "os", "crypto", "stream",
                     "events", "buffer", "url", "util", "child_process",
                     "cluster", "net", "dns", "readline", "worker_threads",
                     "assert", "process", "console", "timers"]:
            if q in mod or mod in q:
                results.append({"url": f"https://nodejs.org/api/{mod}.html", "title": f"Node.js: {mod}"})
                if len(results) >= max_results:
                    break
        if not results:
            results.append({"url": "https://nodejs.org/api/", "title": "Node.js API Reference"})
        return results[:max_results]


class ReactSource(_Source):
    name = "React"
    id = "react"
    aliases = ["react", "reactjs"]
    base_url = "https://react.dev/reference/react"

    def search(self, query, max_results=5):
        q = query.strip()
        q_lower = q.lower()
        results = []
        for prefix in ["react", "react-dom", "react-dom/client", "react-dom/server"]:
            url = f"https://react.dev/reference/{prefix}/{q}"
            if _head_ok(url):
                results.append({"url": url, "title": f"React: {q}"})
                return results
        hooks = {"usestate": "useState", "useeffect": "useEffect",
                 "usecontext": "useContext", "useref": "useRef",
                 "usememo": "useMemo", "usecallback": "useCallback",
                 "usereducer": "useReducer", "uselayouteffect": "useLayoutEffect",
                 "usetransition": "useTransition", "useid": "useId",
                 "suspense": "Suspense", "fragment": "Fragment",
                 "createcontext": "createContext", "forwardref": "forwardRef",
                 "lazy": "lazy", "memo": "memo", "startTransition": "startTransition"}
        matched = hooks.get(q_lower)
        if matched:
            url = f"https://react.dev/reference/react/{matched}"
            results.append({"url": url, "title": f"React: {matched}"})
            return results
        for topic in ["state", "effects", "context", "refs", "components",
                       "props", "rendering", "events", "forms"]:
            if topic in q_lower:
                results.append({"url": "https://react.dev/learn", "title": "React Learn"})
                return results
        results.append({"url": "https://react.dev/reference/react", "title": "React API Reference"})
        return results[:max_results]


class VueSource(_Source):
    name = "Vue.js"
    id = "vue"
    aliases = ["vue", "vuejs", "vue3"]
    base_url = "https://vuejs.org/api/"

    def search(self, query, max_results=5):
        q = query.strip()
        q_lower = q.lower().replace(" ", "-")
        results = []
        sections = {
            "ref": "reactivity-core", "reactive": "reactivity-core",
            "computed": "reactivity-core", "watch": "reactivity-core",
            "watcheffect": "reactivity-core", "readonly": "reactivity-core",
            "shallowref": "reactivity-advanced", "triggerref": "reactivity-advanced",
            "onmounted": "composition-api-lifecycle", "onupdated": "composition-api-lifecycle",
            "onunmounted": "composition-api-lifecycle", "setup": "composition-api-setup",
            "defineprops": "sfc-script-setup", "defineemits": "sfc-script-setup",
            "provide": "composition-api-dependency-injection",
            "inject": "composition-api-dependency-injection",
            "v-if": "built-in-directives", "v-for": "built-in-directives",
            "v-model": "built-in-directives", "v-bind": "built-in-directives",
            "v-on": "built-in-directives", "component": "built-in-special-elements",
        }
        matched_section = sections.get(q_lower)
        if matched_section:
            url = f"https://vuejs.org/api/{matched_section}.html#{q_lower}"
            results.append({"url": url, "title": f"Vue: {q}"})
            return results
        url = f"https://vuejs.org/api/{q_lower}.html"
        if _head_ok(url):
            results.append({"url": url, "title": f"Vue: {q}"})
            return results
        url = f"https://vuejs.org/guide/{q_lower}.html"
        if _head_ok(url):
            results.append({"url": url, "title": f"Vue Guide: {q}"})
            return results
        results.append({"url": "https://vuejs.org/api/", "title": "Vue.js API Reference"})
        return results[:max_results]


class GoSource(_Source):
    name = "Go"
    id = "go"
    aliases = ["go", "golang", "gin", "echo", "fiber"]
    base_url = "https://pkg.go.dev/"

    def search(self, query, max_results=5):
        try:
            url = f"https://pkg.go.dev/search?q={urllib.parse.quote(query)}&m=symbol"
            _, html = _fetch(url)
            results = []
            seen = set()
            for m in re.finditer(
                r'<a\s+href="(/[^"]+)"[^>]*data-test-id="snippet-title"[^>]*>\s*'
                r'<span[^>]*>[^<]*</span>\s*([^<]+)</a>\s*'
                r'<span[^>]*>in</span>\s*'
                r'<a\s+href="(/[^"?]+)"[^>]*>([^<]+)</a>',
                html, re.DOTALL,
            ):
                symbol_path = m.group(1).strip()
                symbol_name = m.group(2).strip()
                pkg_name = m.group(4).strip()
                title = f"{pkg_name}.{symbol_name}"
                full_url = f"https://pkg.go.dev{symbol_path}"
                if full_url not in seen:
                    seen.add(full_url)
                    results.append({"url": full_url, "title": title})
                    if len(results) >= max_results:
                        break
            return results[:max_results]
        except Exception:
            return []


class RustSource(_Source):
    name = "Rust"
    id = "rust"
    aliases = ["rust", "rs", "tokio", "serde", "actix"]
    base_url = "https://doc.rust-lang.org/std/"

    def search(self, query, max_results=5):
        results = []
        q = query.strip()
        q_lower = q.lower().replace("::", "/")
        std_patterns = [
            f"https://doc.rust-lang.org/std/{q_lower}/index.html",
            f"https://doc.rust-lang.org/std/{q_lower}/struct.{q.split('::')[-1]}.html",
            f"https://doc.rust-lang.org/std/{q_lower}/enum.{q.split('::')[-1]}.html",
            f"https://doc.rust-lang.org/std/{q_lower}/trait.{q.split('::')[-1]}.html",
            f"https://doc.rust-lang.org/std/{q_lower}/fn.{q.split('::')[-1]}.html",
        ]
        if "::" not in q:
            for mod in ["vec", "option", "result", "string", "collections",
                         "sync", "io", "fs", "net", "fmt", "iter"]:
                std_patterns.insert(0, f"https://doc.rust-lang.org/std/{mod}/struct.{q}.html")
                std_patterns.insert(1, f"https://doc.rust-lang.org/std/{mod}/enum.{q}.html")
                std_patterns.insert(2, f"https://doc.rust-lang.org/std/{mod}/trait.{q}.html")
        for url in std_patterns:
            if _head_ok(url, timeout=4):
                return [{"url": url, "title": q}]
        try:
            url = f"https://docs.rs/releases/search?query={urllib.parse.quote(query)}"
            _, html = _fetch(url)
            links = _parse_links(html, "https://docs.rs")
            for link in links:
                if "/latest/" in link["url"] or "/releases/" not in link["url"]:
                    results.append(link)
                    if len(results) >= max_results:
                        break
        except Exception:
            pass
        if not results:
            results.append({"url": f"https://doc.rust-lang.org/std/?search={urllib.parse.quote(query)}", "title": f"Rust std: {query}"})
        return results[:max_results]


class JavaSource(_Source):
    name = "Java"
    id = "java"
    aliases = ["java", "jdk"]
    base_url = "https://docs.oracle.com/en/java/javase/21/docs/api/"

    def search(self, query, max_results=5):
        q = query.strip()
        results = []
        if "." in q:
            parts = q.rsplit(".", 1)
            package_path = parts[0].replace(".", "/")
            class_name = parts[1]
            url = f"https://docs.oracle.com/en/java/javase/21/docs/api/java.base/{package_path}/{class_name}.html"
            if _head_ok(url):
                return [{"url": url, "title": q}]
        for pkg in ["java/lang", "java/util", "java/io", "java/nio", "java/net",
                     "java/util/stream", "java/util/concurrent"]:
            url = f"https://docs.oracle.com/en/java/javase/21/docs/api/java.base/{pkg}/{q}.html"
            if _head_ok(url, timeout=3):
                results.append({"url": url, "title": f"{pkg.replace('/', '.')}.{q}"})
                if len(results) >= max_results:
                    return results
        if not results:
            results.append({"url": f"https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/lang/{q}.html", "title": f"Java docs: {q}"})
        return results[:max_results]


class SpringSource(_Source):
    name = "Spring"
    id = "spring"
    aliases = ["spring", "springboot", "spring-boot"]
    base_url = "https://docs.spring.io/spring-framework/reference/"

    def search(self, query, max_results=5):
        q = query.strip().lower().replace(" ", "-")
        results = []
        for section in ["web", "data-access", "core", "integration", "testing",
                        "web/webmvc", "web/webflux", "data-access/jdbc",
                        "data-access/orm", "core/beans", "core/aop"]:
            if q in section or section.split("/")[-1] in q:
                url = f"https://docs.spring.io/spring-framework/reference/{section}.html"
                if _head_ok(url):
                    results.append({"url": url, "title": f"Spring: {section}"})
                    return results
        url = f"https://docs.spring.io/spring-boot/reference/{q}.html"
        if _head_ok(url):
            results.append({"url": url, "title": f"Spring Boot: {query}"})
            return results
        class_name = query.strip().split(".")[-1]
        if class_name[0:1].isupper():
            results.append({"url": f"https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/{q.replace('.', '/')}.html", "title": f"Spring API: {class_name}"})
        if not results:
            results.append({"url": self.base_url, "title": "Spring Framework Reference"})
        return results[:max_results]


class RubySource(_Source):
    name = "Ruby"
    id = "ruby"
    aliases = ["ruby", "rb"]
    base_url = "https://docs.ruby-lang.org/en/"

    def search(self, query, max_results=5):
        q = query.strip()
        results = []
        class_name = q.split(".")[0].split("#")[0].strip()
        if class_name[0:1].isupper():
            url = f"https://docs.ruby-lang.org/en/master/{class_name}.html"
            if _head_ok(url):
                results.append({"url": url, "title": f"{class_name} — Ruby docs"})
                return results
        results.append({"url": f"https://ruby-doc.org/search.html?q={urllib.parse.quote(query)}", "title": f"Search Ruby docs for: {query}"})
        return results[:max_results]


class RailsSource(_Source):
    name = "Ruby on Rails"
    id = "rails"
    aliases = ["rails", "rubyonrails", "ror"]
    base_url = "https://api.rubyonrails.org/"

    def search(self, query, max_results=5):
        q = query.strip()
        results = []
        class_name = q.split(".")[0].split("#")[0].split("::")[0].strip()
        if class_name[0:1].isupper():
            for ns in ["", "ActiveRecord/", "ActionController/", "ActiveModel/",
                        "ActionView/", "ActiveSupport/", "ActionMailer/"]:
                url = f"https://api.rubyonrails.org/classes/{ns}{class_name}.html"
                if _head_ok(url):
                    results.append({"url": url, "title": f"Rails: {ns}{class_name}"})
                    return results
        slug = q.lower().replace(" ", "_").replace("::", "_")
        for guide in ["active_record_basics", "active_record_migrations",
                       "active_record_validations", "active_record_associations",
                       "active_record_querying", "action_controller_overview",
                       "routing", "layouts_and_rendering", "action_view_overview",
                       "action_mailer_basics", "active_job_basics",
                       "active_storage_overview", "action_cable_overview",
                       "testing", "security", "configuring"]:
            if slug in guide or guide.split("_")[0] in slug:
                url = f"https://guides.rubyonrails.org/{guide}.html"
                results.append({"url": url, "title": f"Rails Guide: {guide.replace('_', ' ').title()}"})
                return results
        results.append({"url": "https://api.rubyonrails.org/", "title": "Rails API Reference"})
        return results[:max_results]


class CSharpSource(_Source):
    name = "C# / .NET"
    id = "csharp"
    aliases = ["csharp", "c#", "cs", "dotnet", "aspnet", "entityframework", "ef"]
    base_url = "https://learn.microsoft.com/en-us/dotnet/"

    def search(self, query, max_results=5):
        try:
            search_q = f"{query} C#"
            url = (
                f"https://learn.microsoft.com/api/search"
                f"?search={urllib.parse.quote(search_q)}"
                f"&locale=en-us&%24top={max_results}"
                f"&scope=.NET"
            )
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
            results = []
            for item in data.get("results", []):
                item_url = item.get("url", "")
                if item_url:
                    results.append({"url": item_url, "title": item.get("title", "")})
            return results[:max_results]
        except Exception:
            return []


class DartSource(_Source):
    name = "Dart"
    id = "dart"
    aliases = ["dart"]
    base_url = "https://api.dart.dev/stable/"

    def search(self, query, max_results=5):
        q = query.strip()
        results = []
        class_name = q.split(".")[0]
        if class_name[0:1].isupper():
            url = f"https://api.dart.dev/stable/dart-core/{class_name}-class.html"
            if _head_ok(url):
                results.append({"url": url, "title": f"{class_name} — Dart core"})
                return results
        results.append({"url": f"https://dart.dev/search?q={urllib.parse.quote(query)}", "title": f"Search Dart docs for: {query}"})
        return results[:max_results]


class FlutterSource(_Source):
    name = "Flutter"
    id = "flutter"
    aliases = ["flutter"]
    base_url = "https://api.flutter.dev/flutter/"

    def search(self, query, max_results=5):
        q = query.strip()
        results = []
        class_name = q.split(".")[0].split("/")[-1]
        for lib in ["widgets", "material", "cupertino", "painting", "rendering",
                     "animation", "gestures", "foundation", "services",
                     "dart-ui", "dart-core", "dart-async"]:
            url = f"https://api.flutter.dev/flutter/{lib}/{class_name}-class.html"
            if _head_ok(url):
                results.append({"url": url, "title": f"Flutter: {class_name} ({lib})"})
                return results
        url = f"https://api.flutter.dev/flutter/widgets/{class_name}-class.html"
        results.append({"url": url, "title": f"Flutter: {class_name}"})
        return results[:max_results]


class CSource(_Source):
    name = "C/C++"
    id = "c"
    aliases = ["c", "cpp", "c++", "cppreference"]
    base_url = "https://en.cppreference.com/w/"

    _INDEX_PAGES = [
        "https://en.cppreference.com/w/c/io",
        "https://en.cppreference.com/w/c/string/byte",
        "https://en.cppreference.com/w/c/memory",
        "https://en.cppreference.com/w/c/numeric/math",
        "https://en.cppreference.com/w/c/language",
        "https://en.cppreference.com/w/cpp/container",
        "https://en.cppreference.com/w/cpp/algorithm",
        "https://en.cppreference.com/w/cpp/string/basic_string",
        "https://en.cppreference.com/w/cpp/io",
        "https://en.cppreference.com/w/cpp/memory",
    ]

    def search(self, query, max_results=5):
        q = query.strip().lower().replace("::", "/").replace("std/", "")
        results = []
        q_url = q.replace(" ", "_")
        for prefix in ["w/c/io/", "w/cpp/io/", "w/c/string/byte/",
                        "w/c/memory/", "w/c/numeric/math/", "w/c/language/",
                        "w/cpp/container/", "w/cpp/algorithm/",
                        "w/cpp/string/basic_string/", "w/cpp/memory/"]:
            probe = f"https://en.cppreference.com/{prefix}{q_url}"
            if _head_ok(probe, timeout=4):
                return [{"url": probe, "title": query}]
        for index_url in self._INDEX_PAGES:
            if len(results) >= max_results:
                break
            try:
                _, html = _fetch(index_url)
                for m in re.finditer(r'href="([^"]*\.html)"[^>]*>(.*?)</a>', html, re.DOTALL | re.IGNORECASE):
                    href = m.group(1)
                    title = re.sub(_RE_HTML_TAGS, "", m.group(2)).strip()
                    if q in title.lower() or q in href.lower():
                        full_url = urllib.parse.urljoin(index_url, href)
                        if full_url not in [r["url"] for r in results]:
                            results.append({"url": full_url, "title": title or query})
                            if len(results) >= max_results:
                                break
            except Exception:
                continue
        return results[:max_results]


class BashSource(_Source):
    name = "Bash"
    id = "bash"
    aliases = ["bash", "sh", "shell", "zsh"]
    base_url = "https://www.gnu.org/software/bash/manual/"

    def search(self, query, max_results=5):
        q = query.strip().lower().replace(" ", "-")
        return [
            {"url": f"https://www.gnu.org/software/bash/manual/bash.html#{urllib.parse.quote(q)}", "title": f"Bash manual: {query}"},
            {"url": "https://www.gnu.org/software/bash/manual/bash.html", "title": "Bash Reference Manual"},
        ]


class SQLSource(_Source):
    name = "SQL (SQLite)"
    id = "sql"
    aliases = ["sql", "sqlite", "mysql", "postgres", "postgresql"]
    base_url = "https://www.sqlite.org/docs.html"

    def search(self, query, max_results=5):
        results = []
        try:
            url = f"https://www.sqlite.org/search?s=d&q={urllib.parse.quote(query)}"
            _, html = _fetch(url)
            links = _parse_links(html, "https://www.sqlite.org")
            for link in links[:max_results]:
                if link["url"].endswith(".html"):
                    results.append(link)
        except Exception:
            pass
        if not results:
            results.append({"url": f"https://www.sqlite.org/lang_{query.lower().replace(' ', '')}.html", "title": f"SQLite: {query}"})
        return results[:max_results]


# ---------------------------------------------------------------------------
# Registry — module-level, built once at import
# ---------------------------------------------------------------------------

_ALL_SOURCES = [
    PHPSource(), PythonSource(), JavaScriptSource(), GoSource(), RustSource(),
    JavaSource(), RubySource(), CSharpSource(), DartSource(), CSource(),
    BashSource(), SQLSource(), LaravelSource(), DjangoSource(), FlaskSource(),
    NumPySource(), PandasSource(), NodeSource(), ReactSource(), VueSource(),
    SpringSource(), RailsSource(), FlutterSource(),
]

_ALIAS_MAP: dict = {}
for _src in _ALL_SOURCES:
    for _alias in _src.aliases:
        _ALIAS_MAP[_alias.lower()] = _src


def resolve_source(language):
    return _ALIAS_MAP.get(language.strip().lower())


def lookup(language, query):
    source = resolve_source(language)
    if source is None:
        langs = ", ".join(s.id for s in _ALL_SOURCES)
        return {"error": f"Unknown language/framework: {language}. Supported: {langs}"}
    results = source.search(query)
    if not results:
        return {"error": f"No results found for '{query}' in {source.name} docs.", "source": source.name, "url": source.base_url}
    top = results[0]
    text = _fetch_and_extract(top["url"])
    return {"text": text, "url": top["url"], "source": source.name}


# ---------------------------------------------------------------------------
# Ability class
# ---------------------------------------------------------------------------

class ProgrammingDocsSearchAbility(Ability):
    NAME = "programming_docs_search"
    SEARCH_TOOLTIP = "programming docs search"
    POLICY_CATEGORY = "Search & Tools"
    POLICY_LABELS = {"": "Search programming docs"}
    SUMMARY = "Search official documentation for 12 programming languages and 11 frameworks by language name and query."
    EXAMPLES = [
        "how do I use asyncio in Python",
        "what is the signature of array_map in PHP",
        "look up the useState hook in React",
        "how does Eloquent ORM work in Laravel",
        "what is the difference between Arc and Rc in Rust",
        "Django queryset filter documentation",
        "how to use goroutines in Go",
        "look up the pandas DataFrame merge method",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "language": {
                "type": "string",
                "enum": [
                    "php", "python", "javascript", "typescript", "go", "rust", "java",
                    "ruby", "csharp", "dart", "c", "cpp", "bash", "sql", "laravel",
                    "django", "flask", "numpy", "pandas", "node", "react", "vue",
                    "spring", "rails", "flutter",
                ],
                "description": "Programming language or framework to search documentation for.",
            },
            "query": {
                "type": "string",
                "description": "What to look up — function name, class, method, module, or concept in plain language.",
            },
        },
        "required": ["language", "query"],
    }
    TIMEOUT = 15

    def run(self, channel: str, params: dict, telemetry: dict | None) -> dict:
        language = (params.get("language") or "").strip()
        query = (params.get("query") or "").strip()
        if not language:
            return {"text": "", "error": "Missing required parameter: language"}
        if not query:
            return {"text": "", "error": "Missing required parameter: query"}
        return lookup(language, query)
