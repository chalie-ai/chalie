"""SearchFilesAbility — locate files by name (glob) or content (grep).

Cross-platform alternative to ``bash find`` / ``bash grep`` so the LLM
gets consistent behaviour across macOS, Linux, and Windows — including
mounted drives and connected storage.

Returns a sealed :class:`abilities._result.ToolResult` (never a wire envelope):

* ``glob`` success → ``{"files": [<abs path>, …]}``.
* ``grep`` success → ``{"matches": [{"file": <abs>, "line": <int>,
  "text": <line>, "context": <surrounding lines>}, …]}`` — one row per matched
  line.
* Results are PAGINATED at 5 per page (a result = a file for glob, a matched
  line for grep). Pass ``page`` to walk the list; when more than one page exists
  the body carries a "use the `page` parameter … page N from M pages" note and
  ``meta page``/``page_count``.
* The full result list for a query is cached to ``/tmp/<md5(action:query)>.json``
  for 10 minutes so paging never re-walks the tree; a cache older than that is
  deleted and rebuilt.
* ``/proc``, ``/sys``, ``/dev`` and every non-regular file are skipped during the
  walk, so a read can never park forever on a special file.
* A walk that hits the time budget or the result ceiling sets
  ``meta truncated=true`` so the cut is never silent.
* Zero hits → SUCCESS with ``count=0`` and a broaden suggestion in the body.
* Bad inputs → ``err()`` with a stable kebab ``code`` (``unknown-action`` /
  ``invalid-param`` / ``empty-query`` / ``directory-not-found`` /
  ``not-a-directory`` / ``invalid-regex``).
"""

import hashlib
import json
import os
import re
import time
from collections import deque
from collections.abc import Generator
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from abilities._ability import Ability
from abilities._result import ToolResult
from configs.enums.param_key import Keys
from contracts.params.search_files_params_bag import (
    DEFAULT_CONTEXT_LINES,
    SearchFilesGlobParams,
    SearchFilesGrepParams,
    SearchFilesParamsBag,
)

if TYPE_CHECKING:
    from contracts.params.param_bag import ParamBag

_RESULTS_PER_PAGE = 5
# Safety ceiling on total results gathered for one query, so a pathological
# match-dense tree (e.g. a 100MB file with a million matching lines) can never
# bloat the cache file or memory. A hit sets truncated=true.
_MAX_RESULTS = 5000
_BUDGET_RATIO = 0.8
# Cooperative wall-clock budget (seconds) for a single walk. NOT a framework
# execution timeout — it bounds the traversal so an enormous tree returns a
# partial result with truncated=true instead of crawling indefinitely.
_WALK_BUDGET_S = 30
_CACHE_DIR = "/tmp"
_CACHE_TTL_S = 600
# Special filesystems whose files can block read() forever. Pruned from the walk.
_SKIP_PREFIXES = ("/proc", "/sys", "/dev")

_NARROW_HINT = "Narrow the directory or tighten the pattern for complete results."
_BROADEN_HINT = "Broaden the pattern or widen the directory and try again."


class SearchFilesAbility(Ability[SearchFilesParamsBag]):
    # The dispatcher pre-gates these BEFORE the policy gate: an unknown action →
    # code=unknown-action with valid=(glob, grep); a present action missing
    # 'query' → a single code=missing-params error. A present-but-whitespace
    # query passes the pre-gate (the value is truthy); the bag's router rejects
    # it as empty-query before run() is reached.
    ACTION_REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "glob": (Keys.query,),
        "grep": (Keys.query,),
    }

    # The typed input contract: the dispatch seam builds the bag via
    # SearchFilesParamsBag.from_params before run() is called.
    PARAMS: ClassVar["type[ParamBag] | None"] = SearchFilesParamsBag

    def get_name(self) -> str:
        return "search_files"

    def get_summary(self) -> str:
        return (
            "Locate files on disk by filename pattern (glob) or by content (grep). "
            "Use this BEFORE reaching for bash when you need to find a file you "
            "don't already know the path of. Use in conjunction with the `read` "
            "tool to then get a located file's contents into context."
        )

    def get_examples(self) -> list[str]:
        return [
            "find all yaml files under config",
            "where is the message_processor file",
            "search the backend for files containing 'PolicyService'",
            "list every markdown doc under docs/",
            "grep for 'TODO' in my notes folder",
            "which file defines _FIND_TOOLS_GUARDRAILS",
            "show me all log files in /tmp",
            "find files matching test_*.py",
        ]

    def get_search_tooltip(self) -> str:
        return "Find files by name or content"

    _PARAMETERS: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            Keys.action: {
                "type": "string",
                "enum": ["glob", "grep"],
                "description": (
                    "'glob' = match files by name/pattern (e.g. '*.py', "
                    "'**/test_*.yaml'). 'grep' = search for content matches "
                    "inside files. Pick 'glob' when you know the filename "
                    "shape, 'grep' when you know what's inside the file."
                ),
            },
            Keys.query: {
                "type": "string",
                "description": (
                    "For 'glob': a filename or glob pattern (e.g. '*.md', "
                    "'**/*.log', 'config.*'). For 'grep': the literal "
                    "string or regex to search file contents for."
                ),
            },
            Keys.directory: {
                "type": "string",
                "description": (
                    "Optional directory to search under. Defaults to the "
                    "filesystem root (/). Provide a narrower path when "
                    "possible for faster results. Absolute path or "
                    "~-prefixed home-relative path."
                ),
            },
            Keys.page: {
                "type": "integer",
                "description": (
                    "Which page of results to return (1-based, defaults to 1). "
                    "Results are paginated 5 per page; when more pages exist the "
                    "result says so — call again with the next page to read more."
                ),
            },
            Keys.context_lines: {
                "type": "integer",
                "description": (
                    "Grep only. Number of lines to show above AND below "
                    "each matched line, carried on each row's 'context' field. "
                    "Defaults to 5. Maximum 20."
                ),
            },
        },
        "required": [Keys.action, Keys.query],
    }

    def get_parameters(self) -> dict[str, object]:
        return self._PARAMETERS

    def run(self, params: SearchFilesParamsBag) -> ToolResult:
        # The router factory only ever yields the two leaves; the isinstance
        # chain hands _search each leaf's exact fields (glob renders no
        # context, but the gather/cache key still carries the default width).
        if isinstance(params, SearchFilesGlobParams):
            return self._search("glob", params.query, params.directory, params.page, DEFAULT_CONTEXT_LINES)
        if isinstance(params, SearchFilesGrepParams):
            return self._search("grep", params.query, params.directory, params.page, params.context_lines)
        return ToolResult.err(
            f"Unknown search_files params bag: {type(params).__name__}.",
            code="unknown-action",
            valid=("glob", "grep"),
        )

    def _search(self, action: str, query: str, directory: str, page: int, context_lines: int) -> ToolResult:
        root = self._resolve_directory(directory)
        if isinstance(root, ToolResult):
            return root
        # re.error is the ONLY caught failure — a bad grep pattern is a user
        # input fault. Every other failure bubbles to the dispatcher.
        try:
            results, truncated = _gather(action, query, root, context_lines)
        except re.error as exc:
            return ToolResult.err(
                f"Invalid regex {query!r}: {str(exc)[:120]}",
                code="invalid-regex",
                hint="escape the special characters or pass a valid Python regex.",
            )
        return _paginate(action, results, page, truncated)

    def _resolve_directory(self, directory: str) -> "Path | ToolResult":
        root_str = directory or "/"
        try:
            root = Path(os.path.expanduser(root_str)).resolve()
        except (OSError, RuntimeError) as exc:
            return ToolResult.err(
                f"Invalid directory {root_str!r}: {str(exc)[:120]}",
                code="directory-not-found",
                hint="pass an absolute path or a ~-prefixed home-relative path.",
            )
        if not root.exists():
            return ToolResult.err(
                f"Directory not found: {root}",
                code="directory-not-found",
                hint="pass a directory that exists; an absolute path is safest.",
            )
        if not root.is_dir():
            return ToolResult.err(
                f"Not a directory: {root}",
                code="not-a-directory",
                hint="pass a directory path, not a file path.",
            )
        return root


def _cache_path(action: str, query: str, context_lines: int, root: Path) -> str:
    # The key covers everything that changes the result set: the search root, the
    # action, the grep context width, and the query. Anything left out would serve
    # a stale-from-elsewhere result — e.g. the same query under two roots, or the
    # same grep at two context widths — collapsing to one scratchpad file.
    return os.path.join(
        _CACHE_DIR, f"{hashlib.md5(f'{root}:{action}:{context_lines}:{query}'.encode()).hexdigest()}.json"
    )


def _gather(action: str, query: str, root: Path, context_lines: int) -> tuple[list[object], bool]:
    """Return ``(results, truncated)`` — served from the /tmp scratchpad when a
    fresh (< 10 min) cache for this query exists, otherwise rebuilt by walking
    the tree and written back. Cache I/O is best-effort: a read/write failure
    falls back to a live walk rather than failing the search.
    """
    path = _cache_path(action, query, context_lines, root)
    try:
        if time.time() - os.path.getmtime(path) <= _CACHE_TTL_S:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
            return blob["results"], blob["truncated"]
        os.remove(path)  # stale — discard so the next caller rebuilds it
    except (OSError, ValueError, KeyError):
        pass

    deadline = time.monotonic() + _WALK_BUDGET_S * _BUDGET_RATIO
    if action == "glob":
        results, truncated = _do_glob(root, query, deadline)
    else:
        results, truncated = _do_grep(root, query, context_lines, deadline)

    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"results": results, "truncated": truncated}, fh)
    except OSError:
        pass
    return results, truncated


def _paginate(action: str, results: list[object], page: int, truncated: bool) -> ToolResult:
    total = len(results)
    page_count = max(1, (total + _RESULTS_PER_PAGE - 1) // _RESULTS_PER_PAGE)
    page = min(page, page_count)
    start = (page - 1) * _RESULTS_PER_PAGE
    shown = results[start : start + _RESULTS_PER_PAGE]

    key = "files" if action == "glob" else "matches"
    body: dict[str, object] = {key: shown}

    notes: list[str] = []
    if not total:
        kind = "files matched" if action == "glob" else "matches found"
        notes.append(f"No {kind}. {_BROADEN_HINT}")
    if page_count > 1:
        notes.append(
            "Result list is too large to fit in 1 response, use the `page` "
            "parameter to view more results. You are currently on page "
            f"{page} from {page_count} pages"
        )
    if truncated:
        notes.append(_NARROW_HINT)
    if notes:
        body["note"] = " ".join(notes)

    return ToolResult.ok(body, count=total, page=page, page_count=page_count, truncated=truncated)


def _skip(path: str) -> bool:
    return any(path == p or path.startswith(p + os.sep) for p in _SKIP_PREFIXES)


def _iter_files(root: Path, deadline: float) -> Generator[str, None, None]:
    """Yield regular files under *root*, pruning ``/proc``/``/sys``/``/dev`` and
    every non-regular entry (fifo, socket, device, broken symlink) so a later
    read can never block forever on a special file.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _skip(os.path.join(dirpath, d))]
        if _skip(dirpath):
            continue
        for name in filenames:
            if time.monotonic() > deadline:
                return
            fp = os.path.join(dirpath, name)
            if os.path.isfile(fp):  # follows symlinks; False for fifo/socket/dev/broken
                yield fp


def _do_glob(root: Path, pattern: str, deadline: float) -> tuple[list[object], bool]:
    """Return ``(paths_newest_first, truncated)`` for every name match under
    *root*, capped at ``_MAX_RESULTS``.
    """
    matches: list[tuple[float, str]] = []
    recursive = "**" in pattern or "/" in pattern
    truncated = False
    for fp in _iter_files(root, deadline):
        if recursive:
            if not (fnmatch(os.path.relpath(fp, root), pattern) or fnmatch(fp, pattern)):
                continue
        elif not fnmatch(os.path.basename(fp), pattern):
            continue
        try:
            mtime = os.path.getmtime(fp)
        except OSError:
            continue
        matches.append((mtime, fp))
        if len(matches) >= _MAX_RESULTS:
            truncated = True
            break

    if time.monotonic() > deadline:
        truncated = True
    matches.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in matches], truncated


def _do_grep(
    root: Path, query: str, context_lines: int, deadline: float
) -> tuple[list[object], bool]:
    """Return ``(rows_newest_file_first, truncated)`` — one row per matched line.
    Files are streamed line-by-line (never slurped whole) and the total is capped
    at ``_MAX_RESULTS`` so a match-dense tree can't blow up memory or the cache.
    """
    pattern = re.compile(query)
    files_with_hits: list[tuple[float, list[object]]] = []
    total = 0
    truncated = False
    for fp in _iter_files(root, deadline):
        rows = _grep_file(fp, pattern, context_lines, _MAX_RESULTS - total)
        if not rows:
            continue
        try:
            mtime = os.path.getmtime(fp)
        except OSError:
            mtime = 0.0
        files_with_hits.append((mtime, rows))
        total += len(rows)
        if total >= _MAX_RESULTS:
            truncated = True
            break

    if time.monotonic() > deadline:
        truncated = True
    files_with_hits.sort(key=lambda t: t[0], reverse=True)
    return [row for _m, rows in files_with_hits for row in rows], truncated


def _grep_file(
    fp: str, pattern: "re.Pattern[str]", context_lines: int, budget: int
) -> list[object]:
    """Stream one file, returning at most *budget* match rows in line order.
    Memory stays bounded: the file is never read whole, only the last
    ``context_lines`` lines are held for look-behind, and at most
    ``context_lines`` rows are open for look-ahead at any moment.
    """
    if budget <= 0:
        return []
    rows: list[object] = []
    before: "deque[str]" = deque(maxlen=context_lines)  # maxlen=0 → look-behind off
    open_rows: list[dict[str, object]] = []
    try:
        with open(fp, encoding="utf-8", errors="ignore") as fh:
            for i, raw in enumerate(fh):
                text = raw.rstrip()
                still_open: list[dict[str, object]] = []
                for m in open_rows:
                    cast("list[str]", m["_after"]).append(text)
                    if len(cast("list[str]", m["_after"])) >= context_lines:
                        rows.append(_finalize(m, context_lines))
                    else:
                        still_open.append(m)
                open_rows = still_open
                if pattern.search(raw):
                    m = {"file": fp, "line": i + 1, "text": text, "_before": list(before), "_after": []}
                    if context_lines > 0:
                        open_rows.append(m)
                    else:
                        rows.append(_finalize(m, context_lines))
                    if len(rows) + len(open_rows) >= budget:
                        break
                before.append(text)  # no-op when context_lines=0 (maxlen=0 deque)
            for m in open_rows:
                rows.append(_finalize(m, context_lines))
    except OSError:
        return rows[:budget]
    return rows[:budget]


def _finalize(m: dict[str, object], context_lines: int) -> dict[str, object]:
    row: dict[str, object] = {"file": m["file"], "line": m["line"], "text": m["text"]}
    if context_lines > 0:
        row["context"] = "\n".join(
            cast("list[str]", m["_before"]) + [cast(str, m["text"])] + cast("list[str]", m["_after"])
        )
    return row
