"""Parameter-key canonicalisation — the shared key registry and the dispatch-seam
key healer that make every tool resilient to the argument keys a weak model emits.

Two cooperating layers, applied to every tool call at the dispatch seam BEFORE
the ACTION_REQUIRED pre-gate, the policy gate, or ``run()`` ever sees the params:

1. **Key sanitisation (generic, all tools).** An incoming argument key is
   lower-cased and stripped of every character outside ``[a-z0-9_-]``. This alone
   heals the single commonest model defect — a stray escaped quote or a capital,
   e.g. ``source"`` → ``source`` and ``MAX_CHARS`` → ``max_chars`` — with zero
   per-tool knowledge. (This is the class of breakage TKT-963 root-caused: a
   model stored ``{"source\"": "https://…"}`` and ``read`` bounced on
   ``source-required`` because the corrupt KEY never matched.)

2. **Variant resolution (registry-driven, per-tool).** A sanitised key that is
   not one of the tool's declared parameters is matched against the variant
   ladders in :data:`VARIANTS`; when it belongs to exactly one of *that tool's
   own* declared parameters it is rewritten to that parameter's canonical key.
   This is what lets ``read({"url": …})`` resolve to ``source`` while ``url``
   stays canonical for ``web_download`` — resolution is scoped to the tool's
   declared keys, never a global rename.

Resolution is two-pass and **declared-canonical-first**, so a parameter that is
itself a synonym of another (calendar's ``summary`` vs ``title``; ``document``'s
``id`` vs the ``list`` tool's ``id`` alias) always wins as itself and is never
rewritten away. An unrecognised key passes through **verbatim** (never the lossy
lower-cased form), so MCP camelCase keys (``serverName``) and any out-of-band key
survive intact for the tool or framework to handle.

:class:`Keys` is the single source of truth for every native parameter name: each
ability references ``Keys.<name>`` in its schema and its reads, so the registry
keys and the wire keys can never drift. :func:`validate_registry` enforces the
no-overlap invariant the whole design rests on — that no variant of one parameter
can collide with another parameter (or a framework key) within the same tool. It
runs as a feature test, NOT at import: importing every ability here would be a
circular import, and the invariant is a property of the live registry.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# The two keys the framework injects into / strips from every call (see
# Ability.get_input_schema and ToolDispatcher.dispatch/_execute). No parameter and
# no variant may ever collide with these, or a model key could hijack a framework
# slot. Compared on the squeezed form (see ``squeeze``).
_FRAMEWORK_KEYS = ("act_summary", "async")


class Keys:
    """Canonical names for every native ability parameter — one source of truth.

    Each constant's VALUE is the exact wire key the model sees in the tool schema;
    abilities reference the constant (``Keys.source``) instead of a bare string so
    the schema, the reads, and the :data:`VARIANTS` ladders can never drift apart.
    Attribute names mirror the wire key 1:1 (``id``/``list`` shadow builtins by
    design — the wire key is what matters).
    """

    action = "action"
    active_only = "active_only"
    area = "area"
    automation_id = "automation_id"
    body = "body"
    buffer_minutes = "buffer_minutes"
    category = "category"
    cc = "cc"
    code = "code"
    command = "command"
    content = "content"
    contents = "contents"
    context_lines = "context_lines"
    date_from = "date_from"
    date_time = "date_time"
    date_to = "date_to"
    destination_location = "destination_location"
    direction = "direction"
    directory = "directory"
    domain = "domain"
    down_kbps = "down_kbps"
    dtend = "dtend"
    dtstart = "dtstart"
    due_at = "due_at"
    duration_seconds = "duration_seconds"
    entity_id = "entity_id"
    evidence_transcript_ids = "evidence_transcript_ids"
    frequency = "frequency"
    goal = "goal"
    headers = "headers"
    host = "host"
    id = "id"  # noqa: A003 — attr name mirrors the wire key (shadows builtin)
    identifier = "identifier"
    image = "image"
    include_subagent_transcripts = "include_subagent_transcripts"
    item_id = "item_id"
    item_type = "item_type"
    items = "items"
    key = "key"
    keyword = "keyword"
    kind = "kind"
    language = "language"
    limit = "limit"
    list = "list"  # noqa: A003 — attr name mirrors the wire key (shadows builtin)
    location = "location"
    max_chars = "max_chars"
    max_files = "max_files"
    message = "message"
    minutes = "minutes"
    name = "name"
    operation = "operation"
    path = "path"
    permissions = "permissions"
    port_idx = "port_idx"
    provider = "provider"
    query = "query"
    quota_mb = "quota_mb"
    recurrence = "recurrence"
    rule = "rule"
    rule_id = "rule_id"
    section = "section"
    select = "select"
    sender = "sender"
    server_id = "server_id"
    service = "service"
    service_data = "service_data"
    source = "source"
    subject = "subject"
    summary = "summary"
    tags = "tags"
    target = "target"
    time_anchor = "time_anchor"
    time_range = "time_range"
    timeout = "timeout"
    timeout_s = "timeout_s"
    title = "title"
    to = "to"
    triage = "triage"
    uid = "uid"
    unanswered = "unanswered"
    up_kbps = "up_kbps"
    updates = "updates"
    url = "url"
    use_for = "use_for"
    value = "value"
    window_end = "window_end"
    window_start = "window_start"
    wlan_id = "wlan_id"

    @classmethod
    def values(cls) -> "frozenset[str]":
        """Every canonical wire key declared on this class (for schema-completeness
        checks: a registered ability's property keys MUST be a subset of these)."""
        return frozenset(
            v for k, v in vars(cls).items()
            if not k.startswith("_") and isinstance(v, str)
        )


# ── The variant registry ──────────────────────────────────────────────────────
#
# ``canonical key → the alternative keys a model emits for it``. Kept DELIBERATELY
# small and evidence-backed: every ladder here traces to a real, observed model
# failure, never a guess. The sanitisation layer (squeeze/lower) already gives
# EVERY parameter junk/case resilience for free; these ladders add only the
# semantic synonyms we have proof a model reaches for. Adding a ladder later is a
# one-line edit — :func:`validate_registry` (a feature test) guarantees the
# no-overlap invariant holds before the change can ship.
#
# Variants are matched on the squeezed form (see ``squeeze``), so listing both
# ``filepath`` and ``file_path`` is redundant — one covers both.
VARIANTS: "dict[str, frozenset[str]]" = {
    # read's historical source-alias ladder (TKT-834 / TKT-899): a model addresses
    # the read target by many names. Canonical = source.
    Keys.source: frozenset({
        Keys.url, "uri", "link", "href", Keys.path, "file", "filepath", "file_path",
    }),
    # web_download earns the SAME family on its canonical 'url' (TKT-964): the keys
    # a model uses for a fetch target are identical to read's. 'source' is included
    # so a model that learned read's key still lands. Canonical = url.
    Keys.url: frozenset({
        "uri", "link", "href", Keys.source, "address", "file_url", "download_url",
    }),
    # the list tool's historical 'id' alias for its list selector. Canonical = list.
    Keys.list: frozenset({Keys.id}),
}


class RegistryOverlapError(Exception):
    """Raised by :func:`validate_registry` when a tool's variant ladders overlap
    with another of its parameters or a framework key — the invariant that keeps
    key healing unambiguous."""


# ── Normalisation primitives ──────────────────────────────────────────────────

_DROP = re.compile(r"[^a-z0-9_-]")


def normalize(key: str) -> str:
    """Lower-case *key* and drop every character outside ``[a-z0-9_-]``.

    This is the generic sanitisation layer: ``source"`` → ``source``,
    ``"URL"`` → ``url``, ``max chars`` → ``maxchars`` (space dropped). Separators
    ``-`` / ``_`` are preserved so ``max_chars`` keeps its shape for the exact
    match; :func:`squeeze` removes them for the loose match.
    """
    return _DROP.sub("", str(key).lower())


def squeeze(key: str) -> str:
    """:func:`normalize` then drop ``-`` / ``_`` — the form keys are matched on.

    Collapses every spelling of one key to a single token, so ``MAX_CHARS``,
    ``max-chars`` and ``maxchars`` all compare equal (``maxchars``). Used for both
    the exact (declared-param) match and the variant match.
    """
    return normalize(key).replace("-", "").replace("_", "")


# ── The per-tool key healer (called at the dispatch seam) ──────────────────────

def canonicalize_keys(params: dict, schema: dict) -> dict:
    """Return *params* with its keys healed against a tool's declared *schema*.

    For each incoming key, in order:
      1. **exact** — if its squeezed form matches a declared parameter, emit that
         parameter's canonical key (heals quotes/case/separators, e.g.
         ``source"`` → ``source``, ``MAX_CHARS`` → ``max_chars``).
      2. **variant** — else if its squeezed form is a variant of exactly one
         declared parameter (via :data:`VARIANTS`), emit that parameter's key
         (e.g. ``url`` → ``source`` for ``read``).
      3. **passthrough** — else emit the ORIGINAL key verbatim, so MCP camelCase
         and any unknown key survive untouched.

    First write wins if two incoming keys resolve to the same canonical, mirroring
    the historical alias-ladder's "first key present wins". *schema* is the bare
    ``get_parameters()`` body (no framework fields); an empty/parameterless schema
    or empty params is returned unchanged.
    """
    properties = schema.get("properties") if schema else None
    if not properties or not params:
        return dict(params)

    declared = list(properties.keys())
    by_squeeze = {squeeze(k): k for k in declared}      # squeezed form → canonical key
    variant_map = _variant_map(declared, by_squeeze)    # squeezed variant → canonical key

    healed: dict = {}
    source_key: dict = {}                               # canonical → the raw key that filled it
    for key, value in params.items():
        sq = squeeze(key)
        canonical = by_squeeze.get(sq) or variant_map.get(sq) or key
        if canonical not in healed:
            healed[canonical] = value
            source_key[canonical] = key
        elif key != source_key[canonical]:
            # Two DISTINCT incoming keys resolved to one canonical (e.g. a model sent
            # both ``source`` and its ``url`` alias). First write wins — mirroring the
            # historical alias ladder — but the dropped value is never silent: a weak
            # model double-emitting a key is exactly the confusion worth surfacing.
            logger.warning(
                "[canonicalize_keys] %r and %r both map to %r — keeping %r, dropping %r",
                source_key[canonical], key, canonical, source_key[canonical], key,
            )
    return healed


def _variant_map(declared: "list[str]", by_squeeze: "dict[str, str]") -> "dict[str, str]":
    """Build ``squeezed variant → canonical key`` for one tool's declared params.

    A variant that squeezes onto a declared parameter is dropped (the real
    parameter always wins as itself), and a variant claimed by two declared
    parameters is dropped as ambiguous. :func:`validate_registry` forbids both for
    the real registry, so this is defensive: a future bad edit degrades to "no
    aliasing for that key", never silent mis-routing.
    """
    out: "dict[str, str | None]" = {}
    for canonical in declared:
        for variant in VARIANTS.get(canonical, ()):
            sq = squeeze(variant)
            if sq in by_squeeze:
                continue                       # a real parameter owns this form
            if sq in out and out[sq] != canonical:
                out[sq] = None                 # ambiguous across two params → drop
            elif sq not in out:
                out[sq] = canonical
    return {sq: c for sq, c in out.items() if c is not None}


# ── The invariant enforcer (run as a feature test, never at import) ────────────

def validate_registry(abilities: "list") -> None:
    """Assert the no-overlap invariant for every ability, or raise.

    For each tool, every variant of every declared parameter must:
      * not squeeze onto a DIFFERENT declared parameter of the same tool, and
      * not be claimed by two declared parameters of the same tool, and
      * not collide with a framework key.

    No declared parameter may collide with a framework key either. Every
    :data:`VARIANTS` key must be a real parameter of at least one ability (catches
    a typo'd canonical). Raises :class:`RegistryOverlapError` listing EVERY
    violation found (not just the first), so one run fixes the whole registry.
    """
    framework_sq = {squeeze(k) for k in _FRAMEWORK_KEYS}
    problems: "list[str]" = []
    all_declared: "set[str]" = set()

    for ability in abilities:
        name = ability.get_name()
        try:
            properties = (ability.get_parameters() or {}).get("properties") or {}
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name}: get_parameters() raised {exc!r}")
            continue
        declared = list(properties.keys())
        all_declared.update(declared)
        by_squeeze = {squeeze(k): k for k in declared}

        for param in declared:
            if squeeze(param) in framework_sq:
                problems.append(f"{name}: parameter {param!r} collides with a framework key")

        seen: "dict[str, str]" = {}
        for canonical in declared:
            for variant in VARIANTS.get(canonical, ()):
                sq = squeeze(variant)
                if sq in framework_sq:
                    problems.append(
                        f"{name}: variant {variant!r} of {canonical!r} collides with a framework key"
                    )
                other = by_squeeze.get(sq)
                if other is not None and other != canonical:
                    problems.append(
                        f"{name}: variant {variant!r} of {canonical!r} collides with declared "
                        f"parameter {other!r}"
                    )
                if sq in seen and seen[sq] != canonical:
                    problems.append(
                        f"{name}: variant {variant!r} is claimed by both {seen[sq]!r} and "
                        f"{canonical!r}"
                    )
                seen[sq] = canonical

    unknown = [k for k in VARIANTS if k not in all_declared]
    if unknown:
        problems.append(
            f"VARIANTS keys are not declared parameters of any ability: {sorted(unknown)}"
        )

    if problems:
        raise RegistryOverlapError(
            "Parameter-key registry overlap(s):\n  " + "\n  ".join(problems)
        )
