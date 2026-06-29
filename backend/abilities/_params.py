"""Parameter-key canonicalisation — the shared key registry and the dispatch-seam
key healer that make every tool resilient to the argument keys a weak model emits.

Two cooperating layers, applied to every tool call at the dispatch seam BEFORE
the ACTION_REQUIRED pre-gate, the policy gate, or ``run()`` ever sees the params:

1. **Key sanitisation (generic, all tools).** An incoming argument key is
   lower-cased and stripped of every character outside ``[a-z0-9_-]``. This alone
   heals the single commonest model defect — a stray escaped quote or a capital,
   e.g. ``source"`` → ``source`` and ``MAX_CHARS`` → ``max_chars`` — with zero
   per-tool knowledge. (This heals a common model defect: a model stores
   ``{"source\"": "https://…"}`` and ``read`` bounces on
   ``source-required`` because the corrupt KEY never matched.) This layer is
   :class:`KeyNormalizer`.

2. **Variant resolution (registry-driven, per-tool).** A sanitised key that is
   not one of the tool's declared parameters is matched against the variant
   ladders in :data:`VARIANTS`; when it belongs to exactly one of *that tool's
   own* declared parameters it is rewritten to that parameter's canonical key.
   This is what lets ``read({"url": …})`` resolve to ``source`` while ``url``
   stays canonical for ``web_download`` — resolution is scoped to the tool's
   declared keys, never a global rename. This layer is :class:`KeyHealer`.

Resolution is two-pass and **declared-canonical-first**, so a parameter that is
itself a synonym of another (calendar's ``summary`` vs ``title``; ``document``'s
``id`` vs the ``list`` tool's ``id`` alias) always wins as itself and is never
rewritten away. An unrecognised key passes through **verbatim** (never the lossy
lower-cased form), so MCP camelCase keys (``serverName``) and any out-of-band key
survive intact for the tool or framework to handle.

:class:`Keys` is the single source of truth for every native parameter name: each
ability references ``Keys.<name>`` in its schema and its reads, so the registry
keys and the wire keys can never drift. The no-overlap invariant the whole design
rests on — that no variant of one parameter can collide with another parameter (or
a framework key) within the same tool — is enforced as a feature test by
``tests/_registry_invariants.py`` (``RegistryInvariant``), NOT at import:
importing every ability here would be a circular import, and the invariant is a
property of the live registry, not of this module.
"""

from __future__ import annotations

import logging
import re
from typing import cast

logger = logging.getLogger(__name__)


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
    message = "message"
    minutes = "minutes"
    name = "name"
    operation = "operation"
    page = "page"
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


# ── The variant registry ──────────────────────────────────────────────────────
#
# ``canonical key → the alternative keys a model emits for it``. Two cooperating
# layers at the dispatch seam make every tool resilient to the argument keys a
# weak model emits: ``KeyNormalizer.squeeze`` (lower-case + strip non-[a-z0-9_-] +
# drop -/_) gives EVERY parameter junk/case/separator resilience for free, so the
# ladders below add only genuine *semantic* synonyms. ``KeyHealer`` then resolves a
# squeezed key that is not one of the tool's declared parameters against this
# registry, scoped to the tool's own declared keys (declared-canonical-first, so a
# parameter that is itself a synonym of another always wins as itself). An
# unrecognised key passes through verbatim.
#
# Evidence basis: each variant here was kept only when TWO weak models (a Sonnet
# agent and a Haiku agent) EACH independently reached for it as an alternative name
# for the canonical parameter — the strongest "a weak model will emit this" signal
# — plus a small curated set of universally-known abbreviations (q, n, num, cmd,
# msg, id). The squeeze layer already covers junk/case/separator, so listing both
# ``filepath`` and ``file_path`` is redundant — one covers both. The no-overlap
# invariant (no variant of one parameter may collide with another parameter or a
# framework key within the same tool) is enforced by ``RegistryInvariant`` — a
# feature test in ``tests/_registry_invariants.py`` — NOT at import: importing
# every ability here would be a circular import, and the invariant is a property
# of the live registry, not of this module. ``test_real_registry_has_no_variant_
# overlaps`` is the ship gate; the change cannot land unless it passes.
#
# Three tokens were auto-dropped during invariant derivation (kept on a better
# semantic home where there is no collision): ``start_date`` (dropped from
# date_from, kept on dtstart — calendar declares both) and ``end_date`` (dropped
# from date_to, kept on dtend — calendar declares both), because ``calendar``
# declares both date_from/date_to and dtstart/dtend and the two pairs would
# collide. ``text`` is registered under ``body`` ONLY (not ``keyword``): email
# declares both body and keyword, and registering text under both would collide,
# so the invariant forces a single home; body is the high-stakes target (a model
# emitting ``text`` on an email send/draft means the message body, not the search
# filter), so text→body is chosen. This means ``text`` never heals to ``keyword``
# for any tool — the only tool that declares keyword (email) also declares body,
# so the variant loses to the declared body; and no other tool declares keyword.
# Known action-dependent limitation: ``name→key`` on memory heals ``name`` to
# ``key`` for store/forget/save_graph (correct) but for recall/reflect (which read
# ``query`` and ignore ``key``) the model more likely means the search query; the
# value lands in the unused ``key`` and recall still bounces no-query-or-location.
# Accepted: the common store/forget path is correct, and dropping ``name`` from
# ``key`` would break those correct heals.
VARIANTS: "dict[str, frozenset[str]]" = {
    # read's historical source-alias ladder: a model addresses the read target by
    # many names. Canonical = source.
    Keys.source: frozenset({
        Keys.url, "uri", "link", "href", Keys.path, "file", "filepath", "file_path",
    }),
    # web_download earns the SAME family on its canonical 'url': the keys a model
    # uses for a fetch target are identical to read's. 'source' is included so a
    # model that learned read's key still lands. Canonical = url.
    Keys.url: frozenset({
        "uri", "link", "href", Keys.source, "address", "file_url", "download_url",
    }),
    # the list tool's historical 'id' alias for its list selector. Canonical = list.
    Keys.list: frozenset({Keys.id}),
    # web_browse names its task slot 'goal'; its sibling delegate web_search names
    # the SAME "what to do" slot 'query'. A model fluent in web_search reaches for
    # 'query' on web_browse — the two delegates are routinely conflated — so 'query'
    # heals to 'goal'. Scoped to web_browse (the only tool that declares 'goal'), so
    # the 14 tools that declare 'query' as their OWN canonical are never touched.
    # Canonical = goal.
    Keys.goal: frozenset({Keys.query}),
    # ── multi-tool parameters (highest blast radius — one heal helps many tools) ──
    Keys.action: frozenset({"method"}),
    Keys.query: frozenset({"keyword", "q", "search_query", "search_term", "term"}),
    Keys.limit: frozenset({"count", "max", "max_results", "n", "num", "top"}),
    Keys.name: frozenset({"filename", "label", "title"}),
    Keys.content: frozenset({Keys.body, "text"}),
    Keys.value: frozenset({"input", "text"}),
    Keys.date_from: frozenset({"begin_date", "from_date", "since"}),
    Keys.date_time: frozenset({"anchor_time", "time", "timestamp"}),
    Keys.date_to: frozenset({"to_date", "until"}),
    Keys.key: frozenset({"field", Keys.name}),
    Keys.location: frozenset({"city", "loc", "place", "region"}),
    Keys.path: frozenset({"directory", "filepath", "location"}),
    Keys.uid: frozenset({"email_id", "event_id", Keys.id, "identifier", "message_id", "uuid"}),
    Keys.use_for: frozenset({"description", "purpose", "when_to_use"}),
    # ── single-tool parameters ───────────────────────────────────────────────────
    Keys.active_only: frozenset({"connected_only"}),
    Keys.area: frozenset({"location", "room", "zone"}),
    Keys.automation_id: frozenset({"automation", "rule_id"}),
    Keys.body: frozenset({"message", "msg", "text"}),
    Keys.buffer_minutes: frozenset({"window"}),
    Keys.category: frozenset({"tag", "topic"}),
    Keys.code: frozenset({"python", "script", Keys.source}),
    Keys.command: frozenset({"bash", "cmd", "script", "shell"}),
    Keys.context_lines: frozenset({"surrounding_lines"}),
    Keys.destination_location: frozenset({"dest", "end_location"}),
    Keys.direction: frozenset({"orientation", "way"}),
    Keys.directory: frozenset({"base_path", "dir", "folder", Keys.path}),
    Keys.domain: frozenset({"category", "device_type", "entity_type"}),
    Keys.down_kbps: frozenset({"bandwidth_down", "download_speed"}),
    Keys.dtend: frozenset({"end_date", "end_datetime", "end_time"}),
    Keys.dtstart: frozenset({"start_date", "start_datetime", "start_time"}),
    Keys.due_at: frozenset({"deadline", "when"}),
    Keys.duration_seconds: frozenset({"duration", "length", "seconds"}),
    Keys.entity_id: frozenset({"device", "device_id", "entity", "ha_id"}),
    Keys.evidence_transcript_ids: frozenset({"evidence_ids", "transcript_ids"}),
    Keys.frequency: frozenset({"recurrence", "repeat", "schedule"}),
    Keys.headers: frozenset({"auth_headers", "extra_headers", "http_headers", "request_headers"}),
    Keys.host: frozenset({"address", "endpoint", "server_url", Keys.url}),
    Keys.id: frozenset({"doc_id", "document_id", "item_id", Keys.uid}),
    Keys.image: frozenset({"doc_id", "document_id", "file_id", "img", "photo"}),
    Keys.include_subagent_transcripts: frozenset({"include_agents", "show_subagents", "with_subagents"}),
    Keys.item_id: frozenset({Keys.id, "reminder_id", "schedule_id"}),
    Keys.item_type: frozenset({"kind", "reminder_type", "schedule_type"}),
    Keys.items: frozenset({"entries", "values"}),
    Keys.keyword: frozenset({"q", Keys.query, "search_query", "search_term", "term"}),
    Keys.max_chars: frozenset({"max_length"}),
    Keys.message: frozenset({Keys.body, "msg", "note", "text"}),
    Keys.minutes: frozenset({"duration", "mins"}),
    Keys.operation: frozenset({"op", "task"}),
    Keys.permissions: frozenset({"chmod", "perms"}),
    Keys.port_idx: frozenset({"port_id", "port_number"}),
    Keys.sender: frozenset({"author", "from", "from_address", "from_email"}),
    Keys.to: frozenset({"dest", "recipient", "recipients"}),
    Keys.triage: frozenset({"category", "label"}),
    Keys.unanswered: frozenset({"pending", "unread"}),
    Keys.up_kbps: frozenset({"uplink", "upload_bandwidth", "upload_limit", "upload_speed"}),
    Keys.updates: frozenset({"changes", "fields"}),
    Keys.window_end: frozenset({"end_hour", "end_time", "stop_time"}),
    Keys.window_start: frozenset({"begin", "from_time", "start_hour", "start_time"}),
    Keys.wlan_id: frozenset({"network", "network_id", "wifi_id"}),
}


class KeyNormalizer:
    """Reduce a raw argument key to the canonical form keys are matched on.

    One responsibility: the character-level sanitisation that gives every tool
    junk/case/separator resilience for free, independent of any registry.
    Stateless; injected into :class:`KeyHealer` so the healer never hard-codes its
    own matching rule and either layer can be swapped or tested in isolation.
    """

    _DROP = re.compile(r"[^a-z0-9_-]")

    def normalize(self, key: str) -> str:
        """Lower-case *key* and drop every character outside ``[a-z0-9_-]``.

        The generic sanitisation layer: ``source"`` → ``source``, ``"URL"`` →
        ``url``, ``max chars`` → ``maxchars`` (space dropped). Separators ``-`` /
        ``_`` are preserved so ``max_chars`` keeps its shape for the exact match;
        :meth:`squeeze` removes them for the loose match.
        """
        return self._DROP.sub("", str(key).lower())

    def squeeze(self, key: str) -> str:
        """:meth:`normalize` then drop ``-`` / ``_`` — the form keys are matched on.

        Collapses every spelling of one key to a single token, so ``MAX_CHARS``,
        ``max-chars`` and ``maxchars`` all compare equal (``maxchars``). Used for
        both the exact (declared-param) match and the variant match.
        """
        return self.normalize(key).replace("-", "").replace("_", "")


class KeyHealer:
    """Heal a tool call's argument KEYS against the tool's declared schema.

    One responsibility: turn the keys a weak model emitted into the tool's
    canonical parameter keys, so a stray quote, a capital, a separator, or a known
    synonym never bounces an otherwise-valid call.

    Dependencies are injected (DIP): the :data:`VARIANTS` registry and the
    :class:`KeyNormalizer` both default to the shared module values but can be
    supplied — for a test with a probe registry, or an alternative normalisation
    policy — without touching this class.
    """

    def __init__(
        self,
        variants: "dict[str, frozenset[str]]" = VARIANTS,
        normalizer: "KeyNormalizer | None" = None,
    ) -> None:
        self._variants = variants
        self._normalizer = normalizer or KeyNormalizer()

    def heal(self, params: dict[str, object], schema: dict[str, object]) -> dict[str, object]:
        """Return *params* with its keys healed against a tool's declared *schema*.

        For each incoming key, in order:
          1. **exact** — if its squeezed form matches a declared parameter, emit
             that parameter's canonical key (heals quotes/case/separators, e.g.
             ``source"`` → ``source``, ``MAX_CHARS`` → ``max_chars``).
          2. **variant** — else if its squeezed form is a variant of exactly one
             declared parameter (via :data:`VARIANTS`), emit that parameter's key
             (e.g. ``url`` → ``source`` for ``read``).
          3. **passthrough** — else emit the ORIGINAL key verbatim, so MCP
             camelCase and any unknown key survive untouched.

        First write wins if two incoming keys resolve to the same canonical,
        mirroring the historical alias-ladder's "first key present wins". *schema*
        is the bare ``get_parameters()`` body (no framework fields); an
        empty/parameterless schema or empty params is returned unchanged.
        """
        properties = cast("dict[str, object]", schema.get("properties")) if schema else None
        if not properties or not params:
            return dict(params)

        squeeze = self._normalizer.squeeze
        declared = list(properties.keys())
        by_squeeze = {squeeze(k): k for k in declared}     # squeezed form → canonical key
        variant_map = self._variant_map(declared, by_squeeze)  # squeezed variant → canonical key

        healed: dict[str, object] = {}
        source_key: dict[str, str] = {}                    # canonical → the raw key that filled it
        for key, value in params.items():
            sq = squeeze(key)
            canonical = by_squeeze.get(sq) or variant_map.get(sq) or key
            if canonical not in healed:
                healed[canonical] = value
                source_key[canonical] = key
            elif key != source_key[canonical]:
                # Two DISTINCT incoming keys resolved to one canonical (e.g. a model
                # sent both ``source`` and its ``url`` alias). First write wins —
                # mirroring the historical alias ladder — but the dropped value is
                # never silent: a weak model double-emitting a key is exactly the
                # confusion worth surfacing.
                logger.warning(
                    "[KeyHealer] %r and %r both map to %r — keeping %r, dropping %r",
                    source_key[canonical], key, canonical, source_key[canonical], key,
                )
        return healed

    def _variant_map(
        self, declared: "list[str]", by_squeeze: "dict[str, str]"
    ) -> "dict[str, str]":
        """Build ``squeezed variant → canonical key`` for one tool's declared params.

        A variant that squeezes onto a declared parameter is dropped (the real
        parameter always wins as itself), and a variant claimed by two declared
        parameters is dropped as ambiguous. ``RegistryInvariant`` forbids both for
        the real registry, so this is defensive: a future bad edit degrades to "no
        aliasing for that key", never silent mis-routing.
        """
        squeeze = self._normalizer.squeeze
        out: "dict[str, str | None]" = {}
        for canonical in declared:
            for variant in self._variants.get(canonical, ()):
                sq = squeeze(variant)
                if sq in by_squeeze:
                    continue                       # a real parameter owns this form
                if sq in out and out[sq] != canonical:
                    out[sq] = None                 # ambiguous across two params → drop
                elif sq not in out:
                    out[sq] = canonical
        return {sq: c for sq, c in out.items() if c is not None}
