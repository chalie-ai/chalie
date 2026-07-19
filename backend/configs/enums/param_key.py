# Copyright 2026 Chalie AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The parameter-key registry — canonical native parameter names and their variants.

Two declarative structures the key healer (:mod:`services.key_healer`) consumes at
the dispatch seam to make every tool resilient to the argument keys a weak model
emits:

:class:`Keys` is the single source of truth for every native parameter name: each
ability references ``Keys.<name>`` in its schema and its reads, so the registry
keys and the wire keys can never drift. Each member's VALUE is the exact wire key
the model sees in the tool schema; the member name mirrors that wire key 1:1.

:data:`VARIANTS` maps each canonical key to the alternative keys a model emits for
it — the semantic-synonym ladders the healer resolves against, scoped per-tool.

The no-overlap invariant the whole design rests on — that no variant of one
parameter can collide with another parameter (or a framework key) within the same
tool — is expressed as ``RegistryInvariant`` in ``tests/_registry_invariants.py``
(a test-only helper), NOT checked at import: importing every ability here would be
a circular import, and the invariant is a property of the live registry, not of
this module.
"""

from __future__ import annotations

from enum import StrEnum


class Keys(StrEnum):
    """Canonical names for every native ability parameter — one source of truth.

    Each member's VALUE is the exact wire key the model sees in the tool schema;
    abilities reference the member (``Keys.source``) instead of a bare string so
    the schema, the reads, and the :data:`VARIANTS` ladders can never drift apart.
    As a :class:`~enum.StrEnum` each member *is* its wire string — ``Keys.source ==
    "source"``, ``str(Keys.source) == "source"``, and it serialises and hashes as
    that string — so it is a drop-in for the literal everywhere the schema, JSON
    encoding, or ``params.get`` sees it. Member names mirror the wire key 1:1
    (``id``/``list`` shadow builtins by design — the wire key is what matters),
    with three exceptions that ``Enum``/``str`` reserve: the wire keys ``name``,
    ``value`` and ``title`` are declared as ``name_``/``value_``/``title_`` (PEP 8
    trailing-underscore) because bare ``name``/``value`` clash with ``Enum``'s own
    descriptors and ``title`` with ``str.title``. The VALUE is unchanged, so the
    model still sees ``name``/``value``/``title`` — only the Python attribute is
    spelled with a trailing underscore.
    """

    action = "action"
    active_only = "active_only"
    area = "area"
    args = "args"
    automation_id = "automation_id"
    body = "body"
    buffer_minutes = "buffer_minutes"
    category = "category"
    cc = "cc"
    command = "command"
    content = "content"
    contents = "contents"
    context_lines = "context_lines"
    current_path = "current_path"
    date_from = "date_from"
    date_time = "date_time"
    date_to = "date_to"
    day = "day"
    destination = "destination"
    destination_location = "destination_location"
    direction = "direction"
    directory = "directory"
    domain = "domain"
    down_kbps = "down_kbps"
    dtend = "dtend"
    dtstart = "dtstart"
    duration_seconds = "duration_seconds"
    end_line = "end_line"
    entity_id = "entity_id"
    evidence_transcript_ids = "evidence_transcript_ids"
    frequency = "frequency"
    fuzzy = "fuzzy"
    glob = "glob"
    goal = "goal"
    headers = "headers"
    host = "host"
    hour = "hour"
    id = "id"  # noqa: A003 — member name mirrors the wire key (shadows builtin)
    identifier = "identifier"
    image = "image"
    include_subagent_transcripts = "include_subagent_transcripts"
    instructions = "instructions"
    item_id = "item_id"
    items = "items"
    key = "key"
    keyword = "keyword"
    kind = "kind"
    language = "language"
    limit = "limit"
    list = "list"  # noqa: A003 — member name mirrors the wire key (shadows builtin)
    location = "location"
    max_chars = "max_chars"
    message = "message"
    minute = "minute"
    minutes = "minutes"
    month = "month"
    name_ = "name"  # trailing underscore: bare ``name`` is reserved by Enum
    new_path = "new_path"
    operation = "operation"
    page = "page"
    path = "path"
    pattern = "pattern"
    permission_code = "permission_code"
    permissions = "permissions"
    port_idx = "port_idx"
    provider = "provider"
    query = "query"
    quota_mb = "quota_mb"
    recurrence = "recurrence"
    replace_ = "replace"  # trailing underscore: bare `replace` is a str method (StrEnum base)
    replaces = "replaces"
    rule = "rule"
    rule_id = "rule_id"
    search = "search"
    section = "section"
    select = "select"
    sender = "sender"
    server_id = "server_id"
    service = "service"
    service_data = "service_data"
    source = "source"
    start_at = "start_at"
    start_line = "start_line"
    subject = "subject"
    summary = "summary"
    tags = "tags"
    target = "target"
    time_anchor = "time_anchor"
    time_range = "time_range"
    timeout = "timeout"
    timeout_s = "timeout_s"
    title_ = "title"  # trailing underscore: bare ``title`` clashes with ``str.title``
    to = "to"
    triage = "triage"
    uid = "uid"
    unanswered = "unanswered"
    up_kbps = "up_kbps"
    updates = "updates"
    url = "url"
    use_for = "use_for"
    value_ = "value"  # trailing underscore: bare ``value`` is reserved by Enum
    weekday = "weekday"
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
# unrecognised key passes through verbatim. Both layers live in
# :mod:`services.key_healer`.
#
# Evidence basis: each variant here was kept only when TWO weak models (a Sonnet
# agent and a Haiku agent) EACH independently reached for it as an alternative name
# for the canonical parameter — the strongest "a weak model will emit this" signal
# — plus a small curated set of universally-known abbreviations (q, n, num, cmd,
# msg, id). The squeeze layer already covers junk/case/separator, so listing both
# ``filepath`` and ``file_path`` is redundant — one covers both. The no-overlap
# invariant (no variant of one parameter may collide with another parameter or a
# framework key within the same tool) is expressed as ``RegistryInvariant`` — a
# test-only helper in ``tests/_registry_invariants.py`` — NOT checked at import:
# importing every ability here would be a circular import, and the invariant is a
# property of the live registry, not of this module. (That helper is not currently
# wired into the suite — see the follow-up on restoring its ship-gate test.)
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
    Keys.name_: frozenset({"filename", "label", "title"}),
    Keys.content: frozenset({Keys.body, "text"}),
    Keys.value_: frozenset({"input", "text"}),
    Keys.date_from: frozenset({"begin_date", "from_date", "since"}),
    Keys.date_time: frozenset({"anchor_time", "time", "timestamp"}),
    Keys.date_to: frozenset({"to_date", "until"}),
    Keys.key: frozenset({"field", Keys.name_}),
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
    Keys.command: frozenset({"bash", "cmd", "script", "shell"}),
    Keys.context_lines: frozenset({"surrounding_lines"}),
    Keys.destination_location: frozenset({"dest", "end_location"}),
    Keys.direction: frozenset({"orientation", "way"}),
    Keys.directory: frozenset({"base_path", "dir", "folder", Keys.path}),
    Keys.domain: frozenset({"category", "device_type", "entity_type"}),
    Keys.down_kbps: frozenset({"bandwidth_down", "download_speed"}),
    Keys.dtend: frozenset({"end_date", "end_datetime", "end_time"}),
    Keys.dtstart: frozenset({"start_date", "start_datetime", "start_time"}),
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
    Keys.items: frozenset({"entries", "values"}),
    Keys.keyword: frozenset({"q", Keys.query, "search_query", "search_term", "term"}),
    Keys.max_chars: frozenset({"max_length"}),
    Keys.message: frozenset({Keys.body, "msg", "note", "text"}),
    Keys.minutes: frozenset({"duration", "mins"}),
    Keys.operation: frozenset({"op", "task"}),
    Keys.permissions: frozenset({"chmod", "perms"}),
    Keys.port_idx: frozenset({"port_id", "port_number"}),
    Keys.recurrence: frozenset({"frequency", "repeat", "rrule", "schedule"}),
    Keys.replaces: frozenset({"old_value", "previous_value"}),
    Keys.sender: frozenset({"author", "from", "from_address", "from_email"}),
    Keys.to: frozenset({"dest", "recipient", "recipients"}),
    Keys.triage: frozenset({"category", "label"}),
    Keys.unanswered: frozenset({"pending", "unread"}),
    Keys.up_kbps: frozenset({"uplink", "upload_bandwidth", "upload_limit", "upload_speed"}),
    Keys.updates: frozenset({"changes", "fields"}),
    Keys.wlan_id: frozenset({"network", "network_id", "wifi_id"}),
}
