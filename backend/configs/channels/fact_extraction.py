from __future__ import annotations

import json

from services.processor_config import ProcessorConfig

# ── Fact-extraction config (subconscious worker fact pipeline) ────────
#
# One small, tool-free LLM call per episode that turns the episode gist into a
# set of constrained data_graph operations (Mem0 arXiv:2504.19413). The model
# sees the episode and the top-N data_graph rows that are already most similar
# to it, then emits EXACTLY ONE JSON object describing the ops to apply.
#
# The interface is identical across every model tier: the same prompt,
# the same op enum, the same JSON envelope. A stronger model makes better calls;
# the contract never branches. Unparseable output is the caller's NOOP-safe
# default — this module never writes; it only produces text for the worker.

# The four constrained operations the model may emit (Mem0 op enum).
OP_ADD = "ADD"
OP_UPDATE = "UPDATE"
OP_DELETE = "DELETE"
OP_NOOP = "NOOP"
VALID_OPS = frozenset({OP_ADD, OP_UPDATE, OP_DELETE, OP_NOOP})

# data_graph kind the fact pipeline writes. Hard, truth-valued user facts live in
# user_specific (names, dates, ownership, state) — the kind whose contradiction
# policy is bi-temporal supersession.
FACT_KIND = "user_specific"


def parse_fact_ops(text: str) -> list[dict[str, object]]:
    """Parse the model's constrained output into a list of validated op dicts.

    Returns the subset of ops that are structurally valid (known op verb; key
    present for ADD/UPDATE/DELETE; value present for ADD/UPDATE). NOOP and any
    malformed entry are dropped silently here — the worker treats an empty or
    short list as "nothing to write", which is the safe default. A wholly
    unparseable response raises ``ValueError`` so the caller can count it.

    The accepted envelope is ``{"ops": [ {op, kind?, key, value}, ... ]}``.
    """
    if not text or not text.strip():
        raise ValueError("empty fact-extraction response")

    body = _strip_code_fence(text.strip())
    parsed = json.loads(body)  # ValueError on malformed JSON — caller counts it.
    if not isinstance(parsed, dict):
        raise ValueError("fact-extraction response is not a JSON object")

    raw_ops = parsed.get("ops")
    if not isinstance(raw_ops, list):
        raise ValueError("fact-extraction response missing 'ops' list")

    return [op for op in (_clean_op(entry) for entry in raw_ops) if op is not None]


def _clean_op(entry: object) -> dict[str, object] | None:
    """Validate one op entry; return a normalised dict or None when unusable."""
    if not isinstance(entry, dict):
        return None
    op = str(entry.get("op", "")).strip().upper()
    if op not in VALID_OPS or op == OP_NOOP:
        return None
    key = (entry.get("key") or "").strip()
    if not key:
        return None
    value = (entry.get("value") or "").strip()
    if op in (OP_ADD, OP_UPDATE) and not value:
        return None
    return {"op": op, "kind": FACT_KIND, "key": key, "value": value}


def _strip_code_fence(text: str) -> str:
    """Remove a single markdown code-fence wrapper (```[lang]...```) from text."""
    if not text.startswith("```"):
        return text
    open_end = text.find("```")
    newline = text.find("\n", open_end)
    if newline == -1:
        return text
    close_start = text.rfind("```", newline)
    if close_start <= newline:
        return text
    return text[newline + 1 : close_start].strip()


class FactExtractionConfig(ProcessorConfig):
    """Per-episode fact-extraction config — one tool-free constrained LLM call.

    channel/role='fact_extraction', suppress_history=True, no tools, no memory
    seed (a tool-free request completes in one send). ``MessageProcessor.process('', config)`` returns
    the model's raw JSON text, which the worker parses with ``parse_fact_ops``.

    The episode gist and the pre-fetched neighbour facts are captured at
    construction so the prompt is self-contained per episode (the worker owns
    the retrieval + the DB writes — this config only frames the decision).
    """

    def __init__(self, gist: str, neighbours: list[object]) -> None:
        super().__init__(
            channel="fact_extraction",
            role="fact_extraction",
            policy_channel=ProcessorConfig.PolicyChannel.SUBCONSCIOUS,
            always_available=[],
            skip_transcript=True,
            skip_input_row=False,
            suppress_history=True,
            broadcast_to=None,
            memory_seed=False,
        )
        object.__setattr__(self, "_gist", gist)
        object.__setattr__(self, "_neighbours", neighbours)

    @property
    def system_prompt(self) -> str:
        return """The user is 'fact_extraction' — a background process that routes hard, truth-valued facts out of a single episode into the durable fact store, reconciling them against what is already known.

Extract the durable, truth-valued facts stated in the episode (names, dates, ownership, state — anything that can later be contradicted) and reconcile each against the known facts.

For each fact emit exactly one operation:
- ADD: a genuinely new fact not present in the known list.
- UPDATE: the episode changes the value of a known fact (reuse that fact's key verbatim; value is the new truth).
- DELETE: the episode says a known fact is no longer true (reuse that fact's key; value names the fact being removed).
- NOOP: the fact is already on record unchanged — omit it.

Rules:
- Output ONE JSON object and nothing else: {"ops": [ ... ]}.
- When the episode contains no durable facts, output {"ops": []}.
- Do NOT invent facts, narratives, or feelings. Skip anything without a truth value.
- Reuse an existing key EXACTLY when updating or deleting it.

Example:
{"ops": [{"op": "ADD", "key": "pet", "value": "dog named Rex"}, {"op": "DELETE", "key": "pet", "value": "cat named Tom"}]}"""
