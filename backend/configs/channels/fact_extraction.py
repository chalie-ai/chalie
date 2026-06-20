from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from services.processor_config import ProcessorConfig

if TYPE_CHECKING:
    from services.message_processor import MessageProcessor

    class _WithFactAttrs:
        _gist: str
        _neighbours: list[object]

# ── Fact-extraction config (subconscious worker fact pipeline, TKT-925) ────────
#
# One small, tool-free LLM call per episode that turns the episode gist into a
# set of constrained data_graph operations (Mem0 arXiv:2504.19413). The model
# sees the episode and the top-N data_graph rows that are already most similar
# to it, then emits EXACTLY ONE JSON object describing the ops to apply.
#
# The interface is identical across every model tier (§4.6.5): the same prompt,
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

    def get_user_definition(self, mp: "MessageProcessor") -> str:
        return (
            "The user is 'fact_extraction' — a background process that routes "
            "hard, truth-valued facts out of a single episode into the durable "
            "fact store, reconciling them against what is already known."
        )

    def get_user_prompt(self, mp: "MessageProcessor") -> str:
        _self = cast("_WithFactAttrs", self)
        if _self._neighbours:
            known = "\n".join(
                f"- key={cast('dict[str, object]', n).get('key')!r} value={cast('dict[str, object]', n).get('value')!r}"
                for n in _self._neighbours
            )
        else:
            known = "(no similar facts on record)"
        return (
            f"Episode:\n{_self._gist}\n\n"
            f"Most similar known facts:\n{known}"
        )

    def get_system_prompt(self, mp: "MessageProcessor") -> str:
        """Constrained-op system prompt — identical for every model tier."""
        example = json.dumps(
            {
                "ops": [
                    {"op": OP_ADD, "key": "pet", "value": "dog named Rex"},
                    {"op": OP_DELETE, "key": "pet", "value": "cat named Tom"},
                ]
            }
        )
        return (
            f"{self.get_user_definition(mp)}\n\n"
            "Extract the durable, truth-valued facts stated in the episode "
            "(names, dates, ownership, state — anything that can later be "
            "contradicted) and reconcile each against the known facts.\n\n"
            "For each fact emit exactly one operation:\n"
            f"- {OP_ADD}: a genuinely new fact not present in the known list.\n"
            f"- {OP_UPDATE}: the episode changes the value of a known fact "
            "(reuse that fact's key verbatim; value is the new truth).\n"
            f"- {OP_DELETE}: the episode says a known fact is no longer true "
            "(reuse that fact's key; value names the fact being removed).\n"
            f"- {OP_NOOP}: the fact is already on record unchanged — omit it.\n\n"
            "Rules:\n"
            "- Output ONE JSON object and nothing else: "
            '{"ops": [ ... ]}.\n'
            "- When the episode contains no durable facts, output "
            '{"ops": []}.\n'
            "- Do NOT invent facts, narratives, or feelings. Skip anything "
            "without a truth value.\n"
            "- Reuse an existing key EXACTLY when updating or deleting it.\n\n"
            f"Example:\n{example}"
        )
