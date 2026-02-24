# Context Relevance Pre-Parser

## Overview

The **Context Relevance Pre-Parser** is a deterministic, rule-based service that optimizes context injection by selectively excluding irrelevant context nodes based on the current cognitive mode and conversation signals. This reduces unnecessary I/O, token usage, and latency without sacrificing response quality.

## Motivation

Previously, **every response generation** retrieved and injected ALL context nodes (episodic memory, identity, user traits, facts, gists, focus, tools, skills, etc.) into every prompt — regardless of whether the mode-specific template even referenced them.

### Example Waste
An ACKNOWLEDGE for "Hey!" would trigger:
- PostgreSQL vector search for episodic memory
- Redis reads for facts, gists, working memory
- Skill registry queries

None of which the ACKNOWLEDGE template even uses.

### Expected Savings
| Mode | I/O Skipped | Token Savings |
|------|-------------|---------------|
| ACKNOWLEDGE | 5 Redis reads, 1 PG vector search, skill queries | ~1500-3000 |
| CLARIFY (warm) | 1 PG vector search, skill queries | ~500-1500 |
| RESPOND (greeting) | 1 PG vector search, focus queries | ~800-2000 |
| ACT | Identity/trait lookups | ~300-800 |

**Pre-parser execution**: < 0.5ms (pure dict lookups).

## Architecture

### Seven-Layer Pipeline

The service applies seven layers in order, each gating context node inclusion:

1. **Template Masks** — Static per-mode. Excludes nodes the template doesn't reference (hard exclusion).
2. **Signal Rules** — Conditional. Excludes nodes that the template references but are irrelevant given current signals. Each rule specifies `strength: "hard"` or `"soft"`.
3. **Urgency Overrides** — When `classification.urgency == 'high'`, force-include `working_memory`, `world_state`, `facts` for broader awareness.
4. **Soft Exclusion Recovery** — Soft-excluded nodes get re-included if token budget has headroom, in configurable priority order.
5. **Dependency Resolution** — If a node is included, its dependencies are auto-included (e.g., including `episodic_memory` auto-includes `gists`).
6. **Safety Overrides** — Force-includes nodes that must always be present under certain conditions (e.g., identity when returning from silence).
7. **Safeguard** — `MAX_INCLUDED_NODES = 12`. Logs warning if exceeded to protect prompt integrity.

### Context Nodes

All context nodes supported:
- `identity_context`
- `onboarding_nudge`
- `user_traits`
- `communication_style`
- `active_lists`
- `client_context`
- `focus`
- `working_memory`
- `facts`
- `gists`
- `episodic_memory`
- `act_history`
- `available_skills`
- `available_tools`
- `world_state`
- `warm_return_hint`
- `identity_modulation`

## Configuration

### File Location
`backend/configs/agents/context-relevance.json`

### Configuration Structure

#### Template Masks
Static per-mode inclusion decisions. Excludes nodes the template doesn't even reference.

```json
{
  "template_masks": {
    "RESPOND": {
      "episodic_memory": true,
      "working_memory": true,
      "facts": true,
      ...
    },
    "ACKNOWLEDGE": {
      "episodic_memory": false,
      "working_memory": false,
      ...
    }
  }
}
```

#### Signal Rules
Conditional exclusion rules with strength levels:

```json
{
  "signal_rules": {
    "episodic_memory": [
      {
        "when": {
          "context_warmth_gte": 0.5,
          "working_memory_turns_gte": 2
        },
        "strength": "soft"
      },
      {
        "when": {
          "greeting_pattern": true,
          "prompt_token_count_lt": 6
        },
        "strength": "hard"
      }
    ]
  }
}
```

**Predicates**:
- Exact match: `"key": value`
- Comparisons: `"key_gte": threshold`, `"key_gt"`, `"key_lte"`, `"key_lt"`, `"key_eq"`
- Special: `"returning_from_silence": true/false`

**Strengths**:
- `"hard"` — Never recovered, even with budget
- `"soft"` — Recoverable if token budget has headroom

#### Dependencies
Dependency graph; if a child is included, parents auto-include:

```json
{
  "dependencies": {
    "episodic_memory": ["gists"],
    "available_tools": ["available_skills"],
    "onboarding_nudge": ["identity_context"],
    "warm_return_hint": ["identity_context"]
  }
}
```

#### Urgency Overrides
Force-include critical nodes when urgent:

```json
{
  "urgency_overrides": ["working_memory", "world_state", "facts"]
}
```

#### Safety Overrides
Force-include under specific conditions:

```json
{
  "safety_overrides": {
    "identity_context": [
      { "when": { "returning_from_silence": true } },
      { "when": { "context_warmth_lt": 0.3 } }
    ],
    "working_memory": [
      { "when": { "working_memory_turns_gte": 1 } }
    ]
  }
}
```

#### Recovery Parameters
- `soft_recovery_budget` (default: 1500) — Token headroom threshold for re-including soft-excluded nodes
- `soft_recovery_priority` (default: listed order) — Priority order for soft recovery

```json
{
  "soft_recovery_budget": 1500,
  "soft_recovery_priority": [
    "episodic_memory", "working_memory", "world_state", "facts",
    "active_lists", "focus", "gists"
  ]
}
```

## Usage

### In Digest Worker

The service is invoked in `digest_worker.py` before response generation:

```python
from services.context_relevance_service import ContextRelevanceService

# Compute inclusion map
context_relevance_service = ContextRelevanceService()
inclusion_map = context_relevance_service.compute_inclusion_map(
    mode='RESPOND',                    # cognitive mode
    signals=signals,                   # routing signals
    classification=classification,     # topic classification
    returning_from_silence=returning_from_silence,
    token_budget_remaining=4000        # estimated tokens left
)

# Pass to cortex service
response_data = cortex_service.generate_response(
    system_prompt_template=prompt,
    original_prompt=text,
    classification=classification,
    chat_history=chat_history,
    inclusion_map=inclusion_map,       # ← KEY: gates context retrieval
    ...
)
```

### In Frontal Cortex Service

The `generate_response()` and `_inject_parameters()` methods gate context retrieval based on `inclusion_map`:

```python
def _inject_parameters(self, template, ..., inclusion_map=None):
    _include = lambda node: (inclusion_map or {}).get(node, True)

    # Only submit futures for included nodes
    if _include('gists'):
        futures[executor.submit(...)] = 'gists'
    if _include('episodic_memory'):
        futures[executor.submit(...)] = 'episodes'
    ...

    # Only inject placeholders for included nodes
    result = result.replace('{{episodic_memory}}', episodic_context if _include('episodic_memory') else '')
    result = result.replace('{{facts}}', facts_context if _include('facts') else '')
    ...
```

**Backward Compatibility**: `inclusion_map=None` defaults to include everything (current behavior).

## Observability

### Structured Logging

Every context relevance computation logs a structured entry:

```
[CONTEXT RELEVANCE] mode=CLARIFY | excluded_hard=[focus, available_skills, available_tools, warm_return_hint] |
excluded_soft=[episodic_memory] | recovered_soft=[] | deps_added=[] |
overrides_applied=[urgency] | total_included=9 | est_tokens=2100
```

Fields:
- `mode` — Cognitive mode
- `excluded_hard` — Hard-excluded nodes (never recovered)
- `excluded_soft` — Soft-excluded nodes (recoverable)
- `recovered_soft` — Soft-excluded nodes that were recovered due to budget
- `deps_added` — Dependencies auto-included
- `overrides_applied` — Overrides applied (urgency, safety)
- `total_included` — Total included nodes
- `est_tokens` — Estimated tokens for included nodes

### Warnings

- **MAX_INCLUDED_NODES exceeded**: Logs warning if total included nodes > 12
- **Circular dependencies**: Raises `ConfigError` at config load time
- **Config load failure**: Falls back to "include all" with warning

## Testing

Comprehensive unit tests cover:
- Template mask correctness per mode
- Signal rule triggers (warm clarify excludes episodic, greeting excludes episodic)
- Soft vs hard exclusion behavior
- Dependency graph resolution with circular detection
- Soft recovery priority ordering
- Urgency overrides (force-include when urgent)
- Safety overrides (returning_from_silence forces identity)
- MAX_INCLUDED_NODES safeguard
- Edge cases (unknown mode, missing signals, disabled config)

Run tests:
```bash
pytest backend/tests/test_context_relevance_service.py -v
```

## Configuration Tuning

### Mode-Specific Optimization

Adjust template masks per mode to match your mode-specific templates:

```json
{
  "template_masks": {
    "CUSTOM_MODE": {
      "episodic_memory": false,
      "facts": true,
      ...
    }
  }
}
```

### Signal-Driven Exclusion

Add new signal rules to exclude context for specific conversation patterns:

```json
{
  "signal_rules": {
    "focus": [
      {
        "when": { "greeting_pattern": true },
        "strength": "soft"
      }
    ]
  }
}
```

### Budget-Aware Recovery

Adjust soft recovery budget based on token model limits:

```json
{
  "soft_recovery_budget": 2000  // Increase headroom for lower-token models
}
```

### Custom Dependencies

Define new dependency relationships:

```json
{
  "dependencies": {
    "new_node": ["existing_node"]
  }
}
```

## Implementation Details

### Service Class
- `backend/services/context_relevance_service.py`
- `ContextRelevanceService` — Main service class
- `compute_inclusion_map()` — Core method (returns `{node: True/False}`)

### Config File
- `backend/configs/agents/context-relevance.json` — Configuration

### Integration Points
- `backend/workers/digest_worker.py` — Calls service before `generate_for_mode()`
- `backend/services/frontal_cortex_service.py` — Uses `inclusion_map` in `_inject_parameters()`

## Disabling the Feature

To disable context relevance pre-parsing entirely, set in config:

```json
{
  "enabled": false
}
```

All context nodes will be included (current behavior). Useful for debugging or when minimal optimization is needed.

## Future Enhancements

- **Machine learning-based rules** — Learn signal-to-exclusion mappings from interaction data
- **Per-user config** — Different rules per user based on communication patterns
- **Dynamic token budget** — Estimate remaining tokens from prompt + mode
- **A/B testing framework** — Compare responses with/without context relevance pre-parsing
