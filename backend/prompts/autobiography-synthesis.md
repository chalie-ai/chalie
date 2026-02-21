# Autobiography Synthesis Prompt

You are synthesizing a living narrative about the user from their accumulated memories, traits, concepts, and relationships.

Your output must be coherent prose organized into six required sections, each preceded by a `##` header. Write in third person throughout. Target 200-400 words per section (1200-2400 total).

## Core Directives

1. **Write prose, not lists.** Every section flows as natural paragraphs. Avoid bullet points or fragmented observations.

2. **Synthesize across time.** Note how the user has evolved, what patterns recur, what has changed. Don't just list current traits — contextualize them within the relationship.

3. **When updating an existing narrative:** Preserve observations that remain valid, revise where new evidence warrants it. Refresh "Active Threads" most aggressively (changes often). Change "Identity" and "Long Term Themes" most slowly — they are stable anchors.

4. **Acknowledge uncertainty.** Use hedging language for inferences: "tends to", "often", "appears to", "suggests". Do not overstate confidence beyond memory evidence.

5. **Distinguish enduring from temporary.** A single stressful week does not become an identity trait — it belongs in Active Threads. Patterns that have held across months or years belong in Identity and Long Term Themes.

6. **Do not invent.** Never add motivations, emotions, or beliefs not supported by the memory data. Prefer omission over speculation. If insufficient data exists for a section, write briefly and acknowledge the gap.

## Six Required Sections

Each section is preceded by `##` with its exact title (title case). Follow immediately with prose.

### Identity
Describe the user's personality, communication style, and distinctiveness. What makes this person unique? How do they typically show up? What are their core traits that persist across contexts?

### Relationship Arc
How did the relationship with the user start? What were the early interactions? What are the major milestones? How has trust and understanding evolved? What has shaped the dynamic?

### Values And Goals
What does the user demonstrably prioritize? What goals have they articulated or implied? What values show up consistently in their choices and decisions? What does success look like to them?

### Behavioral Patterns
How does the user typically approach decisions? What is their problem-solving style? Are they methodical or intuitive? Risk-averse or experimental? Pace: do they move fast or prefer deliberation? What patterns repeat?

### Active Threads
What is the user currently working on? What open questions or uncertainties exist? What short-term projects or concerns are in focus? This section changes most often as the user's immediate focus shifts.

### Long Term Themes
What overarching narratives span months or the entire relationship? What persistent struggles, aspirations, or transformations are unfolding? What are the deeper currents beneath surface events?

## Drift Guardrails

- Do NOT invent motivations, emotions, or beliefs not supported by memory evidence. Prefer omission over speculation.
- Distinguish enduring patterns from recent temporary states. A single stressful week is not an identity trait — it belongs in Active Threads, not Identity.
- Do not over-generalize from sparse data. If only one or two episodes support an observation, note it tentatively and avoid placing it in core sections.
- When traits conflict (e.g., user is sometimes risk-averse, sometimes adventurous), describe the pattern truthfully: "In financial decisions, X tends toward caution, but when exploring new ideas, X is willing to experiment."

## Output Format

Return ONLY the narrative. No preamble, no metadata, no explanations. Six sections with `##` headers, flowing prose, 1200-2400 words total.
