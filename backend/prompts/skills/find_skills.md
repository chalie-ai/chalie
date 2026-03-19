## `find_skills` — Skill Discovery

Discover which cognitive skills are available by describing what you need. Returns full documentation for matching skills so you can use them immediately.

### Usage
```json
{"type": "find_skills", "query": "I need to set a reminder for tomorrow"}
```
Parameters:
- `query` (required): Natural language description of what you want to do
- `limit` (optional): Max results (default 3, max 5)

Use when you need further capability or are unsure what's available. Returns full skill documentation ready for immediate use.
