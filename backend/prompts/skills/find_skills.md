## `find_skills` — Skill Discovery

Discover which cognitive skills are available by describing what you need. Returns full documentation for matching skills so you can use them immediately.

### Usage
```json
{"type": "find_skills", "query": "I need to set a reminder for tomorrow"}
```
Parameters:
- `query` (required): Natural language description of what you want to do
- `limit` (optional): Max results (default 3, max 5)

### When to use
- You need a capability beyond recall/memorize/associate (the always-available primitives)
- You're unsure which skill handles a specific task
- You want to check what's available before choosing an approach

### Output
Returns full skill documentation for the best matches, ready for immediate use.
