## `document` — Document Search and Management
Search uploaded documents (warranties, contracts, manuals, receipts, etc.) and manage the document library.

Parameters:
- `action` (required): `"search"`, `"list"`, `"view"`, `"delete"`, `"restore"`
- `query` (required for search): Text to search across all documents
- `name` (optional): Document name (fuzzy matched)
- `id` (optional): Document ID (exact match)

Use when: User asks about information that might be in uploaded documents (warranties, contracts, policies, manuals, receipts, invoices), or wants to manage their document library.

Do NOT use when: The question is about general knowledge not related to any uploaded personal document.

### Two-phase retrieval (IMPORTANT)

When the user asks a question about their documents, always use two steps:

1. **Search** — identifies WHICH documents are relevant (returns names, types, IDs only — no content)
2. **View** — loads the full document text for analysis (pass the `id` from step 1)

Never try to answer from search results alone — search only tells you which documents exist. You must call view to read the actual content.

Common patterns:
- "Is my fridge under warranty?" →
  Step 1: `{"type": "document", "action": "search", "query": "fridge warranty"}`
  Step 2: `{"type": "document", "action": "view", "id": "<id from step 1>"}`
- "What documents do I have?" → `{"type": "document", "action": "list"}`
- "Show me the Samsung warranty" → `{"type": "document", "action": "view", "name": "Samsung warranty"}`
- "When does my insurance expire?" →
  Step 1: `{"type": "document", "action": "search", "query": "insurance expiration"}`
  Step 2: `{"type": "document", "action": "view", "id": "<id from step 1>"}`
- "Delete the old receipt" → `{"type": "document", "action": "delete", "name": "old receipt"}`
