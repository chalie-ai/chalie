## `document` — Document Search and Management
Search, create, and manage documents (warranties, contracts, manuals, receipts, research notes, etc.).

Parameters:
- `action` (required): `"search"`, `"list"`, `"view"`, `"delete"`, `"restore"`, `"create"`
- `query` (required for search): Text to search across all documents
- `name` (required for create, optional for others): Document filename (e.g. `"research-notes.md"`)
- `content` (required for create): The text content to store in the document
- `id` (optional): Document ID (exact match)

Use when: User asks about information that might be in uploaded documents, wants to manage their document library, or asks you to save/store/write content as a document.

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
- "Save this as a document" → `{"type": "document", "action": "create", "name": "document-title.md", "content": "<the content>"}`
- "Store these research notes" → `{"type": "document", "action": "create", "name": "research-notes.md", "content": "<the notes>"}`
