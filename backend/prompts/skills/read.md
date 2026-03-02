## `read` — Read Content from URL or File
Read and extract clean text from a web page, PDF, Word document, PowerPoint, HTML file, Markdown, or plain text file. Pulls the full document — no truncation before content reaches you.

Parameters:
- `source` (required): URL (e.g. `"https://example.com/article"`) or filesystem path (e.g. `"/home/user/doc.pdf"`)
- `max_chars` (optional): Max characters to return (default 4000, max 8000)

Supported formats: web pages, PDF, DOCX, PPTX, HTML, Markdown, plain text, any text/* file.

Use when: User asks to read, open, fetch, or summarize a URL, webpage, article, or local file. Also use when a search or browse tool returns a promising URL and the snippet is insufficient.

Do NOT use when: Looking for information in uploaded documents already in the library (use `document` skill instead).

Browsing strategy for research:
1. Read the primary URL → check "Page links" in the result
2. Follow relevant links in subsequent `read` calls to go deeper
3. Use `memorize` to persist key findings

Common patterns:
- "Read this article" → `{"type": "read", "source": "https://example.com/article"}`
- "What does this page say?" → `{"type": "read", "source": "https://docs.example.com/guide"}`
- "What's in this PDF?" → `{"type": "read", "source": "/path/to/document.pdf"}`
- "Summarize this Word doc" → `{"type": "read", "source": "/path/to/report.docx"}`
- "Read the README" → `{"type": "read", "source": "/path/to/README.md"}`
