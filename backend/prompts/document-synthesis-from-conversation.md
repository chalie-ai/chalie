You are extracting a clean, standalone document from a conversation.

The user and assistant discussed creating a **{{content_type}}**. Your task is to extract ONLY the final version of this content and format it as a clean markdown document.

## Rules

- Output ONLY the document content — no conversation, no meta-commentary, no preamble
- Use proper markdown formatting (headers, lists, bold for emphasis)
- If the content evolved through multiple iterations, use the FINAL version only
- Add a descriptive title as an H1 header
- Keep the user's voice and choices intact — do not add or change content
- Do not wrap the output in code fences

## Conversation

{{conversation}}
