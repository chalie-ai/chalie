#!/usr/bin/env python3
"""
Script to automate cross-linking in documentation markdown files.

This script:
1. Scans all .md files in docs/ directory
2. Injects inline links for referenced concepts
3. Appends a "## Related Documentation" footer section
"""

import os
import re
from pathlib import Path


# Mapping of concept keywords to their documentation file paths
CONCEPT_LINKS = {
    # Core documents
    "vision": ("00-VISION.md", "Chalie Vision & Philosophy"),
    "philosophy": ("00-VISION.md", "Product Design Principles"),
    "quick start": ("01-QUICK-START.md", "Quick Start Guide"),
    "installation": ("01-QUICK-START.md", "Installation Instructions"),
    "deployment": ("01-QUICK-START.md", "Deployment Guide"),
    
    # Setup & Configuration
    "providers setup": ("02-PROVIDERS-SETUP.md", "LLM Providers Configuration"),
    "llm providers": ("02-PROVIDERS-SETUP.md", "Configure LLM Providers"),
    "ollama": ("02-PROVIDERS-SETUP.md", "Ollama Setup"),
    "anthropic": ("02-PROVIDERS-SETUP.md", "Anthropic Configuration"),
    "openai": ("02-PROVIDERS-SETUP.md", "OpenAI Configuration"),
    "gemini": ("02-PROVIDERS-SETUP.md", "Gemini Setup"),
    
    # Interface
    "web interface": ("03-WEB-INTERFACE.md", "Web Interface Guide"),
    "ui setup": ("03-WEB-INTERFACE.md", "UI Configuration"),
    
    # Architecture & System
    "architecture": ("04-ARCHITECTURE.md", "System Architecture Overview"),
    "system architecture": ("04-ARCHITECTURE.md", "Complete System Architecture"),
    "services": ("04-ARCHITECTURE.md", "Services Overview"),
    
    # Workflow & Processing
    "workflow": ("05-WORKFLOW.md", "Workflow Guide"),
    "prompt processing": ("05-WORKFLOW.md", "Prompt Processing Flow"),
    "request pipeline": ("05-WORKFLOW.md", "Request Pipeline Steps"),
    
    # Workers
    "workers": ("06-WORKERS.md", "Workers Overview"),
    "worker processes": ("06-WORKERS.md", "Worker Processes Guide"),
    
    # Cognitive Architecture
    "cognitive architecture": ("07-COGNITIVE-ARCHITECTURE.md", "Cognitive Architecture"),
    "deterministic mode": ("07-COGNITIVE-ARCHITECTURE.md", "Deterministic Mode Router"),
    "decision flow": ("07-COGNITIVE-ARCHITECTURE.md", "Decision Flow Logic"),
    
    # Data & Schemas
    "data schemas": ("08-DATA-SCHEMAS.md", "Data Schemas Reference"),
    "memorystore": ("08-DATA-SCHEMAS.md", "MemoryStore Schema"),
    "sqlite schema": ("08-DATA-SCHEMAS.md", "SQLite Data Model"),
    
    # Tools & Extensions
    "tools": ("09-TOOLS.md", "Tools Guide"),
    "sandboxed tools": ("09-TOOLS.md", "Sandboxed Tool Execution"),
    "tool extensions": ("09-TOOLS.md", "Extending Chalie with Tools"),
    
    # Context & Relevance
    "context relevance": ("10-CONTEXT-RELEVANCE.md", "Context Relevance Guide"),
    "token optimization": ("10-CONTEXT-RELEVANCE.md", "Token Optimization Strategies"),
    "selective context injection": ("10-CONTEXT-RELEVANCE.md", "Selective Context Injection"),
    
    # Testing
    "testing": ("12-TESTING.md", "Testing Guide"),
    "test suite": ("12-TESTING.md", "Test Suite Overview"),
    
    # Message Flow
    "message flow": ("13-MESSAGE-FLOW.md", "Message Flow Diagrams"),
    "visual flow diagrams": ("13-MESSAGE-FLOW.md", "Visual Flow Documentation"),
    
    # Default Tools
    "default tools": ("14-DEFAULT-TOOLS.md", "Default Tools Reference"),
}

# List of all documentation files for the footer section
DOC_FILES = [
    ("00-VISION.md", "Vision & Philosophy"),
    ("01-QUICK-START.md", "Quick Start Guide"),
    ("02-PROVIDERS-SETUP.md", "LLM Providers Setup"),
    ("03-WEB-INTERFACE.md", "Web Interface"),
    ("04-ARCHITECTURE.md", "System Architecture"),
    ("05-WORKFLOW.md", "Workflow Guide"),
    ("06-WORKERS.md", "Workers Overview"),
    ("07-COGNITIVE-ARCHITECTURE.md", "Cognitive Architecture"),
    ("08-DATA-SCHEMAS.md", "Data Schemas"),
    ("09-TOOLS.md", "Tools & Extensions"),
    ("10-CONTEXT-RELEVANCE.md", "Context Relevance"),
    ("12-TESTING.md", "Testing Guide"),
    ("13-MESSAGE-FLOW.md", "Message Flow Diagrams"),
    ("14-DEFAULT-TOOLS.md", "Default Tools"),
]


def get_relative_path(current_file: str, target_file: str) -> str:
    """Get the relative path from current file to target file."""
    # Both files are in docs/ directory, so just return the filename
    return target_file


def protect_content(content: str) -> tuple[str, dict]:
    """
    Protect existing markdown elements by replacing them with placeholders.
    
    Returns the protected content and a dictionary mapping placeholder IDs to original text.
    This prevents cross-link injection from modifying links, code blocks, or URLs.
    """
    protected = {}
    counter = [0]  # Use list for mutable closure
    
    def make_placeholder():
        counter[0] += 1
        return f"\x00PROTECTED_{counter[0]}_END\x00"
    
    result = content
    
    # Protect fenced code blocks first (```...```) - multiline
    code_block_pattern = r'(```[\s\S]*?```)'
    for match in re.finditer(code_block_pattern, result):
        placeholder = make_placeholder()
        protected[placeholder] = match.group(0)
    
    # Protect inline links [text](url)
    link_pattern = r'\[[^\]]+\]\([^)]+\)'
    for match in re.finditer(link_pattern, result):
        placeholder = make_placeholder()
        protected[placeholder] = match.group(0)
    
    # Protect inline code `...` (but not inside already-protected regions)
    inline_code_pattern = r'(?<!`)`[^`]+`(?!`)'
    for match in re.finditer(inline_code_pattern, result):
        placeholder = make_placeholder()
        protected[placeholder] = match.group(0)
    
    # Protect raw URLs (http:// or https://)
    url_pattern = r'https?://[^\s<>"\]]+'
    for match in re.finditer(url_pattern, result):
        placeholder = make_placeholder()
        protected[placeholder] = match.group(0)
    
    return result, protected


def restore_content(content: str, protected: dict) -> str:
    """Restore original content from placeholders."""
    for placeholder, original in protected.items():
        content = content.replace(placeholder, original)
    return content


def inject_inline_links(content: str, current_filename: str) -> str:
    """Inject inline links for concept keywords in the content."""
    # Protect existing markdown elements first
    protected_content, placeholders = protect_content(content)
    
    result = protected_content
    
    # Sort concepts by length (longest first) to avoid partial replacements
    sorted_concepts = sorted(CONCEPT_LINKS.items(), key=lambda x: len(x[0]), reverse=True)
    
    for keyword, (target_file, link_text) in sorted_concepts:
        # Skip if this is the target file itself
        if current_filename == target_file:
            continue
            
        # Create regex pattern for word boundaries (case-insensitive)
        pattern = r'\b' + re.escape(keyword) + r'\b'
        
        # Replace with markdown link, preserving case of original text
        def replace_func(match):
            original_text = match.group(0)
            return f"[{original_text}]({get_relative_path(current_filename, target_file)})"
        
        result = re.sub(pattern, replace_func, result, flags=re.IGNORECASE)
    
    # Restore protected content (links, code blocks, URLs)
    result = restore_content(result, placeholders)
    
    return result


def generate_related_docs_footer(current_filename: str) -> str:
    """Generate the Related Documentation footer section."""
    lines = ["", "## Related Documentation"]
    
    for target_file, title in DOC_FILES:
        if target_file != current_filename:
            rel_path = get_relative_path(current_filename, target_file)
            lines.append(f"- [{title}]({rel_path})")
    
    return "\n".join(lines)


def process_markdown_file(filepath: Path) -> bool:
    """Process a single markdown file and apply cross-linking."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original_content = content
        
        # Inject inline links for concepts
        modified_content = inject_inline_links(content, filepath.name)
        
        # Append Related Documentation footer if not already present
        if "## Related Documentation" not in modified_content:
            modified_content += generate_related_docs_footer(filepath.name)
        
        # Only write if content changed
        if modified_content != original_content:
            filepath.write_text(modified_content, encoding='utf-8')
            print(f"Updated: {filepath}")
            return True
        else:
            print(f"No changes needed: {filepath}")
            return False
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main function to process all markdown files in docs/."""
    docs_dir = Path("docs")
    
    if not docs_dir.exists():
        print(f"Error: Directory '{docs_dir}' does not exist.")
        return
    
    # Find all .md files in docs/ (excluding subdirectories like images/)
    md_files = list(docs_dir.glob("*.md"))
    
    if not md_files:
        print(f"No markdown files found in '{docs_dir}'.")
        return
    
    print(f"Found {len(md_files)} markdown file(s) to process.")
    print("-" * 50)
    
    updated_count = 0
    for md_file in sorted(md_files):
        if process_markdown_file(md_file):
            updated_count += 1
    
    print("-" * 50)
    print(f"Processing complete. {updated_count} file(s) updated.")


if __name__ == "__main__":
    main()
