#!/usr/bin/env python3
"""
SEO Optimization Script for Chalie Documentation

This script dynamically identifies all 15 documentation files in the docs/ directory,
updates H1 titles to be keyword-rich and search-query optimized, and inserts an
SEO-optimized introductory paragraph with relevant keywords and cross-links.
"""

import os
import re
from pathlib import Path


def get_doc_files():
    """Dynamically identify all Markdown files in the docs/ directory."""
    docs_dir = Path("docs")
    if not docs_dir.exists():
        print(f"Error: {docs_dir} directory not found.")
        return []
    
    md_files = sorted(docs_dir.glob("*.md"))
    return md_files


def generate_keyword_rich_title(original_title, filename):
    """Generate a keyword-rich H1 title based on the original and filename."""
    # Remove markdown formatting from original title
    clean_title = re.sub(r'^#+\s*', '', original_title).strip()
    
    # Map filenames to SEO keywords
    seo_keywords = {
        "INDEX.md": "Chalie Documentation Index - Complete Guide & Reference",
        "00-VISION.md": "Chalie Vision & Philosophy - Product Design Principles & Cognitive Assistant Goals",
        "01-QUICK-START.md": "Chalie Quick Start Guide - Installation, Setup & Deployment Instructions",
        "02-PROVIDERS-SETUP.md": "Chalie LLM Providers Setup - Configure Ollama, Anthropic, OpenAI, Gemini API",
        "03-WEB-INTERFACE.md": "Chalie Web Interface Documentation - UI Requirements & Frontend Architecture",
        "04-ARCHITECTURE.md": "Chalie System Architecture - Services, Workers, Data Flow & Memory Hierarchy",
        "05-WORKFLOW.md": "Chalie Request Workflow - Step-by-Step Prompt Processing Pipeline Guide",
        "06-WORKERS.md": "Chalie Worker Processes Overview - Background Tasks & Service Management",
        "07-COGNITIVE-ARCHITECTURE.md": "Chalie Cognitive Architecture - Mode Router, Decision Flow & AI Cognition",
        "08-DATA-SCHEMAS.md": "Chalie Data Schemas Documentation - MemoryStore & SQLite Database Structures",
        "09-TOOLS.md": "Chalie Tools System Guide - Create Custom Tools & Sandbox Extensions",
        "10-CONTEXT-RELEVANCE.md": "Chalie Context Relevance Pre-parser - Token Optimization & Selective Injection",
        "12-TESTING.md": "Chalie Testing Documentation - Test Conventions, Fixtures & Mock Strategies",
        "13-MESSAGE-FLOW.md": "Chalie Message Flow Diagrams - Visual Paths, MemoryStore Hits & LLM Calls",
        "14-DEFAULT-TOOLS.md": "Chalie Default Tools Reference - Auto-install Behavior & Built-in Toolset"
    }
    
    filename_upper = filename
    if filename_upper in seo_keywords:
        return f"# {seo_keywords[filename_upper]}"
    
    # Fallback: enhance with generic keywords
    enhanced_title = clean_title
    if "guide" not in enhanced_title.lower():
        enhanced_title += " Guide"
    if "chalie" not in enhanced_title.lower():
        enhanced_title = f"Chalie {enhanced_title}"
    
    return f"# {enhanced_title}"


def generate_seo_intro_paragraph(filename, all_doc_files):
    """Generate an SEO-optimized introductory paragraph with keywords and cross-links."""
    
    # Define related docs for each file (2-3 links)
    related_docs = {
        "INDEX.md": [
            ("01-QUICK-START.md", "Quick Start Guide"),
            ("04-ARCHITECTURE.md", "System Architecture Overview"),
            ("00-VISION.md", "Chalie Vision & Philosophy")
        ],
        "00-VISION.md": [
            ("INDEX.md", "Documentation Index"),
            ("01-QUICK-START.md", "Quick Start Guide"),
            ("04-ARCHITECTURE.md", "System Architecture Overview")
        ],
        "01-QUICK-START.md": [
            ("INDEX.md", "Documentation Index"),
            ("02-PROVIDERS-SETUP.md", "LLM Providers Configuration"),
            ("03-WEB-INTERFACE.md", "Web Interface Setup")
        ],
        "02-PROVIDERS-SETUP.md": [
            ("01-QUICK-START.md", "Quick Start Guide"),
            ("04-ARCHITECTURE.md", "System Architecture Overview"),
            ("09-TOOLS.md", "Tools System Documentation")
        ],
        "03-WEB-INTERFACE.md": [
            ("INDEX.md", "Documentation Index"),
            ("01-QUICK-START.md", "Quick Start Guide"),
            ("04-ARCHITECTURE.md", "System Architecture Overview")
        ],
        "04-ARCHITECTURE.md": [
            ("INDEX.md", "Documentation Index"),
            ("05-WORKFLOW.md", "Request Processing Workflow"),
            ("07-COGNITIVE-ARCHITECTURE.md", "Cognitive Architecture Guide")
        ],
        "05-WORKFLOW.md": [
            ("04-ARCHITECTURE.md", "System Architecture Overview"),
            ("13-MESSAGE-FLOW.md", "Message Flow Diagrams"),
            ("07-COGNITIVE-ARCHITECTURE.md", "Cognitive Architecture Guide")
        ],
        "06-WORKERS.md": [
            ("04-ARCHITECTURE.md", "System Architecture Overview"),
            ("05-WORKFLOW.md", "Request Processing Workflow"),
            ("09-TOOLS.md", "Tools System Documentation")
        ],
        "07-COGNITIVE-ARCHITECTURE.md": [
            ("04-ARCHITECTURE.md", "System Architecture Overview"),
            ("05-WORKFLOW.md", "Request Processing Workflow"),
            ("13-MESSAGE-FLOW.md", "Message Flow Diagrams")
        ],
        "08-DATA-SCHEMAS.md": [
            ("04-ARCHITECTURE.md", "System Architecture Overview"),
            ("12-TESTING.md", "Testing Documentation"),
            ("INDEX.md", "Documentation Index")
        ],
        "09-TOOLS.md": [
            ("04-ARCHITECTURE.md", "System Architecture Overview"),
            ("14-DEFAULT-TOOLS.md", "Default Tools Reference"),
            ("02-PROVIDERS-SETUP.md", "LLM Providers Configuration")
        ],
        "10-CONTEXT-RELEVANCE.md": [
            ("05-WORKFLOW.md", "Request Processing Workflow"),
            ("04-ARCHITECTURE.md", "System Architecture Overview"),
            ("12-TESTING.md", "Testing Documentation")
        ],
        "12-TESTING.md": [
            ("INDEX.md", "Documentation Index"),
            ("08-DATA-SCHEMAS.md", "Data Schemas Reference"),
            ("09-TOOLS.md", "Tools System Documentation")
        ],
        "13-MESSAGE-FLOW.md": [
            ("04-ARCHITECTURE.md", "System Architecture Overview"),
            ("05-WORKFLOW.md", "Request Processing Workflow"),
            ("07-COGNITIVE-ARCHITECTURE.md", "Cognitive Architecture Guide")
        ],
        "14-DEFAULT-TOOLS.md": [
            ("09-TOOLS.md", "Tools System Documentation"),
            ("INDEX.md", "Documentation Index"),
            ("02-PROVIDERS-SETUP.md", "LLM Providers Configuration")
        ]
    }
    
    # Define SEO keywords for each file
    seo_keywords_map = {
        "INDEX.md": ["Chalie documentation", "cognitive assistant guide", "AI system reference"],
        "00-VISION.md": ["Chalie vision", "product philosophy", "design principles", "cognitive AI goals"],
        "01-QUICK-START.md": ["Chalie installation", "quick start tutorial", "deployment guide", "setup instructions"],
        "02-PROVIDERS-SETUP.md": ["LLM provider configuration", "Ollama setup", "Anthropic API", "OpenAI integration", "Gemini API"],
        "03-WEB-INTERFACE.md": ["Chalie web UI", "frontend interface", "user dashboard", "web application design"],
        "04-ARCHITECTURE.md": ["system architecture", "service overview", "data flow pipeline", "memory hierarchy"],
        "05-WORKFLOW.md": ["request workflow", "prompt processing", "step-by-step guide", "processing pipeline"],
        "06-WORKERS.md": ["worker processes", "background tasks", "service management", "task scheduling"],
        "07-COGNITIVE-ARCHITECTURE.md": ["cognitive architecture", "mode router", "decision flow", "AI cognition system"],
        "08-DATA-SCHEMAS.md": ["data schemas", "MemoryStore structure", "SQLite database", "data models"],
        "09-TOOLS.md": ["Chalie tools", "custom tool creation", "sandbox extensions", "tool architecture"],
        "10-CONTEXT-RELEVANCE.md": ["context relevance", "token optimization", "selective context injection", "performance tuning"],
        "12-TESTING.md": ["testing guide", "test conventions", "mock strategies", "unit testing framework"],
        "13-MESSAGE-FLOW.md": ["message flow diagrams", "visual paths", "MemoryStore hits", "LLM call tracking"],
        "14-DEFAULT-TOOLS.md": ["default tools", "built-in toolset", "auto-install behavior", "pre-installed tools"]
    }
    
    filename_upper = filename
    keywords = seo_keywords_map.get(filename_upper, ["Chalie documentation", "technical guide"])
    links = related_docs.get(filename_upper, [])
    
    # Build keyword phrase
    keyword_phrase = ", ".join(keywords[:3])
    
    # Build cross-links
    link_text = ""
    for i, (link_file, link_label) in enumerate(links):
        if i > 0:
            link_text += " | "
        link_text += f"[{link_label}](../docs/{link_file})" if not filename_upper == "INDEX.md" else f"[{link_label}]({link_file})"
    
    # Generate intro paragraph
    intro = (f"This comprehensive guide covers {keyword_phrase}, providing essential information for developers and users. "
             f"For related topics, see: {link_text}")
    
    return intro


def update_documentation_file(filepath):
    """Update a single documentation file with SEO optimizations."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        filename = os.path.basename(filepath)
        
        # Find and replace H1 title (first line starting with # )
        lines = content.split('\n')
        if not lines:
            return False, "Empty file"
        
        new_lines = []
        h1_replaced = False
        intro_inserted = False
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            
            # Check if this is the H1 header (first line with single #)
            if not h1_replaced and re.match(r'^#\s+.+$', line.strip()):
                # Generate new keyword-rich title
                new_title = generate_keyword_rich_title(line, filename)
                
                # Replace the H1 line
                new_lines[-1] = new_title
                
                # Generate SEO intro paragraph
                all_files = [os.path.basename(f.name) for f in get_doc_files()]
                seo_intro = generate_seo_intro_paragraph(filename, all_files)
                
                # Insert intro paragraph after H1 (add blank line before and after)
                new_lines.append("")  # Blank line after H1
                new_lines.append(seo_intro)
                new_lines.append("")  # Blank line after intro
                
                h1_replaced = True
                intro_inserted = True
        
        if not h1_replaced:
            return False, "No H1 header found"
        
        # Write updated content back to file
        new_content = '\n'.join(new_lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True, "Updated successfully"
    
    except Exception as e:
        return False, str(e)


def main():
    """Main function to run SEO optimization on all documentation files."""
    print("=" * 60)
    print("Chalie Documentation SEO Optimization Script")
    print("=" * 60)
    
    # Get all doc files
    doc_files = get_doc_files()
    
    if not doc_files:
        print("No documentation files found in docs/ directory.")
        return
    
    print(f"\nFound {len(doc_files)} Markdown files to optimize:\n")
    
    success_count = 0
    fail_count = 0
    
    for filepath in doc_files:
        filename = filepath.name
        print(f"Processing: {filename}... ", end="")
        
        success, message = update_documentation_file(filepath)
        
        if success:
            print("✓ Updated")
            success_count += 1
        else:
            print(f"✗ Failed ({message})")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"SEO Optimization Complete!")
    print(f"Successfully updated: {success_count} files")
    print(f"Failed: {fail_count} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
