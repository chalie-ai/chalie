#!/usr/bin/env python3
"""
Script to remove corrupted inline markdown links from docs/*.md files.

This script searches for patterns like [word](NN-NAME.md) in the main body of documents
and replaces them with just the word, leaving "See also" intros and footers untouched.
"""

import os
import re
from pathlib import Path


def process_file(file_path: str) -> bool:
    """
    Process a single markdown file to remove corrupted inline links from the body only.
    
    Returns True if changes were made, False otherwise.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to match corrupted inline links in the main body
    # Matches [word](NN-NAME.md) where word is alphabetic and link has 2 digits followed by uppercase name
    pattern = r'\[([a-zA-Z]+)\]\(([0-9]{2}-[A-Z-]+\.md)\)'
    
    lines = content.split('\n')
    modified_lines = []
    
    in_footer = False
    
    for line in lines:
        # Detect footer (## Related Documentation header) - everything after this is protected
        if re.match(r'^##\s*Related\s+Documentation\s*$', line):
            in_footer = True
        
        # If we're not in a protected section, process the line
        if not in_footer:
            # Check if this line contains "See also" - protect entire line if so
            if 'See also' in line:
                modified_lines.append(line)  # Keep unchanged
            else:
                # Replace corrupted links with just the word
                def replace_link(match):
                    return match.group(1)  # Return just the captured word
                
                modified_line = re.sub(pattern, replace_link, line)
                modified_lines.append(modified_line)
        else:
            # Keep footer sections unchanged
            modified_lines.append(line)
    
    new_content = '\n'.join(modified_lines)
    
    if content != new_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    
    return False


def main():
    """Main function to process all markdown files in docs/ directory."""
    docs_dir = Path('docs')
    
    if not docs_dir.exists():
        print(f"Error: {docs_dir} directory does not exist")
        return 1
    
    md_files = list(docs_dir.glob('*.md'))
    
    if not md_files:
        print("No markdown files found in docs/ directory")
        return 0
    
    modified_count = 0
    
    for file_path in sorted(md_files):
        if process_file(str(file_path)):
            print(f"Modified: {file_path}")
            modified_count += 1
        else:
            print(f"No changes: {file_path}")
    
    print(f"\nTotal files modified: {modified_count}/{len(md_files)}")
    return 0


if __name__ == '__main__':
    exit(main())
