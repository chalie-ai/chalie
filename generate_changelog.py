#!/usr/bin/env python3
"""Generate CHANGELOG.md from git history following Keep a Changelog standard."""

import subprocess
import re
from collections import defaultdict


def get_git_log():
    """Retrieve recent commits from git log."""
    result = subprocess.run(
        ['git', 'log', '--pretty=format:%h|%s'],
        capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return []
    
    lines = result.stdout.strip().split('\n')
    commits = []
    for line in lines:
        if '|' in line:
            hash_, msg = line.split('|', 1)
            commits.append((hash_.strip(), msg.strip()))
    return commits


def categorize_commit(msg):
    """Categorize commit message into Keep a Changelog categories."""
    # Skip merge commits
    if msg.startswith('Merge'):
        return None
    
    # Normalize message for matching
    lower_msg = msg.lower()
    
    # Check for conventional commit prefixes or keywords
    if any(x in lower_msg for x in ['feat:', 'feature:', '+', 'add ', 'added']):
        return 'Added'
    elif any(x in lower_msg for x in ['fix:', 'bugfix:', 'patch:', 'repair', 'correct']):
        return 'Fixed'
    elif any(x in lower_msg for x in ['chore:', 'refactor:', 'update:', 'change ', 'changed', 'improve', 'purge', 'switch', 'expand', 'remove', 'delete']):
        return 'Changed'
    elif any(x in lower_msg for x in ['docs:', 'documentation:']):
        return 'Documentation'
    elif any(x in lower_msg for x in ['perf:', 'performance:', 'optimize']):
        return 'Performance'
    else:
        # Default to Changed if no clear category
        return 'Changed'


def clean_commit_message(msg):
    """Clean and format commit message for changelog."""
    # Remove conventional commit prefixes like "fix:", "feat:", etc.
    msg = re.sub(r'^[a-z]+:\s*', '', msg, flags=re.IGNORECASE)
    
    # Handle branch/task patterns: extract description after the last slash if present
    # e.g., "create-faq/fix-1-create-faq-file" -> "fix-1-create-faq-file"
    # But only if it looks like a task pattern (contains word-number-word pattern)
    if '/' in msg and re.search(r'[\w\-]+/\d+|[\w\-]+/\w+-\d+', msg):
        parts = msg.rsplit('/', 1)
        msg = parts[1]
    
    # Remove task numbering patterns like "fix-1-", "feat-2-" at the start
    msg = re.sub(r'^\w+-\d+\s*-\s*', '', msg, flags=re.IGNORECASE)
    
    # Clean up any leading/trailing whitespace and dashes
    msg = msg.strip()
    if msg.startswith('- '):
        msg = msg[2:].strip()
    
    # Capitalize first letter for readability (unless it's already a sentence or acronym)
    if msg and not re.match(r'^[A-Z]', msg) and len(msg) > 0:
        msg = msg[0].upper() + msg[1:]
    
    return msg


def generate_changelog(commits):
    """Generate changelog content from commits."""
    categories = defaultdict(list)
    seen_messages = set()
    
    for hash_, msg in commits:
        category = categorize_commit(msg)
        if not category:
            continue
            
        formatted_msg = clean_commit_message(msg)
        
        # Skip empty messages after formatting
        if not formatted_msg.strip():
            continue
        
        # Avoid duplicates
        if formatted_msg in seen_messages:
            continue
        seen_messages.add(formatted_msg)
        
        categories[category].append(formatted_msg)
    
    # Build changelog content
    lines = [
        "# Changelog",
        "",
        "All notable changes to this project will be documented in this file.",
        "",
        "[Unreleased]",
        ""
    ]
    
    # Define order of categories
    category_order = ['Added', 'Changed', 'Fixed', 'Documentation', 'Performance']
    
    for category in category_order:
        if category in categories and categories[category]:
            lines.append(f"### {category}")
            lines.append("")
            for msg in categories[category]:
                lines.append(f"- {msg}")
            lines.append("")
    
    return '\n'.join(lines)


def main():
    commits = get_git_log()
    changelog_content = generate_changelog(commits)
    
    with open('CHANGELOG.md', 'w') as f:
        f.write(changelog_content)
    
    print("CHANGELOG.md generated successfully!")


if __name__ == '__main__':
    main()
