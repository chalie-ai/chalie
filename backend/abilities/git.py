"""GitAbility — read/write GitHub & GitLab repos via the Git capability.

Delegates all dispatch to GitCapability's tool handlers exactly like
ubiquiti.py delegates to UbiquitiCapability.
"""

import json
import logging

from abilities._base import Ability
from services.innate_skills._tag import tag as _skill_tag

logger = logging.getLogger(__name__)


class GitAbility(Ability):
    NAME = "git"
    SEARCH_TOOLTIP = "GitHub and GitLab repository operations"
    POLICY_CATEGORY = "Git"
    POLICY_LABELS = {
        # READ
        "get_repo": "Get repository info",
        "list_branches": "List branches",
        "list_commits": "List commits",
        "get_commit": "Get commit details",
        "list_prs": "List pull/merge requests",
        "get_pr": "Get PR/MR details",
        "list_issues": "List issues",
        "get_issue": "Get issue details",
        "list_releases": "List releases",
        "search_repos": "Search repositories",
        "read_file": "Read file from repo",
        "clone": "Clone repository",
        "diff": "Show git diff",
        # WRITE
        "commit": "Commit changes",
        "push": "Push to remote",
        "create_pr": "Create PR/MR",
        "merge_pr": "Merge PR/MR",
    }
    SUMMARY = (
        "Read public GitHub or GitLab repositories without a token; with a "
        "Personal Access Token, read private repos and write code — clone a repo, "
        "edit files, commit, push to a feature branch, open a pull/merge request, "
        "and merge it. Also list commits, PRs/MRs, issues, releases, and search repos."
    )
    EXAMPLES = [
        "Show me the open pull requests in owner/repo",
        "Clone owner/repo so I can edit the README",
        "What commits were made to main last week?",
        "List the open issues in that GitLab project",
        "Read the contents of src/main.py from owner/repo",
        "Commit my changes to a new branch called fix/typo",
        "Push this branch and open a PR targeting main",
        "Merge PR #42 in owner/repo",
    ]
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "get_repo",
                    "list_branches",
                    "list_commits",
                    "get_commit",
                    "list_prs",
                    "get_pr",
                    "list_issues",
                    "get_issue",
                    "list_releases",
                    "search_repos",
                    "read_file",
                    "clone",
                    "diff",
                    "commit",
                    "push",
                    "create_pr",
                    "merge_pr",
                ],
                "description": "The Git action to perform.",
            },
            # Repo identity (most actions)
            "owner": {
                "type": "string",
                "description": "Repository owner (user or organisation name).",
            },
            "repo": {
                "type": "string",
                "description": "Repository name.",
            },
            # Commit / branch params
            "sha": {
                "type": "string",
                "description": "Commit SHA (full or short) for get_commit.",
            },
            "branch": {
                "type": "string",
                "description": "Branch name for list_commits or clone.",
            },
            "ref": {
                "type": "string",
                "description": "Branch, tag, or commit ref for read_file or diff.",
            },
            # PR/issue params
            "pr_number": {
                "type": "integer",
                "description": "PR/MR number for get_pr or merge_pr.",
            },
            "issue_number": {
                "type": "integer",
                "description": "Issue number for get_issue.",
            },
            "state": {
                "type": "string",
                "enum": ["open", "closed", "all"],
                "description": "State filter for list_prs or list_issues.",
                "default": "open",
            },
            # Search
            "query": {
                "type": "string",
                "description": "Search keywords for search_repos.",
            },
            # File read
            "path": {
                "type": "string",
                "description": "File path within the repository for read_file.",
            },
            # Clone
            "full": {
                "type": "boolean",
                "description": "When true, perform a full clone instead of shallow (--depth 1).",
                "default": False,
            },
            # Workspace (diff / commit / push)
            "workspace": {
                "type": "string",
                "description": "Path to an existing cloned workspace for diff, commit, or push.",
            },
            # Diff
            "staged": {
                "type": "boolean",
                "description": "For diff: when true show staged changes (--cached).",
                "default": False,
            },
            # Commit
            "message": {
                "type": "string",
                "description": "Commit message for commit.",
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Explicit file paths to stage for commit (relative to workspace root).",
            },
            "new_branch": {
                "type": "string",
                "description": "For commit: create and switch to this branch before committing.",
            },
            # Push
            "remote_branch": {
                "type": "string",
                "description": "For push: remote branch name; defaults to the local branch name.",
            },
            # create_pr
            "title": {
                "type": "string",
                "description": "PR/MR title for create_pr.",
            },
            "head": {
                "type": "string",
                "description": "Source branch for create_pr.",
            },
            "base_branch": {
                "type": "string",
                "description": "Target branch for create_pr.",
            },
            "body": {
                "type": "string",
                "description": "PR/MR description for create_pr.",
            },
            # merge_pr
            "merge_message": {
                "type": "string",
                "description": "Optional merge commit message for merge_pr.",
            },
        },
        "required": ["action"],
    }
    TIMEOUT = 120  # clone and push can be slow on large repos

    def execute(self, channel: str, params: dict, telemetry: dict | None) -> dict | str:
        """Delegate to GitCapability.act() — the single source of name→handler dispatch."""
        action = params.get("action", "").lower()

        from capabilities import load_capabilities
        cap = load_capabilities().get("git")

        if cap is None:
            result = {
                "status": "error",
                "error": "Git capability not found. Configure it in the Brain dashboard.",
            }
            return {"text": _skill_tag("git", json.dumps(result), action=action)}

        if not cap.is_connected():
            result = {
                "status": "error",
                "error": "Git capability not connected. Check your settings in Brain → Capabilities.",
            }
            return {"text": _skill_tag("git", json.dumps(result), action=action)}

        # Strip internal bookkeeping keys before forwarding to the capability.
        action_params = {k: v for k, v in params.items() if not k.startswith("_") and k != "action"}
        result = cap.act(action, action_params)
        return {"text": _skill_tag("git", json.dumps(result), action=action)}
