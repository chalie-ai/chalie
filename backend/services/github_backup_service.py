"""GitHub Backup Service — store encrypted brain backups in a GitHub repo.

Uses the GitHub Contents API (no SDK dependency) to commit the backup
file directly to a private repository. Simple stopgap for off-machine
backup storage.

Auth via Personal Access Token (classic: repo scope, or fine-grained: contents:write).
"""

import base64
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"


class GitHubBackupService:

    def upload(
        self,
        file_path: Path,
        repo: str,
        token: str,
    ) -> dict:
        """Upload a backup file to a GitHub repo via the Contents API.

        Overwrites the file at backups/brain.chalie-backup on each upload.

        Args:
            file_path: Path to the .chalie-backup file.
            repo: GitHub repo in "owner/repo" format.
            token: GitHub Personal Access Token.

        Returns:
            {"status": "ok", "commit_url": ..., "file_size": ...}

        Raises:
            ValueError: On invalid input.
            RuntimeError: On GitHub API errors.
        """
        if "/" not in repo:
            raise ValueError(f"Invalid repo format: {repo!r} — expected 'owner/repo'")

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }

        owner, name = repo.split("/", 1)
        self._ensure_repo(headers, owner, name)

        content_b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        existing_sha = self._get_file_sha(headers, repo, "backups/brain.chalie-backup")

        body = {
            "message": f"Brain backup {stamp}",
            "content": content_b64,
            "branch": "main",
        }
        if existing_sha:
            body["sha"] = existing_sha

        resp = requests.put(
            f"{_GITHUB_API}/repos/{repo}/contents/backups/brain.chalie-backup",
            headers=headers,
            json=body,
            timeout=120,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Failed to upload backup ({resp.status_code}): {resp.text[:300]}"
            )

        commit_url = resp.json().get("commit", {}).get("html_url", "")

        return {
            "status": "ok",
            "commit_url": commit_url,
            "file_size": file_path.stat().st_size,
        }

    def test_connection(self, token: str) -> dict:
        """Verify a GitHub token works and return user info."""
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }
        resp = requests.get(f"{_GITHUB_API}/user", headers=headers, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"GitHub auth failed ({resp.status_code}): {resp.text[:200]}")
        data = resp.json()
        return {
            "username": data.get("login"),
            "name": data.get("name", ""),
        }

    def _ensure_repo(self, headers: dict, owner: str, name: str) -> None:
        """Check if repo exists; create as private if not."""
        resp = requests.get(
            f"{_GITHUB_API}/repos/{owner}/{name}", headers=headers, timeout=10
        )
        if resp.status_code == 200:
            return

        if resp.status_code == 404:
            logger.info(f"[GitHubBackup] Creating private repo {owner}/{name}")
            create_resp = requests.post(
                f"{_GITHUB_API}/user/repos",
                headers=headers,
                json={"name": name, "private": True, "description": "Chalie brain backups"},
                timeout=15,
            )
            if create_resp.status_code not in (200, 201):
                raise RuntimeError(
                    f"Failed to create repo ({create_resp.status_code}): {create_resp.text[:300]}"
                )
            return

        raise RuntimeError(f"GitHub repo check failed ({resp.status_code}): {resp.text[:200]}")

    def _get_file_sha(self, headers: dict, repo: str, path: str) -> str | None:
        """Get the SHA of an existing file, or None if it doesn't exist."""
        resp = requests.get(
            f"{_GITHUB_API}/repos/{repo}/contents/{path}",
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("sha")
        return None
