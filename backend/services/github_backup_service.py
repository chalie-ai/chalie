"""GitHub Backup Service — upload encrypted brain backups to GitHub Releases.

Uses the GitHub REST API (no SDK dependency) to:
  1. Verify the target repo exists (create as private if not)
  2. Create a GitHub Release tagged brain-YYYY-MM-DD-HHMMSS
  3. Upload the .chalie-backup file as a release asset

Auth via Personal Access Token (classic: repo scope, or fine-grained: contents:write).
"""

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
        """Upload a backup file to a GitHub Release.

        Args:
            file_path: Path to the .chalie-backup file.
            repo: GitHub repo in "owner/repo" format.
            token: GitHub Personal Access Token.

        Returns:
            {"status": "ok", "release_url": ..., "asset_name": ..., "asset_size": ..., "tag": ...}

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

        tag = "brain-" + datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
        filename = file_path.name

        release = self._create_release(headers, repo, tag, filename)
        upload_url_template = release["upload_url"].split("{")[0]

        self._upload_asset(
            headers, upload_url_template, filename, file_path
        )

        return {
            "status": "ok",
            "release_url": release["html_url"],
            "asset_name": filename,
            "asset_size": file_path.stat().st_size,
            "tag": tag,
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

    def _ensure_repo(self, headers: dict, owner: str, name: str) -> str:
        """Check if repo exists; create as private if not. Returns repo URL."""
        resp = requests.get(
            f"{_GITHUB_API}/repos/{owner}/{name}", headers=headers, timeout=10
        )
        if resp.status_code == 200:
            return resp.json()["html_url"]

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
            return create_resp.json()["html_url"]

        raise RuntimeError(f"GitHub repo check failed ({resp.status_code}): {resp.text[:200]}")

    def _create_release(self, headers: dict, repo: str, tag: str, filename: str) -> dict:
        """Create a GitHub Release and return its JSON."""
        resp = requests.post(
            f"{_GITHUB_API}/repos/{repo}/releases",
            headers=headers,
            json={
                "tag_name": tag,
                "name": f"Brain Backup {tag}",
                "body": f"Automated Chalie brain backup.\n\nAsset: `{filename}`",
                "draft": False,
                "prerelease": False,
            },
            timeout=15,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Failed to create release ({resp.status_code}): {resp.text[:300]}")
        return resp.json()

    def _upload_asset(
        self, headers: dict, upload_url: str, filename: str, file_path: Path
    ) -> str:
        """Upload a file as a release asset. Returns the asset browser URL."""
        size = file_path.stat().st_size
        upload_headers = {
            "Authorization": headers["Authorization"],
            "Content-Type": "application/octet-stream",
        }
        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{upload_url}?name={filename}&size={size}",
                headers=upload_headers,
                data=f,
                timeout=300,
            )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Asset upload failed ({resp.status_code}): {resp.text[:300]}")
        return resp.json().get("browser_download_url", "")
