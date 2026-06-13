"""Brain blueprint — /brain/info, /brain/export, /brain/import, /brain/export/upload.

Exports and imports the entire Chalie brain (SQLite DB + files) as an
encrypted .chalie-backup archive.  Optional GitHub repo upload.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_file
from werkzeug.utils import secure_filename

from .auth import require_session

logger = logging.getLogger(__name__)

brain_bp = Blueprint("brain", __name__)


@brain_bp.route("/brain/info", methods=["GET"])
@require_session
def brain_info():
    try:
        from services.brain_export_service import BrainExportService
        svc = BrainExportService()
        info = svc.get_brain_info()
        return jsonify(info), 200
    except Exception as e:
        logger.exception(f"[REST API] brain/info error: {e}")
        return jsonify({"error": "Failed to retrieve brain info"}), 500


@brain_bp.route("/brain/export", methods=["POST"])
@require_session
def brain_export():
    try:
        body = request.get_json(silent=True) or {}
        include_files = body.get("include_files", True)
        encrypt = body.get("encrypt", True)
        password = body.get("password")

        from services.brain_export_service import BrainExportService
        svc = BrainExportService()
        backup_path = svc.export_brain(
            include_files=include_files,
            encrypt=encrypt,
            password=password,
        )

        return send_file(
            str(backup_path),
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name=backup_path.name,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(f"[REST API] brain/export error: {e}")
        return jsonify({"error": "Export failed"}), 500


@brain_bp.route("/brain/import", methods=["POST"])
@require_session
def brain_import():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded = request.files["file"]
    password = request.form.get("password")

    tmp_dir = tempfile.mkdtemp(prefix="chalie-upload-")
    try:
        safe_name = secure_filename(uploaded.filename or "backup.chalie-backup")
        tmp_path = Path(tmp_dir) / safe_name
        uploaded.save(str(tmp_path))

        from services.brain_import_service import BrainImportService
        svc = BrainImportService()
        result = svc.import_brain(tmp_path, password=password)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception(f"[REST API] brain/import error: {e}")
        return jsonify({"error": "Import failed"}), 500
    finally:
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


@brain_bp.route("/brain/export/upload", methods=["POST"])
@require_session
def brain_export_upload():
    try:
        body = request.get_json(silent=True) or {}
        include_files = body.get("include_files", True)
        encrypt = body.get("encrypt", True)
        password = body.get("password")
        github_repo = body.get("github_repo", "")
        github_token = body.get("github_token", "")

        if not github_repo or not github_token:
            return jsonify({"error": "github_repo and github_token are required"}), 400

        from services.brain_export_service import BrainExportService
        from services.github_backup_service import GitHubBackupService

        export_svc = BrainExportService()
        backup_path = export_svc.export_brain(
            include_files=include_files,
            encrypt=encrypt,
            password=password,
        )

        github_svc = GitHubBackupService()
        result = github_svc.upload(backup_path, github_repo, github_token)

        try:
            backup_path.unlink()
        except Exception:
            pass

        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        logger.exception(f"[REST API] brain/export/upload error: {e}")
        return jsonify({"error": "Export + upload failed"}), 500


@brain_bp.route("/brain/github/test", methods=["POST"])
@require_session
def brain_github_test():
    try:
        body = request.get_json(silent=True) or {}
        token = body.get("github_token", "")
        if not token:
            return jsonify({"error": "github_token is required"}), 400

        from services.github_backup_service import GitHubBackupService
        svc = GitHubBackupService()
        result = svc.test_connection(token)
        return jsonify(result), 200
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        logger.exception(f"[REST API] brain/github/test error: {e}")
        return jsonify({"error": "Connection test failed"}), 500
