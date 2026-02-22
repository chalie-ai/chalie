"""
Lists API — CRUD endpoints over ListService.

Routes (all require session auth):
  GET    /lists                      — all active lists with summary counts
  POST   /lists                      — create a new list
  GET    /lists/<id>                 — get list with items array
  PUT    /lists/<id>/rename          — rename list
  DELETE /lists/<id>                 — soft-delete list
  POST   /lists/<id>/items           — add items (dedup applied)
  DELETE /lists/<id>/items           — clear all items
  DELETE /lists/<id>/items/batch     — remove specific items by content
  PUT    /lists/<id>/items/check     — check items
  PUT    /lists/<id>/items/uncheck   — uncheck items
"""

import logging
from datetime import datetime

from flask import Blueprint, jsonify, request

from .auth import require_session

logger = logging.getLogger(__name__)

lists_bp = Blueprint("lists", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_list_service():
    from services.database_service import get_shared_db_service
    from services.list_service import ListService
    return ListService(get_shared_db_service())


def _serialize_dt(val):
    if isinstance(val, datetime):
        return val.isoformat()
    return val


def _serialize_list(lst: dict) -> dict:
    """Convert datetime fields to ISO strings for JSON serialisation."""
    out = dict(lst)
    for field in ("updated_at", "created_at"):
        if field in out:
            out[field] = _serialize_dt(out[field])
    return out


def _serialize_item(item: dict) -> dict:
    """Convert datetime fields to ISO strings for JSON serialisation."""
    out = dict(item)
    for field in ("added_at", "updated_at"):
        if field in out:
            out[field] = _serialize_dt(out[field])
    return out


def _validate_name(name) -> tuple:
    name = (name or "").strip()
    if not name:
        return None, "name is required"
    if len(name) > 200:
        return None, "name must be 200 characters or fewer"
    return name, None


def _validate_items(items) -> tuple:
    if not isinstance(items, list) or not items:
        return None, "items must be a non-empty array"
    cleaned = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            return None, "each item must be a non-empty string"
        if len(item.strip()) > 500:
            return None, "each item must be 500 characters or fewer"
        cleaned.append(item.strip())
    return cleaned, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@lists_bp.route("/lists", methods=["GET"])
@require_session
def get_lists():
    """Return all active lists with summary counts."""
    try:
        svc = _get_list_service()
        lists = svc.get_all_lists()
        return jsonify({"items": [_serialize_list(lst) for lst in lists]})
    except Exception as e:
        logger.error(f"[LISTS API] get_lists error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@lists_bp.route("/lists", methods=["POST"])
@require_session
def create_list():
    """Create a new list."""
    data = request.get_json(silent=True) or {}
    name, err = _validate_name(data.get("name"))
    if err:
        return jsonify({"error": err}), 400

    list_type = (data.get("list_type") or "checklist").strip()

    try:
        svc = _get_list_service()
        list_id = svc.create_list(name, list_type=list_type)
        lst = svc.get_list(list_id)
        lst["items"] = [_serialize_item(i) for i in lst.get("items", [])]
        return jsonify({"item": _serialize_list(lst)}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 409
    except Exception as e:
        logger.error(f"[LISTS API] create_list error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@lists_bp.route("/lists/<list_id>", methods=["GET"])
@require_session
def get_list(list_id):
    """Get a list with its active items."""
    try:
        svc = _get_list_service()
        lst = svc.get_list(list_id)
        if lst is None:
            return jsonify({"error": "Not found"}), 404
        lst = dict(lst)
        lst["items"] = [_serialize_item(i) for i in lst.get("items", [])]
        return jsonify({"item": _serialize_list(lst)})
    except Exception as e:
        logger.error(f"[LISTS API] get_list error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@lists_bp.route("/lists/<list_id>/rename", methods=["PUT"])
@require_session
def rename_list(list_id):
    """Rename a list."""
    data = request.get_json(silent=True) or {}
    name, err = _validate_name(data.get("name"))
    if err:
        return jsonify({"error": err}), 400

    try:
        svc = _get_list_service()
        ok = svc.rename_list(list_id, name)
        if not ok:
            return jsonify({"error": "Not found or name already in use"}), 404
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"[LISTS API] rename_list error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@lists_bp.route("/lists/<list_id>", methods=["DELETE"])
@require_session
def delete_list(list_id):
    """Soft-delete a list."""
    try:
        svc = _get_list_service()
        ok = svc.delete_list(list_id)
        if not ok:
            return jsonify({"error": "Not found"}), 404
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"[LISTS API] delete_list error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@lists_bp.route("/lists/<list_id>/items", methods=["POST"])
@require_session
def add_items(list_id):
    """Add items to a list (dedup applied, list must exist)."""
    data = request.get_json(silent=True) or {}
    items, err = _validate_items(data.get("items"))
    if err:
        return jsonify({"error": err}), 400

    try:
        svc = _get_list_service()
        if svc.get_list(list_id) is None:
            return jsonify({"error": "Not found"}), 404
        added = svc.add_items(list_id, items, auto_create=False)
        return jsonify({"added": added})
    except Exception as e:
        logger.error(f"[LISTS API] add_items error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@lists_bp.route("/lists/<list_id>/items", methods=["DELETE"])
@require_session
def clear_items(list_id):
    """Clear all items from a list."""
    try:
        svc = _get_list_service()
        count = svc.clear_list(list_id)
        if count == -1:
            return jsonify({"error": "Not found"}), 404
        return jsonify({"cleared": count})
    except Exception as e:
        logger.error(f"[LISTS API] clear_items error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@lists_bp.route("/lists/<list_id>/items/batch", methods=["DELETE"])
@require_session
def remove_items(list_id):
    """Remove specific items from a list by content (case-insensitive)."""
    data = request.get_json(silent=True) or {}
    items, err = _validate_items(data.get("items"))
    if err:
        return jsonify({"error": err}), 400

    try:
        svc = _get_list_service()
        if svc.get_list(list_id) is None:
            return jsonify({"error": "Not found"}), 404
        removed = svc.remove_items(list_id, items)
        return jsonify({"removed": removed})
    except Exception as e:
        logger.error(f"[LISTS API] remove_items error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@lists_bp.route("/lists/<list_id>/items/check", methods=["PUT"])
@require_session
def check_items(list_id):
    """Check items in a list."""
    data = request.get_json(silent=True) or {}
    items, err = _validate_items(data.get("items"))
    if err:
        return jsonify({"error": err}), 400

    try:
        svc = _get_list_service()
        if svc.get_list(list_id) is None:
            return jsonify({"error": "Not found"}), 404
        checked = svc.check_items(list_id, items)
        return jsonify({"checked": checked})
    except Exception as e:
        logger.error(f"[LISTS API] check_items error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@lists_bp.route("/lists/<list_id>/items/uncheck", methods=["PUT"])
@require_session
def uncheck_items(list_id):
    """Uncheck items in a list."""
    data = request.get_json(silent=True) or {}
    items, err = _validate_items(data.get("items"))
    if err:
        return jsonify({"error": err}), 400

    try:
        svc = _get_list_service()
        if svc.get_list(list_id) is None:
            return jsonify({"error": "Not found"}), 404
        unchecked = svc.uncheck_items(list_id, items)
        return jsonify({"unchecked": unchecked})
    except Exception as e:
        logger.error(f"[LISTS API] uncheck_items error: {e}")
        return jsonify({"error": "Internal server error"}), 500
