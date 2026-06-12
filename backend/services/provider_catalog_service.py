"""Provider catalog service — loads and serves provider presets from models.dev data."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

_CATALOG: Dict[str, Any] = {}


def _load_catalog() -> Dict[str, Any]:
    global _CATALOG
    if _CATALOG:
        return _CATALOG
    path = Path(__file__).resolve().parent / "provider_catalog.json"
    if not path.exists():
        logger.warning("[ProviderCatalog] catalog file not found at %s", path)
        return _CATALOG
    try:
        with open(path, "r", encoding="utf-8") as f:
            _CATALOG = json.load(f)
        logger.info("[ProviderCatalog] loaded %d providers", len(_CATALOG))
    except Exception as e:
        logger.error("[ProviderCatalog] failed to load catalog: %s", e)
    return _CATALOG


def get_catalog_providers() -> Dict[str, Any]:
    return _load_catalog()


def get_catalog_provider(provider_id: str) -> Optional[Dict[str, Any]]:
    return _load_catalog().get(provider_id)


def is_catalog_provider(provider_id: str) -> bool:
    return provider_id in _load_catalog()


def get_catalog_summary() -> List[Dict[str, Any]]:
    catalog = _load_catalog()
    return [
        {
            "id": pid,
            "name": info["name"],
            "api": info.get("api", ""),
            "model_count": len(info.get("models", [])),
        }
        for pid, info in sorted(catalog.items(), key=lambda x: x[1]["name"].lower())
    ]


def get_catalog_models(provider_id: str) -> List[str]:
    provider = get_catalog_provider(provider_id)
    if not provider:
        return []
    models = provider.get("models", [])
    if isinstance(models, list):
        return sorted(models)
    if isinstance(models, str):
        return [models]
    return []


def invalidate_catalog():
    global _CATALOG
    _CATALOG.clear()
