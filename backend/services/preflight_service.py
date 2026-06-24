"""Read-only readiness pre-flight for the ``/ready`` probe.

Exercises each runtime dependency (SQLite, MemoryStore, the embedding ONNX
session, and the classifier heads) and reports a verified per-component status.
Pure inspection — it emits no log lines, so the 2-second ``/ready`` poll never
floods the Cognition → Errors panel.
"""


def run_preflight() -> dict[str, dict[str, object]]:
    """Return ``{component: {status, ...}}`` for database, memory_store, embeddings, onnx."""
    components: dict[str, dict[str, object]] = {}

    # SQLite
    try:
        from services.database_service import get_shared_db_service
        with get_shared_db_service().connection() as conn:
            conn.execute('SELECT 1')
        components['database'] = {'status': 'ok', 'connected': True}
    except Exception as e:
        components['database'] = {'status': 'error', 'connected': False, 'message': str(e)}

    # MemoryStore
    try:
        from services.memory_client import MemoryClientService
        MemoryClientService.create_connection().ping()
        components['memory_store'] = {'status': 'ok'}
    except Exception as e:
        components['memory_store'] = {'status': 'error', 'message': str(e)}

    # Embedding ONNX session — distinguishes "still warming up" from "broken".
    try:
        from services.embedding_service import get_embedding_service
        if get_embedding_service().is_loaded:
            components['embeddings'] = {'status': 'ok'}
        else:
            try:
                import onnxruntime  # noqa: F401
                components['embeddings'] = {'status': 'loading'}
            except Exception as e:
                components['embeddings'] = {
                    'status': 'error',
                    'message': f'onnxruntime failed to import: {e} — reinstall the venv',
                }
    except Exception as e:
        components['embeddings'] = {'status': 'error', 'message': str(e)}

    # ONNX classifier heads — preloaded in a background thread on boot. Surfaces
    # degraded (registration partially failed) loudly so a wrong/incomplete
    # classifier can't run silently.
    try:
        from services.onnx_inference_service import get_onnx_inference_service
        svc = get_onnx_inference_service()
        if svc.ready:
            components['onnx'] = {'status': 'ok'}
        elif svc.degraded:
            failed = svc.failed_registrations
            components['onnx'] = {
                'status': 'degraded',
                'failed_tasks': [t for t, _ in failed],
                'message': '; '.join(f'{t}: {err}' for t, err in failed),
            }
        else:
            components['onnx'] = {'status': 'loading'}
    except Exception as e:
        components['onnx'] = {'status': 'error', 'message': str(e)}

    return components
