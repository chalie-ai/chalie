"""Read-only readiness pre-flight for the ``/ready`` probe.

Exercises each runtime dependency (SQLite, MemoryStore, the embedding ONNX
session, and the classifier heads) and reports a verified per-component status.
Pure inspection — it emits no log lines, so the 2-second ``/ready`` poll never
floods the Cognition → Errors panel. The single actionable ERROR line for a
broken runtime is logged once at boot by
:meth:`services.runtime_deps_service.RuntimeDepsService.ensure_onnxruntime`.
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

    # Embedding ONNX session — distinguishes "still warming up" from "runtime
    # broken". A null session with a failed self-heal is a terminal error (the
    # hint), not the eternal 'loading' the probe used to report.
    try:
        from services import embedding_service
        from services.runtime_deps_service import RuntimeDepsService
        if embedding_service._session is not None:
            components['embeddings'] = {'status': 'ok'}
        elif RuntimeDepsService.onnxruntime_status() == 'failed':
            components['embeddings'] = {
                'status': 'error',
                'message': RuntimeDepsService.onnxruntime_hint() or 'onnxruntime unavailable',
            }
        else:
            components['embeddings'] = {'status': 'loading'}
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
