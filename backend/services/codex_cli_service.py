"""
Codex CLI LLM provider — communicates with ``codex app-server`` via JSON-RPC 2.0
over stdio.

Architecture:
  ``CodexSession`` manages one ``codex app-server`` subprocess and its JSON-RPC
  wire protocol.  ``CodexCliProviderService`` is the LLM service interface that
  wraps a session sourced from the module-level session cache.

Session cache rationale: ``Providers._resolve()`` creates a fresh service
instance on every call via ``create_llm_service(config)``.  A naively constructed
service would lose the subprocess between calls.  The module-level ``_sessions``
dict (keyed by ``(codex_bin, model, channel)``) survives re-instantiation and
reuses live subprocesses across turns.

v1 limitation: system prompt (``base_instructions``) is injected once at
``thread/start``.  If subsequent turns pass a different system prompt the
session logs a warning but does NOT re-inject — codex has no mid-thread
instruction update RPC.
"""

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from services.llm_service import LLMResponse, estimate_tokens

logger = logging.getLogger(__name__)

# --- Constants ----------------------------------------------------------------

_DEFAULT_MODEL = 'o4-mini'
_DEFAULT_CODEX_BIN = 'codex'
_CONTEXT_LIMIT = 128_000
_HANDSHAKE_INIT_TIMEOUT = 10.0
_HANDSHAKE_THREAD_TIMEOUT = 15.0
_TURN_START_TIMEOUT = 10.0
_DEFAULT_TURN_TIMEOUT = 600
_DETECTION_TURN_TIMEOUT = 30
_POLL_INTERVAL = 0.1

# OAuth failure patterns found in codex stderr output
_AUTH_FAILURE_HINTS = frozenset({
    'invalid_grant', 'token refresh', 'not authenticated', '401', 'unauthorized',
})

# Turn completion statuses
_TURN_STATUS_COMPLETED = 'completed'
_TURN_STATUS_INTERRUPTED = 'interrupted'
_TURN_STATUS_FAILED = 'failed'

# Item types that carry the final answer
_ANSWER_ITEM_TYPE = 'agentMessage'


# --- Module-level session cache -----------------------------------------------

_sessions: dict[tuple, '_CodexSession'] = {}
_sessions_lock = threading.Lock()


def _get_or_create_session(codex_bin: str, model: Optional[str],
                           cwd: Optional[str], channel: str = 'default') -> '_CodexSession':
    """Return a live session from the cache, creating one if necessary.

    Entries are keyed by ``(codex_bin, model, channel)`` so each distinct
    binary+model+channel triple gets its own subprocess.  Stale or auth-failed
    sessions are replaced.
    """
    key = (codex_bin, model or '', channel)
    stale = None
    with _sessions_lock:
        session = _sessions.get(key)
        if session and session.is_alive() and not session.should_retire:
            return session
        stale = session
        session = _CodexSession(codex_bin=codex_bin, model=model, cwd=cwd)
        _sessions[key] = session
    if stale:
        stale.close()
    return session


# --- CodexSession -------------------------------------------------------------

class _CodexSession:
    """Manages a single ``codex app-server`` subprocess and JSON-RPC exchange.

    Spawns the process, performs the initialize/thread/start handshake, and
    provides ``run_turn()`` for blocking turn execution.
    """

    def __init__(self, codex_bin: str = _DEFAULT_CODEX_BIN,
                 model: Optional[str] = None,
                 cwd: Optional[str] = None):
        """Create a session (does not spawn yet — call ``start()`` first)."""
        self._codex_bin = codex_bin
        self._model = model
        self._cwd = cwd or str(Path.cwd())
        self._proc: Optional[subprocess.Popen] = None
        self._thread_id: Optional[str] = None
        self._request_id = 0
        self._pending: dict[int, dict] = {}   # id → {'event': Event, 'result': ..., 'error': ...}
        self._notifications: list[dict] = []
        self._server_requests: list[dict] = []
        self._lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_lines: list[str] = []
        self._should_retire = False
        self._started = False
        self._base_instructions: Optional[str] = None

    # --- Public interface ----------------------------------------------------

    @property
    def should_retire(self) -> bool:
        """True when the session detected an auth failure and should be replaced."""
        return self._should_retire

    @property
    def started(self) -> bool:
        """True after a successful handshake."""
        with self._lock:
            return self._started

    def start(self, base_instructions: str = '') -> None:
        """Spawn the subprocess and perform the full handshake.

        Idempotent — no-op if already started.  Thread-safe.

        Raises:
            RuntimeError: If the handshake fails or times out.
        """
        with self._lock:
            if self._started:
                return
        self._spawn()
        self._handshake(base_instructions)
        with self._lock:
            self._base_instructions = base_instructions
            self._started = True

    def check_instructions(self, base_instructions: str) -> None:
        """Log a warning if the caller's system prompt differs from the session's."""
        if self._base_instructions is not None and base_instructions != self._base_instructions:
            logger.warning(
                '[CodexSession] system prompt changed since thread/start — '
                'codex does not support mid-thread instruction updates (v1 limitation)',
            )

    def is_alive(self) -> bool:
        """Return True if the subprocess is still running."""
        return self._proc is not None and self._proc.poll() is None

    def list_models(self) -> list[str]:
        """Return model IDs via the model/list RPC.  Returns [] on failure."""
        try:
            result = self._request('model/list', {}, timeout=8.0)
            raw = result.get('models') or []
            return [m.get('id') or m if isinstance(m, dict) else str(m) for m in raw if m]
        except Exception as exc:
            logger.debug('[CodexSession] model/list failed: %s', exc)
            return []

    def get_auth_error(self) -> Optional[str]:
        """Return an auth error message if one was detected, else None."""
        if not self._should_retire:
            return None
        return 'Not authenticated — run `codex login` first'

    def run_turn(self, user_text: str, timeout: int = _DEFAULT_TURN_TIMEOUT) -> str:
        """Send a user message and block until ``turn/completed``.

        Returns the accumulated ``agentMessage`` text from the turn.

        Raises:
            RuntimeError: If the turn fails, the subprocess dies, or timeout fires.
        """
        turn_id = self._start_turn(user_text)
        return self._collect_turn(turn_id, timeout)

    def close(self) -> None:
        """Terminate the subprocess cleanly.

        Takes a local reference under the instance lock, then releases the
        lock before blocking on ``wait()`` so other threads are not starved.
        """
        with self._lock:
            proc = self._proc
            self._proc = None
            self._thread_id = None
            self._started = False
            self._base_instructions = None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    # --- Spawn & reader threads ----------------------------------------------

    def _spawn(self) -> None:
        """Launch the ``codex app-server`` subprocess."""
        env = {**os.environ, 'RUST_LOG': 'warn'}
        self._proc = subprocess.Popen(
            [self._codex_bin, 'app-server'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        logger.info('[CodexSession] spawned pid=%d', self._proc.pid)

    def _read_stdout(self) -> None:
        """Read stdout line by line and dispatch JSON-RPC messages."""
        try:
            for raw in self._proc.stdout:
                line = raw.decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                self._dispatch(line)
        except Exception as exc:
            logger.debug('[CodexSession] stdout reader exiting: %s', exc)

    def _read_stderr(self) -> None:
        """Read stderr and check for auth failure signals."""
        try:
            for raw in self._proc.stderr:
                line = raw.decode('utf-8', errors='replace').strip()
                if line:
                    self._stderr_lines.append(line)
                    if self._is_auth_error(line):
                        logger.warning('[CodexSession] auth failure detected in stderr')
                        self._should_retire = True
        except Exception as exc:
            logger.debug('[CodexSession] stderr reader exiting: %s', exc)

    def _is_auth_error(self, text: str) -> bool:
        """Return True when any auth-failure hint is present in the text."""
        lower = text.lower()
        return any(hint in lower for hint in _AUTH_FAILURE_HINTS)

    # --- JSON-RPC dispatch ---------------------------------------------------

    def _dispatch(self, line: str) -> None:
        """Parse one stdout line and route it to the correct handler."""
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logger.debug('[CodexSession] non-JSON stdout line ignored: %s', line[:120])
            return

        msg_id = msg.get('id')
        method = msg.get('method')

        if msg_id is not None and method is None:
            # Response to a client request
            self._resolve_pending(msg_id, msg)
        elif msg_id is not None and method is not None:
            # Server-initiated request (e.g. dynamicTool/call)
            with self._lock:
                self._server_requests.append(msg)
        else:
            # Notification (no id, has method)
            with self._lock:
                self._notifications.append(msg)

    def _resolve_pending(self, msg_id: int, msg: dict) -> None:
        """Deliver a response to the waiting caller thread."""
        with self._lock:
            slot = self._pending.get(msg_id)
        if not slot:
            return
        slot['result'] = msg.get('result')
        slot['error'] = msg.get('error')
        slot['event'].set()

    # --- RPC helpers ---------------------------------------------------------

    def _next_id(self) -> int:
        """Return the next monotonic request ID."""
        with self._lock:
            self._request_id += 1
            return self._request_id

    def _send(self, msg: dict) -> None:
        """Serialise and write one JSON-RPC message to stdin."""
        payload = (json.dumps(msg) + '\n').encode('utf-8')
        try:
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise RuntimeError(f'[CodexSession] failed to write to stdin: {exc}') from exc

    def _request(self, method: str, params: dict, timeout: float) -> dict:
        """Send a request and block until the response arrives.

        Returns the ``result`` dict.

        Raises:
            RuntimeError: On RPC error, timeout, or subprocess death.
        """
        req_id = self._next_id()
        event = threading.Event()
        slot: dict = {'event': event, 'result': None, 'error': None}
        with self._lock:
            self._pending[req_id] = slot
        try:
            self._send({'id': req_id, 'method': method, 'params': params})
            if not event.wait(timeout):
                raise RuntimeError(f'[CodexSession] {method} timed out after {timeout}s')
        finally:
            with self._lock:
                self._pending.pop(req_id, None)
        if slot['error']:
            raise RuntimeError(f'[CodexSession] {method} RPC error: {slot["error"]}')
        return slot['result'] or {}

    def _notify(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no id, no reply expected)."""
        self._send({'method': method, 'params': params})

    # --- Handshake sequence --------------------------------------------------

    def _handshake(self, base_instructions: str) -> None:
        """Perform initialize → initialized → thread/start."""
        self._request('initialize', _build_init_params(), _HANDSHAKE_INIT_TIMEOUT)
        self._notify('initialized', {})
        self._thread_id = self._start_thread(base_instructions)
        logger.info('[CodexSession] thread started id=%s', self._thread_id)

    def _start_thread(self, base_instructions: str) -> str:
        """Send thread/start and extract the thread ID from the response."""
        params: dict = {'cwd': self._cwd}
        if self._model:
            params['model'] = self._model
        if base_instructions:
            params['base_instructions'] = base_instructions
        result = self._request('thread/start', params, _HANDSHAKE_THREAD_TIMEOUT)
        return _extract_thread_id(result)

    # --- Turn lifecycle ------------------------------------------------------

    def _start_turn(self, user_text: str) -> str:
        """Send turn/start and return the new turn ID."""
        result = self._request(
            'turn/start',
            {'threadId': self._thread_id, 'input': [{'type': 'text', 'text': user_text}]},
            _TURN_START_TIMEOUT,
        )
        return _extract_turn_id(result)

    def _collect_turn(self, turn_id: str, timeout: int) -> str:
        """Poll notifications until turn/completed and return accumulated answer text."""
        deadline = time.monotonic() + timeout
        answer_parts: list[str] = []
        processed = 0

        while True:
            if not self.is_alive():
                raise RuntimeError('[CodexSession] subprocess died during turn')
            if time.monotonic() > deadline:
                raise RuntimeError(f'[CodexSession] turn timed out after {timeout}s')

            with self._lock:
                new_notes = self._notifications[processed:]

            for note in new_notes:
                processed += 1
                completed = _process_notification(note, answer_parts, turn_id)
                if completed is not None:
                    with self._lock:
                        self._notifications = self._notifications[processed:]
                    if completed in (_TURN_STATUS_FAILED, _TURN_STATUS_INTERRUPTED):
                        raise RuntimeError(f'[CodexSession] turn ended with status={completed}')
                    return ''.join(answer_parts)

            time.sleep(_POLL_INTERVAL)


# --- Build helpers -----------------------------------------------------------

def _build_init_params() -> dict:
    """Return the initialize request params."""
    return {
        'clientInfo': {'name': 'chalie', 'title': 'Chalie', 'version': '0.1'},
        'capabilities': {},
    }


def _extract_thread_id(result: dict) -> str:
    """Extract the thread ID from a thread/start response, tolerating version differences."""
    tid = (
        result.get('thread', {}).get('id')
        or result.get('thread', {}).get('sessionId')
        or result.get('sessionId')
        or result.get('threadId')
    )
    if not tid:
        raise RuntimeError(f'[CodexSession] thread/start returned no thread id: {result}')
    return tid


def _extract_turn_id(result: dict) -> str:
    """Extract the turn ID from a turn/start response."""
    tid = result.get('turn', {}).get('id') or result.get('turnId') or ''
    if not tid:
        raise RuntimeError(f'[CodexSession] turn/start returned no turn id: {result}')
    return tid


def _process_notification(note: dict, answer_parts: list, _turn_id: str) -> Optional[str]:
    """Handle one notification.  Returns a completion status string when the turn ends.

    ``answer_parts`` is mutated in place for ``agentMessage`` items.

    Returns:
        A turn status string (``'completed'``, ``'interrupted'``, ``'failed'``)
        when a ``turn/completed`` notification arrives, otherwise ``None``.
    """
    method = note.get('method', '')
    params = note.get('params', {})

    if method == 'item/completed':
        item = params.get('item', {})
        if item.get('type') == _ANSWER_ITEM_TYPE:
            text = item.get('text', '')
            if text:
                answer_parts.append(text)
        return None

    if method == 'turn/completed':
        status = params.get('turn', {}).get('status') or _TURN_STATUS_COMPLETED
        return status

    return None


# --- CodexCliProviderService -------------------------------------------------

class CodexCliProviderService:
    """LLM provider service backed by ``codex app-server``.

    Communicates with OpenAI Codex via JSON-RPC 2.0 over stdio.  The auth
    token is managed by the codex binary itself (``codex login``); no API key
    is stored in Chalie.

    On each ``send_messages()`` call the service checks the module-level session
    cache, respawns if necessary, and forwards the user message as a codex turn.
    Tool bridging is not implemented in v1 — ``tool_calls`` is always ``None``
    and Chalie's ACT loop exits after one iteration.
    """

    # JSON path of the user-visible text field in this provider's response.
    CONTENT_FIELD_LABEL = 'item.text'

    def __init__(self, config: dict):
        """Initialise the service from a provider config dict.

        Args:
            config: Provider config dict.  Recognised keys:
                - ``platform`` (str): Must be ``'codex_cli'``.
                - ``model`` (str): Codex model name (default ``'o4-mini'``).
                - ``codex_bin`` (str): Path to the codex binary
                  (default ``'codex'``).
                - ``cwd`` (str): Working directory for the subprocess
                  (defaults to CWD at import time).

        Raises:
            ValueError: If ``config['platform']`` is not ``'codex_cli'``.
        """
        platform = config.get('platform', 'codex_cli')
        if platform != 'codex_cli':
            raise ValueError(f"CodexCliProviderService does not support platform '{platform}'")
        self._config = config
        self.model = config.get('model', _DEFAULT_MODEL)
        self._codex_bin = config.get('codex_bin', _DEFAULT_CODEX_BIN)
        self._cwd = config.get('cwd')

    # --- LLM service interface -----------------------------------------------

    def send_message(self, system_prompt: str, user_message: str) -> LLMResponse:
        """Send a single-turn message and return the response."""
        return self.send_messages(system_prompt, [{'role': 'user', 'content': user_message}])

    def send_messages(self, system_prompt: str, messages: list,
                      _cache_prefix: bool = False,
                      tools: list = None,
                      thinking_mode: str = None) -> LLMResponse:
        """Execute a turn on the codex app-server and return the response.

        ``_cache_prefix``, ``tools``, and ``thinking_mode`` are accepted for
        interface parity but are not used — v1 is pass-through only.

        Extracts the last user message from ``messages`` as the turn input.
        System prompt is injected via ``base_instructions`` on first turn.

        Returns:
            LLMResponse with ``tool_calls=None`` and ``stop_reason='end_turn'``.

        Raises:
            RuntimeError: On auth failure or unrecoverable subprocess error
                after one retry.
        """
        user_text = _extract_user_text(messages)
        start_time = time.time()

        try:
            result_text = self._execute_turn(system_prompt, user_text)
        except RuntimeError as exc:
            if 'auth' in str(exc).lower() or 'not authenticated' in str(exc).lower():
                raise
            logger.warning('[CodexCliProviderService] turn failed, retrying once: %s', exc)
            result_text = self._execute_turn(system_prompt, user_text)

        latency_ms = int((time.time() - start_time) * 1000)
        tokens_in = estimate_tokens(f'{system_prompt}\n{user_text}')
        tokens_out = estimate_tokens(result_text)

        logger.info(
            '[CodexCliProviderService] model=%s est_tokens=%d+%d latency=%dms',
            self.model, tokens_in, tokens_out, latency_ms,
        )

        return LLMResponse(
            text=result_text,
            model=self.model,
            provider='codex_cli',
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            tool_calls=None,
            stop_reason='end_turn',
            latency_ms=latency_ms,
        )

    def build_request_body(self, system_prompt: str, messages: list, tools: list = None, **_kwargs) -> str:
        """Return a JSON representation of the turn/start params for token estimation."""
        user_text = _extract_user_text(messages)
        params = {
            'threadId': '<thread-id>',
            'input': [{'type': 'text', 'text': user_text}],
        }
        return json.dumps({'method': 'turn/start', 'params': params, 'system': system_prompt})

    def get_context_limit(self) -> int:
        """Codex uses OpenAI models — 128k context."""
        return _CONTEXT_LIMIT

    def count_tokens(self, messages: list, system_prompt: str = '', tools: list = None) -> int:
        """Estimate tokens (codex does not expose a token-count RPC)."""
        parts = [system_prompt] if system_prompt else []
        for msg in messages:
            parts.append(msg.get('content', '') or '')
        return estimate_tokens(' '.join(parts))

    # --- Internal helpers ----------------------------------------------------

    def _execute_turn(self, system_prompt: str, user_text: str) -> str:
        """Ensure the session is started and execute one turn."""
        channel = self._resolve_channel()
        session = _get_or_create_session(self._codex_bin, self.model, self._cwd, channel)

        if not session.is_alive():
            raise RuntimeError('[CodexCliProviderService] session subprocess is not alive')

        if not session.started:
            session.start(base_instructions=system_prompt)
        else:
            session.check_instructions(system_prompt)

        return session.run_turn(user_text)

    @staticmethod
    def _resolve_channel() -> str:
        """Return the current processor's channel name for session isolation."""
        try:
            from services.message_processor import current_processor
            proc = current_processor()
            return getattr(proc, 'CHANNEL', 'default') if proc else 'default'
        except Exception:
            logger.debug('[CodexCliProviderService] channel resolution failed, using default')
            return 'default'


# --- Utility functions -------------------------------------------------------

def _extract_user_text(messages: list) -> str:
    """Return the content of the last user message in the messages list.

    Chalie sends all history in a single large user message, so this is
    always the only element.  Falls back to empty string if the list is empty.
    """
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            content = msg.get('content', '')
            return content if isinstance(content, str) else str(content)
    return ''


# --- Detection helper --------------------------------------------------------

def check_codex_cli(codex_bin: str = _DEFAULT_CODEX_BIN) -> dict:
    """Quick check: binary exists, handshake succeeds, model list returned.

    Does NOT send a test turn — use ``detect_codex_cli`` for a full round-trip
    test.  Suitable for the model-list dropdown in the provider form.
    """
    return _run_detection(codex_bin, full_check=False)


def detect_codex_cli(codex_bin: str = _DEFAULT_CODEX_BIN) -> dict:
    """Full check: binary + handshake + model list + test turn round-trip.

    Spawns a temporary session, lists models, sends a test turn, then tears
    down.  The subprocess is not added to the module-level cache.  Use for
    the Test button in the provider form.
    """
    return _run_detection(codex_bin, full_check=True)


def _run_detection(codex_bin: str, full_check: bool) -> dict:
    """Shared detection logic for check/detect variants.

    Returns:
        Dict with ``available``, ``error``, ``models``.
    """
    if not shutil.which(codex_bin):
        return {'available': False, 'error': f"'{codex_bin}' not found in PATH", 'models': []}

    session = _CodexSession(codex_bin=codex_bin, model=None, cwd=str(Path.cwd()))
    try:
        session.start(base_instructions='')
        models = session.list_models()
        if full_check:
            session.run_turn('hi', timeout=_DETECTION_TURN_TIMEOUT)
        return {'available': True, 'error': None, 'models': models}
    except Exception as exc:
        error_msg = session.get_auth_error() or str(exc)
        if not session.get_auth_error() and _is_auth_failure(str(exc), []):
            error_msg = 'Not authenticated — run `codex login` first'
        return {'available': False, 'error': error_msg[:300], 'models': []}
    finally:
        session.close()


def _is_auth_failure(error_msg: str, stderr_lines: list[str]) -> bool:
    """Return True when auth failure signals are present in error message or stderr."""
    combined = (error_msg + ' ' + ' '.join(stderr_lines)).lower()
    return any(hint in combined for hint in _AUTH_FAILURE_HINTS)
