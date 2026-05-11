"""
Memory Store — Thread-safe in-memory replacement for Redis.

Implements the subset of redis.Redis API actually used by Chalie.
All data is ephemeral — loss on restart is acceptable by design.

Data structures:
- STRING: dict[key] → (value, expiry_timestamp|None)
- LIST: dict[key] → (list, expiry_timestamp|None)
- SORTED SET: dict[key] → (SortedList, expiry_timestamp|None)

Thread safety: one RLock per keyspace.
TTL management: lazy eviction on read + background reaper every 60s.
"""

import logging
import re
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


class MemoryStore:
    """Thread-safe in-memory store with MemoryStore-compatible API."""

    def __init__(self):
        """Initialise all keyspace dicts, per-keyspace locks, pub/sub state, and the background reaper thread."""
        # Keyspaces
        self._strings: Dict[str, Tuple[Any, Optional[float]]] = {}
        self._lists: Dict[str, Tuple[list, Optional[float]]] = {}
        self._sorted_sets: Dict[str, Tuple[Any, Optional[float]]] = {}

        self._sets: Dict[str, Tuple[set, Optional[float]]] = {}

        # Locks per keyspace
        self._str_lock = threading.RLock()
        self._list_lock = threading.RLock()
        self._zset_lock = threading.RLock()
        self._set_lock = threading.RLock()

        # Pub/Sub
        self._pubsub_lock = threading.RLock()
        self._channels: Dict[str, list] = defaultdict(list)  # channel → [queue.Queue, ...]

        # Background reaper
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True, name="memory-store-reaper")
        self._reaper.start()

    # ── TTL helpers ────────────────────────────────────────────

    def _is_expired(self, expiry: Optional[float]) -> bool:
        """Return ``True`` if ``expiry`` is set and has already passed."""
        return expiry is not None and time.time() > expiry

    def _expiry_from_seconds(self, seconds: Optional[int]) -> Optional[float]:
        """Convert a TTL in seconds to an absolute UNIX timestamp, or ``None`` for no expiry.

        Args:
            seconds: Relative TTL in seconds. ``None`` or ``<= 0`` means no expiry.

        Returns:
            Absolute expiry timestamp (``time.time() + seconds``) or ``None``.
        """
        if seconds is None or seconds <= 0:
            return None
        return time.time() + seconds

    def _reap_loop(self):
        """Background daemon: scan and remove expired keys every 60s."""
        while True:
            time.sleep(60)
            try:
                self._reap_keyspace(self._strings, self._str_lock)
                self._reap_keyspace(self._lists, self._list_lock)
                self._reap_keyspace(self._sorted_sets, self._zset_lock)
                self._reap_keyspace(self._sets, self._set_lock)
            except Exception as e:
                logger.debug(f"[MemoryStore] Reaper error: {e}")

    def _reap_keyspace(self, store: dict, lock: threading.RLock):
        """Delete all expired entries from a single keyspace dict under its lock.

        Args:
            store: One of the internal keyspace dicts (e.g. ``_strings``).
            lock: The ``RLock`` that guards ``store``.
        """
        now = time.time()
        with lock:
            expired = [k for k, (_, exp) in store.items() if exp is not None and now > exp]
            for k in expired:
                del store[k]

    # ── Connection / health ────────────────────────────────────

    def ping(self) -> bool:
        """Always returns True — MemoryStore is in-process and never unavailable."""
        return True

    # ── STRING operations ──────────────────────────────────────

    def get(self, key: str) -> Optional[str]:
        """Return the string value stored at ``key``, or ``None`` if absent or expired.

        Args:
            key: The string key to look up.

        Returns:
            The stored string value, or ``None`` if the key does not exist or has expired.
        """
        with self._str_lock:
            entry = self._strings.get(key)
            if entry is None:
                return None
            val, expiry = entry
            if self._is_expired(expiry):
                del self._strings[key]
                return None
            return val

    def set(self, key: str, value: str, ex: Optional[int] = None, nx: bool = False):
        """Store ``value`` at ``key`` with an optional TTL and NX (set-if-not-exists) flag.

        Args:
            key: Destination key.
            value: Value to store (coerced to ``str``).
            ex: Optional TTL in seconds. ``None`` means no expiry.
            nx: If ``True``, only set the key when it does not already exist (or is expired).

        Returns:
            ``True`` on success, ``False`` if ``nx=True`` and the key already exists.
        """
        with self._str_lock:
            if nx and key in self._strings:
                _, expiry = self._strings[key]
                if not self._is_expired(expiry):
                    return False
            self._strings[key] = (str(value), self._expiry_from_seconds(ex))
            return True

    def setex(self, key: str, seconds: int, value: str):
        """Store ``value`` at ``key`` with a mandatory TTL (set + expire in one operation).

        Args:
            key: Destination key.
            seconds: TTL in seconds (must be > 0).
            value: Value to store (coerced to ``str``).

        Returns:
            ``True`` always.
        """
        with self._str_lock:
            self._strings[key] = (str(value), self._expiry_from_seconds(seconds))
            return True

    def incr(self, key: str) -> int:
        """Atomically increment the integer value at ``key`` by 1.

        If the key does not exist or has expired it is initialised to ``0`` before
        incrementing, mirroring Redis semantics.

        Args:
            key: The key whose value should be incremented.

        Returns:
            The new integer value after incrementing.
        """
        with self._str_lock:
            entry = self._strings.get(key)
            if entry is None or self._is_expired(entry[1]):
                self._strings[key] = ("1", None)
                return 1
            val, expiry = entry
            new_val = int(val) + 1
            self._strings[key] = (str(new_val), expiry)
            return new_val

    def incrby(self, key: str, amount: int = 1) -> int:
        with self._str_lock:
            entry = self._strings.get(key)
            if entry is None or self._is_expired(entry[1]):
                self._strings[key] = (str(amount), None)
                return amount
            val, expiry = entry
            new_val = int(val) + amount
            self._strings[key] = (str(new_val), expiry)
            return new_val

    # ── LIST operations ────────────────────────────────────────

    def _get_list(self, key: str) -> Optional[list]:
        """Return the live list object for ``key``, evicting it on expiry.

        Must be called while ``_list_lock`` is held.

        Args:
            key: List key to look up.

        Returns:
            The mutable list, or ``None`` if the key is absent or expired.
        """
        entry = self._lists.get(key)
        if entry is None:
            return None
        lst, expiry = entry
        if self._is_expired(expiry):
            del self._lists[key]
            return None
        return lst

    def rpush(self, key: str, *values) -> int:
        """Append one or more values to the tail of the list at ``key``.

        Creates the list if it does not exist.

        Args:
            key: List key.
            *values: One or more values to append (coerced to ``str``).

        Returns:
            The length of the list after the push.
        """
        with self._list_lock:
            lst = self._get_list(key)
            if lst is None:
                lst = []
                self._lists[key] = (lst, None)
            for v in values:
                lst.append(str(v))
            return len(lst)

    def lpush(self, key: str, *values) -> int:
        """Prepend one or more values to the head of the list at ``key``.

        Creates the list if it does not exist. Values are inserted one at a time
        in argument order, so the last argument ends up at index 0.

        Args:
            key: List key.
            *values: One or more values to prepend (coerced to ``str``).

        Returns:
            The length of the list after the push.
        """
        with self._list_lock:
            lst = self._get_list(key)
            if lst is None:
                lst = []
                self._lists[key] = (lst, None)
            for v in values:
                lst.insert(0, str(v))
            return len(lst)

    def ltrim(self, key: str, start: int, stop: int) -> bool:
        """Trim the list at ``key`` so it only contains elements in [``start``, ``stop``].

        Supports negative indices (Redis-compatible semantics).
        Returns ``False`` if the key does not exist (no-op), ``True`` otherwise.

        Args:
            key: List key.
            start: Inclusive start index (may be negative).
            stop: Inclusive stop index (may be negative).
        """
        with self._list_lock:
            lst = self._get_list(key)
            if lst is None:
                return False
            length = len(lst)
            if start < 0:
                start = max(0, length + start)
            if stop < 0:
                stop = length + stop
            lst[:] = lst[start:stop + 1]
            return True

    def lrange(self, key: str, start: int, stop: int) -> list:
        """Return a slice of the list at ``key`` between indices ``start`` and ``stop`` (inclusive).

        Supports negative indices (Redis-compatible semantics).

        Args:
            key: List key.
            start: Inclusive start index (may be negative).
            stop: Inclusive stop index (may be negative).

        Returns:
            A new list containing the requested elements, or ``[]`` if the key is absent.
        """
        with self._list_lock:
            lst = self._get_list(key)
            if lst is None:
                return []
            length = len(lst)
            if start < 0:
                start = max(0, length + start)
            if stop < 0:
                stop = length + stop
            return lst[start:stop + 1]

    def llen(self, key: str) -> int:
        """Return the number of elements in the list at ``key``, or ``0`` if absent.

        Args:
            key: List key.

        Returns:
            Length of the list, or ``0`` if the key does not exist or has expired.
        """
        with self._list_lock:
            lst = self._get_list(key)
            return len(lst) if lst is not None else 0

    def lpop(self, key: str) -> Optional[str]:
        """Remove and return the first (head) element of the list at ``key``.

        Args:
            key: List key.

        Returns:
            The removed element, or ``None`` if the list is empty or does not exist.
        """
        with self._list_lock:
            lst = self._get_list(key)
            if not lst:
                return None
            return lst.pop(0)

    def brpop(self, key: str, timeout: int = 0) -> Optional[Tuple[str, str]]:
        """Blocking right-pop. Polls with sleep for simplicity."""
        deadline = time.time() + timeout if timeout > 0 else None
        while True:
            with self._list_lock:
                lst = self._get_list(key)
                if lst:
                    val = lst.pop()
                    return (key, val)
            if deadline and time.time() >= deadline:
                return None
            time.sleep(0.1)

    # ── SORTED SET operations ──────────────────────────────────

    def _get_zset(self, key: str) -> Optional[Any]:
        """Return the live sorted-set list for ``key``, evicting it on expiry.

        Must be called while ``_zset_lock`` is held.

        Args:
            key: Sorted-set key to look up.

        Returns:
            The mutable list of ``(score, member)`` tuples sorted by score,
            or ``None`` if the key is absent or expired.
        """
        entry = self._sorted_sets.get(key)
        if entry is None:
            return None
        zset, expiry = entry
        if self._is_expired(expiry):
            del self._sorted_sets[key]
            return None
        return zset

    def zadd(self, key: str, mapping: dict = None, **kwargs):
        """Add members with scores. mapping = {member: score}."""
        if mapping is None:
            mapping = kwargs
        with self._zset_lock:
            zset = self._get_zset(key)
            if zset is None:
                zset = []
                self._sorted_sets[key] = (zset, None)
            for member, score in mapping.items():
                # Remove existing entry if present
                zset[:] = [(s, m) for s, m in zset if m != str(member)]
                zset.append((float(score), str(member)))
            zset.sort(key=lambda x: x[0])
            return len(mapping)

    def zcard(self, key: str) -> int:
        """Return the number of members in the sorted set at ``key``.

        Args:
            key: Sorted-set key.

        Returns:
            Cardinality of the set, or ``0`` if the key does not exist.
        """
        with self._zset_lock:
            zset = self._get_zset(key)
            return len(zset) if zset is not None else 0

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        """Remove all members with scores between min_score and max_score (inclusive)."""
        with self._zset_lock:
            zset = self._get_zset(key)
            if zset is None:
                return 0
            if isinstance(min_score, str) and min_score == '-inf':
                min_score = float('-inf')
            if isinstance(max_score, str) and max_score == '+inf':
                max_score = float('inf')
            before = len(zset)
            zset[:] = [(s, m) for s, m in zset if not (float(min_score) <= s <= float(max_score))]
            return before - len(zset)

    # ── SET operations ─────────────────────────────────────────

    def _get_set(self, key: str) -> Optional[set]:
        """Return the live set object for ``key``, evicting it on expiry.

        Must be called while ``_set_lock`` is held.

        Args:
            key: Set key to look up.

        Returns:
            The mutable ``set``, or ``None`` if the key is absent or expired.
        """
        entry = self._sets.get(key)
        if entry is None:
            return None
        s, expiry = entry
        if self._is_expired(expiry):
            del self._sets[key]
            return None
        return s

    def sadd(self, key: str, *values) -> int:
        """Add one or more ``values`` to the set at ``key``.

        Creates the set if it does not exist. Values already present are silently ignored.

        Args:
            key: Set key.
            *values: Values to add (coerced to ``str``).

        Returns:
            The number of elements that were newly added to the set.
        """
        with self._set_lock:
            s = self._get_set(key)
            if s is None:
                s = set()
                self._sets[key] = (s, None)
            added = 0
            for v in values:
                sv = str(v)
                if sv not in s:
                    s.add(sv)
                    added += 1
            return added

    def srem(self, key: str, *values) -> int:
        """Remove one or more ``values`` from the set at ``key``.

        Args:
            key: Set key.
            *values: Values to remove (coerced to ``str``).

        Returns:
            The number of elements that were actually removed.
            Returns ``0`` if the key does not exist.
        """
        with self._set_lock:
            s = self._get_set(key)
            if s is None:
                return 0
            removed = 0
            for v in values:
                sv = str(v)
                if sv in s:
                    s.discard(sv)
                    removed += 1
            return removed

    def smembers(self, key: str) -> set:
        """Return all members of the set at ``key``.

        Args:
            key: Set key.

        Returns:
            A shallow copy of the set, or an empty ``set`` if the key does not exist.
        """
        with self._set_lock:
            s = self._get_set(key)
            return set(s) if s is not None else set()

    # ── KEY operations ─────────────────────────────────────────

    def delete(self, *keys) -> int:
        """Delete one or more keys across all keyspaces.

        Removes the key from every keyspace (string, list, sorted-set, set)
        where it may exist, mirroring Redis behaviour where a key belongs to exactly
        one data structure.

        Args:
            *keys: Key names to delete.

        Returns:
            Total number of key-in-keyspace entries removed.
        """
        count = 0
        for key in keys:
            with self._str_lock:
                if key in self._strings:
                    del self._strings[key]
                    count += 1
            with self._list_lock:
                if key in self._lists:
                    del self._lists[key]
                    count += 1
            with self._zset_lock:
                if key in self._sorted_sets:
                    del self._sorted_sets[key]
                    count += 1
            with self._set_lock:
                if key in self._sets:
                    del self._sets[key]
                    count += 1
        return count

    def exists(self, key: str) -> bool:
        """Return ``True`` if ``key`` exists and has not expired in any keyspace.

        Args:
            key: Key to check.

        Returns:
            ``True`` if the key is present and live, ``False`` otherwise.
        """
        with self._str_lock:
            entry = self._strings.get(key)
            if entry and not self._is_expired(entry[1]):
                return True
        with self._list_lock:
            entry = self._lists.get(key)
            if entry and not self._is_expired(entry[1]):
                return True
        with self._zset_lock:
            entry = self._sorted_sets.get(key)
            if entry and not self._is_expired(entry[1]):
                return True
        with self._set_lock:
            entry = self._sets.get(key)
            if entry and not self._is_expired(entry[1]):
                return True
        return False

    def expire(self, key: str, seconds: int) -> bool:
        """Set a TTL on ``key`` in the first keyspace where it is found.

        Args:
            key: Key to update.
            seconds: New TTL in seconds.

        Returns:
            ``True`` if the key was found and updated, ``False`` if it does not exist.
        """
        new_expiry = self._expiry_from_seconds(seconds)
        for store, lock in [
            (self._strings, self._str_lock),
            (self._lists, self._list_lock),
            (self._sorted_sets, self._zset_lock),
            (self._sets, self._set_lock),
        ]:
            with lock:
                if key in store:
                    val, _ = store[key]
                    store[key] = (val, new_expiry)
                    return True
        return False

    def ttl(self, key: str) -> int:
        """Return TTL in seconds. -1 = no expiry, -2 = key doesn't exist."""
        for store, lock in [
            (self._strings, self._str_lock),
            (self._lists, self._list_lock),
            (self._sorted_sets, self._zset_lock),
            (self._sets, self._set_lock),
        ]:
            with lock:
                entry = store.get(key)
                if entry:
                    _, expiry = entry
                    if self._is_expired(expiry):
                        del store[key]
                        continue
                    if expiry is None:
                        return -1
                    return max(0, int(expiry - time.time()))
        return -2

    def keys(self, pattern: str = "*") -> list:
        """Return keys matching glob pattern."""
        regex = re.compile(
            pattern.replace("*", ".*").replace("?", ".").replace("[", "[")
        )
        result = set()
        now = time.time()
        for store, lock in [
            (self._strings, self._str_lock),
            (self._lists, self._list_lock),
            (self._sorted_sets, self._zset_lock),
            (self._sets, self._set_lock),
        ]:
            with lock:
                for k, (_, expiry) in store.items():
                    if (expiry is None or now <= expiry) and regex.fullmatch(k):
                        result.add(k)
        return list(result)

    def scan(self, _cursor: int = 0, match: str = "*", _count: int = 100) -> Tuple[int, list]:
        """Simplified scan — returns all matching keys at once (cursor always 0)."""
        matched = self.keys(match)
        return (0, matched)

    # ── PUB/SUB ────────────────────────────────────────────────

    def publish(self, channel: str, message: str) -> int:
        """Publish a message to all subscribers of a channel."""
        import queue as queue_module
        with self._pubsub_lock:
            subscribers = self._channels.get(channel, [])
            for q in subscribers:
                try:
                    q.put_nowait({
                        "type": "message",
                        "channel": channel,
                        "data": message
                    })
                except queue_module.Full:
                    pass  # Drop if subscriber is backed up
            return len(subscribers)

    def pubsub(self, **kwargs) -> 'PubSubProxy':
        """Create a pub/sub subscriber."""
        return PubSubProxy(self)

    # ── PIPELINE ───────────────────────────────────────────────

    def pipeline(self, _transaction: bool = True) -> 'PipelineProxy':
        """Return a ``PipelineProxy`` that queues commands for batched execution.

        Args:
            transaction: Accepted for API compatibility; ignored (all operations are
                applied sequentially and immediately on ``execute()``).

        Returns:
            A new :class:`PipelineProxy` bound to this store.
        """
        return PipelineProxy(self)

    def export_matching(self, patterns: list) -> dict:
        """
        Return all matching keys with their type and value in a single pass.

        Instead of keys() × type() × get() per key (O(n×m×5) lock acquisitions),
        this iterates each keyspace once and applies all patterns simultaneously —
        O(n) total, where n = number of live keys across all keyspaces.

        Returns: {key: {"type": str, "value": any}}
        """
        import re as _re
        regexes = [_re.compile(p.replace("*", ".*").replace("?", ".")) for p in patterns]

        def _matches(k):
            return any(rx.fullmatch(k) for rx in regexes)

        result = {}
        now = time.time()

        with self._str_lock:
            for k, (v, expiry) in self._strings.items():
                if (expiry is None or now <= expiry) and _matches(k):
                    result[k] = {"type": "string", "value": v}

        with self._list_lock:
            for k, (v, expiry) in self._lists.items():
                if (expiry is None or now <= expiry) and _matches(k):
                    result[k] = {"type": "list", "value": list(v)}

        with self._zset_lock:
            for k, (v, expiry) in self._sorted_sets.items():
                if (expiry is None or now <= expiry) and _matches(k):
                    result[k] = {"type": "zset", "value": [m for m, _ in v]}

        with self._set_lock:
            for k, (v, expiry) in self._sets.items():
                if (expiry is None or now <= expiry) and _matches(k):
                    result[k] = {"type": "set", "value": list(v)}

        return result

    # ── Type method (compatibility) ────────────────────────────

    def type(self, key: str) -> str:
        """Return the data-structure type of ``key`` as a Redis-compatible string.

        Args:
            key: Key to inspect.

        Returns:
            One of ``"string"``, ``"list"``, ``"zset"``, ``"set"``,
            or ``"none"`` if the key does not exist in any keyspace.
        """
        with self._str_lock:
            if key in self._strings:
                return "string"
        with self._list_lock:
            if key in self._lists:
                return "list"
        with self._zset_lock:
            if key in self._sorted_sets:
                return "zset"
        with self._set_lock:
            if key in self._sets:
                return "set"
        return "none"


class PubSubProxy:
    """PubSub interface (Redis-compatible API) using queue.Queue per subscriber."""

    def __init__(self, store: MemoryStore):
        """Initialise the proxy with a private message queue and an empty channel subscription set.

        Args:
            store: The :class:`MemoryStore` instance that owns the channel registry.
        """
        import queue as queue_module
        self._store = store
        self._queue = queue_module.Queue(maxsize=1000)
        self._subscribed_channels: set = set()

    def subscribe(self, *channels):
        """Subscribe to one or more ``channels`` so that published messages are delivered to this proxy.

        Idempotent — subscribing to an already-subscribed channel is a no-op.

        Args:
            *channels: Channel names to subscribe to.
        """
        with self._store._pubsub_lock:
            for ch in channels:
                if ch not in self._subscribed_channels:
                    self._store._channels[ch].append(self._queue)
                    self._subscribed_channels.add(ch)

    def unsubscribe(self, *channels):
        """Unsubscribe from one or more ``channels``, stopping future message delivery.

        Silently ignores channels that are not currently subscribed.

        Args:
            *channels: Channel names to unsubscribe from.
        """
        with self._store._pubsub_lock:
            for ch in channels:
                if ch in self._subscribed_channels:
                    try:
                        self._store._channels[ch].remove(self._queue)
                    except ValueError:
                        pass
                    self._subscribed_channels.discard(ch)

    def get_message(self, timeout: float = None) -> Optional[dict]:
        """Get next message. Blocks up to timeout seconds."""
        import queue as queue_module
        try:
            if timeout is not None:
                return self._queue.get(timeout=timeout)
            else:
                return self._queue.get_nowait()
        except queue_module.Empty:
            return None

    def close(self):
        """Unsubscribe from all channels and release this proxy's queue from the store."""
        self.unsubscribe(*list(self._subscribed_channels))


class PipelineProxy:
    """Pipeline proxy (Redis-compatible API) — collects operations, executes sequentially on .execute()."""

    def __init__(self, store: MemoryStore):
        """Initialise the proxy with an empty command queue.

        Args:
            store: The :class:`MemoryStore` instance that will execute queued commands.
        """
        self._store = store
        self._commands: list = []

    def __getattr__(self, name):
        """Intercept attribute access to capture any public store method call for later execution.

        Private names (starting with ``_``) raise ``AttributeError`` immediately.

        Args:
            name: Name of the store method being called.

        Returns:
            A callable that records the method call and returns ``self`` for chaining.

        Raises:
            AttributeError: If ``name`` starts with ``_``.
        """
        if name.startswith('_'):
            raise AttributeError(name)

        def _capture(*args, **kwargs):
            """Record the method call and return the pipeline for chaining."""
            self._commands.append((name, args, kwargs))
            return self  # Allow chaining
        return _capture

    def execute(self) -> list:
        """Execute all queued commands sequentially and return their results.

        Each command is dispatched to the underlying :class:`MemoryStore`. Exceptions
        raised by individual commands are caught and included in the result list rather
        than aborting the pipeline, matching redis-py behaviour.

        Returns:
            A list of return values (or ``Exception`` instances) in the same order as
            the queued commands. The internal command queue is cleared after execution.
        """
        results = []
        for method_name, args, kwargs in self._commands:
            method = getattr(self._store, method_name, None)
            if method:
                try:
                    results.append(method(*args, **kwargs))
                except Exception as e:
                    results.append(e)
            else:
                results.append(None)
        self._commands.clear()
        return results

    def __enter__(self):
        """Support use as a context manager; returns ``self`` for command chaining."""
        return self

    def __exit__(self, *args):
        """Exit context manager; commands are not automatically executed on exit."""
        pass


# ── Shared singleton ─────────────────────────────────────────────────────────

_shared_store = None
_shared_store_lock = threading.Lock()


def get_shared_store() -> MemoryStore:
    """Return the process-wide MemoryStore singleton (thread-safe).

    This is the canonical way to obtain the shared store.  All callers that
    previously went through ``MemoryClientService.create_connection()`` should
    use this instead.
    """
    global _shared_store
    if _shared_store is None:
        with _shared_store_lock:
            if _shared_store is None:
                _shared_store = MemoryStore()
    return _shared_store
