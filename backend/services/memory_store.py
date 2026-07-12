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

Note: pub/sub (publish/subscribe/pubsub/_channels) has been removed.
Websocket.broadcast() is the authoritative push path.
"""

import logging
import re
import threading
import time
from typing import Callable, Dict, List, Optional, Set, Tuple, cast


logger = logging.getLogger(__name__)


class MemoryStore:

    def __init__(self) -> None:
        # Keyspaces
        self._strings: Dict[str, Tuple[str, Optional[float]]] = {}
        self._lists: Dict[str, Tuple[List[str], Optional[float]]] = {}
        self._sorted_sets: Dict[str, Tuple[List[Tuple[float, str]], Optional[float]]] = {}

        self._sets: Dict[str, Tuple[Set[str], Optional[float]]] = {}

        # Locks per keyspace
        self._str_lock = threading.RLock()
        self._list_lock = threading.RLock()
        self._zset_lock = threading.RLock()
        self._set_lock = threading.RLock()

        # Background reaper
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True, name="memory-store-reaper")
        self._reaper.start()

    # ── TTL helpers ────────────────────────────────────────────

    def _is_expired(self, expiry: Optional[float]) -> bool:
        return expiry is not None and time.time() > expiry

    def _expiry_from_seconds(self, seconds: Optional[int]) -> Optional[float]:
        if seconds is None or seconds <= 0:
            return None
        return time.time() + seconds

    def _reap_loop(self) -> None:
        """Background daemon: scan and remove expired keys every 60s."""
        while True:
            time.sleep(60)
            try:
                self._reap_keyspace(cast(Dict[str, Tuple[object, object]], self._strings), self._str_lock)
                self._reap_keyspace(cast(Dict[str, Tuple[object, object]], self._lists), self._list_lock)
                self._reap_keyspace(cast(Dict[str, Tuple[object, object]], self._sorted_sets), self._zset_lock)
                self._reap_keyspace(cast(Dict[str, Tuple[object, object]], self._sets), self._set_lock)
            except Exception as e:
                logger.debug(f"[MemoryStore] Reaper error: {e}")

    def _reap_keyspace(self, store: Dict[str, Tuple[object, object]], lock: threading.RLock) -> None:
        now = time.time()
        with lock:
            expired = [k for k, (_, exp) in store.items() if exp is not None and now > cast(float, exp)]
            for k in expired:
                del store[k]

    # ── Connection / health ────────────────────────────────────

    def ping(self) -> bool:
        """Always returns True — MemoryStore is in-process and never unavailable."""
        return True

    # ── STRING operations ──────────────────────────────────────

    def get(self, key: str) -> Optional[str]:
        with self._str_lock:
            entry = self._strings.get(key)
            if entry is None:
                return None
            val, expiry = entry
            if self._is_expired(expiry):
                del self._strings[key]
                return None
            return val

    def set(self, key: str, value: str, ex: Optional[int] = None, nx: bool = False) -> bool:
        with self._str_lock:
            if nx and key in self._strings:
                _, expiry = self._strings[key]
                if not self._is_expired(expiry):
                    return False
            self._strings[key] = (str(value), self._expiry_from_seconds(ex))
            return True

    def setex(self, key: str, seconds: int, value: str) -> bool:
        with self._str_lock:
            self._strings[key] = (str(value), self._expiry_from_seconds(seconds))
            return True

    def incr(self, key: str) -> int:
        """Missing or expired key is initialised to 0 before incrementing, mirroring Redis semantics."""
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

    def _get_list(self, key: str) -> Optional[List[str]]:
        """Must be called while ``_list_lock`` is held."""
        entry = self._lists.get(key)
        if entry is None:
            return None
        lst, expiry = entry
        if self._is_expired(expiry):
            del self._lists[key]
            return None
        return lst

    def rpush(self, key: str, *values: object) -> int:
        with self._list_lock:
            lst = self._get_list(key)
            if lst is None:
                lst = []
                self._lists[key] = (lst, None)
            for v in values:
                lst.append(str(v))
            return len(lst)

    def lpush(self, key: str, *values: object) -> int:
        """Values are inserted one at a time in argument order, so the last argument ends up at index 0."""
        with self._list_lock:
            lst = self._get_list(key)
            if lst is None:
                lst = []
                self._lists[key] = (lst, None)
            for v in values:
                lst.insert(0, str(v))
            return len(lst)

    def ltrim(self, key: str, start: int, stop: int) -> bool:
        """Supports negative indices (Redis-compatible semantics). Returns ``False`` if the key does not exist."""
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

    def lrange(self, key: str, start: int, stop: int) -> List[str]:
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
        with self._list_lock:
            lst = self._get_list(key)
            return len(lst) if lst is not None else 0

    def lpop(self, key: str) -> Optional[str]:
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

    def _get_zset(self, key: str) -> Optional[List[Tuple[float, str]]]:
        """Must be called while ``_zset_lock`` is held."""
        entry = self._sorted_sets.get(key)
        if entry is None:
            return None
        zset, expiry = entry
        if self._is_expired(expiry):
            del self._sorted_sets[key]
            return None
        return zset

    def zadd(self, key: str, mapping: Optional[Dict[str, float]] = None, **kwargs: float) -> int:
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
        with self._zset_lock:
            zset = self._get_zset(key)
            return len(zset) if zset is not None else 0

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
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

    def _get_set(self, key: str) -> Optional[Set[str]]:
        """Must be called while ``_set_lock`` is held."""
        entry = self._sets.get(key)
        if entry is None:
            return None
        s, expiry = entry
        if self._is_expired(expiry):
            del self._sets[key]
            return None
        return s

    def sadd(self, key: str, *values: object) -> int:
        """Creates the set if it does not exist. Values already present are silently ignored."""
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

    def srem(self, key: str, *values: object) -> int:
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

    def smembers(self, key: str) -> Set[str]:
        """Returns a shallow copy, not a live reference to the internal set."""
        with self._set_lock:
            s = self._get_set(key)
            return set(s) if s is not None else set()

    # ── KEY operations ─────────────────────────────────────────

    def delete(self, *keys: str) -> int:
        """Removes the key from every keyspace where it exists (string, list, sorted-set, set)."""
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
        with self._str_lock:
            entry: Optional[Tuple[object, Optional[float]]] = self._strings.get(key)
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
        new_expiry = self._expiry_from_seconds(seconds)
        for store, lock in [
            (cast("Dict[str, Tuple[object, Optional[float]]]", self._strings), self._str_lock),
            (cast("Dict[str, Tuple[object, Optional[float]]]", self._lists), self._list_lock),
            (cast("Dict[str, Tuple[object, Optional[float]]]", self._sorted_sets), self._zset_lock),
            (cast("Dict[str, Tuple[object, Optional[float]]]", self._sets), self._set_lock),
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
            (cast("Dict[str, Tuple[object, Optional[float]]]", self._strings), self._str_lock),
            (cast("Dict[str, Tuple[object, Optional[float]]]", self._lists), self._list_lock),
            (cast("Dict[str, Tuple[object, Optional[float]]]", self._sorted_sets), self._zset_lock),
            (cast("Dict[str, Tuple[object, Optional[float]]]", self._sets), self._set_lock),
        ]:
            with lock:
                entry: Optional[Tuple[object, Optional[float]]] = store.get(key)
                if entry:
                    _, expiry = entry
                    if self._is_expired(expiry):
                        del store[key]
                        continue
                    if expiry is None:
                        return -1
                    return max(0, int(expiry - time.time()))
        return -2

    def keys(self, pattern: str = "*") -> List[str]:
        regex = re.compile(
            pattern.replace("*", ".*").replace("?", ".").replace("[", "[")
        )
        result: Set[str] = set()
        now = time.time()
        for store, lock in [
            (cast("Dict[str, Tuple[object, Optional[float]]]", self._strings), self._str_lock),
            (cast("Dict[str, Tuple[object, Optional[float]]]", self._lists), self._list_lock),
            (cast("Dict[str, Tuple[object, Optional[float]]]", self._sorted_sets), self._zset_lock),
            (cast("Dict[str, Tuple[object, Optional[float]]]", self._sets), self._set_lock),
        ]:
            with lock:
                for k, (_, expiry) in store.items():
                    if (expiry is None or now <= expiry) and regex.fullmatch(k):
                        result.add(k)
        return list(result)

    def scan(self, _cursor: int = 0, match: str = "*", _count: int = 100) -> Tuple[int, List[str]]:
        """Simplified scan — returns all matching keys at once (cursor always 0)."""
        matched = self.keys(match)
        return (0, matched)

    # ── PIPELINE ───────────────────────────────────────────────

    def pipeline(self, _transaction: bool = True) -> 'PipelineProxy':
        return PipelineProxy(self)

    def export_matching(self, patterns: List[str]) -> Dict[str, Dict[str, object]]:
        """
        Instead of keys() × type() × get() per key (O(n×m×5) lock acquisitions),
        iterates each keyspace once and applies all patterns simultaneously —
        O(n) total, where n = number of live keys across all keyspaces.
        """
        import re as _re
        regexes = [_re.compile(p.replace("*", ".*").replace("?", ".")) for p in patterns]

        def _matches(k: str) -> bool:
            return any(rx.fullmatch(k) for rx in regexes)

        result: Dict[str, Dict[str, object]] = {}
        now = time.time()

        with self._str_lock:
            for k, (v, expiry) in cast("Dict[str, Tuple[object, Optional[float]]]", self._strings).items():
                if (expiry is None or now <= expiry) and _matches(k):
                    result[k] = {"type": "string", "value": v}

        with self._list_lock:
            for k, (v, expiry) in cast("Dict[str, Tuple[object, Optional[float]]]", self._lists).items():
                if (expiry is None or now <= expiry) and _matches(k):
                    result[k] = {"type": "list", "value": list(cast("List[str]", v))}

        with self._zset_lock:
            for k, (v, expiry) in cast("Dict[str, Tuple[object, Optional[float]]]", self._sorted_sets).items():
                if (expiry is None or now <= expiry) and _matches(k):
                    result[k] = {"type": "zset", "value": [m for m, _ in cast("List[Tuple[float, str]]", v)]}

        with self._set_lock:
            for k, (v, expiry) in cast("Dict[str, Tuple[object, Optional[float]]]", self._sets).items():
                if (expiry is None or now <= expiry) and _matches(k):
                    result[k] = {"type": "set", "value": list(cast("Set[str]", v))}

        return result

    # ── Type method (compatibility) ────────────────────────────

    def type(self, key: str) -> str:
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


class PipelineProxy:
    """Pipeline proxy (Redis-compatible API) — collects operations, executes sequentially on .execute()."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._commands: List[Tuple[str, tuple[object, ...], Dict[str, object]]] = []

    def __getattr__(self, name: str) -> Callable[..., "PipelineProxy"]:
        """Private names (starting with ``_``) raise ``AttributeError`` immediately."""
        if name.startswith('_'):
            raise AttributeError(name)

        def _capture(*args: object, **kwargs: object) -> "PipelineProxy":
            self._commands.append((name, args, kwargs))
            return self  # Allow chaining
        return _capture

    def execute(self) -> List[object]:
        """Exceptions from individual commands are caught and included in results rather than aborting the pipeline."""
        results = []
        for method_name, args, kwargs in self._commands:
            method = getattr(self._store, method_name, None)
            if method:
                try:
                    results.append(cast(Callable[..., object], method)(*args, **kwargs))
                except Exception as e:
                    results.append(e)
            else:
                results.append(None)
        self._commands.clear()
        return results

    def __enter__(self) -> "PipelineProxy":
        return self

    def __exit__(self, *args: object) -> None:
        """Commands are not automatically executed on context manager exit."""
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
