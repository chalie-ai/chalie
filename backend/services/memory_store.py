"""
Memory Store — Thread-safe in-memory replacement for Redis.

Implements the subset of redis.Redis API actually used by Chalie.
All data is ephemeral — loss on restart is acceptable by design.

Data structures:
- STRING: dict[key] → (value, expiry_timestamp|None)
- LIST: dict[key] → (list, expiry_timestamp|None)
- HASH: dict[key] → (dict, expiry_timestamp|None)
- SORTED SET: dict[key] → (SortedList, expiry_timestamp|None)

Thread safety: one RLock per keyspace.
TTL management: lazy eviction on read + background reaper every 60s.
"""

import json
import logging
import re
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from sortedcontainers import SortedList
except ImportError:
    SortedList = None  # Graceful fallback — sorted set ops will raise

logger = logging.getLogger(__name__)


class MemoryStore:
    """Thread-safe in-memory store with MemoryStore-compatible API."""

    def __init__(self):
        """Initialise all keyspace dicts, per-keyspace locks, pub/sub state, and the background reaper thread."""
        # Keyspaces
        self._strings: Dict[str, Tuple[Any, Optional[float]]] = {}
        self._lists: Dict[str, Tuple[list, Optional[float]]] = {}
        self._hashes: Dict[str, Tuple[dict, Optional[float]]] = {}
        self._sorted_sets: Dict[str, Tuple[Any, Optional[float]]] = {}

        self._sets: Dict[str, Tuple[set, Optional[float]]] = {}

        # Locks per keyspace
        self._str_lock = threading.RLock()
        self._list_lock = threading.RLock()
        self._hash_lock = threading.RLock()
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
                self._reap_keyspace(self._hashes, self._hash_lock)
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

    def setnx(self, key: str, value: str) -> bool:
        """Set ``key`` to ``value`` only if the key does not already exist.

        Args:
            key: Destination key.
            value: Value to store (coerced to ``str``).

        Returns:
            ``True`` if the key was set, ``False`` if it already existed.
        """
        return self.set(key, value, nx=True)

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

    def decr(self, key: str) -> int:
        """Atomically decrement the integer value at ``key`` by 1.

        If the key does not exist or has expired it is initialised to ``0`` before
        decrementing, mirroring Redis semantics.

        Args:
            key: The key whose value should be decremented.

        Returns:
            The new integer value after decrementing.
        """
        with self._str_lock:
            entry = self._strings.get(key)
            if entry is None or self._is_expired(entry[1]):
                self._strings[key] = ("-1", None)
                return -1
            val, expiry = entry
            new_val = int(val) - 1
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

    def ltrim(self, key: str, start: int, stop: int):
        """Trim the list at ``key`` so it only contains elements in [``start``, ``stop``].

        Supports negative indices (Redis-compatible semantics).
        Is a no-op if the key does not exist.

        Args:
            key: List key.
            start: Inclusive start index (may be negative).
            stop: Inclusive stop index (may be negative).

        Returns:
            ``True`` always.
        """
        with self._list_lock:
            lst = self._get_list(key)
            if lst is None:
                return True
            # Python slice: handle negative indexes (Redis-compatible semantics)
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

    def lindex(self, key: str, index: int) -> Optional[str]:
        """Return the element at ``index`` in the list at ``key``.

        Supports negative indices.

        Args:
            key: List key.
            index: Zero-based position (may be negative).

        Returns:
            The element at the given position, or ``None`` if out of range or key absent.
        """
        with self._list_lock:
            lst = self._get_list(key)
            if lst is None:
                return None
            try:
                return lst[index]
            except IndexError:
                return None

    def lset(self, key: str, index: int, value: str):
        """Set the list element at ``index`` to ``value``.

        Args:
            key: List key.
            index: Zero-based position to update (may be negative).
            value: New value (coerced to ``str``).

        Returns:
            ``True`` on success.

        Raises:
            Exception: If the key does not exist.
            IndexError: If ``index`` is out of range.
        """
        with self._list_lock:
            lst = self._get_list(key)
            if lst is None:
                raise Exception("no such key")
            lst[index] = str(value)
            return True

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

    def blpop(self, keys, timeout: int = 0):
        """Blocking left-pop from first non-empty key."""
        if isinstance(keys, str):
            keys = [keys]
        deadline = time.time() + timeout if timeout > 0 else None
        while True:
            with self._list_lock:
                for key in keys:
                    lst = self._get_list(key)
                    if lst:
                        val = lst.pop(0)
                        return (key, val)
            if deadline and time.time() >= deadline:
                return None
            time.sleep(0.1)

    # ── HASH operations ────────────────────────────────────────

    def _get_hash(self, key: str) -> Optional[dict]:
        """Return the live hash dict for ``key``, evicting it on expiry.

        Must be called while ``_hash_lock`` is held.

        Args:
            key: Hash key to look up.

        Returns:
            The mutable dict, or ``None`` if the key is absent or expired.
        """
        entry = self._hashes.get(key)
        if entry is None:
            return None
        d, expiry = entry
        if self._is_expired(expiry):
            del self._hashes[key]
            return None
        return d

    def hset(self, key: str, field: str = None, value: str = None, mapping: dict = None):
        """Set one field (or multiple via ``mapping``) in the hash at ``key``.

        Creates the hash if it does not exist. Mirrors the Redis 4+ ``HSET`` signature
        that accepts either a single ``field``/``value`` pair or a ``mapping`` dict.

        Args:
            key: Hash key.
            field: Field name (used when ``mapping`` is ``None``).
            value: Field value (coerced to ``str``; used when ``mapping`` is ``None``).
            mapping: Optional dict of ``{field: value}`` pairs to set in bulk.

        Returns:
            ``1`` always (simplified; Redis returns the number of *new* fields added).
        """
        with self._hash_lock:
            d = self._get_hash(key)
            if d is None:
                d = {}
                self._hashes[key] = (d, None)
            if mapping:
                for k, v in mapping.items():
                    d[str(k)] = str(v)
            elif field is not None:
                d[str(field)] = str(value)
            return 1

    def hget(self, key: str, field: str) -> Optional[str]:
        """Return the value of ``field`` in the hash at ``key``.

        Args:
            key: Hash key.
            field: Field name.

        Returns:
            The field value as a string, or ``None`` if the key or field does not exist.
        """
        with self._hash_lock:
            d = self._get_hash(key)
            if d is None:
                return None
            return d.get(str(field))

    def hgetall(self, key: str) -> dict:
        """Return all field-value pairs in the hash at ``key``.

        Args:
            key: Hash key.

        Returns:
            A shallow copy of the hash dict, or ``{}`` if the key does not exist.
        """
        with self._hash_lock:
            d = self._get_hash(key)
            return dict(d) if d else {}

    def hdel(self, key: str, *fields) -> int:
        """Delete one or more ``fields`` from the hash at ``key``.

        Args:
            key: Hash key.
            *fields: Field names to remove.

        Returns:
            The number of fields that were actually removed (fields that did not
            exist are not counted).
        """
        with self._hash_lock:
            d = self._get_hash(key)
            if d is None:
                return 0
            count = 0
            for f in fields:
                if str(f) in d:
                    del d[str(f)]
                    count += 1
            return count

    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        """Increment the integer value of ``field`` in the hash at ``key`` by ``amount``.

        Initialises the field to ``0`` before incrementing when it does not exist.

        Args:
            key: Hash key.
            field: Field whose integer value should be incremented.
            amount: Amount to add (default ``1``; may be negative).

        Returns:
            The new integer value of the field.
        """
        with self._hash_lock:
            d = self._get_hash(key)
            if d is None:
                d = {}
                self._hashes[key] = (d, None)
            current = int(d.get(str(field), 0))
            new_val = current + amount
            d[str(field)] = str(new_val)
            return new_val

    def hexists(self, key: str, field: str) -> bool:
        """Return ``True`` if ``field`` exists in the hash at ``key``.

        Args:
            key: Hash key.
            field: Field name to check.

        Returns:
            ``True`` if the field is present, ``False`` otherwise.
        """
        with self._hash_lock:
            d = self._get_hash(key)
            return str(field) in d if d else False

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

    def zrange(self, key: str, start: int, stop: int, withscores: bool = False) -> list:
        """Return members of the sorted set at ``key`` ordered by ascending score.

        Args:
            key: Sorted-set key.
            start: Inclusive start rank (0-based; negative indices supported).
            stop: Inclusive stop rank (negative indices supported).
            withscores: If ``True``, returns ``[(member, score), ...]`` instead of
                just ``[member, ...]``.

        Returns:
            A list of members (strings), or ``(member, score)`` tuples when
            ``withscores=True``. Returns ``[]`` if the key does not exist.
        """
        with self._zset_lock:
            zset = self._get_zset(key)
            if zset is None:
                return []
            length = len(zset)
            if stop < 0:
                stop = length + stop
            items = zset[start:stop + 1]
            if withscores:
                return [(m, s) for s, m in items]
            return [m for s, m in items]

    def zrangebyscore(self, key: str, min_score: float, max_score: float,
                      start: int = None, num: int = None, withscores: bool = False) -> list:
        """Return members of the sorted set at ``key`` with scores in [``min_score``, ``max_score``].

        Accepts the special string values ``'-inf'`` and ``'+inf'`` for unbounded ranges,
        mirroring Redis behaviour.

        Args:
            key: Sorted-set key.
            min_score: Minimum score (inclusive), or the string ``'-inf'``.
            max_score: Maximum score (inclusive), or the string ``'+inf'``.
            start: Optional offset into the filtered result list (LIMIT offset).
            num: Optional maximum number of results to return (LIMIT count).
            withscores: If ``True``, returns ``[(member, score), ...]``.

        Returns:
            A list of members, or ``(member, score)`` tuples when ``withscores=True``.
            Returns ``[]`` if the key does not exist.
        """
        with self._zset_lock:
            zset = self._get_zset(key)
            if zset is None:
                return []

            # Handle special min/max values
            if isinstance(min_score, str) and min_score == '-inf':
                min_score = float('-inf')
            if isinstance(max_score, str) and max_score == '+inf':
                max_score = float('inf')

            items = [(s, m) for s, m in zset if float(min_score) <= s <= float(max_score)]
            if start is not None and num is not None:
                items = items[start:start + num]
            if withscores:
                return [(m, s) for s, m in items]
            return [m for s, m in items]

    def zrevrange(self, key: str, start: int, stop: int, withscores: bool = False) -> list:
        """Return members of the sorted set at ``key`` ordered by descending score.

        Args:
            key: Sorted-set key.
            start: Inclusive start rank in the reversed ordering (0-based).
            stop: Inclusive stop rank in the reversed ordering (negative indices supported).
            withscores: If ``True``, returns ``[(member, score), ...]``.

        Returns:
            A list of members, or ``(member, score)`` tuples when ``withscores=True``.
            Returns ``[]`` if the key does not exist.
        """
        with self._zset_lock:
            zset = self._get_zset(key)
            if zset is None:
                return []
            length = len(zset)
            if stop < 0:
                stop = length + stop
            items = list(reversed(zset))[start:stop + 1]
            if withscores:
                return [(m, s) for s, m in items]
            return [m for s, m in items]

    def zrem(self, key: str, *members) -> int:
        """Remove one or more ``members`` from the sorted set at ``key``.

        Args:
            key: Sorted-set key.
            *members: Member values to remove (coerced to ``str``).

        Returns:
            The number of members actually removed (non-existent members are not counted).
            Returns ``0`` if the key does not exist.
        """
        with self._zset_lock:
            zset = self._get_zset(key)
            if zset is None:
                return 0
            before = len(zset)
            member_set = {str(m) for m in members}
            zset[:] = [(s, m) for s, m in zset if m not in member_set]
            return before - len(zset)

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

    def zscore(self, key: str, member: str) -> Optional[float]:
        """Return score of member in sorted set, or None if not present."""
        with self._zset_lock:
            zset = self._get_zset(key)
            if zset is None:
                return None
            member = str(member)
            for s, m in zset:
                if m == member:
                    return s
            return None

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

    def sismember(self, key: str, value: str) -> bool:
        """Return ``True`` if ``value`` is a member of the set at ``key``.

        Args:
            key: Set key.
            value: Value to check (coerced to ``str``).

        Returns:
            ``True`` if the value is present, ``False`` otherwise.
        """
        with self._set_lock:
            s = self._get_set(key)
            return str(value) in s if s else False

    def scard(self, key: str) -> int:
        """Return the number of members in the set at ``key``.

        Args:
            key: Set key.

        Returns:
            Cardinality of the set, or ``0`` if the key does not exist.
        """
        with self._set_lock:
            s = self._get_set(key)
            return len(s) if s is not None else 0

    # ── KEY operations ─────────────────────────────────────────

    def delete(self, *keys) -> int:
        """Delete one or more keys across all keyspaces.

        Removes the key from every keyspace (string, list, hash, sorted-set, set)
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
            with self._hash_lock:
                if key in self._hashes:
                    del self._hashes[key]
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
        with self._hash_lock:
            entry = self._hashes.get(key)
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
            (self._hashes, self._hash_lock),
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
            (self._hashes, self._hash_lock),
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
            (self._hashes, self._hash_lock),
            (self._sorted_sets, self._zset_lock),
            (self._sets, self._set_lock),
        ]:
            with lock:
                for k, (_, expiry) in store.items():
                    if (expiry is None or now <= expiry) and regex.fullmatch(k):
                        result.add(k)
        return list(result)

    def scan(self, cursor: int = 0, match: str = "*", count: int = 100) -> Tuple[int, list]:
        """Simplified scan — returns all matching keys at once (cursor always 0)."""
        matched = self.keys(match)
        return (0, matched)

    def scan_iter(self, match: str = "*", count: int = 100):
        """Iterate over keys matching pattern."""
        return iter(self.keys(match))

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

    def pipeline(self, transaction: bool = True) -> 'PipelineProxy':
        """Return a ``PipelineProxy`` that queues commands for batched execution.

        Args:
            transaction: Accepted for API compatibility; ignored (all operations are
                applied sequentially and immediately on ``execute()``).

        Returns:
            A new :class:`PipelineProxy` bound to this store.
        """
        return PipelineProxy(self)

    # ── Type method (compatibility) ────────────────────────────

    def type(self, key: str) -> str:
        """Return the data-structure type of ``key`` as a Redis-compatible string.

        Args:
            key: Key to inspect.

        Returns:
            One of ``"string"``, ``"list"``, ``"hash"``, ``"zset"``, ``"set"``,
            or ``"none"`` if the key does not exist in any keyspace.
        """
        with self._str_lock:
            if key in self._strings:
                return "string"
        with self._list_lock:
            if key in self._lists:
                return "list"
        with self._hash_lock:
            if key in self._hashes:
                return "hash"
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

    def listen(self):
        """Generator that yields messages (blocking)."""
        while True:
            msg = self.get_message(timeout=1.0)
            if msg:
                yield msg

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
