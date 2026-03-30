"""
Radix Trie implementation for KV-cache-aware routing.

This module implements a production-grade Radix Trie (compressed prefix tree)
similar to vLLM's approach for matching incoming prompts against cached
KV caches on workers.

Key features:
- O(L) prefix matching where L is the prompt length in tokens
- Multi-worker support (same prefix may exist on multiple workers)
- Thread-safe operations with fine-grained locking
- Memory-efficient storage with reference counting
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(slots=True)
class RadixNode:
    """
    A node in the Radix Trie.

    Each node stores:
    - token_ids: The token sequence for this edge (compressed path)
    - children: Mapping from first token to child node
    - workers: Set of worker IDs that have this prefix cached
    - ref_count: Number of active requests using this node
    - last_accessed: Timestamp for LRU eviction
    """

    token_ids: tuple[int, ...]
    children: dict[int, RadixNode] = field(default_factory=dict)
    workers: set[str] = field(default_factory=set)
    ref_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    # Metadata for debugging/monitoring
    cached_tokens: int = 0  # Total tokens cached at this node

    def update_access_time(self) -> None:
        """Update the last accessed timestamp."""
        self.last_accessed = time.time()

    def add_worker(self, worker_id: str) -> None:
        """Add a worker that has this prefix cached."""
        self.workers.add(worker_id)
        self.update_access_time()

    def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker. Returns True if node has no workers left."""
        self.workers.discard(worker_id)
        return len(self.workers) == 0 and self.ref_count == 0

    def acquire(self) -> None:
        """Increment reference count (request using this cache)."""
        self.ref_count += 1
        self.update_access_time()

    def release(self) -> None:
        """Decrement reference count."""
        self.ref_count = max(0, self.ref_count - 1)
        self.update_access_time()


@dataclass
class MatchResult:
    """Result of a prefix match operation."""

    matched_tokens: tuple[int, ...]
    matched_length: int
    workers: frozenset[str]
    remaining_tokens: tuple[int, ...]
    hit_ratio: float  # matched_length / total_length
    node: RadixNode | None = None  # Reference to the matched node

    @property
    def is_full_match(self) -> bool:
        """Check if the entire prompt was matched."""
        return len(self.remaining_tokens) == 0

    @property
    def is_miss(self) -> bool:
        """Check if no tokens were matched."""
        return self.matched_length == 0


class RadixTrie:
    """
    Radix Trie for efficient prefix matching of token sequences.

    This implementation follows vLLM's design for prefix caching:
    - Compressed paths reduce memory overhead
    - Reference counting prevents eviction of active caches
    - LRU tracking for memory-constrained eviction

    Thread safety:
    - All public methods are thread-safe
    - Uses fine-grained locking at the operation level
    """

    def __init__(self) -> None:
        """Initialize an empty Radix Trie."""
        self.root = RadixNode(token_ids=())
        self._lock = threading.RLock()
        self._stats = {
            "insertions": 0,
            "matches": 0,
            "evictions": 0,
            "total_tokens_stored": 0,
        }
        self._stats_lock = threading.Lock()

    def insert(
        self, token_ids: tuple[int, ...], worker_id: str, acquire: bool = False
    ) -> RadixNode:
        """
        Insert a token sequence into the trie, associating it with a worker.

        Args:
            token_ids: The token sequence to insert
            worker_id: The worker that has this sequence cached
            acquire: Whether to increment the reference count immediately

        Returns:
            The node where the sequence ends
        """
        with self._lock:
            node = self._insert_recursive(self.root, token_ids, worker_id)
            if acquire:
                node.acquire()
            self._update_stats("insertions")
            return node

    def _insert_recursive(
        self, node: RadixNode, token_ids: tuple[int, ...], worker_id: str
    ) -> RadixNode:
        """Recursively insert token_ids into the trie."""
        if not token_ids:
            node.add_worker(worker_id)
            return node

        first_token = token_ids[0]

        if first_token in node.children:
            child = node.children[first_token]
            common_len = self._common_prefix_length(child.token_ids, token_ids)

            if common_len == len(child.token_ids):
                # Child is fully contained in token_ids, recurse
                if common_len == len(token_ids):
                    # Exact match
                    child.add_worker(worker_id)
                    return child
                return self._insert_recursive(
                    child, token_ids[common_len:], worker_id
                )

            # Partial match - need to split the child
            return self._split_node(node, first_token, token_ids, worker_id, common_len)

        # No matching child, create new node
        new_node = RadixNode(token_ids=token_ids)
        new_node.add_worker(worker_id)
        node.children[first_token] = new_node
        self._stats["total_tokens_stored"] += len(token_ids)
        return new_node

    def _split_node(
        self,
        parent: RadixNode,
        first_token: int,
        new_token_ids: tuple[int, ...],
        worker_id: str,
        common_len: int,
    ) -> RadixNode:
        """
        Split an existing node at common_len.

        Before: parent -> child(abc)
        After:  parent -> new_node(ab) -> old_child(c)
                               \
                                -> new_child(xyz)
        """
        old_child = parent.children[first_token]

        # Create intermediate node with common prefix
        common_prefix = new_token_ids[:common_len]
        new_node = RadixNode(
            token_ids=common_prefix,
            children={},
            workers=set(),  # Intermediate nodes don't have workers initially
        )

        # Reparent old child
        old_child.token_ids = old_child.token_ids[common_len:]
        new_node.children[old_child.token_ids[0]] = old_child

        # Insert new child if there's remaining tokens
        if common_len < len(new_token_ids):
            remaining = new_token_ids[common_len:]
            new_leaf = RadixNode(token_ids=remaining)
            new_leaf.add_worker(worker_id)
            new_node.children[remaining[0]] = new_leaf
            self._stats["total_tokens_stored"] += len(remaining)
        else:
            # New token_ids ends at this node
            new_node.add_worker(worker_id)

        parent.children[first_token] = new_node
        self._stats["total_tokens_stored"] += len(common_prefix)
        return new_node if common_len == len(new_token_ids) else new_leaf

    def match(self, token_ids: tuple[int, ...]) -> MatchResult:
        """
        Find the longest matching prefix for the given token sequence.

        Args:
            token_ids: The token sequence to match

        Returns:
            MatchResult containing matched prefix and available workers
        """
        with self._lock:
            matched_tokens: list[int] = []
            node = self.root
            remaining = list(token_ids)

            while remaining:
                first_token = remaining[0]
                if first_token not in node.children:
                    break

                child = node.children[first_token]
                common_len = self._common_prefix_length(
                    child.token_ids, tuple(remaining)
                )

                if common_len == 0:
                    break

                matched_tokens.extend(child.token_ids[:common_len])
                remaining = remaining[common_len:]
                node = child

                if common_len < len(child.token_ids):
                    # Partial match within node
                    break

            node.update_access_time()
            self._update_stats("matches")

            hit_ratio = len(matched_tokens) / len(token_ids) if token_ids else 0.0

            return MatchResult(
                matched_tokens=tuple(matched_tokens),
                matched_length=len(matched_tokens),
                workers=frozenset(node.workers) if matched_tokens else frozenset(),
                remaining_tokens=tuple(remaining),
                hit_ratio=hit_ratio,
                node=node if matched_tokens else None,
            )

    def find_best_worker(
        self, token_ids: tuple[int, ...], worker_loads: dict[str, float]
    ) -> tuple[str | None, MatchResult]:
        """
        Find the best worker for serving this prompt based on cache hit and load.

        Args:
            token_ids: The token sequence to match
            worker_loads: Mapping from worker_id to current load (0.0-1.0)

        Returns:
            Tuple of (best_worker_id, match_result)
        """
        result = self.match(token_ids)

        if not result.workers:
            return None, result

        # Score workers by: cache_hit_ratio * (1 - load)
        # This prioritizes workers with cache AND low load
        best_worker = None
        best_score = -1.0

        for worker_id in result.workers:
            load = worker_loads.get(worker_id, 1.0)
            # Score: balance cache reuse with load distribution
            score = result.hit_ratio * (1.0 - load)
            if score > best_score:
                best_score = score
                best_worker = worker_id

        return best_worker, result

    def remove_worker(self, worker_id: str) -> int:
        """
        Remove all references to a worker from the trie.

        Args:
            worker_id: The worker to remove

        Returns:
            Number of nodes that had this worker removed
        """
        with self._lock:
            count = self._remove_worker_recursive(self.root, worker_id)
            return count

    def _remove_worker_recursive(self, node: RadixNode, worker_id: str) -> int:
        """Recursively remove worker references."""
        count = 0
        if worker_id in node.workers:
            node.workers.discard(worker_id)
            count += 1

        # Clean up empty leaf nodes
        to_remove = []
        for first_token, child in node.children.items():
            child_count = self._remove_worker_recursive(child, worker_id)
            count += child_count

            # Remove child if empty and has no workers
            if not child.workers and not child.children and child.ref_count == 0:
                to_remove.append(first_token)
                self._stats["total_tokens_stored"] -= len(child.token_ids)

        for first_token in to_remove:
            del node.children[first_token]

        return count

    def evict_lru(self, max_tokens: int) -> int:
        """
        Evict least-recently-used entries to keep total tokens under limit.

        Args:
            max_tokens: Maximum tokens to keep

        Returns:
            Number of tokens evicted
        """
        with self._lock:
            if self._stats["total_tokens_stored"] <= max_tokens:
                return 0

            # Collect all evictable nodes (no active refs, has workers)
            evictable: list[tuple[float, RadixNode, RadixNode, int]] = []
            self._collect_evictable(self.root, evictable, self.root, 0)

            # Sort by last_accessed (oldest first)
            evictable.sort(key=lambda x: x[0])

            tokens_to_evict = self._stats["total_tokens_stored"] - max_tokens
            evicted = 0

            for _, node, parent, first_token in evictable:
                if evicted >= tokens_to_evict:
                    break
                if node.ref_count == 0:
                    evicted += len(node.token_ids)
                    node.workers.clear()
                    # Remove from parent if no children
                    if not node.children and first_token in parent.children:
                        del parent.children[first_token]
                        self._stats["total_tokens_stored"] -= len(node.token_ids)

            self._stats["evictions"] += evicted
            return evicted

    def _collect_evictable(
        self,
        node: RadixNode,
        result: list,
        parent: RadixNode,
        first_token: int,
    ) -> None:
        """Collect nodes that can potentially be evicted."""
        if node.workers and node.ref_count == 0:
            result.append((node.last_accessed, node, parent, first_token))

        for ft, child in node.children.items():
            self._collect_evictable(child, result, node, ft)

    def _common_prefix_length(
        self, a: tuple[int, ...], b: tuple[int, ...]
    ) -> int:
        """Find the length of common prefix between two token sequences."""
        min_len = min(len(a), len(b))
        for i in range(min_len):
            if a[i] != b[i]:
                return i
        return min_len

    def _update_stats(self, key: str) -> None:
        """Thread-safe stats update."""
        with self._stats_lock:
            self._stats[key] += 1

    def get_stats(self) -> dict:
        """Get current statistics."""
        with self._stats_lock:
            return dict(self._stats)

    def __len__(self) -> int:
        """Return total tokens stored."""
        return self._stats["total_tokens_stored"]

    def iter_nodes(self) -> Iterator[RadixNode]:
        """Iterate over all nodes in the trie."""
        def _iter(node: RadixNode) -> Iterator[RadixNode]:
            yield node
            for child in node.children.values():
                yield from _iter(child)

        return _iter(self.root)
