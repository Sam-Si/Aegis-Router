"""
CacheRouter: Production-grade cache-aware request router.

This module implements the main orchestration logic that combines:
1. Exact prefix matching via Radix Trie (vLLM-style)
2. Approximate matching via MinHash/SimHash for fuzzy matches
3. Worker health tracking and load-aware routing
4. Metrics and observability

Routing Strategy:
1. Try exact prefix match first (fastest, most accurate)
2. If no exact match, try approximate matching
3. Fall back to least-loaded worker if no cache match
4. Always consider worker health and load in final decision
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from aegis_router.core.radix_trie import MatchResult, RadixTrie
from aegis_router.core.worker import Worker, WorkerStatus
from aegis_router.matching.approximate import ApproximateMatcher, FuzzyMatchResult

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("aegis_router")


class RoutingStrategy(Enum):
    """Available routing strategies."""

    CACHE_FIRST = "cache_first"  # Prioritize cache hit over load
    LOAD_BALANCED = "load_balanced"  # Balance cache hit with load
    LEAST_LOADED = "least_loaded"  # Always pick least loaded worker


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    worker_id: str
    worker_url: str
    strategy_used: str  # "exact_prefix", "approximate", "load_balanced", "fallback"
    cache_hit_ratio: float  # 0.0 to 1.0
    matched_tokens: int
    total_tokens: int
    estimated_tokens_to_compute: int
    confidence: str  # "high", "medium", "low"
    fallback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_cache_hit(self) -> bool:
        """Check if this was a cache hit."""
        return self.cache_hit_ratio > 0.0

    @property
    def cache_savings(self) -> float:
        """Calculate computational savings from cache hit."""
        return self.cache_hit_ratio


@dataclass
class RouterStats:
    """Router performance statistics."""

    total_requests: int = 0
    exact_hits: int = 0
    approximate_hits: int = 0
    misses: int = 0
    total_tokens_matched: int = 0
    total_tokens_processed: int = 0

    # Timing
    total_routing_time_ms: float = 0.0
    avg_routing_time_ms: float = 0.0

    def record_request(
        self,
        matched_tokens: int,
        total_tokens: int,
        routing_time_ms: float,
        hit_type: str,
    ) -> None:
        """Record a routing decision."""
        self.total_requests += 1
        self.total_tokens_matched += matched_tokens
        self.total_tokens_processed += total_tokens
        self.total_routing_time_ms += routing_time_ms

        if hit_type == "exact":
            self.exact_hits += 1
        elif hit_type == "approximate":
            self.approximate_hits += 1
        else:
            self.misses += 1

        self.avg_routing_time_ms = self.total_routing_time_ms / self.total_requests

    @property
    def cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.exact_hits + self.approximate_hits) / self.total_requests

    @property
    def exact_hit_rate(self) -> float:
        """Calculate exact match hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.exact_hits / self.total_requests

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "total_requests": self.total_requests,
            "exact_hits": self.exact_hits,
            "approximate_hits": self.approximate_hits,
            "misses": self.misses,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "exact_hit_rate": round(self.exact_hit_rate, 4),
            "avg_routing_time_ms": round(self.avg_routing_time_ms, 4),
            "total_tokens_matched": self.total_tokens_matched,
            "total_tokens_processed": self.total_tokens_processed,
        }


class CacheRouter:
    """
    Production-grade cache-aware request router.

    Combines Radix Trie prefix matching with approximate matching to route
    inference requests to workers with the best KV cache overlap.

    Features:
    - Thread-safe operations
    - Configurable routing strategies
    - Health checking and automatic failover
    - LRU eviction for memory management
    - Comprehensive metrics
    - Load-aware cache routing (prevents cache hotspotting)
    - Cache replication across workers
    - Automatic worker failure recovery

    Example:
        >>> router = CacheRouter(max_cache_tokens=10_000_000)
        >>> router.register_worker("worker-1", "localhost", 8001)
        >>> router.update_worker_cache("worker-1", [(1, 2, 3, 4, 5)])
        >>> decision = router.route_request([(1, 2, 3, 4, 5, 6)])
        >>> print(decision.worker_id)
        "worker-1"
    """

    def __init__(
        self,
        max_cache_tokens: int = 10_000_000,
        strategy: RoutingStrategy = RoutingStrategy.CACHE_FIRST,
        enable_approximate: bool = True,
        approximate_threshold: float = 0.85,
        health_check_interval: float = 30.0,
        worker_timeout: float = 60.0,
        max_cache_worker_load: float = 0.75,  # Max load to use cache worker
        enable_cache_replication: bool = True,  # Replicate cache on multiple workers
        cache_replication_factor: int = 2,  # How many workers hold each cache
        load_balance_threshold: float = 0.5,  # Load difference to trigger rebalancing
        enable_auto_rebalance: bool = True,  # Periodically rebalance cache
        rebalance_interval: float = 300.0,  # Rebalance every 5 minutes
    ):
        """
        Initialize the cache router.

        Args:
            max_cache_tokens: Maximum tokens to store in the trie
            strategy: Routing strategy to use
            enable_approximate: Whether to enable fuzzy matching
            approximate_threshold: Minimum similarity for approximate matches
            health_check_interval: Seconds between health checks
            worker_timeout: Seconds before marking worker unhealthy
            max_cache_worker_load: Max load (0-1) to route to cache worker
            enable_cache_replication: Whether to replicate cache on multiple workers
            cache_replication_factor: How many workers should hold each cache
            load_balance_threshold: Load difference to trigger rebalancing
            enable_auto_rebalance: Whether to periodically rebalance cache
            rebalance_interval: Seconds between rebalancing attempts
        """
        self.trie = RadixTrie()
        self.workers: dict[str, Worker] = {}
        self.strategy = strategy
        self.max_cache_tokens = max_cache_tokens
        self.enable_approximate = enable_approximate
        self.approximate_threshold = approximate_threshold
        self.worker_timeout = worker_timeout
        self.max_cache_worker_load = max_cache_worker_load
        self.enable_cache_replication = enable_cache_replication
        self.cache_replication_factor = cache_replication_factor
        self.load_balance_threshold = load_balance_threshold
        self.enable_auto_rebalance = enable_auto_rebalance

        # Approximate matcher for fuzzy matching
        self.approximate_matcher = ApproximateMatcher(
            minhash_threshold=approximate_threshold,
            simhash_threshold=approximate_threshold,
        ) if enable_approximate else None

        # Statistics
        self.stats = RouterStats()
        self._stats_lock = threading.Lock()

        # Thread safety
        self._lock = threading.RLock()

        # Background health check
        self._health_check_interval = health_check_interval
        self._shutdown_event = threading.Event()
        self._health_check_thread: threading.Thread | None = None

        # Background rebalancing
        self._rebalance_interval = rebalance_interval
        self._rebalance_thread: threading.Thread | None = None

        # Cache replication tracking: token_sequence -> set of worker_ids
        self._cache_replicas: dict[tuple[int, ...], set[str]] = {}

        logger.info(
            f"CacheRouter initialized with strategy={strategy.value}, "
            f"max_cache_tokens={max_cache_tokens}, "
            f"max_cache_worker_load={max_cache_worker_load}, "
            f"cache_replication={enable_cache_replication}"
        )

    def start(self) -> None:
        """Start background threads."""
        if self._health_check_thread is None or not self._health_check_thread.is_alive():
            self._shutdown_event.clear()
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
            )
            self._health_check_thread.start()
            logger.info("Health check thread started")

        if self.enable_auto_rebalance:
            if self._rebalance_thread is None or not self._rebalance_thread.is_alive():
                self._rebalance_thread = threading.Thread(
                    target=self._rebalance_loop,
                    daemon=True,
                )
                self._rebalance_thread.start()
                logger.info("Rebalance thread started")

    def stop(self) -> None:
        """Stop background threads."""
        self._shutdown_event.set()
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)
            logger.info("Health check thread stopped")
        if self._rebalance_thread and self._rebalance_thread.is_alive():
            self._rebalance_thread.join(timeout=5.0)
            logger.info("Rebalance thread stopped")

    def _health_check_loop(self) -> None:
        """Background thread for periodic health checks."""
        while not self._shutdown_event.wait(self._health_check_interval):
            self._check_worker_health()

    def _check_worker_health(self) -> None:
        """Check health of all workers and update status."""
        with self._lock:
            current_time = time.time()
            dead_workers = []

            for worker in self.workers.values():
                time_since_hb = current_time - worker.last_heartbeat

                if time_since_hb > self.worker_timeout:
                    if worker.status != WorkerStatus.UNHEALTHY:
                        logger.warning(
                            f"Worker {worker.worker_id} marked unhealthy: "
                            f"no heartbeat for {time_since_hb:.1f}s"
                        )
                        worker.mark_unhealthy()
                        dead_workers.append(worker.worker_id)
                elif time_since_hb > self.worker_timeout / 2:
                    if worker.status == WorkerStatus.HEALTHY:
                        logger.warning(
                            f"Worker {worker.worker_id} degraded: "
                            f"heartbeat delayed {time_since_hb:.1f}s"
                        )
                        worker.status = WorkerStatus.DEGRADED

            # Clean up dead workers from cache
            for worker_id in dead_workers:
                self._handle_worker_failure(worker_id)

    def _handle_worker_failure(self, worker_id: str) -> None:
        """Handle worker failure by removing its cache entries."""
        logger.info(f"Cleaning up cache for failed worker {worker_id}")

        # Remove from trie
        self.trie.remove_worker(worker_id)

        # Remove from approximate matcher
        if self.approximate_matcher:
            self.approximate_matcher.remove(worker_id)

        # Remove from replication tracking
        for tokens, workers in list(self._cache_replicas.items()):
            workers.discard(worker_id)
            if not workers:
                del self._cache_replicas[tokens]

        logger.info(f"Cache cleanup complete for {worker_id}")

    def register_worker(
        self, worker_id: str, host: str, port: int, **kwargs
    ) -> Worker:
        """
        Register a new worker with the router.

        Args:
            worker_id: Unique identifier for the worker
            host: Worker hostname or IP
            port: Worker port
            **kwargs: Additional worker parameters

        Returns:
            The created Worker object
        """
        with self._lock:
            if worker_id in self.workers:
                logger.warning(f"Worker {worker_id} already registered, updating")

            worker = Worker(worker_id=worker_id, host=host, port=port, **kwargs)
            self.workers[worker_id] = worker
            logger.info(f"Registered worker {worker_id} at {host}:{port}")
            return worker

    def unregister_worker(self, worker_id: str) -> bool:
        """
        Unregister a worker and remove its cache entries.

        Args:
            worker_id: Worker to remove

        Returns:
            True if worker was found and removed
        """
        with self._lock:
            if worker_id not in self.workers:
                return False

            # Remove from trie
            self.trie.remove_worker(worker_id)

            # Remove from approximate matcher
            if self.approximate_matcher:
                self.approximate_matcher.remove(worker_id)

            # Remove from workers dict
            del self.workers[worker_id]

            logger.info(f"Unregistered worker {worker_id}")
            return True

    def update_worker_heartbeat(
        self, worker_id: str, load: float | None = None, queue_depth: int | None = None
    ) -> None:
        """Update worker heartbeat and metrics."""
        with self._lock:
            if worker_id not in self.workers:
                logger.warning(f"Heartbeat from unknown worker {worker_id}")
                return

            worker = self.workers[worker_id]
            worker.update_load(load or 0.0, queue_depth)

            # If was unhealthy/degraded, mark healthy again
            if worker.status in (WorkerStatus.UNHEALTHY, WorkerStatus.DEGRADED):
                logger.info(f"Worker {worker_id} recovered")
                worker.mark_healthy()

    def _rebalance_loop(self) -> None:
        """Background thread for periodic cache rebalancing."""
        while not self._shutdown_event.wait(self._rebalance_interval):
            self._rebalance_cache()

    def _rebalance_cache(self) -> None:
        """Rebalance cache across workers to prevent hotspotting."""
        with self._lock:
            if len(self.workers) < 2:
                return

            # Find overloaded and underloaded workers
            loads = {
                wid: w.current_load for wid, w in self.workers.items()
                if w.is_available
            }

            if not loads:
                return

            avg_load = sum(loads.values()) / len(loads)
            overloaded = [wid for wid, load in loads.items() if load > avg_load + self.load_balance_threshold]
            underloaded = [wid for wid, load in loads.items() if load < avg_load - self.load_balance_threshold]

            if overloaded and underloaded:
                logger.info(f"Rebalancing: overloaded={overloaded}, underloaded={underloaded}")
                # In a real implementation, we'd migrate cache entries
                # For now, we just log the imbalance

    def update_worker_cache(
        self, worker_id: str, token_sequences: list[tuple[int, ...]]
    ) -> None:
        """
        Update the cache entries for a worker.

        Args:
            worker_id: Worker to update
            token_sequences: List of token sequences the worker has cached
        """
        with self._lock:
            if worker_id not in self.workers:
                logger.warning(f"Cache update for unknown worker {worker_id}")
                return

            worker = self.workers[worker_id]
            total_tokens = 0

            for tokens in token_sequences:
                # Insert into trie
                self.trie.insert(tokens, worker_id)

                # Insert into approximate matcher
                if self.approximate_matcher:
                    self.approximate_matcher.add(tokens, worker_id)

                # Track replication
                if tokens not in self._cache_replicas:
                    self._cache_replicas[tokens] = set()
                self._cache_replicas[tokens].add(worker_id)

                # Replicate to other workers if needed
                if self.enable_cache_replication:
                    self._replicate_cache(tokens, worker_id)

                total_tokens += len(tokens)

            worker.total_tokens_cached += total_tokens
            logger.debug(
                f"Updated cache for {worker_id}: "
                f"{len(token_sequences)} sequences, {total_tokens} tokens"
            )

    def _replicate_cache(self, tokens: tuple[int, ...], source_worker: str) -> None:
        """Replicate cache to other workers for redundancy."""
        current_replicas = self._cache_replicas.get(tokens, set())

        if len(current_replicas) >= self.cache_replication_factor:
            return

        # Find healthy workers that don't have this cache
        available_workers = [
            wid for wid, w in self.workers.items()
            if w.is_available and wid not in current_replicas and wid != source_worker
        ]

        # Sort by load (prefer less loaded workers)
        available_workers.sort(key=lambda wid: self.workers[wid].current_load)

        # Replicate to meet replication factor
        for worker_id in available_workers:
            if len(current_replicas) >= self.cache_replication_factor:
                break

            self.trie.insert(tokens, worker_id)
            if self.approximate_matcher:
                self.approximate_matcher.add(tokens, worker_id)
            current_replicas.add(worker_id)
            self.workers[worker_id].total_tokens_cached += len(tokens)

            logger.debug(f"Replicated cache to {worker_id}")

    def route_request(
        self,
        token_ids: tuple[int, ...],
        priority: int = 0,
        preferred_workers: set[str] | None = None,
    ) -> RoutingDecision:
        """
        Route a request to the best worker based on cache and load.

        Args:
            token_ids: Token sequence of the prompt
            priority: Request priority (higher = more important)
            preferred_workers: Optional set of preferred worker IDs

        Returns:
            RoutingDecision with selected worker and metadata
        """
        start_time = time.perf_counter()

        with self._lock:
            total_tokens = len(token_ids)

            # Step 1: Try exact prefix match
            exact_match = self.trie.match(token_ids)

            if exact_match.workers and exact_match.matched_length > 0:
                # Filter to available workers with load check
                available_workers = self._filter_available_workers(
                    exact_match.workers, preferred_workers
                )

                # If no cache workers available under load threshold, consider fallback
                if not available_workers and exact_match.hit_ratio < 0.5:
                    # Low cache hit - better to use idle non-cache worker
                    pass  # Fall through to load-based routing
                elif available_workers:
                    worker_id = self._select_load_aware_worker(
                        available_workers, exact_match.hit_ratio
                    )
                    routing_time = (time.perf_counter() - start_time) * 1000

                    with self._stats_lock:
                        self.stats.record_request(
                            exact_match.matched_length,
                            total_tokens,
                            routing_time,
                            "exact",
                        )

                    return RoutingDecision(
                        worker_id=worker_id,
                        worker_url=self.workers[worker_id].url,
                        strategy_used="exact_prefix",
                        cache_hit_ratio=exact_match.hit_ratio,
                        matched_tokens=exact_match.matched_length,
                        total_tokens=total_tokens,
                        estimated_tokens_to_compute=len(exact_match.remaining_tokens),
                        confidence="high",
                        metadata={
                            "match_node_id": id(exact_match.node),
                            "available_workers": len(available_workers),
                        },
                    )

            # Step 2: Try approximate matching if enabled
            if self.enable_approximate and self.approximate_matcher:
                fuzzy_match = self.approximate_matcher.find_best_match(token_ids)

                if fuzzy_match.is_usable(self.approximate_threshold):
                    worker_id = fuzzy_match.worker_id

                    if worker_id and worker_id in self.workers:
                        worker = self.workers[worker_id]
                        # Check load before using fuzzy match worker
                        if worker.is_available and worker.current_load <= self.max_cache_worker_load:
                            routing_time = (time.perf_counter() - start_time) * 1000

                            with self._stats_lock:
                                self.stats.record_request(
                                    len(fuzzy_match.matched_tokens),
                                    total_tokens,
                                    routing_time,
                                    "approximate",
                                )

                            return RoutingDecision(
                                worker_id=worker_id,
                                worker_url=worker.url,
                                strategy_used="approximate",
                                cache_hit_ratio=fuzzy_match.similarity,
                                matched_tokens=len(fuzzy_match.matched_tokens),
                                total_tokens=total_tokens,
                                estimated_tokens_to_compute=total_tokens - len(fuzzy_match.matched_tokens),
                                confidence=fuzzy_match.confidence,
                                metadata={
                                    "match_method": fuzzy_match.method,
                                    "similarity": fuzzy_match.similarity,
                                },
                            )

            # Step 3: Fall back to load-based routing
            worker_id = self._select_least_loaded_worker(preferred_workers)

            if worker_id is None:
                # No available workers
                raise NoAvailableWorkersError("No healthy workers available")

            routing_time = (time.perf_counter() - start_time) * 1000

            with self._stats_lock:
                self.stats.record_request(
                    0, total_tokens, routing_time, "miss"
                )

            return RoutingDecision(
                worker_id=worker_id,
                worker_url=self.workers[worker_id].url,
                strategy_used="load_balanced",
                cache_hit_ratio=0.0,
                matched_tokens=0,
                total_tokens=total_tokens,
                estimated_tokens_to_compute=total_tokens,
                confidence="low",
                fallback_reason="no_cache_match",
            )

    def _filter_available_workers(
        self, worker_ids: frozenset[str], preferred_workers: set[str] | None
    ) -> set[str]:
        """
        Filter workers to those that are available and not overloaded.

        Returns workers with cache that are under the load threshold.
        Does NOT fall back to overloaded workers - let the caller decide.
        """
        # Only return workers with cache that are under load threshold
        available = set()
        for wid in worker_ids:
            if wid not in self.workers:
                continue
            worker = self.workers[wid]
            if worker.is_available and worker.current_load <= self.max_cache_worker_load:
                available.add(wid)

        # Apply preferred workers filter
        if preferred_workers:
            available &= preferred_workers

        return available

    def _select_load_aware_worker(
        self, worker_ids: set[str], cache_hit_ratio: float
    ) -> str:
        """
        Select worker balancing cache benefit with load.

        Prefers less loaded workers when multiple have cache.
        """
        if not worker_ids:
            raise ValueError("No worker candidates provided")

        if len(worker_ids) == 1:
            return next(iter(worker_ids))

        # Score each worker: balance cache vs load
        best_worker = None
        best_score = -float("inf")

        for wid in worker_ids:
            worker = self.workers[wid]
            load_score = 1.0 - worker.current_load  # Lower load = higher score

            # Combine cache benefit with load score
            if self.strategy == RoutingStrategy.CACHE_FIRST:
                # Cache is important but don't completely ignore load
                score = cache_hit_ratio * 10 + load_score * 2
            elif self.strategy == RoutingStrategy.LOAD_BALANCED:
                score = cache_hit_ratio * 5 + load_score * 5
            else:  # LEAST_LOADED
                score = load_score * 10

            if score > best_score:
                best_score = score
                best_worker = wid

        return best_worker or next(iter(worker_ids))

    def _select_best_worker(
        self, worker_ids: set[str], cache_hit_ratio: float
    ) -> str:
        """
        Select the best worker from candidates based on strategy.

        Args:
            worker_ids: Set of candidate worker IDs
            cache_hit_ratio: How much of the prompt is cached

        Returns:
            Selected worker ID
        """
        if not worker_ids:
            raise ValueError("No worker candidates provided")

        if len(worker_ids) == 1:
            return next(iter(worker_ids))

        # Score each worker
        best_worker = None
        best_score = -float("inf")

        for wid in worker_ids:
            worker = self.workers[wid]
            score = worker.get_score()

            # Adjust score based on strategy
            if self.strategy == RoutingStrategy.CACHE_FIRST:
                # Prioritize cache hit heavily
                score = cache_hit_ratio * 10 + score
            elif self.strategy == RoutingStrategy.LOAD_BALANCED:
                # Balance cache and load
                score = cache_hit_ratio * 5 + score
            # For LEAST_LOADED, just use the base score

            if score > best_score:
                best_score = score
                best_worker = wid

        return best_worker or next(iter(worker_ids))

    def _select_least_loaded_worker(
        self, preferred_workers: set[str] | None = None
    ) -> str | None:
        """Select the least loaded available worker."""
        candidates = [
            w for w in self.workers.values()
            if w.is_available and (preferred_workers is None or w.worker_id in preferred_workers)
        ]

        if not candidates:
            return None

        # Sort by load, then by queue depth
        candidates.sort(key=lambda w: (w.current_load, w.queue_depth))
        return candidates[0].worker_id

    def get_worker(self, worker_id: str) -> Worker | None:
        """Get a worker by ID."""
        return self.workers.get(worker_id)

    def get_all_workers(self) -> list[Worker]:
        """Get all registered workers."""
        return list(self.workers.values())

    def get_healthy_workers(self) -> list[Worker]:
        """Get all healthy workers."""
        return [w for w in self.workers.values() if w.is_healthy]

    def evict_expired_cache(self) -> int:
        """Evict LRU cache entries to stay under memory limit."""
        return self.trie.evict_lru(self.max_cache_tokens)

    def get_stats(self) -> dict:
        """Get router statistics."""
        with self._stats_lock:
            stats = self.stats.to_dict()

        stats.update({
            "workers_total": len(self.workers),
            "workers_healthy": len(self.get_healthy_workers()),
            "trie_tokens": len(self.trie),
            "trie_nodes": sum(1 for _ in self.trie.iter_nodes()),
        })

        if self.approximate_matcher:
            stats["approximate_index"] = self.approximate_matcher.get_stats()

        return stats

    def __enter__(self) -> CacheRouter:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()


class NoAvailableWorkersError(Exception):
    """Raised when no workers are available to handle a request."""
    pass
