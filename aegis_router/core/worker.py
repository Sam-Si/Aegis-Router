"""
Worker abstraction for the cache-aware router.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class WorkerStatus(Enum):
    """Worker health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # High load but responsive
    UNHEALTHY = "unhealthy"  # Not responding or failed
    DRAINING = "draining"  # No new requests, finish existing


@dataclass
class Worker:
    """
    Represents a model worker instance.

    Tracks health, load, and cache state for intelligent routing decisions.
    """

    worker_id: str
    host: str
    port: int
    status: WorkerStatus = WorkerStatus.HEALTHY

    # Load metrics (0.0 - 1.0)
    current_load: float = 0.0
    queue_depth: int = 0
    avg_latency_ms: float = 0.0

    # Cache metrics
    cache_hit_rate: float = 0.0
    total_tokens_cached: int = 0

    # Timing
    last_heartbeat: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        """Initialize computed fields."""
        self._url = f"http://{self.host}:{self.port}"

    @property
    def url(self) -> str:
        """Return the worker URL."""
        return self._url

    @property
    def is_available(self) -> bool:
        """Check if worker can accept new requests."""
        with self._lock:
            return self.status in (WorkerStatus.HEALTHY, WorkerStatus.DEGRADED)

    @property
    def is_healthy(self) -> bool:
        """Check if worker is fully healthy."""
        with self._lock:
            return self.status == WorkerStatus.HEALTHY

    def update_load(self, load: float, queue_depth: int | None = None) -> None:
        """Update worker load metrics."""
        with self._lock:
            self.current_load = max(0.0, min(1.0, load))
            if queue_depth is not None:
                self.queue_depth = queue_depth
            self.last_heartbeat = time.time()

            # Auto-update status based on load
            if self.current_load > 0.95:
                self.status = WorkerStatus.DEGRADED
            elif self.current_load < 0.8 and self.status == WorkerStatus.DEGRADED:
                self.status = WorkerStatus.HEALTHY

    def update_latency(self, latency_ms: float) -> None:
        """Update average latency using exponential moving average."""
        with self._lock:
            alpha = 0.3  # Smoothing factor
            self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms
            self.last_heartbeat = time.time()

    def record_cache_hit(self, hit: bool) -> None:
        """Update cache hit rate using exponential moving average."""
        with self._lock:
            alpha = 0.1
            self.cache_hit_rate = alpha * (1.0 if hit else 0.0) + (1 - alpha) * self.cache_hit_rate

    def mark_unhealthy(self) -> None:
        """Mark worker as unhealthy."""
        with self._lock:
            self.status = WorkerStatus.UNHEALTHY

    def mark_draining(self) -> None:
        """Mark worker as draining (no new requests)."""
        with self._lock:
            self.status = WorkerStatus.DRAINING

    def mark_healthy(self) -> None:
        """Mark worker as healthy."""
        with self._lock:
            self.status = WorkerStatus.HEALTHY

    def get_score(self, cache_hit_bonus: float = 0.5) -> float:
        """
        Calculate a routing score for this worker.

        Higher score = better candidate for routing.
        Considers: load, latency, and cache performance.
        """
        with self._lock:
            if not self.is_available:
                return -1.0

            # Base score from inverse load
            load_score = 1.0 - self.current_load

            # Latency penalty
            latency_score = 1.0 / (1.0 + self.avg_latency_ms / 100)

            # Cache performance bonus
            cache_score = self.cache_hit_rate * cache_hit_bonus

            return load_score * 0.5 + latency_score * 0.3 + cache_score * 0.2

    def time_since_heartbeat(self) -> float:
        """Return seconds since last heartbeat."""
        return time.time() - self.last_heartbeat

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        with self._lock:
            return {
                "worker_id": self.worker_id,
                "host": self.host,
                "port": self.port,
                "url": self.url,
                "status": self.status.value,
                "current_load": round(self.current_load, 3),
                "queue_depth": self.queue_depth,
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "cache_hit_rate": round(self.cache_hit_rate, 3),
                "total_tokens_cached": self.total_tokens_cached,
                "last_heartbeat": self.last_heartbeat,
            }

    def __hash__(self) -> int:
        """Hash based on worker_id."""
        return hash(self.worker_id)

    def __eq__(self, other: object) -> bool:
        """Equality based on worker_id."""
        if not isinstance(other, Worker):
            return NotImplemented
        return self.worker_id == other.worker_id
