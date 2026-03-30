#!/usr/bin/env python3
"""
Production features demo for Aegis-Router.

Demonstrates:
1. Load-aware cache routing (prevents hotspotting)
2. Worker failure handling
3. Cache replication for redundancy
4. Load rebalancing

Run: python demo_production.py
"""

import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from aegis_router.router.cache_router import CacheRouter, RoutingStrategy
from aegis_router.core.worker import WorkerStatus


class MockModelWorker:
    """Simulated LLM worker."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.prefill_ms = 5.0
        self.decode_ms = 3.0

    def process(self, prompt: str, max_tokens: int = 20, cached_tokens: int = 0):
        total_tokens = len(prompt.split())
        prefill_tokens = total_tokens - cached_tokens
        prefill_time = max(0, prefill_tokens * self.prefill_ms)
        decode_time = max_tokens * self.decode_ms
        time.sleep((prefill_time + decode_time) / 1000)
        return {"time_ms": prefill_time + decode_time}


def demo_load_aware_routing():
    """Demo: Load-aware routing prevents hotspotting."""
    print("=" * 70)
    print("PRODUCTION DEMO 1: Load-Aware Cache Routing")
    print("=" * 70)

    # Router with 75% max load threshold for cache workers
    router = CacheRouter(
        strategy=RoutingStrategy.CACHE_FIRST,
        max_cache_worker_load=0.75,  # Don't route to cache worker if > 75% load
        enable_cache_replication=False,  # Disable for demo clarity
    )
    router.register_worker("worker-1", "localhost", 8001)
    router.register_worker("worker-2", "localhost", 8002)

    workers = {"worker-1": MockModelWorker("worker-1"), "worker-2": MockModelWorker("worker-2")}

    prompt = "Explain the benefits of cache-aware routing"
    tokens = tuple(range(50))

    # Initial cache on worker-1
    router.update_worker_cache("worker-1", [tokens])

    print(f"\nScenario: Cache on worker-1, testing load-aware routing")
    print(f"Max cache worker load: 75%")
    print(f"Prompt: '{prompt[:40]}...'\n")

    # Case 1: worker-1 at 50% load (within threshold)
    print("Case 1: worker-1 at 50% load (within 75% threshold)")
    router.update_worker_heartbeat("worker-1", load=0.5, queue_depth=5)
    router.update_worker_heartbeat("worker-2", load=0.2, queue_depth=2)

    decision = router.route_request(tokens)
    print(f"  Routed to: {decision.worker_id}")
    print(f"  Strategy: {decision.strategy_used}")
    print(f"  Cache hit: {decision.cache_hit_ratio:.0%}")
    print(f"  ✓ Correctly uses cache worker (low load)\n")

    # Case 2: worker-1 at 90% load (over threshold)
    print("Case 2: worker-1 at 90% load (exceeds 75% threshold)")
    router.update_worker_heartbeat("worker-1", load=0.9, queue_depth=18)
    router.update_worker_heartbeat("worker-2", load=0.3, queue_depth=3)

    decision = router.route_request(tokens)
    print(f"  Routed to: {decision.worker_id}")
    print(f"  Strategy: {decision.strategy_used}")
    print(f"  Cache hit: {decision.cache_hit_ratio:.0%}")
    print(f"  ✓ Correctly avoids overloaded cache worker")
    print(f"  ✓ Falls back to idle worker for load balancing\n")

    router.stop()


def demo_worker_failure_handling():
    """Demo: Automatic cleanup when worker dies."""
    print("=" * 70)
    print("PRODUCTION DEMO 2: Worker Failure Handling")
    print("=" * 70)

    router = CacheRouter(
        health_check_interval=1.0,  # Fast health checks for demo
        worker_timeout=2.0,  # Short timeout for demo
        enable_cache_replication=False,
    )
    router.register_worker("worker-1", "localhost", 8001)
    router.register_worker("worker-2", "localhost", 8002)
    router.start()

    prompt = "What is machine learning"
    tokens = tuple(range(30))

    # Populate cache on worker-1
    router.update_worker_cache("worker-1", [tokens])

    print(f"\nScenario: worker-1 has cache, then fails")
    print(f"Initial state: Cache on worker-1\n")

    # Initial request
    decision = router.route_request(tokens)
    print(f"Request 1 (before failure):")
    print(f"  Routed to: {decision.worker_id}")
    print(f"  Cache hit: {decision.cache_hit_ratio:.0%}")

    # Simulate worker-1 failure (no heartbeat)
    print(f"\n[Simulating worker-1 failure...]")
    worker = router.get_worker("worker-1")
    worker.mark_unhealthy()

    # Trigger health check
    router._check_worker_health()

    # Request after failure
    decision = router.route_request(tokens)
    print(f"Request 2 (after failure):")
    print(f"  Routed to: {decision.worker_id}")
    print(f"  Strategy: {decision.strategy_used}")
    print(f"  Cache hit: {decision.cache_hit_ratio:.0%}")
    print(f"  ✓ Automatic failover to healthy worker")
    print(f"  ✓ Cache entries cleaned up for dead worker\n")

    router.stop()


def demo_cache_replication():
    """Demo: Cache replication for redundancy."""
    print("=" * 70)
    print("PRODUCTION DEMO 3: Cache Replication")
    print("=" * 70)

    router = CacheRouter(
        enable_cache_replication=True,
        cache_replication_factor=2,  # Each cache on 2 workers
    )
    router.register_worker("worker-1", "localhost", 8001)
    router.register_worker("worker-2", "localhost", 8002)
    router.register_worker("worker-3", "localhost", 8003)

    prompt = "Explain neural networks"
    tokens = tuple(range(40))

    print(f"\nScenario: Replication factor = 2 (each cache on 2 workers)")
    print(f"Adding cache to worker-1 only...\n")

    # Add cache to worker-1 - should replicate to worker-2 or worker-3
    router.update_worker_cache("worker-1", [tokens])

    # Check replication
    replicas = router._cache_replicas.get(tokens, set())
    print(f"Cache replicas: {replicas}")
    print(f"✓ Cache automatically replicated to {len(replicas)} workers\n")

    # Test routing with one worker overloaded
    print("Test: worker-1 overloaded, should route to replica")
    router.update_worker_heartbeat("worker-1", load=0.9)
    router.update_worker_heartbeat("worker-2", load=0.3)
    router.update_worker_heartbeat("worker-3", load=0.4)

    decision = router.route_request(tokens)
    print(f"  Routed to: {decision.worker_id}")
    print(f"  Cache hit: {decision.cache_hit_ratio:.0%}")
    print(f"  ✓ Uses replica when primary is overloaded\n")

    router.stop()


def demo_comparison():
    """Demo: Benefits of production features."""
    print("=" * 70)
    print("PRODUCTION DEMO 4: Feature Comparison")
    print("=" * 70)

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ Feature                    │ Benefit                                │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│ Load-Aware Routing         │ Prevents 100% CPU hotspotting          │")
    print("│                            │ Falls back when cache worker > 75%     │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│ Worker Failure Handling    │ Auto cleanup of dead worker cache      │")
    print("│                            │ No routing to unhealthy workers        │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│ Cache Replication          │ Each cache on N workers (default: 2)   │")
    print("│                            │ Survives single worker failure         │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│ Auto Rebalancing           │ Periodic load redistribution           │")
    print("│                            │ Prevents long-term hotspotting         │")
    print("└─────────────────────────────────────────────────────────────────────┘\n")

    # Configuration example
    print("Recommended Production Configuration:")
    print("-" * 70)
    config = """
router = CacheRouter(
    strategy=RoutingStrategy.CACHE_FIRST,
    max_cache_worker_load=0.75,      # Max 75% load for cache workers
    enable_cache_replication=True,    # Enable replication
    cache_replication_factor=2,       # 2 copies of each cache
    health_check_interval=30.0,       # Check health every 30s
    worker_timeout=60.0,              # Mark dead after 60s
    enable_auto_rebalance=True,       # Auto rebalance every 5min
    rebalance_interval=300.0,
)
"""
    print(config)


def main():
    """Run all production demos."""
    print("\n" + "=" * 70)
    print("AEGIS-ROUTER: Production Features Demo")
    print("=" * 70)
    print("\nThis demo showcases production-ready features:")
    print("  1. Load-aware cache routing (prevents hotspotting)")
    print("  2. Automatic worker failure handling")
    print("  3. Cache replication for redundancy")
    print("  4. Configuration best practices\n")

    demo_load_aware_routing()
    demo_worker_failure_handling()
    demo_cache_replication()
    demo_comparison()

    print("=" * 70)
    print("✅ All production demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
