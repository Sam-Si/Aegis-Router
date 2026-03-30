"""
Benchmark script for Aegis-Router.

Tests routing performance under various conditions.
"""

import random
import time
from typing import Callable

from aegis_router.router.cache_router import CacheRouter


def benchmark_routing(
    num_workers: int = 10,
    num_requests: int = 10000,
    tokens_per_request: int = 500,
    cache_hit_rate: float = 0.7,
) -> dict:
    """
    Benchmark routing performance.

    Args:
        num_workers: Number of workers to simulate
        num_requests: Number of routing requests to make
        tokens_per_request: Average tokens per request
        cache_hit_rate: Target cache hit rate (0.0-1.0)
    """
    print(f"Benchmarking with {num_workers} workers, {num_requests} requests...")

    router = CacheRouter()

    # Register workers
    for i in range(num_workers):
        router.register_worker(f"worker-{i}", "localhost", 8000 + i)

    # Generate some common prefixes (system prompts, templates)
    common_prefixes = [
        tuple(range(100)),  # System prompt 1
        tuple(range(100, 200)),  # System prompt 2
        tuple(range(200, 300)),  # System prompt 3
    ]

    # Populate caches
    for i, prefix in enumerate(common_prefixes):
        # Each prefix is cached on multiple workers
        for j in range(num_workers // 3):
            worker_id = f"worker-{j + (i * num_workers // 3)}"
            router.update_worker_cache(worker_id, [prefix])

    # Generate request tokens
    requests = []
    for _ in range(num_requests):
        if random.random() < cache_hit_rate:
            # Cache hit: use common prefix + random suffix
            prefix = random.choice(common_prefixes)
            suffix = tuple(random.randint(1000, 2000) for _ in range(tokens_per_request - len(prefix)))
            requests.append(prefix + suffix)
        else:
            # Cache miss: completely random tokens
            requests.append(tuple(random.randint(3000, 4000) for _ in range(tokens_per_request)))

    # Benchmark routing
    start_time = time.perf_counter()

    for tokens in requests:
        try:
            router.route_request(tokens)
        except Exception as e:
            print(f"Routing error: {e}")

    elapsed = time.perf_counter() - start_time

    stats = router.get_stats()

    results = {
        "total_requests": num_requests,
        "elapsed_time_sec": elapsed,
        "requests_per_sec": num_requests / elapsed,
        "avg_latency_ms": (elapsed / num_requests) * 1000,
        "cache_hit_rate": stats["cache_hit_rate"],
        "exact_hit_rate": stats["exact_hit_rate"],
    }

    return results


def benchmark_approximate_matching(
    num_sequences: int = 1000,
    similarity: float = 0.95,
) -> dict:
    """Benchmark approximate matching performance."""
    print(f"Benchmarking approximate matching with {num_sequences} sequences...")

    from aegis_router.matching.approximate import ApproximateMatcher

    matcher = ApproximateMatcher()

    # Generate base sequences
    base_sequences = []
    for i in range(num_sequences):
        seq = tuple(random.randint(0, 50000) for _ in range(200))
        base_sequences.append(seq)
        matcher.add(seq, f"worker-{i}")

    # Generate queries: some exact, some similar
    queries = []
    for i, seq in enumerate(base_sequences):
        if random.random() < 0.5:
            # Exact match
            queries.append((seq, f"worker-{i}"))
        else:
            # Similar (95% same tokens)
            modified = list(seq)
            num_changes = int(len(modified) * (1 - similarity))
            for _ in range(num_changes):
                idx = random.randint(0, len(modified) - 1)
                modified[idx] = random.randint(0, 50000)
            queries.append((tuple(modified), f"worker-{i}"))

    # Benchmark
    start_time = time.perf_counter()

    correct = 0
    for query, expected in queries:
        result = matcher.find_best_match(query)
        if result.worker_id == expected:
            correct += 1

    elapsed = time.perf_counter() - start_time

    return {
        "total_queries": len(queries),
        "correct_matches": correct,
        "accuracy": correct / len(queries),
        "elapsed_time_sec": elapsed,
        "queries_per_sec": len(queries) / elapsed,
        "avg_latency_ms": (elapsed / len(queries)) * 1000,
    }


def print_results(name: str, results: dict) -> None:
    """Print benchmark results."""
    print(f"\n{'=' * 50}")
    print(f"{name} Results")
    print(f"{'=' * 50}")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main():
    """Run all benchmarks."""
    print("Aegis-Router Benchmark Suite")
    print("=" * 50)

    # Benchmark 1: Routing performance
    routing_results = benchmark_routing(
        num_workers=10,
        num_requests=10000,
        tokens_per_request=500,
        cache_hit_rate=0.7,
    )
    print_results("Routing Performance", routing_results)

    # Benchmark 2: Approximate matching
    approx_results = benchmark_approximate_matching(
        num_sequences=1000,
        similarity=0.95,
    )
    print_results("Approximate Matching", approx_results)

    print("\n" + "=" * 50)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
