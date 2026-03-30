#!/usr/bin/env python3
"""
Standalone demo of Aegis-Router cache-aware routing benefits.

This script demonstrates:
1. Identical requests hitting cache (5 requests, 1 sec delay)
2. Fuzzy matching with 85% similar prompts
3. Round-robin vs cache-aware comparison

Run: python demo.py
"""

import sys
import time

# Add the package to path if running directly
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from aegis_router.router.cache_router import CacheRouter, RoutingStrategy


class MockModelWorker:
    """Simulated LLM worker with realistic timing."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.prefill_ms = 5.0  # ms per input token
        self.decode_ms = 3.0   # ms per output token

    def process(self, prompt: str, max_tokens: int = 20, cached_tokens: int = 0):
        """Simulate inference with cache benefit."""
        total_tokens = len(prompt.split())
        prefill_tokens = total_tokens - cached_tokens

        prefill_time = max(0, prefill_tokens * self.prefill_ms)
        decode_time = max_tokens * self.decode_ms

        time.sleep((prefill_time + decode_time) / 1000)

        return {
            "worker_id": self.worker_id,
            "time_ms": prefill_time + decode_time,
            "cached": cached_tokens,
        }


def demo_identical_requests():
    """Demo: 5 identical requests showing cache benefits."""
    print("=" * 70)
    print("DEMO 1: Identical Requests (5 requests, 1 sec delay)")
    print("=" * 70)

    router = CacheRouter()
    router.register_worker("w1", "localhost", 8001)
    router.register_worker("w2", "localhost", 8002)

    workers = {"w1": MockModelWorker("w1"), "w2": MockModelWorker("w2")}

    prompt = "Explain the benefits of cache-aware routing in LLM inference"
    tokens = tuple(range(50))  # 50 tokens

    print(f"\nPrompt: {prompt[:50]}...")
    print(f"Token count: {len(tokens)}")
    print(f"Delay: 1 second between requests\n")

    times = []
    for i in range(5):
        time.sleep(1.0)

        # Route and process
        decision = router.route_request(tokens)
        router.update_worker_cache(decision.worker_id, [tokens])

        worker = workers[decision.worker_id]
        cached = int(decision.cache_hit_ratio * len(tokens))
        result = worker.process(prompt, max_tokens=15, cached_tokens=cached)

        times.append(result["time_ms"])

        status = "🚀 CACHE HIT" if decision.cache_hit_ratio > 0 else "❄️  COLD"
        print(f"  Request {i+1}: {result['time_ms']:5.1f}ms | {decision.worker_id} | {status}")

    print(f"\nResults:")
    print(f"  Cold start:     {times[0]:.1f}ms")
    print(f"  Cached avg:     {sum(times[1:])/4:.1f}ms")
    print(f"  Speedup:        {times[0] / (sum(times[1:])/4):.1f}x")
    print(f"  Time saved:     {times[0] - (sum(times[1:])/4):.1f}ms per request")
    router.stop()


def demo_fuzzy_matching():
    """Demo: Fuzzy matching with 85% similar prompts."""
    print("\n" + "=" * 70)
    print("DEMO 2: Fuzzy Matching (85% Similarity)")
    print("=" * 70)

    router = CacheRouter(enable_approximate=True, approximate_threshold=0.8)
    router.register_worker("w1", "localhost", 8001)

    workers = {"w1": MockModelWorker("w1")}

    # Base prompt (100 tokens)
    base_prompt = "Explain the benefits of cache-aware routing in LLM inference systems"
    base_tokens = tuple(range(100))

    # 85% similar (change last 15 tokens)
    similar_prompt = "Explain the benefits of cache-aware routing for LLM inference systems"
    similar_tokens = tuple(range(85)) + tuple(range(1000, 1015))

    print(f"\nBase:    {base_prompt}")
    print(f"Similar: {similar_prompt}")
    print(f"Similarity: ~85%\n")

    # First request
    print("Request 1 (base):")
    d1 = router.route_request(base_tokens)
    router.update_worker_cache(d1.worker_id, [base_tokens])
    r1 = workers[d1.worker_id].process(base_prompt, cached_tokens=0)
    print(f"  Time: {r1['time_ms']:.1f}ms (cold)")

    # Second request (fuzzy match)
    print("\nRequest 2 (85% similar):")
    d2 = router.route_request(similar_tokens)
    cached = int(d2.cache_hit_ratio * len(similar_tokens))
    r2 = workers[d2.worker_id].process(similar_prompt, cached_tokens=cached)

    print(f"  Strategy: {d2.strategy_used}")
    print(f"  Cache hit: {d2.cache_hit_ratio:.0%}")
    print(f"  Time: {r2['time_ms']:.1f}ms")
    print(f"  Benefit: {r1['time_ms'] - r2['time_ms']:.1f}ms faster")
    router.stop()


def demo_round_robin_comparison():
    """Demo: Compare round-robin vs cache-aware routing."""
    print("\n" + "=" * 70)
    print("DEMO 3: Round-Robin vs Cache-Aware Comparison")
    print("=" * 70)

    prompts = [
        "What is machine learning",
        "Explain neural networks",
        "What are transformers",
        "How does attention work",
        "Difference between GPT and BERT",
    ]
    tokens_list = [tuple(range(i*20, (i+1)*20)) for i in range(5)]

    print(f"\n5 prompts, each sent twice (10 total requests)\n")

    # Round-robin
    print("ROUND-ROBIN:")
    rr_times = []
    workers = {"w1": MockModelWorker("w1"), "w2": MockModelWorker("w2")}

    for round_num in range(2):
        for i, (prompt, tokens) in enumerate(zip(prompts, tokens_list)):
            worker_id = f"w{(i + round_num*5) % 2 + 1}"
            result = workers[worker_id].process(prompt, cached_tokens=0)
            rr_times.append(result["time_ms"])
            print(f"  Req {round_num*5 + i + 1} -> {worker_id}: {result['time_ms']:.1f}ms")

    # Cache-aware
    print("\nCACHE-AWARE:")
    router = CacheRouter()
    router.register_worker("w1", "localhost", 8001)
    router.register_worker("w2", "localhost", 8002)

    cache_times = []
    for round_num in range(2):
        for i, (prompt, tokens) in enumerate(zip(prompts, tokens_list)):
            decision = router.route_request(tokens)
            router.update_worker_cache(decision.worker_id, [tokens])

            worker = workers[decision.worker_id]
            cached = int(decision.cache_hit_ratio * len(tokens))
            result = worker.process(prompt, cached_tokens=cached)
            cache_times.append(result["time_ms"])

            status = "HIT" if decision.cache_hit_ratio > 0 else "COLD"
            print(f"  Req {round_num*5 + i + 1} -> {decision.worker_id}: "
                  f"{result['time_ms']:.1f}ms ({status})")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"  Round-robin total:   {sum(rr_times):.1f}ms")
    print(f"  Cache-aware total:   {sum(cache_times):.1f}ms")
    improvement = (sum(rr_times) - sum(cache_times)) / sum(rr_times) * 100
    print(f"  Improvement:         {improvement:.1f}% faster")
    print(f"  Throughput gain:     {(sum(rr_times)/sum(cache_times) - 1)*100:.1f}%")
    print(f"{'='*70}")
    router.stop()


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("AEGIS-ROUTER: Cache-Aware LLM Routing Demo")
    print("=" * 70)

    demo_identical_requests()
    demo_fuzzy_matching()
    demo_round_robin_comparison()

    print("\n✅ All demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
