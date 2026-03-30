"""
Integration tests for RadixAttention + Continuous Batching Engine.

These tests use the ACTUAL TinyLlama model at /models/tinyllama.gguf
No mocking - real inference with real performance characteristics.

Run all tests with: pytest tests/test_radix_attention_integration.py -v
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import pytest

# Skip all tests if model not available
MODEL_PATH = "/models/tinyllama.gguf"
model_exists = os.path.exists(MODEL_PATH)

if TYPE_CHECKING:
    from aegis_router.core.radix_attention_engine import RadixAttentionEngine


pytestmark = pytest.mark.skipif(
    not model_exists,
    reason=f"Model not found at {MODEL_PATH}"
)


@pytest.fixture(scope="module")
def engine():
    """Create a RadixAttention engine with the real model."""
    from aegis_router.core.radix_attention_engine import RadixAttentionEngine
    
    print(f"\n{'='*60}")
    print(f"Loading model from {MODEL_PATH}...")
    print(f"{'='*60}")
    
    eng = RadixAttentionEngine(
        model_path=MODEL_PATH,
        max_total_tokens=2048,
        max_batch_size=4,
        max_new_tokens=20,
        temperature=0.7,
    )
    
    print("Model loaded successfully!")
    yield eng
    
    # Cleanup
    print("\nCleaning up engine...")


@pytest.fixture
def fresh_engine():
    """Create a fresh engine for each test."""
    from aegis_router.core.radix_attention_engine import RadixAttentionEngine
    
    eng = RadixAttentionEngine(
        model_path=MODEL_PATH,
        max_total_tokens=2048,
        max_batch_size=4,
        max_new_tokens=10,  # Shorter for faster tests
        temperature=0.7,
    )
    yield eng


class TestRadixAttentionCaching:
    """Tests for RadixAttention prefix caching with real model."""
    
    def test_identical_prompts_get_cache_hits(self, fresh_engine):
        """
        Test that identical prompts benefit from RadixAttention caching.
        
        First request: Full prefill (cache miss)
        Second request: Should hit cache for the entire prefix
        """
        engine = fresh_engine
        
        # Use a simple, consistent prompt
        prompt = "The capital of France is"
        prompt_tokens = engine.model.tokenize(prompt.encode(), add_bos=True)
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Prompt tokens: {len(prompt_tokens)}")
        
        # First request - should be cache miss
        print("\n--- Request 1 (Expected: Cache Miss) ---")
        req1_id = engine.add_request(prompt_tokens, max_new_tokens=5)
        completed1 = engine.run_until_complete(timeout=60)
        
        stats1 = engine.get_stats()
        print(f"Cache hits: {stats1['cache_hits']}, misses: {stats1['cache_misses']}")
        print(f"Tokens saved by cache: {stats1['total_tokens_saved_by_cache']}")
        
        assert stats1["cache_misses"] == 1, "First request should be cache miss"
        assert stats1["cache_hits"] == 0, "First request should not hit cache"
        
        # Second request with identical prompt - should be cache hit
        print("\n--- Request 2 (Expected: Cache Hit) ---")
        engine.reset_stats()
        req2_id = engine.add_request(prompt_tokens, max_new_tokens=5)
        completed2 = engine.run_until_complete(timeout=60)
        
        stats2 = engine.get_stats()
        print(f"Cache hits: {stats2['cache_hits']}, misses: {stats2['cache_misses']}")
        print(f"Tokens saved by cache: {stats2['total_tokens_saved_by_cache']}")
        
        assert stats2["cache_hits"] == 1, "Second request should hit cache"
        assert stats2["total_tokens_saved_by_cache"] == len(prompt_tokens), \
            f"Should save all {len(prompt_tokens)} prompt tokens"
    
    def test_prefix_matching_partial_cache(self, fresh_engine):
        """
        Test partial prefix matching.
        
        Request 1: "The quick brown fox"
        Request 2: "The quick brown fox jumps over"
        
        Request 2 should get a partial cache hit for "The quick brown fox"
        """
        engine = fresh_engine
        
        prompt1 = "The quick brown fox"
        prompt2 = "The quick brown fox jumps over"
        
        tokens1 = engine.model.tokenize(prompt1.encode(), add_bos=True)
        tokens2 = engine.model.tokenize(prompt2.encode(), add_bos=True)
        
        print(f"\nPrompt 1: '{prompt1}' ({len(tokens1)} tokens)")
        print(f"Prompt 2: '{prompt2}' ({len(tokens2)} tokens)")
        print(f"Expected shared prefix: {len(tokens1)} tokens")
        
        # First request
        print("\n--- Request 1 ---")
        engine.add_request(tokens1, max_new_tokens=3)
        engine.run_until_complete(timeout=60)
        
        # Second request with extended prompt
        print("\n--- Request 2 (should have partial cache hit) ---")
        engine.reset_stats()
        req2_id = engine.add_request(tokens2, max_new_tokens=3)
        completed = engine.run_until_complete(timeout=60)
        
        stats = engine.get_stats()
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Tokens saved: {stats['total_tokens_saved_by_cache']}")
        
        # Should have saved at least the length of prompt1 tokens
        assert stats["cache_hits"] == 1, "Should hit cache for shared prefix"
        assert stats["total_tokens_saved_by_cache"] >= len(tokens1) - 2, \
            f"Should save approximately {len(tokens1)} tokens from prefix"
    
    def test_shared_system_prompt_cache(self, fresh_engine):
        """
        Test that shared system prompts get cached.
        
        Common scenario: Multiple users with same system prompt but different queries.
        
        Note: Cache is populated AFTER requests complete, so we need to process
        the first request before adding the second to see cache benefits.
        """
        engine = fresh_engine
        
        system_prompt = "You are a helpful AI assistant. Answer questions concisely."
        query1 = "What is 2+2?"
        query2 = "What is the capital of Italy?"
        
        full_prompt1 = f"{system_prompt}\nUser: {query1}\nAssistant:"
        full_prompt2 = f"{system_prompt}\nUser: {query2}\nAssistant:"
        
        tokens1 = engine.model.tokenize(full_prompt1.encode(), add_bos=True)
        tokens2 = engine.model.tokenize(full_prompt2.encode(), add_bos=True)
        
        system_tokens = engine.model.tokenize(system_prompt.encode(), add_bos=True)
        
        print(f"\nSystem prompt tokens: {len(system_tokens)}")
        print(f"Full prompt 1 tokens: {len(tokens1)}")
        print(f"Full prompt 2 tokens: {len(tokens2)}")
        
        # First request - add, process, complete (populates cache)
        print("\n--- User 1 Request ---")
        engine.add_request(tokens1, max_new_tokens=5)
        engine.run_until_complete(timeout=60)
        
        # Second request - added AFTER first completes, should hit cache
        print("\n--- User 2 Request (should reuse system prompt cache) ---")
        engine.reset_stats()  # Reset stats but cache remains
        engine.add_request(tokens2, max_new_tokens=5)
        engine.run_until_complete(timeout=60)
        
        stats = engine.get_stats()
        print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"Tokens saved: {stats['total_tokens_saved_by_cache']}")
        
        # Should have cache hit for shared prefix
        assert stats["cache_hits"] == 1, f"Expected 1 cache hit, got {stats['cache_hits']}"
        assert stats["total_tokens_saved_by_cache"] >= len(system_tokens) * 0.5


class TestContinuousBatching:
    """Tests for continuous batching with real model."""
    
    def test_multiple_requests_batched_together(self, fresh_engine):
        """
        Test that multiple requests are processed in the same batch.
        
        With continuous batching, both requests should complete faster
        than sequential processing.
        """
        engine = fresh_engine
        
        prompts = [
            "The color of the sky is",
            "The capital of Japan is",
        ]
        
        print(f"\nProcessing {len(prompts)} requests with continuous batching...")
        
        # Add all requests
        for prompt in prompts:
            tokens = engine.model.tokenize(prompt.encode(), add_bos=True)
            engine.add_request(tokens, max_new_tokens=5)
        
        # Process all
        start_time = time.perf_counter()
        completed = engine.run_until_complete(timeout=120)
        elapsed = time.perf_counter() - start_time
        
        stats = engine.get_stats()
        
        print(f"\nCompleted {len(completed)} requests in {elapsed:.2f}s")
        print(f"Average time per request: {elapsed/len(prompts):.2f}s")
        print(f"Total tokens - Prefill: {stats['total_prefill_tokens']}, Decode: {stats['total_decode_tokens']}")
        
        assert len(completed) == len(prompts), "All requests should complete"
        assert stats["completed_requests"] == len(prompts)
    
    def test_requests_arrive_at_different_times(self, fresh_engine):
        """
        Test continuous batching where requests arrive while others are decoding.
        
        This is the key scenario: User B arrives while User A is still generating.
        """
        engine = fresh_engine
        
        # User A starts first
        prompt_a = "Write a haiku about nature:"
        tokens_a = engine.model.tokenize(prompt_a.encode(), add_bos=True)
        req_a = engine.add_request(tokens_a, max_new_tokens=10)
        
        print(f"\nUser A started with prompt: '{prompt_a}'")
        
        # Run a few steps for User A
        for _ in range(3):
            engine.step()
        
        status_a = engine.get_request_status(req_a)
        print(f"User A status after 3 steps: {status_a}")
        
        # Now User B arrives while A is still active
        prompt_b = "What is machine learning?"
        tokens_b = engine.model.tokenize(prompt_b.encode(), add_bos=True)
        req_b = engine.add_request(tokens_b, max_new_tokens=5)
        
        print(f"User B arrived with prompt: '{prompt_b}'")
        print("Both requests should now be in the same batch...")
        
        # Continue until complete
        completed = engine.run_until_complete(timeout=120)
        
        stats = engine.get_stats()
        print(f"\nBoth completed! Total: {stats['completed_requests']} requests")
        
        # Both should complete
        assert stats["completed_requests"] == 2
    
    def test_batch_size_limits_respected(self, fresh_engine):
        """
        Test that max_batch_size is respected.
        """
        engine = fresh_engine
        max_batch = engine.max_batch_size
        
        # Add more requests than max_batch_size
        num_requests = max_batch + 2
        
        print(f"\nTesting batch size limit: {max_batch}")
        print(f"Adding {num_requests} requests...")
        
        for i in range(num_requests):
            prompt = f"Question {i}: What is {i}+{i}?"
            tokens = engine.model.tokenize(prompt.encode(), add_bos=True)
            engine.add_request(tokens, max_new_tokens=3)
        
        # Check initial state
        stats = engine.get_stats()
        print(f"Pending: {stats['pending_requests']}, Active: {stats['active_requests']}")
        
        # Active should be at most max_batch_size
        assert stats["active_requests"] <= max_batch
        assert stats["pending_requests"] == num_requests - stats["active_requests"]
        
        # Complete all
        completed = engine.run_until_complete(timeout=120)
        
        stats = engine.get_stats()
        assert stats["completed_requests"] == num_requests


class TestCombinedBenefits:
    """Tests showing combined benefits of RadixAttention + Continuous Batching."""
    
    def test_chat_conversation_simulation(self, fresh_engine):
        """
        Simulate a chat conversation with multiple users.
        
        This demonstrates real-world benefits:
        1. Shared system prompt gets cached (RadixAttention)
        2. Multiple users processed together (Continuous Batching)
        
        Note: To see cache benefits, we process sequentially so each request
        completes before the next is added.
        """
        engine = fresh_engine
        
        system_prompt = "You are a helpful assistant."
        
        # Multiple users with same system prompt
        conversations = [
            ("User1", f"{system_prompt}\nUser: Hello!\nAssistant:"),
            ("User2", f"{system_prompt}\nUser: Hi there!\nAssistant:"),
            ("User3", f"{system_prompt}\nUser: Hey!\nAssistant:"),
        ]
        
        print(f"\n{'='*60}")
        print("CHAT CONVERSATION SIMULATION")
        print(f"{'='*60}")
        print(f"System prompt: '{system_prompt}'")
        print(f"Number of users: {len(conversations)}")
        
        # Process first user (cold start - no cache)
        user1_id, prompt1 = conversations[0]
        tokens1 = engine.model.tokenize(prompt1.encode(), add_bos=True)
        print(f"\nProcessing {user1_id} (cold start, no cache)...")
        engine.add_request(tokens1, max_new_tokens=5)
        engine.run_until_complete(timeout=60)
        
        cold_stats = engine.get_stats()
        print(f"  Cache hits: {cold_stats['cache_hits']}, misses: {cold_stats['cache_misses']}")
        
        # Process remaining users (should benefit from cache)
        engine.reset_stats()  # Reset stats but keep cache
        
        for user_id, prompt in conversations[1:]:
            tokens = engine.model.tokenize(prompt.encode(), add_bos=True)
            print(f"\nProcessing {user_id}...")
            engine.add_request(tokens, max_new_tokens=5)
            engine.run_until_complete(timeout=60)
        
        warm_stats = engine.get_stats()
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Total completed: {cold_stats['completed_requests'] + warm_stats['completed_requests']}")
        print(f"Cold run - Cache hits: {cold_stats['cache_hits']}, misses: {cold_stats['cache_misses']}")
        print(f"Warm run - Cache hits: {warm_stats['cache_hits']}, misses: {warm_stats['cache_misses']}")
        print(f"Tokens saved by cache: {warm_stats['total_tokens_saved_by_cache']}")
        
        # Verify results
        total_completed = cold_stats['completed_requests'] + warm_stats['completed_requests']
        assert total_completed == len(conversations)
        
        # Should have cache hits for shared system prompt in warm run
        assert warm_stats["cache_hits"] >= len(conversations) - 2, \
            f"Expected at least {len(conversations)-2} cache hits in warm run, got {warm_stats['cache_hits']}"
    
    def test_performance_comparison(self, fresh_engine):
        """
        Compare performance metrics between first (cold) and second (warm) runs.
        
        The warm run should be significantly faster due to cache hits.
        """
        engine = fresh_engine
        
        prompts = [
            "The Earth orbits around",
            "Water freezes at",
            "The largest planet is",
        ]
        
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON: Cold vs Warm Cache")
        print(f"{'='*60}")
        
        # Cold run
        print("\n--- COLD RUN (Cache Empty) ---")
        for prompt in prompts:
            tokens = engine.model.tokenize(prompt.encode(), add_bos=True)
            engine.add_request(tokens, max_new_tokens=5)
        
        start = time.perf_counter()
        engine.run_until_complete(timeout=120)
        cold_time = time.perf_counter() - start
        
        cold_stats = engine.get_stats()
        print(f"Time: {cold_time:.2f}s")
        print(f"Cache hits: {cold_stats['cache_hits']}, misses: {cold_stats['cache_misses']}")
        
        # Warm run - same prompts
        print("\n--- WARM RUN (Cache Populated) ---")
        engine.reset_stats()
        
        for prompt in prompts:
            tokens = engine.model.tokenize(prompt.encode(), add_bos=True)
            engine.add_request(tokens, max_new_tokens=5)
        
        start = time.perf_counter()
        engine.run_until_complete(timeout=120)
        warm_time = time.perf_counter() - start
        
        warm_stats = engine.get_stats()
        print(f"Time: {warm_time:.2f}s")
        print(f"Cache hits: {warm_stats['cache_hits']}, misses: {warm_stats['cache_misses']}")
        print(f"Tokens saved: {warm_stats['total_tokens_saved_by_cache']}")
        
        # Calculate speedup
        if warm_time > 0:
            speedup = cold_time / warm_time
            print(f"\nSpeedup: {speedup:.2f}×")
        
        # Warm run should have all cache hits
        assert warm_stats["cache_hits"] == len(prompts), "Warm run should hit cache for all"
        assert warm_stats["cache_misses"] == 0, "Warm run should have no misses"


def test_engine_initialization():
    """Test that engine can be initialized with real model."""
    if not model_exists:
        pytest.skip(f"Model not found at {MODEL_PATH}")
    
    from aegis_router.core.radix_attention_engine import RadixAttentionEngine
    
    print(f"\nInitializing engine with model: {MODEL_PATH}")
    engine = RadixAttentionEngine(
        model_path=MODEL_PATH,
        max_total_tokens=1024,
        max_batch_size=2,
        max_new_tokens=5,
    )
    
    assert engine.model is not None
    assert engine.max_batch_size == 2
    print("Engine initialized successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
