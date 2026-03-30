"""
Tests for Radix Trie implementation.
"""

import pytest

from aegis_router.core.radix_trie import MatchResult, RadixNode, RadixTrie


class TestRadixNode:
    """Test RadixNode functionality."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = RadixNode(token_ids=(1, 2, 3))
        assert node.token_ids == (1, 2, 3)
        assert node.children == {}
        assert node.workers == set()
        assert node.ref_count == 0

    def test_add_worker(self):
        """Test adding workers to node."""
        node = RadixNode(token_ids=(1, 2))
        node.add_worker("worker-1")
        assert "worker-1" in node.workers

        node.add_worker("worker-2")
        assert "worker-1" in node.workers
        assert "worker-2" in node.workers

    def test_remove_worker(self):
        """Test removing workers from node."""
        node = RadixNode(token_ids=(1, 2))
        node.add_worker("worker-1")

        is_empty = node.remove_worker("worker-1")
        assert is_empty
        assert "worker-1" not in node.workers

    def test_reference_counting(self):
        """Test reference counting."""
        node = RadixNode(token_ids=(1, 2))

        node.acquire()
        assert node.ref_count == 1

        node.acquire()
        assert node.ref_count == 2

        node.release()
        assert node.ref_count == 1

        node.release()
        assert node.ref_count == 0


class TestRadixTrie:
    """Test RadixTrie functionality."""

    def test_empty_trie(self):
        """Test operations on empty trie."""
        trie = RadixTrie()
        result = trie.match((1, 2, 3))

        assert result.is_miss
        assert result.matched_length == 0
        assert result.workers == frozenset()

    def test_single_insert_and_match(self):
        """Test basic insert and match."""
        trie = RadixTrie()
        tokens = (1, 2, 3, 4, 5)

        trie.insert(tokens, "worker-1")
        result = trie.match(tokens)

        assert result.matched_length == 5
        assert result.is_full_match
        assert "worker-1" in result.workers
        assert result.hit_ratio == 1.0

    def test_prefix_match(self):
        """Test matching a prefix of stored sequence."""
        trie = RadixTrie()
        stored = (1, 2, 3, 4, 5)
        query = (1, 2, 3)

        trie.insert(stored, "worker-1")
        result = trie.match(query)

        assert result.matched_length == 3
        assert result.is_full_match  # Query fully matched
        assert "worker-1" in result.workers

    def test_partial_match(self):
        """Test partial match where query extends stored sequence."""
        trie = RadixTrie()
        stored = (1, 2, 3)
        query = (1, 2, 3, 4, 5)

        trie.insert(stored, "worker-1")
        result = trie.match(query)

        assert result.matched_length == 3
        assert not result.is_full_match
        assert result.remaining_tokens == (4, 5)
        assert result.hit_ratio == 0.6

    def test_multiple_workers_same_prefix(self):
        """Test multiple workers with same prefix."""
        trie = RadixTrie()
        tokens = (1, 2, 3, 4, 5)

        trie.insert(tokens, "worker-1")
        trie.insert(tokens, "worker-2")

        result = trie.match(tokens)

        assert "worker-1" in result.workers
        assert "worker-2" in result.workers
        assert len(result.workers) == 2

    def test_different_prefixes(self):
        """Test storing different prefixes."""
        trie = RadixTrie()

        trie.insert((1, 2, 3), "worker-1")
        trie.insert((4, 5, 6), "worker-2")

        result1 = trie.match((1, 2, 3))
        assert "worker-1" in result1.workers
        assert "worker-2" not in result1.workers

        result2 = trie.match((4, 5, 6))
        assert "worker-2" in result2.workers
        assert "worker-1" not in result2.workers

    def test_shared_prefix_splitting(self):
        """Test that shared prefixes are properly split."""
        trie = RadixTrie()

        # Insert sequences with shared prefix
        trie.insert((1, 2, 3, 4), "worker-1")
        trie.insert((1, 2, 5, 6), "worker-2")

        # Both should be matchable
        result1 = trie.match((1, 2, 3, 4))
        assert result1.matched_length == 4
        assert "worker-1" in result1.workers

        result2 = trie.match((1, 2, 5, 6))
        assert result2.matched_length == 4
        assert "worker-2" in result2.workers

        # Common prefix should match both
        common = trie.match((1, 2))
        assert common.matched_length == 2
        # At the common prefix node, both workers should be present
        assert len(common.workers) == 0  # Intermediate node has no workers

    def test_node_splitting(self):
        """Test node splitting on partial match."""
        trie = RadixTrie()

        # Insert first sequence
        trie.insert((1, 2, 3, 4), "worker-1")
        # Insert overlapping sequence
        trie.insert((1, 2, 5, 6), "worker-2")

        # Verify structure by matching
        result = trie.match((1, 2))
        assert result.matched_length == 2

    def test_remove_worker(self):
        """Test removing a worker."""
        trie = RadixTrie()
        tokens = (1, 2, 3, 4, 5)

        trie.insert(tokens, "worker-1")
        trie.insert(tokens, "worker-2")

        count = trie.remove_worker("worker-1")
        assert count > 0

        result = trie.match(tokens)
        assert "worker-1" not in result.workers
        assert "worker-2" in result.workers

    def test_find_best_worker(self):
        """Test finding best worker with load consideration."""
        trie = RadixTrie()
        tokens = (1, 2, 3, 4, 5)

        trie.insert(tokens, "worker-1")
        trie.insert(tokens, "worker-2")

        # worker-1 has higher load
        loads = {"worker-1": 0.8, "worker-2": 0.3}

        best_worker, result = trie.find_best_worker(tokens, loads)

        assert best_worker == "worker-2"  # Lower load wins
        assert result.hit_ratio == 1.0

    def test_lru_eviction(self):
        """Test LRU eviction."""
        trie = RadixTrie()

        # Insert sequences
        trie.insert((1, 2, 3), "worker-1")
        trie.insert((4, 5, 6), "worker-2")

        initial_tokens = len(trie)
        assert initial_tokens > 0

        # Evict to 0
        evicted = trie.evict_lru(0)
        assert evicted >= initial_tokens

    def test_stats_tracking(self):
        """Test statistics tracking."""
        trie = RadixTrie()

        initial_stats = trie.get_stats()
        assert initial_stats["insertions"] == 0
        assert initial_stats["matches"] == 0

        trie.insert((1, 2, 3), "worker-1")
        trie.match((1, 2, 3))

        stats = trie.get_stats()
        assert stats["insertions"] == 1
        assert stats["matches"] == 1

    def test_long_sequence(self):
        """Test with long token sequences."""
        trie = RadixTrie()
        tokens = tuple(range(1000))  # 1000 tokens

        trie.insert(tokens, "worker-1")
        result = trie.match(tokens)

        assert result.matched_length == 1000
        assert result.is_full_match

    def test_empty_sequence(self):
        """Test with empty sequence."""
        trie = RadixTrie()

        result = trie.match(())
        assert result.is_miss
        assert result.matched_length == 0

    def test_single_token(self):
        """Test with single token."""
        trie = RadixTrie()

        trie.insert((42,), "worker-1")
        result = trie.match((42,))

        assert result.matched_length == 1
        assert "worker-1" in result.workers


class TestRadixTrieConcurrency:
    """Test thread safety."""

    def test_concurrent_inserts(self):
        """Test concurrent insertions."""
        import threading

        trie = RadixTrie()
        tokens_list = [(i, i + 1, i + 2) for i in range(100)]
        errors = []

        def insert_worker(tokens, worker_id):
            try:
                trie.insert(tokens, worker_id)
            except Exception as e:
                errors.append(e)

        threads = []
        for i, tokens in enumerate(tokens_list):
            t = threading.Thread(target=insert_worker, args=(tokens, f"worker-{i}"))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert not errors

        # Verify all were inserted
        for i, tokens in enumerate(tokens_list):
            result = trie.match(tokens)
            assert result.matched_length == 3, f"Failed for tokens {tokens}"


class TestMatchResult:
    """Test MatchResult dataclass."""

    def test_full_match_property(self):
        """Test is_full_match property."""
        full = MatchResult(
            matched_tokens=(1, 2, 3),
            matched_length=3,
            workers=frozenset(["w1"]),
            remaining_tokens=(),
            hit_ratio=1.0,
        )
        assert full.is_full_match

        partial = MatchResult(
            matched_tokens=(1, 2),
            matched_length=2,
            workers=frozenset(["w1"]),
            remaining_tokens=(3,),
            hit_ratio=0.67,
        )
        assert not partial.is_full_match

    def test_miss_property(self):
        """Test is_miss property."""
        miss = MatchResult(
            matched_tokens=(),
            matched_length=0,
            workers=frozenset(),
            remaining_tokens=(1, 2, 3),
            hit_ratio=0.0,
        )
        assert miss.is_miss

        hit = MatchResult(
            matched_tokens=(1,),
            matched_length=1,
            workers=frozenset(["w1"]),
            remaining_tokens=(2, 3),
            hit_ratio=0.33,
        )
        assert not hit.is_miss
