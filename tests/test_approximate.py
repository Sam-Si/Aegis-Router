"""
Tests for approximate matching implementations.
"""

import pytest

from aegis_router.matching.approximate import (
    ApproximateMatcher,
    FuzzyMatchResult,
    MinHashMatcher,
    MinHashSignature,
    SimHashIndex,
)


class TestMinHashSignature:
    """Test MinHash signature computation."""

    def test_empty_sequence(self):
        """Test with empty sequence."""
        mh = MinHashSignature(num_hashes=64)
        sig = mh.compute(())
        assert len(sig) == 64
        assert all(v == 0 for v in sig)

    def test_identical_sequences(self):
        """Test identical sequences have same signature."""
        mh = MinHashSignature(num_hashes=64)
        tokens = (1, 2, 3, 4, 5)

        sig1 = mh.compute(tokens)
        sig2 = mh.compute(tokens)

        assert sig1 == sig2

    def test_similar_sequences_high_similarity(self):
        """Test similar sequences have high estimated similarity."""
        mh = MinHashSignature(num_hashes=128)

        # Two sequences with minor differences
        seq1 = tuple(range(100))
        seq2 = tuple(range(95)) + (1000, 1001, 1002, 1003, 1004)

        sig1 = mh.compute(seq1)
        sig2 = mh.compute(seq2)

        sim = mh.estimate_similarity(sig1, sig2)
        assert sim > 0.8  # Should be quite similar

    def test_different_sequences_low_similarity(self):
        """Test different sequences have low similarity."""
        mh = MinHashSignature(num_hashes=128)

        seq1 = tuple(range(100))
        seq2 = tuple(range(100, 200))

        sig1 = mh.compute(seq1)
        sig2 = mh.compute(seq2)

        sim = mh.estimate_similarity(sig1, sig2)
        assert sim < 0.3  # Should be quite different

    def test_band_generation(self):
        """Test signature banding."""
        mh = MinHashSignature(num_hashes=128, num_bands=16)
        sig = list(range(128))

        bands = mh.get_bands(sig)

        assert len(bands) == 16
        assert len(bands[0]) == 8  # 128 / 16 = 8 rows per band


class TestMinHashMatcher:
    """Test MinHash-based matching."""

    def test_add_and_find_exact(self):
        """Test adding and finding exact match."""
        matcher = MinHashMatcher()
        tokens = tuple(range(50))

        matcher.add(tokens, "worker-1")
        result = matcher.find_best_match(tokens)

        assert result.worker_id == "worker-1"
        assert result.similarity > 0.9
        assert result.is_usable()

    def test_find_similar(self):
        """Test finding similar but not exact match."""
        matcher = MinHashMatcher(similarity_threshold=0.7)

        # Original sequence
        original = tuple(range(100))
        # Slightly modified (10% different)
        modified = tuple(range(90)) + (1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009)

        matcher.add(original, "worker-1")
        result = matcher.find_best_match(modified)

        assert result.worker_id == "worker-1"
        assert result.similarity > 0.8

    def test_no_match(self):
        """Test when no good match exists."""
        matcher = MinHashMatcher()

        matcher.add(tuple(range(100)), "worker-1")
        result = matcher.find_best_match(tuple(range(100, 200)))

        assert result.worker_id is None
        assert not result.is_usable()

    def test_remove_worker(self):
        """Test removing a worker."""
        matcher = MinHashMatcher()
        tokens = tuple(range(50))

        matcher.add(tokens, "worker-1")
        assert matcher.remove("worker-1")

        result = matcher.find_best_match(tokens)
        assert result.worker_id is None

    def test_multiple_workers_best_match(self):
        """Test finding best match among multiple workers."""
        matcher = MinHashMatcher(similarity_threshold=0.5)

        # Add two different sequences with significant overlap
        seq1 = tuple(range(100))
        seq2 = tuple(range(80, 180))  # 20 token overlap with seq1
        matcher.add(seq1, "worker-1")
        matcher.add(seq2, "worker-2")

        # Query closer to worker-1 (first 70 tokens of seq1)
        query = tuple(range(70))
        result = matcher.find_best_match(query)

        # Should match worker-1 due to higher overlap
        assert result.worker_id == "worker-1"


class TestSimHashIndex:
    """Test SimHash-based matching."""

    def test_add_and_find_exact(self):
        """Test adding and finding exact match."""
        index = SimHashIndex()
        tokens = tuple(range(50))

        index.add(tokens, "worker-1")
        result = index.find_best_match(tokens)

        assert result.worker_id == "worker-1"
        assert result.similarity == 1.0

    def test_near_duplicate_detection(self):
        """Test detecting near-duplicates with small edits."""
        index = SimHashIndex(max_hamming_distance=6)

        # Original sequence
        original = tuple(range(100))
        # Single token change at the end
        modified = tuple(range(99)) + (9999,)

        index.add(original, "worker-1")
        result = index.find_best_match(modified)

        assert result.worker_id == "worker-1"
        assert result.similarity > 0.9

    def test_no_match_too_different(self):
        """Test when sequences are too different."""
        index = SimHashIndex(max_hamming_distance=6)

        index.add(tuple(range(100)), "worker-1")
        result = index.find_best_match(tuple(range(100, 200)))

        assert result.worker_id is None

    def test_hamming_distance(self):
        """Test Hamming distance calculation."""
        index = SimHashIndex()

        # Same number
        assert index._hamming_distance(0b1111, 0b1111) == 0

        # One bit different
        assert index._hamming_distance(0b1111, 0b1110) == 1

        # All bits different
        assert index._hamming_distance(0b1111, 0b0000) == 4


class TestApproximateMatcher:
    """Test hybrid approximate matcher."""

    def test_exact_match(self):
        """Test exact match detection."""
        matcher = ApproximateMatcher()
        tokens = tuple(range(100))

        matcher.add(tokens, "worker-1")
        result = matcher.find_best_match(tokens)

        assert result.worker_id == "worker-1"
        assert result.similarity > 0.9

    def test_near_match(self):
        """Test near-match detection."""
        matcher = ApproximateMatcher()

        original = tuple(range(100))
        # Small modification
        modified = tuple(range(95)) + (1000, 1001, 1002, 1003, 1004)

        matcher.add(original, "worker-1")
        result = matcher.find_best_match(modified)

        assert result.worker_id == "worker-1"
        assert result.confidence in ("high", "medium")

    def test_no_match(self):
        """Test when no match exists."""
        matcher = ApproximateMatcher()

        matcher.add(tuple(range(100)), "worker-1")
        result = matcher.find_best_match(tuple(range(1000, 1100)))

        assert result.worker_id is None
        assert result.confidence == "none"

    def test_remove_worker(self):
        """Test removing a worker."""
        matcher = ApproximateMatcher()
        tokens = tuple(range(50))

        matcher.add(tokens, "worker-1")
        matcher.remove("worker-1")

        result = matcher.find_best_match(tokens)
        assert result.worker_id is None

    def test_get_stats(self):
        """Test statistics retrieval."""
        matcher = ApproximateMatcher()

        matcher.add(tuple(range(50)), "worker-1")
        matcher.add(tuple(range(50, 100)), "worker-2")

        stats = matcher.get_stats()
        assert stats["minhash_signatures"] == 2
        assert stats["simhash_entries"] == 2


class TestFuzzyMatchResult:
    """Test FuzzyMatchResult dataclass."""

    def test_is_usable_threshold(self):
        """Test usability threshold."""
        result_high = FuzzyMatchResult(
            worker_id="w1",
            similarity=0.9,
            matched_tokens=(1, 2, 3),
            confidence="high",
            method="minhash",
        )
        assert result_high.is_usable(threshold=0.85)

        result_low = FuzzyMatchResult(
            worker_id="w1",
            similarity=0.7,
            matched_tokens=(1, 2, 3),
            confidence="low",
            method="minhash",
        )
        assert not result_low.is_usable(threshold=0.85)
