"""
Approximate matching for fuzzy prompt matching.

This module implements Locality Sensitive Hashing (LSH) techniques to find
approximately similar prompts when exact prefix matching fails. This handles
cases where prompts are slightly different (typos, formatting, minor edits)
but semantically equivalent and likely to benefit from KV cache reuse.

Techniques implemented:
1. MinHash + LSH for Jaccard similarity estimation
2. SimHash for near-duplicate detection
3. Hybrid approach combining both

Trade-offs:
- MinHash: Better for longer sequences, tunable false positive rate
- SimHash: Faster, better for near-exact matches with small edits
"""

from __future__ import annotations

import hashlib
import heapq
import random
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mmh3

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class FuzzyMatchResult:
    """Result of an approximate match."""

    worker_id: str | None
    similarity: float  # 0.0 to 1.0
    matched_tokens: tuple[int, ...]
    confidence: str  # "high", "medium", "low"
    method: str  # "minhash", "simhash", "exact"

    def is_usable(self, threshold: float = 0.85) -> bool:
        """Check if the match is usable based on similarity threshold."""
        return self.similarity >= threshold


class MinHashSignature:
    """
    MinHash signature for Jaccard similarity estimation.

    Uses k hash functions to create a compact signature.
    Similar sets have similar signatures with high probability.
    """

    def __init__(self, num_hashes: int = 128, num_bands: int = 16, seed: int = 42):
        """
        Initialize MinHash parameters.

        Args:
            num_hashes: Number of hash functions (signature length)
            num_bands: Number of bands for LSH (affects collision probability)
            seed: Random seed for reproducibility
        """
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.seed = seed

        # Generate random seeds for each hash function
        rng = random.Random(seed)
        self.hash_seeds = [rng.randint(0, 2**32 - 1) for _ in range(num_hashes)]

    def compute(self, tokens: tuple[int, ...]) -> list[int]:
        """
        Compute MinHash signature for token sequence.

        Uses k-shingles (n-grams) of tokens for locality sensitivity.
        """
        if not tokens:
            return [0] * self.num_hashes

        # Create 3-gram shingles
        shingles = set()
        for i in range(len(tokens) - 2):
            shingle = (tokens[i], tokens[i + 1], tokens[i + 2])
            shingles.add(shingle)

        if len(tokens) < 3:
            # For short sequences, use individual tokens
            shingles = {(t,) for t in tokens}

        signature = []
        for hash_seed in self.hash_seeds:
            min_hash = float("inf")
            for shingle in shingles:
                # Combine shingle into a single hash
                shingle_bytes = str(shingle).encode()
                h = mmh3.hash(shingle_bytes, hash_seed)
                min_hash = min(min_hash, h)
            signature.append(min_hash)

        return signature

    def get_bands(self, signature: list[int]) -> list[tuple[int, ...]]:
        """Split signature into bands for LSH."""
        bands = []
        for i in range(self.num_bands):
            start = i * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(signature[start:end])
            bands.append(band)
        return bands

    def estimate_similarity(self, sig1: list[int], sig2: list[int]) -> float:
        """Estimate Jaccard similarity from two signatures."""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / self.num_hashes


class MinHashMatcher:
    """
    LSH-based approximate matcher using MinHash.

    Efficiently finds similar prompts by hashing signatures into buckets.
    Two prompts hash to the same bucket iff they have at least one band in common.
    """

    def __init__(
        self,
        num_hashes: int = 128,
        num_bands: int = 16,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize the MinHash matcher.

        Args:
            num_hashes: Number of hash functions
            num_bands: Number of bands for LSH (more bands = stricter matching)
            similarity_threshold: Minimum similarity for a match
        """
        self.minhash = MinHashSignature(num_hashes, num_bands)
        self.threshold = similarity_threshold

        # LSH buckets: band_hash -> {signature -> worker_id}
        self.buckets: dict[tuple[int, ...], dict[tuple[int, ...], str]] = {}

        # Store all signatures for exact similarity calculation
        self.signatures: dict[str, tuple[list[int], tuple[int, ...]]] = {}

        self._lock = threading.RLock()

    def add(self, tokens: tuple[int, ...], worker_id: str) -> None:
        """Add a token sequence to the index."""
        signature = self.minhash.compute(tokens)
        bands = self.minhash.get_bands(signature)

        with self._lock:
            # Store signature for later similarity calculation
            sig_key = tuple(signature)
            self.signatures[worker_id] = (signature, tokens)

            # Add to LSH buckets
            for band in bands:
                if band not in self.buckets:
                    self.buckets[band] = {}
                self.buckets[band][sig_key] = worker_id

    def remove(self, worker_id: str) -> bool:
        """Remove a worker's signature from the index."""
        with self._lock:
            if worker_id not in self.signatures:
                return False

            signature, _ = self.signatures[worker_id]
            sig_key = tuple(signature)
            bands = self.minhash.get_bands(signature)

            for band in bands:
                if band in self.buckets and sig_key in self.buckets[band]:
                    del self.buckets[band][sig_key]
                    if not self.buckets[band]:
                        del self.buckets[band]

            del self.signatures[worker_id]
            return True

    def find_best_match(self, tokens: tuple[int, ...]) -> FuzzyMatchResult:
        """
        Find the best matching worker for the given tokens.

        Uses LSH for candidate generation, then exact similarity for ranking.
        """
        query_sig = self.minhash.compute(tokens)
        query_bands = self.minhash.get_bands(query_sig)

        with self._lock:
            # Collect candidates from LSH buckets
            candidates: dict[str, float] = {}

            for band in query_bands:
                if band in self.buckets:
                    for sig_key, worker_id in self.buckets[band].items():
                        if worker_id not in candidates:
                            candidates[worker_id] = 0.0
                        candidates[worker_id] += 1

            if not candidates:
                return FuzzyMatchResult(
                    worker_id=None,
                    similarity=0.0,
                    matched_tokens=(),
                    confidence="none",
                    method="minhash",
                )

            # Calculate exact similarity for top candidates
            best_worker = None
            best_sim = 0.0

            for worker_id in candidates:
                if worker_id not in self.signatures:
                    continue
                sig, _ = self.signatures[worker_id]
                sim = self.minhash.estimate_similarity(query_sig, sig)
                if sim > best_sim:
                    best_sim = sim
                    best_worker = worker_id

            if best_sim < self.threshold:
                return FuzzyMatchResult(
                    worker_id=None,
                    similarity=best_sim,
                    matched_tokens=(),
                    confidence="low",
                    method="minhash",
                )

            _, matched_tokens = self.signatures[best_worker]
            confidence = "high" if best_sim > 0.9 else "medium"

            return FuzzyMatchResult(
                worker_id=best_worker,
                similarity=best_sim,
                matched_tokens=matched_tokens,
                confidence=confidence,
                method="minhash",
            )


class SimHashIndex:
    """
    SimHash-based near-duplicate detection.

    SimHash is a locality-sensitive hash that produces similar hashes for
    similar inputs. It's particularly effective for detecting near-duplicates
    (e.g., prompts with minor edits, whitespace differences).

    The Hamming distance between two SimHashes approximates the similarity
    between the original documents.
    """

    def __init__(self, hash_bits: int = 64, max_hamming_distance: int = 6):
        """
        Initialize SimHash index.

        Args:
            hash_bits: Number of bits in the hash (64 or 128)
            max_hamming_distance: Maximum Hamming distance for a match
        """
        self.hash_bits = hash_bits
        self.max_distance = max_hamming_distance

        # Index: hamming_prefix -> {hash -> worker_id}
        # We use multiple prefix lengths for efficient lookup
        self.index: dict[int, dict[int, str]] = {}
        self.hashes: dict[str, int] = {}  # worker_id -> simhash

        self._lock = threading.RLock()

    def _compute_simhash(self, tokens: tuple[int, ...]) -> int:
        """Compute SimHash for token sequence."""
        if not tokens:
            return 0

        # Create feature hashes for each token and position
        feature_hashes = []
        for i, token in enumerate(tokens):
            # Combine token value and position
            feature = f"{i}:{token}".encode()
            h = mmh3.hash64(feature)[0]  # Get 64-bit hash
            feature_hashes.append(h)

        # Compute weighted sum of bit vectors
        vec = [0] * self.hash_bits
        for h in feature_hashes:
            for i in range(self.hash_bits):
                bit = (h >> i) & 1
                vec[i] += 1 if bit else -1

        # Build final hash from sign of each dimension
        simhash = 0
        for i in range(self.hash_bits):
            if vec[i] > 0:
                simhash |= 1 << i

        return simhash

    def _hamming_distance(self, a: int, b: int) -> int:
        """Compute Hamming distance between two hashes."""
        x = a ^ b
        count = 0
        while x:
            count += x & 1
            x >>= 1
        return count

    def add(self, tokens: tuple[int, ...], worker_id: str) -> None:
        """Add a token sequence to the index."""
        simhash = self._compute_simhash(tokens)

        with self._lock:
            self.hashes[worker_id] = simhash

            # Index by multiple prefix lengths for efficient lookup
            for prefix_bits in [16, 32, 48]:
                prefix = simhash >> (self.hash_bits - prefix_bits)
                if prefix_bits not in self.index:
                    self.index[prefix_bits] = {}
                self.index[prefix_bits][prefix] = worker_id

    def remove(self, worker_id: str) -> bool:
        """Remove a worker from the index."""
        with self._lock:
            if worker_id not in self.hashes:
                return False

            simhash = self.hashes[worker_id]

            for prefix_bits in [16, 32, 48]:
                prefix = simhash >> (self.hash_bits - prefix_bits)
                idx = self.index.get(prefix_bits, {})
                if prefix in idx and idx[prefix] == worker_id:
                    del idx[prefix]

            del self.hashes[worker_id]
            return True

    def find_best_match(self, tokens: tuple[int, ...]) -> FuzzyMatchResult:
        """Find the best matching worker using SimHash."""
        query_hash = self._compute_simhash(tokens)

        with self._lock:
            # Search for nearest neighbor
            best_worker = None
            best_distance = self.max_distance + 1

            for worker_id, h in self.hashes.items():
                dist = self._hamming_distance(query_hash, h)
                if dist < best_distance:
                    best_distance = dist
                    best_worker = worker_id

            if best_worker is None or best_distance > self.max_distance:
                return FuzzyMatchResult(
                    worker_id=None,
                    similarity=0.0,
                    matched_tokens=(),
                    confidence="none",
                    method="simhash",
                )

            # Convert Hamming distance to similarity
            similarity = 1.0 - (best_distance / self.hash_bits)

            # Determine confidence
            if best_distance <= 3:
                confidence = "high"
            elif best_distance <= 6:
                confidence = "medium"
            else:
                confidence = "low"

            return FuzzyMatchResult(
                worker_id=best_worker,
                similarity=similarity,
                matched_tokens=(),  # SimHash doesn't preserve token info
                confidence=confidence,
                method="simhash",
            )


class ApproximateMatcher:
    """
    Hybrid approximate matcher combining multiple techniques.

    Strategy:
    1. First try SimHash (fast, good for near-duplicates)
    2. If no high-confidence match, try MinHash (better for partial similarity)
    3. Return the best result based on confidence and similarity
    """

    def __init__(
        self,
        minhash_threshold: float = 0.8,
        simhash_threshold: float = 0.9,
        enable_minhash: bool = True,
        enable_simhash: bool = True,
    ):
        """
        Initialize the hybrid approximate matcher.

        Args:
            minhash_threshold: Minimum Jaccard similarity for MinHash matches
            simhash_threshold: Minimum similarity for SimHash matches
            enable_minhash: Whether to enable MinHash matching
            enable_simhash: Whether to enable SimHash matching
        """
        self.minhash = MinHashMatcher(similarity_threshold=minhash_threshold)
        self.simhash = SimHashIndex(max_hamming_distance=6)
        self.minhash_threshold = minhash_threshold
        self.simhash_threshold = simhash_threshold
        self.enable_minhash = enable_minhash
        self.enable_simhash = enable_simhash

    def add(self, tokens: tuple[int, ...], worker_id: str) -> None:
        """Add a token sequence to all enabled indices."""
        if self.enable_simhash:
            self.simhash.add(tokens, worker_id)
        if self.enable_minhash:
            self.minhash.add(tokens, worker_id)

    def remove(self, worker_id: str) -> None:
        """Remove a worker from all indices."""
        if self.enable_simhash:
            self.simhash.remove(worker_id)
        if self.enable_minhash:
            self.minhash.remove(worker_id)

    def find_best_match(self, tokens: tuple[int, ...]) -> FuzzyMatchResult:
        """
        Find the best matching worker using all enabled methods.

        Returns the highest-confidence match across all methods.
        """
        results = []

        if self.enable_simhash:
            simhash_result = self.simhash.find_best_match(tokens)
            if simhash_result.is_usable(self.simhash_threshold):
                results.append(simhash_result)

        if self.enable_minhash:
            minhash_result = self.minhash.find_best_match(tokens)
            if minhash_result.is_usable(self.minhash_threshold):
                results.append(minhash_result)

        if not results:
            # Return the best non-usable result for diagnostics
            if self.enable_simhash:
                return self.simhash.find_best_match(tokens)
            if self.enable_minhash:
                return self.minhash.find_best_match(tokens)
            return FuzzyMatchResult(
                worker_id=None,
                similarity=0.0,
                matched_tokens=(),
                confidence="none",
                method="none",
            )

        # Return the highest similarity result
        return max(results, key=lambda r: r.similarity)

    def get_stats(self) -> dict:
        """Get statistics about the matcher."""
        return {
            "minhash_signatures": len(self.minhash.signatures),
            "minhash_buckets": len(self.minhash.buckets),
            "simhash_entries": len(self.simhash.hashes),
        }
