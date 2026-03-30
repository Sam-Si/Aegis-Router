# Aegis-Router: Cache-Aware LLM Orchestrator

A production-grade request router for LLM inference that uses **Radix Trie-based prefix matching** (vLLM-style) combined with **LSH-based approximate matching** to route requests to workers with existing KV caches.

## Features

- **Exact Prefix Matching**: Radix Trie implementation similar to vLLM for O(L) prefix lookups
- **Approximate Matching**: MinHash + SimHash for fuzzy prompt matching (handles typos, minor edits)
- **Load-Aware Routing**: Considers worker load, queue depth, and health status
- **Multi-Worker Support**: Same prefix can exist on multiple workers with intelligent selection
- **Health Monitoring**: Automatic health checking and failover
- **Production Ready**: Thread-safe, LRU eviction, comprehensive metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Request                              │
│                    (token sequence)                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   CacheRouter                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  1. Exact Prefix Match (Radix Trie)                  │  │
│  │     └── O(L) lookup for cached prefixes              │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                    │
│                         ▼                                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  2. Approximate Match (MinHash/SimHash)              │  │
│  │     └── LSH for near-duplicate detection             │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                    │
│                         ▼                                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  3. Load-Based Fallback                              │  │
│  │     └── Route to least-loaded healthy worker         │  │
│  └───────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Selected Worker with KV Cache                   │
│         (maximizing prefix reuse, minimizing compute)        │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

**Required step** - the package must be installed to run tests:

```bash
# Install in editable mode (recommended for development)
pip install -e .

# With dev dependencies (pytest, black, ruff, mypy)
pip install -e ".[dev]"

# With benchmark dependencies
pip install -e ".[benchmark]"
```

**Note**: The package must be installed (`pip install -e .`) before running tests, even when testing locally.

### Running the Server

```bash
# Run as a module (recommended)
python -m aegis_router --port 8080

# Or use the CLI entry point
aegis-router --port 8080

# With auto-reload for development
python -m aegis_router --reload
```

### API Usage

#### Register a Worker

```bash
curl -X POST http://localhost:8080/workers/register \
  -H "Content-Type: application/json" \
  -d '{
    "worker_id": "worker-1",
    "host": "localhost",
    "port": 8001
  }'
```

#### Update Worker Cache

```bash
curl -X POST http://localhost:8080/workers/worker-1/cache \
  -H "Content-Type: application/json" \
  -d '{
    "token_sequences": [[1, 2, 3, 4, 5], [10, 20, 30]]
  }'
```

#### Route a Request

```bash
curl -X POST http://localhost:8080/route \
  -H "Content-Type: application/json" \
  -d '{
    "token_ids": [1, 2, 3, 4, 5, 6, 7],
    "priority": 0
  }'
```

Response:
```json
{
  "worker_id": "worker-1",
  "worker_url": "http://localhost:8001",
  "strategy_used": "exact_prefix",
  "cache_hit_ratio": 0.71,
  "matched_tokens": 5,
  "total_tokens": 7,
  "estimated_tokens_to_compute": 2,
  "confidence": "high",
  "metadata": {}
}
```

#### Send Heartbeat

```bash
curl -X POST http://localhost:8080/workers/worker-1/heartbeat \
  -H "Content-Type: application/json" \
  -d '{
    "load": 0.5,
    "queue_depth": 3
  }'
```

#### Get Statistics

```bash
curl http://localhost:8080/stats
```

## Python API

```python
from aegis_router import CacheRouter

# Create router
router = CacheRouter(
    max_cache_tokens=10_000_000,
    strategy="cache_first",
    enable_approximate=True,
)

# Register workers
router.register_worker("worker-1", "localhost", 8001)
router.register_worker("worker-2", "localhost", 8002)

# Update cache (called by workers after processing)
router.update_worker_cache("worker-1", [
    tuple(tokenizer.encode("System: You are a helpful assistant...")),
])

# Route a request
decision = router.route_request(tuple(token_ids))
print(f"Routed to {decision.worker_id} with {decision.cache_hit_ratio:.0%} cache hit")
```

## Routing Strategies

1. **CACHE_FIRST** (default): Prioritize cache hit over load balancing
2. **LOAD_BALANCED**: Balance cache hit ratio with worker load
3. **LEAST_LOADED**: Always pick the least loaded worker (ignore cache)

## How It Works

### Exact Prefix Matching (Radix Trie)

When a request comes in:
1. Tokenize the prompt into token IDs
2. Walk the Radix Trie to find the longest matching prefix
3. Return workers that have this prefix cached
4. Score workers by cache hit ratio and current load

**Example:**
- Worker A has cached: `[1, 2, 3, 4, 5]`
- Request comes in: `[1, 2, 3, 4, 5, 6, 7]`
- Match: 5/7 tokens (71% hit ratio)
- Route to Worker A, skip computing first 5 tokens

### Approximate Matching (LSH)

When exact match fails:
1. Compute MinHash signature of the query
2. Look up in LSH buckets for candidate matches
3. Calculate exact similarity with candidates
4. Route to best match if above threshold

**Use case:** User sends almost identical prompt with minor edits

## Benchmarks

Run the benchmark suite:

```bash
python benchmark.py
```

Example output:
```
Routing Performance Results
  total_requests: 10000
  elapsed_time_sec: 0.5234
  requests_per_sec: 19105.23
  avg_latency_ms: 0.0523
  cache_hit_rate: 0.7000
  exact_hit_rate: 0.6500

Approximate Matching Results
  total_queries: 1000
  correct_matches: 987
  accuracy: 0.9870
  queries_per_sec: 15420.50
  avg_latency_ms: 0.0648
```

## Testing

```bash
# Run all tests (with coverage, configured in pyproject.toml)
pytest

# Run without coverage
pytest --no-cov

# Run specific test file
pytest tests/test_radix_trie.py -v

# Run functional tests (demonstrates cache routing benefits)
pytest tests/test_functional.py -v -s --no-cov

# Run with parallel execution
pytest -n auto

# View HTML coverage report
open htmlcov/index.html
```

### Functional Tests

The functional tests demonstrate real-world benefits of cache-aware routing:

```bash
# Run all functional demos
pytest tests/test_functional.py -v -s --no-cov

# Or run the standalone demo script
python demo.py
```

**Test 1: Identical Requests**
- 5 identical requests with 1 second delay
- Shows 2x speedup on cached requests
- Demonstrates routing overhead < 1ms

**Test 2: Fuzzy Matching (85% Similarity)**
- Base prompt cached, similar prompt routed via fuzzy match
- Shows 45-50ms benefit from partial cache reuse
- Demonstrates approximate matching working

**Test 3: Round-Robin vs Cache-Aware**
- 5 prompts × 2 rounds = 10 requests
- Round-robin: No cache benefit, evenly distributed
- Cache-aware: 12-18% faster, 13-22% higher throughput
- Shows benefit increases with more repeated prompts

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_cache_tokens` | 10M | Maximum tokens to store in trie |
| `strategy` | CACHE_FIRST | Routing strategy |
| `enable_approximate` | True | Enable fuzzy matching |
| `approximate_threshold` | 0.85 | Minimum similarity for fuzzy matches |
| `health_check_interval` | 30s | Seconds between health checks |
| `worker_timeout` | 60s | Seconds before marking worker unhealthy |

## Project Structure

Simplified layout using modern Python packaging (PEP 420):

```
aegis_router/               # Main package (only one __init__.py)
├── __init__.py            # Public API exports
├── __main__.py            # python -m aegis_router entry point
├── main.py                # CLI entry point
├── core/
│   ├── radix_trie.py      # Radix Trie implementation
│   └── worker.py          # Worker abstraction
├── matching/
│   └── approximate.py     # MinHash/SimHash matching
├── router/
│   └── cache_router.py    # Main orchestrator
└── api/
    └── server.py          # FastAPI endpoints

tests/                     # Test files (no __init__.py needed)
├── test_radix_trie.py
├── test_approximate.py
└── test_cache_router.py

pyproject.toml             # Modern Python project configuration
requirements.txt           # Dependencies
```

**Key simplifications:**
- Only one `__init__.py` at the package root (modern Python uses implicit namespace packages)
- All configuration centralized in `pyproject.toml`
- No need for `setup.py` or `setup.cfg`

## Design Decisions

### Why Radix Trie?

- **O(L) matching**: Where L is prompt length - very fast
- **Memory efficient**: Compressed paths reduce node count
- **Industry proven**: vLLM uses same approach
- **Perfect for prefixes**: LLM prompts share common prefixes (system prompts)

### Why MinHash + SimHash?

- **MinHash**: Good for longer sequences, tunable false positive rate via LSH bands
- **SimHash**: Fast, excellent for near-duplicates (typos, formatting changes)
- **Hybrid approach**: Combines strengths of both

### Thread Safety

- All operations use `threading.RLock()` for safety
- Fine-grained locking at operation level
- Stats use separate lock to reduce contention

## License

MIT License - See LICENSE file for details.
