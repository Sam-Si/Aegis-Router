#!/bin/bash
#
# Run all tests including RadixAttention+Batching integration tests
#
# Usage:
#   ./run_all_tests.sh              # Run all tests
#   ./run_all_tests.sh --quick      # Skip slow integration tests
#   ./run_all_tests.sh --no-cov     # Run without coverage

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  AEGIS ROUTER - COMPLETE TEST SUITE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if model exists
if [ -f "/models/tinyllama.gguf" ]; then
    echo "✓ Model found: /models/tinyllama.gguf"
    MODEL_STATUS="available"
else
    echo "⚠ Model not found at /models/tinyllama.gguf"
    echo "  Integration tests will be skipped"
    MODEL_STATUS="missing"
fi

echo ""
echo "───────────────────────────────────────────────────────────────"
echo "  1. UNIT TESTS (Radix Trie, Approximate Matching, Router)"
echo "───────────────────────────────────────────────────────────────"

# Run unit tests first
if [[ "$*" == *"--no-cov"* ]]; then
    pytest tests/test_radix_trie.py tests/test_approximate.py tests/test_cache_router.py -v --no-cov
else
    pytest tests/test_radix_trie.py tests/test_approximate.py tests/test_cache_router.py -v
fi

echo ""
echo "───────────────────────────────────────────────────────────────"
echo "  2. FUNCTIONAL TESTS (Cache-Aware Routing Demo)"
echo "───────────────────────────────────────────────────────────────"

if [[ "$*" == *"--no-cov"* ]]; then
    pytest tests/test_functional.py -v --no-cov -s
else
    pytest tests/test_functional.py -v -s
fi

# Check if we should run integration tests
if [[ "$*" == *"--quick"* ]]; then
    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo "  3. INTEGRATION TESTS (RadixAttention+Batching)"
echo "───────────────────────────────────────────────────────────────"
    echo "  ⚠ Skipped (--quick flag used)"
else
    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo "  3. INTEGRATION TESTS (RadixAttention+Batching with Real Model)"
    echo "───────────────────────────────────────────────────────────────"
    
    if [ "$MODEL_STATUS" = "available" ]; then
        echo "  Running with real TinyLlama model..."
        echo "  This may take 2-5 minutes..."
        echo ""
        
        if [[ "$*" == *"--no-cov"* ]]; then
            pytest tests/test_radix_attention_integration.py -v --no-cov -s --timeout=300
        else
            pytest tests/test_radix_attention_integration.py -v -s --timeout=300
        fi
    else
        echo "  ⚠ Skipped (model not available)"
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ALL TESTS COMPLETED SUCCESSFULLY ✓"
echo "═══════════════════════════════════════════════════════════════"

if [ "$MODEL_STATUS" = "available" ] && [[ "$*" != *"--quick"* ]]; then
    echo ""
    echo "Summary:"
    echo "  - Unit tests: ✓ Passed"
    echo "  - Functional tests: ✓ Passed"
    echo "  - Integration tests (RadixAttention+Batching): ✓ Passed"
    echo ""
    echo "The RadixAttention+Batching engine has been validated with:"
    echo "  ✓ Real TinyLlama model inference"
    echo "  ✓ Prefix caching (RadixAttention)"
    echo "  ✓ Continuous batching"
    echo "  ✓ Combined performance benefits"
fi

echo ""
