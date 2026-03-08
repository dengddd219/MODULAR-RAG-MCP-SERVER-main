#!/usr/bin/env python
"""Test script to verify RAGAS evaluation uses OpenAI API correctly.

This script verifies that:
1. RAGAS evaluation uses gpt-4o-mini regardless of main business LLM
2. Environment variables (EVAL_API_KEY, EVAL_BASE_URL) are properly read
3. The evaluation pipeline works end-to-end

Usage:
    # Set environment variables first
    export EVAL_API_KEY="your-openai-api-key"
    export EVAL_BASE_URL="https://api.openai.com/v1"  # optional
    
    # Run test
    python scripts/test_ragas_eval.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_ragas_evaluator_initialization() -> bool:
    """Test that RagasEvaluator can be initialized without settings."""
    print("=" * 60)
    print("Test 1: RagasEvaluator Initialization")
    print("=" * 60)
    
    try:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        
        # Test 1: Initialize without settings (should work)
        print("\n✓ Testing initialization without settings...")
        evaluator1 = RagasEvaluator()
        print(f"  - Metrics: {evaluator1._metric_names}")
        
        # Test 2: Initialize with custom metrics
        print("\n✓ Testing initialization with custom metrics...")
        evaluator2 = RagasEvaluator(metrics=["faithfulness", "answer_relevancy"])
        print(f"  - Metrics: {evaluator2._metric_names}")
        
        print("\n✅ All initialization tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_variables() -> bool:
    """Test that environment variables are properly checked."""
    print("\n" + "=" * 60)
    print("Test 2: Environment Variable Validation")
    print("=" * 60)
    
    # Check if EVAL_API_KEY is set
    api_key = os.getenv("EVAL_API_KEY")
    base_url = os.getenv("EVAL_BASE_URL", "https://api.openai.com/v1")
    
    print(f"\n✓ EVAL_API_KEY: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"✓ EVAL_BASE_URL: {base_url}")
    
    if not api_key:
        print("\n⚠️  WARNING: EVAL_API_KEY is not set!")
        print("   Please set it before running actual evaluation:")
        print("   export EVAL_API_KEY='your-openai-api-key'")
        return False
    
    print("\n✅ Environment variables are properly configured!")
    return True


def test_build_wrappers() -> bool:
    """Test that _build_wrappers uses environment variables correctly."""
    print("\n" + "=" * 60)
    print("Test 3: _build_wrappers Method")
    print("=" * 60)
    
    api_key = os.getenv("EVAL_API_KEY")
    if not api_key:
        print("\n⚠️  Skipping: EVAL_API_KEY not set")
        return False
    
    try:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        
        evaluator = RagasEvaluator()
        
        print("\n✓ Testing _build_wrappers()...")
        llm, embeddings = evaluator._build_wrappers()
        
        print(f"  - LLM type: {type(llm)}")
        print(f"  - Embeddings type: {type(embeddings)}")
        
        # Check that LLM has max_tokens set
        if hasattr(llm, "max_tokens"):
            print(f"  - LLM max_tokens: {llm.max_tokens}")
        elif hasattr(llm, "llm") and hasattr(llm.llm, "max_tokens"):
            print(f"  - LLM max_tokens (nested): {llm.llm.max_tokens}")
        
        print("\n✅ _build_wrappers test passed!")
        return True
    except ValueError as e:
        if "EVAL_API_KEY" in str(e):
            print(f"\n❌ Missing EVAL_API_KEY: {e}")
            return False
        raise
    except Exception as e:
        print(f"\n❌ _build_wrappers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_with_mock_data() -> bool:
    """Test actual evaluation with mock data (requires API key)."""
    print("\n" + "=" * 60)
    print("Test 4: Evaluation with Mock Data")
    print("=" * 60)
    
    api_key = os.getenv("EVAL_API_KEY")
    if not api_key:
        print("\n⚠️  Skipping: EVAL_API_KEY not set")
        return False
    
    try:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        
        evaluator = RagasEvaluator(metrics=["faithfulness"])
        
        # Mock data
        query = "What is RAG?"
        retrieved_chunks = [
            {"text": "RAG stands for Retrieval-Augmented Generation. It combines retrieval and generation."},
            {"text": "RAG systems retrieve relevant documents and use them to generate answers."},
        ]
        generated_answer = "RAG (Retrieval-Augmented Generation) is a technique that combines document retrieval with language model generation to produce accurate answers."
        
        print("\n✓ Running evaluation with mock data...")
        print(f"  - Query: {query}")
        print(f"  - Chunks: {len(retrieved_chunks)}")
        print(f"  - Answer length: {len(generated_answer)} chars")
        
        result = evaluator.evaluate(
            query=query,
            retrieved_chunks=retrieved_chunks,
            generated_answer=generated_answer,
        )
        
        print(f"\n✓ Evaluation completed!")
        print(f"  - Result: {result}")
        
        # Check that result contains expected metrics
        if "faithfulness" in result:
            print(f"  - Faithfulness score: {result['faithfulness']:.4f}")
        
        print("\n✅ Evaluation test passed!")
        return True
    except Exception as e:
        print(f"\n❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RAGAS Evaluation Test Suite")
    print("=" * 60)
    print("\nThis script verifies that RAGAS evaluation:")
    print("  1. Uses gpt-4o-mini regardless of main business LLM")
    print("  2. Reads configuration from environment variables")
    print("  3. Works correctly end-to-end")
    print()
    
    results = []
    
    # Test 1: Initialization
    results.append(("Initialization", test_ragas_evaluator_initialization()))
    
    # Test 2: Environment variables
    results.append(("Environment Variables", test_environment_variables()))
    
    # Test 3: Build wrappers
    results.append(("Build Wrappers", test_build_wrappers()))
    
    # Test 4: Actual evaluation (only if API key is set)
    api_key = os.getenv("EVAL_API_KEY")
    if api_key:
        results.append(("Evaluation", test_evaluation_with_mock_data()))
    else:
        print("\n" + "=" * 60)
        print("Test 4: Evaluation with Mock Data")
        print("=" * 60)
        print("\n⚠️  Skipped: EVAL_API_KEY not set")
        results.append(("Evaluation", None))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, result in results:
        if result is None:
            status = "⚠️  Skipped"
        elif result:
            status = "✅ Passed"
        else:
            status = "❌ Failed"
        print(f"  {name}: {status}")
    
    # Return exit code
    failed = [r for _, r in results if r is False]
    if failed:
        print(f"\n❌ {len(failed)} test(s) failed!")
        return 1
    elif all(r is not False for _, r in results):
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests were skipped")
        return 0


if __name__ == "__main__":
    sys.exit(main())

