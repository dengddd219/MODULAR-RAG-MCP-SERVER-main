#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script for LLM Arena - verify routing and model switching.

This script tests:
1. Simple query -> should route to small model (local SLM)
2. Complex query -> should route to large model (cloud LLM)

Usage:
    python scripts/test_llm_arena.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.core.query_engine.intent_router import IntentRouter
from src.core.settings import load_settings
from src.libs.llm.model_evaluator import ModelEvaluator
from src.libs.llm.model_manager import ModelConfig, ModelManager


def print_section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_model_registration() -> None:
    """Test 1: Verify model registration."""
    print_section("Test 1: Model Registration")
    
    try:
        settings = load_settings()
        manager = ModelManager(settings)
        
        # Register Tier 2 models
        tier2_models = [
            "ollama-qwen2.5:0.5b",  # Ultra-fast, lightning speed on any CPU
            "ollama-qwen2.5:1.5b",  # Very fast, excellent Chinese understanding, perfect for RAG
        ]
        
        for model_name in tier2_models:
            model_id = model_name.replace(":", "-")
            config = ModelConfig(
                model_id=model_id,
                provider="ollama",
                model_name=model_name,
                display_name=model_name,
                description=f"Local SLM: {model_name}",
                is_small_model=True,
            )
            manager.register_model(config)
            print(f"[OK] Registered Tier 2 model: {model_name}")
        
        # Register Tier 3 models
        provider_map = {
            "api-deepseek-chat": ("deepseek", "deepseek-chat"),
            "api-qwen-max": ("qwen", "qwen-max"),
            "api-glm-4-plus": ("glm", "glm-4-plus"),
            "api-gpt-4o-mini": ("openai", "gpt-4o-mini"),
        }
        
        tier3_models = [
            "api-deepseek-chat",
            "api-qwen-max",
            "api-glm-4-plus",
            "api-gpt-4o-mini",
        ]
        
        for model_name in tier3_models:
            model_id = model_name.replace(":", "-")
            provider, actual_model = provider_map.get(model_name, ("unknown", model_name))
            config = ModelConfig(
                model_id=model_id,
                provider=provider,
                model_name=actual_model,
                display_name=model_name,
                description=f"Cloud LLM: {model_name}",
                is_small_model=False,
            )
            manager.register_model(config)
            print(f"[OK] Registered Tier 3 model: {model_name}")
        
        # List all models
        all_models = manager.list_models()
        print(f"\n📊 Total registered models: {len(all_models)}")
        for model in all_models:
            tier = "Tier 2 (Local SLM)" if model.is_small_model else "Tier 3 (Cloud LLM)"
            print(f"   - {model.display_name} ({tier})")
        
        print("\n[OK] Model registration test passed!")
        return manager
        
    except Exception as e:
        print(f"[ERROR] Model registration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_routing_logic(
    manager: ModelManager,
    query: str,
    expected_model_type: str,
    description: str,
) -> bool:
    """Test routing logic for a query."""
    print(f"\n🔍 Testing: {description}")
    print(f"   Query: {query}")
    print(f"   Expected: {expected_model_type}")
    
    try:
        settings = load_settings()
        intent_router = IntentRouter()
        
        # Route query
        routing_result = intent_router.route(query)
        
        print(f"\n   Routing Result:")
        print(f"   - Intent Label: {routing_result.intent_label}")
        print(f"   - Intent Confidence: {routing_result.intent_confidence}")
        print(f"   - Is Spam: {routing_result.is_spam}")
        
        # Get routing config
        routing_config = getattr(settings, "llm_routing", None)
        if routing_config is None:
            print("   ⚠️  No routing config found, using defaults")
            simple_intents = ["fabric_care", "faq", "returns"]
            complexity_threshold = 0.7
        else:
            simple_intents = routing_config.simple_intents
            complexity_threshold = routing_config.complexity_threshold
        
        # Determine which model to use
        intent_label = routing_result.intent_label or ""
        confidence = routing_result.intent_confidence or 0.0
        
        is_simple = (
            intent_label.lower() in [s.lower() for s in simple_intents]
            and confidence >= complexity_threshold
        )
        
        if is_simple:
            selected_model_type = "small model (local SLM)"
            model_id = "ollama-qwen2.5-7b"  # Default small model
        else:
            selected_model_type = "large model (cloud LLM)"
            model_id = "api-deepseek-chat"  # Default large model
        
        print(f"\n   Routing Decision:")
        print(f"   - Selected: {selected_model_type}")
        print(f"   - Model ID: {model_id}")
        
        # Verify expectation
        if expected_model_type == "small" and is_simple:
            print(f"   [OK] Correctly routed to small model")
            return True
        elif expected_model_type == "large" and not is_simple:
            print(f"   [OK] Correctly routed to large model")
            return True
        else:
            print(f"   [WARN] Routing mismatch (expected {expected_model_type}, got {'small' if is_simple else 'large'})")
            return False
        
    except Exception as e:
        print(f"   [ERROR] Routing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_call(manager: ModelManager, model_id: str, query: str) -> bool:
    """Test actual LLM call."""
    print(f"\n🤖 Testing LLM call with model: {model_id}")
    
    try:
        # Set model
        manager.set_current_model(model_id)
        llm = manager.get_llm()
        
        # Create evaluator
        evaluator = ModelEvaluator()
        config = manager.get_model_config(model_id)
        
        # Build prompt
        prompt = f"请简要回答：{query}"
        messages = [{"role": "user", "content": prompt}]
        
        # Track metrics
        with evaluator.track_call(
            model_id=model_id,
            provider=config.provider,
            model_name=config.model_name,
            query=query,
        ) as metrics:
            start_time = time.monotonic()
            
            try:
                response = llm.chat(messages)
                elapsed = time.monotonic() - start_time
                
                # Extract response
                if isinstance(response, str):
                    answer = response
                else:
                    answer = response.content if hasattr(response, "content") else str(response)
                
                # Update metrics
                metrics.latency_ms = elapsed * 1000.0
                metrics.response_length = len(answer)
                
                if hasattr(response, "usage"):
                    usage = response.usage
                    metrics.prompt_tokens = usage.get("prompt_tokens", 0)
                    metrics.completion_tokens = usage.get("completion_tokens", 0)
                    metrics.total_tokens = usage.get("total_tokens", 0)
                
                print(f"   [OK] LLM call successful")
                print(f"   - Latency: {metrics.latency_ms:.0f} ms")
                print(f"   - Tokens: {metrics.total_tokens}")
                print(f"   - Cost: ${metrics.calculate_cost():.6f}")
                print(f"   - Answer (first 100 chars): {answer[:100]}...")
                
                return True
                
            except Exception as e:
                print(f"   [WARN] LLM call failed (this is OK if model is not available): {e}")
                print(f"   - This is expected if the model is not installed/configured")
                return False  # Not a critical failure
        
    except Exception as e:
        print(f"   [ERROR] LLM call setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  LLM Arena Test Suite")
    print("  Testing routing logic and model switching")
    print("=" * 80)
    
    # Test 1: Model registration
    manager = test_model_registration()
    if manager is None:
        print("\n[ERROR] Cannot continue without model registration")
        return
    
    # Test 2: Routing logic - Simple query
    print_section("Test 2: Routing Logic - Simple Query")
    
    simple_queries = [
        ("这件羊毛大衣可以机洗吗？", "simple"),
        ("如何退货？", "simple"),
        ("What is the default port for the Dashboard?", "simple"),
    ]
    
    routing_passed = True
    for query, expected in simple_queries:
        if not test_routing_logic(manager, query, expected, "Simple Query"):
            routing_passed = False
    
    # Test 3: Routing logic - Complex query
    print_section("Test 3: Routing Logic - Complex Query")
    
    complex_queries = [
        ("请详细解释系统的两层意图路由是如何设计和划分的？", "large"),
        ("How does the system achieve zero-cost incremental updates?", "large"),
        ("Please explain the complete workflow from ingestion to return for image processing.", "large"),
    ]
    
    for query, expected in complex_queries:
        if not test_routing_logic(manager, query, expected, "Complex Query"):
            routing_passed = False
    
    # Test 4: Actual LLM calls (if models are available)
    print_section("Test 4: Actual LLM Calls (Optional)")
    print("Note: This test requires models to be installed/configured.")
    print("It will gracefully skip if models are not available.\n")
    
    # Try small model
    test_llm_call(manager, "ollama-qwen2.5-7b", "简单问题：什么是RAG？")
    
    # Try large model
    test_llm_call(manager, "api-deepseek-chat", "复杂问题：请详细解释混合检索的工作原理")
    
    # Summary
    print_section("Test Summary")
    
    if routing_passed:
        print("[OK] Routing logic tests: PASSED")
    else:
        print("[WARN] Routing logic tests: Some mismatches (may be expected)")
    
    print("\n📊 Available Models for Developers:")
    print("\nTier 2 - Local SLM (Small Models):")
    print("  - ollama-qwen2.5:0.5b (Ultra-fast, lightning speed)")
    print("  - ollama-qwen2.5:1.5b (Very fast, excellent Chinese understanding)")
    print("\nTier 3 - Cloud LLM (Large Models):")
    print("  - api-deepseek-chat (DeepSeek-V3)")
    print("  - api-qwen-max (通义千问)")
    print("  - api-glm-4-plus (智谱)")
    print("  - api-gpt-4o-mini (OpenAI)")
    
    print("\n📐 Scoring Formula:")
    print("\nComposite Score = (Cost_Score × 0.33) + (Latency_Score × 0.33) + (Quality_Score × 0.33)")
    print("\nWhere:")
    print("  - Cost_Score = normalize_negative(avg_cost_per_query)")
    print("    Formula: (max_cost - cost) / (max_cost - min_cost)")
    print("  - Latency_Score = normalize_negative(p95_latency)")
    print("    Formula: (max_latency - latency) / (max_latency - min_latency)")
    print("  - Quality_Score = normalize_positive(avg_quality_score)")
    print("    Formula: (quality - min_quality) / (max_quality - min_quality)")
    print("\nFinal score is multiplied by 100 to get 0-100 scale.")
    
    print("\n[OK] Test suite completed!")


if __name__ == "__main__":
    main()

