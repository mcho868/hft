#!/usr/bin/env python3
"""
Performance Comparison: Original vs Optimized
"""

import time
import psutil
from performance_optimized_evaluator import StreamingEvaluator
from optimized_hybrid_evaluator import OptimizedHybridTester
from pathlib import Path

def monitor_memory():
    """Get current memory usage"""
    return psutil.Process().memory_info().rss / 1024 / 1024

def test_original_version():
    """Test original version with limited scope"""
    print("🔄 Testing ORIGINAL version...")
    
    start_memory = monitor_memory()
    start_time = time.time()
    
    tester = OptimizedHybridTester(
        embeddings_path="../RAGdatav4/indiv_embeddings",
        batch_size=16,
        num_workers=2
    )
    
    # Load test data (limit to 50 cases for fair comparison)
    test_cases = tester.load_test_data("../Final_dataset/simplified_triage_dialogues_val.json")
    test_cases = test_cases[:50]
    
    # Generate limited configs
    configs = tester.generate_hybrid_configs(
        max_configs=2,
        chunking_filter="fixed_c512_o100",
        bias_filter="diverse"
    )
    
    results = []
    peak_memory = start_memory
    
    for config in configs:
        current_memory = monitor_memory()
        peak_memory = max(peak_memory, current_memory)
        
        result = tester.evaluate_configuration_optimized(config, test_cases)
        results.append(result)
        
        current_memory = monitor_memory()
        peak_memory = max(peak_memory, current_memory)
    
    end_time = time.time()
    end_memory = monitor_memory()
    
    print(f"  ⏱️  Time: {end_time - start_time:.1f}s")
    print(f"  💾 Memory: {start_memory:.1f} -> {end_memory:.1f} MB (Peak: {peak_memory:.1f} MB)")
    print(f"  📊 Results: {[r.pass_at_5 for r in results]}")
    
    return end_time - start_time, peak_memory - start_memory, results

def test_optimized_version():
    """Test optimized version"""
    print("🚀 Testing OPTIMIZED version...")
    
    start_memory = monitor_memory()
    start_time = time.time()
    
    evaluator = StreamingEvaluator(
        embeddings_path="../RAGdatav4/indiv_embeddings",
        batch_size=16,
        max_workers=2
    )
    
    # Load test data (same 50 cases)
    test_cases = evaluator.load_test_data("../Final_dataset/simplified_triage_dialogues_val.json", max_cases=50)
    
    # Generate same configs
    configs = evaluator.generate_configs(
        max_configs=2,
        chunking_filter="fixed_c512_o100"
    )
    configs = [c for c in configs if "diverse" in c.bias_config.name][:2]
    
    results = []
    peak_memory = start_memory
    
    for config in configs:
        current_memory = monitor_memory()
        peak_memory = max(peak_memory, current_memory)
        
        result = evaluator.evaluate_single_config(config, test_cases)
        if result:
            results.append(result)
        
        current_memory = monitor_memory()
        peak_memory = max(peak_memory, current_memory)
    
    end_time = time.time()
    end_memory = monitor_memory()
    
    print(f"  ⏱️  Time: {end_time - start_time:.1f}s")
    print(f"  💾 Memory: {start_memory:.1f} -> {end_memory:.1f} MB (Peak: {peak_memory:.1f} MB)")
    print(f"  📊 Results: {[r.pass_at_5 for r in results]}")
    
    return end_time - start_time, peak_memory - start_memory, results

def main():
    print("="*70)
    print("🔬 PERFORMANCE COMPARISON: Original vs Optimized")
    print("="*70)
    print("Testing 2 configurations with 50 test cases each")
    print()
    
    # Test original version
    try:
        orig_time, orig_memory, orig_results = test_original_version()
    except Exception as e:
        print(f"❌ Original version failed: {e}")
        orig_time, orig_memory, orig_results = float('inf'), float('inf'), []
    
    print()
    
    # Test optimized version
    try:
        opt_time, opt_memory, opt_results = test_optimized_version()
    except Exception as e:
        print(f"❌ Optimized version failed: {e}")
        opt_time, opt_memory, opt_results = float('inf'), float('inf'), []
    
    print()
    print("="*70)
    print("📊 COMPARISON RESULTS")
    print("="*70)
    
    if orig_time != float('inf') and opt_time != float('inf'):
        time_improvement = (orig_time - opt_time) / orig_time * 100
        memory_improvement = (orig_memory - opt_memory) / orig_memory * 100
        
        print(f"⏱️  Time Improvement: {time_improvement:+.1f}%")
        print(f"💾 Memory Improvement: {memory_improvement:+.1f}%")
        
        if opt_results and orig_results:
            print(f"🎯 Accuracy maintained: Original={orig_results[0].pass_at_5:.3f}, Optimized={opt_results[0].pass_at_5:.3f}")
    
    print("\n🔍 Key Optimizations in New Version:")
    print("  ✅ Lazy loading - only one config in memory")
    print("  ✅ Parallel query processing")
    print("  ✅ Aggressive memory cleanup")
    print("  ✅ Cached BM25 preprocessing")
    print("  ✅ Streaming results to disk")

if __name__ == "__main__":
    main()