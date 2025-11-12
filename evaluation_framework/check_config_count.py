#!/usr/bin/env python3
"""
Check Configuration Count - Shows how many configurations will be evaluated
"""

import sys
from pathlib import Path

# Add evaluation framework to path
sys.path.insert(0, str(Path(__file__).parent))

from optimized_config_generator import OptimizedConfigMatrixGenerator

def main():
    print("📊 Configuration Count Analysis")
    print("=" * 50)
    
    # Initialize generator
    config_generator = OptimizedConfigMatrixGenerator()
    
    # Check RAG configs
    print(f"📋 RAG Configuration Sources:")
    print(f"   Top pass@5 file: {config_generator.pass5_results_file.exists()}")
    print(f"   Top pass@10 file: {config_generator.pass10_results_file.exists()}")
    
    # Load configs
    rag_configs = config_generator._load_validated_rag_configs()
    print(f"\n🎯 RAG Configurations: {len(rag_configs)}")
    for i, config in enumerate(rag_configs, 1):
        print(f"   {i}. {config.config_id}")
        print(f"      Method: {config.chunking_method} / {config.retrieval_type}")
        print(f"      Bias: {config.bias_config}")
        print(f"      Performance: P@5={config.pass_at_5:.3f}, P@10={config.pass_at_10:.3f}")
    
    # Check adapters
    adapter_configs = config_generator.scan_adapter_directories()
    print(f"\n🤖 Adapter Configurations: {len(adapter_configs)}")
    
    model_counts = {}
    for adapter in adapter_configs:
        model = adapter.model_name
        if model not in model_counts:
            model_counts[model] = 0
        model_counts[model] += 1
    
    for model, count in model_counts.items():
        print(f"   {model}: {count} adapters")
    
    # Calculate total configurations
    chunk_variants = 2  # 5 and 10 chunks
    total_configs = len(rag_configs) * len(adapter_configs) * chunk_variants
    
    print(f"\n🚀 Total Evaluation Matrix:")
    print(f"   RAG configs: {len(rag_configs)}")
    print(f"   × Adapters: {len(adapter_configs)}")
    print(f"   × Chunk variants: {chunk_variants} (5 and 10 chunks)")
    print(f"   = Total configurations: {total_configs}")
    
    # Estimate time
    seconds_per_config = 30  # Rough estimate with 20 test cases
    total_minutes = (total_configs * seconds_per_config) / 60
    total_hours = total_minutes / 60
    
    print(f"\n⏱️  Time Estimates (with 20 test cases):")
    print(f"   ~{seconds_per_config}s per config")
    print(f"   Total time: ~{total_minutes:.1f} minutes ({total_hours:.1f} hours)")
    
    if total_hours > 2:
        print(f"\n⚠️  Consider further optimizations:")
        print(f"   - Set EVAL_SUBSET_SIZE=10 for even faster testing")
        print(f"   - Use --skip-clinical to disable LLM-as-judge")
        print(f"   - Test subset first with quick_test_runner.py")

if __name__ == "__main__":
    main()