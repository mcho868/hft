#!/usr/bin/env python3
"""
Quick Test Runner - Tests just the first few configurations for rapid iteration
"""

import sys
from pathlib import Path

# Add evaluation framework to path
sys.path.insert(0, str(Path(__file__).parent))

from optimized_config_generator import OptimizedConfigMatrixGenerator
from enhanced_evaluation_pipeline import EnhancedEvaluationPipeline

def main():
    print("🚀 Quick Test Runner - Testing First 5 Configurations")
    print("=" * 60)
    
    # Generate configurations
    config_generator = OptimizedConfigMatrixGenerator()
    all_combos = config_generator.generate_evaluation_matrix()
    
    # Take only first 5 for quick testing
    quick_combos = all_combos[:5]
    
    print(f"📊 Testing {len(quick_combos)} configurations (out of {len(all_combos)} total)")
    for i, combo in enumerate(quick_combos):
        chunk_limit = combo.rag_config.get('chunk_limit', 10)
        print(f"   {i+1}. {combo.combo_id} ({chunk_limit} chunks)")
    
    # Initialize pipeline with faster settings
    pipeline = EnhancedEvaluationPipeline(
        base_dir="/Users/choemanseung/789/hft",
        batch_size=1,
        max_workers=1,
        enable_clinical_judge=False,  # Disable for speed
        skip_models=False
    )
    
    # Run evaluation
    print(f"\n🚀 Starting quick evaluation...")
    results = pipeline.run_enhanced_evaluation(quick_combos, resume=False)
    
    # Display results
    print(f"\n📊 Quick Test Results:")
    print("-" * 60)
    
    valid_results = [r for r in results if r.error_message is None]
    if valid_results:
        for result in valid_results:
            chunk_info = "C5" if "_C5_" in result.combo_id else "C10" if "_C10_" in result.combo_id else "C?"
            print(f"✅ {result.combo_id} ({chunk_info}):")
            print(f"   Accuracy: {result.triage_accuracy:.3f}, F2: {result.f2_score:.3f}")
            print(f"   Speed: {result.inference_speed_tps:.1f} tps, Memory: {result.memory_usage_mb:.1f}MB")
        
        # Best performer
        best = max(valid_results, key=lambda x: x.f2_score)
        print(f"\n🏆 Best F2 Score: {best.combo_id} ({best.f2_score:.3f})")
        
        # Speed comparison
        chunk5_results = [r for r in valid_results if "_C5_" in r.combo_id]
        chunk10_results = [r for r in valid_results if "_C10_" in r.combo_id]
        
        if chunk5_results and chunk10_results:
            avg_f2_5 = sum(r.f2_score for r in chunk5_results) / len(chunk5_results)
            avg_f2_10 = sum(r.f2_score for r in chunk10_results) / len(chunk10_results)
            print(f"\n📈 Chunk Comparison:")
            print(f"   5 chunks: {avg_f2_5:.3f} F2 (avg)")
            print(f"   10 chunks: {avg_f2_10:.3f} F2 (avg)")
    else:
        print("❌ No successful evaluations")
    
    print(f"\n✅ Quick test completed! Run full evaluation if results look good.")

if __name__ == "__main__":
    main()