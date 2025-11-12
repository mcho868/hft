#!/usr/bin/env python3
"""
Streamlined Medical Triage Evaluation Framework
Entry point script for the enhanced evaluation pipeline with TinfoilAgent.
"""

import sys
import argparse

def main():
    """Main entry point for streamlined evaluation"""
    parser = argparse.ArgumentParser(
        description="Medical Triage System - Enhanced Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete evaluation with clinical assessment
  python run_evaluation.py
  
  # Quick test with first 50 combinations  
  python run_evaluation.py --max-combinations 50
  
  # Skip clinical evaluation (default - faster)
  python run_evaluation.py
  
  # Enable clinical evaluation with TinfoilAgent
  python run_evaluation.py --enable-clinical
  
  # Test TinfoilAgent integration
  python test_tinfoil_integration.py
        """
    )
    
    parser.add_argument("--max-combinations", type=int, default=600,
                       help="Maximum combinations to evaluate (default: 600)")
    parser.add_argument("--enable-clinical", action="store_true",
                       help="Enable TinfoilAgent clinical evaluation (default: False)")
    parser.add_argument("--skip-models", action="store_true",
                       help="Skip model loading for faster testing")
    parser.add_argument("--base-dir", default="/Users/choemanseung/789/hft",
                       help="Base directory path")
    
    args = parser.parse_args()
    
    print("🚀 Medical Triage System - Enhanced Evaluation Pipeline")
    print("=" * 60)
    print(f"📊 Max combinations: {args.max_combinations}")
    print(f"🤖 Clinical evaluation: {'Enabled (TinfoilAgent)' if args.enable_clinical else 'Disabled (default)'}")
    print(f"🔧 Model loading: {'Disabled (Mock)' if args.skip_models else 'Enabled (MLX)'}")
    print("=" * 60)
    
    # Import and run optimized master runner
    from optimized_master_runner import OptimizedMasterEvaluationRunner
    
    # Configure based on arguments
    config = {
        "phases": {
            "generate_matrix": True,
            "run_evaluation": True, 
            "clinical_judgement": args.enable_clinical,
            "performance_profiling": True,
            "statistical_analysis": True
        },
        "evaluation": {
            "max_combinations": args.max_combinations,
            "batch_size": 8,
            "max_workers": 2,
            "resume": True,
            "skip_models": args.skip_models
        },
        "output": {"verbose": True}
    }
    
    try:
        # Run enhanced evaluation
        runner = OptimizedMasterEvaluationRunner(args.base_dir, config)
        results = runner.run_complete_optimized_evaluation()
        
        print("\n🎉 Enhanced evaluation completed successfully!")
        print(f"📁 Results saved to: {results['session_dir']}")
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())