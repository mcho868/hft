#!/usr/bin/env python3
"""
Main entry point for the Retrieval Performance Testing System

This script provides a unified interface to all testing functionalities.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print system banner"""
    print("="*70)
    print("🔬 RETRIEVAL PERFORMANCE TESTING SYSTEM")
    print("   Medical Triage Document Retrieval Evaluation")
    print("="*70)

def setup_environment():
    """Run environment setup"""
    print("🔧 Setting up environment...")
    script_path = Path(__file__).parent / 'setup_test_environment.py'
    result = subprocess.run([sys.executable, str(script_path)])
    return result.returncode == 0

def create_test_data():
    """Create sample test data"""
    print("📝 Creating test data...")
    script_path = Path(__file__).parent / 'create_sample_test_data.py'
    result = subprocess.run([sys.executable, str(script_path)])
    return result.returncode == 0

def run_quick_test():
    """Run quick test with 5 configurations"""
    print("🚀 Running quick test...")
    script_path = Path(__file__).parent / 'retrieval_performance_tester.py'
    result = subprocess.run([
        sys.executable, str(script_path),
        '--test-data', 'eval/test_data.json',
        '--results-dir', 'results/quick_test',
        '--max-configs', '5'
    ])
    return result.returncode == 0

def run_full_test():
    """Run comprehensive test with all configurations"""
    print("🔍 Running comprehensive test...")
    script_path = Path(__file__).parent / 'retrieval_performance_tester.py'
    result = subprocess.run([
        sys.executable, str(script_path),
        '--test-data', 'eval/test_data.json',
        '--results-dir', 'results/full_test'
    ])
    return result.returncode == 0

def run_contextual_test():
    """Run test with only contextual configurations"""
    print("🧠 Running contextual-only test...")
    script_path = Path(__file__).parent / 'retrieval_performance_tester.py'
    result = subprocess.run([
        sys.executable, str(script_path),
        '--test-data', 'eval/test_data.json',
        '--results-dir', 'results/contextual_test',
        '--config-filter', 'contextual'
    ])
    return result.returncode == 0

def visualize_results(results_dir='results'):
    """Generate visualizations"""
    print("📊 Generating visualizations...")
    
    # Find the most recent results directory
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return False
    
    subdirs = [d for d in results_path.iterdir() if d.is_dir()]
    if subdirs:
        latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
        results_dir = str(latest_dir)
    
    script_path = Path(__file__).parent / 'visualize_results.py'
    result = subprocess.run([
        sys.executable, str(script_path),
        '--results-dir', results_dir,
        '--output-dir', 'visualizations'
    ])
    return result.returncode == 0

def interactive_menu():
    """Interactive menu for user selection"""
    while True:
        print("\n" + "="*50)
        print("📋 SELECT OPERATION:")
        print("="*50)
        print("1. 🔧 Setup Environment")
        print("2. 📝 Create Test Data")
        print("3. 🚀 Quick Test (5 configs)")
        print("4. 🔍 Full Test (all configs)")
        print("5. 🧠 Contextual Only Test")
        print("6. 📊 Visualize Results")
        print("7. 🎯 Complete Pipeline")
        print("8. ❌ Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            setup_environment()
        elif choice == '2':
            create_test_data()
        elif choice == '3':
            if run_quick_test():
                visualize_results('results/quick_test')
        elif choice == '4':
            if run_full_test():
                visualize_results('results/full_test')
        elif choice == '5':
            if run_contextual_test():
                visualize_results('results/contextual_test')
        elif choice == '6':
            results_dir = input("Enter results directory (default: results): ").strip()
            if not results_dir:
                results_dir = 'results'
            visualize_results(results_dir)
        elif choice == '7':
            print("🔄 Running complete pipeline...")
            if (setup_environment() and 
                create_test_data() and 
                run_quick_test()):
                visualize_results('results/quick_test')
                print("✅ Complete pipeline finished!")
            else:
                print("❌ Pipeline failed at some step")
        elif choice == '8':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-8.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Retrieval Performance Testing System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # Interactive menu
  python main.py --setup             # Setup environment only
  python main.py --quick             # Quick test
  python main.py --full              # Full test
  python main.py --pipeline          # Complete pipeline
        """
    )
    
    parser.add_argument('--setup', action='store_true', help='Setup environment')
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    parser.add_argument('--full', action='store_true', help='Run full test')
    parser.add_argument('--contextual', action='store_true', help='Run contextual test')
    parser.add_argument('--visualize', help='Visualize results from directory')
    parser.add_argument('--pipeline', action='store_true', help='Run complete pipeline')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Command line mode
    if args.setup:
        return 0 if setup_environment() else 1
    elif args.quick:
        success = create_test_data() and run_quick_test()
        if success:
            visualize_results('results/quick_test')
        return 0 if success else 1
    elif args.full:
        success = create_test_data() and run_full_test()
        if success:
            visualize_results('results/full_test')
        return 0 if success else 1
    elif args.contextual:
        success = create_test_data() and run_contextual_test()
        if success:
            visualize_results('results/contextual_test')
        return 0 if success else 1
    elif args.visualize:
        return 0 if visualize_results(args.visualize) else 1
    elif args.pipeline:
        success = (setup_environment() and 
                  create_test_data() and 
                  run_quick_test())
        if success:
            visualize_results('results/quick_test')
        return 0 if success else 1
    else:
        # Interactive mode
        interactive_menu()
        return 0

if __name__ == "__main__":
    exit(main())