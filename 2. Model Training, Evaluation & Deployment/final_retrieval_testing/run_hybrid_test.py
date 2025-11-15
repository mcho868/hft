#!/usr/bin/env python3
"""
Convenience script to run hybrid retrieval tests with predefined configurations.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def print_banner():
    """Print system banner"""
    print("="*80)
    print("🔬 HYBRID RETRIEVAL PERFORMANCE TESTING")
    print("   Multi-Source Bias + Pure RAG vs Contextual RAG")
    print("="*80)

def create_test_data():
    """Create sample test data if it doesn't exist"""
    test_data_file = "eval/test_data.json"
    
    if not os.path.exists(test_data_file):
        print("Creating sample test data...")
        result = subprocess.run([sys.executable, "create_sample_test_data.py"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error creating test data: {result.stderr}")
            return False
        print("Sample test data created successfully")
    else:
        print(f"Using existing test data: {test_data_file}")
    
    return True

def run_quick_hybrid_test():
    """Run a quick test with limited configurations"""
    print("Running quick hybrid test (limited configurations)...")
    
    cmd = [
        sys.executable, "hybrid_retrieval_evaluator.py",
        "--test-data", "eval/test_data.json",
        "--results-dir", "results/hybrid_quick",
        "--max-configs", "10"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_bias_comparison_test():
    """Run test comparing different bias configurations"""
    print("Running bias configuration comparison test...")
    
    cmd = [
        sys.executable, "hybrid_retrieval_evaluator.py",
        "--test-data", "eval/test_data.json",
        "--results-dir", "results/bias_comparison",
        "--chunking-filter", "fixed_c1024_o150"  # Use one chunking method for fair comparison
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_rag_type_comparison():
    """Run test comparing Pure RAG vs Contextual RAG"""
    print("Running Pure RAG vs Contextual RAG comparison...")
    
    cmd = [
        sys.executable, "hybrid_retrieval_evaluator.py",
        "--test-data", "eval/test_data.json",
        "--results-dir", "results/rag_comparison",
        "--bias-filter", "balanced"  # Use balanced bias for fair comparison
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_contextual_focused_test():
    """Run test focused on contextual configurations only"""
    print("Running contextual-focused test...")
    
    cmd = [
        sys.executable, "hybrid_retrieval_evaluator.py",
        "--test-data", "eval/test_data.json",
        "--results-dir", "results/contextual_focused",
        "--chunking-filter", "contextual"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_full_hybrid_test():
    """Run comprehensive test with all configurations"""
    print("Running full hybrid test (all configurations)...")
    print("⚠️  Warning: This may take a long time!")
    
    # Ask about dataset choice
    print("\nDataset options:")
    print("1. Small test dataset (5 cases) - Quick")
    print("2. Validation dataset (1,975 cases) - Comprehensive but slow")
    
    dataset_choice = input("Choose dataset (1 or 2): ").strip()
    
    if dataset_choice == "2":
        test_data_file = "../generated_triage_dialogues_val.json"
        print("🔍 Using validation dataset with 1,975 test cases")
        print("⏰ Estimated time: 2-4 hours depending on hardware")
    else:
        test_data_file = "eval/test_data.json"
        print("🚀 Using small test dataset with 5 test cases")
    
    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Test cancelled.")
        return False
    
    cmd = [
        sys.executable, "hybrid_retrieval_evaluator.py",
        "--test-data", test_data_file,
        "--results-dir", "results/hybrid_full"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_validation_dataset_test():
    """Run test with validation dataset and top performing configurations"""
    print("Running validation dataset test...")
    print("📊 Using validation dataset with 1,975 test cases")
    print("🎯 Testing top 5 chunking methods with memory monitoring")
    print("⏰ Estimated time: 1-2 hours")
    
    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Test cancelled.")
        return False
    
    cmd = [
        sys.executable, "hybrid_retrieval_evaluator.py",
        "--test-data", "../generated_triage_dialogues_val.json",
        "--results-dir", "results/validation_test",
        "--max-configs", "30"  # Top 5 chunking methods × 6 bias configs
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def show_results_summary():
    """Show a summary of available results"""
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found.")
        return
    
    print("\n📊 AVAILABLE RESULTS:")
    print("=" * 40)
    
    subdirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('hybrid')]
    if not subdirs:
        print("No hybrid test results found.")
        return
    
    for subdir in sorted(subdirs, key=lambda x: x.stat().st_mtime, reverse=True):
        # Get latest timestamp
        json_files = list(subdir.glob("hybrid_summary_*.json"))
        if json_files:
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            timestamp = latest_file.stem.split('_')[-1]
            print(f"📁 {subdir.name} (Latest: {timestamp})")
            
            # Show some files
            files = list(subdir.glob("*"))
            print(f"   📄 {len(files)} files")
            
            if subdir.glob("hybrid_detailed_report_*.txt"):
                print("   📋 Detailed report available")
            if subdir.glob("hybrid_summary_*.json"):
                print("   📊 Summary data available")
        else:
            print(f"📁 {subdir.name} (No results yet)")

def main():
    """Main function with user menu"""
    print_banner()
    
    # Check if required files exist
    required_files = [
        "hybrid_retrieval_evaluator.py",
        "offline_contextual_retrieval.py", 
        "../RAGdatav4"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("ERROR: Missing required files/directories:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease ensure all files are in place before running tests.")
        return
    
    # Create test data if needed
    if not create_test_data():
        return
    
    while True:
        print("\n" + "="*60)
        print("📋 SELECT HYBRID TEST TYPE:")
        print("="*60)
        print("1. 🚀 Quick Test (10 configs)")
        print("2. ⚖️  Bias Configuration Comparison")
        print("3. 🧠 Pure RAG vs Contextual RAG Comparison")
        print("4. 🎯 Contextual-Focused Test")
        print("5. 🔍 Full Test (all configs - choose dataset)")
        print("6. 🎯 Validation Dataset Test (1,975 cases)")
        print("7. 📊 Show Results Summary")
        print("8. ❌ Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        success = False
        
        if choice == "1":
            success = run_quick_hybrid_test()
        elif choice == "2":
            success = run_bias_comparison_test()
        elif choice == "3":
            success = run_rag_type_comparison()
        elif choice == "4":
            success = run_contextual_focused_test()
        elif choice == "5":
            success = run_full_hybrid_test()
        elif choice == "6":
            success = run_validation_dataset_test()
        elif choice == "7":
            show_results_summary()
            continue
        elif choice == "8":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-8.")
            continue
        
        if success:
            print("\n" + "="*60)
            print("✅ TEST COMPLETED SUCCESSFULLY")
            print("="*60)
            print("Check the results directory for detailed output.")
            print("Run option 6 to see available results.")
        else:
            print("\n" + "="*60)
            print("❌ TEST FAILED")
            print("="*60)
            print("Check the error messages above for details.")

if __name__ == "__main__":
    main()