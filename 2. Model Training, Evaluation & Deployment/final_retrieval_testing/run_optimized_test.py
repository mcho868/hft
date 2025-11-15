#!/usr/bin/env python3
"""
Test runner for optimized hybrid retrieval evaluator
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def print_banner():
    print("="*80)
    print("🚀 OPTIMIZED HYBRID RETRIEVAL PERFORMANCE TESTING")
    print("   Key Optimizations:")
    print("   ✅ Persistent embedding model in memory")
    print("   ✅ Cached indices and chunks")
    print("   ✅ Batch processing for query embeddings")
    print("   ✅ Memory-efficient evaluation")
    print("="*80)

def run_quick_optimized_test():
    """Run a quick test with the optimized version"""
    print("🚀 Running quick optimized test (2 configs, batch processing)...")
    
    cmd = [
        sys.executable, "optimized_hybrid_evaluator.py",
        "--test-data", "../generated_triage_dialogues_val.json",
        "--results-dir", "results/optimized_quick",
        "--max-configs", "2",
        "--batch-size", "64"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ Quick test completed in {end_time - start_time:.1f} seconds")
        return True
    else:
        print("❌ Quick test failed")
        return False

def run_validation_optimized_test():
    """Run validation dataset test with optimized version"""
    print("🎯 Running validation dataset test (optimized)...")
    print("📊 Using validation dataset with 1,975 test cases")
    print("⚡ Optimized with batch processing and persistent models")
    print("⏰ Estimated time: 15-30 minutes (vs 2-3 hours)")
    
    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Test cancelled.")
        return False
    
    cmd = [
        sys.executable, "optimized_hybrid_evaluator.py",
        "--test-data", "../generated_triage_dialogues_val.json",
        "--results-dir", "results/optimized_validation",
        "--max-configs", "30",  # Top configurations
        "--batch-size", "64"    # Larger batch size for efficiency
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ Validation test completed in {end_time - start_time:.1f} seconds")
        return True
    else:
        print("❌ Validation test failed")
        return False

def compare_performance():
    """Run both original and optimized versions for comparison"""
    print("⚖️  Running performance comparison...")
    print("🐌 Original version (first 100 test cases)")
    print("🚀 Optimized version (first 100 test cases)")
    
    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Comparison cancelled.")
        return False
    
    # Create small test dataset
    print("Creating small test dataset (100 cases)...")
    small_dataset_cmd = [
        "head", "-102", "../generated_triage_dialogues_val.json"  # 100 cases + array brackets
    ]
    
    with open("small_test_data.json", "w") as f:
        result = subprocess.run(small_dataset_cmd, stdout=f)
        if result.returncode != 0:
            print("Failed to create small test dataset")
            return False
    
    # Fix JSON format
    with open("small_test_data.json", "r") as f:
        content = f.read()
    
    # Ensure proper JSON closing
    if not content.strip().endswith("]"):
        content = content.rstrip().rstrip(",") + "\n]"
        with open("small_test_data.json", "w") as f:
            f.write(content)
    
    print("📊 Running original version...")
    original_cmd = [
        sys.executable, "hybrid_retrieval_evaluator.py",
        "--test-data", "small_test_data.json",
        "--results-dir", "results/comparison_original",
        "--max-configs", "2"
    ]
    
    start_time = time.time()
    original_result = subprocess.run(original_cmd)
    original_time = time.time() - start_time
    
    print(f"Original version: {original_time:.1f} seconds")
    
    print("🚀 Running optimized version...")
    optimized_cmd = [
        sys.executable, "optimized_hybrid_evaluator.py",
        "--test-data", "small_test_data.json",
        "--results-dir", "results/comparison_optimized",
        "--max-configs", "2",
        "--batch-size", "32"
    ]
    
    start_time = time.time()
    optimized_result = subprocess.run(optimized_cmd)
    optimized_time = time.time() - start_time
    
    print(f"Optimized version: {optimized_time:.1f} seconds")
    
    if original_time > 0 and optimized_time > 0:
        speedup = original_time / optimized_time
        print(f"🎯 Speedup: {speedup:.1f}x faster!")
    
    # Cleanup
    os.remove("small_test_data.json")
    
    return True

def main():
    print_banner()
    
    # Check if required files exist
    required_files = [
        "optimized_hybrid_evaluator.py",
        "../generated_triage_dialogues_val.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("ERROR: Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return
    
    while True:
        print("\n" + "="*60)
        print("📋 SELECT OPTIMIZED TEST TYPE:")
        print("="*60)
        print("1. 🚀 Quick Optimized Test (2 configs)")
        print("2. 🎯 Validation Dataset Test (1,975 cases, optimized)")
        print("3. ⚖️  Performance Comparison (Original vs Optimized)")
        print("4. ❌ Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        success = False
        
        if choice == "1":
            success = run_quick_optimized_test()
        elif choice == "2":
            success = run_validation_optimized_test()
        elif choice == "3":
            success = compare_performance()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")
            continue
        
        if success:
            print("\n" + "="*60)
            print("✅ TEST COMPLETED SUCCESSFULLY")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ TEST FAILED")
            print("="*60)

if __name__ == "__main__":
    main()