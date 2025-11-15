#!/usr/bin/env python3
"""
Convenience script to run retrieval performance tests with common configurations.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

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

def run_quick_test():
    """Run a quick test with limited configurations"""
    print("Running quick test (limited configurations)...")
    
    cmd = [
        sys.executable, "retrieval_performance_tester.py",
        "--test-data", "eval/test_data.json",
        "--results-dir", "results/quick_test",
        "--max-configs", "5"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_full_test():
    """Run full test with all configurations"""
    print("Running full test (all configurations)...")
    
    cmd = [
        sys.executable, "retrieval_performance_tester.py", 
        "--test-data", "eval/test_data.json",
        "--results-dir", "results/full_test"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_contextual_only():
    """Run test with only contextual configurations"""
    print("Running contextual-only test...")
    
    cmd = [
        sys.executable, "retrieval_performance_tester.py",
        "--test-data", "eval/test_data.json", 
        "--results-dir", "results/contextual_test",
        "--config-filter", "contextual"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    """Main function with user menu"""
    print("="*60)
    print("RETRIEVAL PERFORMANCE TESTING")
    print("="*60)
    
    # Check if required files exist
    required_files = [
        "retrieval_performance_tester.py",
        "offline_contextual_retrieval.py",
        "RAGdatav4/embeddings"
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
    
    print("\nSelect test type:")
    print("1. Quick test (5 configurations)")
    print("2. Full test (all configurations)")
    print("3. Contextual only test")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            success = run_quick_test()
            break
        elif choice == "2":
            success = run_full_test()
            break
        elif choice == "3":
            success = run_contextual_only()
            break
        elif choice == "4":
            print("Exiting...")
            return
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
    
    if success:
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Check the results directory for detailed output.")
    else:
        print("\n" + "="*60)
        print("TEST FAILED")
        print("="*60)
        print("Check the error messages above for details.")

if __name__ == "__main__":
    main()