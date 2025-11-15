"""
Setup script for Offline Contextual Retrieval System
Run this first to check dependencies and prepare data
"""

import os
import sys
import subprocess
import json

def check_dependencies():
    """Check if required packages are installed"""
    
    required_packages = [
        'numpy',
        'scipy', 
        'sentence_transformers',
        'scikit-learn',
        'faiss',
        'rank_bm25',
        'tqdm'
    ]
    
    missing_packages = []
    
    print("Checking dependencies...")
    print("-" * 30)
    
    for package in required_packages:
        try:
            if package == 'faiss':
                import faiss
            elif package == 'rank_bm25':
                from rank_bm25 import BM25Okapi
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with:")
        
        # Special handling for faiss
        if 'faiss' in missing_packages:
            print("  pip install faiss-cpu  # or faiss-gpu if you have GPU")
            missing_packages.remove('faiss')
        
        if 'rank_bm25' in missing_packages:
            print("  pip install rank-bm25")
            missing_packages.remove('rank_bm25')
        
        if missing_packages:
            print(f"  pip install {' '.join(missing_packages)}")
        
        print("\nOr install all at once:")
        print("  pip install -r requirements_contextual.txt")
        
        return False
    
    print("\nAll dependencies are installed! ✓")
    return True

def check_data_files():
    """Check available chunk data files"""
    
    print("\nChecking data files...")
    print("-" * 30)
    
    base_dir = "RAGdatav4"
    
    if not os.path.exists(base_dir):
        print(f"✗ Directory {base_dir} not found")
        return []
    
    # Look for chunk files
    chunk_files = []
    for file in os.listdir(base_dir):
        if file.endswith('.json') and 'chunks' in file:
            file_path = os.path.join(base_dir, file)
            
            # Check if file is readable
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    chunk_count = len(data)
                
                print(f"✓ {file} ({chunk_count} chunks)")
                chunk_files.append(file_path)
                
            except Exception as e:
                print(f"✗ {file} - Error: {e}")
    
    if not chunk_files:
        print("No valid chunk files found!")
        print("Expected files like:")
        print("  - healthify_chunks_*.json")
        print("  - mayo_chunks_*.json") 
        print("  - nhs_chunks_*.json")
        return []
    
    print(f"\nFound {len(chunk_files)} valid chunk files")
    return chunk_files

def create_sample_config():
    """Create a sample configuration file"""
    
    config = {
        "model_name": "all-MiniLM-L6-v2",
        "rerank_method": "feature",
        "cache_dir": "./cache",
        "search_defaults": {
            "top_k": 10,
            "use_reranking": True,
            "semantic_weight": 0.7,
            "bm25_weight": 0.3
        },
        "chunk_files": [
            "RAGdatav4/healthify_chunks_contextual_fixed_c512_o100.json",
            "RAGdatav4/mayo_chunks_contextual_fixed_c512_o100.json",
            "RAGdatav4/nhs_chunks_contextual_fixed_c512_o100.json"
        ]
    }
    
    config_file = "contextual_retrieval_config.json"
    
    with open(config_file, 'w') as f:
        json.dump(config, indent=2, fp=f)
    
    print(f"\nCreated configuration file: {config_file}")
    print("Edit this file to customize your setup")

def run_quick_test():
    """Run a quick test to verify everything works"""
    
    print("\nRunning quick test...")
    print("-" * 30)
    
    try:
        from offline_contextual_retrieval import OfflineContextualRetrieval
        
        # Create a minimal test
        print("✓ Import successful")
        
        # Test with tiny model for speed
        system = OfflineContextualRetrieval(
            model_name='all-MiniLM-L6-v2',
            rerank_method='feature'
        )
        print("✓ System initialization successful")
        
        print("\nQuick test passed! ✓")
        print("Ready to run full system with your data")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    
    print("Offline Contextual Retrieval System Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("✗ Python 3.7+ required")
        return
    
    print(f"✓ Python {sys.version.split()[0]}")
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check data files
    data_files = check_data_files()
    
    # Create config
    create_sample_config()
    
    # Run test if everything looks good
    if deps_ok and data_files:
        test_ok = run_quick_test()
        
        if test_ok:
            print("\n" + "=" * 50)
            print("SETUP COMPLETE!")
            print("=" * 50)
            print("Next steps:")
            print("1. Run: python test_contextual_retrieval.py")
            print("2. Or use the system in your own code:")
            print("   from offline_contextual_retrieval import OfflineContextualRetrieval")
            print("\nSee the example in offline_contextual_retrieval.py main() function")
        else:
            print("\n✗ Setup completed with errors")
            print("Check the error messages above")
    else:
        print("\n✗ Setup incomplete")
        print("Please fix the issues above and run again")

if __name__ == "__main__":
    main()