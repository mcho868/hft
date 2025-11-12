#!/usr/bin/env python3
"""
Setup script for the retrieval performance testing environment.
Checks dependencies and creates necessary directories.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is adequate"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'faiss',
        'sentence_transformers',
        'rank_bm25',
        'scikit-learn',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'faiss':
                importlib.import_module('faiss')
            else:
                importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing packages"""
    if not packages:
        return True
    
    print(f"\n📦 Installing missing packages: {', '.join(packages)}")
    
    # Map package names to pip install names
    pip_names = {
        'faiss': 'faiss-cpu',
        'rank_bm25': 'rank-bm25',
        'scikit-learn': 'scikit-learn'
    }
    
    for package in packages:
        pip_name = pip_names.get(package, package)
        print(f"Installing {pip_name}...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
            print(f"✅ Successfully installed {pip_name}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {pip_name}")
            return False
    
    return True

def check_required_files():
    """Check if required files exist"""
    required_files = [
        'offline_contextual_retrieval.py',
        'RAGdatav4',
        'RAGdatav4/embeddings'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    return missing_files

def create_directories():
    """Create necessary directories"""
    directories = [
        'eval',
        'results',
        'visualizations'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_embedding_files():
    """Check available embedding files"""
    embeddings_dir = Path('RAGdatav4/embeddings')
    
    if not embeddings_dir.exists():
        print("❌ RAGdatav4/embeddings directory not found")
        return False
    
    pkl_files = list(embeddings_dir.glob('*.pkl'))
    index_files = list(embeddings_dir.glob('*.index'))
    
    print(f"📊 Found {len(pkl_files)} embedding files (.pkl)")
    print(f"📊 Found {len(index_files)} index files (.index)")
    
    if len(pkl_files) == 0 or len(index_files) == 0:
        print("⚠️  Warning: No embedding or index files found")
        print("   You may need to generate embeddings first")
        return False
    
    return True

def run_quick_verification():
    """Run a quick verification test"""
    print("\n🧪 Running quick verification...")
    
    try:
        # Try to import the main classes
        from offline_contextual_retrieval import OfflineContextualRetrieval
        print("✅ Can import OfflineContextualRetrieval")
        
        # Try to create test data
        result = subprocess.run([sys.executable, 'create_sample_test_data.py'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Can create test data")
        else:
            print(f"❌ Test data creation failed: {result.stderr}")
            return False
        
        print("✅ Basic verification passed")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("RETRIEVAL PERFORMANCE TESTING - SETUP")
    print("="*60)
    
    print("\n1. Checking Python version...")
    if not check_python_version():
        return 1
    
    print("\n2. Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        install_choice = input(f"\nInstall missing packages? (y/n): ").lower().strip()
        if install_choice == 'y':
            if not install_dependencies(missing_packages):
                print("❌ Failed to install some dependencies")
                return 1
        else:
            print("⚠️  Some dependencies are missing. Tests may not work properly.")
    
    print("\n3. Checking required files...")
    missing_files = check_required_files()
    
    if missing_files:
        print(f"\n⚠️  Missing required files: {missing_files}")
        print("Please ensure all required files are in place:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    
    print("\n4. Creating directories...")
    create_directories()
    
    print("\n5. Checking embedding files...")
    check_embedding_files()
    
    print("\n6. Running verification...")
    if run_quick_verification():
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE!")
        print("="*60)
        print("\nYou can now run:")
        print("  python run_retrieval_test.py")
        print("\nOr for advanced usage:")
        print("  python retrieval_performance_tester.py --help")
        return 0
    else:
        print("\n" + "="*60)
        print("❌ SETUP INCOMPLETE")
        print("="*60)
        print("Please fix the issues above before running tests.")
        return 1

if __name__ == "__main__":
    exit(main())