"""
Index-Only Builder Script
Builds individual FAISS indices from existing chunk files in RAGdatav4/
Uses cosine similarity and saves to RAGdatav4/indiv_embeddings/
"""

import os
import json
import numpy as np
import faiss
import time
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import glob

# --- Configuration ---
CHUNK_DIR = '/Users/choemanseung/789/hft/RAGdatav4'
CACHE_DIR = '/Users/choemanseung/789/hft/RAGdatav4/indiv_embeddings'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def discover_chunk_files() -> Dict[str, List[str]]:
    """Discover all chunk files and group by strategy and source"""
    chunk_files = {}
    
    # Find all chunk JSON files
    pattern = os.path.join(CHUNK_DIR, "*_chunks_*.json")
    files = glob.glob(pattern)
    
    for file_path in files:
        filename = os.path.basename(file_path)
        
        # Parse filename: {source}_chunks_{strategy}.json
        if "_chunks_" in filename:
            parts = filename.replace('.json', '').split('_chunks_')
            if len(parts) == 2:
                source = parts[0]  # healthify, mayo, nhs
                strategy = parts[1]  # fixed_c256_o0, sentence_t384_o1, etc.
                
                if strategy not in chunk_files:
                    chunk_files[strategy] = {}
                
                chunk_files[strategy][source] = file_path
    
    return chunk_files

def load_chunks_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load chunks from a JSON file"""
    print(f"Loading chunks from {os.path.basename(file_path)}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Ensure chunks have required fields
    for chunk in chunks:
        if 'original_text' not in chunk and 'text' in chunk:
            chunk['original_text'] = chunk['text']
        if 'is_contextual' not in chunk:
            chunk['is_contextual'] = 'contextual_info' in chunk and chunk['contextual_info']
    
    print(f"Loaded {len(chunks)} chunks")
    return chunks

def get_text_for_embedding(chunk: Dict[str, Any]) -> str:
    """Extract text for embedding, handling contextual chunks"""
    if chunk.get('is_contextual') and chunk.get('contextual_info'):
        # For contextual chunks, combine original text + context
        return f"{chunk['original_text']}\n\nContext: {chunk['contextual_info']}"
    else:
        return chunk.get('original_text', chunk.get('text', ''))

def generate_embeddings(chunks: List[Dict[str, Any]], model_name: str) -> np.ndarray:
    """Generate embeddings for chunks"""
    print(f"Generating embeddings with {model_name}...")
    
    model = SentenceTransformer(model_name)
    
    # Extract texts for embedding
    texts_to_embed = [get_text_for_embedding(chunk) for chunk in chunks]
    
    # Generate embeddings in batches
    batch_size = 8
    embeddings = []
    
    for i in tqdm(range(0, len(texts_to_embed), batch_size), 
                 desc="Generating embeddings"):
        batch = texts_to_embed[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings).astype('float32')
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def build_faiss_index(embeddings: np.ndarray, index_path: str) -> None:
    """Build and save FAISS index with cosine similarity"""
    print(f"Building FAISS index (cosine similarity)...")
    
    if embeddings.size == 0:
        print("Warning: No embeddings to index")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    d = embeddings.shape[1]
    
    # Use IndexFlatIP (Inner Product) for cosine similarity
    index = faiss.IndexFlatIP(d)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path} ({index.ntotal} vectors)")

def build_index_for_source_strategy(source: str, strategy: str, chunk_file: str) -> bool:
    """Build index for a specific source and strategy"""
    print(f"\n{'='*60}")
    print(f"Building index: {source} - {strategy}")
    print('='*60)
    
    # Define output path
    index_filename = f"{source}_vector_db_{strategy}.index"
    index_path = os.path.join(CACHE_DIR, index_filename)
    
    # Skip if already exists
    if os.path.exists(index_path):
        print(f"⏭️  Skipping: {index_filename} already exists")
        return True
    
    try:
        # Load chunks
        chunks = load_chunks_from_file(chunk_file)
        if not chunks:
            print(f"❌ No chunks found in {chunk_file}")
            return False
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks, EMBEDDING_MODEL_NAME)
        
        # Build and save index
        build_faiss_index(embeddings, index_path)
        
        print(f"✅ Successfully built index for {source} - {strategy}")
        return True
        
    except Exception as e:
        print(f"❌ Error building index for {source} - {strategy}: {e}")
        return False

def main():
    """Main execution function"""
    print("Individual Source Index Builder")
    print("=" * 50)
    print(f"Chunk directory: {CHUNK_DIR}")
    print(f"Output directory: {CACHE_DIR}")
    print(f"Model: {EMBEDDING_MODEL_NAME}")
    print(f"Similarity: Cosine (IndexFlatIP)")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Discover chunk files
    print("\nDiscovering chunk files...")
    chunk_files = discover_chunk_files()
    
    if not chunk_files:
        print("❌ No chunk files found!")
        print(f"Expected files like: healthify_chunks_fixed_c256_o0.json in {CHUNK_DIR}")
        return
    
    # Display found strategies and sources
    print(f"\nFound {len(chunk_files)} strategies:")
    total_combinations = 0
    for strategy, sources in chunk_files.items():
        print(f"  📋 {strategy}: {list(sources.keys())}")
        total_combinations += len(sources)
    
    print(f"\nTotal source-strategy combinations: {total_combinations}")
    
    # Ask user what to build
    print(f"\nOptions:")
    print("1. Build all indices")
    print("2. Build specific strategy")
    print("3. Build specific source")
    
    choice = input("Enter choice (1-3): ").strip()
    
    start_time = time.time()
    successful_builds = 0
    
    if choice == "1":
        # Build all
        print(f"\n🔨 Building all {total_combinations} indices...")
        for strategy, sources in chunk_files.items():
            for source, chunk_file in sources.items():
                if build_index_for_source_strategy(source, strategy, chunk_file):
                    successful_builds += 1
    
    elif choice == "2":
        # Build specific strategy
        print("\nAvailable strategies:")
        strategies = list(chunk_files.keys())
        for i, strategy in enumerate(strategies, 1):
            print(f"  {i}. {strategy}")
        
        try:
            strategy_idx = int(input("Enter strategy number: ")) - 1
            selected_strategy = strategies[strategy_idx]
            sources = chunk_files[selected_strategy]
            
            print(f"\n🔨 Building indices for strategy: {selected_strategy}")
            for source, chunk_file in sources.items():
                if build_index_for_source_strategy(source, selected_strategy, chunk_file):
                    successful_builds += 1
        except (ValueError, IndexError):
            print("❌ Invalid strategy selection")
            return
    
    elif choice == "3":
        # Build specific source
        all_sources = set()
        for sources in chunk_files.values():
            all_sources.update(sources.keys())
        all_sources = sorted(list(all_sources))
        
        print("\nAvailable sources:")
        for i, source in enumerate(all_sources, 1):
            print(f"  {i}. {source}")
        
        try:
            source_idx = int(input("Enter source number: ")) - 1
            selected_source = all_sources[source_idx]
            
            print(f"\n🔨 Building indices for source: {selected_source}")
            for strategy, sources in chunk_files.items():
                if selected_source in sources:
                    chunk_file = sources[selected_source]
                    if build_index_for_source_strategy(selected_source, strategy, chunk_file):
                        successful_builds += 1
        except (ValueError, IndexError):
            print("❌ Invalid source selection")
            return
    
    else:
        print("❌ Invalid choice")
        return
    
    # Summary
    build_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("INDEX BUILDING COMPLETED")
    print('='*60)
    print(f"✅ Successfully built: {successful_builds} indices")
    print(f"⏱️  Total time: {build_time:.2f} seconds")
    print(f"📁 Indices saved to: {CACHE_DIR}")
    
    # List created files
    if successful_builds > 0:
        print(f"\nCreated index files:")
        index_files = glob.glob(os.path.join(CACHE_DIR, "*.index"))
        for index_file in sorted(index_files):
            filename = os.path.basename(index_file)
            file_size = os.path.getsize(index_file) / (1024 * 1024)  # MB
            print(f"  📄 {filename} ({file_size:.1f} MB)")

if __name__ == "__main__":
    main()