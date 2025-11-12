#!/usr/bin/env python3
"""
Agent Chunk Deduplicator Script

This script identifies and removes duplicate chunks from agent-generated JSON files
in the RAGdata folder, then regenerates the corresponding FAISS index files.

The script:
1. Finds all agent-generated chunk JSON files (*_chunks_agent_*.json)
2. Removes duplicate chunks using various similarity measures
3. Saves cleaned files with "_agent_cleaned" suffix
4. Regenerates FAISS index files for the cleaned chunks
"""

import os
import json
import glob
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

# Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
RAG_DATA_DIR = "/Users/choemanseung/789/hft/RAGdatav2"
MIN_CHUNK_LENGTH = 50  # Minimum chunk length to keep

def find_duplicate_chunks(chunk_objects: List[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Find and remove duplicate chunks using simple, fast methods:
    1. Remove short chunks
    2. Remove exact duplicates based on text content
    3. Remove chunks that are exact subsets of other chunks
    
    Returns:
        - List of unique chunk objects with reindexed chunk_ids
        - Dictionary with statistics about duplicates removed
    """
    print(f"  Processing {len(chunk_objects)} chunks for duplicates...")
    
    stats = {
        "original_count": len(chunk_objects),
        "exact_duplicates": 0,
        "subset_duplicates": 0,
        "short_chunks_removed": 0,
        "final_count": 0
    }
    
    # Step 1: Remove short chunks
    print("    Removing short chunks...")
    valid_chunks = []
    for chunk_obj in chunk_objects:
        text = chunk_obj.get("text", "").strip()
        if len(text) >= MIN_CHUNK_LENGTH:
            valid_chunks.append(chunk_obj)
        else:
            stats["short_chunks_removed"] += 1
    
    print(f"    After removing short chunks: {len(valid_chunks)} remaining")
    
    # Step 2: Remove exact duplicates based on text content
    print("    Removing exact duplicates...")
    seen_texts = set()
    unique_chunks = []
    
    for chunk_obj in valid_chunks:
        text = chunk_obj.get("text", "").strip()
        if text not in seen_texts:
            seen_texts.add(text)
            unique_chunks.append(chunk_obj)
        else:
            stats["exact_duplicates"] += 1
    
    print(f"    After removing exact duplicates: {len(unique_chunks)} remaining")
    
    # Step 3: Remove subset chunks (chunks whose text is completely contained in other chunks)
    print("    Removing subset chunks...")
    final_chunks = []
    
    # Sort by text length (longest first) for efficient subset detection
    unique_chunks.sort(key=lambda x: len(x.get("text", "")), reverse=True)
    
    for chunk_obj in unique_chunks:
        text = chunk_obj.get("text", "").strip()
        is_subset = False
        
        # Check if this chunk's text is a subset of any already accepted chunk's text
        for accepted_chunk_obj in final_chunks:
            accepted_text = accepted_chunk_obj.get("text", "").strip()
            if text in accepted_text and text != accepted_text:
                is_subset = True
                stats["subset_duplicates"] += 1
                break
        
        if not is_subset:
            final_chunks.append(chunk_obj)
    
    # Step 4: Reindex chunk_ids
    base_name = ""
    if final_chunks:
        original_id = final_chunks[0].get("chunk_id", "")
        # Extract base name from chunk_id (everything before the last "_chunk_X")
        parts = original_id.split("_chunk_")
        if len(parts) >= 2:
            base_name = parts[0]
    
    for i, chunk_obj in enumerate(final_chunks):
        chunk_obj["chunk_id"] = f"{base_name}_chunk_{i}"
    
    stats["final_count"] = len(final_chunks)
    
    print(f"  Removed {stats['short_chunks_removed']} short chunks")
    print(f"  Removed {stats['exact_duplicates']} exact duplicates")
    print(f"  Removed {stats['subset_duplicates']} subset duplicates")
    print(f"  Final count: {stats['final_count']} unique chunks")
    
    return final_chunks, stats

def generate_embeddings(chunk_objects: List[Dict], model_name: str) -> np.ndarray:
    """Generate embeddings for chunks using SentenceTransformer"""
    print(f"  Generating embeddings for {len(chunk_objects)} chunks...")
    texts = [chunk_obj.get("text", "") for chunk_obj in chunk_objects]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.astype('float32')

def build_and_save_faiss_index(embeddings: np.ndarray, index_path: str):
    """Build and save FAISS index"""
    print(f"  Building FAISS index with {embeddings.shape[0]} vectors...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"  FAISS index saved to: {index_path}")

def process_agent_file(json_file_path: str) -> Dict[str, any]:
    """Process a single agent-generated JSON file"""
    print(f"\nProcessing: {json_file_path}")
    
    # Load original chunks
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            original_chunks = json.load(f)
    except Exception as e:
        print(f"  Error loading file: {e}")
        return {"error": str(e)}
    
    if not isinstance(original_chunks, list):
        print(f"  Error: Expected list of chunks, got {type(original_chunks)}")
        return {"error": "Invalid file format"}
    
    # Remove duplicates
    cleaned_chunks, stats = find_duplicate_chunks(original_chunks)
    
    if len(cleaned_chunks) == 0:
        print(f"  Warning: No chunks remaining after deduplication!")
        return {"error": "No chunks remaining"}
    
    # Generate output filenames
    base_name = os.path.basename(json_file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Create deduplicated filename by adding "_deduplicated" suffix
    cleaned_name = f"{name_without_ext}_deduplicated"
    
    cleaned_json_path = os.path.join(RAG_DATA_DIR, f"{cleaned_name}.json")
    cleaned_index_path = os.path.join(RAG_DATA_DIR, f"{cleaned_name.replace('_chunks_', '_vector_db_')}.index")
    
    # Save cleaned chunks
    try:
        with open(cleaned_json_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_chunks, f, ensure_ascii=False, indent=2)
        print(f"  Cleaned chunks saved to: {cleaned_json_path}")
    except Exception as e:
        print(f"  Error saving cleaned chunks: {e}")
        return {"error": f"Error saving chunks: {e}"}
    
    # Generate embeddings and FAISS index
    try:
        embeddings = generate_embeddings(cleaned_chunks, EMBEDDING_MODEL_NAME)
        build_and_save_faiss_index(embeddings, cleaned_index_path)
    except Exception as e:
        print(f"  Error generating embeddings/index: {e}")
        return {"error": f"Error generating index: {e}"}
    
    result = {
        "original_file": json_file_path,
        "cleaned_json": cleaned_json_path,
        "cleaned_index": cleaned_index_path,
        "stats": stats,
        "success": True
    }
    
    return result

def main():
    """Main function to process all agent-generated files"""
    print("="*80)
    print("AGENT CHUNK DEDUPLICATOR (SIMPLIFIED)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Data directory: {RAG_DATA_DIR}")
    print(f"  Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"  Minimum chunk length: {MIN_CHUNK_LENGTH}")
    print(f"  Method: Exact duplicates + subset removal")
    print("="*80)
    
    # Find all agent-generated JSON files with _chunks_agent_mlx pattern
    agent_pattern = os.path.join(RAG_DATA_DIR, "*_chunks_agent_mlx*.json")
    agent_files = glob.glob(agent_pattern)
    
    if not agent_files:
        print("No agent-generated chunk files found!")
        return
    
    print(f"Found {len(agent_files)} agent-generated files:")
    for file in agent_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each file
    results = []
    total_stats = {
        "files_processed": 0,
        "files_failed": 0,
        "total_original_chunks": 0,
        "total_final_chunks": 0,
        "total_exact_duplicates": 0,
        "total_subset_duplicates": 0,
        "total_short_chunks": 0
    }
    
    for agent_file in agent_files:
        result = process_agent_file(agent_file)
        results.append(result)
        
        if result.get("success"):
            total_stats["files_processed"] += 1
            stats = result["stats"]
            total_stats["total_original_chunks"] += stats["original_count"]
            total_stats["total_final_chunks"] += stats["final_count"]
            total_stats["total_exact_duplicates"] += stats["exact_duplicates"]
            total_stats["total_subset_duplicates"] += stats["subset_duplicates"]
            total_stats["total_short_chunks"] += stats["short_chunks_removed"]
        else:
            total_stats["files_failed"] += 1
    
    # Print summaryj
    print("\n" + "="*80)
    print("DEDUPLICATION COMPLETE - SUMMARY")
    print("="*80)
    print(f"Files processed successfully: {total_stats['files_processed']}")
    print(f"Files failed: {total_stats['files_failed']}")
    print(f"Total original chunks: {total_stats['total_original_chunks']:,}")
    print(f"Total final chunks: {total_stats['total_final_chunks']:,}")
    print(f"Chunks removed:")
    print(f"  - Exact duplicates: {total_stats['total_exact_duplicates']:,}")
    print(f"  - Subset duplicates: {total_stats['total_subset_duplicates']:,}")
    print(f"  - Short chunks: {total_stats['total_short_chunks']:,}")
    
    total_removed = (total_stats['total_exact_duplicates'] + 
                    total_stats['total_subset_duplicates'] + 
                    total_stats['total_short_chunks'])
    
    if total_stats['total_original_chunks'] > 0:
        reduction_percentage = (total_removed / total_stats['total_original_chunks']) * 100
        print(f"Total reduction: {total_removed:,} chunks ({reduction_percentage:.1f}%)")
    
    print("\nCleaned files created:")
    for result in results:
        if result.get("success"):
            print(f"  - {os.path.basename(result['cleaned_json'])}")
            print(f"  - {os.path.basename(result['cleaned_index'])}")
    
    if total_stats['files_failed'] > 0:
        print(f"\nFailed files:")
        for result in results:
            if not result.get("success"):
                print(f"  - {result.get('original_file', 'Unknown')}: {result.get('error', 'Unknown error')}")
    
    print("="*80)

if __name__ == "__main__":
    main()