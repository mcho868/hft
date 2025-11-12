#!/usr/bin/env python3
"""
Basic Retrieval Demo - Simple Semantic Search

This script demonstrates basic semantic retrieval using FAISS and sentence transformers.
It provides a CLI interface to query the medical knowledge base and returns top 10 results.

Usage:
    python basic_retrieval_demo.py
    Then enter your queries interactively
"""

import os
import json
import numpy as np
import faiss
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicRetriever:
    """Simple semantic retriever using FAISS and SentenceTransformers"""
    
    def __init__(self, data_path: str, embeddings_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.data_path = Path(data_path)
        self.embeddings_path = Path(embeddings_path)
        self.model_name = model_name
        
        # Initialize the sentence transformer model
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Storage for loaded data
        self.sources = {}  # source_name -> {index, chunks}
        self.available_sources = []
        self.current_chunking_method = None
        
        logger.info(f"BasicRetriever initialized")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Embeddings path: {self.embeddings_path}")
    
    def discover_available_sources_and_methods(self):
        """Discover available data sources and chunking methods"""
        sources_methods = {}
        
        # Look for chunk files
        for chunk_file in self.data_path.glob("*_chunks_*.json"):
            filename = chunk_file.stem
            if "_chunks_" in filename:
                parts = filename.split("_chunks_", 1)
                source = parts[0]
                method = parts[1]
                
                if source not in sources_methods:
                    sources_methods[source] = []
                sources_methods[source].append(method)
        
        logger.info("Available sources and chunking methods:")
        for source, methods in sources_methods.items():
            logger.info(f"  {source}: {methods}")
        
        return sources_methods
    
    def load_source(self, source_name: str, chunking_method: str):
        """Load a specific source with given chunking method"""
        
        # Load chunks
        chunk_file = self.data_path / f"{source_name}_chunks_{chunking_method}.json"
        if not chunk_file.exists():
            logger.error(f"Chunk file not found: {chunk_file}")
            return False
        
        logger.info(f"Loading chunks from: {chunk_file}")
        with open(chunk_file, 'r') as f:
            chunks = json.load(f)
        
        # Load FAISS index
        index_file = self.embeddings_path / f"{source_name}_vector_db_{chunking_method}.index"
        if not index_file.exists():
            logger.error(f"Index file not found: {index_file}")
            return False
        
        logger.info(f"Loading FAISS index from: {index_file}")
        index = faiss.read_index(str(index_file))
        
        # Store loaded data
        self.sources[source_name] = {
            'index': index,
            'chunks': chunks
        }
        
        logger.info(f"Successfully loaded {source_name} with {len(chunks)} chunks")
        return True
    
    def load_all_sources(self, chunking_method: str, sources: List[str] = None):
        """Load all available sources for a specific chunking method"""
        if sources is None:
            sources = ['healthify', 'mayo', 'nhs']
        
        self.current_chunking_method = chunking_method
        logger.info(f"Loading all sources with chunking method: {chunking_method}")
        
        loaded_sources = []
        for source in sources:
            if self.load_source(source, chunking_method):
                loaded_sources.append(source)
        
        self.available_sources = loaded_sources
        logger.info(f"Loaded sources: {self.available_sources}")
        return len(loaded_sources) > 0
    
    def search(self, query: str, top_k: int = 10, sources: List[str] = None) -> List[Dict[str, Any]]:
        """Perform semantic search across specified sources"""
        
        if not self.available_sources:
            logger.error("No sources loaded. Please load sources first.")
            return []
        
        if sources is None:
            sources = self.available_sources
        
        # Encode the query
        logger.debug(f"Encoding query: {query}")
        query_embedding = self.model.encode([query]).astype('float32')
        
        all_results = []
        
        # Search in each source
        for source_name in sources:
            if source_name not in self.sources:
                logger.warning(f"Source {source_name} not loaded, skipping")
                continue
            
            source_data = self.sources[source_name]
            index = source_data['index']
            chunks = source_data['chunks']
            
            # Perform search
            distances, indices = index.search(query_embedding, min(top_k, len(chunks)))
            
            # Collect results from this source
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx < len(chunks):
                    chunk = chunks[idx].copy()
                    
                    # Add retrieval metadata
                    chunk['retrieval_score'] = float(1.0 / (1.0 + dist))  # Convert distance to similarity
                    chunk['retrieval_rank'] = i + 1
                    chunk['retrieval_source'] = source_name
                    chunk['retrieval_distance'] = float(dist)
                    
                    all_results.append(chunk)
        
        # Sort all results by score and return top_k
        all_results.sort(key=lambda x: x['retrieval_score'], reverse=True)
        return all_results[:top_k]
    
    def format_result(self, result: Dict[str, Any], rank: int) -> str:
        """Format a single search result for display"""
        score = result.get('retrieval_score', 0)
        source = result.get('retrieval_source', 'unknown')
        source_doc = result.get('source_document', 'unknown')
        text = result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', '')
        
        return f"""
Rank {rank}:
  Score: {score:.4f}
  Source: {source}
  Document: {source_doc}
  Text: {text}
  ---
"""

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Basic Retrieval Demo")
    parser.add_argument("--data-path", default="RAGdatav4", help="Path to data directory")
    parser.add_argument("--embeddings-path", default="RAGdatav4/indiv_embeddings", help="Path to embeddings directory")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--chunking-method", default="fixed_c512_o100", help="Chunking method to use")
    parser.add_argument("--sources", nargs="+", default=["healthify", "mayo", "nhs"], help="Sources to search")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 BASIC RETRIEVAL DEMO")
    print("="*80)
    print(f"Data Path: {args.data_path}")
    print(f"Embeddings Path: {args.embeddings_path}")
    print(f"Model: {args.model}")
    print(f"Chunking Method: {args.chunking_method}")
    print(f"Sources: {args.sources}")
    print(f"Top-K: {args.top_k}")
    print("="*80)
    
    # Initialize retriever
    retriever = BasicRetriever(
        data_path=args.data_path,
        embeddings_path=args.embeddings_path,
        model_name=args.model
    )
    
    # Discover available data
    sources_methods = retriever.discover_available_sources_and_methods()
    
    # Load specified sources
    print(f"\nLoading sources with chunking method: {args.chunking_method}")
    if not retriever.load_all_sources(args.chunking_method, args.sources):
        print("❌ Failed to load any sources. Please check your data paths and chunking method.")
        sys.exit(1)
    
    print(f"✅ Successfully loaded sources: {retriever.available_sources}")
    print("\n" + "="*80)
    print("Ready to search! Enter your queries below.")
    print("Commands:")
    print("  - Enter any text to search")
    print("  - 'quit' or 'exit' to quit")
    print("  - 'info' to show current configuration")
    print("="*80)
    
    # Interactive query loop
    while True:
        try:
            query = input("\n🔍 Enter your query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'info':
                print(f"""
Current Configuration:
  Chunking Method: {retriever.current_chunking_method}
  Loaded Sources: {retriever.available_sources}
  Model: {args.model}
  Top-K: {args.top_k}
""")
                continue
            
            print(f"\n🔎 Searching for: '{query}'")
            print("-" * 80)
            
            # Perform search
            results = retriever.search(query, top_k=args.top_k, sources=args.sources)
            
            if not results:
                print("❌ No results found.")
                continue
            
            print(f"📋 Found {len(results)} results:\n")
            
            # Display results
            for i, result in enumerate(results, 1):
                print(retriever.format_result(result, i))
            
            print("=" * 80)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Search error: {e}")

if __name__ == "__main__":
    main()