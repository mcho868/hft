#!/usr/bin/env python3
"""
Contextual Retrieval Demo - Semantic Search + BM25 + Rank Fusion

This script demonstrates contextual retrieval using:
1. Semantic search (FAISS + SentenceTransformers)
2. BM25 keyword search
3. Reciprocal Rank Fusion to combine results

Usage:
    python contextual_retrieval_demo.py
    Then enter your queries interactively
"""

import os
import json
import numpy as np
import faiss
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BM25Index:
    """BM25 search index for keyword-based retrieval"""
    
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        
        # Extract text for BM25 indexing
        texts = []
        for chunk in chunks:
            # Use contextual text if available, otherwise original text
            if chunk.get('is_contextual') and chunk.get('contextual_info'):
                text = f"{chunk['text']} {chunk['contextual_info']}"
            else:
                text = chunk['text']
            texts.append(text)
        
        # Tokenize and build BM25 index
        logger.debug(f"Building BM25 index for {len(texts)} chunks")
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        logger.debug("BM25 index built successfully")
    
    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Search and return (chunk_index, score) pairs"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

def reciprocal_rank_fusion(semantic_results: List[Tuple[int, float]], 
                          bm25_results: List[Tuple[int, float]], 
                          semantic_weight: float = 0.7, 
                          bm25_weight: float = 0.3,
                          k: int = 60) -> List[Tuple[int, float]]:
    """
    Combine semantic and BM25 results using Reciprocal Rank Fusion (RRF)
    
    Args:
        semantic_results: List of (chunk_index, score) from semantic search
        bm25_results: List of (chunk_index, score) from BM25 search
        semantic_weight: Weight for semantic results
        bm25_weight: Weight for BM25 results
        k: RRF parameter (typically 60)
    
    Returns:
        List of (chunk_index, combined_score) sorted by combined score
    """
    
    chunk_scores = {}
    
    # Add semantic scores using RRF
    for rank, (chunk_idx, score) in enumerate(semantic_results):
        rrf_score = semantic_weight / (k + rank + 1)
        chunk_scores[chunk_idx] = chunk_scores.get(chunk_idx, 0) + rrf_score
    
    # Add BM25 scores using RRF
    for rank, (chunk_idx, score) in enumerate(bm25_results):
        rrf_score = bm25_weight / (k + rank + 1)
        chunk_scores[chunk_idx] = chunk_scores.get(chunk_idx, 0) + rrf_score
    
    # Sort by combined score
    sorted_results = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

class ContextualRetriever:
    """Contextual retriever using semantic search + BM25 + rank fusion"""
    
    def __init__(self, data_path: str, embeddings_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.data_path = Path(data_path)
        self.embeddings_path = Path(embeddings_path)
        self.model_name = model_name
        
        # Initialize the sentence transformer model
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Storage for loaded data
        self.sources = {}  # source_name -> {index, chunks, bm25}
        self.available_sources = []
        self.current_chunking_method = None
        
        logger.info(f"ContextualRetriever initialized")
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
        
        # Build BM25 index
        logger.info(f"Building BM25 index for {source_name}")
        bm25_index = BM25Index(chunks)
        
        # Store loaded data
        self.sources[source_name] = {
            'index': index,
            'chunks': chunks,
            'bm25': bm25_index
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
    
    def search_contextual(self, query: str, top_k: int = 10, sources: List[str] = None,
                         semantic_weight: float = 0.7, bm25_weight: float = 0.3,
                         retrieval_k: int = 50) -> List[Dict[str, Any]]:
        """
        Perform contextual search using semantic + BM25 + rank fusion
        
        Args:
            query: Search query
            top_k: Number of final results to return
            sources: Sources to search in
            semantic_weight: Weight for semantic search results
            bm25_weight: Weight for BM25 search results
            retrieval_k: Number of candidates to retrieve from each method before fusion
        """
        
        if not self.available_sources:
            logger.error("No sources loaded. Please load sources first.")
            return []
        
        if sources is None:
            sources = self.available_sources
        
        # Encode the query for semantic search
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
            bm25 = source_data['bm25']
            
            # 1. Semantic search
            num_chunks = len(chunks)
            search_k = min(retrieval_k, num_chunks)
            
            distances, indices = index.search(query_embedding, search_k)
            semantic_results = [(int(idx), float(1.0 / (1.0 + dist))) 
                              for dist, idx in zip(distances[0], indices[0]) 
                              if idx >= 0 and idx < num_chunks]
            
            # 2. BM25 search
            bm25_results = bm25.search(query, top_k=search_k)
            
            # 3. Rank fusion
            fused_results = reciprocal_rank_fusion(
                semantic_results, bm25_results, 
                semantic_weight, bm25_weight
            )
            
            # 4. Collect final results for this source
            source_top_k = min(top_k, len(fused_results))
            for i, (chunk_idx, score) in enumerate(fused_results[:source_top_k]):
                if chunk_idx < len(chunks):
                    chunk = chunks[chunk_idx].copy()
                    
                    # Add retrieval metadata
                    chunk['retrieval_score'] = float(score)
                    chunk['retrieval_rank'] = i + 1
                    chunk['retrieval_source'] = source_name
                    chunk['retrieval_type'] = 'contextual_rag'
                    
                    # Add component scores for debugging
                    semantic_score = next((s for idx, s in semantic_results if idx == chunk_idx), 0.0)
                    bm25_score = next((s for idx, s in bm25_results if idx == chunk_idx), 0.0)
                    chunk['semantic_score'] = semantic_score
                    chunk['bm25_score'] = bm25_score
                    
                    all_results.append(chunk)
        
        # Sort all results by combined score and return top_k
        all_results.sort(key=lambda x: x['retrieval_score'], reverse=True)
        return all_results[:top_k]
    
    def search_semantic_only(self, query: str, top_k: int = 10, sources: List[str] = None) -> List[Dict[str, Any]]:
        """Perform semantic-only search for comparison"""
        
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
                    chunk['retrieval_score'] = float(1.0 / (1.0 + dist))
                    chunk['retrieval_rank'] = i + 1
                    chunk['retrieval_source'] = source_name
                    chunk['retrieval_type'] = 'semantic_only'
                    chunk['retrieval_distance'] = float(dist)
                    
                    all_results.append(chunk)
        
        # Sort all results by score and return top_k
        all_results.sort(key=lambda x: x['retrieval_score'], reverse=True)
        return all_results[:top_k]
    
    def format_result(self, result: Dict[str, Any], rank: int, show_debug: bool = False) -> str:
        """Format a single search result for display"""
        score = result.get('retrieval_score', 0)
        source = result.get('retrieval_source', 'unknown')
        source_doc = result.get('source_document', 'unknown')
        ret_type = result.get('retrieval_type', 'unknown')
        text = result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', '')
        
        debug_info = ""
        if show_debug and ret_type == 'contextual_rag':
            semantic_score = result.get('semantic_score', 0)
            bm25_score = result.get('bm25_score', 0)
            debug_info = f"  Semantic: {semantic_score:.4f}, BM25: {bm25_score:.4f}\n"
        
        return f"""
Rank {rank}:
  Score: {score:.4f} ({ret_type})
  Source: {source}
  Document: {source_doc}
{debug_info}  Text: {text}
  ---
"""

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Contextual Retrieval Demo")
    parser.add_argument("--data-path", default="RAGdatav4", help="Path to data directory")
    parser.add_argument("--embeddings-path", default="RAGdatav4/indiv_embeddings", help="Path to embeddings directory")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--chunking-method", default="fixed_c512_o100", help="Chunking method to use")
    parser.add_argument("--sources", nargs="+", default=["healthify", "mayo", "nhs"], help="Sources to search")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--semantic-weight", type=float, default=0.7, help="Weight for semantic search")
    parser.add_argument("--bm25-weight", type=float, default=0.3, help="Weight for BM25 search")
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 CONTEXTUAL RETRIEVAL DEMO (Semantic + BM25 + Rank Fusion)")
    print("="*80)
    print(f"Data Path: {args.data_path}")
    print(f"Embeddings Path: {args.embeddings_path}")
    print(f"Model: {args.model}")
    print(f"Chunking Method: {args.chunking_method}")
    print(f"Sources: {args.sources}")
    print(f"Top-K: {args.top_k}")
    print(f"Weights: Semantic={args.semantic_weight}, BM25={args.bm25_weight}")
    print("="*80)
    
    # Initialize retriever
    retriever = ContextualRetriever(
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
    print("  - Enter any text to search (contextual retrieval)")
    print("  - 'semantic [query]' for semantic-only search")
    print("  - 'contextual [query]' for contextual search (default)")
    print("  - 'compare [query]' to see both methods side by side")
    print("  - 'debug [query]' to show score breakdowns")
    print("  - 'quit' or 'exit' to quit")
    print("  - 'info' to show current configuration")
    print("="*80)
    
    # Interactive query loop
    while True:
        try:
            user_input = input("\n🔍 Enter your query: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'info':
                print(f"""
Current Configuration:
  Chunking Method: {retriever.current_chunking_method}
  Loaded Sources: {retriever.available_sources}
  Model: {args.model}
  Top-K: {args.top_k}
  Semantic Weight: {args.semantic_weight}
  BM25 Weight: {args.bm25_weight}
""")
                continue
            
            # Parse command
            parts = user_input.split(' ', 1)
            command = parts[0].lower()
            query = parts[1] if len(parts) > 1 else user_input
            
            if command == 'semantic' and len(parts) > 1:
                # Semantic-only search
                print(f"\n🔎 Semantic-only search for: '{query}'")
                print("-" * 80)
                
                results = retriever.search_semantic_only(query, top_k=args.top_k, sources=args.sources)
                
                if not results:
                    print("❌ No results found.")
                    continue
                
                print(f"📋 Found {len(results)} results (semantic only):\n")
                for i, result in enumerate(results, 1):
                    print(retriever.format_result(result, i))
            
            elif command == 'contextual' and len(parts) > 1:
                query = parts[1]
                
            elif command == 'debug' and len(parts) > 1:
                # Contextual search with debug info
                query = parts[1]
                print(f"\n🔎 Contextual search (with debug) for: '{query}'")
                print("-" * 80)
                
                results = retriever.search_contextual(
                    query, top_k=args.top_k, sources=args.sources,
                    semantic_weight=args.semantic_weight, bm25_weight=args.bm25_weight
                )
                
                if not results:
                    print("❌ No results found.")
                    continue
                
                print(f"📋 Found {len(results)} results (contextual with debug):\n")
                for i, result in enumerate(results, 1):
                    print(retriever.format_result(result, i, show_debug=True))
                
                continue
            
            elif command == 'compare' and len(parts) > 1:
                # Compare both methods
                query = parts[1]
                print(f"\n🔎 Comparing methods for: '{query}'")
                print("=" * 80)
                
                # Semantic search
                print("📊 SEMANTIC-ONLY RESULTS:")
                print("-" * 40)
                semantic_results = retriever.search_semantic_only(query, top_k=5, sources=args.sources)
                for i, result in enumerate(semantic_results, 1):
                    print(retriever.format_result(result, i))
                
                # Contextual search
                print("\n📊 CONTEXTUAL RESULTS (Semantic + BM25 + Fusion):")
                print("-" * 40)
                contextual_results = retriever.search_contextual(
                    query, top_k=5, sources=args.sources,
                    semantic_weight=args.semantic_weight, bm25_weight=args.bm25_weight
                )
                for i, result in enumerate(contextual_results, 1):
                    print(retriever.format_result(result, i, show_debug=True))
                
                continue
            
            # Default: contextual search
            if command not in ['semantic', 'contextual', 'debug', 'compare']:
                query = user_input
            
            print(f"\n🔎 Contextual search for: '{query}'")
            print("-" * 80)
            
            # Perform contextual search
            results = retriever.search_contextual(
                query, top_k=args.top_k, sources=args.sources,
                semantic_weight=args.semantic_weight, bm25_weight=args.bm25_weight
            )
            
            if not results:
                print("❌ No results found.")
                continue
            
            print(f"📋 Found {len(results)} results (contextual):\n")
            
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