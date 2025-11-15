#!/usr/bin/env python3
"""
Simplified Offline Contextual Retrieval System
Based on the contextual retrieval notebook but adapted for offline use with local embeddings
"""

import os
import json
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import time
from collections import defaultdict

class OfflineVectorDB:
    """Simple vector database for semantic search using preloaded embeddings"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None
    
    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """Semantic search using preloaded FAISS index"""
        if self.index is None:
            raise ValueError("Index not loaded. Load data first.")
        
        # Encode query
        query_embedding = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and idx < len(self.chunks):  # Valid result
                results.append({
                    'chunk': self.chunks[idx],
                    'similarity_score': float(score),
                    'rank': i
                })
        
        return results

class BM25Search:
    """BM25 search implementation"""
    
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.corpus = []
        self.bm25 = None
        self._build_index()
    
    def _build_index(self):
        """Build BM25 index"""
        for chunk in self.chunks:
            # Use text field for BM25 search
            text = chunk.get('text', chunk.get('original_text', ''))
            if chunk.get('is_contextual') and chunk.get('contextual_info'):
                text = f"{text} {chunk['contextual_info']}"
            
            # Tokenize
            tokens = text.lower().split()
            self.corpus.append(tokens)
        
        self.bm25 = BM25Okapi(self.corpus)
    
    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """BM25 search"""
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if idx < len(self.chunks):
                results.append({
                    'chunk': self.chunks[idx],
                    'bm25_score': float(scores[idx]),
                    'rank': i
                })
        
        return results

class OfflineReranker:
    """Simple offline reranking methods"""
    
    def __init__(self, method: str = "feature"):
        self.method = method
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    def rerank(self, query: str, chunks: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rerank chunks using specified method"""
        if self.method == "feature":
            return self._feature_rerank(query, chunks, top_k)
        elif self.method == "tfidf":
            return self._tfidf_rerank(query, chunks, top_k)
        else:
            return chunks[:top_k]  # No reranking
    
    def _get_text_for_chunk(self, chunk: Dict) -> str:
        """Extract text from chunk for reranking"""
        if isinstance(chunk, dict) and 'chunk' in chunk:
            chunk_data = chunk['chunk']
        else:
            chunk_data = chunk
        
        # Handle different chunk formats
        text = chunk_data.get('text', chunk_data.get('original_text', ''))
        if chunk_data.get('is_contextual') and chunk_data.get('contextual_info'):
            return f"{text} {chunk_data['contextual_info']}"
        return text
    
    def _feature_rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        """Feature-based reranking"""
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            text = self._get_text_for_chunk(chunk)
            text_words = text.lower().split()
            text_set = set(text_words)
            
            # Calculate features
            exact_matches = len(query_words.intersection(text_set))
            jaccard = len(query_words.intersection(text_set)) / len(query_words.union(text_set)) if query_words.union(text_set) else 0
            query_coverage = len(query_words.intersection(text_set)) / len(query_words) if query_words else 0
            
            # Combined score
            score = exact_matches * 2.0 + jaccard * 1.5 + query_coverage * 1.0
            
            scored_chunks.append({
                'chunk': chunk,
                'relevance_score': score,
                'rerank_method': 'feature'
            })
        
        scored_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_chunks[:top_k]
    
    def _tfidf_rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        """TF-IDF based reranking"""
        texts = [self._get_text_for_chunk(chunk) for chunk in chunks]
        corpus = [query] + texts
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            query_vec = tfidf_matrix[0]
            doc_vecs = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vec, doc_vecs).flatten()
            ranked_indices = np.argsort(similarities)[::-1]
            
            reranked = []
            for i in ranked_indices[:top_k]:
                reranked.append({
                    'chunk': chunks[i],
                    'relevance_score': float(similarities[i]),
                    'rerank_method': 'tfidf'
                })
            
            return reranked
        except Exception as e:
            return chunks[:top_k]

class HybridRetriever:
    """Hybrid retrieval with RRF (Reciprocal Rank Fusion)"""
    
    def __init__(self, vector_db: OfflineVectorDB, chunks: List[Dict[str, Any]]):
        self.vector_db = vector_db
        self.bm25_search = BM25Search(chunks)
        self.chunks = chunks
    
    def reciprocal_rank_fusion(self, 
                             semantic_results: List[Dict], 
                             bm25_results: List[Dict], 
                             k: int = 60,
                             semantic_weight: float = 0.7,
                             bm25_weight: float = 0.3) -> List[Tuple[str, float]]:
        """Implement Reciprocal Rank Fusion"""
        scores = defaultdict(float)
        
        # Score semantic results
        for rank, result in enumerate(semantic_results):
            chunk_id = result['chunk'].get('chunk_id', str(rank))
            scores[chunk_id] += semantic_weight * (1 / (k + rank + 1))
        
        # Score BM25 results
        for rank, result in enumerate(bm25_results):
            chunk_id = result['chunk'].get('chunk_id', str(rank))
            scores[chunk_id] += bm25_weight * (1 / (k + rank + 1))
        
        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
    
    def search(self, query: str, 
               top_k: int = 20, 
               semantic_weight: float = 0.7, 
               bm25_weight: float = 0.3,
               initial_k: int = None) -> List[Dict]:
        """Hybrid search with RRF"""
        if initial_k is None:
            initial_k = top_k * 2
        
        # Get results from both methods
        semantic_results = self.vector_db.search(query, k=initial_k)
        bm25_results = self.bm25_search.search(query, k=initial_k)
        
        # Apply RRF
        fused_rankings = self.reciprocal_rank_fusion(
            semantic_results, bm25_results, 
            semantic_weight=semantic_weight, 
            bm25_weight=bm25_weight
        )
        
        # Convert back to chunk objects - simplified approach
        results = []
        for i, (chunk_id, fusion_score) in enumerate(fused_rankings[:top_k]):
            # Find chunk by ID or use semantic results as fallback
            chunk = None
            for res in semantic_results + bm25_results:
                if res['chunk'].get('chunk_id', '') == chunk_id:
                    chunk = res['chunk']
                    break
            
            if chunk is None and i < len(semantic_results):
                chunk = semantic_results[i]['chunk']
            
            if chunk:
                results.append({
                    'chunk': chunk,
                    'fusion_score': fusion_score,
                    'retrieval_method': 'hybrid_rrf'
                })
        
        return results

class OfflineContextualRetrieval:
    """Complete offline contextual retrieval system"""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 rerank_method: str = "feature"):
        
        self.vector_db = OfflineVectorDB(model_name)
        self.hybrid_retriever = None
        self.reranker = OfflineReranker(rerank_method)
        self.chunks = []
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               use_reranking: bool = True,
               semantic_weight: float = 0.7,
               bm25_weight: float = 0.3,
               initial_k: int = None) -> List[Dict]:
        """Complete search pipeline"""
        
        if self.hybrid_retriever is None:
            raise ValueError("No data loaded. Load data first.")
        
        start_time = time.time()
        
        # Stage 1: Hybrid retrieval
        if initial_k is None:
            initial_k = top_k * 3 if use_reranking else top_k
        
        hybrid_results = self.hybrid_retriever.search(
            query, 
            top_k=initial_k,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight
        )
        
        # Stage 2: Optional reranking
        if use_reranking and len(hybrid_results) > top_k:
            reranked_results = self.reranker.rerank(query, hybrid_results, top_k)
            final_results = reranked_results
        else:
            final_results = hybrid_results[:top_k]
        
        search_time = time.time() - start_time
        
        # Add timing info
        for result in final_results:
            result['search_time'] = search_time
            result['total_chunks'] = len(self.chunks)
        
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_chunks': len(self.chunks),
            'contextual_chunks': sum(1 for c in self.chunks if c.get('is_contextual', False)),
            'model': self.vector_db.model_name,
            'reranker': self.reranker.method
        }

def main():
    """Example usage"""
    print("OfflineContextualRetrieval system loaded successfully!")
    print("This is a standalone retrieval system for use with preloaded embeddings.")

if __name__ == "__main__":
    main()