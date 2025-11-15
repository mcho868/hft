#!/usr/bin/env python3
"""
Optimized Hybrid Retrieval Evaluator - High Performance Version

Key optimizations:
1. Persistent embedding model in memory
2. Cached indices and chunks (loaded once per chunking method)
3. Batch processing for query embeddings
4. Parallel processing support
5. Memory-efficient evaluation

Usage:
    python optimized_hybrid_evaluator.py --test-data ../generated_triage_dialogues_val.json --results-dir results/optimized_test
"""

import os
import json
import pickle
import numpy as np
import faiss
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging
from tqdm import tqdm
import re
from datetime import datetime
import psutil
import gc
import tracemalloc

# Import existing classes
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Test case from simplified format"""
    symptom: str
    query: str
    final_triage_decision: str
    next_step: str
    reasoning: str
    id: Optional[int] = None

    def get_combined_query(self) -> str:
        """Get the query for retrieval"""
        return self.query
    
    def get_expected_source_doc(self) -> str:
        """Extract the expected source document name from symptom"""
        clean_symptom = re.sub(r'\s*\(Case\s+\d+\)$', '', self.symptom)
        return f"{clean_symptom}.txt"
    
    def get_keywords_for_matching(self) -> List[str]:
        """Extract keywords from symptom for flexible matching"""
        clean_symptom = re.sub(r'\s*\(Case\s+\d+\)$', '', self.symptom)
        # Split by underscores and extract meaningful keywords
        keywords = clean_symptom.lower().replace('_', ' ').split()
        # Filter out common words
        meaningful_keywords = [k for k in keywords if len(k) > 2 and k not in ['the', 'and', 'for', 'with']]
        return meaningful_keywords

@dataclass
class BiasConfig:
    """Bias configuration for multi-source retrieval"""
    name: str
    healthify: int
    mayo: int
    nhs: int
    description: str

@dataclass
class RetrievalConfig:
    """Configuration for a specific retrieval method"""
    name: str
    chunking_method: str
    retrieval_type: str
    bias_config: BiasConfig
    embedding_file: Optional[str] = None
    index_files: Optional[Dict[str, str]] = None
    chunk_files: Optional[Dict[str, str]] = None

@dataclass
class TestResult:
    """Results from testing a specific configuration"""
    config: RetrievalConfig
    pass_at_1: float
    pass_at_2: float
    pass_at_3: float
    pass_at_4: float
    pass_at_5: float
    pass_at_10: float
    pass_at_20: float
    avg_retrieval_time: float
    total_test_cases: int
    successful_retrievals: int
    detailed_results: List[Dict[str, Any]]
    source_distribution: Dict[str, int]
    memory_stats: Dict[str, float]

class HybridBM25:
    """Optimized BM25 search for hybrid retrieval"""
    
    def __init__(self, chunks: List[Dict]):
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
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        self.chunks = chunks
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search and return (chunk_index, score) pairs"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

def reciprocal_rank_fusion(semantic_results: List[Tuple[int, float]], 
                          bm25_results: List[Tuple[int, float]], 
                          semantic_weight: float = 0.7, 
                          bm25_weight: float = 0.3) -> List[Tuple[int, float]]:
    """Combine semantic and BM25 results using reciprocal rank fusion"""
    
    chunk_scores = {}
    
    # Add semantic scores (RRF)
    for rank, (chunk_idx, score) in enumerate(semantic_results):
        rrf_score = semantic_weight / (60 + rank + 1)
        chunk_scores[chunk_idx] = chunk_scores.get(chunk_idx, 0) + rrf_score
    
    # Add BM25 scores (RRF)
    for rank, (chunk_idx, score) in enumerate(bm25_results):
        rrf_score = bm25_weight / (60 + rank + 1)
        chunk_scores[chunk_idx] = chunk_scores.get(chunk_idx, 0) + rrf_score
    
    # Sort by combined score
    sorted_results = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

class OptimizedMultiSourceRetriever:
    """Optimized multi-source retriever with persistent models and caching"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', embeddings_path: str = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings_path = Path(embeddings_path) if embeddings_path else Path("../RAGdatav4/indiv_embeddings")
        self.data_path = self.embeddings_path.parent  # RAGdatav4 directory
        
        # Caches to avoid reloading
        self.source_cache = {}  # chunking_method -> {source -> {index, chunks, bm25}}
        self.loaded_chunking_methods = set()
        
        logger.info(f"Initialized OptimizedMultiSourceRetriever with model: {model_name}")
        logger.info(f"Data path: {self.data_path}, Embeddings path: {self.embeddings_path}")
    
    def load_chunking_method(self, chunking_method: str, sources: List[str] = None):
        """Load all sources for a specific chunking method (cached)"""
        if chunking_method in self.loaded_chunking_methods:
            logger.info(f"Chunking method {chunking_method} already loaded (cached)")
            return
        
        if sources is None:
            sources = ['healthify', 'mayo', 'nhs']
        
        logger.info(f"Loading chunking method: {chunking_method}")
        self.source_cache[chunking_method] = {}
        
        for source in sources:
            # Load chunks (JSON format in parent directory)
            chunk_file = self.data_path / f"{source}_chunks_{chunking_method}.json"
            if not chunk_file.exists():
                logger.warning(f"Chunk file not found: {chunk_file}")
                continue
            
            with open(chunk_file, 'r') as f:
                chunks = json.load(f)
            
            # Load FAISS index
            index_file = self.embeddings_path / f"{source}_vector_db_{chunking_method}.index"
            if not index_file.exists():
                logger.warning(f"Index file not found: {index_file}")
                continue
            
            index = faiss.read_index(str(index_file))
            
            # Build BM25 index
            bm25 = HybridBM25(chunks)
            
            self.source_cache[chunking_method][source] = {
                'index': index,
                'chunks': chunks,
                'bm25': bm25
            }
            
            logger.info(f"Loaded {source} with {len(chunks)} chunks for {chunking_method}")
        
        self.loaded_chunking_methods.add(chunking_method)
    
    def batch_encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode multiple queries at once for efficiency"""
        return self.model.encode(queries, batch_size=32, show_progress_bar=False).astype('float32')
    
    def search_batch_with_bias(self, queries: List[str], bias_config: BiasConfig, 
                              chunking_method: str, retrieval_type: str = 'pure_rag') -> List[List[Dict]]:
        """Search multiple queries at once with bias configuration"""
        
        # Ensure chunking method is loaded
        self.load_chunking_method(chunking_method)
        
        if chunking_method not in self.source_cache:
            logger.error(f"Chunking method {chunking_method} not available")
            return [[] for _ in queries]
        
        # Encode all queries at once
        query_embeddings = self.batch_encode_queries(queries)
        
        # Process each query
        all_results = []
        for i, query in enumerate(queries):
            query_embedding = query_embeddings[i:i+1]
            
            if retrieval_type == 'contextual_rag':
                results = self._contextual_search_with_bias(query, query_embedding, bias_config, chunking_method)
            else:
                results = self._pure_rag_search_with_bias(query, query_embedding, bias_config, chunking_method)
            
            all_results.append(results)
        
        return all_results
    
    def _pure_rag_search_with_bias(self, query: str, query_embedding: np.ndarray, 
                                  bias_config: BiasConfig, chunking_method: str) -> List[Dict]:
        """Pure RAG search (semantic only)"""
        all_results = []
        
        source_mapping = {
            'healthify': bias_config.healthify,
            'mayo': bias_config.mayo,
            'nhs': bias_config.nhs
        }
        
        source_data = self.source_cache[chunking_method]
        
        for source_name, num_chunks in source_mapping.items():
            if source_name not in source_data:
                logger.warning(f"Source {source_name} not available")
                continue
            
            if num_chunks <= 0:
                # Skip sources with 0 chunks (e.g., healthify_mayo bias excludes NHS)
                continue
            
            index = source_data[source_name]['index']
            chunks = source_data[source_name]['chunks']
            
            # Perform semantic search
            distances, indices = index.search(query_embedding, num_chunks)
            
            # Collect results
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx < len(chunks):
                    chunk = chunks[idx].copy()
                    chunk['retrieval_score'] = float(1.0 / (1.0 + dist))
                    chunk['retrieval_rank'] = i + 1
                    chunk['retrieval_source'] = source_name
                    chunk['retrieval_type'] = 'pure_rag'
                    all_results.append(chunk)
        
        return all_results
    
    def _contextual_search_with_bias(self, query: str, query_embedding: np.ndarray, 
                                   bias_config: BiasConfig, chunking_method: str) -> List[Dict]:
        """Contextual RAG search (semantic + BM25 + fusion)"""
        all_results = []
        
        source_mapping = {
            'healthify': bias_config.healthify,
            'mayo': bias_config.mayo,
            'nhs': bias_config.nhs
        }
        
        source_data = self.source_cache[chunking_method]
        
        for source_name, num_chunks in source_mapping.items():
            if source_name not in source_data:
                continue
            
            if num_chunks <= 0:
                # Skip sources with 0 chunks (e.g., healthify_mayo bias excludes NHS)
                continue
            
            index = source_data[source_name]['index']
            chunks = source_data[source_name]['chunks']
            bm25 = source_data[source_name]['bm25']
            
            # Retrieve more candidates for fusion
            retrieval_k = min(num_chunks * 3, len(chunks))
            
            # Semantic search
            distances, indices = index.search(query_embedding, retrieval_k)
            semantic_results = [(int(idx), float(1.0 / (1.0 + dist))) 
                              for dist, idx in zip(distances[0], indices[0]) if idx >= 0]
            
            # BM25 search
            bm25_results = bm25.search(query, top_k=retrieval_k)
            
            # Rank fusion
            fused_results = reciprocal_rank_fusion(semantic_results, bm25_results)
            
            # Take top num_chunks after fusion
            for i, (chunk_idx, score) in enumerate(fused_results[:num_chunks]):
                if chunk_idx < len(chunks):
                    chunk = chunks[chunk_idx].copy()
                    chunk['retrieval_score'] = float(score)
                    chunk['retrieval_rank'] = i + 1
                    chunk['retrieval_source'] = source_name
                    chunk['retrieval_type'] = 'contextual_rag'
                    all_results.append(chunk)
        
        return all_results

class MemoryMonitor:
    """Memory monitoring utility"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.start_memory = 0
        
    def start_monitoring(self):
        """Start memory monitoring"""
        tracemalloc.start()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
    def update_peak(self):
        """Update peak memory usage"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        current_trace, peak_trace = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'start_memory_mb': self.start_memory,
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'peak_traced_mb': peak_trace / 1024 / 1024
        }

class OptimizedHybridTester:
    """Optimized hybrid retrieval performance tester"""
    
    def __init__(self, embeddings_path: str = "../RAGdatav4/indiv_embeddings", 
                 batch_size: int = 32):
        self.embeddings_path = Path(embeddings_path)
        self.data_path = self.embeddings_path.parent  # RAGdatav4 directory
        self.batch_size = batch_size
        
        # Initialize persistent retriever
        self.retriever = OptimizedMultiSourceRetriever(embeddings_path=str(self.embeddings_path))
        self.memory_monitor = MemoryMonitor()
        
        logger.info(f"Initialized OptimizedHybridTester with batch_size={batch_size}")
    
    def generate_bias_configs(self) -> List[BiasConfig]:
        """Generate bias configurations"""
        configs = [
            BiasConfig("healthify_focused", 6, 2, 2, "Focus on Healthify content"),
            BiasConfig("diverse", 4, 3, 3, "slight preference for healthify")
        ]
        return configs
    
    def discover_chunking_methods(self) -> List[str]:
        """Discover available chunking methods"""
        methods = set()
        
        # Look for chunk files in data directory (JSON format)
        for file_path in self.data_path.glob("*_chunks_*.json"):
            # Extract chunking method from filename
            filename = file_path.stem
            if "_chunks_" in filename:
                method = filename.split("_chunks_", 1)[1]
                methods.add(method)
        
        return sorted(list(methods))
    
    def generate_hybrid_configs(self, max_configs: Optional[int] = None, 
                               chunking_filter: Optional[str] = None,
                               bias_filter: Optional[str] = None) -> List[RetrievalConfig]:
        """Generate hybrid configurations"""
        
        # Get chunking methods
        chunking_methods = self.discover_chunking_methods()
        if chunking_filter:
            if chunking_filter == "contextual":
                chunking_methods = [m for m in chunking_methods if "contextual" in m.lower()]
            else:
                chunking_methods = [m for m in chunking_methods if chunking_filter in m]
        
        # Get bias configs
        bias_configs = self.generate_bias_configs()
        if bias_filter:
            bias_configs = [c for c in bias_configs if bias_filter in c.name]
        
        # Generate all combinations
        configs = []
        retrieval_types = ['pure_rag', 'contextual_rag']
        
        for chunking in chunking_methods:
            for bias in bias_configs:
                for ret_type in retrieval_types:
                    config_name = f"{chunking}_{ret_type}_{bias.name}"
                    configs.append(RetrievalConfig(
                        name=config_name,
                        chunking_method=chunking,
                        retrieval_type=ret_type,
                        bias_config=bias
                    ))
        
        if max_configs and len(configs) > max_configs:
            configs = configs[:max_configs]
        
        logger.info(f"Generated {len(configs)} hybrid configurations")
        return configs
    
    def evaluate_configuration_optimized(self, config: RetrievalConfig, test_cases: List[TestCase],
                                       k_values: List[int] = [1, 2, 3, 4, 5, 10, 20]) -> TestResult:
        """Optimized evaluation of a single configuration"""
        
        logger.info(f"Evaluating configuration: {config.name} with {len(test_cases)} test cases")
        
        self.memory_monitor.start_monitoring()
        
        # Pre-load chunking method
        self.retriever.load_chunking_method(config.chunking_method)
        self.memory_monitor.update_peak()
        
        # Prepare batch processing
        batch_results = []
        total_time = 0
        
        # Process in batches
        for i in range(0, len(test_cases), self.batch_size):
            batch = test_cases[i:i + self.batch_size]
            batch_queries = [tc.get_combined_query() for tc in batch]
            
            # Time batch processing
            start_time = time.time()
            
            # Search batch
            search_results = self.retriever.search_batch_with_bias(
                batch_queries, config.bias_config, config.chunking_method, config.retrieval_type
            )
            
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # Process results
            for j, (test_case, results) in enumerate(zip(batch, search_results)):
                expected_doc = test_case.get_expected_source_doc()
                keywords = test_case.get_keywords_for_matching()
                
                # Calculate metrics
                pass_results = {k: False for k in k_values}
                
                # Debug: Print first test case details
                if i * self.batch_size + j < 3:  # Only first 3 test cases
                    logger.info(f"Debug Test Case {i * self.batch_size + j}:")
                    logger.info(f"  Symptom: '{test_case.symptom}'")
                    logger.info(f"  Expected doc: '{expected_doc}'")
                    logger.info(f"  Keywords: {keywords}")
                    logger.info(f"  Top 3 retrieved docs: {[r.get('source_document', 'NO_SOURCE') for r in results[:3]]}")
                
                for k in k_values:
                    top_k_results = results[:k]
                    for result in top_k_results:
                        source_doc = result.get('source_document', '')
                        source_clean = source_doc.replace('.txt', '').replace('_', ' ').lower()
                        
                        # Method 1: Exact match (original logic)
                        expected_clean = expected_doc.replace('.txt', '').replace('_', ' ').lower()
                        if expected_clean in source_clean:
                            pass_results[k] = True
                            break
                        
                        # Method 2: Keyword-based matching (more flexible)
                        # Check if at least 2 keywords match (or all keywords if fewer than 2)
                        matching_keywords = sum(1 for keyword in keywords if keyword in source_clean)
                        required_matches = min(2, len(keywords))
                        
                        if matching_keywords >= required_matches and required_matches > 0:
                            pass_results[k] = True
                            if i * self.batch_size + j < 3:
                                logger.info(f"  ✓ Match found: {source_doc} (keywords: {matching_keywords}/{len(keywords)})")
                            break
                
                # Track source distribution
                source_dist = {'healthify': 0, 'mayo': 0, 'nhs': 0}
                for result in results:
                    source = result.get('retrieval_source', 'unknown')
                    if source in source_dist:
                        source_dist[source] += 1
                
                batch_results.append({
                    'test_case_id': getattr(test_case, 'id', i * self.batch_size + j),
                    'pass_at_k': pass_results,
                    'source_distribution': source_dist,
                    'num_results': len(results),
                    'expected_doc': expected_doc
                })
            
            self.memory_monitor.update_peak()
        
        # Calculate final metrics
        total_cases = len(test_cases)
        pass_counts = {k: 0 for k in k_values}
        total_source_dist = {'healthify': 0, 'mayo': 0, 'nhs': 0}
        
        for result in batch_results:
            for k in k_values:
                if result['pass_at_k'][k]:
                    pass_counts[k] += 1
            
            for source, count in result['source_distribution'].items():
                total_source_dist[source] += count
        
        # Calculate pass rates
        pass_rates = {k: pass_counts[k] / total_cases if total_cases > 0 else 0 for k in k_values}
        
        memory_stats = self.memory_monitor.get_stats()
        avg_time = total_time / total_cases if total_cases > 0 else 0
        
        return TestResult(
            config=config,
            pass_at_1=pass_rates[1],
            pass_at_2=pass_rates[2],
            pass_at_3=pass_rates[3],
            pass_at_4=pass_rates[4],
            pass_at_5=pass_rates[5],
            pass_at_10=pass_rates[10],
            pass_at_20=pass_rates[20],
            avg_retrieval_time=avg_time,
            total_test_cases=total_cases,
            successful_retrievals=total_cases,
            detailed_results=batch_results,
            source_distribution=total_source_dist,
            memory_stats=memory_stats
        )
    
    def load_test_data(self, test_data_file: str) -> List[TestCase]:
        """Load test cases from JSON file"""
        if not os.path.exists(test_data_file):
            logger.error(f"Test data file not found: {test_data_file}")
            return []
        
        test_cases = []
        with open(test_data_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Take only first 200 test cases for faster evaluation
            data = data[:200]
            logger.info(f"Using first 200 test cases from {len(data)} total")
            
            for item in data:
                try:
                    test_case = TestCase(
                        id=item.get('id'),
                        symptom=item['symptom'],
                        query=item['query'],
                        final_triage_decision=item['final_triage_decision'],
                        next_step=item['next_step'],
                        reasoning=item['reasoning']
                    )
                    test_cases.append(test_case)
                except Exception as e:
                    logger.warning(f"Failed to parse test case: {e}")
                    continue
        
        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases
    
    def run_comprehensive_test(self, test_data_file: str, results_dir: str, 
                             max_configs: Optional[int] = None,
                             chunking_filter: Optional[str] = None,
                             bias_filter: Optional[str] = None) -> List[TestResult]:
        """Run comprehensive hybrid retrieval test"""
        
        # Load test data
        test_cases = self.load_test_data(test_data_file)
        if not test_cases:
            return []
        
        # Generate configurations
        configs = self.generate_hybrid_configs(max_configs, chunking_filter, bias_filter)
        
        logger.info(f"Testing {len(configs)} hybrid configurations with {len(test_cases)} test cases")
        
        # Create results directory
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamp once for consistent filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Run evaluations
        all_results = []
        for config in tqdm(configs, desc="Evaluating Configurations"):
            result = self.evaluate_configuration_optimized(config, test_cases)
            all_results.append(result)

            # Save intermediate results with consistent timestamp
            self.save_results(all_results, results_path, timestamp)

        return all_results
    
    def save_results(self, results: List[TestResult], results_dir: Path, timestamp: str):
        """Save results to files"""
        
        # Save summary
        summary_data = []
        for result in results:
            summary_data.append({
                'config_name': result.config.name,
                'chunking_method': result.config.chunking_method,
                'retrieval_type': result.config.retrieval_type,
                'bias_config': result.config.bias_config.name,
                'pass_at_1': result.pass_at_1,
                'pass_at_2': result.pass_at_2,
                'pass_at_3': result.pass_at_3,
                'pass_at_4': result.pass_at_4,
                'pass_at_5': result.pass_at_5,
                'pass_at_10': result.pass_at_10,
                'pass_at_20': result.pass_at_20,
                'avg_retrieval_time': result.avg_retrieval_time,
                'total_test_cases': result.total_test_cases,
                'memory_stats': result.memory_stats,
                'source_distribution': result.source_distribution
            })
        
        summary_file = results_dir / f"optimized_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Results saved to {summary_file}")

def main():
    # Default settings for full comprehensive test  
    script_dir = Path(__file__).parent
    test_data_file = script_dir / "../Final_dataset/simplified_triage_dialogues_val.json"
    results_dir = script_dir / "results/optimized_comprehensive"
    batch_size = 64
    embeddings_path = script_dir / "../RAGdatav4/indiv_embeddings"
    
    print("="*80)
    print("🚀 OPTIMIZED HYBRID RETRIEVAL - FULL COMPREHENSIVE TEST")
    print("="*80)
    print(f"📊 Test Data: {test_data_file}")
    print(f"📁 Results Dir: {results_dir}")
    print(f"⚡ Batch Size: {batch_size}")
    print(f"🧠 All chunking methods, bias configs, and retrieval types")
    print(f"📈 Estimated ~432 configurations")
    print(f"⏰ Estimated time: 2-4 hours")
    print("="*80)
    
    confirm = input("Start full comprehensive test? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Test cancelled.")
        return
    
    logger.info("Starting optimized hybrid retrieval performance testing...")
    logger.info("Optimizations: Persistent models, batch processing, memory caching")
    logger.info("Running FULL COMPREHENSIVE TEST with all configurations")
    
    tester = OptimizedHybridTester(
        embeddings_path=str(embeddings_path),
        batch_size=batch_size
    )
    
    results = tester.run_comprehensive_test(
        test_data_file=str(test_data_file),
        results_dir=str(results_dir),
        max_configs=None,  # No limit - test all configs
        chunking_filter=None,
        bias_filter=None
    )
    
    if results:
        logger.info(f"Testing completed! {len(results)} configurations evaluated.")
        logger.info(f"Results saved to: {results_dir}")
        
        # Print top performers
        sorted_results = sorted(results, key=lambda x: x.pass_at_5, reverse=True)
        logger.info("Top 3 configurations by Pass@5:")
        for i, result in enumerate(sorted_results[:3]):
            logger.info(f"{i+1}. {result.config.name}: {result.pass_at_5:.3f}")
        
        print("\n" + "="*80)
        print("✅ FULL COMPREHENSIVE TEST COMPLETED!")
        print("="*80)
        print(f"📈 {len(results)} configurations tested")
        print(f"📁 Results saved to: {results_dir}")
        print("🏆 Top 3 performers:")
        for i, result in enumerate(sorted_results[:3]):
            print(f"   {i+1}. {result.config.name}: {result.pass_at_5:.3f}")
    else:
        logger.error("No results generated.")

if __name__ == "__main__":
    main()