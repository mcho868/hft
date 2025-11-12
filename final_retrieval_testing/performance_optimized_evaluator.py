#!/usr/bin/env python3
"""
Performance-Optimized Hybrid Retrieval Evaluator

Key optimizations:
1. Lazy loading - only load needed configurations
2. Memory management - cleanup after each config
3. True parallel processing for batch queries
4. Cached BM25 indices
5. Streaming evaluation to reduce memory
"""

import os
import json
import gc
import numpy as np
import faiss
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
import re
from datetime import datetime
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import lru_cache

# Import existing classes
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

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
        keywords = clean_symptom.lower().replace('_', ' ').split()
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

@dataclass
class TestResult:
    """Results from testing a specific configuration"""
    config: RetrievalConfig
    pass_at_5: float
    pass_at_10: float
    pass_at_20: float
    avg_retrieval_time: float
    total_test_cases: int
    successful_retrievals: int
    memory_peak_mb: float

class PerformanceOptimizedRetriever:
    """Memory-efficient retriever with lazy loading"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', embeddings_path: str = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings_path = Path(embeddings_path) if embeddings_path else Path("../RAGdatav4/indiv_embeddings")
        self.data_path = self.embeddings_path.parent
        
        # Only cache BM25 preprocessing, not full data
        self.bm25_cache = {}  # chunking_method -> {source -> tokenized_texts}
        self.current_config = None
        self.current_data = {}  # Only hold current configuration data
        
        logger.info(f"Initialized PerformanceOptimizedRetriever with model: {model_name}")
    
    def cleanup_memory(self):
        """Aggressive memory cleanup"""
        if self.current_data:
            # Clear FAISS indices
            for source_data in self.current_data.values():
                if 'index' in source_data:
                    del source_data['index']
                if 'chunks' in source_data:
                    del source_data['chunks']
                if 'bm25' in source_data:
                    del source_data['bm25']
            
            self.current_data.clear()
            
        # Force garbage collection
        gc.collect()
        
        # Log memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory after cleanup: {memory_mb:.1f} MB")
    
    @lru_cache(maxsize=100)
    def get_bm25_tokenized_texts(self, chunking_method: str, source: str) -> tuple:
        """Cache BM25 tokenized texts to avoid reprocessing"""
        chunk_file = self.data_path / f"{source}_chunks_{chunking_method}.json"
        if not chunk_file.exists():
            return tuple()
        
        with open(chunk_file, 'r') as f:
            chunks = json.load(f)
        
        texts = []
        for chunk in chunks:
            if chunk.get('is_contextual') and chunk.get('contextual_info'):
                text = f"{chunk['text']} {chunk['contextual_info']}"
            else:
                text = chunk['text']
            texts.append(text)
        
        # Return tuple for caching (lists aren't hashable)
        tokenized = tuple(text.lower().split() for text in texts)
        return tokenized
    
    def load_single_configuration(self, chunking_method: str, sources: List[str] = None):
        """Load only one configuration at a time"""
        # Clean up previous configuration
        self.cleanup_memory()
        
        if sources is None:
            sources = ['healthify', 'mayo', 'nhs']
        
        logger.info(f"Loading configuration: {chunking_method}")
        self.current_data = {}
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        for source in sources:
            # Load chunks
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
            
            # Build BM25 efficiently using cached tokenization
            tokenized_texts = self.get_bm25_tokenized_texts(chunking_method, source)
            if tokenized_texts:
                bm25 = BM25Okapi(list(tokenized_texts))
            else:
                continue
            
            self.current_data[source] = {
                'index': index,
                'chunks': chunks,
                'bm25': bm25
            }
            
            logger.info(f"Loaded {source}: {len(chunks)} chunks")
        
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage: {start_memory:.1f} -> {end_memory:.1f} MB (+{end_memory-start_memory:.1f} MB)")
        
        self.current_config = chunking_method
        return len(self.current_data) > 0
    
    def batch_search_parallel(self, queries: List[str], bias_config: BiasConfig, 
                             retrieval_type: str = 'pure_rag', max_workers: int = 4) -> List[List[Dict]]:
        """Parallel processing for batch queries"""
        
        if not self.current_data:
            return [[] for _ in queries]
        
        # Encode all queries at once (most efficient)
        query_embeddings = self.model.encode(queries, batch_size=32, show_progress_bar=False).astype('float32')
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, query in enumerate(queries):
                future = executor.submit(
                    self._single_search,
                    query, 
                    query_embeddings[i:i+1], 
                    bias_config, 
                    retrieval_type
                )
                futures.append(future)
            
            # Collect results in order
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout per query
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Query failed: {e}")
                    results.append([])
        
        return results
    
    def _single_search(self, query: str, query_embedding: np.ndarray, 
                      bias_config: BiasConfig, retrieval_type: str) -> List[Dict]:
        """Single query search - designed for parallel execution"""
        
        all_results = []
        source_mapping = {
            'healthify': bias_config.healthify,
            'mayo': bias_config.mayo,
            'nhs': bias_config.nhs
        }
        
        for source_name, num_chunks in source_mapping.items():
            if source_name not in self.current_data or num_chunks <= 0:
                continue
            
            source_data = self.current_data[source_name]
            index = source_data['index']
            chunks = source_data['chunks']
            
            if retrieval_type == 'contextual_rag':
                # Contextual RAG with fusion
                bm25 = source_data['bm25']
                retrieval_k = min(num_chunks * 3, len(chunks))
                
                # Semantic search
                distances, indices = index.search(query_embedding, retrieval_k)
                semantic_results = [(int(idx), float(1.0 / (1.0 + dist))) 
                                  for dist, idx in zip(distances[0], indices[0]) if idx >= 0]
                
                # BM25 search
                tokenized_query = query.lower().split()
                scores = bm25.get_scores(tokenized_query)
                top_indices = np.argsort(scores)[-retrieval_k:][::-1]
                bm25_results = [(int(idx), float(scores[idx])) for idx in top_indices]
                
                # Simple rank fusion
                chunk_scores = {}
                for rank, (chunk_idx, _) in enumerate(semantic_results):
                    chunk_scores[chunk_idx] = chunk_scores.get(chunk_idx, 0) + 0.7 / (60 + rank + 1)
                for rank, (chunk_idx, _) in enumerate(bm25_results):
                    chunk_scores[chunk_idx] = chunk_scores.get(chunk_idx, 0) + 0.3 / (60 + rank + 1)
                
                sorted_results = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
                final_indices = [idx for idx, _ in sorted_results[:num_chunks]]
                
            else:
                # Pure RAG (semantic only)
                distances, indices = index.search(query_embedding, num_chunks)
                final_indices = [int(idx) for idx in indices[0] if idx >= 0]
            
            # Collect final results
            for i, chunk_idx in enumerate(final_indices):
                if chunk_idx < len(chunks):
                    chunk = chunks[chunk_idx].copy()
                    chunk['retrieval_score'] = float(1.0 / (1.0 + i))  # Simple scoring
                    chunk['retrieval_rank'] = i + 1
                    chunk['retrieval_source'] = source_name
                    chunk['retrieval_type'] = retrieval_type
                    all_results.append(chunk)
        
        return all_results

class StreamingEvaluator:
    """Memory-efficient evaluator with streaming results"""
    
    def __init__(self, embeddings_path: str = "../RAGdatav4/indiv_embeddings", 
                 batch_size: int = 32, max_workers: int = 4):
        self.embeddings_path = Path(embeddings_path)
        self.data_path = self.embeddings_path.parent
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        self.retriever = PerformanceOptimizedRetriever(embeddings_path=str(self.embeddings_path))
        
        logger.info(f"Initialized StreamingEvaluator with batch_size={batch_size}, workers={max_workers}")
    
    def generate_bias_configs(self) -> List[BiasConfig]:
        """Generate bias configurations"""
        return [
            BiasConfig("balanced", 4, 4, 4, "Equal bias across all sources"),
            BiasConfig("healthify_focused", 6, 2, 2, "Focus on Healthify content"),
            BiasConfig("diverse", 3, 3, 4, "Slight preference for NHS")
        ]
    
    def discover_chunking_methods(self) -> List[str]:
        """Discover available chunking methods"""
        methods = set()
        for file_path in self.data_path.glob("*_chunks_*.json"):
            filename = file_path.stem
            if "_chunks_" in filename:
                method = filename.split("_chunks_", 1)[1]
                methods.add(method)
        return sorted(list(methods))
    
    def generate_configs(self, max_configs: Optional[int] = None, 
                        chunking_filter: Optional[str] = None) -> List[RetrievalConfig]:
        """Generate configurations with filtering"""
        chunking_methods = self.discover_chunking_methods()
        if chunking_filter:
            chunking_methods = [m for m in chunking_methods if chunking_filter in m]
        
        bias_configs = self.generate_bias_configs()
        retrieval_types = ['pure_rag', 'contextual_rag']
        
        configs = []
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
        
        logger.info(f"Generated {len(configs)} configurations")
        return configs
    
    def evaluate_single_config(self, config: RetrievalConfig, test_cases: List[TestCase]) -> TestResult:
        """Evaluate single configuration with memory efficiency"""
        
        logger.info(f"Evaluating: {config.name}")
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Load configuration
        if not self.retriever.load_single_configuration(config.chunking_method):
            logger.error(f"Failed to load chunking method: {config.chunking_method}")
            return None
        
        # Process in batches with parallel processing
        all_results = []
        total_time = 0
        
        for i in range(0, len(test_cases), self.batch_size):
            batch = test_cases[i:i + self.batch_size]
            batch_queries = [tc.get_combined_query() for tc in batch]
            
            start_time = time.time()
            search_results = self.retriever.batch_search_parallel(
                batch_queries, config.bias_config, config.retrieval_type, self.max_workers
            )
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # Process results efficiently
            for j, (test_case, results) in enumerate(zip(batch, search_results)):
                keywords = test_case.get_keywords_for_matching()
                
                # Efficient matching
                pass_results = {5: False, 10: False, 20: False}
                for k in [5, 10, 20]:
                    for result in results[:k]:
                        source_doc = result.get('source_document', '').lower()
                        
                        # Keyword matching
                        matching_keywords = sum(1 for keyword in keywords if keyword in source_doc)
                        if matching_keywords >= min(2, len(keywords)) and len(keywords) > 0:
                            pass_results[k] = True
                            break
                
                all_results.append(pass_results)
        
        # Calculate metrics
        total_cases = len(test_cases)
        pass_counts = {k: sum(1 for r in all_results if r[k]) for k in [5, 10, 20]}
        pass_rates = {k: pass_counts[k] / total_cases if total_cases > 0 else 0 for k in [5, 10, 20]}
        
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
        avg_time = total_time / total_cases if total_cases > 0 else 0
        
        return TestResult(
            config=config,
            pass_at_5=pass_rates[5],
            pass_at_10=pass_rates[10],
            pass_at_20=pass_rates[20],
            avg_retrieval_time=avg_time,
            total_test_cases=total_cases,
            successful_retrievals=total_cases,
            memory_peak_mb=peak_memory - start_memory
        )
    
    def load_test_data(self, test_data_file: str, max_cases: int = 200) -> List[TestCase]:
        """Load test cases efficiently"""
        if not os.path.exists(test_data_file):
            logger.error(f"Test data file not found: {test_data_file}")
            return []
        
        test_cases = []
        with open(test_data_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            data = data[:max_cases]
            logger.info(f"Loading {len(data)} test cases")
            
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
        
        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases
    
    def run_evaluation(self, test_data_file: str, results_dir: str, 
                      max_configs: Optional[int] = None,
                      chunking_filter: Optional[str] = None) -> List[TestResult]:
        """Run evaluation with streaming results"""
        
        # Load test data
        test_cases = self.load_test_data(test_data_file)
        if not test_cases:
            return []
        
        # Generate configurations
        configs = self.generate_configs(max_configs, chunking_filter)
        
        logger.info(f"Evaluating {len(configs)} configurations with {len(test_cases)} test cases")
        
        # Create results directory
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Process configurations one by one
        all_results = []
        for i, config in enumerate(tqdm(configs, desc="Configurations")):
            try:
                result = self.evaluate_single_config(config, test_cases)
                if result:
                    all_results.append(result)
                    
                    # Save intermediate results
                    self.save_streaming_results(all_results, results_path, i)
                    
                    logger.info(f"✅ {config.name}: Pass@5={result.pass_at_5:.3f}, Memory={result.memory_peak_mb:.1f}MB")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {config.name}: {e}")
                continue
            
            # Force cleanup between configurations
            self.retriever.cleanup_memory()
        
        return all_results
    
    def save_streaming_results(self, results: List[TestResult], results_dir: Path, iteration: int):
        """Save results incrementally"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary_data = []
        for result in results:
            summary_data.append({
                'config_name': result.config.name,
                'chunking_method': result.config.chunking_method,
                'retrieval_type': result.config.retrieval_type,
                'bias_config': result.config.bias_config.name,
                'pass_at_5': result.pass_at_5,
                'pass_at_10': result.pass_at_10,
                'pass_at_20': result.pass_at_20,
                'avg_retrieval_time': result.avg_retrieval_time,
                'memory_peak_mb': result.memory_peak_mb,
                'total_test_cases': result.total_test_cases
            })
        
        summary_file = results_dir / f"streaming_results_{timestamp}_iter_{iteration}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

def main():
    """Main function for performance-optimized evaluation"""
    script_dir = Path(__file__).parent
    test_data_file = script_dir / "../Final_dataset/simplified_triage_dialogues_val.json"
    results_dir = script_dir / "results/performance_optimized"
    
    print("="*80)
    print("🚀 PERFORMANCE-OPTIMIZED HYBRID RETRIEVAL EVALUATOR")
    print("="*80)
    print("✅ Lazy loading - only one config in memory at a time")
    print("✅ Parallel query processing with ThreadPoolExecutor")
    print("✅ Aggressive memory management and cleanup")
    print("✅ Streaming results to disk")
    print("✅ Cached BM25 preprocessing")
    print("="*80)
    
    # Configuration
    max_configs = int(input("Max configurations to test (e.g., 20): ").strip() or "20")
    chunking_filter = input("Chunking filter (e.g., 'fixed_c512_o100', or press Enter): ").strip() or None
    batch_size = int(input("Batch size (default 16): ").strip() or "16")
    max_workers = int(input("Max workers (default 4): ").strip() or "4")
    
    evaluator = StreamingEvaluator(
        embeddings_path=str(script_dir / "../RAGdatav4/indiv_embeddings"),
        batch_size=batch_size,
        max_workers=max_workers
    )
    
    start_time = time.time()
    results = evaluator.run_evaluation(
        test_data_file=str(test_data_file),
        results_dir=str(results_dir),
        max_configs=max_configs,
        chunking_filter=chunking_filter
    )
    
    total_time = time.time() - start_time
    
    if results:
        logger.info(f"✅ Evaluation completed in {total_time:.1f} seconds!")
        logger.info(f"📊 {len(results)} configurations evaluated")
        
        # Show top performers
        sorted_results = sorted(results, key=lambda x: x.pass_at_5, reverse=True)
        print("\n🏆 TOP 5 PERFORMERS:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {result.config.name}: {result.pass_at_5:.3f} (Memory: {result.memory_peak_mb:.1f}MB)")
    else:
        logger.error("No results generated")

if __name__ == "__main__":
    main()