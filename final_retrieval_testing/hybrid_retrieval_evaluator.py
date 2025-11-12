#!/usr/bin/env python3
"""
Hybrid Retrieval Evaluator - Combines Multi-Source Bias Configuration with Pure RAG vs Contextual Retrieval

This script merges the best of both approaches:
1. Multi-source bias configurations (like retrieval_testv2.py)
2. Pure RAG vs Contextual retrieval evaluation (like final_retrieval_testing)
3. Comprehensive performance metrics and analysis

Usage:
    python hybrid_retrieval_evaluator.py --test-data eval/test_data.json --results-dir results/hybrid_test
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

# Import existing retrieval classes
from offline_contextual_retrieval import OfflineContextualRetrieval, OfflineVectorDB, BM25Search, OfflineReranker, HybridRetriever
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Test case from final_retrieval_testing format"""
    symptom: str
    patient_query: str
    patient_response: str
    final_triage_decision: str
    next_step: str
    reasoning_question: str
    reasoning_decision: str
    clarifying_question: str
    generation_timestamp: Optional[float] = None  # Optional for validation dataset
    id: Optional[int] = None  # For validation dataset compatibility
    
    def get_combined_query(self) -> str:
        """Combine patient query and response for retrieval"""
        return f"{self.patient_query} {self.patient_response}"
    
    def get_expected_source_doc(self) -> str:
        """Extract the expected source document name from symptom"""
        # Remove (Case N) suffix and normalize
        clean_symptom = re.sub(r'\s*\(Case\s+\d+\)$', '', self.symptom)
        return f"{clean_symptom}.txt"

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
    retrieval_type: str  # 'pure_rag', 'contextual_rag'
    bias_config: BiasConfig
    embedding_file: Optional[str] = None
    index_files: Optional[Dict[str, str]] = None  # source -> index_file mapping
    chunk_files: Optional[Dict[str, str]] = None  # source -> chunk_file mapping

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
    detailed_results: List[Dict[str, Any]]
    source_distribution: Dict[str, int]  # Actual source distribution in results
    memory_stats: Dict[str, float]  # Memory usage statistics

class HybridBM25:
    """BM25 implementation for contextual retrieval"""
    
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.corpus = []
        self.bm25 = None
        self._build_index()
    
    def _build_index(self):
        """Build BM25 index from chunks"""
        for chunk in self.chunks:
            # Use contextual text if available, otherwise original text
            if chunk.get('is_contextual') and chunk.get('contextual_info'):
                text = f"{chunk.get('text', '')} {chunk.get('contextual_info', '')}"
            else:
                text = chunk.get('text', chunk.get('original_text', ''))
            
            # Simple tokenization
            tokens = text.lower().split()
            self.corpus.append(tokens)
        
        self.bm25 = BM25Okapi(self.corpus)
    
    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Search and return (index, score) tuples"""
        if not self.bm25:
            return []
        
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices with scores
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

def reciprocal_rank_fusion(semantic_results: List[Tuple[int, float]], 
                          bm25_results: List[Tuple[int, float]], 
                          semantic_weight: float = 0.7, 
                          bm25_weight: float = 0.3, 
                          k: int = 60) -> List[Tuple[int, float]]:
    """Implement reciprocal rank fusion as in the Anthropic notebook"""
    chunk_scores = {}
    
    # Add semantic search scores
    for rank, (idx, score) in enumerate(semantic_results):
        rrf_score = semantic_weight / (k + rank + 1)
        chunk_scores[idx] = chunk_scores.get(idx, 0) + rrf_score
    
    # Add BM25 scores  
    for rank, (idx, score) in enumerate(bm25_results):
        rrf_score = bm25_weight / (k + rank + 1)
        chunk_scores[idx] = chunk_scores.get(idx, 0) + rrf_score
    
    # Sort by combined score
    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_chunks

class MultiSourceRetriever:
    """Multi-source retriever supporting bias configurations and contextual retrieval"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.source_retrievers = {}  # source_name -> retriever
        self.source_bm25 = {}  # source_name -> BM25 index
        
    def add_source(self, source_name: str, index_path: str, chunks: List[Dict]):
        """Add a source retriever with BM25 support"""
        self.source_retrievers[source_name] = {
            'index': faiss.read_index(index_path),
            'chunks': chunks
        }
        # Build BM25 index for this source
        self.source_bm25[source_name] = HybridBM25(chunks)
        logger.info(f"Added source {source_name} with {len(chunks)} chunks and BM25 index")
    
    def search_with_bias(self, query: str, bias_config: BiasConfig, retrieval_type: str = 'pure_rag') -> List[Dict]:
        """Search with bias configuration across sources"""
        if retrieval_type == 'contextual_rag':
            return self._contextual_search_with_bias(query, bias_config)
        else:
            return self._pure_rag_search_with_bias(query, bias_config)
    
    def _pure_rag_search_with_bias(self, query: str, bias_config: BiasConfig) -> List[Dict]:
        """Pure RAG search (semantic only)"""
        all_results = []
        
        # Encode query once
        query_embedding = self.model.encode([query]).astype('float32')
        
        source_mapping = {
            'healthify': bias_config.healthify,
            'mayo': bias_config.mayo,
            'nhs': bias_config.nhs
        }
        
        # Retrieve from each source according to bias config
        for source_name, num_chunks in source_mapping.items():
            if source_name not in self.source_retrievers:
                logger.warning(f"Source {source_name} not available")
                continue
                
            source_data = self.source_retrievers[source_name]
            index = source_data['index']
            chunks = source_data['chunks']
            
            # Perform semantic search
            distances, indices = index.search(query_embedding, num_chunks)
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1 and idx < len(chunks):
                    chunk = chunks[idx]
                    
                    result = {
                        'chunk': chunk,
                        'source': source_name,
                        'distance': float(distance),
                        'similarity_score': float(1.0 / (1.0 + distance)),
                        'rank_in_source': i,
                        'retrieval_type': 'pure_rag'
                    }
                    all_results.append(result)
        
        # Sort by similarity (lower distance = higher similarity)
        all_results.sort(key=lambda x: x['distance'])
        return all_results
    
    def _contextual_search_with_bias(self, query: str, bias_config: BiasConfig) -> List[Dict]:
        """Contextual RAG search with semantic + BM25 fusion"""
        all_results = []
        
        # Encode query once
        query_embedding = self.model.encode([query]).astype('float32')
        
        source_mapping = {
            'healthify': bias_config.healthify,
            'mayo': bias_config.mayo,
            'nhs': bias_config.nhs
        }
        
        # Retrieve from each source with hybrid approach
        for source_name, num_chunks in source_mapping.items():
            if source_name not in self.source_retrievers:
                logger.warning(f"Source {source_name} not available")
                continue
                
            source_data = self.source_retrievers[source_name]
            index = source_data['index']
            chunks = source_data['chunks']
            bm25_index = self.source_bm25[source_name]
            
            # Get more results for fusion (2x the target)
            retrieval_k = min(num_chunks * 2, len(chunks))
            
            # Semantic search
            distances, indices = index.search(query_embedding, retrieval_k)
            semantic_results = [(int(idx), float(1.0 / (1.0 + dist))) 
                              for dist, idx in zip(distances[0], indices[0]) 
                              if idx != -1 and idx < len(chunks)]
            
            # BM25 search  
            bm25_results = bm25_index.search(query, top_k=retrieval_k)
            
            # Rank fusion
            fused_results = reciprocal_rank_fusion(semantic_results, bm25_results)
            
            # Take top num_chunks after fusion
            for rank, (idx, score) in enumerate(fused_results[:num_chunks]):
                if idx < len(chunks):
                    chunk = chunks[idx]
                    
                    result = {
                        'chunk': chunk,
                        'source': source_name,
                        'distance': float(1.0 - score),  # Convert score back to distance-like
                        'similarity_score': float(score),
                        'rank_in_source': rank,
                        'retrieval_type': 'contextual_rag',
                        'fusion_score': float(score)
                    }
                    all_results.append(result)
        
        # Sort by fusion score (higher is better)
        all_results.sort(key=lambda x: x.get('fusion_score', x['similarity_score']), reverse=True)
        return all_results
    

class HybridChunkLoader:
    """Loads chunks for multi-source hybrid approach"""
    
    def __init__(self, ragdata_path: str = "RAGdatav4"):
        parent_dir = Path(__file__).parent.parent
        self.ragdata_path = parent_dir / ragdata_path
        self.chunks_cache = {}
    
    def load_source_chunks(self, source: str, chunking_method: str) -> List[Dict[str, Any]]:
        """Load chunks for a specific source"""
        cache_key = f"{source}_{chunking_method}"
        if cache_key in self.chunks_cache:
            return self.chunks_cache[cache_key]
        
        chunk_file = f"{source}_chunks_{chunking_method}.json"
        file_path = self.ragdata_path / chunk_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Standardize chunk format
        standardized_chunks = []
        for chunk in chunks:
            standardized_chunk = {
                'chunk_id': chunk.get('chunk_id', ''),
                'text': chunk.get('text', ''),
                'original_text': chunk.get('original_text', chunk.get('text', '')),
                'contextual_info': chunk.get('contextual_info', ''),
                'is_contextual': chunk.get('is_contextual', False),
                'source_document': chunk.get('source_document', ''),
                'source_char_start': chunk.get('source_char_start', 0),
                'source_char_end': chunk.get('source_char_end', 0),
                'metadata': chunk
            }
            standardized_chunks.append(standardized_chunk)
        
        self.chunks_cache[cache_key] = standardized_chunks
        logger.info(f"Loaded {len(standardized_chunks)} chunks for {source} with {chunking_method}")
        return standardized_chunks

class HybridIndexLoader:
    """Loads FAISS indices for multi-source approach"""
    
    def __init__(self, embeddings_path: str = "RAGdatav4/indiv_embeddings"):
        parent_dir = Path(__file__).parent.parent
        self.embeddings_path = parent_dir / embeddings_path
        self.indices_cache = {}
    
    def load_source_index(self, source: str, chunking_method: str) -> faiss.Index:
        """Load FAISS index for a specific source"""
        cache_key = f"{source}_{chunking_method}"
        if cache_key in self.indices_cache:
            return self.indices_cache[cache_key]
        
        index_file = f"{source}_vector_db_{chunking_method}.index"
        file_path = self.embeddings_path / index_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Index file not found: {file_path}")
        
        index = faiss.read_index(str(file_path))
        self.indices_cache[cache_key] = index
        logger.info(f"Loaded FAISS index for {source} with {chunking_method}")
        return index

class ConfigurationScanner:
    """Scans for available chunking methods and creates hybrid configurations"""
    
    def __init__(self, ragdata_path: str = "RAGdatav4", embeddings_path: str = "RAGdatav4/indiv_embeddings"):
        parent_dir = Path(__file__).parent.parent
        self.ragdata_path = parent_dir / ragdata_path
        self.embeddings_path = parent_dir / embeddings_path
    
    def scan_available_configurations(self) -> List[RetrievalConfig]:
        """Scan for available chunking methods and create configurations"""
        configs = []
        
        # Define bias configurations
        bias_configs = [
            BiasConfig("balanced", 4, 4, 4, "Equal representation from all sources"),
            BiasConfig("healthify_bias_1", 6, 2, 2, "Strong Healthify bias"),
            BiasConfig("healthify_bias_2", 5, 3, 2, "Moderate Healthify bias"),
            BiasConfig("mayo_bias", 2, 6, 2, "Strong Mayo Clinic bias"),
            BiasConfig("nhs_bias", 2, 2, 6, "Strong NHS bias"),
            BiasConfig("medical_authority", 3, 4, 3, "Mayo-focused medical authority")
        ]
        
        # Define retrieval types
        retrieval_types = [
            ("pure_rag", "Pure RAG using original text only"),
            ("contextual_rag", "Contextual RAG using contextual information when available")
        ]
        
        # Find available chunking methods
        chunking_methods = self._discover_chunking_methods()
        
        # Create configurations for each combination
        for chunking_method in chunking_methods:
            for bias_config in bias_configs:
                for retrieval_type, description in retrieval_types:
                    # Check if all required files exist
                    if self._validate_chunking_method(chunking_method):
                        config_name = f"{chunking_method}_{retrieval_type}_{bias_config.name}"
                        
                        config = RetrievalConfig(
                            name=config_name,
                            chunking_method=chunking_method,
                            retrieval_type=retrieval_type,
                            bias_config=bias_config
                        )
                        configs.append(config)
        
        logger.info(f"Generated {len(configs)} hybrid configurations")
        if len(configs) == 0:
            logger.warning("No configurations found! Checking data structure...")
            logger.warning(f"Looking for chunks in: {self.ragdata_path}")
            logger.warning(f"Looking for indices in: {self.embeddings_path}")
            
            # Debug: show what files we actually have
            if self.ragdata_path.exists():
                chunk_files = list(self.ragdata_path.glob("*_chunks_*.json"))
                logger.warning(f"Found {len(chunk_files)} chunk files")
                for f in chunk_files[:3]:
                    logger.warning(f"  Example chunk file: {f.name}")
            
            if self.embeddings_path.exists():
                index_files = list(self.embeddings_path.glob("*.index"))
                logger.warning(f"Found {len(index_files)} index files")
                for f in index_files[:3]:
                    logger.warning(f"  Example index file: {f.name}")
        
        return configs
    
    def _discover_chunking_methods(self) -> List[str]:
        """Discover available chunking methods"""
        chunking_methods = set()
        
        # Look for chunk files pattern: {source}_chunks_{method}.json
        for source in ['healthify', 'mayo', 'nhs']:
            chunk_files = list(self.ragdata_path.glob(f"{source}_chunks_*.json"))
            for chunk_file in chunk_files:
                # Extract chunking method
                filename = chunk_file.stem
                method = filename.replace(f"{source}_chunks_", "")
                chunking_methods.add(method)
        
        return sorted(list(chunking_methods))
    
    def _validate_chunking_method(self, chunking_method: str) -> bool:
        """Check if all required files exist for a chunking method"""
        sources = ['healthify', 'mayo', 'nhs']
        
        for source in sources:
            # Check chunk file
            chunk_file = self.ragdata_path / f"{source}_chunks_{chunking_method}.json"
            if not chunk_file.exists():
                return False
            
            # Check index file
            index_file = self.embeddings_path / f"{source}_vector_db_{chunking_method}.index"
            if not index_file.exists():
                return False
        
        return True

class MemoryMonitor:
    """Monitors memory usage during evaluation"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = 0
        self.peak_memory = 0
        
    def start_monitoring(self):
        """Start memory monitoring"""
        tracemalloc.start()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
    def get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
        
    def update_peak(self):
        """Update peak memory usage"""
        current = self.get_current_memory()
        if current > self.peak_memory:
            self.peak_memory = current
            
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        current = self.get_current_memory()
        current_traced, peak_traced = tracemalloc.get_traced_memory()
        
        return {
            'start_memory_mb': self.start_memory,
            'current_memory_mb': current,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': current - self.start_memory,
            'traced_current_mb': current_traced / 1024 / 1024,
            'traced_peak_mb': peak_traced / 1024 / 1024
        }
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        tracemalloc.stop()

class HybridEvaluator:
    """Evaluates hybrid retrieval configurations"""
    
    def __init__(self):
        self.chunk_loader = HybridChunkLoader()
        self.index_loader = HybridIndexLoader()
        self.memory_monitor = MemoryMonitor()
    
    def is_correct_retrieval(self, test_case: TestCase, retrieved_chunk: Dict[str, Any]) -> bool:
        """Check if retrieved chunk is correct for the test case"""
        # Extract the symptom name without (Case N) suffix
        symptom = re.sub(r'\s*\(Case\s+\d+\)$', '', test_case.symptom)
        actual_source = retrieved_chunk.get('source_document', '').lower()
        
        # Define alternative document name mappings
        condition_mappings = {
            'abdominal_aortic_aneurysm': ['aortic', 'aneurysm'],
            'acute_myocardial_infarction': ['heart_attack', 'myocardial', 'cardiac'],
            'asthma': ['asthma'],
            'migraine': ['migraine']
        }
        
        symptom_lower = symptom.lower()
        
        # Check direct match first
        if symptom_lower in actual_source:
            return True
            
        # Check alternative terms
        for condition, alternatives in condition_mappings.items():
            if condition in symptom_lower:
                return any(alt in actual_source for alt in alternatives)
        
        # Fallback to simple symptom name check
        return symptom_lower in actual_source
    
    def evaluate_configuration(self, config: RetrievalConfig, test_cases: List[TestCase], 
                             k_values: List[int] = [5, 10, 20]) -> TestResult:
        """Evaluate a single hybrid configuration"""
        logger.info(f"Evaluating configuration: {config.name} with {len(test_cases)} test cases")
        
        try:
            # Start memory monitoring
            self.memory_monitor.start_monitoring()
            
            # Initialize multi-source retriever
            retriever = MultiSourceRetriever()
            
            # Load data for each source
            sources = ['healthify', 'mayo', 'nhs']
            for source in sources:
                chunks = self.chunk_loader.load_source_chunks(source, config.chunking_method)
                index = self.index_loader.load_source_index(source, config.chunking_method)
                retriever.add_source(source, str(self.index_loader.embeddings_path / f"{source}_vector_db_{config.chunking_method}.index"), chunks)
                
            # Update memory after loading
            self.memory_monitor.update_peak()
            
            # Run evaluation
            results = {k: [] for k in k_values}
            retrieval_times = []
            detailed_results = []
            source_distribution = {'healthify': 0, 'mayo': 0, 'nhs': 0}
            
            for i, test_case in enumerate(tqdm(test_cases, desc=f"Testing {config.name}")):
                query = test_case.get_combined_query()
                
                # Measure retrieval time
                start_time = time.time()
                retrieved_docs = retriever.search_with_bias(
                    query, 
                    config.bias_config, 
                    config.retrieval_type
                )
                retrieval_time = time.time() - start_time
                retrieval_times.append(retrieval_time)
                
                # Update memory monitoring periodically
                if i % 100 == 0:
                    self.memory_monitor.update_peak()
                
                # Count source distribution
                for doc in retrieved_docs[:max(k_values)]:
                    source = doc.get('source', 'unknown')
                    if source in source_distribution:
                        source_distribution[source] += 1
                
                # Check results for each k value
                case_results = {}
                for k in k_values:
                    top_k_docs = retrieved_docs[:k]
                    is_correct = any(self.is_correct_retrieval(test_case, doc['chunk']) 
                                   for doc in top_k_docs)
                    results[k].append(is_correct)
                    case_results[f'pass_at_{k}'] = is_correct
                
                # Store detailed results
                detailed_results.append({
                    'test_case': test_case,
                    'query': query,
                    'results': case_results,
                    'retrieval_time': retrieval_time,
                    'top_retrieved': retrieved_docs[:5] if retrieved_docs else [],
                    'source_distribution': {doc.get('source', 'unknown'): 1 for doc in retrieved_docs[:10]}
                })
            
            # Calculate metrics
            pass_rates = {k: np.mean(results[k]) * 100 for k in k_values}
            
            # Get final memory stats
            memory_stats = self.memory_monitor.get_stats()
            self.memory_monitor.stop_monitoring()
            
            # Force garbage collection
            gc.collect()
            
            return TestResult(
                config=config,
                pass_at_5=pass_rates.get(5, 0),
                pass_at_10=pass_rates.get(10, 0),
                pass_at_20=pass_rates.get(20, 0),
                avg_retrieval_time=np.mean(retrieval_times),
                total_test_cases=len(test_cases),
                successful_retrievals=sum(results[max(k_values)]),
                detailed_results=detailed_results,
                source_distribution=source_distribution,
                memory_stats=memory_stats
            )
            
        except Exception as e:
            logger.error(f"Error evaluating config {config.name}: {e}")
            return TestResult(
                config=config,
                pass_at_5=0, pass_at_10=0, pass_at_20=0,
                avg_retrieval_time=0, total_test_cases=len(test_cases),
                successful_retrievals=0, detailed_results=[],
                source_distribution={'healthify': 0, 'mayo': 0, 'nhs': 0},
                memory_stats={'start_memory_mb': 0, 'current_memory_mb': 0, 'peak_memory_mb': 0, 'memory_increase_mb': 0, 'traced_current_mb': 0, 'traced_peak_mb': 0}
            )

class HybridResultsAnalyzer:
    """Analyzes and reports hybrid test results"""
    
    def __init__(self, results_dir: str = "results/hybrid_test"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results: List[TestResult], timestamp: str):
        """Save comprehensive results"""
        # Save summary results
        summary_file = self.results_dir / f"hybrid_summary_{timestamp}.json"
        summary_data = []
        
        for result in results:
            summary_data.append({
                'config_name': result.config.name,
                'chunking_method': result.config.chunking_method,
                'retrieval_type': result.config.retrieval_type,
                'bias_config': {
                    'name': result.config.bias_config.name,
                    'healthify': result.config.bias_config.healthify,
                    'mayo': result.config.bias_config.mayo,
                    'nhs': result.config.bias_config.nhs,
                    'description': result.config.bias_config.description
                },
                'pass_at_5': result.pass_at_5,
                'pass_at_10': result.pass_at_10,
                'pass_at_20': result.pass_at_20,
                'avg_retrieval_time_ms': result.avg_retrieval_time * 1000,
                'total_test_cases': result.total_test_cases,
                'successful_retrievals': result.successful_retrievals,
                'source_distribution': result.source_distribution,
                'memory_stats': result.memory_stats
            })
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Hybrid summary results saved to {summary_file}")
        
        # Save detailed analysis report
        self._save_detailed_report(results, timestamp)
    
    def _save_detailed_report(self, results: List[TestResult], timestamp: str):
        """Save detailed text report"""
        report_file = self.results_dir / f"hybrid_detailed_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("HYBRID RETRIEVAL EVALUATION REPORT\n")
            f.write("Multi-Source Bias Configuration + Pure RAG vs Contextual RAG\n")
            f.write("=" * 100 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Configurations Tested: {len(results)}\n")
            f.write(f"Evaluation Method: Multi-source stratified sampling with bias configurations\n")
            f.write(f"Retrieval Types: Pure RAG (original text) vs Contextual RAG (with context)\n\n")
            
            # Sort results by Pass@5 (primary metric for small models)
            sorted_results = sorted(results, key=lambda x: x.pass_at_5, reverse=True)
            
            # Overall rankings
            f.write("OVERALL PERFORMANCE RANKINGS\n")
            f.write("=" * 50 + "\n")
            f.write(f"{'Rank':<4} {'Configuration':<50} {'Pass@5':<8} {'Pass@10':<9} {'Time(ms)':<10}\n")
            f.write("-" * 85 + "\n")
            
            for rank, result in enumerate(sorted_results[:20], 1):
                f.write(f"{rank:<4} {result.config.name[:49]:<50} "
                       f"{result.pass_at_5:.1f}%{'':<3} "
                       f"{result.pass_at_10:.1f}%{'':<4} "
                       f"{result.avg_retrieval_time*1000:.1f}{'':<6}\n")
            
            # Analysis by retrieval type
            f.write(f"\n\nANALYSIS BY RETRIEVAL TYPE\n")
            f.write("=" * 40 + "\n")
            
            retrieval_performance = defaultdict(list)
            for result in results:
                retrieval_performance[result.config.retrieval_type].append(result)
            
            for ret_type, ret_results in retrieval_performance.items():
                avg_pass_10 = np.mean([r.pass_at_10 for r in ret_results])
                avg_time = np.mean([r.avg_retrieval_time for r in ret_results])
                best_config = max(ret_results, key=lambda x: x.pass_at_10)
                
                f.write(f"\n{ret_type.upper()}\n")
                f.write("-" * len(ret_type) + "\n")
                f.write(f"Average Pass@10: {avg_pass_10:.1f}%\n")
                f.write(f"Average Time: {avg_time*1000:.1f}ms\n")
                f.write(f"Best Configuration: {best_config.config.name}\n")
                f.write(f"Best Pass@10: {best_config.pass_at_10:.1f}%\n")
            
            # Analysis by bias configuration
            f.write(f"\n\nANALYSIS BY BIAS CONFIGURATION\n")
            f.write("=" * 45 + "\n")
            
            bias_performance = defaultdict(list)
            for result in results:
                bias_performance[result.config.bias_config.name].append(result)
            
            for bias_name, bias_results in bias_performance.items():
                avg_pass_10 = np.mean([r.pass_at_10 for r in bias_results])
                best_config = max(bias_results, key=lambda x: x.pass_at_10)
                
                f.write(f"\n{bias_name.upper()}\n")
                f.write("-" * len(bias_name) + "\n")
                f.write(f"Configuration: {best_config.config.bias_config.healthify}:{best_config.config.bias_config.mayo}:{best_config.config.bias_config.nhs} (H:M:N)\n")
                f.write(f"Description: {best_config.config.bias_config.description}\n")
                f.write(f"Average Pass@10: {avg_pass_10:.1f}%\n")
                f.write(f"Best Configuration: {best_config.config.name}\n")
                f.write(f"Best Pass@10: {best_config.pass_at_10:.1f}%\n")
            
            # Analysis by chunking method
            f.write(f"\n\nANALYSIS BY CHUNKING METHOD\n")
            f.write("=" * 40 + "\n")
            
            chunking_performance = defaultdict(list)
            for result in results:
                chunking_performance[result.config.chunking_method].append(result)
            
            for method, method_results in chunking_performance.items():
                avg_pass_10 = np.mean([r.pass_at_10 for r in method_results])
                best_config = max(method_results, key=lambda x: x.pass_at_10)
                
                f.write(f"\n{method}\n")
                f.write("-" * len(method) + "\n")
                f.write(f"Average Pass@10: {avg_pass_10:.1f}%\n")
                f.write(f"Best Configuration: {best_config.config.name}\n")
                f.write(f"Best Pass@10: {best_config.pass_at_10:.1f}%\n")
        
        logger.info(f"Detailed report saved to {report_file}")
    
    def print_summary(self, results: List[TestResult]):
        """Print summary of results"""
        print("\n" + "="*100)
        print("HYBRID RETRIEVAL EVALUATION RESULTS")
        print("Multi-Source Bias + Pure RAG vs Contextual RAG")
        print("="*100)
        
        # Sort by Pass@5 performance (primary metric for small models)
        sorted_results = sorted(results, key=lambda x: x.pass_at_5, reverse=True)
        
        print(f"{'Rank':<4} {'Configuration':<35} {'Type':<12} {'Bias':<15} {'Pass@5':<8} {'Time(ms)':<8} {'Memory(MB)':<10}")
        print("-" * 105)
        
        for i, result in enumerate(sorted_results[:15], 1):
            bias_ratio = f"{result.config.bias_config.healthify}:{result.config.bias_config.mayo}:{result.config.bias_config.nhs}"
            memory_peak = result.memory_stats.get('peak_memory_mb', 0)
            print(f"{i:<4} {result.config.chunking_method[:34]:<35} "
                  f"{result.config.retrieval_type[:11]:<12} "
                  f"{bias_ratio:<15} "
                  f"{result.pass_at_5:.1f}%{'':<3} "
                  f"{result.avg_retrieval_time*1000:.1f}{'':<4} "
                  f"{memory_peak:.0f}{'':<6}")
        
        # Quick analysis
        print(f"\n" + "="*80)
        print("QUICK ANALYSIS")
        print("="*80)
        
        # Best overall
        if sorted_results:
            best = sorted_results[0]
            print(f"🏆 Best Overall: {best.config.name}")
            print(f"   Pass@5: {best.pass_at_5:.1f}% (Primary Metric)")
            print(f"   Pass@10: {best.pass_at_10:.1f}%")
            print(f"   Retrieval Type: {best.config.retrieval_type}")
            print(f"   Bias Config: {best.config.bias_config.healthify}:{best.config.bias_config.mayo}:{best.config.bias_config.nhs} (H:M:N)")
            print(f"   Peak Memory: {best.memory_stats.get('peak_memory_mb', 0):.1f} MB")
            print(f"   Memory Increase: {best.memory_stats.get('memory_increase_mb', 0):.1f} MB")
        else:
            print("❌ No configurations were successfully evaluated.")
            print("   Check that RAGdatav4/indiv_embeddings contains the required index files.")
            return
        
        # Compare retrieval types
        pure_rag_results = [r for r in results if r.config.retrieval_type == 'pure_rag']
        contextual_rag_results = [r for r in results if r.config.retrieval_type == 'contextual_rag']
        
        if pure_rag_results and contextual_rag_results:
            pure_avg = np.mean([r.pass_at_5 for r in pure_rag_results])
            contextual_avg = np.mean([r.pass_at_5 for r in contextual_rag_results])
            
            print(f"\n📊 Retrieval Type Comparison (Pass@5):")
            print(f"   Pure RAG Average: {pure_avg:.1f}%")
            print(f"   Contextual RAG Average: {contextual_avg:.1f}%")
            
            if contextual_avg > pure_avg:
                print(f"   ✅ Contextual RAG outperforms Pure RAG by {contextual_avg - pure_avg:.1f}%")
            else:
                print(f"   ⚠️  Pure RAG outperforms Contextual RAG by {pure_avg - contextual_avg:.1f}%")

class HybridRetrievalTester:
    """Main class for hybrid retrieval testing"""
    
    def __init__(self, ragdata_path: str = "RAGdatav4"):
        self.ragdata_path = ragdata_path
        self.config_scanner = ConfigurationScanner(ragdata_path)
        self.evaluator = HybridEvaluator()
        self.analyzer = HybridResultsAnalyzer()
    
    def load_test_data(self, test_data_file: str) -> List[TestCase]:
        """Load test cases from JSON file"""
        if not os.path.exists(test_data_file):
            logger.error(f"Test data file not found: {test_data_file}")
            return []
        
        test_cases = []
        with open(test_data_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                try:
                    # Clean quoted strings if present
                    clean_query = item['patient_query'].strip('"') if item['patient_query'].startswith('"') else item['patient_query']
                    clean_response = item['patient_response'].strip('"') if item['patient_response'].startswith('"') else item['patient_response']
                    clean_clarifying = item.get('clarifying_question', '').strip('"') if item.get('clarifying_question', '').startswith('"') else item.get('clarifying_question', '')
                    
                    test_case = TestCase(
                        id=item.get('id'),
                        symptom=item['symptom'],
                        patient_query=clean_query,
                        patient_response=clean_response,
                        final_triage_decision=item['final_triage_decision'],
                        next_step=item['next_step'],
                        reasoning_question=item.get('reasoning_question', ''),
                        reasoning_decision=item.get('reasoning_decision', ''),
                        clarifying_question=clean_clarifying,
                        generation_timestamp=item.get('generation_timestamp', time.time())
                    )
                    test_cases.append(test_case)
                except Exception as e:
                    logger.warning(f"Failed to parse test case: {e}, item keys: {item.keys()}")
                    continue
        else:
            logger.error("Test data should be a list of test cases")
            return []
        
        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases
    
    def run_comprehensive_test(self, test_data_file: str, max_configs: Optional[int] = None,
                             chunking_filter: Optional[str] = None,
                             bias_filter: Optional[str] = None) -> List[TestResult]:
        """Run comprehensive hybrid testing"""
        # Load test data
        test_cases = self.load_test_data(test_data_file)
        if not test_cases:
            logger.error("No test cases loaded. Exiting.")
            return []
        
        # Scan available configurations
        configs = self.config_scanner.scan_available_configurations()
        
        # Apply filters
        if chunking_filter:
            configs = [c for c in configs if chunking_filter in c.chunking_method]
        if bias_filter:
            configs = [c for c in configs if bias_filter in c.bias_config.name]
        if max_configs:
            configs = configs[:max_configs]
        
        logger.info(f"Testing {len(configs)} hybrid configurations with {len(test_cases)} test cases")
        
        # Run evaluations
        results = []
        for config in configs:
            result = self.evaluator.evaluate_configuration(config, test_cases)
            results.append(result)
        
        # Save and analyze results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.analyzer.save_results(results, timestamp)
        self.analyzer.print_summary(results)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Hybrid Retrieval Performance Testing')
    parser.add_argument('--test-data', required=True, help='Path to test data JSON file')
    parser.add_argument('--ragdata-path', default='RAGdatav4', help='Path to RAGdata directory')
    parser.add_argument('--results-dir', default='results/hybrid_test', help='Directory to save results')
    parser.add_argument('--max-configs', type=int, help='Maximum number of configurations to test')
    parser.add_argument('--chunking-filter', help='Filter configurations by chunking method')
    parser.add_argument('--bias-filter', help='Filter configurations by bias configuration name')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = HybridRetrievalTester(args.ragdata_path)
    tester.analyzer = HybridResultsAnalyzer(args.results_dir)
    
    # Run tests
    logger.info("Starting hybrid retrieval performance testing...")
    logger.info("Testing: Multi-source bias configurations + Pure RAG vs Contextual RAG")
    
    results = tester.run_comprehensive_test(
        args.test_data, 
        args.max_configs,
        args.chunking_filter,
        args.bias_filter
    )
    
    logger.info(f"Hybrid testing completed. Results saved to {args.results_dir}")

if __name__ == "__main__":
    main()