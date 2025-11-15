#!/usr/bin/env python3
"""
Comprehensive Retrieval Performance Testing System

This script systematically evaluates different chunking methods and retrieval techniques
for medical triage dialogues. It tests the retrieval accuracy by checking if the 
correct medical condition documents are retrieved based on patient queries.

Usage:
    python retrieval_performance_tester.py --test-data eval/test_data.json --results-dir results/
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

# Import existing retrieval classes
from offline_contextual_retrieval import OfflineContextualRetrieval, OfflineVectorDB, BM25Search, OfflineReranker, HybridRetriever
from sentence_transformers import SentenceTransformer

# Simple retrieval system based on working rag_chat.py
class SimpleRetriever:
    """Simple FAISS-based retriever like the working rag_chat.py"""
    
    def __init__(self, model_name: str, index_path: str, chunks: List[Dict]):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.chunks = chunks
        
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Simple search using FAISS index"""
        # Encode query
        query_embedding = self.model.encode([query]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1 and idx < len(self.chunks):
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(1.0 / (1.0 + distance)),  # Convert distance to similarity
                    'distance': float(distance),
                    'rank': i
                })
        
        return results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Represents a single test case with patient dialogue and expected answer"""
    symptom: str
    patient_query: str
    patient_response: str
    final_triage_decision: str
    next_step: str
    reasoning_question: str
    reasoning_decision: str
    clarifying_question: str
    generation_timestamp: float
    
    def get_combined_query(self) -> str:
        """Combine patient query and response for retrieval"""
        return f"{self.patient_query} {self.patient_response}"
    
    def get_expected_source_doc(self) -> str:
        """Extract the expected source document name from symptom"""
        # Remove (Case N) suffix and normalize
        clean_symptom = re.sub(r'\s*\(Case\s+\d+\)$', '', self.symptom)
        return f"{clean_symptom}.txt"

@dataclass
class RetrievalConfig:
    """Configuration for a specific retrieval method"""
    name: str
    chunking_method: str
    embedding_file: str
    index_file: str
    chunk_data_file: str
    retrieval_type: str  # 'semantic', 'bm25', 'hybrid', 'contextual'
    rerank_method: Optional[str] = None

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

class ChunkLoader:
    """Loads and manages chunk data from RAGdatav4"""
    
    def __init__(self, ragdata_path: str = "RAGdatav4"):
        # Get the parent directory of the final_retrieval_testing folder
        parent_dir = Path(__file__).parent.parent
        self.ragdata_path = parent_dir / ragdata_path
        self.chunks_cache = {}
    
    def load_chunks(self, chunk_file: str) -> List[Dict[str, Any]]:
        """Load chunks from JSON file or combine multiple source files"""
        if chunk_file in self.chunks_cache:
            return self.chunks_cache[chunk_file]
        
        # Check if this is a request for combined chunks
        if chunk_file.startswith('combined_chunks_'):
            return self._load_combined_chunks(chunk_file)
        
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
        
        self.chunks_cache[chunk_file] = standardized_chunks
        return standardized_chunks
    
    def _load_combined_chunks(self, combined_file: str) -> List[Dict[str, Any]]:
        """Load and combine chunks from all three sources to match combined embeddings"""
        if combined_file in self.chunks_cache:
            return self.chunks_cache[combined_file]
        
        # Extract chunking method from combined filename
        # e.g., "combined_chunks_fixed_c1024_o150.json" -> "fixed_c1024_o150"
        chunking_method = combined_file.replace('combined_chunks_', '').replace('.json', '')
        
        # Load chunks from all three sources
        all_chunks = []
        sources = ['healthify', 'mayo', 'nhs']
        
        for source in sources:
            source_file = f"{source}_chunks_{chunking_method}.json"
            source_path = self.ragdata_path / source_file
            
            if source_path.exists():
                with open(source_path, 'r', encoding='utf-8') as f:
                    source_chunks = json.load(f)
                
                # Standardize and add to combined list
                for chunk in source_chunks:
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
                    all_chunks.append(standardized_chunk)
                    
                logger.info(f"Loaded {len(source_chunks)} chunks from {source}")
            else:
                logger.warning(f"Source file not found: {source_path}")
        
        logger.info(f"Combined total: {len(all_chunks)} chunks for {chunking_method}")
        self.chunks_cache[combined_file] = all_chunks
        return all_chunks

class EmbeddingLoader:
    """Loads and manages embedding files and FAISS indices"""
    
    def __init__(self, embeddings_path: str = "RAGdatav4/embeddings"):
        # Get the parent directory of the final_retrieval_testing folder
        parent_dir = Path(__file__).parent.parent
        self.embeddings_path = parent_dir / embeddings_path
        self.embeddings_cache = {}
        self.indices_cache = {}
    
    def load_embeddings(self, embedding_file: str) -> np.ndarray:
        """Load embeddings from pickle file"""
        if embedding_file in self.embeddings_cache:
            return self.embeddings_cache[embedding_file]
        
        file_path = self.embeddings_path / embedding_file
        if not file_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        self.embeddings_cache[embedding_file] = embeddings
        return embeddings
    
    def load_faiss_index(self, index_file: str) -> faiss.Index:
        """Load FAISS index from file"""
        if index_file in self.indices_cache:
            return self.indices_cache[index_file]
        
        file_path = self.embeddings_path / index_file
        if not file_path.exists():
            raise FileNotFoundError(f"Index file not found: {file_path}")
        
        index = faiss.read_index(str(file_path))
        self.indices_cache[index_file] = index
        return index

class ConfigurationScanner:
    """Scans and identifies available retrieval configurations"""
    
    def __init__(self, ragdata_path: str = "RAGdatav4", embeddings_path: str = "RAGdatav4/embeddings"):
        # Get the parent directory of the final_retrieval_testing folder
        parent_dir = Path(__file__).parent.parent
        self.ragdata_path = parent_dir / ragdata_path
        self.embeddings_path = parent_dir / embeddings_path
    
    def scan_available_configs(self) -> List[RetrievalConfig]:
        """Scan for all available chunking and embedding combinations"""
        configs = []
        
        # Find all embedding files
        embedding_files = list(self.embeddings_path.glob("embeddings_*.pkl"))
        
        for emb_file in embedding_files:
            # Extract chunking method from filename
            filename = emb_file.stem
            # Pattern: embeddings_{chunking_method}_all-MiniLM-L6-v2
            parts = filename.split('_')
            if len(parts) >= 3:
                chunking_method = '_'.join(parts[1:-1])  # Remove 'embeddings' and model name
                
                # Find corresponding index file
                index_file = emb_file.with_name(f"faiss_index_{chunking_method}_all-MiniLM-L6-v2.index")
                
                if index_file.exists():
                    # Check if we have all three source chunk files for combined loading
                    required_sources = ['healthify', 'mayo', 'nhs']
                    all_sources_exist = True
                    
                    for source in required_sources:
                        source_file = self.ragdata_path / f"{source}_chunks_{chunking_method}.json"
                        if not source_file.exists():
                            all_sources_exist = False
                            break
                    
                    if all_sources_exist:
                        # Determine retrieval types to test
                        retrieval_types = ['semantic', 'bm25', 'hybrid']
                        if 'contextual' in chunking_method:
                            retrieval_types.append('contextual')
                        
                        for ret_type in retrieval_types:
                            config = RetrievalConfig(
                                name=f"{chunking_method}_{ret_type}",
                                chunking_method=chunking_method,
                                embedding_file=emb_file.name,
                                index_file=index_file.name,
                                chunk_data_file=f"combined_chunks_{chunking_method}.json",  # Use combined chunks
                                retrieval_type=ret_type
                            )
                            configs.append(config)
        
        logger.info(f"Found {len(configs)} retrieval configurations")
        return configs

class RetrievalEvaluator:
    """Evaluates retrieval performance for different configurations"""
    
    def __init__(self):
        self.chunk_loader = ChunkLoader()
        self.embedding_loader = EmbeddingLoader()
    
    def is_correct_retrieval(self, test_case: TestCase, retrieved_chunk: Dict[str, Any]) -> bool:
        """Check if retrieved chunk is correct for the test case"""
        # Extract the symptom name without (Case N) suffix
        symptom = re.sub(r'\s*\(Case\s+\d+\)$', '', test_case.symptom)
        actual_source = retrieved_chunk.get('source_document', '').lower()
        
        # Define alternative document name mappings for different conditions
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
    
    def evaluate_single_config(self, config: RetrievalConfig, test_cases: List[TestCase], 
                             k_values: List[int] = [5, 10, 20]) -> TestResult:
        """Evaluate a single retrieval configuration"""
        logger.info(f"Evaluating configuration: {config.name}")
        
        try:
            # Load required data
            chunks = self.chunk_loader.load_chunks(config.chunk_data_file)
            embeddings = self.embedding_loader.load_embeddings(config.embedding_file)
            faiss_index = self.embedding_loader.load_faiss_index(config.index_file)
            
            # Initialize retrieval system
            retrieval_system = self._create_retrieval_system(config, chunks, embeddings, faiss_index)
            
            # Run evaluation
            results = {k: [] for k in k_values}
            retrieval_times = []
            detailed_results = []
            
            for test_case in tqdm(test_cases, desc=f"Testing {config.name}"):
                query = test_case.get_combined_query()
                
                # Measure retrieval time
                start_time = time.time()
                retrieved_docs = retrieval_system.search(query, top_k=max(k_values))
                retrieval_time = time.time() - start_time
                retrieval_times.append(retrieval_time)
                
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
                    'top_retrieved': retrieved_docs[:5] if retrieved_docs else []
                })
            
            # Calculate metrics
            pass_rates = {k: np.mean(results[k]) * 100 for k in k_values}
            
            return TestResult(
                config=config,
                pass_at_5=pass_rates.get(5, 0),
                pass_at_10=pass_rates.get(10, 0),
                pass_at_20=pass_rates.get(20, 0),
                avg_retrieval_time=np.mean(retrieval_times),
                total_test_cases=len(test_cases),
                successful_retrievals=sum(results[max(k_values)]),
                detailed_results=detailed_results
            )
            
        except Exception as e:
            logger.error(f"Error evaluating config {config.name}: {e}")
            return TestResult(
                config=config,
                pass_at_5=0, pass_at_10=0, pass_at_20=0,
                avg_retrieval_time=0, total_test_cases=len(test_cases),
                successful_retrievals=0, detailed_results=[]
            )
    
    def _create_retrieval_system(self, config: RetrievalConfig, chunks: List[Dict], 
                               embeddings: np.ndarray, faiss_index: faiss.Index):
        """Create retrieval system based on configuration"""
        # Handle the mismatch between embeddings and chunks
        if len(embeddings) > len(chunks):
            logger.warning(f"Truncating embeddings from {len(embeddings)} to {len(chunks)} to match chunks")
            embeddings = embeddings[:len(chunks)]
        elif len(chunks) > len(embeddings):
            logger.warning(f"Truncating chunks from {len(chunks)} to {len(embeddings)} to match embeddings")
            chunks = chunks[:len(embeddings)]
        
        # Use simple retriever for semantic search (like working rag_chat.py)
        if config.retrieval_type == "semantic":
            # Create index file path from config
            index_path = self.embedding_loader.embeddings_path / config.index_file
            return SimpleRetriever(
                model_name='all-MiniLM-L6-v2',
                index_path=str(index_path),
                chunks=chunks
            )
        
        # Use complex system for other retrieval types
        retrieval_system = OfflineContextualRetrieval(
            model_name='all-MiniLM-L6-v2',
            rerank_method=config.rerank_method or "feature"
        )
        
        # Set preloaded data properly
        retrieval_system.chunks = chunks
        retrieval_system.vector_db.chunks = chunks
        retrieval_system.vector_db.embeddings = embeddings
        retrieval_system.vector_db.index = faiss_index
        
        # Create hybrid retriever
        retrieval_system.hybrid_retriever = HybridRetriever(retrieval_system.vector_db, chunks)
        
        return retrieval_system

class ResultsAnalyzer:
    """Analyzes and reports test results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def save_results(self, results: List[TestResult], timestamp: str):
        """Save detailed results to files"""
        # Save summary results
        summary_file = self.results_dir / f"summary_{timestamp}.json"
        summary_data = []
        
        for result in results:
            summary_data.append({
                'config_name': result.config.name,
                'chunking_method': result.config.chunking_method,
                'retrieval_type': result.config.retrieval_type,
                'pass_at_5': result.pass_at_5,
                'pass_at_10': result.pass_at_10,
                'pass_at_20': result.pass_at_20,
                'avg_retrieval_time_ms': result.avg_retrieval_time * 1000,
                'total_test_cases': result.total_test_cases,
                'successful_retrievals': result.successful_retrievals
            })
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Summary results saved to {summary_file}")
        
        # Save detailed results for top performing configs
        top_configs = sorted(results, key=lambda x: x.pass_at_10, reverse=True)[:5]
        for i, result in enumerate(top_configs):
            detail_file = self.results_dir / f"detailed_{result.config.name}_{timestamp}.json"
            with open(detail_file, 'w') as f:
                json.dump({
                    'config': result.config.__dict__,
                    'metrics': {
                        'pass_at_5': result.pass_at_5,
                        'pass_at_10': result.pass_at_10,
                        'pass_at_20': result.pass_at_20,
                        'avg_retrieval_time_ms': result.avg_retrieval_time * 1000
                    },
                    'detailed_results': result.detailed_results[:10]  # Save first 10 for space
                }, f, indent=2, default=str)
    
    def print_summary(self, results: List[TestResult]):
        """Print summary of results"""
        print("\n" + "="*80)
        print("RETRIEVAL PERFORMANCE TESTING RESULTS")
        print("="*80)
        
        # Sort by Pass@10 performance
        sorted_results = sorted(results, key=lambda x: x.pass_at_10, reverse=True)
        
        print(f"{'Rank':<4} {'Configuration':<30} {'Pass@5':<8} {'Pass@10':<9} {'Pass@20':<9} {'Avg Time (ms)':<12}")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results[:10], 1):
            print(f"{i:<4} {result.config.name:<30} {result.pass_at_5:.1f}%{'':<3} "
                  f"{result.pass_at_10:.1f}%{'':<4} {result.pass_at_20:.1f}%{'':<4} "
                  f"{result.avg_retrieval_time*1000:.1f}{'':<8}")
        
        # Analysis by chunking method
        print("\n" + "="*60)
        print("ANALYSIS BY CHUNKING METHOD")
        print("="*60)
        
        chunking_performance = defaultdict(list)
        for result in results:
            chunking_performance[result.config.chunking_method].append(result)
        
        for method, method_results in chunking_performance.items():
            avg_pass_10 = np.mean([r.pass_at_10 for r in method_results])
            best_retrieval = max(method_results, key=lambda x: x.pass_at_10).config.retrieval_type
            print(f"{method:<25} Avg Pass@10: {avg_pass_10:.1f}%  Best: {best_retrieval}")
        
        # Analysis by retrieval type
        print("\n" + "="*60)
        print("ANALYSIS BY RETRIEVAL TYPE")
        print("="*60)
        
        retrieval_performance = defaultdict(list)
        for result in results:
            retrieval_performance[result.config.retrieval_type].append(result)
        
        for ret_type, ret_results in retrieval_performance.items():
            avg_pass_10 = np.mean([r.pass_at_10 for r in ret_results])
            avg_time = np.mean([r.avg_retrieval_time for r in ret_results])
            print(f"{ret_type:<15} Avg Pass@10: {avg_pass_10:.1f}%  Avg Time: {avg_time*1000:.1f}ms")

class RetrievalPerformanceTester:
    """Main class orchestrating the testing process"""
    
    def __init__(self, ragdata_path: str = "RAGdatav4"):
        self.ragdata_path = ragdata_path
        self.config_scanner = ConfigurationScanner(ragdata_path)
        self.evaluator = RetrievalEvaluator()
        self.analyzer = ResultsAnalyzer()
    
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
                test_case = TestCase(**item)
                test_cases.append(test_case)
        else:
            logger.error("Test data should be a list of test cases")
            return []
        
        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases
    
    def run_comprehensive_test(self, test_data_file: str, max_configs: Optional[int] = None) -> List[TestResult]:
        """Run comprehensive testing across all configurations"""
        # Load test data
        test_cases = self.load_test_data(test_data_file)
        if not test_cases:
            logger.error("No test cases loaded. Exiting.")
            return []
        
        # Scan available configurations
        configs = self.config_scanner.scan_available_configs()
        if max_configs:
            configs = configs[:max_configs]
        
        logger.info(f"Testing {len(configs)} configurations with {len(test_cases)} test cases")
        
        # Run evaluations
        results = []
        for config in configs:
            result = self.evaluator.evaluate_single_config(config, test_cases)
            results.append(result)
        
        # Save and analyze results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.analyzer.save_results(results, timestamp)
        self.analyzer.print_summary(results)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Test retrieval performance across chunking methods')
    parser.add_argument('--test-data', required=True, help='Path to test data JSON file')
    parser.add_argument('--ragdata-path', default='RAGdatav4', help='Path to RAGdata directory')
    parser.add_argument('--results-dir', default='results', help='Directory to save results')
    parser.add_argument('--max-configs', type=int, help='Maximum number of configurations to test')
    parser.add_argument('--config-filter', help='Filter configurations by name pattern')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = RetrievalPerformanceTester(args.ragdata_path)
    tester.analyzer = ResultsAnalyzer(args.results_dir)
    
    # Run tests
    logger.info("Starting comprehensive retrieval performance testing...")
    results = tester.run_comprehensive_test(args.test_data, args.max_configs)
    
    logger.info(f"Testing completed. Results saved to {args.results_dir}")

if __name__ == "__main__":
    main()