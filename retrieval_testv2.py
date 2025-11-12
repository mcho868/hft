from retrieval_function import MultiSourceRagRetriever
import time
import json
import os
import glob
from datetime import datetime
import re
# No additional imports needed for source-based evaluation

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
RAGDATA_PATH = '/Users/choemanseung/789/hft/RAGdatav2'

def parse_config_name(filename):
    """Parse configuration details from filename"""
    # Remove source prefix and file extension
    config_part = filename.replace('_chunks_', '_').replace('_vector_db_', '_')
    for source in ['nhs_', 'mayo_', 'healthify_']:
        config_part = config_part.replace(source, '')
    config_part = config_part.replace('.json', '').replace('.index', '')
    # Remove the leading underscore if present
    if config_part.startswith('_'):
        config_part = config_part[1:]
    
    config = {"config_name": config_part}
    
    # Parse different configuration types
    if config_part.startswith('fixed_'):
        # e.g., fixed_c512_o100
        match = re.search(r'fixed_c(\d+)_o(\d+)', config_part)
        if match:
            config.update({
                "strategy": "fixed",
                "chunk_size": int(match.group(1)),
                "overlap": int(match.group(2))
            })

    elif config_part.startswith('sentence_'):
        # e.g., sentence_t512_o2
        match = re.search(r'sentence_t(\d+)_o(\d+)', config_part)
        if match:
            config.update({
                "strategy": "sentence", 
                "target_size": int(match.group(1)),
                "overlap_sentences": int(match.group(2))
            })

    elif config_part.startswith('paragraph_'):
        # e.g., paragraph_m100
        match = re.search(r'paragraph_m(\d+)', config_part)
        if match:
            config.update({
                "strategy": "paragraph",
                "min_length": int(match.group(1))
            })

    elif config_part.startswith('structured_agent_'):
        # e.g., structured_agent_gemma3_4b_medical_json
        config.update({
            "strategy": "structured_agent",
            "chunk_size": "Variable",
            "overlap": "AI-structured"
        })

    elif config_part.startswith('agent_'):
        # e.g., agent_mlx_gemma3_4b_1_you_are_an_expert_me or agent_mlx_gemma3_4b_1_you_are_an_expert_me_deduplicated
        if 'deduplicated' in config_part:
            config.update({
                "strategy": "agent_deduplicated",
                "chunk_size": "Variable",
                "overlap": "Deduplicated"
            })
        else:
            config.update({
                "strategy": "agent",
                "chunk_size": "Variable", 
                "overlap": "AI-generated"
            })

    return config

def get_chunk_display_text(chunk):
    """Return a concise, human-readable preview text for a chunk.

    For structured-agent chunks, build a summary from structured fields.
    Fallback to plain text when not available.
    """
    try:
        if isinstance(chunk, dict) and chunk.get('is_structured') and isinstance(chunk.get('structured_data'), dict):
            sd = chunk['structured_data']
            parts = []
            condition = sd.get('condition')
            if condition:
                parts.append(f"Condition: {condition}")
            symptoms = sd.get('symptoms')
            if isinstance(symptoms, list) and symptoms:
                parts.append("Symptoms: " + ", ".join([str(s) for s in symptoms[:5]]))
            triage = sd.get('triage_level') or sd.get('triage')
            if triage:
                parts.append(f"Triage: {triage}")
            steps = sd.get('next_steps')
            if isinstance(steps, list) and steps:
                parts.append("Next steps: " + str(steps[0]))
            elif isinstance(steps, str) and steps:
                parts.append("Next steps: " + steps)
            if parts:
                return " | ".join(parts)
    except Exception:
        pass
    # Fallback: plain text
    if isinstance(chunk, dict):
        return str(chunk.get('text', ''))
    return str(chunk)

def normalize_source_label(source_name: str) -> str:
    """Map arbitrary source name strings to canonical keys: healthify, mayo, nhs."""
    s = str(source_name or '').strip().lower()
    if 'health' in s:
        return 'healthify'
    if 'mayo' in s:
        return 'mayo'
    if 'nhs' in s:
        return 'nhs'
    return s

def discover_configurations():
    """Discover all available configurations from RAGdatav2 directory"""
    configurations = {}
    
    # Find all chunk files
    chunk_files = glob.glob(os.path.join(RAGDATA_PATH, '*_chunks_*.json'))
    
    for chunk_file in chunk_files:
        filename = os.path.basename(chunk_file)
        config = parse_config_name(filename)
        config_name = config["config_name"]
        
        # Find corresponding index files
        index_files = []
        for source in ['nhs', 'mayo', 'healthify']:
            index_file = os.path.join(RAGDATA_PATH, f'{source}_vector_db_{config_name}.index')
            if os.path.exists(index_file):
                index_files.append(index_file)
        
        # Find corresponding chunk files for all sources
        chunk_files_all = []
        for source in ['nhs', 'mayo', 'healthify']:
            chunk_file_source = os.path.join(RAGDATA_PATH, f'{source}_chunks_{config_name}.json')
            if os.path.exists(chunk_file_source):
                chunk_files_all.append(chunk_file_source)
        
        # Only include if we have files for all 3 sources
        if len(index_files) == 3 and len(chunk_files_all) == 3:
            configurations[config_name] = {
                "config": config,
                "index_paths": index_files,
                "chunks_paths": chunk_files_all,
                "source_names": ["NHS", "Mayo", "Healthify"]
            }
    
    return configurations

# Source-based evaluation - no similarity calculation needed

def evaluate_metrics_at_k(retrieved_chunk_objects, golden_sources, k_values=[5, 10]):
    """
    Evaluate Hit@K, Recall@K and Precision@K for retrieved chunks
    
    Args:
        retrieved_chunk_objects: List of chunk objects with source_document field
        golden_sources: List of source document paths from golden answer
        k_values: List of k values to evaluate
        
    Returns:
        Dict with hit@k, recall@k and precision@k metrics
        
    Definitions:
        - Hit@K = 1 if at least one golden source found in top-k, 0 otherwise (binary success)
        - Recall@K = (# unique golden sources found in top-k) / (total # golden sources)
        - Precision@K = (# chunks from golden sources in top-k) / k
    """
    results = {}
    
    # Create a mapping function between golden source format and chunk source format
    def normalize_source_for_matching(source_path, is_golden_source=False):
        """
        Normalize paths for comparison:
        - Golden sources: /cleaned_healthify_data/file.txt -> healthify/file.txt
        - Chunk sources: healthify/file.txt -> healthify/file.txt (no change)
        """
        if is_golden_source:
            # Convert cleaned_X_data format to X format
            if 'cleaned_healthify_data' in source_path:
                filename = source_path.split('/')[-1]
                print(f"healthify/{filename}")
                return f"healthify/{filename}"
            elif 'cleaned_mayo_data' in source_path:
                filename = source_path.split('/')[-1]
                print(f"mayo/{filename}")
                return f"mayo/{filename}"
            elif 'cleaned_nhs_data' in source_path:
                filename = source_path.split('/')[-1]
                print(f"nhs/{filename}")
                return f"nhs/{filename}"
        
        # For chunk sources or if no mapping found, use last two parts
        path_parts = source_path.split('/')
        if len(path_parts) >= 2:
            return '/'.join(path_parts[-2:])
        


        return source_path
    
    # Normalize golden sources
    normalized_golden_sources = set()
    for source_path in golden_sources:
        normalized_path = normalize_source_for_matching(source_path, is_golden_source=True)
        normalized_golden_sources.add(normalized_path)
    
    total_golden_sources = len(normalized_golden_sources)
    
    # Debug: Print normalized golden sources
    print(f"    Debug normalized golden sources: {list(normalized_golden_sources)}")
    
    # For each k value, calculate recall and precision
    for k in k_values:
        top_k_chunks = retrieved_chunk_objects[:k] if len(retrieved_chunk_objects) >= k else retrieved_chunk_objects
        actual_k = len(top_k_chunks)
        
        # Count chunks from correct sources and which golden sources were found
        correct_source_chunks = 0  # For precision
        found_golden_sources = set()  # For recall
        matching_chunks = []
        
        for i, chunk_obj in enumerate(top_k_chunks):
            chunk_source_raw = chunk_obj.get('source_document', '')
            
            # Debug: Print first few chunk sources to see the format
            if i < 3:
                print(f"    Debug chunk {i}: raw='{chunk_source_raw}', retriever_source='{chunk_obj.get('retriever_source', 'N/A')}'")
            
            # Use retriever_source to construct proper path format
            retriever_source = chunk_obj.get('retriever_source', '')
            if retriever_source and chunk_source_raw:
                # Map retriever source names to directory prefixes
                if retriever_source.lower() in ['nhs', 'mayo', 'healthify']:
                    chunk_source = f"{retriever_source.lower()}/{chunk_source_raw}"
                else:
                    chunk_source = chunk_source_raw
            else:
                chunk_source = normalize_source_for_matching(chunk_source_raw, is_golden_source=False)
            
            if i < 3:
                print(f"    Debug chunk {i}: final_source='{chunk_source}'")
            
            # Check if this chunk's source matches any golden source
            if chunk_source in normalized_golden_sources:
                correct_source_chunks += 1
                found_golden_sources.add(chunk_source)
                matching_chunks.append({
                    'chunk_idx': i,
                    'source_document': chunk_source,
                    'chunk_id': chunk_obj.get('chunk_id', '')
                })
        
        # Calculate metrics
        # Hit@K: Binary success - 1 if at least one golden source found, 0 otherwise
        hit_at_k = 1 if len(found_golden_sources) > 0 else 0
        
        # Recall@K: How many of the golden sources were found
        recall_at_k = len(found_golden_sources) / total_golden_sources if total_golden_sources > 0 else 0
        
        # Precision@K: How many of the retrieved chunks were relevant
        precision_at_k = correct_source_chunks / actual_k if actual_k > 0 else 0
        
        # Binary recall success (for compatibility with existing code)
        recall_success = recall_at_k > 0
        
        results[f"recall@{k}"] = {
            "success": recall_success,
            "hit_score": hit_at_k,
            "recall_score": recall_at_k,
            "precision_score": precision_at_k,
            "correct_source_chunks": correct_source_chunks,
            "found_golden_sources": len(found_golden_sources),
            "total_golden_sources": total_golden_sources,
            "matching_chunks": matching_chunks,
            "chunks_evaluated": actual_k,
            "golden_sources": list(normalized_golden_sources)
        }
    
    return results

# Load test queries with source information
print("Loading test queries...")
with open('/Users/choemanseung/789/hft/golden_answer_claude.json', 'r', encoding='utf-8') as file:
    test_queries = json.load(file)

# Source-based evaluation - no additional models needed

# Discover all configurations
print("Discovering configurations...")
configurations = discover_configurations()

print(f"\n{'='*100}")
print("RETRIEVAL EVALUATION WITH HIT@K METRICS (PRIMARY)")
print(f"{'='*100}")
print(f"Discovered {len(configurations)} configurations across all chunking strategies")
print(f"Testing {len(test_queries)} queries per configuration")
print(f"Total evaluations: {len(configurations) * len(test_queries)}")
print(f"Evaluation focus: Hit@5 (Primary), Hit@10, Recall@5, Recall@10")
print(f"{'='*100}\n")

# Store all results
all_results = {}
config_counter = 0
total_configs = len(configurations)

# Process each configuration
for config_name, config_data in configurations.items():
    config_counter += 1
    config = config_data["config"]
    
    print(f"\n{'='*80}")
    print(f"TESTING CONFIGURATION {config_counter}/{total_configs}: {config_name}")
    print(f"Strategy: {config.get('strategy', 'unknown')}")
    if 'chunk_size' in config:
        print(f"Chunk Size: {config['chunk_size']}, Overlap: {config['overlap']}")
    elif 'target_size' in config:
        print(f"Target Size: {config['target_size']}, Overlap Sentences: {config['overlap_sentences']}")
    elif 'min_length' in config:
        print(f"Min Length: {config['min_length']}")
    print(f"{'='*80}")
    
    try:
        # Initialize retriever
        retriever = MultiSourceRagRetriever(
            model_name=EMBEDDING_MODEL_NAME,
            index_paths=config_data["index_paths"],
            chunks_paths=config_data["chunks_paths"], 
            source_names=config_data["source_names"]
        )
        
        # Initialize results for this configuration
        config_results = {
            "config_name": config_name,
            "config": config,
            "query_results": {},
            "performance_metrics": {
                "total_queries": len(test_queries),
                "successful_retrievals": 0,
                "failed_retrievals": 0,
                "avg_response_time": 0,
                "total_response_time": 0,
                "hit_metrics": {
                    "hit@5": {"successes": 0, "total": 0, "avg_hit": 0},
                    "hit@10": {"successes": 0, "total": 0, "avg_hit": 0}
                },
                "recall_metrics": {
                    "recall@5": {"successes": 0, "total": 0, "avg_recall": 0},
                    "recall@10": {"successes": 0, "total": 0, "avg_recall": 0}
                }
            }
        }
        
        # Test each query
        for query_idx, test_case in enumerate(test_queries, 1):
            print(f"\n--- Query {query_idx}/{len(test_queries)}: {test_case['caseID']} ---")
            print(f"Query: '{test_case['query'][:100]}...'")
            print(f"Category: {test_case['healthify_topic']}")
            
            start_time = time.time()
            query_result = {
                "caseID": test_case['caseID'],
                "query": test_case['query'],
                "healthify_topic": test_case['healthify_topic'],
                "golden_answer": test_case['answer'],
                "success": False,
                "response_time": 0,
                "retrieved_chunks": [],
                "recall_results": {},
                "precision_results": {},
                "error": None
            }
            
            try:
                # Test different bias configurations for stratified sampling
                bias_configurations = [
                    {"name": "balanced", "healthify": 4, "mayo": 4, "nhs": 4},  # 4:4:4
                    {"name": "healthify_bias_1", "healthify": 6, "mayo": 3, "nhs": 3},  # 6:3:3
                    {"name": "healthify_bias_2", "healthify": 7, "mayo": 3, "nhs": 2},  # 7:3:2
                    {"name": "healthify_bias_3", "healthify": 8, "mayo": 2, "nhs": 2},  # 8:2:2
                    {"name": "healthify_bias_4", "healthify": 6, "mayo": 4, "nhs": 2},  # 6:4:2
                    {"name": "healthify_bias_5", "healthify": 5, "mayo": 4, "nhs": 3}   # 5:4:3
                ]
                
                # Test all bias configurations for this query
                query_result["bias_results"] = {}
                
                for bias_config in bias_configurations:
                    bias_name = bias_config["name"]
                    all_chunk_objects = []
                    
                    # Stratified sampling with bias toward Healthify
                    source_mapping = {
                        "Healthify": "healthify",
                        "Mayo": "mayo", 
                        "NHS": "nhs"
                    }
                    
                    for source_name, source_retriever in retriever.retrievers.items():
                        # Determine how many chunks to get from this source
                        source_key = source_mapping.get(source_name, source_name.lower())
                        num_chunks = bias_config.get(source_key, 4)
                        
                        # Perform direct search to get indices
                        query_embedding = source_retriever.model.encode([test_case['query']]).astype('float32')
                        distances, indices = source_retriever.index.search(query_embedding, num_chunks)
                        
                        # Get the chunk objects using the indices
                        for idx in indices[0]:
                            if idx < len(source_retriever.chunks):
                                chunk_obj = source_retriever.chunks[idx]
                                # Add source name for tracking
                                chunk_obj_with_source = chunk_obj.copy() if isinstance(chunk_obj, dict) else {"text": chunk_obj}
                                chunk_obj_with_source["retriever_source"] = source_name
                                chunk_obj_with_source["distance"] = distances[0][list(indices[0]).index(idx)]
                                all_chunk_objects.append(chunk_obj_with_source)
                    
                    # Sort by distance (lower is better/more similar)
                    all_chunk_objects.sort(key=lambda x: x.get('distance', float('inf')))
                    
                    # Evaluate recall for this bias configuration
                    bias_recall_results = evaluate_metrics_at_k(
                        all_chunk_objects, test_case['source']
                    )
                    
                    # Store results for this bias configuration
                    # Compute normalized source counts
                    norm_counts = {"healthify": 0, "mayo": 0, "nhs": 0}
                    for c in all_chunk_objects:
                        key = normalize_source_label(c.get('retriever_source', ''))
                        if key in norm_counts:
                            norm_counts[key] += 1

                    query_result["bias_results"][bias_name] = {
                        "recall_results": bias_recall_results,
                        "chunks_retrieved": len(all_chunk_objects),
                        "source_distribution": norm_counts
                    }
                
                # Use balanced configuration (4:4:4) for main evaluation and backward compatibility
                all_chunk_objects = []
                for source_name, source_retriever in retriever.retrievers.items():
                    query_embedding = source_retriever.model.encode([test_case['query']]).astype('float32')
                    distances, indices = source_retriever.index.search(query_embedding, 4)  # 4 from each
                    
                    for idx in indices[0]:
                        if idx < len(source_retriever.chunks):
                            chunk_obj = source_retriever.chunks[idx]
                            chunk_obj_with_source = chunk_obj.copy() if isinstance(chunk_obj, dict) else {"text": chunk_obj}
                            chunk_obj_with_source["retriever_source"] = source_name
                            chunk_obj_with_source["distance"] = distances[0][list(indices[0]).index(idx)]
                            all_chunk_objects.append(chunk_obj_with_source)
                
                # Sort by distance for main evaluation
                all_chunk_objects.sort(key=lambda x: x.get('distance', float('inf')))
                
                if all_chunk_objects:
                    query_result["success"] = True
                    query_result["retrieved_chunk_objects"] = all_chunk_objects[:10]  # Limit for storage
                    
                    # Evaluate Recall@K and Precision@K using source matching
                    recall_results = evaluate_metrics_at_k(
                        all_chunk_objects, test_case['source']
                    )
                    query_result["recall_results"] = recall_results
                    
                    # Print results
                    print(f"  ✓ Retrieved {len(all_chunk_objects)} chunks")
                    print(f"  Golden sources ({len(test_case['source'])}): {[s.split('/')[-1] for s in test_case['source']]}")
                    for k in [5, 10]:
                        if f"recall@{k}" in recall_results:
                            r = recall_results[f"recall@{k}"]
                            status = "✓" if r["success"] else "✗"
                            print(f"    @{k}: {status} Hit={r['hit_score']} R={r['recall_score']:.2f} ({r['found_golden_sources']}/{r['total_golden_sources']} sources)")
                    
                    config_results["performance_metrics"]["successful_retrievals"] += 1
                    
                    # Update aggregate metrics
                    for k in [5, 10]:
                        recall_key = f"recall@{k}"
                        if recall_key in recall_results:
                            # Update Hit@K metrics
                            hit_metrics = config_results["performance_metrics"]["hit_metrics"][f"hit@{k}"]
                            hit_metrics["total"] += 1
                            hit_score = recall_results[recall_key]["hit_score"]
                            hit_metrics["avg_hit"] += hit_score
                            if hit_score > 0:
                                hit_metrics["successes"] += 1
                            
                            # Update Recall@K metrics  
                            recall_metrics = config_results["performance_metrics"]["recall_metrics"][recall_key]
                            recall_metrics["total"] += 1
                            if recall_results[recall_key]["success"]:
                                recall_metrics["successes"] += 1
                            # Track average recall scores
                            recall_metrics["avg_recall"] += recall_results[recall_key]["recall_score"]
                
                else:
                    query_result["error"] = "No chunks retrieved"
                    print(f"  ✗ Failed - No chunks retrieved")
                    config_results["performance_metrics"]["failed_retrievals"] += 1
                    
            except Exception as e:
                query_result["error"] = str(e)
                print(f"  ✗ Exception - {str(e)}")
                config_results["performance_metrics"]["failed_retrievals"] += 1
            
            # Record response time
            end_time = time.time()
            query_result["response_time"] = end_time - start_time
            config_results["performance_metrics"]["total_response_time"] += query_result["response_time"]
            
            # Store query result
            config_results["query_results"][f"query_{query_idx}"] = query_result
        
        # Calculate final metrics for this configuration
        total_queries = len(test_queries)
        successful_queries = config_results["performance_metrics"]["successful_retrievals"]
        
        config_results["performance_metrics"]["avg_response_time"] = (
            config_results["performance_metrics"]["total_response_time"] / total_queries
        )
        
        # Calculate final metrics percentages and averages
        for k in [5, 10]:
            # Hit@K metrics
            hit_key = f"hit@{k}"
            hit_metrics = config_results["performance_metrics"]["hit_metrics"][hit_key]
            if hit_metrics["total"] > 0:
                hit_metrics["hit_percentage"] = (hit_metrics["successes"] / hit_metrics["total"]) * 100
                hit_metrics["avg_hit_rate"] = hit_metrics["avg_hit"] / hit_metrics["total"]
            else:
                hit_metrics["hit_percentage"] = 0
                hit_metrics["avg_hit_rate"] = 0
                
            # Recall@K metrics
            recall_key = f"recall@{k}"
            recall_metrics = config_results["performance_metrics"]["recall_metrics"][recall_key]
            if recall_metrics["total"] > 0:
                recall_metrics["recall_percentage"] = (recall_metrics["successes"] / recall_metrics["total"]) * 100
                recall_metrics["avg_recall_score"] = recall_metrics["avg_recall"] / recall_metrics["total"]
            else:
                recall_metrics["recall_percentage"] = 0
                recall_metrics["avg_recall_score"] = 0
        
        # Store configuration results
        all_results[config_name] = config_results
        
        # Print configuration summary
        print(f"\n--- CONFIGURATION {config_name} SUMMARY ---")
        print(f"Successful retrievals: {successful_queries}/{total_queries}")
        print(f"Average response time: {config_results['performance_metrics']['avg_response_time']:.3f}s")
        
        hit_metrics = config_results["performance_metrics"]["hit_metrics"]
        recall_metrics = config_results["performance_metrics"]["recall_metrics"]
        print("Performance Metrics:")
        for k in [5, 10]:
            hit_key = f"hit@{k}"
            recall_key = f"recall@{k}"
            hit_m = hit_metrics[hit_key]
            recall_m = recall_metrics[recall_key]
            print(f"  @{k}: Hit={hit_m['hit_percentage']:.1f}% R={recall_m['avg_recall_score']:.3f} ({recall_m['recall_percentage']:.1f}% success)")
        
    except Exception as e:
        print(f"CRITICAL ERROR with configuration {config_name}: {str(e)}")
        all_results[config_name] = {
            "config_name": config_name,
            "config": config,
            "error": str(e),
            "performance_metrics": {"successful_retrievals": 0, "failed_retrievals": total_queries}
        }

# Generate comprehensive summary report
print(f"\n{'='*100}")
print("COMPREHENSIVE RECALL@K EVALUATION COMPLETE - GENERATING REPORT")
print(f"{'='*100}")

# Analyze and rank configurations
valid_configs = []
strategy_performance = {
    "fixed": [], "sentence": [], "paragraph": [], 
    "agent": [], "agent_deduplicated": []
}

for config_name, results in all_results.items():
    if "error" not in results:
        metrics = results["performance_metrics"]
        config = results["config"]
        strategy = config.get("strategy", "unknown")
        
        # Use hit@5 as the primary ranking metric
        hit_metrics = metrics.get("hit_metrics", {})
        hit_5_score = 0
        
        if "hit@5" in hit_metrics:
            hit_5_score = hit_metrics["hit@5"].get("hit_percentage", 0)  # Already 0-100
        
        success_rate = metrics["successful_retrievals"] / metrics["total_queries"]
        avg_response_time = metrics["avg_response_time"]
        
        config_data = {
            "config_name": config_name,
            "strategy": strategy,
            "config": config,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "hit_5_score": hit_5_score,
            "hit_metrics": hit_metrics,
            "recall_metrics": metrics.get("recall_metrics", {}),
            "successful_retrievals": metrics["successful_retrievals"],
            "results": results
        }
        
        valid_configs.append(config_data)
        if strategy in strategy_performance:
            strategy_performance[strategy].append(config_data)

# Sort by hit@5 score (higher is better)
valid_configs.sort(key=lambda x: x["hit_5_score"], reverse=True)

# Sort each strategy's configs by hit@5 score
for strategy in strategy_performance:
    strategy_performance[strategy].sort(key=lambda x: x["hit_5_score"], reverse=True)

# Generate detailed summary file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_filename = f"retrieval_hit_evaluation_{timestamp}.txt"

with open(summary_filename, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("COMPREHENSIVE HIT@K RETRIEVAL EVALUATION SUMMARY\n") 
    f.write("=" * 100 + "\n")
    f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Configurations Tested: {len(valid_configs)}\n")
    f.write(f"Total Test Cases: {len(test_queries)}\n")
    f.write(f"Total Evaluations: {len(valid_configs) * len(test_queries)}\n")
    f.write(f"Embedding Model: {EMBEDDING_MODEL_NAME}\n")
    f.write(f"Evaluation Method: Source Document-based Hit@K (Primary), Recall@K (Secondary)\n")
    f.write(f"Evaluation Criteria: Hit@K measures binary success (≥1 relevant source found), Recall@K measures coverage\n")
    f.write("=" * 100 + "\n\n")
    
    # Strategy analysis
    f.write("ANALYSIS BY CHUNKING STRATEGY\n")
    f.write("=" * 50 + "\n\n")
    
    strategy_names = {
        "fixed": "Fixed-Length Chunking",
        "sentence": "Sentence-Based Chunking",
        "paragraph": "Paragraph-Based Chunking", 
        "agent": "AI Agent-Based Chunking",
        "agent_deduplicated": "AI Agent-Based Chunking (Deduplicated)"
    }
    
    best_configs_by_strategy = {}
    
    for strategy, configs in strategy_performance.items():
        if not configs:
            continue
            
        strategy_name = strategy_names.get(strategy, strategy.title())
        f.write(f"{strategy_name}\n")
        f.write("-" * len(strategy_name) + "\n")
        
        # Performance table
        f.write(f"{'Rank':<4} {'Config Name':<30} {'Success':<8} {'H@5':<8} {'H@10':<8} {'Response':<9}\n")
        f.write("-" * 65 + "\n")
        
        for rank, config in enumerate(configs[:10], 1):
            hit_metrics = config.get("hit_metrics", {})
            h5 = hit_metrics.get("hit@5", {}).get("hit_percentage", 0)
            h10 = hit_metrics.get("hit@10", {}).get("hit_percentage", 0)
            
            f.write(f"{rank:<4} {config['config_name'][:29]:<30} "
                   f"{config['success_rate']:.1%}     "
                   f"{h5:.1f}%    "
                   f"{h10:.1f}%    "
                   f"{config['avg_response_time']:.3f}s\n")
        
        # Best config for strategy
        if configs:
            best_config = configs[0]
            best_configs_by_strategy[strategy] = best_config
            
            f.write(f"\nBest Configuration: {best_config['config_name']}\n")
            f.write(f"  - Hit@5 Score: {best_config['hit_5_score']:.1f}%\n")
            
            best_hit_metrics = best_config.get("hit_metrics", {})
            best_recall_metrics = best_config.get("recall_metrics", {})
            for k in [5, 10]:
                hit_key = f"hit@{k}"
                recall_key = f"recall@{k}"
                if hit_key in best_hit_metrics:
                    hit_m = best_hit_metrics[hit_key]
                    f.write(f"  - @{k}: H={hit_m.get('hit_percentage', 0):.1f}%")
                if recall_key in best_recall_metrics:
                    recall_m = best_recall_metrics[recall_key]
                    f.write(f" R={recall_m.get('avg_recall_score', 0):.3f}")
                f.write("\n")
            
            f.write(f"  - Success Rate: {best_config['success_rate']:.1%}\n")
            f.write(f"  - Avg Response Time: {best_config['avg_response_time']:.3f}s\n")
        
        f.write("\n" + "=" * 40 + "\n\n")
    
    # Overall rankings
    f.write("OVERALL CONFIGURATION RANKINGS (Top 20)\n")
    f.write("=" * 45 + "\n")
    
    f.write(f"{'Rank':<4} {'Config Name':<35} {'Strategy':<12} {'H@5':<8} {'H@10':<8}\n")
    f.write("-" * 70 + "\n")
    
    for rank, config in enumerate(valid_configs[:20], 1):
        hit_metrics = config.get("hit_metrics", {})
        h5 = hit_metrics.get("hit@5", {}).get("hit_percentage", 0)
        h10 = hit_metrics.get("hit@10", {}).get("hit_percentage", 0)
        
        f.write(f"{rank:<4} {config['config_name'][:34]:<35} "
               f"{config['strategy'][:11]:<12} "
               f"{h5:.1f}%    "
               f"{h10:.1f}%\n")
    
    # Best overall configuration analysis
    if valid_configs:
        best_config = valid_configs[0]
        f.write(f"\n\nBEST OVERALL CONFIGURATION\n")
        f.write("=" * 30 + "\n")
        f.write(f"Winner: {best_config['config_name']}\n")
        f.write(f"Strategy: {strategy_names.get(best_config['strategy'], best_config['strategy'])}\n")
        f.write(f"Hit@5 Score: {best_config['hit_5_score']:.1f}%\n")
        
        # Configuration details
        config_details = best_config['config']
        if 'chunk_size' in config_details and 'overlap' in config_details:
            f.write(f"Chunk Size: {config_details['chunk_size']}, Overlap: {config_details['overlap']}\n")
        elif 'target_size' in config_details:
            f.write(f"Target Size: {config_details['target_size']}, Overlap Sentences: {config_details['overlap_sentences']}\n")
        elif 'min_length' in config_details:
            f.write(f"Min Length: {config_details['min_length']}\n")
        
        best_hit_metrics = best_config.get("hit_metrics", {})
        best_recall_metrics = best_config.get("recall_metrics", {})
        f.write("\nPerformance Summary:\n")
        for k in [5, 10]:
            hit_key = f"hit@{k}"
            recall_key = f"recall@{k}"
            if hit_key in best_hit_metrics:
                hit_m = best_hit_metrics[hit_key]
                f.write(f"  @{k}: H={hit_m.get('hit_percentage', 0):.1f}%")
            if recall_key in best_recall_metrics:
                recall_m = best_recall_metrics[recall_key]
                f.write(f" R={recall_m.get('avg_recall_score', 0):.3f}")
            f.write("\n")
        
        f.write(f"\nWhy this configuration excels:\n")
        f.write(f"- Highest Hit@5 score among all configurations\n")
        f.write(f"- Best at finding at least one relevant source in top 5 results\n")
        f.write(f"- Optimal for answerable queries where any golden source suffices\n")
    
    # Bias Configuration Analysis Summary
    f.write(f"\n\nBIAS CONFIGURATION ANALYSIS SUMMARY\n")
    f.write("=" * 40 + "\n")
    
    # Aggregate bias results across all configurations and queries
    if valid_configs:
        # Calculate average performance for each bias configuration across all queries
        bias_summary = {}
        total_queries_with_bias = 0
        
        for config in valid_configs[:5]:  # Top 5 configs
            config_results = config['results']
            for query_key, query_result in config_results["query_results"].items():
                if 'bias_results' in query_result:
                    total_queries_with_bias += 1
                    for bias_name, bias_data in query_result['bias_results'].items():
                        if bias_name not in bias_summary:
                            bias_summary[bias_name] = {
                                'r5_scores': [], 'r10_scores': [],
                                'total_healthify': 0, 'total_mayo': 0, 'total_nhs': 0
                            }
                        
                        recall_results = bias_data['recall_results']
                        source_dist = bias_data['source_distribution']
                        
                        bias_summary[bias_name]['r5_scores'].append(
                            recall_results.get('recall@5', {}).get('recall_score', 0)
                        )
                        bias_summary[bias_name]['r10_scores'].append(
                            recall_results.get('recall@10', {}).get('recall_score', 0)
                        )
                        bias_summary[bias_name]['total_healthify'] += source_dist['healthify']
                        bias_summary[bias_name]['total_mayo'] += source_dist['mayo']
                        bias_summary[bias_name]['total_nhs'] += source_dist['nhs']
        
        # Write bias configuration summary
        f.write(f"{'Config':<20} {'Avg R@5':<8} {'Avg R@10':<9} {'H:M:N Ratio':<15} {'Description':<25}\n")
        f.write("-" * 80 + "\n")
        
        bias_descriptions = {
            "balanced": "Equal representation",
            "healthify_bias_1": "Moderate Healthify bias", 
            "healthify_bias_2": "Strong Healthify bias",
            "healthify_bias_3": "Heavy Healthify bias",
            "healthify_bias_4": "Healthify + Mayo focus",
            "healthify_bias_5": "Balanced Healthify bias"
        }
        
        for bias_name, data in bias_summary.items():
            if data['r5_scores']:
                avg_r5 = sum(data['r5_scores']) / len(data['r5_scores'])
                avg_r10 = sum(data['r10_scores']) / len(data['r10_scores'])
                
                # Calculate average ratio per query
                num_queries = len(data['r5_scores'])
                avg_h = data['total_healthify'] / num_queries if num_queries > 0 else 0
                avg_m = data['total_mayo'] / num_queries if num_queries > 0 else 0
                avg_n = data['total_nhs'] / num_queries if num_queries > 0 else 0
                
                ratio_str = f"{avg_h:.1f}:{avg_m:.1f}:{avg_n:.1f}"
                description = bias_descriptions.get(bias_name, "Custom configuration")
                
                f.write(f"{bias_name:<20} {avg_r5:.3f}    {avg_r10:.3f}     {ratio_str:<15} {description:<25}\n")
        
        f.write(f"\nKey Insights:\n")
        f.write(f"- Total queries analyzed: {total_queries_with_bias // len(bias_summary) if bias_summary else 0}\n")
        f.write(f"- Bias configurations test different source sampling strategies\n")
        f.write(f"- H:M:N shows average Healthify:Mayo:NHS chunks per query\n")
        f.write(f"- Compare recall scores to identify optimal bias strategy\n\n")
    
    # Detailed per-query analysis for top configs (first 5)
    f.write(f"\n\nDETAILED QUERY-BY-QUERY ANALYSIS (Top 5 Configurations)\n")
    f.write("=" * 65 + "\n")
    
    for config_rank, config in enumerate(valid_configs[:5], 1):
        f.write(f"\nCONFIGURATION RANK {config_rank}: {config['config_name']}\n")
        f.write("-" * 50 + "\n")
        
        results = config['results']
        for query_key, query_result in results["query_results"].items():
            f.write(f"\n{query_result['caseID']} - {query_result['healthify_topic']}\n")
            f.write(f"Query: {query_result['query'][:100]}...\n")
            
            if query_result['success'] and 'recall_results' in query_result:
                recall_results = query_result['recall_results']
                f.write("Recall Results: ")
                for k in [5, 10]:
                    recall_key = f"recall@{k}"
                    if recall_key in recall_results:
                        r = recall_results[recall_key]
                        status = "✓" if r["success"] else "✗"
                        f.write(f"@{k}:{status}(R={r['recall_score']:.2f}) ")
                f.write(f"\nChunks Retrieved: {len(query_result.get('retrieved_chunk_objects', []))}\n")
                
                # Show retrieved chunks with their sources
                retrieved_chunks = query_result.get('retrieved_chunk_objects', [])
                if retrieved_chunks:
                    f.write("Retrieved Chunks:\n")
                    for i, chunk in enumerate(retrieved_chunks[:10], 1):  # Show top 10
                        chunk_source = chunk.get('source_document', 'Unknown')
                        retriever_source = chunk.get('retriever_source', 'Unknown')
                        chunk_text = get_chunk_display_text(chunk)
                        chunk_preview = (chunk_text[:150] + "...") if len(chunk_text) > 150 else chunk_text
                        
                        # Check if this chunk matches golden sources
                        is_relevant = False
                        if f"recall@5" in recall_results:
                            matching_chunks = recall_results[f"recall@5"].get('matching_chunks', [])
                            is_relevant = any(match.get('chunk_idx') == i-1 for match in matching_chunks)
                        
                        relevance_marker = " [RELEVANT]" if is_relevant else ""
                        f.write(f"  {i}. Source: {retriever_source}/{chunk_source}{relevance_marker}\n")
                        f.write(f"     Text: {chunk_preview}\n\n")
                
                # Show bias configuration results
                if 'bias_results' in query_result:
                    f.write("Bias Configuration Analysis:\n")
                    bias_results = query_result['bias_results']
                    for bias_name, bias_data in bias_results.items():
                        recall_results = bias_data['recall_results']
                        source_dist = bias_data['source_distribution']
                        
                        r5 = recall_results.get('recall@5', {}).get('recall_score', 0)
                        r10 = recall_results.get('recall@10', {}).get('recall_score', 0)
                        
                        f.write(f"  {bias_name}: R@5={r5:.2f} R@10={r10:.2f} ")
                        f.write(f"[H:{source_dist['healthify']} M:{source_dist['mayo']} N:{source_dist['nhs']}]\n")
                    f.write("\n")
            else:
                f.write(f"Error: {query_result.get('error', 'Unknown error')}\n")
    
    # Test cases summary
    f.write(f"\n\nTEST CASES EVALUATED\n")
    f.write("=" * 25 + "\n")
    f.write(f"{'Case ID':<15} {'Topic':<25} {'Triage':<8} {'Query Preview':<50}\n")
    f.write("-" * 100 + "\n")
    
    for test_case in test_queries:
        f.write(f"{test_case['caseID']:<15} "
               f"{test_case['healthify_topic'][:24]:<25} "
               f"{test_case['triage']:<8} "
               f"{test_case['query'][:49]:<50}\n")

print(f"\nDetailed evaluation report saved to: {summary_filename}")

# Print quick summary
print(f"\nQUICK SUMMARY:")
print(f"Configurations evaluated: {len(valid_configs)}")
if valid_configs:
    best = valid_configs[0]
    print(f"Best configuration: {best['config_name']}")
    print(f"Hit@5 score: {best['hit_5_score']:.1f}%")
    
    best_hit_metrics = best.get("hit_metrics", {})
    best_recall_metrics = best.get("recall_metrics", {})
    for k in [5, 10]:
        hit_key = f"hit@{k}"
        recall_key = f"recall@{k}"
        if hit_key in best_hit_metrics:
            hit_m = best_hit_metrics[hit_key]
            print(f"  @{k}: H={hit_m.get('hit_percentage', 0):.1f}%", end="")
        if recall_key in best_recall_metrics:
            recall_m = best_recall_metrics[recall_key]
            print(f" R={recall_m.get('avg_recall_score', 0):.3f}")
        else:
            print()
    
    print(f"Success rate: {best['success_rate']:.1%}")
    print(f"Average response time: {best['avg_response_time']:.3f}s")

print("=" * 100)
print("SOURCE-BASED HIT@K EVALUATION COMPLETE")
print("Focus: Retrieval success measurement - can queries be answered with retrieved sources?")
print("Primary Metric: Hit@5 and Hit@10 (binary success - ≥1 relevant source found)")
print("Secondary Metrics: Recall@5 and Recall@10 (source coverage percentage)")
print("=" * 100)