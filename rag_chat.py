#!/usr/bin/env python3

import os
import sys
import glob
import json
from retrieval_function import MultiSourceRagRetriever
import mlx.core as mx
from mlx_lm import load, generate

# Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
RAGDATA_PATH = '/Users/choemanseung/789/hft/RAGdatav2'
LLM_MODEL_PATH = '/Users/choemanseung/789/hft/mlx_models/SmolLM2-360M-Instruct-MLX'

def discover_chunking_configurations():
    """Discover available chunking configurations"""
    configurations = {}
    
    # Find all chunk files
    chunk_files = glob.glob(os.path.join(RAGDATA_PATH, '*_chunks_*.json'))
    
    for chunk_file in chunk_files:
        filename = os.path.basename(chunk_file)
        
        # Extract config name (remove source prefix and extensions)
        config_part = filename.replace('_chunks_', '_').replace('_vector_db_', '_')
        for source in ['nhs_', 'mayo_', 'healthify_']:
            config_part = config_part.replace(source, '')
        config_part = config_part.replace('.json', '').replace('.index', '')
        if config_part.startswith('_'):
            config_part = config_part[1:]
        
        # Find corresponding files for all sources
        index_files = []
        chunk_files_all = []
        
        for source in ['nhs', 'mayo', 'healthify']:
            index_file = os.path.join(RAGDATA_PATH, f'{source}_vector_db_{config_part}.index')
            chunk_file_source = os.path.join(RAGDATA_PATH, f'{source}_chunks_{config_part}.json')
            
            if os.path.exists(index_file) and os.path.exists(chunk_file_source):
                index_files.append(index_file)
                chunk_files_all.append(chunk_file_source)
        
        # Only include if we have files for all 3 sources
        if len(index_files) == 3 and len(chunk_files_all) == 3:
            configurations[config_part] = {
                "config_name": config_part,
                "index_paths": index_files,
                "chunks_paths": chunk_files_all,
                "source_names": ["NHS", "Mayo", "Healthify"]
            }
    
    return configurations

def select_configuration(configurations):
    """Let user select a chunking configuration"""
    print("\n" + "="*60)
    print("AVAILABLE CHUNKING CONFIGURATIONS")
    print("="*60)
    
    config_list = list(configurations.keys())
    for i, config_name in enumerate(config_list, 1):
        # Parse configuration type for display
        if config_name.startswith('fixed_'):
            parts = config_name.replace('fixed_', '').split('_')
            if len(parts) >= 2:
                chunk_size = parts[0][1:] if parts[0].startswith('c') else parts[0]
                overlap = parts[1][1:] if parts[1].startswith('o') else parts[1]
                display = f"Fixed chunking (size: {chunk_size}, overlap: {overlap})"
            else:
                display = "Fixed chunking"
        elif config_name.startswith('sentence_'):
            parts = config_name.replace('sentence_', '').split('_')
            if len(parts) >= 2:
                target = parts[0][1:] if parts[0].startswith('t') else parts[0]
                overlap = parts[1][1:] if parts[1].startswith('o') else parts[1]
                display = f"Sentence-based (target: {target}, overlap: {overlap})"
            else:
                display = "Sentence-based chunking"
        elif config_name.startswith('paragraph_'):
            parts = config_name.replace('paragraph_', '').split('_')
            if len(parts) >= 1:
                min_len = parts[0][1:] if parts[0].startswith('m') else parts[0]
                display = f"Paragraph-based (min length: {min_len})"
            else:
                display = "Paragraph-based chunking"
        elif config_name.startswith('agent_'):
            if 'deduplicated' in config_name:
                display = "AI Agent-based chunking (deduplicated)"
            else:
                display = "AI Agent-based chunking"
        else:
            display = config_name
        
        print(f"{i:2d}. {config_name:<30} - {display}")
    
    print("\nSelect a configuration by number:")
    while True:
        try:
            choice = input("Enter number (1-{}): ".format(len(config_list)))
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(config_list):
                selected_config = config_list[choice_idx]
                print(f"\nSelected: {selected_config}")
                return configurations[selected_config]
            else:
                print(f"Please enter a number between 1 and {len(config_list)}")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)

def retrieve_with_bias(retriever, query, bias_config="balanced"):
    """Retrieve chunks with specified bias configuration"""
    bias_configurations = {
        "balanced": {"healthify": 4, "mayo": 4, "nhs": 4},
        "healthify_bias": {"healthify": 6, "mayo": 3, "nhs": 3},
        "healthify_strong": {"healthify": 8, "mayo": 2, "nhs": 2},
    }
    
    config = bias_configurations.get(bias_config, bias_configurations["balanced"])
    all_chunk_objects = []
    
    # Source mapping
    source_mapping = {
        "Healthify": "healthify",
        "Mayo": "mayo", 
        "NHS": "nhs"
    }
    
    # Stratified sampling
    for source_name, source_retriever in retriever.retrievers.items():
        source_key = source_mapping.get(source_name, source_name.lower())
        num_chunks = config.get(source_key, 4)
        
        # Perform search
        query_embedding = source_retriever.model.encode([query]).astype('float32')
        distances, indices = source_retriever.index.search(query_embedding, num_chunks)
        
        # Get chunks
        for idx in indices[0]:
            if idx < len(source_retriever.chunks):
                chunk_obj = source_retriever.chunks[idx]
                chunk_obj_with_source = chunk_obj.copy() if isinstance(chunk_obj, dict) else {"text": chunk_obj}
                chunk_obj_with_source["retriever_source"] = source_name
                chunk_obj_with_source["distance"] = distances[0][list(indices[0]).index(idx)]
                all_chunk_objects.append(chunk_obj_with_source)
    
    # Sort by similarity (lower distance = higher similarity)
    all_chunk_objects.sort(key=lambda x: x.get('distance', float('inf')))
    
    return all_chunk_objects

def format_context(chunks, max_chunks=5):
    """Format retrieved chunks into context for LLM"""
    context_parts = []
    
    for i, chunk in enumerate(chunks[:max_chunks]):
        source = chunk.get('retriever_source', 'Unknown')
        source_doc = chunk.get('source_document', 'Unknown')
        text = chunk.get('text', '')
        
        context_parts.append(f"Source {i+1} ({source} - {source_doc}):\n{text}\n")
    
    return "\n".join(context_parts)

def create_prompt(user_query, context):
    """Create prompt for the LLM"""
    return f"""<|im_start|>system
You are a helpful medical assistant. Use the provided medical information sources to answer the user's health question. Be accurate, helpful, and cite which sources you're using. If you are confused between two potential topics, (like heart attack or indigestion, always choose the more urgent one), and determine if user should stay home, visit GP or go to ED.  If you're unsure or if the question requires immediate medical attention, advise the user to consult a healthcare professional.

Medical Information Sources:
{context}
<|im_end|>
<|im_start|>user
{user_query}
<|im_end|>
<|im_start|>assistant
"""

def main():
    print("🏥 Medical RAG Chat System")
    print("Powered by SmolLM2-360M")
    
    # Discover configurations
    print("\nDiscovering chunking configurations...")
    configurations = discover_chunking_configurations()
    
    if not configurations:
        print("❌ No valid configurations found! Check RAGdatav2 directory.")
        sys.exit(1)
    
    print(f"Found {len(configurations)} configurations")
    
    # Let user select configuration
    selected_config = select_configuration(configurations)
    
    # Initialize retriever
    print(f"\n🔧 Initializing retriever with configuration: {selected_config['config_name']}")
    try:
        retriever = MultiSourceRagRetriever(
            model_name=EMBEDDING_MODEL_NAME,
            index_paths=selected_config["index_paths"],
            chunks_paths=selected_config["chunks_paths"], 
            source_names=selected_config["source_names"]
        )
        print("✅ Retriever initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize retriever: {e}")
        sys.exit(1)
    
    # Load LLM
    print(f"\n🤖 Loading SmolLM2 model from {LLM_MODEL_PATH}")
    try:
        model, tokenizer = load(LLM_MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Chat loop
    print("\n" + "="*60)
    print("🩺 MEDICAL RAG CHAT - Ready!")
    print("="*60)
    print("Commands:")
    print("  - Type your medical question")
    print("  - '/bias balanced' - Use balanced retrieval (4:4:4)")
    print("  - '/bias healthify' - Use Healthify bias (6:3:3)")  
    print("  - '/bias strong' - Use strong Healthify bias (8:2:2)")
    print("  - '/quit' - Exit")
    print("="*60)
    
    bias_setting = "balanced"
    
    while True:
        try:
            user_input = input(f"\n[{bias_setting}] 🩺 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['/quit', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            # Handle bias commands
            if user_input.startswith('/bias '):
                new_bias = user_input.split(' ', 1)[1].lower()
                if new_bias in ['balanced', 'healthify', 'strong']:
                    bias_setting = new_bias
                    bias_map = {
                        'balanced': 'balanced (4:4:4)',
                        'healthify': 'healthify_bias (6:3:3)',  
                        'strong': 'healthify_strong (8:2:2)'
                    }
                    print(f"🔧 Bias setting changed to: {bias_map[new_bias]}")
                else:
                    print("❌ Invalid bias. Use: balanced, healthify, or strong")
                continue
            
            print("🔍 Retrieving relevant information...")
            
            # Retrieve relevant chunks
            bias_map = {
                'balanced': 'balanced',
                'healthify': 'healthify_bias',
                'strong': 'healthify_strong'
            }
            chunks = retrieve_with_bias(retriever, user_input, bias_map[bias_setting])
            
            if not chunks:
                print("❌ No relevant information found.")
                continue
            
            # Show retrieved sources
            print(f"📚 Retrieved {len(chunks)} chunks from:")
            source_counts = {}
            for chunk in chunks:
                source = chunk.get('retriever_source', 'Unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            
            for source, count in source_counts.items():
                print(f"  - {source}: {count} chunks")
            
            # Format context and create prompt
            context = format_context(chunks)
            print("context:  ", context)

            prompt = create_prompt(user_input, context)
            
            print("\n🤖 Assistant: ", end="", flush=True)
            
            # Generate response
            try:    
                response = generate(
                    model, tokenizer, prompt,
                    max_tokens=500,
                    verbose=False
                )
                print(response)
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()