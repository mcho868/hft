#!/usr/bin/env python3
"""
Modular RAG Chunking Script v4
Uses modular components from RAGdatav3/scripts package
"""

import os
import sys
import time

# Add the RAGdatav3 directory to the Python path for imports
sys.path.append('/Users/choemanseung/789/hft')

# Import new chunker classes and utility functions
from RAGdatav3.scripts import (
    # New chunker classes
    FixedLengthChunker, SentenceBasedChunker, ParagraphBasedChunker,
    AgentBasedChunker, StructuredAgentChunker, ContextualRetrievalChunker,
    # Utility functions
    save_text_chunks
)

# --- Configuration ---
DATA_DIR = ["RAGdatav3/nhs", "RAGdatav3/mayo", "RAGdatav3/healthify"]

TEST_CONFIGS = [
    (256, 0), (512, 0), (768, 0), (256, 50), (384, 50), (384, 100),
    (448, 112), (512, 100), (512, 150), (640, 64), (768, 100), (768, 200), (1024, 150)
]

SENTENCE_CONFIGS = [
    (384, 0), (512, 0), (768, 0), (384, 1), (384, 2), (512, 1), (512, 2),
    (512, 3), (768, 1), (768, 2), (768, 3), (1024, 2), (1024, 3)
]

PARAGRAPH_CONFIGS = [25, 50, 100]

AGENT_PROMPTS = [
    "You are an expert medical writer creating a knowledge base for a retrieval system. Your task is to break down the following document into a series of self-contained, semantically complete chunks. Each chunk must contain a single, coherent piece of medical advice, a description of a symptom, or a specific instruction. Preserve all critical details, such as symptoms, dosages, and emergency warnings. Do not summarize away important information. Present the output as a numbered list.",
]

# Best performing configs for contextual retrieval
CONTEXTUAL_CONFIGS = [
    ('fixed', 512, 100),  # fixed_c512_o100
    ('sentence', 1024, 2)  # sentence_t1024_o2
]


def main():
    """Main execution function to build all chunking strategies"""
    
    start_time = time.time()
    total_databases_created = 0
    
    print("🚀 RAG Chunking Script v4 (Modular)")
    print("=" * 50)
    print("Select chunking strategies to build:")
    print("1. Fixed-length chunking")
    print("2. Sentence-based chunking")
    print("3. Paragraph-based chunking")
    print("4. Agent-based chunking (uses LM Studio API or TinfoilAgent API)")
    print("5. Structured agent chunking (extracts medical data)")
    print("6. Contextual retrieval chunking (Anthropic method)")
    print("7. All strategies")
    
    choice = input("Enter your choice (1-7): ").strip()
    
    strategies_to_run = []
    if choice in ["1", "7"]: strategies_to_run.append("fixed")
    if choice in ["2", "7"]: strategies_to_run.append("sentence")
    if choice in ["3", "7"]: strategies_to_run.append("paragraph")
    if choice in ["4", "7"]: strategies_to_run.append("agent")
    if choice in ["5", "7"]: strategies_to_run.append("structured")
    if choice in ["6", "7"]: strategies_to_run.append("contextual")
    
    if not strategies_to_run:
        print("Invalid choice. Exiting.")
        return
    
    # Ask for API choice if any agentic methods are selected
    use_tinfoil = False
    if any(strategy in strategies_to_run for strategy in ["agent", "structured", "contextual"]):
        print("\nSelect API for agentic methods:")
        print("1. LM Studio (local)")
        print("2. TinfoilAgent (batch processing)")
        api_choice = input("Enter your choice (1-2): ").strip()
        use_tinfoil = (api_choice == "2")
    
    # Ensure output directory exists
    os.makedirs("RAGdatav4", exist_ok=True)
    
    for data_dir in DATA_DIR:
            
        print(f"\n{'='*80}\nProcessing data directory: {data_dir}\n{'='*80}")
        base_name = data_dir.split('/')[-1]  # e.g., 'nhs', 'mayo', 'healthify'
        
        if "fixed" in strategies_to_run:
            print(f"\n{'='*60}\nFIXED-LENGTH CHUNKING\n{'='*60}")
            for config_idx, (size, overlap) in enumerate(TEST_CONFIGS):
                    
                config_name = f"fixed_c{size}_o{overlap}"
                print(f"\n--- Fixed Config {config_idx + 1}/{len(TEST_CONFIGS)}: {config_name} ---")
                chunks_filename = f"RAGdatav4/{base_name}_chunks_{config_name}.json"
                if os.path.exists(chunks_filename):
                    print(f"⏭️  Skipping: File already exists")
                    continue
                chunker = FixedLengthChunker(data_dir, base_name, config_name, size, overlap)
                text_chunks = chunker.chunk()
                if text_chunks:
                    save_text_chunks(text_chunks, chunks_filename)
                    total_databases_created += 1

        if "sentence" in strategies_to_run:
            print(f"\n{'='*60}\nSENTENCE-BASED CHUNKING\n{'='*60}")
            for config_idx, (size, overlap) in enumerate(SENTENCE_CONFIGS):
                    
                config_name = f"sentence_t{size}_o{overlap}"
                print(f"\n--- Sentence Config {config_idx + 1}/{len(SENTENCE_CONFIGS)}: {config_name} ---")
                chunks_filename = f"RAGdatav4/{base_name}_chunks_{config_name}.json"
                if os.path.exists(chunks_filename):
                    print(f"⏭️  Skipping: File already exists")
                    continue
                chunker = SentenceBasedChunker(data_dir, base_name, config_name, size, overlap)
                text_chunks = chunker.chunk()
                if text_chunks:
                    save_text_chunks(text_chunks, chunks_filename)
                    total_databases_created += 1

        if "paragraph" in strategies_to_run:
            print(f"\n{'='*60}\nPARAGRAPH-BASED CHUNKING\n{'='*60}")
            for config_idx, min_len in enumerate(PARAGRAPH_CONFIGS):
                    
                config_name = f"paragraph_m{min_len}"
                print(f"\n--- Paragraph Config {config_idx + 1}/{len(PARAGRAPH_CONFIGS)}: {config_name} ---")
                chunks_filename = f"RAGdatav4/{base_name}_chunks_{config_name}.json"
                if os.path.exists(chunks_filename):
                    print(f"⏭️  Skipping: File already exists")
                    continue
                chunker = ParagraphBasedChunker(data_dir, base_name, config_name, min_len)
                text_chunks = chunker.chunk()
                if text_chunks:
                    save_text_chunks(text_chunks, chunks_filename)
                    total_databases_created += 1

        if "agent" in strategies_to_run:
            print(f"\n{'='*60}\nAGENT-BASED CHUNKING\n{'='*60}")
            for config_idx, prompt in enumerate(AGENT_PROMPTS):
                    
                prompt_key = prompt.lower().replace(" ", "_")[:20]
                api_name = "tinfoil" if use_tinfoil else "lmstudio"
                config_name = f"agent_{api_name}_qwen3_4b_{config_idx + 1}_{prompt_key}"
                print(f"\n--- Agent Config {config_idx + 1}/{len(AGENT_PROMPTS)}: {config_name} ---")
                chunks_filename = f"RAGdatav4/{base_name}_chunks_{config_name}.json"
                if os.path.exists(chunks_filename):
                    print(f"⏭️  Skipping: File already exists")
                    continue
                chunker = AgentBasedChunker(data_dir, base_name, config_name, prompt, use_tinfoil)
                text_chunks = chunker.chunk()
                if text_chunks:
                    save_text_chunks(text_chunks, chunks_filename)
                    total_databases_created += 1

        if "structured" in strategies_to_run:
            print(f"\n{'='*60}\nSTRUCTURED AGENT-BASED CHUNKING\n{'='*60}")
            api_name = "tinfoil" if use_tinfoil else "qwen3_4b"
            config_name = f"structured_agent_{api_name}_medical"
            print(f"\n--- Structured Agent Config: {config_name} ---")
            chunks_filename = f"RAGdatav4/{base_name}_chunks_{config_name}.json"
            if os.path.exists(chunks_filename):
                print(f"⏭️  Skipping: File already exists")
            else:
                chunker = StructuredAgentChunker(data_dir, base_name, config_name, use_tinfoil)
                text_chunks = chunker.chunk()
                if text_chunks:
                    save_text_chunks(text_chunks, chunks_filename)
                    total_databases_created += 1

        if "contextual" in strategies_to_run:
            print(f"\n{'='*60}\nCONTEXTUAL RETRIEVAL CHUNKING\n{'='*60}")
            for config_idx, (chunk_type, size, overlap) in enumerate(CONTEXTUAL_CONFIGS):
                    
                api_suffix = "_tinfoil" if use_tinfoil else ""
                config_name = f"contextual_{chunk_type}_c{size}_o{overlap}{api_suffix}"
                print(f"\n--- Contextual Config {config_idx + 1}/{len(CONTEXTUAL_CONFIGS)}: {config_name} ---")
                chunks_filename = f"RAGdatav4/{base_name}_chunks_{config_name}.json"
                if os.path.exists(chunks_filename):
                    print(f"⏭️  Skipping: File already exists")
                    continue
                chunker = ContextualRetrievalChunker(data_dir, base_name, config_name, chunk_type, size, overlap, use_tinfoil)
                text_chunks = chunker.chunk()
                if text_chunks:
                    save_text_chunks(text_chunks, chunks_filename)
                    total_databases_created += 1

    end_time = time.time()
    print(f"\n{'='*80}\nALL CHUNKING STRATEGIES COMPLETED\n{'='*80}")
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")
    print(f"Total chunk files created in this run: {total_databases_created}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n🛑 Interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ An unexpected error occurred: {e}")
        sys.exit(1)