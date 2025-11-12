#!/usr/bin/env python3
"""
AI-powered Medical Conditions Filter using TinfoilAgent
Splits the medical conditions file into chunks of 100 and sends to AI agent for filtering.
"""

import os
import sys
import re
import time
import json
from typing import List, Set, Dict, Any
from pathlib import Path

# Add the RAGdatav3 directory to the Python path for TinfoilAgent imports
sys.path.append('/Users/choemanseung/789/hft')

# Import TinfoilAgent
try:
    from mlx_models.tinfoilAgent import TinfoilAgent
    TINFOIL_AVAILABLE = True
except ImportError:
    print("❌ TinfoilAgent not available. Please ensure it's properly installed.")
    TINFOIL_AVAILABLE = False
    sys.exit(1)


def load_medical_conditions(file_path: str) -> List[str]:
    """Load medical conditions from the numbered list file."""
    conditions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Extract condition from numbered format: "   123. Condition Name"
            match = re.match(r'\s*\d+\.\s+(.+)', line.strip())
            if match:
                conditions.append(match.group(1).strip())
    return conditions


def create_filtering_prompt(conditions_batch: List[str]) -> str:
    """Create a prompt for the AI agent to filter medical conditions."""
    conditions_text = "\n".join([f"{i+1}. {condition}" for i, condition in enumerate(conditions_batch)])
    
    prompt = f"""You are a medical expert tasked with filtering a list of terms to keep only legitimate medical conditions, diseases, disorders, and syndromes.

FILTER OUT:
- Medication names (drugs like amoxicillin, metformin, etc.)
- Medical procedures (surgeries, tests like CT scan, MRI, etc.)
- Medical equipment and devices
- Generic topic categories (like "Cancer Topics", "Diabetes Topics")
- Treatment methods and therapies
- Diagnostic tools and tests
- Vaccines and immunizations
- Medical specialties and departments
- Very short or incomplete terms (less than 3 meaningful characters)

KEEP ONLY:
- Legitimate medical conditions, diseases, disorders, and syndromes
- Specific illnesses and health problems
- Mental health conditions
- Genetic conditions
- Chronic diseases
- Acute conditions
- Injuries and trauma (if they represent medical conditions)

Here are the terms to filter (numbered list):

{conditions_text}

INSTRUCTIONS:
1. Review each term carefully
2. Return ONLY the numbers (from the list above) of terms that are legitimate medical conditions
3. Separate the numbers with commas
4. Do NOT include any explanations, just the numbers
5. If none qualify, return "NONE"

Example response format: 1,3,7,12,15,18,22

Numbers of legitimate medical conditions:"""

    return prompt


def call_tinfoil_agent(prompt: str, model_name: str = "llama3-3-70b") -> str:
    """Call TinfoilAgent API to filter medical conditions."""
    try:
        agent = TinfoilAgent(model_name)
        response = agent.getResponse(prompt)
        return response if response else ""
    except Exception as e:
        print(f"❌ Error calling TinfoilAgent: {e}")
        return ""


def parse_agent_response(response: str) -> List[int]:
    """Parse the agent's response to extract selected condition numbers."""
    if not response or "NONE" in response.upper():
        return []
    
    # Clean the response
    response = response.strip()
    
    # Extract numbers from the response
    numbers = []
    # Look for comma-separated numbers
    if "," in response:
        parts = response.split(",")
        for part in parts:
            # Extract number from each part
            match = re.search(r'\d+', part.strip())
            if match:
                numbers.append(int(match.group()))
    else:
        # Look for space-separated or single numbers
        matches = re.findall(r'\d+', response)
        numbers = [int(match) for match in matches]
    
    return sorted(list(set(numbers)))  # Remove duplicates and sort


def chunk_conditions(conditions: List[str], chunk_size: int = 100) -> List[List[str]]:
    """Split conditions into chunks of specified size."""
    chunks = []
    for i in range(0, len(conditions), chunk_size):
        chunk = conditions[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def save_progress(progress_file: str, processed_chunks: int, filtered_conditions: Set[str]):
    """Save progress to a file."""
    progress_data = {
        "processed_chunks": processed_chunks,
        "filtered_conditions": list(filtered_conditions),
        "timestamp": time.time()
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)


def load_progress(progress_file: str) -> tuple[int, Set[str]]:
    """Load progress from file if it exists."""
    if not os.path.exists(progress_file):
        return 0, set()
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        
        processed_chunks = progress_data.get("processed_chunks", 0)
        filtered_conditions = set(progress_data.get("filtered_conditions", []))
        return processed_chunks, filtered_conditions
    except Exception as e:
        print(f"⚠️  Error loading progress: {e}")
        return 0, set()


def main():
    """Main function to filter medical conditions using AI agent."""
    if not TINFOIL_AVAILABLE:
        print("❌ TinfoilAgent is required for this script")
        return
    
    # File paths
    input_file = "/Users/choemanseung/789/hft/unique_medical_conditions.txt"
    output_file = "/Users/choemanseung/789/hft/ai_filtered_medical_conditions.txt"
    progress_file = "/Users/choemanseung/789/hft/ai_filtering_progress.json"
    detailed_log_file = "/Users/choemanseung/789/hft/ai_filtering_detailed_log.txt"
    
    print("🤖 AI-Powered Medical Conditions Filter")
    print("=" * 50)
    
    # Load medical conditions
    print("📋 Loading medical conditions...")
    conditions = load_medical_conditions(input_file)
    print(f"Loaded {len(conditions)} medical conditions")
    
    # Load existing progress
    processed_chunks_count, filtered_conditions = load_progress(progress_file)
    
    # Split into chunks
    chunk_size = 100
    condition_chunks = chunk_conditions(conditions, chunk_size)
    total_chunks = len(condition_chunks)
    
    print(f"📦 Split into {total_chunks} chunks of {chunk_size} conditions each")
    
    if processed_chunks_count > 0:
        print(f"🔄 Resuming from chunk {processed_chunks_count + 1} (found {len(filtered_conditions)} conditions so far)")
    
    # Process chunks
    start_time = time.time()
    
    with open(detailed_log_file, 'w', encoding='utf-8') as log_file:
        log_file.write("AI Medical Conditions Filtering - Detailed Log\n")
        log_file.write("=" * 55 + "\n\n")
        log_file.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total conditions: {len(conditions)}\n")
        log_file.write(f"Total chunks: {total_chunks}\n")
        log_file.write(f"Chunk size: {chunk_size}\n\n")
        
        for chunk_idx in range(processed_chunks_count, total_chunks):
            chunk = condition_chunks[chunk_idx]
            
            print(f"\n📤 Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk)} conditions)")
            log_file.write(f"--- Chunk {chunk_idx + 1}/{total_chunks} ---\n")
            log_file.write(f"Conditions in chunk: {len(chunk)}\n")
            
            # Create prompt
            prompt = create_filtering_prompt(chunk)
            
            # Call AI agent
            print("🤖 Calling TinfoilAgent...")
            response = call_tinfoil_agent(prompt)
            
            if not response:
                print("❌ No response from agent, skipping chunk")
                log_file.write("ERROR: No response from agent\n\n")
                continue
            
            # Parse response
            selected_numbers = parse_agent_response(response)
            print(f"✅ Agent selected {len(selected_numbers)} conditions from chunk")
            
            log_file.write(f"Agent response: {response}\n")
            log_file.write(f"Selected numbers: {selected_numbers}\n")
            log_file.write(f"Selected conditions:\n")
            
            # Add selected conditions to filtered set
            for num in selected_numbers:
                if 1 <= num <= len(chunk):  # Validate number is in range
                    condition = chunk[num - 1]  # Convert to 0-based index
                    filtered_conditions.add(condition)
                    log_file.write(f"  {num}. {condition}\n")
                else:
                    print(f"⚠️  Invalid selection number: {num}")
                    log_file.write(f"  INVALID: {num}\n")
            
            log_file.write(f"Total unique conditions so far: {len(filtered_conditions)}\n\n")
            
            # Save progress
            save_progress(progress_file, chunk_idx + 1, filtered_conditions)
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time_per_chunk = elapsed / (chunk_idx - processed_chunks_count + 1)
            remaining_chunks = total_chunks - (chunk_idx + 1)
            eta = remaining_chunks * avg_time_per_chunk
            
            print(f"⏱️  Progress: {chunk_idx + 1}/{total_chunks} chunks")
            print(f"📊 Total filtered conditions: {len(filtered_conditions)}")
            print(f"🕒 ETA: {eta/60:.1f} minutes")
            
            # Small delay to be nice to the API
            time.sleep(1)
    
    # Convert to sorted list and save final results
    final_conditions = sorted(list(filtered_conditions))
    
    # Save filtered conditions
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("AI-Filtered Medical Conditions\n")
        f.write("=" * 32 + "\n\n")
        f.write(f"Original count: {len(conditions)}\n")
        f.write(f"Filtered count: {len(final_conditions)}\n")
        f.write(f"Filtered out: {len(conditions) - len(final_conditions)}\n")
        f.write(f"Filter rate: {((len(conditions) - len(final_conditions)) / len(conditions) * 100):.1f}%\n\n")
        
        for i, condition in enumerate(final_conditions, 1):
            f.write(f"{i:4d}. {condition}\n")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n🎉 AI Filtering Complete!")
    print(f"📋 Original conditions: {len(conditions)}")
    print(f"✅ Filtered conditions: {len(final_conditions)}")
    print(f"❌ Filtered out: {len(conditions) - len(final_conditions)}")
    print(f"📊 Filter rate: {((len(conditions) - len(final_conditions)) / len(conditions) * 100):.1f}%")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"💾 Results saved to: {output_file}")
    print(f"📝 Detailed log: {detailed_log_file}")
    
    # Clean up progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print("🧹 Cleaned up progress file")
    
    # Show sample of filtered conditions
    print(f"\n📋 Sample of AI-filtered conditions:")
    print("-" * 40)
    for i, condition in enumerate(final_conditions[:20], 1):
        print(f"{i:2d}. {condition}")
    
    if len(final_conditions) > 20:
        print(f"... and {len(final_conditions) - 20} more")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n🛑 Interrupted by user. Progress has been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ An unexpected error occurred: {e}")
        sys.exit(1)