#!/usr/bin/env python3
"""
Dataset Transformation Script
Transforms triage dialogue datasets into simplified single-turn format
"""

import json
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from typing import List, Dict, Any

# Add the project directory to the Python path for imports
sys.path.append('/Users/choemanseung/789/hft')
from mlx_models.tinfoilAgent import TinfoilAgent

# Global lock for thread-safe operations
output_lock = Lock()
progress_lock = Lock()

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON dataset from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} entries from {file_path}")
        return data
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []

def create_single_turn_query(agent: TinfoilAgent, patient_query: str, clarifying_question: str, patient_response: str) -> str:
    """Combine the three dialogue parts into a single comprehensive patient statement"""
    # Remove surrounding quotes if present
    patient_query = patient_query.strip('"')
    clarifying_question = clarifying_question.strip('"')
    patient_response = patient_response.strip('"')
    
    prompt = f"""Given the following dialogue between a patient and a nurse, create a single comprehensive patient statement that includes all the information from both the initial patient query and the patient's response to the clarifying question. The result should be a natural, single statement from the patient that provides all relevant details upfront.

Original patient query: {patient_query}

Nurse's clarifying question: {clarifying_question}

Patient's response: {patient_response}

Please combine all the information into one cohesive patient statement that sounds natural and includes all relevant details. Start with just the patient's words (no "Patient:" prefix needed):"""

    try:
        response = agent.getResponse(prompt)
        if response:
            return response.strip()
        else:
            # Fallback: simple combination
            return f"{patient_query} Additionally, {patient_response.lower()}"
    except Exception as e:
        print(f"⚠️ Error creating single turn query: {e}")
        return f"{patient_query} Additionally, {patient_response.lower()}"

def generate_new_reasoning(agent: TinfoilAgent, entry: Dict[str, Any], single_turn_query: str) -> str:
    """Generate new combined reasoning using Tinfoil API"""
    
    prompt = f"""Given the following patient statement and triage decision, provide a very brief reasoning (maximum 2 sentences) for why this triage decision ({entry['final_triage_decision']}) is appropriate:

Patient Statement: {single_turn_query}

Triage Decision: {entry['final_triage_decision']}
Next Step: {entry['next_step']}

Please provide a concise reasoning (maximum 2 sentences) that explains why this triage decision is appropriate given the patient's symptoms.

Reasoning:"""

    try:
        response = agent.getResponse(prompt)
        if response:
            return response.strip()
        else:
            # Fallback reasoning if API fails
            return f"Based on the patient's symptoms and responses, a {entry['final_triage_decision']} referral is appropriate for proper assessment and management."
    except Exception as e:
        print(f"⚠️ Error generating reasoning for ID {entry['id']}: {e}")
        return f"Based on the patient's symptoms and responses, a {entry['final_triage_decision']} referral is appropriate for proper assessment and management."

def transform_entry(agent: TinfoilAgent, entry: Dict[str, Any], progress_counter: List[int], total: int) -> Dict[str, Any]:
    """Transform a single entry into the new format"""
    try:
        # Create single turn query
        single_turn_query = create_single_turn_query(
            agent,
            entry['patient_query'], 
            entry['clarifying_question'], 
            entry['patient_response']
        )
        
        # Generate new reasoning
        new_reasoning = generate_new_reasoning(agent, entry, single_turn_query)
        
        # Create transformed entry
        transformed = {
            "id": entry["id"],
            "symptom": entry["symptom"],
            "query": single_turn_query,
            "final_triage_decision": entry["final_triage_decision"],
            "next_step": entry["next_step"],
            "reasoning": new_reasoning
        }
        
        # Thread-safe progress update
        with progress_lock:
            progress_counter[0] += 1
            if progress_counter[0] % 10 == 0:
                print(f"📊 Progress: {progress_counter[0]}/{total} entries processed ({progress_counter[0]/total*100:.1f}%)")
        
        return transformed
        
    except Exception as e:
        print(f"❌ Error transforming entry ID {entry.get('id', 'unknown')}: {e}")
        return None

def process_dataset_parallel(dataset: List[Dict[str, Any]], dataset_name: str, num_workers: int = 200) -> List[Dict[str, Any]]:
    """Process dataset entries in parallel using ThreadPoolExecutor"""
    print(f"\n🔄 Processing {dataset_name} with {num_workers} workers...")
    
    transformed_entries = []
    progress_counter = [0]  # Use list to allow modification in nested function
    total = len(dataset)
    
    # Create agent instances for each worker
    agents = [TinfoilAgent("llama3-3-70b") for _ in range(num_workers)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_entry = {}
        for i, entry in enumerate(dataset):
            agent = agents[i % num_workers]  # Round-robin agent assignment
            future = executor.submit(transform_entry, agent, entry, progress_counter, total)
            future_to_entry[future] = entry
        
        # Collect results as they complete
        for future in as_completed(future_to_entry):
            try:
                result = future.result()
                if result:
                    with output_lock:
                        transformed_entries.append(result)
            except Exception as e:
                entry = future_to_entry[future]
                print(f"❌ Worker error for entry ID {entry.get('id', 'unknown')}: {e}")
    
    print(f"✅ Completed processing {dataset_name}: {len(transformed_entries)}/{total} entries successfully transformed")
    return transformed_entries

def save_dataset(data: List[Dict[str, Any]], output_path: str):
    """Save transformed dataset to JSON file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(data)} entries to {output_path}")
    except Exception as e:
        print(f"❌ Error saving to {output_path}: {e}")

def main():
    """Main execution function"""
    print("🚀 Dataset Transformation Script")
    print("=" * 60)
    
    # Input file paths
    input_files = [
        "/Users/choemanseung/789/hft/Final_dataset/generated_triage_dialogues_test_new.json",
        "/Users/choemanseung/789/hft/Final_dataset/generated_triage_dialogues_val_new.json", 
        "/Users/choemanseung/789/hft/Final_dataset/generated_triage_dialogues_train_new.json"
    ]
    
    # Output file paths
    output_files = [
        "/Users/choemanseung/789/hft/Final_dataset/simplified_triage_dialogues_test.json",
        "/Users/choemanseung/789/hft/Final_dataset/simplified_triage_dialogues_val.json",
        "/Users/choemanseung/789/hft/Final_dataset/simplified_triage_dialogues_train.json"
    ]
    
    dataset_names = ["Test", "Validation", "Training"]
    
    total_start_time = time.time()
    
    for input_file, output_file, dataset_name in zip(input_files, output_files, dataset_names):
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name} Dataset")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print(f"{'='*80}")
        
        # Check if output already exists
        if os.path.exists(output_file):
            print(f"⏭️ Skipping {dataset_name}: Output file already exists")
            continue
        
        start_time = time.time()
        
        # Load dataset
        dataset = load_dataset(input_file)
        if not dataset:
            print(f"⚠️ Skipping {dataset_name}: No data to process")
            continue
        
        # Process dataset in parallel
        transformed_data = process_dataset_parallel(dataset, dataset_name, num_workers=200)
        
        # Save results
        if transformed_data:
            save_dataset(transformed_data, output_file)
        
        end_time = time.time()
        print(f"⏱️ {dataset_name} processing completed in {end_time - start_time:.2f} seconds")
    
    total_end_time = time.time()
    print(f"\n{'='*80}")
    print("🎉 ALL DATASETS PROCESSED SUCCESSFULLY")
    print(f"⏱️ Total processing time: {total_end_time - total_start_time:.2f} seconds")
    print(f"{'='*80}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n🛑 Interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ An unexpected error occurred: {e}")
        sys.exit(1)