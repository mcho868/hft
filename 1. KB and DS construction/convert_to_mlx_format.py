#!/usr/bin/env python3
"""
Convert triage dialogues to MLX-compatible JSONL format.
Concatenates final_triage_decision, next_step, and reasoning into a completion field.
"""

import json
import os
from pathlib import Path

def convert_triage_to_mlx(input_file, output_file):
    """
    Convert triage dialogue JSON to MLX-compatible JSONL format.
    
    Args:
        input_file: Path to input JSON file with triage dialogues
        output_file: Path to output JSONL file
    """
    
    # Ensure output directory exists
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to MLX format
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            # Create prompt from symptom and query
            prompt = f"Patient presents with: {item['symptom']}\n\nPatient query: {item['query']}\n\nProvide triage decision, next steps, and reasoning:"
            
            # Concatenate completion fields
            completion = f"Triage Decision: {item['final_triage_decision']}\n\nNext Step: {item['next_step']}\n\nReasoning: {item['reasoning']}"
            
            # Create MLX-compatible entry
            mlx_entry = {
                "prompt": prompt,
                "completion": completion
            }
            
            # Write as JSONL (one JSON object per line)
            f.write(json.dumps(mlx_entry, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(data)} entries from {input_file} to {output_file}")

if __name__ == "__main__":
    # Convert all three splits
    base_input_dir = "/Users/choemanseung/789/hft/Final_dataset"
    base_output_dir = "/Users/choemanseung/789/hft/Final_dataset/triage_dialogues_mlx"
    
    splits = [
        ("simplified_triage_dialogues_train.json", "train.jsonl"),
        ("simplified_triage_dialogues_val.json", "valid.jsonl"), 
        ("simplified_triage_dialogues_test.json", "test.jsonl")
    ]
    
    for input_name, output_name in splits:
        input_file = f"{base_input_dir}/{input_name}"
        output_file = f"{base_output_dir}/{output_name}"
        convert_triage_to_mlx(input_file, output_file)