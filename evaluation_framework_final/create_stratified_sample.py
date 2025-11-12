#!/usr/bin/env python3
"""
Create a stratified 200-case sample maintaining exact triage distribution
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

def create_stratified_sample():
    """Create 200-case stratified sample"""
    
    # Load full validation data
    data_file = Path("/Users/choemanseung/789/hft/Final_dataset/simplified_triage_dialogues_val.json")
    with open(data_file, 'r') as f:
        full_data = json.load(f)
    
    print(f"Loaded {len(full_data)} total cases")
    
    # Group by triage decision
    grouped = defaultdict(list)
    for case in full_data:
        grouped[case['final_triage_decision']].append(case)
    
    # Target distribution for 200 cases
    target_counts = {
        'GP': 152,    # 76.2%
        'ED': 36,     # 17.9% 
        'HOME': 12    # 6.0%
    }
    
    # Sample from each group
    stratified_sample = []
    
    for decision, target_count in target_counts.items():
        available_cases = grouped[decision]
        print(f"{decision}: {len(available_cases)} available, sampling {target_count}")
        
        if len(available_cases) >= target_count:
            sampled = random.sample(available_cases, target_count)
        else:
            print(f"Warning: Only {len(available_cases)} {decision} cases available")
            sampled = available_cases
        
        stratified_sample.extend(sampled)
    
    # Shuffle the combined sample
    random.shuffle(stratified_sample)
    
    print(f"\nCreated stratified sample with {len(stratified_sample)} cases")
    
    # Verify distribution
    from collections import Counter
    sample_counts = Counter(case['final_triage_decision'] for case in stratified_sample)
    print(f"Sample distribution:")
    for decision, count in sample_counts.items():
        percentage = (count / len(stratified_sample)) * 100
        print(f"  {decision}: {count} cases ({percentage:.1f}%)")
    
    # Save stratified sample
    output_file = Path("/Users/choemanseung/789/hft/evaluation_framework_final/stratified_sample_200.json")
    with open(output_file, 'w') as f:
        json.dump(stratified_sample, f, indent=2)
    
    print(f"\nSaved to: {output_file}")
    return stratified_sample

if __name__ == "__main__":
    create_stratified_sample()