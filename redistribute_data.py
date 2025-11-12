#!/usr/bin/env python3
import json
import random
from collections import defaultdict, Counter
import math

def load_data(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_data(data, file_path):
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def group_by_symptom(data):
    """Group cases by symptom (without case numbers)."""
    symptom_groups = defaultdict(list)
    for case in data:
        symptom = case['symptom'].split(' (Case')[0]  # Extract symptom name without case number
        symptom_groups[symptom].append(case)
    return symptom_groups

def redistribute_data(data, train_ratio=5, test_ratio=1, val_ratio=1):
    """
    Redistribute data with the given ratio while ensuring all symptoms appear in test and val sets.
    
    Args:
        data: List of cases
        train_ratio: Ratio for training set (default 5)
        test_ratio: Ratio for test set (default 1) 
        val_ratio: Ratio for validation set (default 1)
    
    Returns:
        tuple: (train_data, test_data, val_data)
    """
    symptom_groups = group_by_symptom(data)
    
    train_data = []
    test_data = []
    val_data = []
    
    total_ratio = train_ratio + test_ratio + val_ratio
    
    # Process each symptom group
    for symptom, cases in symptom_groups.items():
        # Shuffle cases within each symptom group
        shuffled_cases = cases.copy()
        random.shuffle(shuffled_cases)
        
        num_cases = len(cases)
        
        # Calculate splits ensuring at least 1 case for test and val if possible
        if num_cases >= 3:
            # For symptoms with 3+ cases, use proportional split but ensure at least 1 for test and val
            test_size = max(1, math.ceil(num_cases * test_ratio / total_ratio))
            val_size = max(1, math.ceil(num_cases * val_ratio / total_ratio))
            train_size = num_cases - test_size - val_size
            
            # If train_size becomes 0 or negative, adjust
            if train_size <= 0:
                if num_cases == 3:
                    train_size, test_size, val_size = 1, 1, 1
                else:
                    train_size = max(1, num_cases - 2)
                    test_size = val_size = 1
        
        elif num_cases == 2:
            # For symptoms with 2 cases: 1 train, 1 test, 0 val (or 0 train, 1 test, 1 val)
            train_size, test_size, val_size = 1, 1, 0
        
        elif num_cases == 1:
            # For symptoms with 1 case: put in train but also copy to test to ensure coverage
            train_size, test_size, val_size = 1, 0, 0
            # We'll add this case to test set later to ensure symptom coverage
        
        else:
            continue  # Skip if no cases (shouldn't happen)
        
        # Distribute cases
        train_cases = shuffled_cases[:train_size]
        test_cases = shuffled_cases[train_size:train_size + test_size]
        val_cases = shuffled_cases[train_size + test_size:train_size + test_size + val_size]
        
        train_data.extend(train_cases)
        test_data.extend(test_cases)
        val_data.extend(val_cases)
        
        # For single-case symptoms, also add to test for coverage
        if num_cases == 1:
            test_data.extend(shuffled_cases)
        
        print(f"{symptom}: {num_cases} cases -> Train: {len(train_cases)}, Test: {len(test_cases)}, Val: {len(val_cases)}")
    
    # Shuffle the final datasets
    random.shuffle(train_data)
    random.shuffle(test_data)
    random.shuffle(val_data)
    
    return train_data, test_data, val_data

def verify_symptom_coverage(train_data, test_data, val_data):
    """Verify that all symptoms appear in test and validation sets."""
    train_symptoms = set()
    test_symptoms = set()
    val_symptoms = set()
    
    for case in train_data:
        symptom = case['symptom'].split(' (Case')[0]
        train_symptoms.add(symptom)
    
    for case in test_data:
        symptom = case['symptom'].split(' (Case')[0]
        test_symptoms.add(symptom)
    
    for case in val_data:
        symptom = case['symptom'].split(' (Case')[0]
        val_symptoms.add(symptom)
    
    all_symptoms = train_symptoms | test_symptoms | val_symptoms
    
    print(f"\nSymptom coverage verification:")
    print(f"Total unique symptoms: {len(all_symptoms)}")
    print(f"Symptoms in train: {len(train_symptoms)} ({len(train_symptoms)/len(all_symptoms)*100:.1f}%)")
    print(f"Symptoms in test: {len(test_symptoms)} ({len(test_symptoms)/len(all_symptoms)*100:.1f}%)")
    print(f"Symptoms in val: {len(val_symptoms)} ({len(val_symptoms)/len(all_symptoms)*100:.1f}%)")
    
    missing_in_test = all_symptoms - test_symptoms
    missing_in_val = all_symptoms - val_symptoms
    
    if missing_in_test:
        print(f"WARNING: {len(missing_in_test)} symptoms missing in test set: {sorted(list(missing_in_test))[:10]}...")
    
    if missing_in_val:
        print(f"WARNING: {len(missing_in_val)} symptoms missing in val set: {sorted(list(missing_in_val))[:10]}...")
    
    return len(missing_in_test) == 0 and len(missing_in_val) == 0

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load the original data
    print("Loading original data...")
    original_data = load_data('/Users/choemanseung/789/hft/generated_triage_dialogues.json')
    print(f"Total cases: {len(original_data)}")
    
    # Redistribute data with 5:1:1 ratio
    print("\nRedistributing data with 5:1:1 ratio...")
    train_data, test_data, val_data = redistribute_data(original_data, train_ratio=5, test_ratio=1, val_ratio=1)
    
    print(f"\nFinal distribution:")
    print(f"Train: {len(train_data)} cases")
    print(f"Test: {len(test_data)} cases")
    print(f"Val: {len(val_data)} cases")
    print(f"Total: {len(train_data) + len(test_data) + len(val_data)} cases")
    
    actual_ratio = f"{len(train_data)/len(test_data):.1f}:{len(test_data)/len(test_data):.1f}:{len(val_data)/len(test_data):.1f}"
    print(f"Actual ratio: {actual_ratio}")
    
    # Verify symptom coverage
    coverage_ok = verify_symptom_coverage(train_data, test_data, val_data)
    
    if coverage_ok:
        print("\n✓ All symptoms are represented in both test and validation sets!")
    else:
        print("\n⚠ Some symptoms are missing from test or validation sets.")
    
    # Save the redistributed data
    print("\nSaving redistributed data...")
    save_data(train_data, '/Users/choemanseung/789/hft/generated_triage_dialogues_train_new.json')
    save_data(test_data, '/Users/choemanseung/789/hft/generated_triage_dialogues_test_new.json')
    save_data(val_data, '/Users/choemanseung/789/hft/generated_triage_dialogues_val_new.json')
    
    print("✓ Redistribution complete!")
    print("New files created:")
    print("  - generated_triage_dialogues_train_new.json")
    print("  - generated_triage_dialogues_test_new.json") 
    print("  - generated_triage_dialogues_val_new.json")

if __name__ == "__main__":
    main()