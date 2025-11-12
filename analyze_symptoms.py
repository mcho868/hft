#!/usr/bin/env python3
import json
from collections import Counter

def analyze_symptoms(file_path):
    """Analyze symptoms distribution in the JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    symptoms = []
    for case in data:
        symptom = case['symptom'].split(' (Case')[0]  # Extract symptom name without case number
        symptoms.append(symptom)
    
    symptom_counts = Counter(symptoms)
    
    print(f"File: {file_path}")
    print(f"Total cases: {len(data)}")
    print(f"Unique symptoms: {len(symptom_counts)}")
    print("\nSymptom distribution:")
    for symptom, count in sorted(symptom_counts.items()):
        print(f"  {symptom}: {count}")
    
    return symptom_counts, data

if __name__ == "__main__":
    print("=== ORIGINAL DATA ===")
    original_symptoms, original_data = analyze_symptoms('/Users/choemanseung/789/hft/generated_triage_dialogues.json')
    
    print("\n=== CURRENT TRAIN DATA ===")
    train_symptoms, _ = analyze_symptoms('/Users/choemanseung/789/hft/generated_triage_dialogues_train.json')
    
    print("\n=== CURRENT VAL DATA ===")
    val_symptoms, _ = analyze_symptoms('/Users/choemanseung/789/hft/generated_triage_dialogues_val.json')
    
    print(f"\nTotal original cases: {len(original_data)}")
    total_split = len(original_data) // 7  # For 5:1:1 ratio
    print(f"Target distribution (5:1:1):")
    print(f"  Train: {5 * total_split} cases")
    print(f"  Test: {total_split} cases") 
    print(f"  Val: {total_split} cases")