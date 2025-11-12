#!/usr/bin/env python3
import json
from collections import Counter, defaultdict

def analyze_duplicates():
    """Analyze duplicates in the healthify original file"""
    
    original_file = "/Users/choemanseung/789/hft/RAGdatav4/healthify_chunks_contextual_fixed_c512_o100.json"
    tinfoil_file = "/Users/choemanseung/789/hft/RAGdatav4/healthify_chunks_contextual_fixed_c512_o100_tinfoil.json"
    
    # Load both files
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(tinfoil_file, 'r', encoding='utf-8') as f:
        tinfoil_data = json.load(f)
    
    print(f"Original entries: {len(original_data)}")
    print(f"Tinfoil entries: {len(tinfoil_data)}")
    print(f"Difference: {len(original_data) - len(tinfoil_data)}")
    print()
    
    def make_content_key(entry):
        return (
            entry.get('source_document', ''),
            entry.get('source_char_start', -1),
            entry.get('source_char_end', -1)
        )
    
    # Analyze duplicates in original file
    print("ORIGINAL FILE DUPLICATE ANALYSIS:")
    print("=" * 50)
    
    original_keys = [make_content_key(entry) for entry in original_data]
    original_key_counts = Counter(original_keys)
    
    duplicates_in_orig = {key: count for key, count in original_key_counts.items() if count > 1}
    
    print(f"Total content keys in original: {len(original_keys)}")
    print(f"Unique content keys in original: {len(original_key_counts)}")
    print(f"Duplicate content keys: {len(duplicates_in_orig)}")
    
    if duplicates_in_orig:
        total_extra_entries = sum(count - 1 for count in duplicates_in_orig.values())
        print(f"Total extra duplicate entries: {total_extra_entries}")
        print()
        
        print("Examples of duplicated content:")
        for i, (key, count) in enumerate(list(duplicates_in_orig.items())[:5]):
            source, start, end = key
            print(f"  {i+1}. {source}:{start}-{end} appears {count} times")
            
            # Find the actual entries
            matching_entries = []
            for j, entry in enumerate(original_data):
                if make_content_key(entry) == key:
                    matching_entries.append((j, entry))
            
            print(f"     Indices: {[idx for idx, _ in matching_entries]}")
            print(f"     Chunk IDs: {[entry.get('chunk_id', 'N/A') for _, entry in matching_entries]}")
            print()
    
    # Analyze tinfoil file
    print("TINFOIL FILE DUPLICATE ANALYSIS:")
    print("=" * 50)
    
    tinfoil_keys = [make_content_key(entry) for entry in tinfoil_data]
    tinfoil_key_counts = Counter(tinfoil_keys)
    
    duplicates_in_tinf = {key: count for key, count in tinfoil_key_counts.items() if count > 1}
    
    print(f"Total content keys in tinfoil: {len(tinfoil_keys)}")
    print(f"Unique content keys in tinfoil: {len(tinfoil_key_counts)}")
    print(f"Duplicate content keys: {len(duplicates_in_tinf)}")
    
    if duplicates_in_tinf:
        total_extra_entries = sum(count - 1 for count in duplicates_in_tinf.values())
        print(f"Total extra duplicate entries: {total_extra_entries}")
    
    # Compare the two files
    print("\nCOMPARISON:")
    print("=" * 50)
    
    # Find which duplicates were removed
    removed_duplicates = set(duplicates_in_orig.keys()) - set(duplicates_in_tinf.keys())
    kept_duplicates = set(duplicates_in_orig.keys()) & set(duplicates_in_tinf.keys())
    
    print(f"Duplicates removed in tinfoil: {len(removed_duplicates)}")
    print(f"Duplicates kept in tinfoil: {len(kept_duplicates)}")
    
    if removed_duplicates:
        print("\nExamples of duplicates that were removed:")
        for i, key in enumerate(list(removed_duplicates)[:3]):
            source, start, end = key
            orig_count = duplicates_in_orig[key]
            print(f"  {i+1}. {source}:{start}-{end} (was {orig_count} copies, now 1)")
    
    # Calculate expected difference
    expected_reduction = sum(duplicates_in_orig[key] - 1 for key in removed_duplicates)
    actual_difference = len(original_data) - len(tinfoil_data)
    
    print(f"\nExpected reduction from deduplication: {expected_reduction}")
    print(f"Actual difference: {actual_difference}")
    
    if expected_reduction == actual_difference:
        print("✅ The difference is exactly explained by duplicate removal!")
    else:
        print(f"⚠️ Difference mismatch: expected {expected_reduction}, got {actual_difference}")

def main():
    print("Analyzing duplicates in healthify files...")
    print("=" * 70)
    analyze_duplicates()

if __name__ == "__main__":
    main()