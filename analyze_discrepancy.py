#!/usr/bin/env python3
import json
from collections import Counter

def analyze_files(original_file, tinfoil_file):
    """Analyze the discrepancy between the two files"""
    
    # Load both files
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(tinfoil_file, 'r', encoding='utf-8') as f:
        tinfoil_data = json.load(f)
    
    print(f"Original file entries: {len(original_data)}")
    print(f"Tinfoil file entries: {len(tinfoil_data)}")
    print(f"Difference: {len(original_data) - len(tinfoil_data)}")
    print()
    
    # Extract chunk_ids
    original_ids = [entry.get('chunk_id', 'NO_ID') for entry in original_data]
    tinfoil_ids = [entry.get('chunk_id', 'NO_ID') for entry in tinfoil_data]
    
    # Check for duplicates within each file
    original_dupes = [id for id, count in Counter(original_ids).items() if count > 1]
    tinfoil_dupes = [id for id, count in Counter(tinfoil_ids).items() if count > 1]
    
    print(f"Duplicate chunk_ids in original: {len(original_dupes)}")
    if original_dupes[:5]:
        print(f"  Examples: {original_dupes[:5]}")
    
    print(f"Duplicate chunk_ids in tinfoil: {len(tinfoil_dupes)}")
    if tinfoil_dupes[:5]:
        print(f"  Examples: {tinfoil_dupes[:5]}")
    print()
    
    # Convert to sets for comparison
    original_set = set(original_ids)
    tinfoil_set = set(tinfoil_ids)
    
    print(f"Unique chunk_ids in original: {len(original_set)}")
    print(f"Unique chunk_ids in tinfoil: {len(tinfoil_set)}")
    print()
    
    # Find missing entries
    missing_in_tinfoil = original_set - tinfoil_set
    extra_in_tinfoil = tinfoil_set - original_set
    
    print(f"Chunk_ids missing from tinfoil: {len(missing_in_tinfoil)}")
    print(f"Chunk_ids only in tinfoil: {len(extra_in_tinfoil)}")
    print()
    
    if missing_in_tinfoil:
        print("Sample missing chunk_ids:")
        for id in list(missing_in_tinfoil)[:10]:
            print(f"  {id}")
        print()
    
    if extra_in_tinfoil:
        print("Sample extra chunk_ids in tinfoil:")
        for id in list(extra_in_tinfoil)[:10]:
            print(f"  {id}")
        print()
    
    # Analyze patterns in missing IDs
    if missing_in_tinfoil:
        print("Analysis of missing chunk_ids:")
        # Extract the numeric part to see if there's a pattern
        missing_numbers = []
        for id in missing_in_tinfoil:
            try:
                # Extract number from end of chunk_id
                parts = id.split('_')
                if parts[-1].isdigit():
                    missing_numbers.append(int(parts[-1]))
            except:
                pass
        
        if missing_numbers:
            missing_numbers.sort()
            print(f"  Missing numbers range: {min(missing_numbers)} to {max(missing_numbers)}")
            print(f"  Total missing numbers: {len(missing_numbers)}")
            
            # Check if it's a contiguous range
            if len(missing_numbers) > 1:
                gaps = []
                for i in range(1, len(missing_numbers)):
                    gap = missing_numbers[i] - missing_numbers[i-1]
                    if gap > 1:
                        gaps.append((missing_numbers[i-1], missing_numbers[i], gap))
                
                if not gaps:
                    print("  ✓ Missing numbers form a contiguous range")
                else:
                    print(f"  Gaps in missing numbers: {len(gaps)}")
                    for start, end, gap in gaps[:5]:
                        print(f"    Gap from {start} to {end} (size: {gap})")
    
    # Check source documents
    original_sources = set(entry.get('source_document', 'NO_SOURCE') for entry in original_data)
    tinfoil_sources = set(entry.get('source_document', 'NO_SOURCE') for entry in tinfoil_data)
    
    print(f"\nSource documents in original: {len(original_sources)}")
    print(f"Source documents in tinfoil: {len(tinfoil_sources)}")
    
    missing_sources = original_sources - tinfoil_sources
    if missing_sources:
        print(f"Source documents missing from tinfoil: {len(missing_sources)}")
        for src in list(missing_sources)[:5]:
            print(f"  {src}")

def main():
    original_file = "/Users/choemanseung/789/hft/RAGdatav4/mayo_chunks_contextual_fixed_c512_o100.json"
    tinfoil_file = "/Users/choemanseung/789/hft/RAGdatav4/mayo_chunks_contextual_fixed_c512_o100_tinfoil.json"
    
    print("Analyzing discrepancy between files...")
    print("=" * 60)
    
    analyze_files(original_file, tinfoil_file)

if __name__ == "__main__":
    main()