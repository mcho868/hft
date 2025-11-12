#!/usr/bin/env python3
import json

def get_missing_sources(original_file, tinfoil_file):
    """Get all source documents missing from tinfoil version"""
    
    # Load both files
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(tinfoil_file, 'r', encoding='utf-8') as f:
        tinfoil_data = json.load(f)
    
    # Get unique source documents from each file
    original_sources = set(entry.get('source_document', 'NO_SOURCE') for entry in original_data)
    tinfoil_sources = set(entry.get('source_document', 'NO_SOURCE') for entry in tinfoil_data)
    
    # Find missing sources
    missing_sources = original_sources - tinfoil_sources
    common_sources = original_sources & tinfoil_sources
    
    print(f"Total source documents in original: {len(original_sources)}")
    print(f"Total source documents in tinfoil: {len(tinfoil_sources)}")
    print(f"Common source documents: {len(common_sources)}")
    print(f"Missing from tinfoil: {len(missing_sources)}")
    print()
    
    print("Source documents missing from tinfoil version:")
    print("=" * 60)
    
    # Sort the missing sources for better readability
    sorted_missing = sorted(missing_sources)
    
    for i, source in enumerate(sorted_missing, 1):
        print(f"{i:4d}. {source}")
    
    # Also save to a file for easy reference
    with open('/Users/choemanseung/789/hft/missing_sources.txt', 'w') as f:
        f.write("Source documents missing from tinfoil version:\n")
        f.write("=" * 60 + "\n\n")
        for i, source in enumerate(sorted_missing, 1):
            f.write(f"{i:4d}. {source}\n")
    
    print(f"\nList also saved to: /Users/choemanseung/789/hft/missing_sources.txt")

def main():
    original_file = "/Users/choemanseung/789/hft/RAGdatav4/mayo_chunks_contextual_fixed_c512_o100.json"
    tinfoil_file = "/Users/choemanseung/789/hft/RAGdatav4/mayo_chunks_contextual_fixed_c512_o100_tinfoil.json"
    
    get_missing_sources(original_file, tinfoil_file)

if __name__ == "__main__":
    main()