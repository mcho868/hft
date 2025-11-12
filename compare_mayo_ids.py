#!/usr/bin/env python3
import json

def compare_mayo_files():
    """Compare chunk_ids between mayo original and tinfoil files"""
    
    original_file = "/Users/choemanseung/789/hft/RAGdatav4/mayo_chunks_contextual_fixed_c512_o100.json"
    tinfoil_file = "/Users/choemanseung/789/hft/RAGdatav4/mayo_chunks_contextual_fixed_c512_o100_tinfoil.json"
    
    # Load both files
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(tinfoil_file, 'r', encoding='utf-8') as f:
        tinfoil_data = json.load(f)
    
    print(f"Original file entries: {len(original_data)}")
    print(f"Tinfoil file entries: {len(tinfoil_data)}")
    print()
    
    # Compare first few entries
    print("FIRST 5 ENTRIES COMPARISON:")
    print("=" * 80)
    
    for i in range(min(5, len(original_data), len(tinfoil_data))):
        orig = original_data[i]
        tinf = tinfoil_data[i]
        
        print(f"Entry {i}:")
        print(f"  Original chunk_id: {orig.get('chunk_id', 'NO_ID')}")
        print(f"  Tinfoil chunk_id:  {tinf.get('chunk_id', 'NO_ID')}")
        print(f"  Original source:   {orig.get('source_document', 'NO_SOURCE')}")
        print(f"  Tinfoil source:    {tinf.get('source_document', 'NO_SOURCE')}")
        print(f"  Same source?       {'✓' if orig.get('source_document') == tinf.get('source_document') else '✗'}")
        print(f"  Same text?         {'✓' if orig.get('original_text') == tinf.get('original_text') else '✗'}")
        print()
    
    # Analyze ID patterns
    print("CHUNK_ID PATTERNS:")
    print("=" * 80)
    
    orig_ids = [entry.get('chunk_id', '') for entry in original_data[:10]]
    tinf_ids = [entry.get('chunk_id', '') for entry in tinfoil_data[:10]]
    
    print("Original IDs (first 10):")
    for i, id in enumerate(orig_ids):
        print(f"  {i}: {id}")
    
    print("\nTinfoil IDs (first 10):")
    for i, id in enumerate(tinf_ids):
        print(f"  {i}: {id}")
    
    # Check if they're in different order
    print("\nORDER ANALYSIS:")
    print("=" * 80)
    
    # Extract the numeric parts of chunk_ids
    def extract_number(chunk_id):
        try:
            parts = chunk_id.split('_')
            return int(parts[-1]) if parts[-1].isdigit() else -1
        except:
            return -1
    
    orig_numbers = [extract_number(entry.get('chunk_id', '')) for entry in original_data[:20]]
    tinf_numbers = [extract_number(entry.get('chunk_id', '')) for entry in tinfoil_data[:20]]
    
    print("Original chunk numbers (first 20):", orig_numbers)
    print("Tinfoil chunk numbers (first 20):", tinf_numbers)
    
    # Check if it's just the naming convention
    print("\nNAMING CONVENTION:")
    print("=" * 80)
    
    orig_pattern = orig_ids[0].replace('0', 'X') if orig_ids else 'N/A'
    tinf_pattern = tinf_ids[0].replace('0', 'X') if tinf_ids else 'N/A'
    
    print(f"Original pattern: {orig_pattern}")
    print(f"Tinfoil pattern:  {tinf_pattern}")
    
    # Check if content is the same but IDs are different
    print("\nCONTENT COMPARISON:")
    print("=" * 80)
    
    same_content = 0
    for i in range(min(10, len(original_data), len(tinfoil_data))):
        orig_text = original_data[i].get('original_text', '')
        tinf_text = tinfoil_data[i].get('original_text', '')
        if orig_text == tinf_text:
            same_content += 1
    
    print(f"Out of first 10 entries, {same_content} have identical original_text")
    
    if same_content == 10:
        print("✓ Content appears to be identical, just IDs are different")
    elif same_content == 0:
        print("✗ Content is completely different")
    else:
        print("⚠️  Content is partially the same")

def main():
    print("Comparing mayo original vs tinfoil chunk_ids...")
    print("=" * 80)
    compare_mayo_files()

if __name__ == "__main__":
    main()