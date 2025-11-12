#!/usr/bin/env python3
import json
from collections import Counter, defaultdict

def check_healthify_consistency():
    """Check consistency between healthify original and tinfoil files"""
    
    original_file = "/Users/choemanseung/789/hft/RAGdatav4/healthify_chunks_contextual_fixed_c512_o100.json"
    tinfoil_file = "/Users/choemanseung/789/hft/RAGdatav4/healthify_chunks_contextual_fixed_c512_o100_tinfoil.json"
    
    print("Checking healthify file consistency...")
    print("=" * 70)
    
    # Check if original file exists first
    try:
        with open(original_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Original file not found: {original_file}")
        return None, None
    except Exception as e:
        print(f"❌ Error reading original file: {e}")
        return None, None
    
    # Load tinfoil file
    try:
        with open(tinfoil_file, 'r', encoding='utf-8') as f:
            tinfoil_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Tinfoil file not found: {tinfoil_file}")
        return original_data, None
    except Exception as e:
        print(f"❌ Error reading tinfoil file: {e}")
        return original_data, None
    
    print(f"Original file entries: {len(original_data)}")
    print(f"Tinfoil file entries: {len(tinfoil_data)}")
    print()
    
    # Check source documents
    orig_sources = set(entry.get('source_document', '') for entry in original_data)
    tinf_sources = set(entry.get('source_document', '') for entry in tinfoil_data)
    
    orig_sources.discard('')
    tinf_sources.discard('')
    
    print(f"Original unique sources: {len(orig_sources)}")
    print(f"Tinfoil unique sources: {len(tinf_sources)}")
    print(f"Common sources: {len(orig_sources & tinf_sources)}")
    print(f"Missing from tinfoil: {len(orig_sources - tinf_sources)}")
    print(f"Extra in tinfoil: {len(tinf_sources - orig_sources)}")
    print()
    
    # Check chunk_id patterns
    print("CHUNK_ID ANALYSIS:")
    print("=" * 50)
    
    # Sample original IDs
    orig_sample = [entry.get('chunk_id', '') for entry in original_data[:10]]
    tinf_sample = [entry.get('chunk_id', '') for entry in tinfoil_data[:10]]
    
    print("Original sample IDs:")
    for i, id in enumerate(orig_sample):
        print(f"  {i}: {id}")
    
    print("\nTinfoil sample IDs:")
    for i, id in enumerate(tinf_sample):
        print(f"  {i}: {id}")
    
    # Check for duplicates in both files
    print("\nDUPLICATE ANALYSIS:")
    print("=" * 50)
    
    orig_chunk_ids = [entry.get('chunk_id', '') for entry in original_data]
    tinf_chunk_ids = [entry.get('chunk_id', '') for entry in tinfoil_data]
    
    orig_dupes = [id for id, count in Counter(orig_chunk_ids).items() if count > 1]
    tinf_dupes = [id for id, count in Counter(tinf_chunk_ids).items() if count > 1]
    
    print(f"Original duplicates: {len(orig_dupes)}")
    print(f"Tinfoil duplicates: {len(tinf_dupes)}")
    
    if tinf_dupes:
        print(f"Sample tinfoil duplicates: {tinf_dupes[:5]}")
    
    # Check content matching
    print("\nCONTENT MATCHING:")
    print("=" * 50)
    
    # Create lookup from original
    orig_lookup = {}
    for entry in original_data:
        key = (
            entry.get('source_document', ''),
            entry.get('source_char_start', -1),
            entry.get('source_char_end', -1)
        )
        orig_lookup[key] = entry
    
    # Check how many tinfoil entries have matches
    matches = 0
    perfect_matches = 0
    
    for tinf_entry in tinfoil_data:
        key = (
            tinf_entry.get('source_document', ''),
            tinf_entry.get('source_char_start', -1),
            tinf_entry.get('source_char_end', -1)
        )
        
        if key in orig_lookup:
            matches += 1
            orig_entry = orig_lookup[key]
            
            # Check if text content is identical
            orig_text = orig_entry.get('original_text', '')
            tinf_text = tinf_entry.get('original_text', '')
            
            if orig_text == tinf_text:
                perfect_matches += 1
    
    print(f"Tinfoil entries with matching source/position: {matches}")
    print(f"Perfect content matches: {perfect_matches}")
    print(f"Match rate: {matches/len(tinfoil_data)*100:.1f}%")
    
    # Determine if files are consistent
    print("\nCONSISTENCY ASSESSMENT:")
    print("=" * 50)
    
    is_consistent = True
    issues = []
    
    if len(tinf_dupes) > 0:
        is_consistent = False
        issues.append(f"Tinfoil has {len(tinf_dupes)} duplicate chunk_ids")
    
    if matches < len(tinfoil_data) * 0.9:  # Less than 90% match
        is_consistent = False
        issues.append(f"Low content matching: {matches/len(tinfoil_data)*100:.1f}%")
    
    if len(orig_sources - tinf_sources) > len(orig_sources) * 0.1:  # More than 10% missing
        is_consistent = False
        issues.append(f"Many sources missing from tinfoil: {len(orig_sources - tinf_sources)}")
    
    if is_consistent:
        print("✅ FILES ARE CONSISTENT")
        print("   - Chunk IDs appear to be properly mapped")
        print("   - Content matches well")
        print("   - No significant duplicate issues")
    else:
        print("❌ FILES ARE INCONSISTENT")
        for issue in issues:
            print(f"   - {issue}")
        print("\n🔧 TINFOIL FILE NEEDS FIXING")
    
    return original_data, tinfoil_data, is_consistent, issues

def main():
    result = check_healthify_consistency()
    if len(result) == 4:
        original_data, tinfoil_data, is_consistent, issues = result
        
        if not is_consistent and tinfoil_data is not None:
            print(f"\n{'='*70}")
            print("RECOMMENDATION: Run mapping fix for healthify tinfoil file")
            print("The tinfoil file appears to have similar issues as mayo had.")
    
if __name__ == "__main__":
    main()