#!/usr/bin/env python3
import json
import sys
from collections import Counter

def check_source_document_field(file_path):
    """Check if any entries are missing the source_document field"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Error: {file_path} does not contain a JSON array")
            return False, 0, 0
        
        missing_source_doc = []
        total_entries = len(data)
        
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                print(f"Warning: Entry {i} is not a dictionary")
                continue
                
            if 'source_document' not in entry:
                missing_source_doc.append(i)
            elif entry['source_document'] is None or entry['source_document'] == '':
                missing_source_doc.append(i)
        
        print(f"File: {file_path}")
        print(f"Total entries: {total_entries}")
        print(f"Entries missing source_document: {len(missing_source_doc)}")
        
        if missing_source_doc:
            print(f"Indices of entries missing source_document: {missing_source_doc[:10]}{'...' if len(missing_source_doc) > 10 else ''}")
            return False, total_entries, len(missing_source_doc)
        else:
            print("✓ All entries have source_document field")
            
            # Check for duplicates
            chunk_ids = [entry.get('chunk_id', 'NO_ID') for entry in data]
            dupes = [id for id, count in Counter(chunk_ids).items() if count > 1]
            if dupes:
                print(f"⚠️  Found {len(dupes)} duplicate chunk_ids")
                print(f"Sample duplicates: {dupes[:5]}")
            
            # Count unique source documents
            sources = set(entry.get('source_document', 'NO_SOURCE') for entry in data)
            sources.discard('NO_SOURCE')
            print(f"Unique source documents: {len(sources)}")
            
            return True, total_entries, 0
            
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return False, 0, 0
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return False, 0, 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0, 0

def main():
    files = [
        "/Users/choemanseung/789/hft/RAGdatav4/healthify_chunks_contextual_fixed_c512_o100_tinfoil.json",
        "/Users/choemanseung/789/hft/RAGdatav4/healthify_chunks_contextual_sentence_c1024_o2_tinfoil.json"
    ]
    
    print("Checking source_document field in healthify tinfoil files...")
    print("=" * 80)
    
    results = []
    for file_path in files:
        success, total, missing = check_source_document_field(file_path)
        results.append((file_path.split('/')[-1], success, total, missing))
        print("-" * 80)
    
    print("\nSUMMARY:")
    print("=" * 80)
    for filename, success, total, missing in results:
        status = "✓" if success else "✗"
        print(f"{status} {filename}")
        print(f"  Total entries: {total}")
        if missing > 0:
            print(f"  Missing source_document: {missing}")
        print()

if __name__ == "__main__":
    main()