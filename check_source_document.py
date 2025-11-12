#!/usr/bin/env python3
import json
import sys

def check_source_document_field(file_path):
    """Check if any entries are missing the source_document field"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Error: {file_path} does not contain a JSON array")
            return False
        
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
        
        print(f"\nFile: {file_path}")
        print(f"Total entries: {total_entries}")
        print(f"Entries missing source_document: {len(missing_source_doc)}")
        
        if missing_source_doc:
            print(f"Indices of entries missing source_document: {missing_source_doc[:10]}{'...' if len(missing_source_doc) > 10 else ''}")
            return False
        else:
            print("✓ All entries have source_document field")
            return True
            
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    original_file = "/Users/choemanseung/789/hft/RAGdatav4/mayo_chunks_contextual_fixed_c512_o100.json"
    tinfoil_file = "/Users/choemanseung/789/hft/RAGdatav4/mayo_chunks_contextual_fixed_c512_o100_tinfoil.json"
    
    print("Checking source_document field in both files...")
    print("=" * 60)
    
    original_ok = check_source_document_field(original_file)
    tinfoil_ok = check_source_document_field(tinfoil_file)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Original file has all source_document fields: {'✓' if original_ok else '✗'}")
    print(f"Tinfoil file has all source_document fields: {'✓' if tinfoil_ok else '✗'}")
    
    if not tinfoil_ok and original_ok:
        print("\n⚠️  TINFOIL VERSION IS MISSING source_document FIELDS")
    elif tinfoil_ok and not original_ok:
        print("\n⚠️  ORIGINAL VERSION IS MISSING source_document FIELDS")
    elif not tinfoil_ok and not original_ok:
        print("\n⚠️  BOTH FILES ARE MISSING source_document FIELDS")
    else:
        print("\n✓ Both files have all required source_document fields")

if __name__ == "__main__":
    main()