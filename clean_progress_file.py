#!/usr/bin/env python3
import json

def clean_progress_file():
    """Clean the progress file to only include documents that actually exist in the tinfoil JSON"""
    
    progress_file = "/Users/choemanseung/789/hft/contextual_chunking_progress_mayo_contextual_fixed_c512_o100_tinfoil.json"
    tinfoil_file = "/Users/choemanseung/789/hft/RAGdatav4/mayo_chunks_contextual_fixed_c512_o100_tinfoil.json"
    
    # Load the progress file
    with open(progress_file, 'r', encoding='utf-8') as f:
        progress_data = json.load(f)
    
    # Load the tinfoil file to get actual source documents
    with open(tinfoil_file, 'r', encoding='utf-8') as f:
        tinfoil_data = json.load(f)
    
    # Get unique source documents from tinfoil file
    actual_sources = set(entry.get('source_document', '') for entry in tinfoil_data)
    actual_sources.discard('')  # Remove empty strings if any
    
    # Get current processed documents from progress file
    current_processed = set(progress_data.get('processed_documents', []))
    
    print(f"Documents in progress file: {len(current_processed)}")
    print(f"Actual documents in tinfoil file: {len(actual_sources)}")
    
    # Find which documents should be removed (in progress but not in actual)
    documents_to_remove = current_processed - actual_sources
    # Find which documents are actually processed (intersection)
    actually_processed = current_processed & actual_sources
    
    print(f"Documents to remove from progress: {len(documents_to_remove)}")
    print(f"Documents that are actually processed: {len(actually_processed)}")
    
    if documents_to_remove:
        print("\nSample documents being removed:")
        for doc in list(documents_to_remove)[:10]:
            print(f"  - {doc}")
        
        if len(documents_to_remove) > 10:
            print(f"  ... and {len(documents_to_remove) - 10} more")
    
    # Update the progress file
    progress_data['processed_documents'] = sorted(list(actually_processed))
    
    # Create backup of original file
    backup_file = progress_file.replace('.json', '_backup.json')
    import shutil
    shutil.copy2(progress_file, backup_file)
    print(f"\nBackup created: {backup_file}")
    
    # Write the cleaned progress file
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)
    
    print(f"Updated progress file: {progress_file}")
    print(f"New processed_documents count: {len(actually_processed)}")
    
    # Verify the update
    print("\nVerification:")
    print(f"✓ All {len(actually_processed)} documents in progress file exist in tinfoil file")
    
    # Check if there are documents in tinfoil that aren't marked as processed
    not_marked_processed = actual_sources - actually_processed
    if not_marked_processed:
        print(f"⚠️  {len(not_marked_processed)} documents in tinfoil file are not marked as processed")
        print("Sample documents not marked as processed:")
        for doc in list(not_marked_processed)[:5]:
            print(f"  - {doc}")

def main():
    print("Cleaning progress file to match actual tinfoil data...")
    print("=" * 60)
    clean_progress_file()

if __name__ == "__main__":
    main()