#!/usr/bin/env python3
"""
Extract unique medical conditions from file names in RAG data directories.

This script traverses three medical data directories and extracts unique
medical condition names from the file names, cleaning and normalizing them.
"""

import os
import re
from pathlib import Path
from typing import Set, List


def clean_filename(filename: str) -> str:
    """
    Clean and normalize a filename to extract the medical condition name.
    
    Args:
        filename: The original filename (without extension)
        
    Returns:
        Cleaned medical condition name
    """
    # Remove common suffixes
    suffixes_to_remove = [
        '_diagnosis_treatment',
        '_treatment',
        '_diagnosis',
        '_(overview)',
        '_overview'
    ]
    
    cleaned = filename
    for suffix in suffixes_to_remove:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            break
    
    # Handle NHS-specific patterns (What_is_X-Y format)
    if cleaned.startswith('What_is_') and '-' in cleaned:
        parts = cleaned.split('-')
        if len(parts) == 2:
            cleaned = parts[1]
    
    # Handle Overview- prefix
    if cleaned.startswith('Overview-'):
        cleaned = cleaned[9:]  # Remove 'Overview-'
    
    # Handle Symptoms- prefix  
    if cleaned.startswith('Symptoms-'):
        cleaned = cleaned[9:]  # Remove 'Symptoms-'
    
    # Replace underscores with spaces
    cleaned = cleaned.replace('_', ' ')
    
    # Remove content in parentheses at the end (like abbreviations)
    cleaned = re.sub(r'\s*\([^)]*\)\s*$', '', cleaned)
    
    # Clean up extra spaces
    cleaned = ' '.join(cleaned.split())
    
    # Convert to title case for consistency
    cleaned = cleaned.title()
    
    return cleaned


def extract_conditions_from_directory(directory_path: str) -> Set[str]:
    """
    Extract medical conditions from all files in a directory.
    
    Args:
        directory_path: Path to the directory containing medical data files
        
    Returns:
        Set of unique medical condition names
    """
    conditions = set()
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Warning: Directory {directory_path} does not exist")
        return conditions
    
    # Get all .txt files in the directory
    txt_files = list(directory.glob('*.txt'))
    print(f"Found {len(txt_files)} files in {directory.name}")
    
    for file_path in txt_files:
        # Get filename without extension
        filename = file_path.stem
        
        # Clean and extract condition name
        condition = clean_filename(filename)
        
        # Skip very short names or single letters
        if len(condition) > 1 and condition.strip():
            conditions.add(condition)
    
    return conditions


def main():
    """Main function to extract medical conditions from all directories."""
    # Define the three data directories
    base_path = "/Users/choemanseung/789/hft/RAGdata"
    directories = [
        os.path.join(base_path, "cleaned_healthify_data"),
        os.path.join(base_path, "cleaned_mayo_data"), 
        os.path.join(base_path, "cleaned_nhs_data")
    ]
    
    all_conditions = set()
    
    print("Extracting medical conditions from data directories...")
    print("=" * 60)
    
    # Process each directory
    for directory in directories:
        print(f"\nProcessing: {os.path.basename(directory)}")
        conditions = extract_conditions_from_directory(directory)
        all_conditions.update(conditions)
        print(f"Unique conditions from this directory: {len(conditions)}")
    
    # Convert to sorted list for output
    unique_conditions = sorted(list(all_conditions))
    
    print("\n" + "=" * 60)
    print(f"Total unique medical conditions found: {len(unique_conditions)}")
    print("=" * 60)
    
    # Save to file
    output_file = "/Users/choemanseung/789/hft/unique_medical_conditions.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Unique Medical Conditions\n")
        f.write("=" * 25 + "\n\n")
        f.write(f"Total count: {len(unique_conditions)}\n\n")
        
        for i, condition in enumerate(unique_conditions, 1):
            f.write(f"{i:4d}. {condition}\n")
    
    print(f"\nResults saved to: {output_file}")
    
    # Display first 20 conditions as a sample
    print("\nSample of medical conditions found:")
    print("-" * 40)
    for i, condition in enumerate(unique_conditions[:20], 1):
        print(f"{i:2d}. {condition}")
    
    if len(unique_conditions) > 20:
        print(f"... and {len(unique_conditions) - 20} more")


if __name__ == "__main__":
    main()