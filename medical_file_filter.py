#!/usr/bin/env python3
"""
Medical File Filter using Qwen3 Model

This script analyzes medical files from three sources (Healthify, Mayo, NHS) and 
identifies those containing condition identification and treatment information.
Files that meet the criteria are moved to RAGdatav3.

Author: Claude Code
Version: 1.0
"""

import os
import shutil
import logging
import json
from typing import List, Tuple, Dict
from datetime import datetime
import argparse
import sys

# Check if required libraries are available
try:
    import requests
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install with: pip install requests")
    sys.exit(1)

# Configuration
SOURCE_DIRECTORIES = [
    "/Users/choemanseung/789/hft/RAGdata/cleaned_healthify_data",
    "/Users/choemanseung/789/hft/RAGdata/cleaned_mayo_data", 
    "/Users/choemanseung/789/hft/RAGdata/cleaned_nhs_data"
]

DESTINATION_DIRECTORY = "/Users/choemanseung/789/hft/RAGdatav3"
LOG_FILE = "medical_file_filter.log"
PROCESSED_FILES_LOG = "processed_files.json"
LARGE_FILES_LOG = "largefile.txt"
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "qwen3-4b-instruct-2507-mlx@8bit"  # Adjust based on your loaded model in LM Studio
MAX_WORDS_FOR_CONTEXT = 3000  # Conservative limit for 4K context window

# Classification prompt for Qwen3 model with real examples and JSON schema
CLASSIFICATION_PROMPT = """
You are a medical content analyzer. Your task is to determine if the given medical text contains BOTH:

1. Information for IDENTIFYING/DIAGNOSING a medical condition (symptoms, tests, criteria, etc.)
2. Information on HOW TO ACT AFTERWARDS (treatment options, management, next steps, recommendations)

EXAMPLES FROM REAL DATA:

Example 1 (YES):
Text: "Symptoms of jock itch are: A spreading rash that begins in the crease of the groin and moves down the upper thigh and buttocks. A rash whose center tends to clear as the rash spreads. A rash that may be full or partially ring shaped. A rash bordered with small blisters. Itchiness. Scaly skin. A rash that might be red, brown, purple or gray depending on your skin color. See your doctor if your rash is painful or you develop a fever. Tips for reducing the risk of jock itch include: Stay dry. Keep the groin area and inner thighs dry by drying with a clean towel after showering or exercising. Wear clean clothes. Change your underwear at least once a day or more often if you sweat a lot. It helps to wear underwear made of cotton or other fabric that breathes and keeps the skin drier. Don't share personal items. Don't let others use your clothing, towels or other personal items."
Response: {{"classification": "YES", "has_identification": true, "has_treatment": true, "explanation": "Contains clear symptoms for identifying jock itch (spreading rash, ring-shaped, itchiness) AND specific prevention/management strategies (stay dry, clean clothes, cotton underwear)"}}

Example 2 (YES):
Text: "Cardiac catheterization Chest X-rays Coronary angiogram Echocardiogram Electrocardiogram (ECG or EKG) Stress test Each minute after a heart attack, more heart tissue is damaged or dies. Urgent treatment is needed to fix blood flow and restore oxygen levels. Medications to treat a heart attack might include: Aspirin. Aspirin reduces blood clotting. It helps keep blood moving through a narrowed artery. Clot busters (thrombolytics or fibrinolytics). These drugs help break up any blood clots that are blocking blood flow to the heart. Other blood-thinning medicines. A medicine called heparin may be given by an intravenous (IV) injection. Nitroglycerin. This medication widens the blood vessels. It helps improve blood flow to the heart. Beta blockers. These medications slow the heartbeat and decrease blood pressure. Coronary angioplasty and stenting. This procedure is done to open clogged heart arteries. During angioplasty, a heart doctor guides a thin, flexible tube (catheter) to the narrowed part of the heart artery."
Response: {{"classification": "YES", "has_identification": true, "has_treatment": true, "explanation": "Contains diagnostic tests for heart attack identification (ECG, catheterization, stress test) AND detailed treatment medications (aspirin, clot busters, heparin) and procedures (angioplasty, stenting)"}}

Example 3 (NO):
Text: "These pages have information about back and neck pain, and exercises and apps that might help you to manage it."
Response: {{"classification": "NO", "has_identification": false, "has_treatment": false, "explanation": "Only a brief topic description without specific identification criteria or treatment details"}}

Example 4 (NO)
Text: "Acarbose Accarb® ACE inhibitors Acetazolamide Aciclovir (cream) Aciclovir (tablets) Acitretin Acne treatment Adalimumab Adapalene ADHD medicines – adults ADHD medicines – children AIR asthma therapy..."
Response: {{"classification": "NO", "has_identification": false, "has_treatment": false, "explanation": "Just a medication list without condition identification or treatment guidance"}}

Now analyze this medical text and respond with the exact JSON format shown above:

Text to analyze:
{text}

Response (JSON only):"""

class MedicalFileFilter:
    def __init__(self):
        self.setup_logging()
        self.processed_files = self.load_processed_files()
        self.large_files = []
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'qualifying_files': 0,
            'moved_files': 0,
            'large_files': 0,
            'errors': 0
        }
    
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_processed_files(self) -> Dict:
        """Load previously processed files to avoid reprocessing"""
        if os.path.exists(PROCESSED_FILES_LOG):
            try:
                with open(PROCESSED_FILES_LOG, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load processed files log: {e}")
        return {}
    
    def save_processed_files(self):
        """Save processed files log"""
        try:
            with open(PROCESSED_FILES_LOG, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save processed files log: {e}")
    
    def test_lm_studio_connection(self) -> bool:
        """Test if LM Studio API is available"""
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": "Test connection. Respond with 'OK'."}
                ],
                "temperature": 0,
                "max_tokens": 3
            }
            
            response = requests.post(
                LM_STUDIO_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                response_json = response.json()
                content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                if 'OK' in content.upper():
                    self.logger.info(f"LM Studio API with model {MODEL_NAME} is available and responding")
                    return True
                else:
                    self.logger.info(f"LM Studio API responded but content unclear: {content}")
                    return True  # Still consider it working
            else:
                self.logger.error(f"LM Studio API returned status {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to connect to LM Studio API: {e}")
            print(f"\nError: Cannot connect to LM Studio API at {LM_STUDIO_URL}")
            print("Please ensure:")
            print("1. LM Studio is running")
            print("2. A model is loaded in LM Studio")
            print("3. The server is running on localhost:1234")
            print("4. Check the 'Developer' tab in LM Studio for the correct endpoint")
            return False
        return False
    
    def read_file_content(self, file_path: str) -> str:
        """Read content from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content.strip()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return ""
    
    def log_large_file(self, file_path: str, word_count: int, source_label: str):
        """Log large files that exceed context window limits"""
        large_file_info = {
            'file_path': file_path,
            'word_count': word_count,
            'source': source_label,
            'timestamp': datetime.now().isoformat()
        }
        self.large_files.append(large_file_info)
        self.stats['large_files'] += 1
        
    def save_large_files_log(self):
        """Save large files log to text file"""
        try:
            with open(LARGE_FILES_LOG, 'w') as f:
                f.write(f"Large Files Report (>{MAX_WORDS_FOR_CONTEXT} words)\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total large files: {len(self.large_files)}\n\n")
                
                for file_info in self.large_files:
                    f.write(f"File: {file_info['file_path']}\n")
                    f.write(f"Words: {file_info['word_count']:,}\n")
                    f.write(f"Source: {file_info['source']}\n")
                    f.write(f"Logged: {file_info['timestamp']}\n")
                    f.write("-" * 40 + "\n")
                    
        except Exception as e:
            self.logger.error(f"Could not save large files log: {e}")
    
    
    def classify_file_content(self, content: str) -> Tuple[bool, str]:
        """
        Use Qwen3 model via LM Studio API to classify if content contains both condition identification and treatment info
        
        Returns:
            Tuple[bool, str]: (classification_result, explanation)
        """
        if not content.strip():
            return False, "Empty content"
        
        # Prepare the prompt
        prompt = CLASSIFICATION_PROMPT.format(text=content)
        
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # Low temperature for consistent results
                "max_tokens": 200,   # Increased for JSON response
                "top_p": 0.9
            }
            
            response = requests.post(
                LM_STUDIO_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60  # Longer timeout for classification
            )
            
            if response.status_code == 200:
                response_json = response.json()
                response_text = response_json.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                
                # Try to parse JSON response
                try:
                    result = json.loads(response_text)
                    classification = result.get('classification', '').upper()
                    explanation = result.get('explanation', 'No explanation provided')
                    
                    if classification == 'YES':
                        return True, explanation
                    elif classification == 'NO':
                        return False, explanation
                    else:
                        # Fallback to simple text parsing
                        if 'YES' in response_text.upper():
                            return True, "Contains both identification and treatment info"
                        elif 'NO' in response_text.upper():
                            return False, "Missing identification or treatment info"
                        else:
                            self.logger.warning(f"Unclear model response: {response_text}")
                            return self.fallback_classification(content)
                            
                except json.JSONDecodeError:
                    # Fallback to simple text parsing if JSON parsing fails
                    if 'YES' in response_text.upper():
                        return True, "Contains both identification and treatment info"
                    elif 'NO' in response_text.upper():
                        return False, "Missing identification or treatment info"
                    else:
                        self.logger.warning(f"Invalid JSON response: {response_text}")
                        return self.fallback_classification(content)
            else:
                self.logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
                return self.fallback_classification(content)
                
        except Exception as e:
            self.logger.error(f"Error in LM Studio classification: {e}")
            return self.fallback_classification(content)
    
    def fallback_classification(self, content: str) -> Tuple[bool, str]:
        """Fallback keyword-based classification if model fails"""
        content_lower = content.lower()
        
        # Keywords for condition identification
        identification_keywords = [
            'symptom', 'diagnos', 'test', 'sign', 'criteria', 'examination',
            'screening', 'detect', 'identify', 'recogni', 'blood test',
            'urine test', 'x-ray', 'scan', 'biopsy', 'check'
        ]
        
        # Keywords for treatment/action
        treatment_keywords = [
            'treatment', 'therap', 'medication', 'medicine', 'drug', 
            'surgery', 'management', 'care', 'prevent', 'avoid',
            'lifestyle', 'exercise', 'diet', 'recommend', 'advice',
            'follow-up', 'monitor', 'insulin', 'prescription'
        ]
        
        has_identification = any(keyword in content_lower for keyword in identification_keywords)
        has_treatment = any(keyword in content_lower for keyword in treatment_keywords)
        
        if has_identification and has_treatment:
            return True, "Fallback: Contains both keywords"
        else:
            return False, f"Fallback: Missing {'identification' if not has_identification else 'treatment'} keywords"
    
    def get_all_files(self) -> List[Tuple[str, str]]:
        """Get all text files from source directories with their source labels"""
        all_files = []
        
        for source_dir in SOURCE_DIRECTORIES:
            if not os.path.exists(source_dir):
                self.logger.warning(f"Source directory does not exist: {source_dir}")
                continue
            
            source_label = os.path.basename(source_dir).replace('cleaned_', '').replace('_data', '')
            
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        all_files.append((file_path, source_label))
        
        return all_files
    
    def move_file_to_destination(self, source_path: str, source_label: str) -> bool:
        """Move qualifying file to destination directory"""
        try:
            # Create destination subdirectory for source
            dest_subdir = os.path.join(DESTINATION_DIRECTORY, source_label)
            os.makedirs(dest_subdir, exist_ok=True)
            
            # Get filename and create destination path
            filename = os.path.basename(source_path)
            dest_path = os.path.join(dest_subdir, filename)
            
            # Handle filename conflicts
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                dest_path = os.path.join(dest_subdir, f"{name}_{counter}{ext}")
                counter += 1
            
            # Copy the file
            shutil.copy2(source_path, dest_path)
            self.logger.info(f"Moved: {source_path} -> {dest_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error moving file {source_path}: {e}")
            return False
    
    def process_files(self, dry_run: bool = False, max_files: int = None):
        """Process all files from source directories"""
        self.logger.info("Starting medical file filtering process")
        self.logger.info(f"LM Studio API: {LM_STUDIO_URL}")
        self.logger.info(f"Model: {MODEL_NAME}")
        self.logger.info(f"Dry run mode: {dry_run}")
        
        # Test LM Studio connection first
        if not self.test_lm_studio_connection():
            self.logger.error("Cannot proceed without LM Studio API connection")
            return False
        
        # Get all files
        all_files = self.get_all_files()
        self.stats['total_files'] = len(all_files)
        
        if max_files:
            all_files = all_files[:max_files]
            self.logger.info(f"Limited to first {max_files} files for testing")
        
        self.logger.info(f"Found {len(all_files)} files to process")
        
        # Process each file
        for i, (file_path, source_label) in enumerate(all_files, 1):
            try:
                # Skip if already processed (unless forced)
                file_key = f"{source_label}:{os.path.basename(file_path)}"
                if file_key in self.processed_files:
                    self.logger.info(f"Skipping already processed file: {file_path}")
                    continue
                
                self.logger.info(f"Processing ({i}/{len(all_files)}): {file_path}")
                
                # Read file content
                content = self.read_file_content(file_path)
                if not content:
                    self.stats['errors'] += 1
                    continue
                
                # Check if file is too large for context window
                word_count = len(content.split())
                if word_count > MAX_WORDS_FOR_CONTEXT:
                    self.logger.warning(f"  Large file ({word_count:,} words) - may be truncated by model")
                    self.log_large_file(file_path, word_count, source_label)
                
                # Classify content using LM Studio API
                is_qualifying, explanation = self.classify_file_content(content)
                
                # Log result
                result_text = "QUALIFYING" if is_qualifying else "NOT QUALIFYING"
                self.logger.info(f"  Result: {result_text} - {explanation}")
                
                # Update processed files log
                self.processed_files[file_key] = {
                    'file_path': file_path,
                    'source_label': source_label,
                    'processed_at': datetime.now().isoformat(),
                    'is_qualifying': is_qualifying,
                    'explanation': explanation,
                    'moved': False
                }
                
                # Move file if it qualifies (and not dry run)
                if is_qualifying:
                    self.stats['qualifying_files'] += 1
                    
                    if not dry_run:
                        if self.move_file_to_destination(file_path, source_label):
                            self.stats['moved_files'] += 1
                            self.processed_files[file_key]['moved'] = True
                        else:
                            self.stats['errors'] += 1
                    else:
                        self.logger.info(f"  [DRY RUN] Would move: {file_path}")
                
                self.stats['processed_files'] += 1
                
                # Save progress periodically
                if i % 10 == 0:
                    self.save_processed_files()
                    self.print_progress()
            
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                self.stats['errors'] += 1
        
        # Final save
        self.save_processed_files()
        self.save_large_files_log()
        self.print_final_results()
        return True
    
    def print_progress(self):
        """Print current progress statistics"""
        print(f"\nProgress: {self.stats['processed_files']}/{self.stats['total_files']} files")
        print(f"Qualifying: {self.stats['qualifying_files']}, Moved: {self.stats['moved_files']}, Large: {self.stats['large_files']}, Errors: {self.stats['errors']}")
    
    def print_final_results(self):
        """Print final processing results"""
        print("\n" + "="*60)
        print("MEDICAL FILE FILTERING COMPLETE")
        print("="*60)
        print(f"Total files found:      {self.stats['total_files']}")
        print(f"Files processed:        {self.stats['processed_files']}")
        print(f"Qualifying files:       {self.stats['qualifying_files']}")
        print(f"Files moved:           {self.stats['moved_files']}")
        print(f"Large files (>{MAX_WORDS_FOR_CONTEXT} words): {self.stats['large_files']}")
        print(f"Errors:                {self.stats['errors']}")
        print(f"Success rate:          {(self.stats['processed_files']/max(1,self.stats['total_files'])*100):.1f}%")
        print(f"Qualification rate:    {(self.stats['qualifying_files']/max(1,self.stats['processed_files'])*100):.1f}%")
        if self.stats['large_files'] > 0:
            print(f"\nSee '{LARGE_FILES_LOG}' for details on large files")
        print("="*60)


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Filter medical files using Qwen3 model for condition identification and treatment information"
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help="Run classification without moving files"
    )
    parser.add_argument(
        '--max-files',
        type=int,
        help="Limit number of files to process (for testing)"
    )
    parser.add_argument(
        '--model',
        default=MODEL_NAME,
        help=f"Model name to use in LM Studio (default: {MODEL_NAME})"
    )
    parser.add_argument(
        '--url',
        default=LM_STUDIO_URL,
        help=f"LM Studio API URL (default: {LM_STUDIO_URL})"
    )
    
    args = parser.parse_args()
    
    # Create filter instance and run
    filter_instance = MedicalFileFilter()
    success = filter_instance.process_files(
        dry_run=args.dry_run,
        max_files=args.max_files
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()