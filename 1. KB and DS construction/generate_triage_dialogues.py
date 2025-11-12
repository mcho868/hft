#!/usr/bin/env python3
"""
Automated Triage Dialogue Generator using TinfoilAgent
Generates synthetic medical triage dialogues with parallel processing for efficiency.
"""

import os
import sys
import re
import time
import json
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Add the RAGdatav3 directory to the Python path for TinfoilAgent imports
sys.path.append('/Users/choemanseung/789/hft')

# Import TinfoilAgent
try:
    from mlx_models.tinfoilAgent import TinfoilAgent
    TINFOIL_AVAILABLE = True
except ImportError:
    print("❌ TinfoilAgent not available. Please ensure it's properly installed.")
    TINFOIL_AVAILABLE = False
    sys.exit(1)


@dataclass
class TriageDialogue:
    """Data class for storing generated triage dialogue"""
    symptom: str
    patient_query: str
    clarifying_question: str
    patient_response: str
    final_triage_decision: str
    next_step: str
    reasoning_question: str
    reasoning_decision: str
    generation_timestamp: float


def load_medical_conditions(file_path: str) -> List[str]:
    """Load medical conditions from the filtered list file."""
    conditions = []
    
    # Try to load from AI-filtered results first, then fall back to manually filtered
    ai_filtered_path = "/Users/choemanseung/789/hft/ai_filtered_medical_conditions.txt"
    manual_filtered_path = "/Users/choemanseung/789/hft/filtered_medical_conditions.txt"
    
    target_file = ai_filtered_path if os.path.exists(ai_filtered_path) else manual_filtered_path
    if not os.path.exists(target_file):
        target_file = file_path  # Fall back to original file
    
    print(f"📋 Loading conditions from: {os.path.basename(target_file)}")
    
    with open(target_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Extract condition from numbered format: "   123. Condition Name"
            match = re.match(r'\s*\d+\.\s+(.+)', line.strip())
            if match:
                conditions.append(match.group(1).strip())
    
    return conditions


def build_triage_lookup() -> Dict[str, Dict[str, Any]]:
    """Build a lookup dictionary from structured triage data, averaging multiple matches."""
    
    ragdata_dir = Path("/Users/choemanseung/789/hft/RAGdatav4")
    structured_files = list(ragdata_dir.glob("*structured_agent_tinfoil_medical*.json"))
    
    if not structured_files:
        print("❌ No structured triage files found in RAGdatav4")
        return {}
    
    print(f"📋 Found {len(structured_files)} structured triage files")
    
    # First collect all matches for each condition
    condition_matches = {}
    
    for file_path in structured_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract structured chunks
            for chunk in data:
                if chunk.get('is_structured') and chunk.get('structured_data'):
                    structured_data = chunk['structured_data']
                    
                    # Only include if we have essential triage information
                    if (structured_data.get('triage_level') and 
                        structured_data.get('symptoms') and 
                        structured_data.get('next_steps')):
                        
                        # Use condition name (without .txt) as key
                        condition_name = structured_data.get('condition', '').replace('.txt', '')
                        condition_key = condition_name.lower().replace('_', ' ').replace('-', ' ')
                        
                        if condition_key not in condition_matches:
                            condition_matches[condition_key] = []
                        
                        condition_matches[condition_key].append({
                            'chunk_id': chunk['chunk_id'],
                            'condition': condition_name,
                            'symptoms': structured_data['symptoms'],
                            'triage_level': structured_data['triage_level'],
                            'next_steps': structured_data['next_steps'],
                            'source_document': chunk.get('source_document', ''),
                            'full_text': chunk.get('text', '')
                        })
            
            print(f"✅ Loaded {len([c for c in data if c.get('is_structured')])} structured chunks from {file_path.name}")
            
        except Exception as e:
            print(f"⚠️  Error loading {file_path.name}: {e}")
            continue
    
    # Now average out multiple matches
    triage_lookup = {}
    
    for condition_key, matches in condition_matches.items():
        if len(matches) == 1:
            # Single match - use as is
            triage_lookup[condition_key] = matches[0]
        else:
            # Multiple matches - average the triage decision
            print(f"🔄 Found {len(matches)} matches for '{condition_key}' - averaging triage...")
            
            # Count triage levels
            triage_counts = {}
            all_symptoms = []
            all_next_steps = []
            
            for match in matches:
                triage = match['triage_level']
                triage_counts[triage] = triage_counts.get(triage, 0) + 1
                all_symptoms.extend(match['symptoms'])
                all_next_steps.extend(match['next_steps'])
            
            # Determine averaged triage with clinical priority for ties
            # Priority order: ED > GP > HOME (err on side of caution)
            triage_priority = {'ED': 3, 'GP': 2, 'HOME': 1}
            
            max_count = max(triage_counts.values())
            tied_triages = [triage for triage, count in triage_counts.items() if count == max_count]
            
            if len(tied_triages) == 1:
                averaged_triage = tied_triages[0]
            else:
                # Tie-breaking: choose highest priority (most cautious)
                averaged_triage = max(tied_triages, key=lambda x: triage_priority.get(x, 0))
                print(f"   🔀 Tie detected {dict((t, triage_counts[t]) for t in tied_triages)} - choosing {averaged_triage} (clinical priority)")
            
            # Combine and deduplicate symptoms and next steps
            unique_symptoms = list(dict.fromkeys(all_symptoms))  # Preserves order, removes duplicates
            unique_next_steps = list(dict.fromkeys(all_next_steps))
            
            # Create averaged entry
            triage_lookup[condition_key] = {
                'chunk_id': f"averaged_{condition_key}",
                'condition': matches[0]['condition'],  # Use first match's condition name
                'symptoms': unique_symptoms,
                'triage_level': averaged_triage,
                'next_steps': unique_next_steps,
                'source_document': f"multiple_sources_{len(matches)}_files",
                'full_text': f"Averaged from {len(matches)} sources with triage distribution: {triage_counts}",
                'is_averaged': True,
                'source_count': len(matches),
                'triage_distribution': triage_counts
            }
            
            print(f"   📊 Averaged triage for '{condition_key}': {averaged_triage} (from {triage_counts})")
    
    print(f"📊 Total triage lookup entries: {len(triage_lookup)}")
    
    # Show overall triage distribution
    triage_counts = {}
    averaged_count = 0
    for entry in triage_lookup.values():
        triage = entry['triage_level']
        triage_counts[triage] = triage_counts.get(triage, 0) + 1
        if entry.get('is_averaged'):
            averaged_count += 1
    
    print(f"🎯 Final triage distribution: {triage_counts}")
    print(f"🔄 Averaged entries: {averaged_count}/{len(triage_lookup)}")
    
    return triage_lookup


def find_matching_triage_data(condition: str, triage_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Find matching triage data for a given medical condition."""
    
    # Normalize the condition name for matching
    condition_normalized = condition.lower().replace('-', ' ').replace('_', ' ')
    
    # Try exact match first
    if condition_normalized in triage_lookup:
        return triage_lookup[condition_normalized]
    
    # Try partial matching for conditions that might have slight variations
    best_match = None
    best_score = 0
    
    for key, data in triage_lookup.items():
        # Calculate similarity score
        condition_words = set(condition_normalized.split())
        key_words = set(key.split())
        
        # Jaccard similarity
        intersection = condition_words.intersection(key_words)
        union = condition_words.union(key_words)
        
        if len(union) > 0:
            similarity = len(intersection) / len(union)
            
            if similarity > best_score and similarity >= 0.5:  # At least 50% similarity
                best_score = similarity
                best_match = data
    
    return best_match


def create_grounded_triage_prompt(triage_data: Dict[str, Any], case_number: int) -> str:
    """Create triage dialogue prompt grounded in actual structured medical data."""
    
    # Add variation guidance for different cases
    variation_guidance = {
        1: "Focus on a young adult (20-35) in an urban setting (Auckland/Wellington).",
        2: "Focus on a middle-aged person (35-55) from a rural area or smaller town.",
        3: "Focus on an elderly person (65+) with potential cultural context (Māori/Pasifika).",
        4: "Focus on a child/teenager (5-18) with parental involvement in the query.",
        5: "Focus on a working-age adult (25-50) in a workplace or telehealth context.",
        6: "Focus on a pregnant/breastfeeding woman (20-40) with family context and specific NZ cultural elements.",
        7: "Focus on an immunocompromised or chronic condition patient (any age) with detailed medication history and complex medical context."
    }
    
    case_guidance = variation_guidance.get(case_number, "Create a diverse, realistic New Zealand scenario.")
    
    # Format symptoms and next steps for the prompt
    symptoms_text = "\n".join([f"- {symptom}" for symptom in triage_data['symptoms']])
    next_steps_text = "\n".join([f"- {step}" for step in triage_data['next_steps']])
    
    prompt = f"""You are an expert medical triage doctor in Aotearoa New Zealand. Generate a realistic triage dialogue based on this ACTUAL medical triage information:

MEDICAL CONDITION: {triage_data['condition']}

DOCUMENTED SYMPTOMS:
{symptoms_text}

CORRECT TRIAGE DECISION: {triage_data['triage_level']}

RECOMMENDED NEXT STEPS:
{next_steps_text}

CASE VARIATION GUIDANCE: {case_guidance}

CRITICAL REQUIREMENTS:
- The patient's symptoms MUST align with the documented symptoms above
- The Final Triage Decision MUST be: {triage_data['triage_level']}
- The Next Step MUST be consistent with the recommended next steps above
- Create realistic New Zealand healthcare context

MANDATORY OUTPUT FORMAT - You MUST include ALL sections exactly as shown:

Patient Query: A short, naturalistic description of symptoms from a layperson in New Zealand. Include age, gender, and NZ context (e.g., "GP clinic," "telehealth," or cultural background where relevant).

Clarifying Question: One clinically appropriate question that helps resolve ambiguity or screen for red flags.

Patient Response: A naturalistic answer to the clarifying question.

Final Triage Decision: One of {{ED, GP, HOME}}.

Next Step for Patient: A short, practical instruction aligned with the triage decision (e.g., "Call 111 immediately," "Book an urgent GP appointment today," "Rest at home and monitor symptoms").

Reasoning for Clarifying Question (CoT): A concise, 2–3 sentence step-by-step reasoning for why this clarifying question is important. THIS SECTION IS MANDATORY.

Reasoning for Triage Decision (CoT): A concise, 2–3 sentence clinical justification for the final decision. THIS SECTION IS MANDATORY.

Constraints:
- Keep dialogues short (1–3 sentences per turn).
- Use New Zealand spelling and healthcare terminology.
- Do not overuse CoT—keep it minimal, clear, and clinically grounded.
- Ensure realism: demographics (Māori, Pasifika, rural/urban NZ) and healthcare contexts (GP, ED, telehealth).

Example format (illustrative only):
Patient Query: "Kia ora, I'm a 65-year-old Māori man in Rotorua with sudden shortness of breath and wheezing."
Clarifying Question: "Did this start suddenly and are you having trouble speaking in full sentences?"
Patient Response: "Yes, it started all of a sudden and I can't finish sentences without stopping to breathe."
Final Triage Decision: ED
Next Step for Patient: Call 111 immediately and request an ambulance.
Reasoning for Clarifying Question (CoT): Acute shortness of breath with inability to speak may indicate severe asthma or cardiac event, requiring urgent clarification.
Reasoning for Triage Decision (CoT): The patient shows red-flag symptoms consistent with severe respiratory distress, so emergency department care is essential.

Generate the dialogue now:"""

    return prompt


def call_tinfoil_agent(prompt: str, model_name: str = "llama3-3-70b") -> str:
    """Call TinfoilAgent API to generate triage dialogue."""
    try:
        agent = TinfoilAgent(model_name)
        response = agent.getResponse(prompt)
        return response if response else ""
    except Exception as e:
        print(f"❌ Error calling TinfoilAgent: {e}")
        return ""


def parse_triage_response(response: str, symptom: str) -> TriageDialogue:
    """Parse the AI response into structured triage dialogue components."""
    
    def extract_section(text: str, section_name: str) -> str:
        """Extract content after a section header."""
        # Try multiple pattern variations - be more flexible with spacing and case
        patterns = [
            # Exact match with (CoT) - must start after newline or beginning
            rf"(?:^|\n){re.escape(section_name)}\s*\(CoT\)\s*:\s*(.*?)(?=\n\s*[A-Z][^:]*:|$)",
            # Without (CoT) - must start after newline or beginning
            rf"(?:^|\n){re.escape(section_name)}\s*:\s*(.*?)(?=\n\s*[A-Z][^:]*:|$)",
            # Case insensitive with (CoT) - must start after newline or beginning
            rf"(?i)(?:^|\n){re.escape(section_name.lower())}\s*\(cot\)\s*:\s*(.*?)(?=\n\s*[A-Za-z][^:]*:|$)",
            # Case insensitive without (CoT) - must start after newline or beginning
            rf"(?i)(?:^|\n){re.escape(section_name.lower())}\s*:\s*(.*?)(?=\n\s*[A-Za-z][^:]*:|$)"
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted:  # Only return non-empty matches
                    # Debug: print which pattern worked
                    # print(f"   🔍 Pattern {i+1} matched for '{section_name}': {extracted[:50]}...")
                    return extracted
        return ""
    
    # Extract each section
    patient_query = extract_section(response, "Patient Query")
    clarifying_question = extract_section(response, "Clarifying Question")
    patient_response = extract_section(response, "Patient Response")
    final_triage = extract_section(response, "Final Triage Decision")
    next_step = extract_section(response, "Next Step for Patient")
    
    # Try multiple variations for reasoning sections
    reasoning_question = (
        extract_section(response, "Reasoning for Clarifying Question (CoT)") or
        extract_section(response, "Reasoning for Clarifying Question") or
        extract_section(response, "Clarifying Question Reasoning") or
        extract_section(response, "Question Reasoning")
    )
    
    reasoning_decision = (
        extract_section(response, "Reasoning for Triage Decision (CoT)") or
        extract_section(response, "Reasoning for Triage Decision") or
        extract_section(response, "Triage Decision Reasoning") or
        extract_section(response, "Decision Reasoning")
    )
    
    # Clean up and validate final triage decision
    if final_triage:
        final_triage = final_triage.upper().strip()
        if final_triage not in ["ED", "GP", "HOME"]:
            # Try to extract from the text
            if "ED" in final_triage or "EMERGENCY" in final_triage.upper():
                final_triage = "ED"
            elif "GP" in final_triage or "DOCTOR" in final_triage.upper():
                final_triage = "GP"
            else:
                final_triage = "HOME"
    
    return TriageDialogue(
        symptom=symptom,
        patient_query=patient_query,
        clarifying_question=clarifying_question,
        patient_response=patient_response,
        final_triage_decision=final_triage,
        next_step=next_step,
        reasoning_question=reasoning_question,
        reasoning_decision=reasoning_decision,
        generation_timestamp=time.time()
    )


def generate_multiple_dialogues(triage_data: Dict[str, Any], num_cases: int = 7) -> Tuple[str, List[TriageDialogue], int]:
    """Generate multiple triage dialogues for a given structured triage entry."""
    generated_dialogues = []
    successful_count = 0
    condition = triage_data['condition']
    
    for case_num in range(1, num_cases + 1):
        try:
            prompt = create_grounded_triage_prompt(triage_data, case_num)
            response = call_tinfoil_agent(prompt)
            
            if not response:
                print(f"⚠️  No response for {condition} case {case_num}")
                continue
            
            dialogue = parse_triage_response(response, f"{condition} (Case {case_num})")
            
            # Enhanced validation - check triage consistency and reasoning sections
            triage_consistent = (dialogue.final_triage_decision == triage_data['triage_level'])
            has_reasoning = (dialogue.reasoning_question and dialogue.reasoning_decision)
            
            if (dialogue.patient_query and dialogue.clarifying_question and 
                dialogue.patient_response and dialogue.final_triage_decision and
                triage_consistent and has_reasoning):
                generated_dialogues.append(dialogue)
                successful_count += 1
                print(f"✅ Generated case {case_num}/7 for: {condition} (Triage: {dialogue.final_triage_decision})")
            else:
                if not triage_consistent:
                    print(f"⚠️  Triage mismatch for {condition} case {case_num}: expected {triage_data['triage_level']}, got {dialogue.final_triage_decision}")
                elif not has_reasoning:
                    print(f"⚠️  Missing reasoning sections for {condition} case {case_num}: question='{dialogue.reasoning_question[:50]}...' decision='{dialogue.reasoning_decision[:50]}...'")
                else:
                    print(f"⚠️  Incomplete dialogue for {condition} case {case_num}")
                
        except Exception as e:
            print(f"❌ Error generating case {case_num} for {condition}: {e}")
            continue
        
        # Small delay between cases to avoid overwhelming the API
        time.sleep(0.5)
    
    return condition, generated_dialogues, successful_count


def save_progress(progress_file: str, completed_symptoms: List[str], generated_dialogues: List[TriageDialogue]):
    """Save progress to file."""
    progress_data = {
        "completed_symptoms": completed_symptoms,
        "generated_count": len(generated_dialogues),
        "timestamp": time.time(),
        "dialogues": [
            {
                "symptom": d.symptom,
                "patient_query": d.patient_query,
                "clarifying_question": d.clarifying_question,
                "patient_response": d.patient_response,
                "final_triage_decision": d.final_triage_decision,
                "next_step": d.next_step,
                "reasoning_question": d.reasoning_question,
                "reasoning_decision": d.reasoning_decision,
                "generation_timestamp": d.generation_timestamp
            } for d in generated_dialogues
        ]
    }
    
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)


def load_progress(progress_file: str) -> Tuple[List[str], List[TriageDialogue]]:
    """Load progress from file if it exists."""
    if not os.path.exists(progress_file):
        return [], []
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        
        completed_symptoms = progress_data.get("completed_symptoms", [])
        dialogue_data = progress_data.get("dialogues", [])
        
        dialogues = []
        for d in dialogue_data:
            dialogue = TriageDialogue(
                symptom=d["symptom"],
                patient_query=d["patient_query"],
                clarifying_question=d["clarifying_question"],
                patient_response=d["patient_response"],
                final_triage_decision=d["final_triage_decision"],
                next_step=d["next_step"],
                reasoning_question=d["reasoning_question"],
                reasoning_decision=d["reasoning_decision"],
                generation_timestamp=d["generation_timestamp"]
            )
            dialogues.append(dialogue)
        
        return completed_symptoms, dialogues
        
    except Exception as e:
        print(f"⚠️  Error loading progress: {e}")
        return [], []


def create_train_val_test_splits(dialogues: List[TriageDialogue], train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """Create train/validation/test splits with random shuffling while maintaining symptom distribution."""
    
    # Group dialogues by base symptom (without case number)
    symptom_groups = {}
    for dialogue in dialogues:
        base_symptom = dialogue.symptom.split(' (Case ')[0]  # Remove case number
        if base_symptom not in symptom_groups:
            symptom_groups[base_symptom] = []
        symptom_groups[base_symptom].append(dialogue)
    
    train_data = []
    val_data = []
    test_data = []
    
    # For each symptom, randomly assign cases to splits
    for symptom, cases in symptom_groups.items():
        random.shuffle(cases)  # Randomize case order
        
        n_cases = len(cases)
        n_train = max(1, int(n_cases * train_ratio))
        n_val = max(1, int(n_cases * val_ratio))
        n_test = n_cases - n_train - n_val
        
        # Ensure we have at least 1 case in test if possible
        if n_test == 0 and n_cases > 2:
            n_train -= 1
            n_test = 1
        
        train_data.extend(cases[:n_train])
        val_data.extend(cases[n_train:n_train + n_val])
        test_data.extend(cases[n_train + n_val:])
    
    # Final shuffle of each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data


def save_final_results(dialogues: List[TriageDialogue], output_file: str):
    """Save final results in multiple formats with train/val/test splits."""
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Create train/val/test splits
    train_data, val_data, test_data = create_train_val_test_splits(dialogues)
    
    print(f"📊 Dataset splits:")
    print(f"   🏋️  Train: {len(train_data)} dialogues ({len(train_data)/len(dialogues)*100:.1f}%)")
    print(f"   🔍 Validation: {len(val_data)} dialogues ({len(val_data)/len(dialogues)*100:.1f}%)")
    print(f"   🧪 Test: {len(test_data)} dialogues ({len(test_data)/len(dialogues)*100:.1f}%)")
    
    # Save complete dataset
    json_output = output_file.replace('.txt', '.json')
    json_data = [
        {
            "id": i + 1,
            "symptom": d.symptom,
            "patient_query": d.patient_query,
            "clarifying_question": d.clarifying_question,
            "patient_response": d.patient_response,
            "final_triage_decision": d.final_triage_decision,
            "next_step": d.next_step,
            "reasoning_question": d.reasoning_question,
            "reasoning_decision": d.reasoning_decision,
            "generation_timestamp": d.generation_timestamp
        } for i, d in enumerate(dialogues)
    ]
    
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    # Save train/val/test splits
    def save_split(data: List[TriageDialogue], split_name: str):
        split_json = [
            {
                "id": i + 1,
                "symptom": d.symptom,
                "patient_query": d.patient_query,
                "clarifying_question": d.clarifying_question,
                "patient_response": d.patient_response,
                "final_triage_decision": d.final_triage_decision,
                "next_step": d.next_step,
                "reasoning_question": d.reasoning_question,
                "reasoning_decision": d.reasoning_decision,
                "generation_timestamp": d.generation_timestamp
            } for i, d in enumerate(data)
        ]
        
        split_file = json_output.replace('.json', f'_{split_name}.json')
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_json, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved {split_name} split: {split_file}")
    
    save_split(train_data, 'train')
    save_split(val_data, 'val')
    save_split(test_data, 'test')
    
    # Save as formatted text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Generated Medical Triage Dialogues\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total dialogues: {len(dialogues)}\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, dialogue in enumerate(dialogues, 1):
            f.write(f"--- Dialogue {i}: {dialogue.symptom} ---\n\n")
            f.write(f"Patient Query: {dialogue.patient_query}\n\n")
            f.write(f"Clarifying Question: {dialogue.clarifying_question}\n\n")
            f.write(f"Patient Response: {dialogue.patient_response}\n\n")
            f.write(f"Final Triage Decision: {dialogue.final_triage_decision}\n\n")
            f.write(f"Next Step for Patient: {dialogue.next_step}\n\n")
            f.write(f"Reasoning for Clarifying Question (CoT): {dialogue.reasoning_question}\n\n")
            f.write(f"Reasoning for Triage Decision (CoT): {dialogue.reasoning_decision}\n\n")
            f.write("-" * 80 + "\n\n")


def main():
    """Main function to generate triage dialogues with parallel processing."""
    if not TINFOIL_AVAILABLE:
        print("❌ TinfoilAgent is required for this script")
        return
    
    # Configuration
    max_workers = 10  # Parallel processing workers
    max_conditions = None  # Set to a number to limit for testing, None for all
    
    # File paths
    input_file = "/Users/choemanseung/789/hft/unique_medical_conditions.txt"
    output_file = "/Users/choemanseung/789/hft/generated_triage_dialogues.txt"
    progress_file = "/Users/choemanseung/789/hft/triage_generation_progress.json"
    
    print("🤖 Automated Triage Dialogue Generator")
    print("=" * 50)
    
    # Load medical conditions and build triage lookup
    print("📋 Loading medical conditions and building triage lookup...")
    
    medical_conditions = load_medical_conditions(input_file)
    triage_lookup = build_triage_lookup()
    
    if not triage_lookup:
        print("❌ No structured triage data found. Please run structured agent chunking first.")
        return
    
    # Match conditions to triage data
    matched_triage_data = []
    unmatched_conditions = []
    
    for condition in medical_conditions:
        triage_data = find_matching_triage_data(condition, triage_lookup)
        if triage_data:
            matched_triage_data.append(triage_data)
        else:
            unmatched_conditions.append(condition)
    
    print(f"📊 Matched {len(matched_triage_data)}/{len(medical_conditions)} conditions to triage data")
    
    if unmatched_conditions:
        print(f"⚠️  {len(unmatched_conditions)} conditions could not be matched:")
        for condition in unmatched_conditions[:10]:  # Show first 10
            print(f"   - {condition}")
        if len(unmatched_conditions) > 10:
            print(f"   ... and {len(unmatched_conditions) - 10} more")
    
    if max_conditions:
        triage_data = matched_triage_data[:max_conditions]
        print(f"🔧 Testing mode: Using first {max_conditions} matched entries")
    else:
        triage_data = matched_triage_data
    
    print(f"📊 Processing {len(triage_data)} matched triage entries")
    
    # Load existing progress
    completed_symptoms, existing_dialogues = load_progress(progress_file)
    
    # Filter out already completed conditions (use condition name as key)
    remaining_triage_data = [t for t in triage_data if t['condition'] not in completed_symptoms]
    
    print(f"✅ Found {len(existing_dialogues)} existing dialogues")
    print(f"📝 Remaining triage entries to process: {len(remaining_triage_data)}")
    
    if not remaining_triage_data:
        print("🎉 All triage entries already processed!")
        save_final_results(existing_dialogues, output_file)
        return
    
    # Start generation with parallel processing
    print(f"🚀 Starting parallel generation with {max_workers} workers...")
    start_time = time.time()
    
    all_dialogues = existing_dialogues.copy()
    completed = completed_symptoms.copy()
    successful_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks (each will generate 5 cases)
        future_to_condition = {
            executor.submit(generate_multiple_dialogues, triage_entry): triage_entry['condition'] 
            for triage_entry in remaining_triage_data
        }
        
        # Process completed futures
        for future in as_completed(future_to_condition):
            condition = future_to_condition[future]
            
            try:
                original_condition, dialogues, cases_generated = future.result()
                
                if dialogues:
                    all_dialogues.extend(dialogues)
                    successful_count += cases_generated
                    print(f"✅ Generated {cases_generated}/5 cases for: {condition}")
                else:
                    failed_count += 1
                    print(f"❌ Failed to generate any cases for: {condition}")
                
                completed.append(condition)
                
                # Save progress every 5 conditions (since each has 7 cases)
                if len(completed) % 5 == 0:
                    save_progress(progress_file, completed, all_dialogues)
                
                # Progress update
                conditions_completed = len(completed)
                remaining_conditions_count = len(remaining_triage_data) - conditions_completed
                elapsed = time.time() - start_time
                
                if conditions_completed > 0:
                    avg_time = elapsed / conditions_completed
                    eta = remaining_conditions_count * avg_time
                    
                    print(f"📊 Progress: {conditions_completed}/{len(remaining_triage_data)} conditions "
                          f"({len(all_dialogues)} total dialogues) "
                          f"ETA: {eta/60:.1f}m")
                
            except Exception as e:
                print(f"❌ Error processing {condition}: {e}")
                failed_count += 1
                completed.append(condition)
    
    # Save final results
    save_progress(progress_file, completed, all_dialogues)
    save_final_results(all_dialogues, output_file)
    
    # Summary
    total_time = time.time() - start_time
    expected_dialogues = len(triage_data) * 7  # 7 cases per condition
    
    print(f"\n🎉 Triage Dialogue Generation Complete!")
    print(f"📊 Total triage entries processed: {len(triage_data)}")
    print(f"✅ Total dialogues generated: {len(all_dialogues)}")
    print(f"🎯 Expected dialogues (7 per entry): {expected_dialogues}")
    print(f"❌ Failed entry processing: {failed_count}")
    print(f"📈 Overall success rate: {(len(all_dialogues) / expected_dialogues * 100):.1f}%")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🚀 Average rate: {len(all_dialogues) / (total_time/60):.1f} dialogues/minute")
    print(f"💾 Results saved to:")
    print(f"   📄 Text: {output_file}")
    print(f"   📊 JSON: {output_file.replace('.txt', '.json')}")
    
    # Show final triage distribution
    final_triage_counts = {}
    for dialogue in all_dialogues:
        triage = dialogue.final_triage_decision
        final_triage_counts[triage] = final_triage_counts.get(triage, 0) + 1
    
    print(f"\n🎯 Final triage distribution: {final_triage_counts}")
    
    # Clean up progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print("🧹 Cleaned up progress file")
    
    # Show sample dialogue
    if all_dialogues:
        print(f"\n📋 Sample generated dialogue:")
        print("-" * 50)
        sample = all_dialogues[0]
        print(f"Symptom: {sample.symptom}")
        print(f"Patient Query: {sample.patient_query}")
        print(f"Clarifying Question: {sample.clarifying_question}")
        print(f"Final Triage: {sample.final_triage_decision}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n🛑 Interrupted by user. Progress has been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ An unexpected error occurred: {e}")
        sys.exit(1)