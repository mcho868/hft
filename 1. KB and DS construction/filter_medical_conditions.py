#!/usr/bin/env python3
"""
Filter and clean medical conditions list to remove medications, procedures, 
and other non-condition entries while handling bilingual content.
"""

import re
from typing import Set, List, Tuple


def load_medical_conditions(file_path: str) -> List[str]:
    """Load medical conditions from the numbered list file."""
    conditions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Extract condition from numbered format: "   123. Condition Name"
            match = re.match(r'\s*\d+\.\s+(.+)', line.strip())
            if match:
                conditions.append(match.group(1))
    return conditions


def create_medication_patterns() -> List[str]:
    """Create regex patterns to identify medication names."""
    # Common medication suffixes and patterns
    return [
        r'.*mab$',  # monoclonal antibodies (adalimumab, infliximab)
        r'.*prazole$',  # proton pump inhibitors (omeprazole, lansoprazole)
        r'.*olol$',  # beta blockers (atenolol, metoprolol)
        r'.*pril$',  # ACE inhibitors (lisinopril, enalapril)
        r'.*sartan$',  # ARBs (losartan, valsartan)
        r'.*pine$',  # calcium channel blockers (amlodipine, nifedipine)
        r'.*ine$',  # many drugs end in -ine (morphine, codeine, but also genuine conditions)
        r'.*statin$',  # statins (atorvastatin, simvastatin)
        r'.*cillin$',  # antibiotics (amoxicillin, penicillin)
        r'.*mycin$',  # antibiotics (erythromycin, clarithromycin)
        r'.*floxacin$',  # fluoroquinolones (ciprofloxacin)
        r'.*tidine$',  # H2 blockers (ranitidine, famotidine)
        r'.*zole$',  # antifungals and others (fluconazole, metronidazole)
        r'.*pam$',  # benzodiazepines (diazepam, lorazepam)
        r'.*barbital$',  # barbiturates
        r'.*thiazide$',  # diuretics
    ]


def create_known_medications() -> Set[str]:
    """Create a set of known medication names to filter out."""
    return {
        'acarbose', 'acetazolamide', 'aciclovir', 'acitretin', 'adalimumab',
        'alendronate', 'alitretinoin', 'allopurinol', 'amgevita', 'amiodarone',
        'amisulpride', 'amitriptyline', 'amlodipine', 'amorolfine', 'amoxicillin',
        'anastrozole', 'antacids', 'antibiotics', 'antihistamines', 'antiseptics',
        'apomorphine', 'aripiprazole', 'aspirin', 'atenolol', 'atorvastatin',
        'atomoxetine', 'azathioprine', 'azithromycin', 'baclofen', 'baricitinib',
        'beclometasone', 'bendroflumethiazide', 'betahistine', 'bevacizumab',
        'bexsero', 'bezafibrate', 'biotin', 'bisoprolol', 'budesonide',
        'bumetanide', 'bupropion', 'buspirone', 'cabergoline', 'calcipotriol',
        'calcitriol', 'candesartan', 'capecitabine', 'capsaicin', 'carbamazepine',
        'carbimazole', 'carvedilol', 'cefaclor', 'cefalexin', 'ceftriaxone',
        'celecoxib', 'cellcept', 'cerazette', 'cerumol', 'cetirizine',
        'chloramphenicol', 'chlortalidone', 'ciclopirox', 'ciclosporin',
        'cilazapril', 'ciprofloxacin', 'citalopram', 'clarithromycin', 'clexane',
        'clindamycin', 'clobazam', 'clomifene', 'clonidine', 'clopidogrel',
        'clozapine', 'codeine', 'colchicine', 'colecalciferol', 'copaxone',
        'cosentyx', 'crotamiton', 'cyclizine', 'cyclophosphamide', 'cyproterone',
        'dabigatran', 'decongestants', 'dexamethasone', 'dexamfetamine',
        'diclofenac', 'digoxin', 'diltiazem', 'disinfectants', 'disulfiram',
        'diuretics', 'domperidone', 'donepezil', 'doxazosin', 'doxycycline',
        'dulaglutide', 'duromine', 'emgality', 'emicizumab', 'empagliflozin',
        'enalapril', 'enbrel', 'entacapone', 'entecavir', 'entresto',
        'eplerenone', 'epoetin', 'erythromycin', 'escitalopram', 'ethambutol',
        'ethosuximide', 'exemestane', 'famciclovir', 'famotidine', 'fasenra',
        'febuxostat', 'felodipine', 'fentanyl', 'fexofenadine', 'finasteride',
        'fingolimod', 'flecainide', 'flixonase', 'flixotide', 'flucloxacillin',
        'fluorouracil', 'fluoxetine', 'folic acid', 'fosfomycin', 'furosemide',
        'gabapentin', 'galantamine', 'galvumet', 'galvus', 'gaviscon',
        'gemfibrozil', 'gliclazide', 'glipizide', 'glucagen', 'glucosamine',
        'glyceryl', 'goserelin', 'humira', 'hydroxyurea', 'ibuprofen',
        'imiquimod', 'indapamide', 'infliximab', 'insulin', 'ipratropium',
        'isotretinoin', 'itraconazole', 'ivermectin', 'keytruda', 'labetalol',
        'lactulose', 'lagevrio', 'lamotrigine', 'lansoprazole', 'leflunomide',
        'lenalidomide', 'letrozole', 'levetiracetam', 'levothyroxine',
        'linezolid', 'liraglutide', 'lisinopril', 'loperamide', 'loratadine',
        'losartan', 'macrogol', 'madopar', 'mebendazole', 'mebeverine',
        'meclozine', 'melatonin', 'memantine', 'metformin', 'methyldopa',
        'metoprolol', 'metronidazole', 'minocycline', 'mirtazapine',
        'modafinil', 'montelukast', 'morphine', 'moxifloxacin', 'nadolol',
        'naproxen', 'natalizumab', 'nifedipine', 'nimodipine', 'nintedanib',
        'norfloxacin', 'nucala', 'ocrelizumab', 'olanzapine', 'olsalazine',
        'omeprazole', 'onbrez', 'ondansetron', 'ornidazole', 'orphenadrine',
        'oseltamivir', 'oxybutynin', 'oxycodone', 'palivizumab',
        'pantoprazole', 'paracetamol', 'paroxetine', 'paxlovid', 'penicillin',
        'perindopril', 'pevaryl', 'phenytoin', 'pimafucort', 'pioglitazone',
        'pirfenidone', 'pizotifen', 'ponstan', 'pramipexole', 'pravastatin',
        'prednisolone', 'prednisone', 'pregabalin', 'primolut', 'probiotics',
        'proctosedyl', 'promethazine', 'propranolol', 'pyrazinamide',
        'quetiapine', 'quinapril', 'raloxifene', 'ramipril', 'ranitidine',
        'rasagiline', 'rectogesic', 'relistor', 'remdesivir', 'rifampicin',
        'rifinah', 'rinvoq', 'risedronate', 'risperidone', 'rituximab',
        'rivaroxaban', 'rivastigmine', 'rizatriptan', 'ropinirole',
        'rosuvastatin', 'roxithromycin', 'salbutamol', 'salmeterol',
        'saxenda', 'seretide', 'sertraline', 'simvastatin', 'sinemet',
        'solifenacin', 'stalevo', 'statins', 'sulfasalazine', 'sumatriptan',
        'symbicort', 'tadalafil', 'tamoxifen', 'tamsulosin', 'tecentriq',
        'tecovirimat', 'tenofovir', 'terbinafine', 'teriflunomide',
        'teriparatide', 'testogel', 'tiotropium', 'tolcapone', 'topiramate',
        'tramadol', 'trastuzumab', 'triazolam', 'trimethoprim', 'triptans',
        'urea cream', 'ustekinumab', 'valaciclovir', 'vannair', 'vardenafil',
        'varenicline', 'vedolizumab', 'venlafaxine', 'verapamil', 'victoza',
        'warfarin', 'wegovy', 'xenical', 'zanamivir', 'ziprasidone',
        'zoledronate', 'zopiclone'
    }


def create_procedure_terms() -> Set[str]:
    """Create a set of medical procedures and diagnostic terms to filter out."""
    return {
        'acupuncture', 'amniocentesis', 'anaesthesia', 'angioplasty and stents',
        'antenatal blood tests', 'barium enema', 'barium swallow and barium meal',
        'bone scan', 'bronchoscopy', 'caesarean section', 'colonoscopy',
        'colposcopy', 'cone biopsy', 'coronary angiography', 'ct scan',
        'ct calcium score', 'dexa scan', 'dialysis', 'echocardiogram',
        'electrocardiograph', 'eye examination', 'fluoroscopy', 'gastroscopy',
        'mammogram', 'mri scan', 'spirometry', 'ultrasound', 'x-ray',
        'blood tests', 'blood transfusion', 'chemotherapy', 'cortisone injections',
        'vaccination', 'vaccines', 'surgery', 'surgical mesh', 'telehealth',
        'wound healing'
    }


def create_generic_terms() -> Set[str]:
    """Create a set of overly generic medical terms to filter out."""
    return {
        'cancer', 'pain', 'surgery', 'ageing topics', 'allergy topics',
        'anxiety topics', 'arthritis topics', 'asthma topics', 'autoimmune disease topics',
        'back and neck pain topics', 'blindness and low vision topics',
        'blood clot topics', 'blood pressure topics', 'blood tests topics',
        'cancer screening topics', 'cancer topics', 'cardiovascular and heart disease topics',
        'children\'s common health problems topics', 'coeliac disease topics',
        'contraception topics', 'copd topics', 'covid-19 management topics',
        'covid-19 medicine topics', 'covid-19 vaccination topics',
        'dementia topics', 'depression topics', 'diabetes test topics',
        'diabetes topics', 'driving and road safety topics', 'eating disorders',
        'eczema topics', 'epilepsy topics', 'eye care and treatment topics',
        'eye condition topics', 'fever topics', 'first aid topics',
        'food allergy topics', 'gambling harm topics', 'gender diversity topics',
        'gout topics', 'lgbtqi topics', 'nsaid topics', 'opioid topics',
        'phobia topics', 'sexual health', 'what is pain', 'brand changes',
        'common symptoms and what to do', 'global dro', 'illegal drugs',
        'theranostics'
    }


def create_bilingual_cleaning_patterns() -> List[Tuple[str, str]]:
    """Create patterns to clean bilingual entries (Māori/English)."""
    return [
        # Remove common Māori additions that don't add medical value
        (r'\s+mate\s+\w+(\s+\w+)*$', ''),  # Remove "mate [māori terms]" at end
        (r'\s+aroreretini(\s+ki\s+ngā\s+pakeke)?$', ''),  # Remove ADHD māori terms
        (r'\s+pokenga\s+\w+(\s+\w+)*$', ''),  # Remove "pokenga [terms]"
        (r'\s+anipā\s+o\s+te\s+huringa\s+āhuarangi$', ''),  # Climate change anxiety
    ]


def should_filter_condition(condition: str, medications: Set[str], 
                          procedures: Set[str], generic_terms: Set[str],
                          med_patterns: List[str]) -> Tuple[bool, str]:
    """
    Determine if a condition should be filtered out and why.
    
    Returns:
        Tuple of (should_filter: bool, reason: str)
    """
    condition_lower = condition.lower()
    
    # Check if it's a known medication
    if condition_lower in medications:
        return True, "medication"
    
    # Check if it's a procedure/diagnostic term
    if condition_lower in procedures:
        return True, "procedure"
    
    # Check if it's a generic term
    if condition_lower in generic_terms:
        return True, "generic"
    
    # Check medication patterns
    for pattern in med_patterns:
        if re.match(pattern, condition_lower):
            return True, f"medication_pattern ({pattern})"
    
    # Filter very short terms (likely abbreviations or incomplete)
    if len(condition.replace(' ', '')) < 3:
        return True, "too_short"
    
    # Filter terms that are mostly punctuation or special characters
    if len(re.sub(r'[a-zA-Z\s]', '', condition)) > len(condition) * 0.3:
        return True, "too_many_special_chars"
    
    return False, ""


def clean_bilingual_entry(condition: str, patterns: List[Tuple[str, str]]) -> str:
    """Clean bilingual entries by removing redundant Māori terms."""
    cleaned = condition
    
    for pattern, replacement in patterns:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    
    # Clean up extra spaces and formatting
    cleaned = ' '.join(cleaned.split())
    cleaned = cleaned.strip()
    
    return cleaned


def main():
    """Main function to filter and clean medical conditions."""
    input_file = "/Users/choemanseung/789/hft/unique_medical_conditions.txt"
    output_file = "/Users/choemanseung/789/hft/filtered_medical_conditions.txt"
    report_file = "/Users/choemanseung/789/hft/filtering_report.txt"
    
    print("Loading medical conditions...")
    conditions = load_medical_conditions(input_file)
    print(f"Loaded {len(conditions)} conditions")
    
    # Create filtering sets and patterns
    medications = create_known_medications()
    procedures = create_procedure_terms()
    generic_terms = create_generic_terms()
    med_patterns = create_medication_patterns()
    bilingual_patterns = create_bilingual_cleaning_patterns()
    
    # Filter and clean conditions
    kept_conditions = []
    filtered_out = []
    
    print("\nFiltering conditions...")
    for condition in conditions:
        # First clean bilingual entries
        cleaned_condition = clean_bilingual_entry(condition, bilingual_patterns)
        
        # Check if should be filtered
        should_filter, reason = should_filter_condition(
            cleaned_condition, medications, procedures, generic_terms, med_patterns
        )
        
        if should_filter:
            filtered_out.append((condition, cleaned_condition, reason))
        else:
            # Only add if it's different from original or wasn't filtered
            if cleaned_condition and cleaned_condition.strip():
                kept_conditions.append(cleaned_condition)
    
    # Remove duplicates while preserving order
    unique_kept = []
    seen = set()
    for condition in kept_conditions:
        condition_lower = condition.lower()
        if condition_lower not in seen:
            seen.add(condition_lower)
            unique_kept.append(condition)
    
    # Sort the final list
    unique_kept.sort()
    
    # Write filtered conditions
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Filtered Medical Conditions\n")
        f.write("=" * 28 + "\n\n")
        f.write(f"Total count: {len(unique_kept)}\n")
        f.write(f"Filtered out: {len(filtered_out)}\n")
        f.write(f"Original count: {len(conditions)}\n\n")
        
        for i, condition in enumerate(unique_kept, 1):
            f.write(f"{i:4d}. {condition}\n")
    
    # Write filtering report
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Medical Conditions Filtering Report\n")
        f.write("=" * 36 + "\n\n")
        f.write(f"Original conditions: {len(conditions)}\n")
        f.write(f"Kept conditions: {len(unique_kept)}\n")
        f.write(f"Filtered out: {len(filtered_out)}\n")
        f.write(f"Duplicates removed: {len(kept_conditions) - len(unique_kept)}\n\n")
        
        # Group filtered items by reason
        reason_groups = {}
        for original, cleaned, reason in filtered_out:
            if reason not in reason_groups:
                reason_groups[reason] = []
            reason_groups[reason].append((original, cleaned))
        
        f.write("Filtered out by category:\n")
        f.write("-" * 25 + "\n")
        for reason, items in sorted(reason_groups.items()):
            f.write(f"\n{reason.upper()} ({len(items)} items):\n")
            for original, cleaned in items[:20]:  # Show first 20 of each category
                if original != cleaned:
                    f.write(f"  {original} -> {cleaned}\n")
                else:
                    f.write(f"  {original}\n")
            if len(items) > 20:
                f.write(f"  ... and {len(items) - 20} more\n")
    
    print(f"\nFiltering complete!")
    print(f"Original conditions: {len(conditions)}")
    print(f"Kept conditions: {len(unique_kept)}")
    print(f"Filtered out: {len(filtered_out)}")
    print(f"Duplicates removed: {len(kept_conditions) - len(unique_kept)}")
    print(f"\nResults saved to: {output_file}")
    print(f"Filtering report saved to: {report_file}")
    
    # Show sample of kept conditions
    print(f"\nSample of filtered conditions:")
    print("-" * 35)
    for i, condition in enumerate(unique_kept[:20], 1):
        print(f"{i:2d}. {condition}")
    
    if len(unique_kept) > 20:
        print(f"... and {len(unique_kept) - 20} more")


if __name__ == "__main__":
    main()