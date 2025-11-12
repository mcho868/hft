# Project Reorganization Plan
## Based on Dissertation Implementation Structure

**Date**: 2025-01-13
**Purpose**: Align actual project structure with dissertation workflow documentation

---

## Section 1: Knowledge Base and Dataset Construction (Stages 1-4)

This section covers the data acquisition and processing pipeline before model training.

### Stage 1: Web Scraping

#### ✅ KEEP - Core Production Scripts

**Location**: `webscrapers/`

| File | Purpose | Status |
|------|---------|--------|
| `nhs_scraper.py` | NHS conditions scraper (~800 conditions) | **KEEP** |
| `mayo_scraper.py` | Mayo Clinic scraper | **KEEP** |
| `mayo_diagnosis_treatment_scraper.py` | Mayo diagnosis/treatment scraper | **KEEP** |
| `healthify_scraper.py` | Healthify NZ health info scraper | **KEEP** |
| `medical_scraper.py` | Base scraper utilities | **KEEP** |

**Action**: ✅ Keep entire `webscrapers/` directory as-is

#### ❌ BACKUP - Experimental/Debug Scripts

**Root directory scripts to backup**:

| File | Reason | Backup Location |
|------|--------|-----------------|
| `check_healthify_consistency.py` | Debug script, one-time use | `backup_bin/data_quality_checks/` |
| `check_healthify_files.py` | Debug script, one-time use | `backup_bin/data_quality_checks/` |
| `analyze_healthify_duplicates.py` | Analysis script, completed | `backup_bin/data_quality_checks/` |
| `check_source_document.py` | Debug utility | `backup_bin/data_quality_checks/` |
| `compare_mayo_ids.py` | Debug utility | `backup_bin/data_quality_checks/` |
| `missing_sources.py` | Analysis script | `backup_bin/data_quality_checks/` |

**Scraped Data Directories**:

| Directory | Contents | Action |
|-----------|----------|--------|
| `RAGdatav3/nhs/` | NHS scraped JSON files | **KEEP** (source data) |
| `RAGdatav3/mayo/` | Mayo scraped JSON files | **KEEP** (source data) |
| `RAGdatav3/healthify/` | Healthify scraped JSON files | **KEEP** (source data) |
| `RAGdata/` | Old scraping format (v1) | **BACKUP** to `backup_bin/RAGdata_v1/` |

---

### Stage 2: Dataset Preparation

#### ✅ KEEP - Core Dataset Scripts

**Location**: `preparing dataset/`

| File | Purpose | Status |
|------|---------|--------|
| `simple_deduplicator.py` | Deduplicates medical content | **KEEP** |
| `clean_triage_data.py` | Cleans triage dialogues | **KEEP** |
| `prepare_mlx_dataset.py` | Converts to MLX format | **KEEP** |

**Root directory - Core pipeline scripts**:

| File | Purpose | Status |
|------|---------|--------|
| `generate_triage_dialogues.py` | **PRIMARY SCRIPT** - Generates 13K dialogues | **KEEP** |
| `extract_medical_conditions.py` | Extracts condition list | **KEEP** |
| `filter_medical_conditions.py` | Filters conditions | **KEEP** |
| `ai_filter_medical_conditions.py` | AI-assisted filtering | **KEEP** |
| `convert_to_mlx_format.py` | MLX conversion utility | **KEEP** |

#### ❌ BACKUP - Analysis/Debug Scripts

**Root directory to backup**:

| File | Reason | Backup Location |
|------|--------|-----------------|
| `analyze_discrepancy.py` | One-time analysis | `backup_bin/dataset_analysis/` |
| `analyze_symptoms.py` | Symptom analysis | `backup_bin/dataset_analysis/` |
| `debug_reasoning_issue.py` | Debug script | `backup_bin/dataset_analysis/` |
| `dataset_transformer.py` | Old transformer (superseded) | `backup_bin/dataset_tools_old/` |
| `redistribute_data.py` | Data reorganization utility | `backup_bin/dataset_tools_old/` |
| `clean_progress_file.py` | Progress file cleaner | `backup_bin/dataset_tools_old/` |

**Preparing dataset directory to backup**:

| File | Reason | Backup Location |
|------|--------|-----------------|
| `compare_datasets.py` | One-time comparison | `backup_bin/dataset_analysis/` |
| `create_reasoning_dataset.py` | Old dataset format | `backup_bin/dataset_tools_old/` |
| `dataset_load.py` | Test loader | `backup_bin/dataset_analysis/` |
| `prepare_simple_dataset.py` | Superseded by prepare_mlx_dataset.py | `backup_bin/dataset_tools_old/` |

**Final Datasets**:

| Directory/File | Contents | Action |
|----------------|----------|--------|
| `Final_dataset/final_triage_dialogues_mlx/` | **PRODUCTION DATASET** (train/valid/test.jsonl) | **KEEP** |
| `Final_dataset/generated_triage_dialogues.json` | Full 13K dialogues | **KEEP** |
| `Final_dataset/generated_triage_dialogues.txt` | Text format | **KEEP** (for reference) |
| `Final_dataset/simplified_triage_dialogues_*.json` | Simplified versions | **KEEP** |
| `Final_dataset/triage_dialogues_mlx/` | Old MLX format (dirty prompts) | **BACKUP** to `backup_bin/old_datasets/` |

**Supporting Files**:

| File | Action |
|------|--------|
| `unique_medical_conditions.txt` | **KEEP** (master condition list) |
| `ai_filtered_medical_conditions.txt` | **KEEP** (AI-filtered list) |
| `filtered_medical_conditions.txt` | **KEEP** (manually filtered) |
| `ai_filtering_detailed_log.txt` | **BACKUP** to `backup_bin/logs/` |

---

### Stage 3: Document Chunking

#### ✅ KEEP - Core Chunking System

**Modular chunking framework**:

| Location | Contents | Status |
|----------|----------|--------|
| `RAGdatav3/scripts/` | **Entire directory** - Modular chunker classes | **KEEP** |
| `main_chunking_script_v4.py` | **PRIMARY SCRIPT** - Main chunking runner | **KEEP** |

**RAGdatav3/scripts/ breakdown**:

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Package init | **KEEP** |
| `chunker_base.py` | Base chunker class | **KEEP** |
| `fixed_length_chunker.py` | Fixed-length chunking | **KEEP** |
| `sentence_based_chunker.py` | Sentence-based chunking | **KEEP** |
| `paragraph_based_chunker.py` | Paragraph-based chunking | **KEEP** |
| `agent_based_chunker.py` | LLM-guided chunking | **KEEP** |
| `structured_agent_chunker.py` | **WINNER** - Structured extraction | **KEEP** |
| `contextual_retrieval_chunker.py` | Contextual chunking | **KEEP** |
| `chunkers.py` | Legacy functions | **KEEP** (backward compat) |
| `utils.py` | Shared utilities | **KEEP** |

**Chunked Data**:

| Directory | Contents | Action |
|-----------|----------|--------|
| `RAGdatav4/` | **PRODUCTION CHUNKS** - All 100+ chunk files | **KEEP** |
| `RAGdatav4/*.json` | Chunk files (e.g., `nhs_chunks_fixed_c512_o100.json`) | **KEEP** |

#### ❌ BACKUP - Experimental/Old Scripts

**Root directory to backup**:

| File | Reason | Backup Location |
|------|--------|-----------------|
| `agent_chunk_deduplicator.py` | Post-processing utility (one-time) | `backup_bin/chunking_tools/` |
| `simple_paragraphchunk_maker_from_deduplicated.py` | Old chunker (superseded) | `backup_bin/chunking_old/` |
| `medical_file_filter.py` | File filtering utility | `backup_bin/chunking_tools/` |

**Progress files in root to backup**:

| File | Reason |
|------|--------|
| `agent_chunking_progress_*.json` | Temporary progress (3 files) | **DELETE** or move to `backup_bin/progress_files/` |

**Progress files in backup_bin**:

| Pattern | Action |
|---------|--------|
| `backup_bin/agent_chunking_progress_*.json` | Already backed up | **KEEP in backup_bin** |
| `backup_bin/contextual_chunking_progress_*.json` | Large temp files (~500MB total) | **DELETE** (can regenerate) |

---

### Stage 4: Vector Index Building

#### ✅ KEEP - Core Index System

**Primary script**:

| File | Purpose | Status |
|------|---------|--------|
| `main_build_script_index_only.py` | **PRIMARY SCRIPT** - Builds all FAISS indices | **KEEP** |

**Index storage**:

| Directory | Contents | Action |
|-----------|----------|--------|
| `RAGdatav4/indiv_embeddings/` | **PRODUCTION INDICES** - All .index files | **KEEP** |
| `RAGdatav4/indiv_embeddings/*.index` | FAISS vector indices (~100-120 files) | **KEEP** |
| `RAGdatav4/embeddings/` | If exists, old embedding files | Check and potentially **BACKUP** |

#### ❌ BACKUP - Old Retrieval/Demo Scripts

**Root directory to backup**:

| File | Reason | Backup Location |
|------|--------|-----------------|
| `basic_retrieval_demo.py` | Demo script | `backup_bin/demos/` |
| `contextual_retrieval_demo.py` | Demo script | `backup_bin/demos/` |
| `retrieval_function.py` | Old retrieval (v1) | `backup_bin/retrieval_old/` |
| `retrieval_function_v2.py` | Old retrieval (v2) | `backup_bin/retrieval_old/` |
| `retrieval_testv2.py` | Old test script | `backup_bin/retrieval_old/` |
| `rag_chat.py` | Demo chat script | `backup_bin/demos/` |

**Directories to backup**:

| Directory | Reason | Backup Location |
|-----------|----------|-----------------|
| `contextual_retrieval _sample/` | Sample/test data | `backup_bin/samples/` |

**Config files**:

| File | Action |
|------|--------|
| `contextual_retrieval_config.json` | **KEEP** (may be referenced) |

---

## Summary for Section 1

### Files to KEEP (Production-Ready)

**Scripts (16 files)**:
```
webscrapers/nhs_scraper.py
webscrapers/mayo_scraper.py
webscrapers/mayo_diagnosis_treatment_scraper.py
webscrapers/healthify_scraper.py
webscrapers/medical_scraper.py

generate_triage_dialogues.py
extract_medical_conditions.py
filter_medical_conditions.py
ai_filter_medical_conditions.py
convert_to_mlx_format.py

preparing dataset/simple_deduplicator.py
preparing dataset/clean_triage_data.py
preparing dataset/prepare_mlx_dataset.py

main_chunking_script_v4.py
main_build_script_index_only.py
contextual_retrieval_config.json
```

**Directories**:
```
webscrapers/                           # All scraper code
RAGdatav3/nhs/                         # Scraped data (source)
RAGdatav3/mayo/                        # Scraped data (source)
RAGdatav3/healthify/                   # Scraped data (source)
RAGdatav3/scripts/                     # Modular chunking framework
RAGdatav4/                             # All chunk files (100+)
RAGdatav4/indiv_embeddings/            # All FAISS indices (100+)
Final_dataset/final_triage_dialogues_mlx/  # Production training data
Final_dataset/generated_triage_dialogues.json
Final_dataset/simplified_triage_dialogues_*.json
preparing dataset/                     # Core dataset scripts only
```

**Reference Files**:
```
unique_medical_conditions.txt
ai_filtered_medical_conditions.txt
filtered_medical_conditions.txt
```

### Files to BACKUP (25+ files)

**Create these backup subdirectories**:
```bash
backup_bin/
├── data_quality_checks/        # 6 files
├── dataset_analysis/           # 4 files
├── dataset_tools_old/          # 5 files
├── chunking_tools/             # 2 files
├── chunking_old/               # 1 file
├── retrieval_old/              # 3 files
├── demos/                      # 3 files
├── samples/                    # 1 directory
├── old_datasets/               # 1 directory
├── RAGdata_v1/                 # 1 directory
└── logs/                       # 1 file
```

### Files to DELETE

**Temporary progress files** (regenerable):
```
backup_bin/contextual_chunking_progress_*.json  # ~500MB of temp data
agent_chunking_progress_*.json (in root)        # If still present
```

---

## Next Steps

1. **Review this plan** - Confirm files to keep/backup/delete
2. **Create backup structure** - Set up backup_bin subdirectories
3. **Move files** - Execute backup operations
4. **Update README** - Align with clean structure
5. **Proceed to Section 2** - Model training (Stages 5-7)

Would you like me to:
- Generate the bash script to execute this reorganization?
- Proceed to Section 2 (Model Fine-Tuning, Retrieval Testing, Generation Validation)?
- Make any adjustments to this plan?
